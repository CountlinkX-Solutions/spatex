//! SKATER (Assuncao et al. 2006) as implemented by `spopt.region.skater.SpanningForest`:
//! a minimum spanning forest over the graph-restricted dissimilarity, pruned greedily by the
//! edge whose removal gives the lowest whole-map score.

use super::util::{edge_components, Adj, Dsu, Mat, Metric};
use rayon::prelude::*;

/// Options of [`skater`].
#[derive(Clone, Debug)]
pub struct SkaterOpts {
    /// Number of regions requested.
    pub n_clusters: usize,
    /// Minimum number of areas per region (`-inf` for none).
    pub floor: f64,
    /// `true` counts islands towards `n_clusters`; `false` adds every component to it.
    pub ignore_islands: bool,
    /// Record the labels after every cut.
    pub trace: bool,
    /// Dissimilarity and score metric (`Manhattan` or `Euclidean`).
    pub metric: Metric,
    /// Remove edges whose dissimilarity is exactly zero, as spopt does.
    pub drop_zero: bool,
}

/// One entry of the pruning trace.
#[derive(Clone, Debug, PartialEq)]
pub struct TraceStep {
    /// Labels after the cut.
    pub labels: Vec<u32>,
    /// Cut edge endpoints (`None` for the initial entry).
    pub edge: Option<(u32, u32)>,
    /// Map score after the cut (`inf` for the initial entry).
    pub score: f64,
}

/// Result of [`skater`].
#[derive(Clone, Debug)]
pub struct SkaterOut {
    /// Region of each area, numbered by lowest member.
    pub labels: Vec<u32>,
    /// Edges `(i, j)`, `i < j`, of the pruned forest in row-major order.
    pub msf: Vec<(u32, u32)>,
    /// Final map score.
    pub score: f64,
    /// Pruning trace when requested.
    pub trace: Vec<TraceStep>,
    /// Human-readable warnings.
    pub warnings: Vec<String>,
}

/// Minimum spanning forest by Kruskal over edges sorted by `(dissimilarity, i, j)`, which is
/// exactly what `scipy.sparse.csgraph.minimum_spanning_tree` returns for a symmetric input.
pub fn spanning_forest(adj: &Adj, x: &Mat, metric: Metric, drop_zero: bool) -> Vec<(u32, u32)> {
    let mut edges: Vec<(f64, u32, u32)> = Vec::new();
    for (i, row) in adj.nb.iter().enumerate() {
        for &j in row {
            if j as usize > i {
                let d = metric.dist(x.row(i), x.row(j as usize));
                if !(drop_zero && d == 0.0) {
                    edges.push((d, i as u32, j));
                }
            }
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut dsu = Dsu::new(adj.n());
    let mut msf: Vec<(u32, u32)> = edges
        .iter()
        .filter(|&&(_, i, j)| dsu.union(i, j))
        .map(|&(_, i, j)| (i, j))
        .collect();
    msf.sort_unstable();
    msf
}

fn region_cost(x: &Mat, metric: Metric, nodes: impl Iterator<Item = u32> + Clone) -> f64 {
    let mean = x.mean_of(nodes.clone());
    nodes.map(|i| metric.dist(x.row(i as usize), &mean)).sum()
}

/// Whole-map score of `labels`; `inf` when a region has fewer than `floor` areas.
pub fn score(x: &Mat, labels: &[u32], metric: Metric, floor: f64) -> f64 {
    let k = labels.iter().max().map_or(0, |m| *m as usize + 1);
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); k];
    for (i, &l) in labels.iter().enumerate() {
        members[l as usize].push(i as u32);
    }
    if members.iter().any(|m| (m.len() as f64) < floor) {
        return f64::INFINITY;
    }
    members
        .iter()
        .map(|m| region_cost(x, metric, m.iter().copied()))
        .sum()
}

/// Per-component rooted structure used to evaluate every cut.
struct Tree {
    /// Nodes in DFS preorder.
    order: Vec<u32>,
    /// Preorder index of every node.
    tin: Vec<u32>,
    /// One past the last preorder index of the subtree.
    tout: Vec<u32>,
    parent: Vec<u32>,
}

/// Runs SKATER on `adj` with attributes `x`.
pub fn skater(adj: &Adj, x: &Mat, opts: &SkaterOpts) -> Result<SkaterOut, String> {
    let n = adj.n();
    if opts.n_clusters == 0 || opts.n_clusters > n {
        return Err(format!(
            "n_clusters must be between 1 and {n}, got {}",
            opts.n_clusters
        ));
    }
    let metric = opts.metric;
    let mut msf = spanning_forest(adj, x, metric, opts.drop_zero);
    let (mut count, mut labels) = edge_components(n, &msf);
    let mut target = opts.n_clusters;
    let mut warnings = Vec::new();
    if count > 1 {
        warnings.push(if opts.ignore_islands {
            format!(
                "By default, the graph is disconnected! Counting islands towards the remaining {} clusters.",
                target as i64 - count as i64
            )
        } else {
            format!(
                "By default, the graph is disconnected! Increasing `n_clusters` from {} to {} in order to account for islands.",
                target,
                target + count
            )
        });
        if !opts.ignore_islands {
            target += count;
        }
        let mut sizes = vec![0usize; count];
        labels.iter().for_each(|&l| sizes[l as usize] += 1);
        if sizes.iter().any(|&s| (s as f64) < opts.floor) {
            return Err(
                "Islands must be larger than the quorum. If not, drop the small islands \
                        and solve for clusters in the remaining field."
                    .into(),
            );
        }
    }
    let mut trace = Vec::new();
    if opts.trace {
        trace.push(TraceStep {
            labels: labels.clone(),
            edge: None,
            score: f64::INFINITY,
        });
    }
    while count < target {
        match find_cut(n, x, metric, opts.floor, &msf, count, &labels) {
            Some((idx, s)) => {
                let edge = msf.remove(idx);
                let (c, l) = edge_components(n, &msf);
                count = c;
                labels = l;
                if opts.trace {
                    trace.push(TraceStep {
                        labels: labels.clone(),
                        edge: Some(edge),
                        score: s,
                    });
                }
            }
            None => {
                warnings.push(format!(
                    "MSF contains no valid moves after finding {count} subtrees. Decrease the \
                     size of your quorum to find the remaining {} subtrees.",
                    target - count
                ));
                break;
            }
        }
    }
    let final_score = score(x, &labels, metric, f64::NEG_INFINITY);
    Ok(SkaterOut {
        labels,
        msf,
        score: final_score,
        trace,
        warnings,
    })
}

/// Index (into `msf`) and score of the best cut: the first edge, in row-major order, with the
/// strictly lowest finite map score.
fn find_cut(
    n: usize,
    x: &Mat,
    metric: Metric,
    floor: f64,
    msf: &[(u32, u32)],
    count: usize,
    labels: &[u32],
) -> Option<(usize, f64)> {
    let forest = Adj::from_edges(n, msf);
    let mut comp_nodes: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (i, &l) in labels.iter().enumerate() {
        comp_nodes[l as usize].push(i as u32);
    }
    let cost: Vec<f64> = comp_nodes
        .iter()
        .map(|m| region_cost(x, metric, m.iter().copied()))
        .collect();
    let trees: Vec<Tree> = comp_nodes
        .iter()
        .map(|m| root_tree(&forest, n, m))
        .collect();
    // Cost of every other region and the smallest of their sizes, per component.
    let others: Vec<f64> = (0..count)
        .map(|c| (0..count).filter(|&o| o != c).map(|o| cost[o]).sum())
        .collect();
    let other_min: Vec<f64> = (0..count)
        .map(|c| {
            (0..count)
                .filter(|&o| o != c)
                .map(|o| comp_nodes[o].len() as f64)
                .fold(f64::INFINITY, f64::min)
        })
        .collect();

    let scored: Vec<f64> = msf
        .par_iter()
        .map(|&(a, b)| {
            let c = labels[a as usize] as usize;
            if other_min[c] < floor {
                return f64::INFINITY;
            }
            let tree = &trees[c];
            let child = if tree.parent[b as usize] == a { b } else { a };
            let (lo, hi) = (tree.tin[child as usize], tree.tout[child as usize]);
            let inside = &tree.order[lo as usize..hi as usize];
            let size_in = inside.len() as f64;
            let size_out = (tree.order.len() - inside.len()) as f64;
            if size_in < floor || size_out < floor {
                return f64::INFINITY;
            }
            let c1 = region_cost(x, metric, inside.iter().copied());
            let outside = tree.order.iter().enumerate().filter_map(|(pos, &v)| {
                if (pos as u32) < lo || (pos as u32) >= hi {
                    Some(v)
                } else {
                    None
                }
            });
            let c2 = region_cost(x, metric, outside);
            others[c] + c1 + c2
        })
        .collect();
    let mut best: Option<(usize, f64)> = None;
    for (idx, &s) in scored.iter().enumerate() {
        if s.is_finite() && best.is_none_or(|(_, b)| s < b) {
            best = Some((idx, s));
        }
    }
    best
}

fn root_tree(forest: &Adj, n: usize, nodes: &[u32]) -> Tree {
    let root = nodes[0];
    let mut order = Vec::with_capacity(nodes.len());
    let mut tin = vec![0u32; n];
    let mut tout = vec![0u32; n];
    let mut parent = vec![u32::MAX; n];
    // Iterative DFS with an explicit (node, next-neighbor) stack.
    let mut stack: Vec<(u32, usize)> = vec![(root, 0)];
    tin[root as usize] = 0;
    order.push(root);
    while let Some(&mut (v, ref mut next)) = stack.last_mut() {
        let row = &forest.nb[v as usize];
        if *next < row.len() {
            let w = row[*next];
            *next += 1;
            if w != parent[v as usize] {
                parent[w as usize] = v;
                tin[w as usize] = order.len() as u32;
                order.push(w);
                stack.push((w, 0));
            }
        } else {
            tout[v as usize] = order.len() as u32;
            stack.pop();
        }
    }
    Tree {
        order,
        tin,
        tout,
        parent,
    }
}
