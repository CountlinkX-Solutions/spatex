//! SA3 (spatial adaptive agglomerative aggregation), a port of `spopt.region.SA3`.
//!
//! Every connected component of the graph larger than `min_cluster_size` is clustered with
//! spatially constrained Ward ([`super::ward`], scikit-learn's connectivity Ward), the merge
//! tree is turned into a scipy-style linkage matrix, monotonised the way spopt does, and
//! `fast_hdbscan`'s extraction (`condense_tree`, `extract_eom_clusters` / `extract_leaves`,
//! `get_cluster_label_vector`) selects the clusters. Everything else is noise (`-1`).
//!
//! `fast_hdbscan` keeps the condensed tree in `float32` (lambdas, sizes, stabilities); those
//! values are `f32` here too so that ties and selections match.

use super::util::Mat;
use super::ward::ward;
use super::Adj;

/// Cluster extraction scheme.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extraction {
    /// Excess of mass: the stability-maximising set of clusters.
    Eom,
    /// The leaves of the condensed cluster tree.
    Leaf,
}

/// One linkage row `(left, right, distance, size)`, scipy layout.
pub type Link = (u32, u32, f64, u32);

/// Linkage matrix of one clustered component, with the global ids of its members.
pub type ComponentTree = (Vec<u32>, Vec<Link>);

/// Result of [`sa3`].
#[derive(Clone, Debug)]
pub struct Sa3Out {
    /// Cluster of each area, `-1` for noise; clusters are numbered across components.
    pub labels: Vec<i64>,
    /// Connected components of the graph.
    pub n_components: usize,
    /// Members and (monotonised) linkage of every component that was clustered.
    pub trees: Vec<ComponentTree>,
}

/// spopt's correction of a non-monotone tree: while some distance is smaller than its
/// predecessor, set every such one to its predecessor plus `0.01`, all at once.
pub fn fix_monotone(d: &mut [f64]) -> Result<(), String> {
    let cap = 10 * d.len() + 100;
    for _ in 0..cap {
        let problems: Vec<usize> = (0..d.len().saturating_sub(1))
            .filter(|&i| d[i + 1] < d[i])
            .collect();
        if problems.is_empty() {
            return Ok(());
        }
        let updates: Vec<(usize, f64)> = problems.iter().map(|&i| (i + 1, d[i] + 0.01)).collect();
        for (i, v) in updates {
            d[i] = v;
        }
    }
    Err("the linkage distances could not be made monotone".into())
}

// -- fast_hdbscan.cluster_trees ---------------------------------------------------------------

/// `condense_tree` output (`parent`, `child`, `lambda_val`, `child_size`).
struct Condensed {
    parent: Vec<usize>,
    child: Vec<usize>,
    lambda: Vec<f32>,
    size: Vec<f32>,
}

fn bfs(link: &[Link], root: usize, n_points: usize) -> Vec<usize> {
    let mut to_process = vec![root];
    let mut out = Vec::new();
    while !to_process.is_empty() {
        out.extend_from_slice(&to_process);
        let mut next = Vec::new();
        for &v in &to_process {
            if v >= n_points {
                let (l, r, _, _) = link[v - n_points];
                next.push(l as usize);
                next.push(r as usize);
            }
        }
        to_process = next;
    }
    out
}

fn lambda_of(distance: f64) -> f32 {
    if distance > 0.0 {
        (1.0 / distance) as f32
    } else {
        f32::INFINITY
    }
}

#[allow(clippy::too_many_arguments)]
fn eliminate(
    branch: usize,
    parent: usize,
    lambda: f32,
    link: &[Link],
    n_points: usize,
    ignore: &mut [bool],
    out: &mut Condensed,
) {
    let mut push = |child: usize| {
        out.parent.push(parent);
        out.child.push(child);
        out.lambda.push(lambda);
        out.size.push(1.0);
    };
    if branch < n_points {
        push(branch);
        return;
    }
    for v in bfs(link, branch, n_points) {
        if v < n_points {
            push(v);
        } else {
            ignore[v] = true;
        }
    }
}

fn condense_tree(link: &[Link], min_size: f64) -> Condensed {
    let root = 2 * link.len();
    let n_points = link.len() + 1;
    let mut next_label = n_points + 1;
    let mut relabel = vec![0usize; root + 1];
    relabel[root] = n_points;
    let mut ignore = vec![false; root + 1];
    let mut out = Condensed {
        parent: Vec::new(),
        child: Vec::new(),
        lambda: Vec::new(),
        size: Vec::new(),
    };
    let count = |v: usize| {
        if v >= n_points {
            f64::from(link[v - n_points].3)
        } else {
            1.0
        }
    };
    for node in bfs(link, root, n_points) {
        if ignore[node] || node < n_points {
            continue;
        }
        let parent = relabel[node];
        let (l, r, d, _) = link[node - n_points];
        let (left, right) = (l as usize, r as usize);
        let lambda = lambda_of(d);
        let (lc, rc) = (count(left), count(right));
        if left < n_points && rc >= min_size {
            relabel[right] = parent;
            out.parent.push(parent);
            out.child.push(left);
            out.lambda.push(lambda);
            out.size.push(1.0);
        } else if lc < min_size && rc >= min_size {
            relabel[right] = parent;
            eliminate(left, parent, lambda, link, n_points, &mut ignore, &mut out);
        } else if lc >= min_size && rc < min_size {
            relabel[left] = parent;
            eliminate(right, parent, lambda, link, n_points, &mut ignore, &mut out);
        } else if lc < min_size && rc < min_size {
            eliminate(left, parent, lambda, link, n_points, &mut ignore, &mut out);
            eliminate(right, parent, lambda, link, n_points, &mut ignore, &mut out);
        } else {
            for (side, c) in [(left, lc), (right, rc)] {
                relabel[side] = next_label;
                out.parent.push(parent);
                out.child.push(next_label);
                out.lambda.push(lambda);
                out.size.push(c as f32);
                next_label += 1;
            }
        }
    }
    out
}

/// `score_condensed_tree_nodes`: stability of every cluster node (`None` for other ids).
fn node_scores(t: &Condensed) -> Vec<Option<f32>> {
    let top = t.parent.iter().chain(&t.child).copied().max().unwrap_or(0);
    let mut score: Vec<Option<f32>> = vec![None; top + 1];
    score[t.parent[0]] = Some(0.0);
    for i in 0..t.parent.len() {
        let value = t.lambda[i] * t.size[i];
        if t.size[i] > 1.0 {
            score[t.child[i]] = Some(-value);
        }
        if let Some(s) = score[t.parent[i]].as_mut() {
            *s += value;
        }
    }
    score
}

/// `extract_eom_clusters` with `allow_single_cluster=False` and no maximum size.
fn extract_eom(t: &Condensed) -> Vec<usize> {
    let score = node_scores(t);
    let cluster_rows: Vec<usize> = (0..t.parent.len()).filter(|&i| t.size[i] > 1.0).collect();
    if cluster_rows.is_empty() {
        return Vec::new();
    }
    let root = cluster_rows.iter().map(|&i| t.parent[i]).min().unwrap_or(0);
    let mut kids: Vec<Vec<usize>> = vec![Vec::new(); score.len()];
    for &i in &cluster_rows {
        kids[t.parent[i]].push(t.child[i]);
    }
    let mut selected = vec![false; score.len()];
    let n_scored = score.iter().flatten().count();
    if n_scored > 1 {
        // post-order over the subtrees of the root's children, like the recursion
        let mut ret = vec![0.0f64; score.len()];
        for &start in &kids[root] {
            let mut order = Vec::new();
            let mut stack = vec![start];
            while let Some(v) = stack.pop() {
                order.push(v);
                stack.extend(kids[v].iter().rev().copied());
            }
            for &v in order.iter().rev() {
                let current = f64::from(score[v].unwrap_or(0.0)).max(0.0);
                let total: f64 = kids[v].iter().fold(0.0, |acc, &c| acc + ret[c]);
                if total > current {
                    ret[v] = total;
                } else {
                    selected[v] = true;
                    let mut below = kids[v].clone();
                    while let Some(c) = below.pop() {
                        selected[c] = false;
                        below.extend(kids[c].iter().copied());
                    }
                    ret[v] = current;
                }
            }
        }
    }
    (0..selected.len()).filter(|&v| selected[v]).collect()
}

/// `extract_leaves`: the cluster nodes without cluster children.
fn extract_leaves(t: &Condensed) -> Vec<usize> {
    let (Some(&top), Some(&n_points)) = (t.parent.iter().max(), t.parent.iter().min()) else {
        return Vec::new();
    };
    let mut leaf = vec![true; top + 1];
    leaf[..n_points].iter_mut().for_each(|v| *v = false);
    for i in 0..t.parent.len() {
        if t.size[i] > 1.0 {
            leaf[t.parent[i]] = false;
        }
    }
    (0..leaf.len()).filter(|&v| leaf[v]).collect()
}

/// `ds_rank_create`, `ds_find` and `ds_union_by_rank` of `fast_hdbscan.disjoint_set`.
struct RankSet {
    parent: Vec<usize>,
    rank: Vec<u32>,
}

impl RankSet {
    fn new(n: usize) -> RankSet {
        RankSet {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&self, mut x: usize) -> usize {
        while self.parent[x] != x {
            x = self.parent[x];
        }
        x
    }

    fn union(&mut self, x: usize, y: usize) {
        let (mut x, mut y) = (self.find(x), self.find(y));
        if x == y {
            return;
        }
        if self.rank[x] < self.rank[y] {
            std::mem::swap(&mut x, &mut y);
        }
        self.parent[y] = x;
        if self.rank[x] == self.rank[y] {
            self.rank[x] += 1;
        }
    }
}

/// `get_cluster_label_vector` with `cluster_selection_epsilon = 0`.
fn label_vector(t: &Condensed, clusters: &[usize], n: usize) -> Vec<i64> {
    let mut result = vec![-1i64; n];
    if t.parent.is_empty() || clusters.is_empty() {
        return result;
    }
    let root = t.parent.iter().copied().min().unwrap_or(0);
    let mut sorted = clusters.to_vec();
    sorted.sort_unstable();
    let label_of = |c: usize| sorted.binary_search(&c).map_or(-1, |k| k as i64);
    let top = t.parent.iter().chain(&t.child).copied().max().unwrap_or(0);
    let mut set = RankSet::new(top + 1);
    for k in 0..t.parent.len() {
        let child = t.child[k];
        if !clusters.contains(&child) {
            if child < n && t.lambda[k] == 0.0 {
                continue;
            }
            set.union(t.parent[k], child);
        }
    }
    for (p, slot) in result.iter_mut().enumerate().take(root) {
        let cluster = set.find(p);
        *slot = match cluster.cmp(&root) {
            std::cmp::Ordering::Less => -1,
            std::cmp::Ordering::Equal => {
                if sorted.len() == 1 {
                    let max_lambda = (0..t.parent.len())
                        .filter(|&k| t.parent[k] == cluster)
                        .map(|k| t.lambda[k])
                        .fold(f32::NEG_INFINITY, f32::max);
                    let cur = (0..t.parent.len())
                        .find(|&k| t.child[k] == p)
                        .map_or(f32::NEG_INFINITY, |k| t.lambda[k]);
                    if cur >= max_lambda {
                        label_of(cluster)
                    } else {
                        -1
                    }
                } else {
                    -1
                }
            }
            std::cmp::Ordering::Greater => label_of(cluster),
        };
    }
    result
}

/// spopt's `extract_clusters`: labels (`-1` is noise) from a linkage matrix in scipy layout.
pub fn extract_clusters(
    link: &[Link],
    min_cluster_size: usize,
    extraction: Extraction,
) -> Vec<i64> {
    let n = link.len() + 1;
    let tree = condense_tree(link, min_cluster_size as f64);
    let selected = match extraction {
        Extraction::Eom => extract_eom(&tree),
        Extraction::Leaf => extract_leaves(&tree),
    };
    label_vector(&tree, &selected, n)
}

// -- SA3 --------------------------------------------------------------------------------------

/// Number of leaves under each node, to turn scikit-learn's children into linkage rows.
fn linkage_rows(children: &[(u32, u32)], distances: &[f64], n: usize) -> Vec<Link> {
    let mut sizes: Vec<u32> = Vec::with_capacity(children.len());
    let mut rows = Vec::with_capacity(children.len());
    for (i, &(a, b)) in children.iter().enumerate() {
        let size_of = |v: u32| {
            if (v as usize) < n {
                1
            } else {
                sizes[v as usize - n]
            }
        };
        let s = size_of(a) + size_of(b);
        sizes.push(s);
        rows.push((a, b, distances[i], s));
    }
    rows
}

/// Runs SA3. `adj` must be symmetric; `min_cluster_size` must be at least 2.
pub fn sa3(
    adj: &Adj,
    x: &Mat,
    min_cluster_size: usize,
    extraction: Extraction,
) -> Result<Sa3Out, String> {
    if min_cluster_size < 2 {
        return Err(format!(
            "min_cluster_size must be an integer >= 2, got {min_cluster_size}"
        ));
    }
    let n = adj.n();
    let (count, comp) = adj.components();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (i, &c) in comp.iter().enumerate() {
        members[c as usize].push(i as u32);
    }
    let mut labels = vec![-1i64; n];
    let mut trees = Vec::new();
    let mut next_cluster = 0i64;
    let mut local_of = vec![0u32; n];
    for nodes in members {
        if nodes.len() <= min_cluster_size {
            continue;
        }
        for (k, &v) in nodes.iter().enumerate() {
            local_of[v as usize] = k as u32;
        }
        let sub = Adj {
            nb: nodes
                .iter()
                .map(|&v| {
                    adj.nb[v as usize]
                        .iter()
                        .map(|&u| local_of[u as usize])
                        .collect()
                })
                .collect(),
        };
        let rows: Vec<f64> = nodes
            .iter()
            .flat_map(|&v| x.row(v as usize).iter().copied())
            .collect();
        let sub_x = Mat::new(&rows, nodes.len(), x.p)?;
        let w = ward(&sub, &sub_x, 2)?;
        let mut distances = w.distances;
        fix_monotone(&mut distances)?;
        let link = linkage_rows(&w.children, &distances, nodes.len());
        let local = extract_clusters(&link, min_cluster_size, extraction);
        if let Some(&top) = local.iter().max().filter(|&&m| m >= 0) {
            for (&v, &l) in nodes.iter().zip(&local) {
                if l >= 0 {
                    labels[v as usize] = l + next_cluster;
                }
            }
            next_cluster += top + 1;
        }
        trees.push((nodes, link));
    }
    Ok(Sa3Out {
        labels,
        n_components: count,
        trees,
    })
}
