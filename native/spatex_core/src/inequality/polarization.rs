//! Spatial polarization index `S` (`inequality.polarization.S`).

use crate::graph::Csr;

/// How observations are split into attribute groups.
#[derive(Clone, Debug, PartialEq)]
pub enum Grouping {
    /// `k` quantile classes (`pandas.qcut(x, k, labels=False, duplicates="drop")`).
    Quantiles(usize),
    /// Explicit right-closed bin edges (`pandas.cut(x, bins)`).
    Bins(Vec<f64>),
}

/// Result of [`polarization`].
#[derive(Clone, Debug)]
pub struct Polarization {
    /// Observed index.
    pub statistic: f64,
    /// Number of attribute groups (`k` or `len(bins) - 1`).
    pub n_a_components: usize,
    /// Number of connected components of the graph.
    pub n_g_components: usize,
    /// Number of connected components of the same-group subgraph, excluded observations aside.
    pub n_i_components: usize,
    /// Same-group component of each observation, `-1` when it has no group.
    pub i_labels: Vec<i64>,
    /// Attribute group of each observation, `-1` when it has none.
    pub a_labels: Vec<i64>,
    /// Graph component of each observation.
    pub g_labels: Vec<i64>,
    /// Pseudo p-value `(#{sim >= statistic} + 1) / (P + 1)` (`NaN` without permutations).
    pub p_value: f64,
    /// Simulated index per permutation.
    pub sim: Vec<f64>,
}

use rayon::prelude::*;

use crate::perm::{replicate_rng, shuffle};

/// `numpy.quantile(sorted, q)` with the default linear method.
fn quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let virtual_index = (n - 1) as f64 * q;
    if virtual_index >= (n - 1) as f64 {
        return sorted[n - 1];
    }
    let previous = virtual_index.floor();
    let gamma = virtual_index - previous;
    let (a, b) = (sorted[previous as usize], sorted[previous as usize + 1]);
    let diff = b - a;
    if gamma >= 0.5 {
        b - diff * (1.0 - gamma)
    } else {
        a + diff * gamma
    }
}

/// The quantile levels of `pandas.qcut(x, k)`: `linspace(0, 1, k + 1)` nudged up one ulp where
/// not exactly representable.
fn qcut_levels(k: usize) -> Vec<f64> {
    let step = 1.0 / k as f64;
    (0..=k)
        .map(|i| {
            let q = if i == k { 1.0 } else { i as f64 * step };
            if k as f64 * q == i as f64 {
                q
            } else {
                q.next_up()
            }
        })
        .collect()
}

/// Group index of `v` in right-closed `edges`, `-1` outside; with `include_lowest` the first
/// edge belongs to the first group (`pandas._bins_to_cuts`).
fn locate(v: f64, edges: &[f64], include_lowest: bool) -> i32 {
    let mut id = edges.partition_point(|&e| e < v);
    if include_lowest && v == edges[0] {
        id = 1;
    }
    if id == 0 || id == edges.len() {
        -1
    } else {
        id as i32 - 1
    }
}

/// Attribute group of each observation (`-1` for none) and the nominal number of groups.
///
/// Quantile groups follow `pandas.qcut(..., labels=False, duplicates="drop")`: collapsed edges
/// are dropped but the number of groups stays `k`, so constant data leaves every observation
/// without a group unless `k == 1`. Explicit bins follow `pandas.cut`: right-closed, the first
/// edge excluded, and values outside the edges are an error (as in PySAL) except those equal to
/// the first edge, which get no group.
pub fn group_labels(x: &[f64], grouping: &Grouping) -> Result<(Vec<i32>, usize), String> {
    match grouping {
        Grouping::Quantiles(k) => {
            if *k < 1 || *k > x.len() {
                return Err(format!(
                    "k must be a positive integer not larger than the {} observations, got {k}",
                    x.len()
                ));
            }
            let mut sorted = x.to_vec();
            sorted.sort_by(f64::total_cmp);
            let mut edges: Vec<f64> = qcut_levels(*k)
                .into_iter()
                .map(|q| quantile(&sorted, q))
                .collect();
            if *k != 1 {
                edges.dedup();
            }
            Ok((x.iter().map(|&v| locate(v, &edges, true)).collect(), *k))
        }
        Grouping::Bins(edges) => {
            if edges.len() < 2 || edges.windows(2).any(|w| w[1] <= w[0]) {
                return Err("bins must be at least two strictly increasing edges".to_string());
            }
            let (lo, hi) = (edges[0], edges[edges.len() - 1]);
            if x.iter().any(|&v| v < lo || v > hi) {
                return Err(format!(
                    "data outside the bin range [{lo}, {hi}]; adjust bins to include the data range"
                ));
            }
            Ok((
                x.iter().map(|&v| locate(v, edges, false)).collect(),
                edges.len() - 1,
            ))
        }
    }
}

/// Union-find over `n` nodes.
struct Dsu(Vec<u32>);

impl Dsu {
    fn new(n: usize) -> Dsu {
        Dsu((0..n as u32).collect())
    }

    fn find(&mut self, mut a: u32) -> u32 {
        while self.0[a as usize] != a {
            self.0[a as usize] = self.0[self.0[a as usize] as usize];
            a = self.0[a as usize];
        }
        a
    }

    fn union(&mut self, a: u32, b: u32) {
        let (ra, rb) = (self.find(a), self.find(b));
        // the smaller index stays the root, so a root is its component's first node
        if ra < rb {
            self.0[rb as usize] = ra;
        } else if rb < ra {
            self.0[ra as usize] = rb;
        }
    }
}

/// Distinct node pairs of the graph's stored entries (its structure, whatever the weights).
fn edge_list(g: &Csr) -> Vec<(u32, u32)> {
    let mut edges = Vec::with_capacity(g.nnz());
    for i in 0..g.n {
        for &j in &g.neighbors[g.offsets[i]..g.offsets[i + 1]] {
            if j as usize != i {
                edges.push((i as u32, j));
            }
        }
    }
    edges
}

/// Components of the same-group subgraph: the roots (first node of each component) and the
/// number of components among the observations that have a group.
fn intersection(labels: &[i32], edges: &[(u32, u32)]) -> (Dsu, usize, usize) {
    let mut dsu = Dsu::new(labels.len());
    for &(i, j) in edges {
        let (a, b) = (labels[i as usize], labels[j as usize]);
        if a >= 0 && a == b {
            dsu.union(i, j);
        }
    }
    let mut components = 0;
    let mut valid = 0;
    for (v, &l) in labels.iter().enumerate() {
        if l >= 0 {
            valid += 1;
            if dsu.find(v as u32) == v as u32 {
                components += 1;
            }
        }
    }
    (dsu, components, valid)
}

fn index_value(components: usize, k: usize, valid: usize) -> f64 {
    let denom = valid.saturating_sub(k).max(1) as f64;
    1.0 - (components as f64 - k as f64) / denom
}

/// Spatial polarization index of `x` over the graph `g`.
///
/// Observations are split into groups (`grouping`); `S = 1 - (c - k) / max(m - k, 1)` with `c`
/// the connected components of the subgraph joining neighbors of the same group, `m` the
/// observations that have a group and `k = max(graph components, groups)`. Only the graph's
/// structure matters: every stored entry is an edge, whatever its weight. With `permutations`
/// the group labels are shuffled over the nodes and `p_value = (#{sim >= S} + 1) / (P + 1)`.
pub fn polarization(
    x: &[f64],
    g: &Csr,
    grouping: &Grouping,
    permutations: usize,
    seed: u64,
) -> Result<Polarization, String> {
    if x.len() != g.n {
        return Err(format!(
            "x has {} values but the graph has {} nodes",
            x.len(),
            g.n
        ));
    }
    let (labels, n_a) = group_labels(x, grouping)?;
    let edges = edge_list(g);
    let n = x.len();
    let mut graph_dsu = Dsu::new(n);
    for &(i, j) in &edges {
        graph_dsu.union(i, j);
    }
    let mut g_labels = vec![0_i64; n];
    let mut n_g = 0;
    let mut root_label = vec![-1_i64; n];
    for (v, label) in g_labels.iter_mut().enumerate() {
        let r = graph_dsu.find(v as u32) as usize;
        if root_label[r] < 0 {
            root_label[r] = n_g as i64;
            n_g += 1;
        }
        *label = root_label[r];
    }
    let k = n_g.max(n_a);
    let (mut dsu, components, valid) = intersection(&labels, &edges);
    let statistic = index_value(components, k, valid);
    let mut i_labels = vec![-1_i64; n];
    let mut next = 0;
    let mut seen = vec![-1_i64; n];
    for v in 0..n {
        if labels[v] >= 0 {
            let r = dsu.find(v as u32) as usize;
            if seen[r] < 0 {
                seen[r] = next;
                next += 1;
            }
            i_labels[v] = seen[r];
        }
    }
    let mut out = Polarization {
        statistic,
        n_a_components: n_a,
        n_g_components: n_g,
        n_i_components: components,
        i_labels,
        a_labels: labels.iter().map(|&l| i64::from(l)).collect(),
        g_labels,
        p_value: f64::NAN,
        sim: Vec::new(),
    };
    if permutations > 0 {
        out.sim = (0..permutations)
            .into_par_iter()
            .map(|r| {
                let mut shuffled = labels.clone();
                shuffle(&mut shuffled, &mut replicate_rng(seed, r as u64));
                let (_, c, m) = intersection(&shuffled, &edges);
                index_value(c, k, m)
            })
            .collect();
        let above = out.sim.iter().filter(|&&v| v >= statistic).count();
        out.p_value = (above as f64 + 1.0) / (permutations as f64 + 1.0);
    }
    Ok(out)
}
