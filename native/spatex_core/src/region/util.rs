//! Shared helpers: adjacency lists, components, attribute distances and label utilities.

use crate::graph::Csr;

/// Undirected adjacency: sorted, duplicate-free neighbor lists without self-loops.
#[derive(Clone, Debug, PartialEq)]
pub struct Adj {
    /// Neighbors of each node in ascending order.
    pub nb: Vec<Vec<u32>>,
}

impl Adj {
    /// Number of nodes.
    pub fn n(&self) -> usize {
        self.nb.len()
    }

    /// Builds the adjacency pattern of `csr`: self-loops (the island encoding) and zero weights
    /// are dropped. An asymmetric pattern is an error unless `symmetrize`, which keeps `B or B'`.
    pub fn from_csr(csr: &Csr, symmetrize: bool) -> Result<Adj, String> {
        let n = csr.n;
        let mut nb: Vec<Vec<u32>> = vec![Vec::new(); n];
        for (i, row) in nb.iter_mut().enumerate() {
            for e in csr.offsets[i]..csr.offsets[i + 1] {
                let j = csr.neighbors[e];
                if j as usize != i && csr.weights[e] != 0.0 {
                    row.push(j);
                }
            }
        }
        let mut missing: Vec<(usize, u32)> = Vec::new();
        for (i, row) in nb.iter().enumerate() {
            for &j in row {
                if nb[j as usize].binary_search(&(i as u32)).is_err() {
                    missing.push((j as usize, i as u32));
                }
            }
        }
        if !missing.is_empty() {
            if !symmetrize {
                return Err(
                    "the graph is not symmetric; pass symmetrize: true to use its symmetrized pattern"
                        .into(),
                );
            }
            for (j, i) in missing {
                nb[j].push(i);
            }
            for row in &mut nb {
                row.sort_unstable();
                row.dedup();
            }
        }
        Ok(Adj { nb })
    }

    /// Builds an adjacency from explicit undirected edges (duplicates and loops ignored).
    pub fn from_edges(n: usize, edges: &[(u32, u32)]) -> Adj {
        let mut nb: Vec<Vec<u32>> = vec![Vec::new(); n];
        for &(a, b) in edges {
            if a != b {
                nb[a as usize].push(b);
                nb[b as usize].push(a);
            }
        }
        for row in &mut nb {
            row.sort_unstable();
            row.dedup();
        }
        Adj { nb }
    }

    /// Connected components, labeled in order of their lowest node (scipy's convention).
    pub fn components(&self) -> (usize, Vec<u32>) {
        let n = self.n();
        let mut label = vec![u32::MAX; n];
        let mut count = 0u32;
        let mut stack = Vec::new();
        for s in 0..n {
            if label[s] != u32::MAX {
                continue;
            }
            label[s] = count;
            stack.push(s as u32);
            while let Some(a) = stack.pop() {
                for &b in &self.nb[a as usize] {
                    if label[b as usize] == u32::MAX {
                        label[b as usize] = count;
                        stack.push(b);
                    }
                }
            }
            count += 1;
        }
        (count as usize, label)
    }

    /// True when `nodes` (already known members, flagged in `member`) induce a connected
    /// subgraph, ignoring `skip`. An empty set is not connected.
    pub fn connected_subset(&self, nodes: &[u32], member: &[bool], skip: Option<u32>) -> bool {
        let Some(&start) = nodes.iter().find(|&&v| Some(v) != skip) else {
            return false;
        };
        let total = nodes.iter().filter(|&&v| Some(v) != skip).count();
        let mut seen = vec![false; self.n()];
        seen[start as usize] = true;
        let mut stack = vec![start];
        let mut reached = 1;
        while let Some(a) = stack.pop() {
            for &b in &self.nb[a as usize] {
                let bi = b as usize;
                if member[bi] && !seen[bi] && Some(b) != skip {
                    seen[bi] = true;
                    reached += 1;
                    stack.push(b);
                }
            }
        }
        reached == total
    }
}

/// Attribute distance used by objectives and dissimilarities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// `sum |a - b|`
    Manhattan,
    /// `sqrt(sum (a - b)^2)`
    Euclidean,
    /// `sum (a - b)^2`
    SqEuclidean,
}

impl Metric {
    /// Parses `manhattan`, `euclidean` or `sqeuclidean`.
    pub fn parse(s: &str) -> Result<Metric, String> {
        match s {
            "manhattan" => Ok(Metric::Manhattan),
            "euclidean" => Ok(Metric::Euclidean),
            "sqeuclidean" => Ok(Metric::SqEuclidean),
            other => Err(format!("unknown metric {other:?}")),
        }
    }

    /// Distance between two attribute rows.
    pub fn dist(self, a: &[f64], b: &[f64]) -> f64 {
        match self {
            Metric::Manhattan => a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum(),
            Metric::Euclidean => sq_dist(a, b).sqrt(),
            Metric::SqEuclidean => sq_dist(a, b),
        }
    }
}

/// Squared Euclidean distance, accumulated feature by feature.
pub fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        let d = x - y;
        s += d * d;
    }
    s
}

/// Row-major `n x p` attribute matrix.
#[derive(Clone, Copy, Debug)]
pub struct Mat<'a> {
    /// Rows.
    pub n: usize,
    /// Columns.
    pub p: usize,
    /// Row-major values.
    pub data: &'a [f64],
}

impl<'a> Mat<'a> {
    /// Wraps a row-major buffer.
    pub fn new(data: &'a [f64], n: usize, p: usize) -> Result<Mat<'a>, String> {
        if p == 0 || n == 0 {
            return Err("the attribute matrix is empty".into());
        }
        if data.len() != n * p {
            return Err(format!(
                "attribute matrix has {} values, expected {n} x {p}",
                data.len()
            ));
        }
        if data.iter().any(|v| !v.is_finite()) {
            return Err("attributes must be finite".into());
        }
        Ok(Mat { n, p, data })
    }

    /// Row `i`.
    #[inline]
    pub fn row(&self, i: usize) -> &'a [f64] {
        &self.data[i * self.p..(i + 1) * self.p]
    }

    /// Mean of the rows listed in `idx`, summed in the given order.
    pub fn mean_of(&self, idx: impl Iterator<Item = u32> + Clone) -> Vec<f64> {
        let mut sum = vec![0.0; self.p];
        let mut count = 0usize;
        for i in idx {
            for (s, v) in sum.iter_mut().zip(self.row(i as usize)) {
                *s += v;
            }
            count += 1;
        }
        let c = count as f64;
        sum.iter_mut().for_each(|s| *s /= c);
        sum
    }
}

/// Converts a column-major buffer (`p` columns of `n` values) into row-major order.
pub fn columns_to_rows(cols: &[f64], n: usize, p: usize) -> Result<Vec<f64>, String> {
    if cols.len() != n * p {
        return Err(format!(
            "attribute buffer has {} values, expected {p} columns of {n}",
            cols.len()
        ));
    }
    let mut rows = vec![0.0; n * p];
    for f in 0..p {
        for i in 0..n {
            rows[i * p + f] = cols[f * n + i];
        }
    }
    Ok(rows)
}

/// Renumbers labels `0..k` by order of first appearance.
pub fn relabel<T: Copy + Eq + std::hash::Hash>(labels: &[T]) -> Vec<u32> {
    let mut map = std::collections::HashMap::new();
    labels
        .iter()
        .map(|l| {
            let next = map.len() as u32;
            *map.entry(*l).or_insert(next)
        })
        .collect()
}

/// Region ids that do not induce a connected subgraph.
pub fn non_contiguous_regions(adj: &Adj, labels: &[u32]) -> Vec<u32> {
    let n = adj.n();
    let mut by_region: std::collections::BTreeMap<u32, Vec<u32>> = Default::default();
    for (i, &l) in labels.iter().enumerate().take(n) {
        by_region.entry(l).or_default().push(i as u32);
    }
    let mut bad = Vec::new();
    let mut member = vec![false; n];
    for (r, nodes) in &by_region {
        nodes.iter().for_each(|&v| member[v as usize] = true);
        if !adj.connected_subset(nodes, &member, None) {
            bad.push(*r);
        }
        nodes.iter().for_each(|&v| member[v as usize] = false);
    }
    bad
}

/// Union-find with path halving.
pub struct Dsu {
    parent: Vec<u32>,
}

impl Dsu {
    /// `n` singletons.
    pub fn new(n: usize) -> Dsu {
        Dsu {
            parent: (0..n as u32).collect(),
        }
    }

    /// Representative of `x`.
    pub fn find(&mut self, mut x: u32) -> u32 {
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        x
    }

    /// Joins the sets of `a` and `b`; false when already joined.
    pub fn union(&mut self, a: u32, b: u32) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        self.parent[ra as usize] = rb;
        true
    }
}

/// Component labels (lowest-node order) of `n` nodes joined by `edges`.
pub fn edge_components(n: usize, edges: &[(u32, u32)]) -> (usize, Vec<u32>) {
    let mut dsu = Dsu::new(n);
    for &(a, b) in edges {
        dsu.union(a, b);
    }
    let mut ids = vec![u32::MAX; n];
    let mut count = 0u32;
    let mut labels = vec![0u32; n];
    for (i, l) in labels.iter_mut().enumerate() {
        let r = dsu.find(i as u32) as usize;
        if ids[r] == u32::MAX {
            ids[r] = count;
            count += 1;
        }
        *l = ids[r];
    }
    (count as usize, labels)
}
