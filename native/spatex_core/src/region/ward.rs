//! Spatially constrained Ward clustering, a port of scikit-learn's `ward_tree` with a
//! connectivity matrix (`AgglomerativeClustering(linkage="ward", connectivity=...)`), including
//! its connectivity completion, tie-breaking and label numbering.

use super::util::{sq_dist, Adj, Mat};
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Result of [`ward`].
#[derive(Clone, Debug)]
pub struct WardOut {
    /// Cluster of each area.
    pub labels: Vec<u32>,
    /// Merges `(a, b)` with `a < b`; merge `i` creates node `n + i`.
    pub children: Vec<(u32, u32)>,
    /// `sqrt(2 * inertia)` of each merge (not monotone under connectivity constraints).
    pub distances: Vec<f64>,
    /// Connected components of the input graph before completion.
    pub n_connected_components: usize,
    /// Warnings.
    pub warnings: Vec<String>,
}

#[derive(Clone, Copy, PartialEq)]
struct Key {
    inertia: f64,
    r: u32,
    c: u32,
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.inertia
            .partial_cmp(&other.inertia)
            .unwrap_or(Ordering::Equal)
            .then(self.r.cmp(&other.r))
            .then(self.c.cmp(&other.c))
    }
}

/// `compute_ward_dist` for one pair, with sklearn's exact expression order.
fn ward_dist(m1: &[f64], m2: &[f64], p: usize, r: usize, c: usize) -> f64 {
    let n = (m1[r] * m1[c]) / (m1[r] + m1[c]);
    let mut pa = 0.0;
    for f in 0..p {
        let d = m2[r * p + f] / m1[r] - m2[c * p + f] / m1[c];
        pa += d * d;
    }
    pa * n
}

/// Adds, for every pair of components `(i, j)` with `j < i`, the edge between their most
/// similar areas in attribute space (`_fix_connected_components`).
fn complete(adj: &mut Adj, x: &Mat, count: usize, labels: &[u32]) {
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); count];
    for (i, &l) in labels.iter().enumerate() {
        members[l as usize].push(i as u32);
    }
    let mut added: Vec<(u32, u32)> = Vec::new();
    for i in 0..count {
        for j in 0..i {
            let mut best = (f64::INFINITY, 0u32, 0u32);
            for &a in &members[i] {
                for &b in &members[j] {
                    let d = sq_dist(x.row(a as usize), x.row(b as usize));
                    if d < best.0 {
                        best = (d, a, b);
                    }
                }
            }
            added.push((best.1, best.2));
        }
    }
    for (a, b) in added {
        adj.nb[a as usize].push(b);
        adj.nb[b as usize].push(a);
    }
    for row in &mut adj.nb {
        row.sort_unstable();
        row.dedup();
    }
}

/// Runs spatial Ward. `adj` must be symmetric (it is completed when disconnected).
pub fn ward(adj: &Adj, x: &Mat, n_clusters: usize) -> Result<WardOut, String> {
    let n = adj.n();
    let p = x.p;
    if n_clusters == 0 || n_clusters > n {
        return Err(format!(
            "n_clusters must be between 1 and {n}, got {n_clusters}"
        ));
    }
    let mut warnings = Vec::new();
    let mut adj = adj.clone();
    let (ncc, comp) = adj.components();
    if ncc > 1 {
        warnings.push(format!(
            "the number of connected components of the connectivity matrix is {ncc} > 1. \
             Completing it to avoid stopping the tree early."
        ));
        complete(&mut adj, x, ncc, &comp);
    }
    if n == 1 {
        return Ok(WardOut {
            labels: vec![0],
            children: Vec::new(),
            distances: Vec::new(),
            n_connected_components: ncc,
            warnings,
        });
    }
    let full = (n_clusters as f64) < f64::max(100.0, 0.02 * n as f64);
    let n_nodes = if full { 2 * n - 1 } else { 2 * n - n_clusters };

    let mut m1 = vec![0.0; n_nodes];
    let mut m2 = vec![0.0; n_nodes * p];
    for i in 0..n {
        m1[i] = 1.0;
        m2[i * p..(i + 1) * p].copy_from_slice(x.row(i));
    }
    let mut a_lists: Vec<Vec<u32>> = adj.nb.clone();
    let mut heap: BinaryHeap<Reverse<Key>> = BinaryHeap::new();
    for (ind, row) in adj.nb.iter().enumerate() {
        for &i in row.iter().filter(|&&i| (i as usize) < ind) {
            heap.push(Reverse(Key {
                inertia: ward_dist(&m1, &m2, p, ind, i as usize),
                r: ind as u32,
                c: i,
            }));
        }
    }
    let mut parent: Vec<u32> = (0..n_nodes as u32).collect();
    let mut used = vec![true; n_nodes];
    let mut children: Vec<(u32, u32)> = Vec::with_capacity(n_nodes - n);
    let mut distances: Vec<f64> = Vec::with_capacity(n_nodes - n);
    let mut not_visited = vec![true; n_nodes];

    for k in n..n_nodes {
        let (inertia, i, j) = loop {
            let Reverse(key) = heap
                .pop()
                .ok_or_else(|| "the connectivity graph ran out of mergeable pairs".to_string())?;
            if used[key.r as usize] && used[key.c as usize] {
                break (key.inertia, key.r as usize, key.c as usize);
            }
        };
        parent[i] = k as u32;
        parent[j] = k as u32;
        children.push((j as u32, i as u32));
        used[i] = false;
        used[j] = false;
        distances.push(inertia);
        m1[k] = m1[i] + m1[j];
        for f in 0..p {
            m2[k * p + f] = m2[i * p + f] + m2[j * p + f];
        }
        not_visited.iter_mut().for_each(|v| *v = true);
        not_visited[k] = false;
        let mut heads: Vec<u32> = Vec::new();
        for src in [i, j] {
            for &node in &a_lists[src] {
                let mut cur = node as usize;
                while parent[cur] as usize != cur {
                    cur = parent[cur] as usize;
                }
                if not_visited[cur] {
                    not_visited[cur] = false;
                    heads.push(cur as u32);
                }
            }
        }
        for &h in &heads {
            a_lists[h as usize].push(k as u32);
        }
        for &h in &heads {
            heap.push(Reverse(Key {
                inertia: ward_dist(&m1, &m2, p, k, h as usize),
                r: k as u32,
                c: h,
            }));
        }
        a_lists.push(heads);
    }
    let distances: Vec<f64> = distances.iter().map(|d| (2.0 * d).sqrt()).collect();

    let labels = if full {
        hc_cut(n_clusters, &children, n)
    } else {
        heads_labels(&parent, n)
    };
    Ok(WardOut {
        labels,
        children,
        distances,
        n_connected_components: ncc,
        warnings,
    })
}

// -- CPython heapq, ported literally: sklearn numbers clusters by the heap's array layout. --

fn siftdown(heap: &mut [i64], startpos: usize, mut pos: usize) {
    let newitem = heap[pos];
    while pos > startpos {
        let parentpos = (pos - 1) >> 1;
        let parent = heap[parentpos];
        if newitem < parent {
            heap[pos] = parent;
            pos = parentpos;
            continue;
        }
        break;
    }
    heap[pos] = newitem;
}

fn siftup(heap: &mut [i64], mut pos: usize) {
    let endpos = heap.len();
    let startpos = pos;
    let newitem = heap[pos];
    let mut childpos = 2 * pos + 1;
    while childpos < endpos {
        let rightpos = childpos + 1;
        if rightpos < endpos && heap[childpos] >= heap[rightpos] {
            childpos = rightpos;
        }
        heap[pos] = heap[childpos];
        pos = childpos;
        childpos = 2 * pos + 1;
    }
    heap[pos] = newitem;
    siftdown(heap, startpos, pos);
}

fn heappush(heap: &mut Vec<i64>, item: i64) {
    heap.push(item);
    let last = heap.len() - 1;
    siftdown(heap, 0, last);
}

fn heappushpop(heap: &mut [i64], item: i64) -> i64 {
    if !heap.is_empty() && heap[0] < item {
        let top = heap[0];
        heap[0] = item;
        siftup(heap, 0);
        return top;
    }
    item
}

fn descendants(node: usize, children: &[(u32, u32)], n: usize, out: &mut Vec<usize>) {
    let mut stack = vec![node];
    while let Some(v) = stack.pop() {
        if v < n {
            out.push(v);
        } else {
            let (a, b) = children[v - n];
            stack.push(b as usize);
            stack.push(a as usize);
        }
    }
}

/// `_hc_cut`: cut the full tree into `n_clusters` clusters.
fn hc_cut(n_clusters: usize, children: &[(u32, u32)], n: usize) -> Vec<u32> {
    let last = children[children.len() - 1];
    let mut nodes: Vec<i64> = vec![-(i64::from(last.0.max(last.1)) + 1)];
    for _ in 0..n_clusters - 1 {
        let (a, b) = children[(-nodes[0]) as usize - n];
        heappush(&mut nodes, -i64::from(a));
        heappushpop(&mut nodes, -i64::from(b));
    }
    let mut labels = vec![0u32; n];
    let mut buf = Vec::new();
    for (i, &node) in nodes.iter().enumerate() {
        buf.clear();
        descendants((-node) as usize, children, n, &mut buf);
        for &leaf in &buf {
            labels[leaf] = i as u32;
        }
    }
    labels
}

/// Partial tree: clusters are numbered by the rank of their head node.
fn heads_labels(parent: &[u32], n: usize) -> Vec<u32> {
    let head = |mut v: usize| {
        while parent[v] as usize != v {
            v = parent[v] as usize;
        }
        v
    };
    let heads: Vec<usize> = (0..n).map(head).collect();
    let mut unique = heads.clone();
    unique.sort_unstable();
    unique.dedup();
    heads
        .iter()
        .map(|h| unique.binary_search(h).unwrap_or(0) as u32)
        .collect()
}

#[cfg(test)]
mod heap_tests {
    use super::*;

    #[test]
    fn heap_layout_matches_cpython() {
        // Verified against CPython's heapq with the same push/pushpop sequence.
        let mut h: Vec<i64> = vec![-20];
        for (a, b) in [(-15, -9), (-14, -8), (-13, -7), (-12, -6)] {
            heappush(&mut h, a);
            heappushpop(&mut h, b);
        }
        assert_eq!(h, vec![-12, -9, -8, -7, -6]);
    }

    #[test]
    fn pushpop_returns_the_item_when_it_is_the_smallest() {
        let mut h: Vec<i64> = vec![1, 5, 3];
        assert_eq!(heappushpop(&mut h, 0), 0);
        assert_eq!(h, vec![1, 5, 3]);
        assert_eq!(heappushpop(&mut h, 4), 1);
        assert_eq!(h[0], 3);
    }
}
