//! Spenc (spatially encouraged spectral clustering, Wolf 2018), a port of `spopt.region.Spenc`
//! and `spopt.region.spenclib.SPENC.fit`.
//!
//! The stages spopt computes deterministically are ported exactly: the affinity matrix (a
//! `sklearn.metrics.pairwise_kernels` kernel of the attributes times the weights `W`
//! elementwise), the normalized Laplacian `L = I - D^-1/2 A D^-1/2` (scipy's `laplacian` with
//! `normed=True`), the `k` eigenvectors of `-L` with the largest eigenvalues, divided by
//! `sqrt(degree)` and sign-normalized (largest magnitude entry positive). spopt finds the
//! eigenpairs with ARPACK (`eigsh`, shift-invert); this port uses a dense symmetric
//! eigendecomposition (`faer`), which yields the same eigenpairs to numerical precision but is
//! cubic in the number of areas.
//!
//! Label assignment is [`super::spenc_labels`] (`discretize` or k-means) from a `ChaCha8` stream,
//! or Shi and Malik's recursive bipartition (`assign_labels="hierarchical"`), which is
//! deterministic.

use std::collections::VecDeque;

use faer::{Mat, Side};

use super::spenc_labels::{discretize, kmeans};
use super::util::{edge_components, Mat as Attrs};
use crate::graph::Csr;
use crate::limits::{checked_product, ensure_within, MAX_SPECTRAL_CELLS};
use crate::perm::{percentile, replicate_rng};

/// Attribute affinity (scikit-learn's `pairwise_kernels` names, `nearest_neighbors`,
/// `precomputed`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// `exp(-gamma |x - y|^2)`
    Rbf,
    /// `exp(-gamma |x - y|_1)`
    Laplacian,
    /// `x . y`
    Linear,
    /// `(gamma x . y + coef0)^degree`
    Polynomial,
    /// `tanh(gamma x . y + coef0)`
    Sigmoid,
    /// Cosine similarity.
    Cosine,
    /// Symmetrized `n_neighbors` connectivity graph (self included); `W` is not applied.
    NearestNeighbors,
    /// The attributes are the affinity matrix; `W` is not applied.
    Precomputed,
}

impl Kernel {
    /// Parses a kernel name (`poly` is accepted for `polynomial`).
    pub fn parse(s: &str) -> Result<Kernel, String> {
        Ok(match s {
            "rbf" => Kernel::Rbf,
            "laplacian" => Kernel::Laplacian,
            "linear" => Kernel::Linear,
            "polynomial" | "poly" => Kernel::Polynomial,
            "sigmoid" => Kernel::Sigmoid,
            "cosine" => Kernel::Cosine,
            "nearest_neighbors" => Kernel::NearestNeighbors,
            "precomputed" => Kernel::Precomputed,
            other => return Err(format!("unknown affinity {other:?}")),
        })
    }
}

/// How labels are assigned in the embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Assign {
    /// Yu and Shi's rotation (`sklearn` `discretize`).
    Discretize,
    /// k-means++ and Lloyd.
    Kmeans,
    /// Recursive spectral bipartition (Shi and Malik).
    Hierarchical,
}

impl Assign {
    /// Parses `discretize`, `kmeans` or `hierarchical`.
    pub fn parse(s: &str) -> Result<Assign, String> {
        Ok(match s {
            "discretize" => Assign::Discretize,
            "kmeans" => Assign::Kmeans,
            "hierarchical" => Assign::Hierarchical,
            other => return Err(format!("unknown assign_labels {other:?}")),
        })
    }
}

/// Kernel parameters.
#[derive(Clone, Debug)]
pub struct KernelParams {
    /// Kernel coefficient (rbf, laplacian, polynomial, sigmoid).
    pub gamma: f64,
    /// Polynomial degree.
    pub degree: f64,
    /// Polynomial and sigmoid offset.
    pub coef0: f64,
    /// Neighbors of the nearest-neighbors affinity.
    pub n_neighbors: usize,
}

/// Options of [`spenc`].
#[derive(Clone, Debug)]
pub struct SpencOpts {
    /// Number of clusters; `None` is infinity (hierarchical only).
    pub n_clusters: Option<usize>,
    /// Attribute affinity.
    pub kernel: Kernel,
    /// Kernel parameters.
    pub params: KernelParams,
    /// Label assignment.
    pub assign: Assign,
    /// Hierarchical: lower limit of the weighted size of both sides of a cut.
    pub floor: f64,
    /// Hierarchical: area weights for the floor (default ones).
    pub floor_weights: Option<Vec<f64>>,
    /// Hierarchical: grid points of the cut search.
    pub grid_resolution: usize,
    /// Seed of the random stages.
    pub seed: u64,
}

/// The spectral embedding of an affinity matrix.
#[derive(Clone, Debug)]
pub struct Embedding {
    /// `n x k`, row-major (spopt's `embedding_`).
    pub vectors: Vec<f64>,
    /// The `k` smallest eigenvalues of the normalized Laplacian, ascending.
    pub eigenvalues: Vec<f64>,
}

/// Result of [`spenc`].
#[derive(Clone, Debug)]
pub struct SpencOut {
    /// Cluster of each area.
    pub labels: Vec<u32>,
    /// Dense `n x n` affinity, row-major.
    pub affinity: Vec<f64>,
    /// The embedding (none for hierarchical).
    pub embedding: Option<Embedding>,
    /// Row `discretize` started from.
    pub start_row: Option<usize>,
    /// Warnings.
    pub warnings: Vec<String>,
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn kernel_value(kernel: Kernel, p: &KernelParams, a: &[f64], b: &[f64]) -> f64 {
    match kernel {
        Kernel::Rbf => {
            (-p.gamma * a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f64>()).exp()
        }
        Kernel::Laplacian => {
            (-p.gamma * a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f64>()).exp()
        }
        Kernel::Linear => dot(a, b),
        Kernel::Polynomial => (p.gamma * dot(a, b) + p.coef0).powf(p.degree),
        Kernel::Sigmoid => (p.gamma * dot(a, b) + p.coef0).tanh(),
        Kernel::Cosine => {
            let (na, nb) = (dot(a, a).sqrt(), dot(b, b).sqrt());
            if na == 0.0 || nb == 0.0 {
                0.0
            } else {
                dot(a, b) / na / nb
            }
        }
        Kernel::NearestNeighbors | Kernel::Precomputed => unreachable!(),
    }
}

/// The dense `n x n` affinity spopt builds: `W.multiply(kernel(X))` for the kernels,
/// `0.5 (C + C')` for `nearest_neighbors` and `X` itself for `precomputed`.
pub fn affinity_matrix(
    csr: &Csr,
    x: &Attrs,
    kernel: Kernel,
    p: &KernelParams,
) -> Result<Vec<f64>, String> {
    let n = csr.n;
    ensure_within(
        "spenc affinity matrix",
        checked_product("spenc affinity matrix", n, n)?,
        MAX_SPECTRAL_CELLS,
        "the dense eigendecomposition limits spenc to about 5000 areas",
    )?;
    if x.n != n {
        return Err(format!(
            "attributes have {} rows but the graph has {n} nodes",
            x.n
        ));
    }
    match kernel {
        Kernel::Precomputed => {
            if x.p != n {
                return Err(format!(
                    "a precomputed affinity must be {n} x {n}, got {} x {}",
                    x.n, x.p
                ));
            }
            Ok(x.data.to_vec())
        }
        Kernel::NearestNeighbors => {
            if p.n_neighbors == 0 || p.n_neighbors > n {
                return Err(format!("n_neighbors must be between 1 and {n}"));
            }
            let mut c = vec![0.0; n * n];
            for i in 0..n {
                let mut order: Vec<(f64, usize)> = (0..n)
                    .map(|j| {
                        let d: f64 = x
                            .row(i)
                            .iter()
                            .zip(x.row(j))
                            .map(|(a, b)| (a - b) * (a - b))
                            .sum();
                        (d, j)
                    })
                    .collect();
                order.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                for &(_, j) in order.iter().take(p.n_neighbors) {
                    c[i * n + j] = 1.0;
                }
            }
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    a[i * n + j] = 0.5 * (c[i * n + j] + c[j * n + i]);
                }
            }
            Ok(a)
        }
        _ => {
            let mut a = vec![0.0; n * n];
            for i in 0..n {
                for e in csr.offsets[i]..csr.offsets[i + 1] {
                    let j = csr.neighbors[e] as usize;
                    let w = csr.weights[e];
                    if j != i && w != 0.0 {
                        a[i * n + j] = w * kernel_value(kernel, p, x.row(i), x.row(j));
                    }
                }
            }
            Ok(a)
        }
    }
}

/// Spectral embedding of the affinity `a` (`n x n`, row-major, symmetric): spopt's `_embed`.
///
/// The diagonal is ignored, as scipy's `laplacian` does. Isolated areas (zero degree) get a
/// unit scale and a zero Laplacian diagonal.
pub fn embed(a: &[f64], n: usize, k: usize) -> Result<Embedding, String> {
    if k == 0 || k >= n {
        return Err(format!(
            "n_clusters must be between 1 and {} for {n} areas to embed",
            n.saturating_sub(1)
        ));
    }
    if a.len() != n * n {
        return Err("the affinity matrix is not square".into());
    }
    for i in 0..n {
        for j in (i + 1)..n {
            let (x, y) = (a[i * n + j], a[j * n + i]);
            if !x.is_finite() || !y.is_finite() {
                return Err("the affinity matrix must be finite".into());
            }
            if (x - y).abs() > 1e-10 * x.abs().max(y.abs()).max(1.0) {
                return Err(
                    "the affinity matrix must be symmetric (the graph weights times the attribute affinity)"
                        .into(),
                );
            }
        }
    }
    let degree: Vec<f64> = (0..n)
        .map(|j| (0..n).filter(|&i| i != j).map(|i| a[i * n + j]).sum())
        .collect();
    let isolated: Vec<bool> = degree.iter().map(|&d| d == 0.0).collect();
    let w: Vec<f64> = degree
        .iter()
        .zip(&isolated)
        .map(|(&d, &iso)| if iso { 1.0 } else { d.sqrt() })
        .collect();
    // -L: off diagonal a_ij / (w_i w_j), diagonal -(1 - isolated).
    let m = Mat::from_fn(n, n, |i, j| {
        if i == j {
            -(if isolated[i] { 0.0 } else { 1.0 })
        } else {
            a[i * n + j] / (w[i] * w[j])
        }
    });
    let eig = m
        .self_adjoint_eigen(Side::Lower)
        .map_err(|e| format!("eigenvalue decomposition failed: {e:?}"))?;
    let (u, s) = (eig.U(), eig.S().column_vector());
    let mut vectors = vec![0.0; n * k];
    let mut eigenvalues = Vec::with_capacity(k);
    for c in 0..k {
        let col = n - 1 - c; // eigenvalues of -L ascending: take the largest first
        eigenvalues.push(0.0 - s[col]);
        let mut big = (0.0f64, 0.0f64);
        for i in 0..n {
            let v = u[(i, col)] / w[i];
            vectors[i * k + c] = v;
            if v.abs() > big.0 {
                big = (v.abs(), v);
            }
        }
        if big.1 < 0.0 {
            (0..n).for_each(|i| vectors[i * k + c] = -vectors[i * k + c]);
        }
    }
    Ok(Embedding {
        vectors,
        eigenvalues,
    })
}

/// Shi and Malik's cut of `second` (spopt's `_make_hierarchical_cut`, grid search): the cut
/// `second <= t` of the `grid` points `t` between the 2nd and 98th percentiles that minimizes
/// the normalized cut of the affinity `a` (`m x m`).
fn grid_cut(second: &[f64], a: &[f64], m: usize, grid: usize) -> Vec<bool> {
    let mut sorted = second.to_vec();
    sorted.sort_by(f64::total_cmp);
    let (lo, hi) = (percentile(&sorted, 2.0), percentile(&sorted, 98.0));
    let step = if grid > 1 {
        (hi - lo) / (grid - 1) as f64
    } else {
        0.0
    };
    let row_sum: Vec<f64> = (0..m).map(|i| a[i * m..(i + 1) * m].iter().sum()).collect();
    let mut best: Option<(f64, f64)> = None; // (score, cutpoint)
    for g in 0..grid {
        let t = if g + 1 == grid && grid > 1 {
            hi
        } else {
            lo + g as f64 * step
        };
        let cut: Vec<bool> = second.iter().map(|&v| v <= t).collect();
        let (mut assoc_a, mut assoc_b, mut cross) = (0.0, 0.0, 0.0);
        for i in 0..m {
            if cut[i] {
                assoc_a += row_sum[i];
                cross += (0..m)
                    .filter(|&j| !cut[j])
                    .map(|j| a[i * m + j])
                    .sum::<f64>();
            } else {
                assoc_b += row_sum[i];
            }
        }
        let cross = cross * 2.0;
        let mut score = cross / assoc_a + cross / assoc_b;
        if score.is_nan() {
            score = f64::INFINITY;
        }
        if best.is_none_or(|b| score < b.0) {
            best = Some((score, t));
        }
    }
    let t = best.map_or(lo, |b| b.1);
    second.iter().map(|&v| v <= t).collect()
}

/// spopt's `_spectral_bipartition` (`cut_method="gridsearch"`).
fn bipartition(
    a: &[f64],
    n: usize,
    n_clusters: Option<usize>,
    floor: f64,
    weights: &[f64],
    grid: usize,
) -> Result<Vec<u32>, String> {
    let threshold = n_clusters.unwrap_or(usize::MAX);
    let mut this_cut = vec![true; n];
    let mut cuts: VecDeque<Vec<bool>> = VecDeque::new();
    let mut accepted: Vec<Vec<bool>> = Vec::new();
    let mut discovered = 1;
    while discovered < threshold {
        let idx: Vec<usize> = (0..n).filter(|&i| this_cut[i]).collect();
        let m = idx.len();
        let sub: Vec<f64> = idx
            .iter()
            .flat_map(|&i| idx.iter().map(move |&j| (i, j)))
            .map(|(i, j)| a[i * n + j])
            .collect();
        let emb = embed(&sub, m, 2).map_err(|e| {
            format!("the hierarchical bipartition cannot split a part of {m} areas: {e}")
        })?;
        let second: Vec<f64> = (0..m).map(|i| emb.vectors[i * 2 + 1]).collect();
        let cut = grid_cut(&second, &sub, m, grid);
        let mut left = this_cut.clone();
        let mut right = this_cut.clone();
        for (pos, &g) in idx.iter().enumerate() {
            left[g] = cut[pos];
            right[g] = !cut[pos];
        }
        let size = |side: &[bool]| -> f64 {
            side.iter()
                .zip(weights)
                .filter(|(&s, _)| s)
                .map(|(_, w)| w)
                .sum()
        };
        if size(&left) > floor
            && size(&right) > floor
            && !accepted.contains(&left)
            && !accepted.contains(&right)
        {
            accepted.push(left.clone());
            accepted.push(right.clone());
            cuts.push_back(left);
            cuts.push_back(right);
        }
        discovered += 1;
        match cuts.pop_front() {
            Some(next) => this_cut = next,
            None => break,
        }
    }
    if accepted.is_empty() {
        return Err(
            "the hierarchical bipartition accepted no cut (check floor and n_clusters)".into(),
        );
    }
    let mut labels: Vec<i64> = vec![-1; n];
    for (i, cut) in accepted.iter().rev().enumerate() {
        for v in 0..n {
            if labels[v] == -1 && cut[v] {
                labels[v] = i as i64;
            }
        }
    }
    let mut uniq = labels.clone();
    uniq.sort_unstable();
    uniq.dedup();
    Ok(labels
        .iter()
        .map(|l| uniq.binary_search(l).expect("present") as u32)
        .collect())
}

/// Runs Spenc on the areas of `csr` with attributes `x`.
pub fn spenc(csr: &Csr, x: &Attrs, opts: &SpencOpts) -> Result<SpencOut, String> {
    let n = csr.n;
    if n < 2 {
        return Err("spenc needs at least two areas".into());
    }
    let assign = if opts.n_clusters.is_none() {
        Assign::Hierarchical
    } else {
        opts.assign
    };
    if let Some(k) = opts.n_clusters {
        if k == 0 || k >= n {
            return Err(format!(
                "n_clusters must be between 1 and {} (fewer than the areas)",
                n - 1
            ));
        }
        if assign == Assign::Hierarchical && k < 2 {
            return Err("the hierarchical bipartition needs n_clusters >= 2".into());
        }
    }
    let weights = match &opts.floor_weights {
        Some(w) if w.len() != n => {
            return Err(format!(
                "floor_weights has {} entries for {n} areas",
                w.len()
            ))
        }
        Some(w) => w.clone(),
        None => vec![1.0; n],
    };
    if opts.grid_resolution == 0 {
        return Err("grid_resolution must be positive".into());
    }
    let affinity = affinity_matrix(csr, x, opts.kernel, &opts.params)?;
    let mut warnings = Vec::new();
    let edges: Vec<(u32, u32)> = (0..n)
        .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
        .filter(|&(i, j)| affinity[i * n + j] != 0.0 || affinity[j * n + i] != 0.0)
        .map(|(i, j)| (i as u32, j as u32))
        .collect();
    let (components, _) = edge_components(n, &edges);
    if components > 1 {
        warnings.push(format!(
            "the affinity graph is disconnected and has {components} subcomponents; this will certainly affect the solution"
        ));
    }
    if assign == Assign::Hierarchical {
        let labels = bipartition(
            &affinity,
            n,
            opts.n_clusters,
            opts.floor,
            &weights,
            opts.grid_resolution,
        )?;
        return Ok(SpencOut {
            labels,
            affinity,
            embedding: None,
            start_row: None,
            warnings,
        });
    }
    let k = opts.n_clusters.expect("finite for non-hierarchical");
    let embedding = embed(&affinity, n, k)?;
    let mut rng = replicate_rng(opts.seed, 0);
    let (labels, start_row) = if assign == Assign::Kmeans {
        (kmeans(&embedding.vectors, n, k, k, &mut rng), None)
    } else {
        let (l, r) = discretize(&embedding.vectors, n, k, &mut rng)?;
        (l, Some(r))
    };
    Ok(SpencOut {
        labels,
        affinity,
        embedding: Some(embedding),
        start_row,
        warnings,
    })
}
