//! Unit tests for Spenc (hand-checkable cases; the spopt oracle lives in the Elixir suite).

use super::spenc::{affinity_matrix, embed, spenc, Assign, Kernel, KernelParams, SpencOpts};
use super::spenc_labels::{discretize, discretize_from_row, kmeans};
use super::util::{relabel, Mat};
use crate::graph::Csr;
use crate::perm::replicate_rng;

fn path(n: usize, w: f64) -> Csr {
    let (mut f, mut t) = (Vec::new(), Vec::new());
    for i in 0..n - 1 {
        f.extend([i as u32, i as u32 + 1]);
        t.extend([i as u32 + 1, i as u32]);
    }
    let weights = vec![w; f.len()];
    Csr::from_arrays(n, &f, &t, &weights).unwrap()
}

fn params() -> KernelParams {
    KernelParams {
        gamma: 0.5,
        degree: 3.0,
        coef0: 1.0,
        n_neighbors: 2,
    }
}

fn opts(k: Option<usize>, assign: Assign) -> SpencOpts {
    SpencOpts {
        n_clusters: k,
        kernel: Kernel::Rbf,
        params: params(),
        assign,
        floor: 0.0,
        floor_weights: None,
        grid_resolution: 100,
        seed: 7,
    }
}

const TWO_GROUPS: [f64; 8] = [0.0, 0.1, 0.2, 0.3, 3.0, 3.1, 3.2, 3.3];

#[test]
fn rbf_affinity_is_kernel_times_weights() {
    let g = path(3, 2.0);
    let x = [0.0, 1.0, 3.0];
    let m = Mat::new(&x, 3, 1).unwrap();
    let a = affinity_matrix(&g, &m, Kernel::Rbf, &params()).unwrap();
    assert_eq!(a.len(), 9);
    assert!((a[1] - 2.0 * (-0.5f64).exp()).abs() < 1e-15);
    assert!((a[5] - 2.0 * (-0.5f64 * 4.0).exp()).abs() < 1e-15);
    assert_eq!(a[2], 0.0, "non-neighbors have zero affinity");
    assert_eq!(a[0], 0.0, "the diagonal of W is zero");
    assert_eq!(a[1], a[3]);
}

#[test]
fn other_kernels_follow_scikit_learn() {
    let g = path(2, 1.0);
    let x = [1.0, 2.0, 2.0, 1.0];
    let m = Mat::new(&x, 2, 2).unwrap();
    let p = params();
    let at = |k| affinity_matrix(&g, &m, k, &p).unwrap()[1];
    assert!((at(Kernel::Laplacian) - (-0.5f64 * 2.0).exp()).abs() < 1e-15);
    assert!((at(Kernel::Linear) - 4.0).abs() < 1e-15);
    assert!((at(Kernel::Polynomial) - (0.5f64 * 4.0 + 1.0).powi(3)).abs() < 1e-12);
    assert!((at(Kernel::Sigmoid) - (0.5f64 * 4.0 + 1.0).tanh()).abs() < 1e-15);
    assert!((at(Kernel::Cosine) - 0.8).abs() < 1e-15);
}

#[test]
fn nearest_neighbors_affinity_ignores_w_and_symmetrizes() {
    let g = path(4, 1.0);
    let x = [0.0, 1.0, 1.5, 5.0];
    let m = Mat::new(&x, 4, 1).unwrap();
    let a = affinity_matrix(&g, &m, Kernel::NearestNeighbors, &params()).unwrap();
    // k = 2 including self: 0 -> {0, 1}, 1 -> {1, 2}, 2 -> {2, 1}, 3 -> {3, 2}.
    assert_eq!(a[1], 0.5, "0 -> 1 only one way");
    assert_eq!(a[4 + 2], 1.0, "1 <-> 2 both ways");
    assert_eq!(a[3 * 4 + 2], 0.5);
    assert_eq!(a[0], 1.0);
    assert_eq!(a[2], 0.0);
    for i in 0..4 {
        for j in 0..4 {
            assert_eq!(a[i * 4 + j], a[j * 4 + i]);
        }
    }
}

#[test]
fn precomputed_affinity_is_the_input() {
    let g = path(2, 1.0);
    let x = [0.0, 0.3, 0.3, 0.0];
    let m = Mat::new(&x, 2, 2).unwrap();
    let a = affinity_matrix(&g, &m, Kernel::Precomputed, &params()).unwrap();
    assert_eq!(a, x.to_vec());
    let bad = Mat::new(&x, 4, 1).unwrap();
    assert!(affinity_matrix(&g, &bad, Kernel::Precomputed, &params()).is_err());
}

#[test]
fn embedding_matches_the_path_laplacian_spectrum() {
    // Normalized Laplacian of the path on 4 nodes: 1 - cos(j pi / 3) = 0, 0.5, 1.5, 2.
    let n = 4;
    let mut a = vec![0.0; 16];
    for i in 0..3 {
        a[i * 4 + i + 1] = 1.0;
        a[(i + 1) * 4 + i] = 1.0;
    }
    let e = embed(&a, n, 2).unwrap();
    assert!(e.eigenvalues[0].abs() < 1e-12);
    assert!((e.eigenvalues[1] - 0.5).abs() < 1e-12);
    // First vector: D^{-1/2} u / sqrt(d) with u ~ sqrt(d): constant, positive max-abs entry.
    let first: Vec<f64> = (0..n).map(|i| e.vectors[i * 2]).collect();
    assert!(first
        .iter()
        .all(|v| (v - first[0]).abs() < 1e-12 && *v > 0.0));
    // Each vector's largest-magnitude entry is positive (spopt's sign convention).
    for c in 0..2 {
        let col: Vec<f64> = (0..n).map(|i| e.vectors[i * 2 + c]).collect();
        let big = col
            .iter()
            .cloned()
            .fold(0.0f64, |m, v| if v.abs() > m.abs() { v } else { m });
        assert!(big > 0.0);
    }
}

#[test]
fn embedding_rejects_bad_sizes_and_asymmetry() {
    let a = vec![0.0, 1.0, 1.0, 0.0];
    assert!(embed(&a, 2, 2).is_err(), "k must be smaller than n");
    let asym = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    assert!(embed(&asym, 3, 2).is_err());
}

fn blocks_vectors() -> Vec<f64> {
    // Two well separated groups in a 2-column embedding.
    let mut v = Vec::new();
    for i in 0..8 {
        if i < 4 {
            v.extend([1.0, 0.1 * i as f64]);
        } else {
            v.extend([0.1 * (i - 4) as f64, 1.0]);
        }
    }
    v
}

#[test]
fn discretize_separates_two_blocks_from_every_start_row() {
    let v = blocks_vectors();
    for row in 0..8 {
        let l = discretize_from_row(&v, 8, 2, row).unwrap();
        assert_eq!(relabel(&l), vec![0, 0, 0, 0, 1, 1, 1, 1], "row {row}");
    }
}

#[test]
fn discretize_draws_its_start_row_from_the_rng() {
    let v = blocks_vectors();
    let (l1, r1) = discretize(&v, 8, 2, &mut replicate_rng(3, 0)).unwrap();
    let (l2, r2) = discretize(&v, 8, 2, &mut replicate_rng(3, 0)).unwrap();
    assert_eq!((l1.clone(), r1), (l2, r2));
    assert!(r1 < 8);
    assert_eq!(l1, discretize_from_row(&v, 8, 2, r1).unwrap());
}

#[test]
fn kmeans_separates_two_blocks_and_is_seeded() {
    let v = blocks_vectors();
    let l = kmeans(&v, 8, 2, 2, &mut replicate_rng(1, 0));
    assert_eq!(relabel(&l), vec![0, 0, 0, 0, 1, 1, 1, 1]);
    assert_eq!(l, kmeans(&v, 8, 2, 2, &mut replicate_rng(1, 0)));
}

#[test]
fn spenc_recovers_two_groups_with_every_assignment() {
    let g = path(8, 1.0);
    let m = Mat::new(&TWO_GROUPS, 8, 1).unwrap();
    for assign in [Assign::Discretize, Assign::Kmeans, Assign::Hierarchical] {
        let mut o = opts(Some(2), assign);
        o.params.gamma = 0.05;
        let out = spenc(&g, &m, &o).unwrap();
        assert_eq!(
            relabel(&out.labels),
            vec![0, 0, 0, 0, 1, 1, 1, 1],
            "{assign:?}"
        );
        assert_eq!(out.affinity.len(), 64);
    }
}

#[test]
fn spenc_discretize_reports_embedding_and_start_row() {
    let g = path(8, 1.0);
    let m = Mat::new(&TWO_GROUPS, 8, 1).unwrap();
    let out = spenc(&g, &m, &opts(Some(2), Assign::Discretize)).unwrap();
    let e = out.embedding.expect("embedding");
    assert_eq!(e.vectors.len(), 16);
    assert_eq!(e.eigenvalues.len(), 2);
    assert!(out.start_row.unwrap() < 8);
    let again = spenc(&g, &m, &opts(Some(2), Assign::Discretize)).unwrap();
    assert_eq!(out.labels, again.labels);
    let h = spenc(&g, &m, &opts(Some(2), Assign::Hierarchical)).unwrap();
    assert!(h.embedding.is_none() && h.start_row.is_none());
}

#[test]
fn spenc_validates_n_clusters_and_floor() {
    let g = path(8, 1.0);
    let m = Mat::new(&TWO_GROUPS, 8, 1).unwrap();
    assert!(spenc(&g, &m, &opts(Some(8), Assign::Discretize)).is_err());
    assert!(spenc(&g, &m, &opts(Some(0), Assign::Discretize)).is_err());
    assert!(spenc(&g, &m, &opts(Some(1), Assign::Hierarchical)).is_err());
    let mut o = opts(None, Assign::Hierarchical);
    o.floor_weights = Some(vec![1.0; 3]);
    assert!(spenc(&g, &m, &o).is_err());
}

#[test]
fn spenc_warns_on_a_disconnected_graph() {
    let g = Csr::from_arrays(4, &[0, 1, 2, 3], &[1, 0, 3, 2], &[1.0; 4]).unwrap();
    let x = [0.0, 0.1, 5.0, 5.1];
    let m = Mat::new(&x, 4, 1).unwrap();
    let out = spenc(&g, &m, &opts(Some(2), Assign::Discretize)).unwrap();
    assert!(out.warnings.iter().any(|w| w.contains("disconnected")));
}
