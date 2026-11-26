use super::polarization::{group_labels, polarization, Grouping};
use crate::graph::Csr;

fn graph(n: usize, edges: &[(u32, u32)]) -> Csr {
    let mut f = Vec::new();
    let mut nb = Vec::new();
    for &(a, b) in edges {
        f.extend([a, b]);
        nb.extend([b, a]);
    }
    let w = vec![1.0; f.len()];
    Csr::from_arrays(n, &f, &nb, &w).unwrap()
}

fn path(n: usize) -> Csr {
    let edges: Vec<(u32, u32)> = (0..n as u32 - 1).map(|i| (i, i + 1)).collect();
    graph(n, &edges)
}

fn quantiles(k: usize) -> Grouping {
    Grouping::Quantiles(k)
}

#[test]
fn quantile_groups_split_sorted_values() {
    let (labels, ca) = group_labels(&[4.0, 1.0, 3.0, 2.0], &quantiles(2)).unwrap();
    assert_eq!(labels, vec![1, 0, 1, 0]);
    assert_eq!(ca, 2);
}

#[test]
fn quantile_groups_use_numpy_linear_interpolation() {
    // edges of 5 values in 3 classes: 1, 2.333.., 3.666.., 5 (h = 4/3 and 8/3)
    let (labels, _) = group_labels(&[1.0, 2.0, 3.0, 4.0, 5.0], &quantiles(3)).unwrap();
    assert_eq!(labels, vec![0, 0, 1, 2, 2]);
}

#[test]
fn quantile_groups_drop_duplicate_edges_but_keep_the_nominal_count() {
    // edges 1, 1, 1, 9: the middle ones collapse, leaving the single bin [1, 9]
    let (labels, ca) = group_labels(&[1.0, 1.0, 1.0, 1.0, 1.0, 9.0], &quantiles(3)).unwrap();
    assert_eq!(ca, 3);
    assert_eq!(labels, vec![0, 0, 0, 0, 0, 0]);
}

#[test]
fn constant_values_have_no_group_unless_one_class() {
    let (labels, _) = group_labels(&[2.0; 4], &quantiles(2)).unwrap();
    assert_eq!(labels, vec![-1; 4]);
    let (one, _) = group_labels(&[2.0; 4], &quantiles(1)).unwrap();
    assert_eq!(one, vec![0; 4]);
}

#[test]
fn explicit_bins_are_right_closed_and_exclude_the_lower_edge() {
    let bins = Grouping::Bins(vec![0.0, 2.0, 5.0]);
    let (labels, ca) = group_labels(&[0.0, 1.0, 2.0, 3.0, 5.0], &bins).unwrap();
    assert_eq!(ca, 2);
    assert_eq!(labels, vec![-1, 0, 0, 1, 1]);
}

#[test]
fn bins_reject_values_outside_the_range_and_bad_edges() {
    assert!(group_labels(&[1.0, 9.0], &Grouping::Bins(vec![2.0, 8.0])).is_err());
    assert!(group_labels(&[1.0], &Grouping::Bins(vec![0.0, 2.0, 2.0])).is_err());
    assert!(group_labels(&[1.0], &Grouping::Bins(vec![3.0])).is_err());
    assert!(group_labels(&[1.0, 2.0], &quantiles(0)).is_err());
    assert!(group_labels(&[1.0, 2.0], &quantiles(3)).is_err());
}

#[test]
fn clustered_values_give_one_and_alternating_values_zero() {
    let g = path(6);
    let clustered = polarization(&[1.0, 1.0, 1.0, 9.0, 9.0, 9.0], &g, &quantiles(2), 0, 0).unwrap();
    assert_eq!(clustered.statistic, 1.0);
    assert_eq!(clustered.n_i_components, 2);
    assert_eq!(clustered.i_labels, vec![0, 0, 0, 1, 1, 1]);
    assert!(clustered.p_value.is_nan() && clustered.sim.is_empty());
    let alternating =
        polarization(&[1.0, 9.0, 1.0, 9.0, 1.0, 9.0], &g, &quantiles(2), 0, 0).unwrap();
    assert_eq!(alternating.statistic, 0.0);
    assert_eq!(alternating.n_i_components, 6);
}

#[test]
fn islands_count_as_graph_components() {
    // path 0-1-2 plus the island 3 (zero-weight self loop) and the pair 4-5
    let f = [0, 1, 1, 2, 3, 4, 5];
    let nb = [1, 0, 2, 1, 3, 5, 4];
    let w = [1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0];
    let g = Csr::from_arrays(6, &f, &nb, &w).unwrap();
    let r = polarization(&[1.0, 2.0, 8.0, 9.0, 1.0, 2.0], &g, &quantiles(2), 0, 0).unwrap();
    assert_eq!(r.n_g_components, 3);
    assert_eq!(r.g_labels, vec![0, 0, 0, 1, 2, 2]);
    assert_eq!(r.n_a_components, 2);
    // groups 0 0 1 1 0 0 give the components {0,1}, {2}, {3}, {4,5}; k = max(3, 2) = 3
    assert_eq!(r.n_i_components, 4);
    assert!((r.statistic - (1.0 - (4.0 - 3.0) / 3.0)).abs() < 1e-15);
}

#[test]
fn excluded_observations_leave_the_count() {
    let g = path(4);
    let bins = Grouping::Bins(vec![1.0, 5.0]);
    let r = polarization(&[1.0, 2.0, 3.0, 4.0], &g, &bins, 0, 0).unwrap();
    assert_eq!(r.a_labels, vec![-1, 0, 0, 0]);
    assert_eq!(r.i_labels, vec![-1, 0, 0, 0]);
    // c = 1, k = max(1, 1) = 1, denom = max(3 - 1, 1)
    assert_eq!(r.statistic, 1.0);
}

#[test]
fn permutations_are_seeded_and_thread_independent() {
    let g = path(30);
    let x: Vec<f64> = (0..30).map(f64::from).collect();
    let run = |threads: usize, seed: u64| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| polarization(&x, &g, &quantiles(3), 99, seed).unwrap())
    };
    let (a, b, c) = (run(1, 5), run(4, 5), run(4, 6));
    assert_eq!(a.sim, b.sim);
    assert_eq!(a.p_value, b.p_value);
    assert_ne!(a.sim, c.sim);
    assert_eq!(a.sim.len(), 99);
    assert!(a.sim.iter().all(|&s| s <= 1.0));
    // a sorted ramp is as clustered as it gets: no permutation beats it
    assert_eq!(a.p_value, 1.0 / 100.0);
}

#[test]
fn length_mismatch_is_an_error() {
    assert!(polarization(&[1.0, 2.0], &path(3), &quantiles(2), 0, 0).is_err());
}
