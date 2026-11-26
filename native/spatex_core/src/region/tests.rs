//! Unit tests for regionalization (hand-checkable cases and invariants; the spopt oracle lives
//! in the Elixir suite).

use super::azp::{azp, AzpOpts, Strategy};
use super::kmeans::{draw_seeds, region_kmeans};
use super::maxp::{max_p, modify_components, MaxPOpts, Policy};
use super::objective::{center, pairwise};
use super::sa3::{extract_clusters, fix_monotone, sa3, Extraction};
use super::skater::{skater, spanning_forest, SkaterOpts};
use super::util::{columns_to_rows, edge_components, non_contiguous_regions, relabel, Mat};
use super::ward::ward;
use super::{Adj, Metric};

/// Rook lattice `rows x cols`.
fn lattice(rows: usize, cols: usize) -> Adj {
    let mut edges = Vec::new();
    for r in 0..rows {
        for c in 0..cols {
            let i = (r * cols + c) as u32;
            if c + 1 < cols {
                edges.push((i, i + 1));
            }
            if r + 1 < rows {
                edges.push((i, i + cols as u32));
            }
        }
    }
    Adj::from_edges(rows * cols, &edges)
}

fn skater_opts(k: usize) -> SkaterOpts {
    SkaterOpts {
        n_clusters: k,
        floor: 0.0,
        ignore_islands: false,
        trace: false,
        metric: Metric::Manhattan,
        drop_zero: false,
    }
}

#[test]
fn adjacency_drops_loops_and_checks_symmetry() {
    use crate::graph::Csr;
    let sym = Csr::from_arrays(3, &[0, 1, 1, 2], &[1, 0, 2, 1], &[1.0; 4]).unwrap();
    let adj = Adj::from_csr(&sym, false).unwrap();
    assert_eq!(adj.nb, vec![vec![1], vec![0, 2], vec![1]]);
    let island = Csr::from_arrays(2, &[0], &[1], &[1.0]).unwrap();
    assert!(Adj::from_csr(&island, false)
        .unwrap_err()
        .contains("symmetric"));
    let both = Adj::from_csr(&island, true).unwrap();
    assert_eq!(both.nb, vec![vec![1], vec![0]]);
    let zero = Csr::from_arrays(2, &[0, 1], &[1, 0], &[0.0, 0.0]).unwrap();
    assert_eq!(
        Adj::from_csr(&zero, false).unwrap().nb,
        vec![vec![], vec![]]
    );
}

#[test]
fn components_are_labeled_by_lowest_node() {
    let adj = Adj::from_edges(6, &[(4, 5), (0, 3), (1, 2)]);
    let (count, labels) = adj.components();
    assert_eq!(count, 3);
    assert_eq!(labels, vec![0, 1, 1, 0, 2, 2]);
    assert_eq!(edge_components(6, &[(4, 5), (0, 3), (1, 2)]).1, labels);
}

#[test]
fn relabel_numbers_by_first_appearance() {
    assert_eq!(relabel(&[7u32, 7, 3, 9, 3]), vec![0, 0, 1, 2, 1]);
}

#[test]
fn columns_become_rows() {
    assert_eq!(
        columns_to_rows(&[1.0, 2.0, 3.0, 10.0, 20.0, 30.0], 3, 2).unwrap(),
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0]
    );
    assert!(columns_to_rows(&[1.0], 3, 2).is_err());
}

#[test]
fn matrix_validation() {
    assert!(Mat::new(&[1.0, f64::NAN], 2, 1).is_err());
    assert!(Mat::new(&[1.0], 2, 1).is_err());
    assert!(Mat::new(&[], 0, 1).is_err());
}

#[test]
fn objectives_by_hand() {
    let data = [0.0, 1.0, 5.0, 7.0];
    let x = Mat::new(&data, 4, 1).unwrap();
    let labels = [0, 0, 1, 1];
    assert_eq!(pairwise(&x, &labels, Metric::Manhattan).unwrap(), 3.0);
    assert_eq!(center(&x, &labels, Metric::Manhattan).unwrap(), 3.0);
    assert_eq!(center(&x, &labels, Metric::SqEuclidean).unwrap(), 2.5);
    assert!(pairwise(&x, &[0, 0], Metric::Manhattan).is_err());
}

#[test]
fn spanning_forest_is_a_spanning_tree_per_component() {
    let adj = lattice(4, 5);
    let data: Vec<f64> = (0..20).map(|i| ((i * 7) % 11) as f64).collect();
    let x = Mat::new(&data, 20, 1).unwrap();
    let msf = spanning_forest(&adj, &x, Metric::Manhattan, false);
    assert_eq!(msf.len(), 19);
    assert_eq!(edge_components(20, &msf).0, 1);
    assert!(msf.windows(2).all(|w| w[0] < w[1]), "row-major order");
}

#[test]
fn skater_cuts_the_largest_gap() {
    let adj = lattice(1, 6);
    let data = [1.0, 1.1, 1.2, 9.0, 9.1, 9.2];
    let x = Mat::new(&data, 6, 1).unwrap();
    let out = skater(&adj, &x, &skater_opts(2)).unwrap();
    assert_eq!(out.labels, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(out.msf.len(), 4);
    assert!(out.warnings.is_empty());
}

#[test]
fn skater_matches_exhaustive_search_of_tree_cuts() {
    // 3x3 lattice, k = 3: the pruned result must be the best two-edge cut of the MSF.
    let adj = lattice(3, 3);
    let data = [1.0, 2.0, 9.0, 1.5, 8.0, 9.5, 20.0, 2.5, 21.0];
    let x = Mat::new(&data, 9, 1).unwrap();
    let out = skater(&adj, &x, &skater_opts(3)).unwrap();
    // exhaustive: every pair of MSF edges to remove, scored directly
    let msf = spanning_forest(&adj, &x, Metric::Manhattan, false);
    let mut best = f64::INFINITY;
    for a in 0..msf.len() {
        for b in (a + 1)..msf.len() {
            let rest: Vec<(u32, u32)> = msf
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != a && *i != b)
                .map(|(_, e)| *e)
                .collect();
            let (_, labels) = edge_components(9, &rest);
            best = best.min(super::skater::score(&x, &labels, Metric::Manhattan, 0.0));
        }
    }
    // greedy pruning is not globally optimal in general; it can never beat the optimum
    assert!(out.score >= best - 1e-12);
    assert_eq!(out.labels.iter().max(), Some(&2));
}

#[test]
fn skater_islands_and_floor() {
    let adj = Adj::from_edges(5, &[(0, 1), (1, 2), (3, 4)]);
    let data = [1.0, 2.0, 9.0, 4.0, 4.5];
    let x = Mat::new(&data, 5, 1).unwrap();
    let inc = skater(&adj, &x, &skater_opts(2)).unwrap();
    assert_eq!(*inc.labels.iter().max().unwrap(), 3, "2 + 2 components");
    assert_eq!(inc.warnings.len(), 1);
    let mut opts = skater_opts(2);
    opts.ignore_islands = true;
    let ign = skater(&adj, &x, &opts).unwrap();
    assert_eq!(ign.labels, vec![0, 0, 0, 1, 1]);
    opts.floor = 3.0;
    assert!(skater(&adj, &x, &opts)
        .unwrap_err()
        .contains("Islands must be larger"));
    let mut floor = skater_opts(5);
    floor.floor = 2.0;
    let capped = skater(&lattice(1, 5), &Mat::new(&data, 5, 1).unwrap(), &floor).unwrap();
    assert!(capped.warnings[0].contains("no valid moves"));
}

#[test]
fn skater_trace_and_validation() {
    let adj = lattice(1, 4);
    let data = [1.0, 1.0, 5.0, 5.0];
    let x = Mat::new(&data, 4, 1).unwrap();
    let mut opts = skater_opts(2);
    opts.trace = true;
    let out = skater(&adj, &x, &opts).unwrap();
    assert_eq!(out.trace.len(), 2);
    assert_eq!(out.trace[0].edge, None);
    assert_eq!(out.trace[1].edge, Some((1, 2)));
    assert_eq!(out.trace[1].score, 0.0);
    assert!(skater(&adj, &x, &skater_opts(0)).is_err());
    assert!(skater(&adj, &x, &skater_opts(5)).is_err());
}

#[test]
fn skater_zero_dissimilarity_option() {
    let adj = lattice(1, 4);
    let data = [1.0, 1.0, 5.0, 5.0];
    let x = Mat::new(&data, 4, 1).unwrap();
    let kept = spanning_forest(&adj, &x, Metric::Manhattan, false);
    let dropped = spanning_forest(&adj, &x, Metric::Manhattan, true);
    assert_eq!(kept.len(), 3);
    assert_eq!(dropped, vec![(1, 2)]);
}

#[test]
fn ward_merges_similar_neighbors_and_orders_children() {
    let adj = lattice(1, 6);
    let data = [1.0, 1.1, 1.2, 9.0, 9.1, 9.2];
    let x = Mat::new(&data, 6, 1).unwrap();
    let out = ward(&adj, &x, 2).unwrap();
    assert_eq!(relabel(&out.labels), vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(out.children.len(), 5);
    assert!(out.children.iter().all(|(a, b)| a < b));
    let one = ward(&adj, &x, 1).unwrap();
    assert!(one.labels.iter().all(|&l| l == 0));
    assert!(ward(&adj, &x, 7).is_err());
}

#[test]
fn ward_completes_a_disconnected_graph() {
    let adj = Adj::from_edges(4, &[(0, 1), (2, 3)]);
    let data = [0.0, 1.0, 1.5, 9.0];
    let x = Mat::new(&data, 4, 1).unwrap();
    let out = ward(&adj, &x, 1).unwrap();
    assert_eq!(out.n_connected_components, 2);
    assert_eq!(out.warnings.len(), 1);
    assert_eq!(out.children.len(), 3);
    // the bridge is the closest pair across components, areas 1 and 2
    let two = ward(&adj, &x, 2).unwrap();
    assert_eq!(relabel(&two.labels), vec![0, 0, 0, 1]);
}

#[test]
fn kmeans_grows_from_seeds() {
    let adj = lattice(1, 6);
    let data = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2];
    let x = Mat::new(&data, 6, 1).unwrap();
    let out = region_kmeans(&adj, &x, &[0, 5], 1000).unwrap();
    assert_eq!(out.labels, vec![0, 0, 0, 1, 1, 1]);
    assert_eq!(out.centroids.len(), 2);
    assert!((out.centroids[0] - 0.1).abs() < 1e-12);
}

#[test]
fn kmeans_moves_misplaced_areas() {
    // seeds in the middle of each side: the improvement phase fixes the border
    let adj = lattice(1, 6);
    let data = [0.0, 0.0, 0.0, 9.0, 9.0, 9.0];
    let x = Mat::new(&data, 6, 1).unwrap();
    let out = region_kmeans(&adj, &x, &[2, 3], 1000).unwrap();
    assert_eq!(out.labels, vec![0, 0, 0, 1, 1, 1]);
    assert!(out.iters >= 1);
}

#[test]
fn kmeans_seed_errors_and_drawing() {
    let adj = Adj::from_edges(4, &[(0, 1), (2, 3)]);
    let data = [0.0, 1.0, 2.0, 3.0];
    let x = Mat::new(&data, 4, 1).unwrap();
    assert!(region_kmeans(&adj, &x, &[0, 1], 100)
        .unwrap_err()
        .contains("no seed"));
    assert!(region_kmeans(&adj, &x, &[0, 0], 100).is_err());
    assert!(region_kmeans(&adj, &x, &[0, 9], 100).is_err());
    let seeds = draw_seeds(&adj, 3, 5).unwrap();
    assert_eq!(seeds.len(), 3);
    assert_eq!(seeds, draw_seeds(&adj, 3, 5).unwrap());
    assert!(draw_seeds(&adj, 1, 5).is_err());
    let comps: std::collections::HashSet<u32> = seeds.iter().map(|s| s / 2).collect();
    assert_eq!(comps.len(), 2);
}

fn azp_opts(k: usize, seed: u64) -> AzpOpts {
    AzpOpts {
        n_clusters: k,
        metric: Metric::Manhattan,
        seed,
        initial: None,
        max_moves: 100_000,
        strategy: Strategy::Basic,
    }
}

const TABU: Strategy = Strategy::Tabu {
    tabu_length: 10,
    repetitions: 5,
};

const ANNEALING: Strategy = Strategy::Annealing {
    init_temperature: 2.0,
    cooling_factor: 0.85,
    sa_moves_term: 10,
    repetitions: 5,
    nonmoving_steps: 3,
};

fn with(strategy: Strategy, k: usize, seed: u64) -> AzpOpts {
    AzpOpts {
        strategy,
        max_moves: 5_000,
        ..azp_opts(k, seed)
    }
}

#[test]
fn tabu_and_annealing_give_valid_improving_deterministic_partitions() {
    let adj = lattice(2, 5);
    let data = [3.0, 4.0, 1.0, 9.0, 8.0, 3.5, 4.5, 0.5, 8.5, 9.5];
    let x = Mat::new(&data, 10, 1).unwrap();
    let optimum = best_partition_objective(&adj, &x, 3);
    for strategy in [TABU, ANNEALING] {
        let mut best = f64::INFINITY;
        for seed in 0..20 {
            let out = azp(&adj, &x, &with(strategy, 3, seed)).unwrap();
            assert!(
                non_contiguous_regions(&adj, &out.labels).is_empty(),
                "{strategy:?}"
            );
            assert_eq!(out.labels.iter().max(), Some(&2), "{strategy:?}");
            let start = pairwise(&x, &out.initial_labels, Metric::Manhattan).unwrap();
            assert!(out.objective <= start + 1e-9, "{strategy:?}");
            assert!(
                (out.objective - pairwise(&x, &out.labels, Metric::Manhattan).unwrap()).abs()
                    < 1e-9
            );
            assert_eq!(
                out.labels,
                azp(&adj, &x, &with(strategy, 3, seed)).unwrap().labels
            );
            best = best.min(out.objective);
        }
        assert!(
            (best - optimum).abs() < 1e-9,
            "{strategy:?}: {best} vs {optimum}"
        );
    }
}

#[test]
fn tabu_and_annealing_end_in_a_strict_local_optimum() {
    let adj = lattice(4, 4);
    let data: Vec<f64> = (0..16).map(|i| ((i * 5) % 7) as f64).collect();
    let x = Mat::new(&data, 16, 1).unwrap();
    for strategy in [TABU, ANNEALING] {
        let a = azp(&adj, &x, &with(strategy, 4, 2)).unwrap();
        for area in 0..16usize {
            for &nb in &adj.nb[area] {
                let to = a.labels[nb as usize];
                if to == a.labels[area] {
                    continue;
                }
                let mut moved = a.labels.clone();
                moved[area] = to;
                if non_contiguous_regions(&adj, &moved).is_empty()
                    && moved.iter().collect::<std::collections::HashSet<_>>().len() == 4
                {
                    let o = pairwise(&x, &moved, Metric::Manhattan).unwrap();
                    assert!(o >= a.objective - 1e-9, "{strategy:?}");
                }
            }
        }
    }
}

#[test]
fn annealing_parameters_are_validated() {
    let adj = lattice(2, 3);
    let x = Mat::new(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 6, 1).unwrap();
    for (t, c) in [(0.0, 0.85), (1.0, 1.0), (1.0, 0.0), (f64::NAN, 0.5)] {
        let s = Strategy::Annealing {
            init_temperature: t,
            cooling_factor: c,
            sa_moves_term: 10,
            repetitions: 5,
            nonmoving_steps: 3,
        };
        assert!(azp(&adj, &x, &with(s, 2, 0)).is_err(), "{t} {c}");
    }
    let s = Strategy::Tabu {
        tabu_length: 0,
        repetitions: 5,
    };
    assert!(azp(&adj, &x, &with(s, 2, 0)).is_err());
}

/// Every contiguous partition of `n` nodes into `k` regions, by brute force (n <= 10).
fn best_partition_objective(adj: &Adj, x: &Mat, k: usize) -> f64 {
    let n = adj.n();
    let mut best = f64::INFINITY;
    let mut labels = vec![0u32; n];
    fn rec(
        i: usize,
        used: u32,
        k: usize,
        labels: &mut Vec<u32>,
        adj: &Adj,
        x: &Mat,
        best: &mut f64,
    ) {
        if i == labels.len() {
            if used as usize == k && non_contiguous_regions(adj, labels).is_empty() {
                *best = best.min(pairwise(x, labels, Metric::Manhattan).unwrap());
            }
            return;
        }
        for l in 0..=used.min(k as u32 - 1) {
            labels[i] = l;
            rec(i + 1, used.max(l + 1), k, labels, adj, x, best);
        }
    }
    rec(0, 0, k, &mut labels, adj, x, &mut best);
    best
}

#[test]
fn azp_finds_the_optimum_on_tiny_graphs_for_some_seed() {
    let adj = lattice(2, 5);
    let data = [3.0, 4.0, 1.0, 9.0, 8.0, 3.5, 4.5, 0.5, 8.5, 9.5];
    let x = Mat::new(&data, 10, 1).unwrap();
    let optimum = best_partition_objective(&adj, &x, 3);
    let mut best = f64::INFINITY;
    for seed in 0..40 {
        let out = azp(&adj, &x, &azp_opts(3, seed)).unwrap();
        assert!(non_contiguous_regions(&adj, &out.labels).is_empty());
        assert_eq!(out.labels.iter().max(), Some(&2));
        assert!(out.objective >= optimum - 1e-9);
        assert!(
            (out.objective - pairwise(&x, &out.labels, Metric::Manhattan).unwrap()).abs() < 1e-9
        );
        let start = pairwise(&x, &out.initial_labels, Metric::Manhattan).unwrap();
        assert!(out.objective <= start + 1e-9);
        best = best.min(out.objective);
    }
    assert!((best - optimum).abs() < 1e-9, "{best} vs {optimum}");
}

#[test]
fn azp_is_a_strict_local_optimum_and_deterministic() {
    let adj = lattice(4, 4);
    let data: Vec<f64> = (0..16).map(|i| ((i * 5) % 7) as f64).collect();
    let x = Mat::new(&data, 16, 1).unwrap();
    let a = azp(&adj, &x, &azp_opts(4, 9)).unwrap();
    let b = azp(&adj, &x, &azp_opts(4, 9)).unwrap();
    assert_eq!(a.labels, b.labels);
    // no single move of one area into an adjacent region lowers the objective
    for area in 0..16usize {
        for &nb in &adj.nb[area] {
            let to = a.labels[nb as usize];
            if to == a.labels[area] {
                continue;
            }
            let mut moved = a.labels.clone();
            moved[area] = to;
            if non_contiguous_regions(&adj, &moved).is_empty()
                && moved.iter().collect::<std::collections::HashSet<_>>().len() == 4
            {
                let o = pairwise(&x, &moved, Metric::Manhattan).unwrap();
                assert!(o >= a.objective - 1e-9);
            }
        }
    }
}

#[test]
fn azp_validation() {
    let adj = Adj::from_edges(4, &[(0, 1), (2, 3)]);
    let data = [0.0, 1.0, 2.0, 3.0];
    let x = Mat::new(&data, 4, 1).unwrap();
    assert!(azp(&adj, &x, &azp_opts(1, 0))
        .unwrap_err()
        .contains("components"));
    assert!(azp(&adj, &x, &azp_opts(5, 0)).is_err());
    let one = azp(&lattice(1, 4), &x, &azp_opts(1, 0)).unwrap();
    assert_eq!(one.labels, vec![0; 4]);
    let mut opts = azp_opts(2, 0);
    opts.initial = Some(vec![0, 1, 0, 1]);
    assert!(azp(&lattice(1, 4), &x, &opts)
        .unwrap_err()
        .contains("contiguous"));
    opts.initial = Some(vec![5, 5, 7, 7]);
    let ok = azp(&lattice(1, 4), &x, &opts).unwrap();
    assert_eq!(ok.initial_labels, vec![0, 0, 1, 1]);
}

fn maxp_opts(threshold: f64, seed: u64) -> MaxPOpts {
    MaxPOpts {
        threshold,
        top_n: 2,
        max_iterations_construction: 30,
        max_iterations_sa: 3,
        seed,
        policy: Policy::Single,
    }
}

#[test]
fn max_p_regions_are_valid_and_reach_the_best_p_on_a_path() {
    let adj = lattice(1, 6);
    let data = [0.0, 0.1, 0.2, 5.0, 5.1, 5.2];
    let x = Mat::new(&data, 6, 1).unwrap();
    let t = [1.0; 6];
    for seed in 0..10 {
        let (labels, p, obj, _) = max_p(&adj, &x, &t, &maxp_opts(2.0, seed)).unwrap();
        assert_eq!(p, 3, "a path of six areas holds three regions of two");
        assert!(non_contiguous_regions(&adj, &labels).is_empty());
        let sizes = (0..p as u32)
            .map(|r| labels.iter().filter(|&&l| l == r).count())
            .collect::<Vec<_>>();
        assert!(sizes.iter().all(|&s| s >= 2));
        assert!(obj >= 0.0);
    }
}

#[test]
fn max_p_is_deterministic_and_validates() {
    let adj = lattice(3, 4);
    let data: Vec<f64> = (0..12).map(|i| ((i * 3) % 5) as f64).collect();
    let x = Mat::new(&data, 12, 1).unwrap();
    let t = [1.0; 12];
    let a = max_p(&adj, &x, &t, &maxp_opts(3.0, 4)).unwrap();
    let b = max_p(&adj, &x, &t, &maxp_opts(3.0, 4)).unwrap();
    assert_eq!(a, b);
    let mut bad = maxp_opts(3.0, 0);
    bad.top_n = 0;
    assert!(max_p(&adj, &x, &t, &bad).is_err());
}

#[test]
fn modify_components_policies() {
    // component 0 = {0,1,2,3} (total 4), component 1 = {4,5} (total 2)
    let adj = Adj::from_edges(6, &[(0, 1), (1, 2), (2, 3), (4, 5)]);
    let t = [1.0; 6];
    let coords = [
        [0.0, 0.0],
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [5.0, 0.0],
        [6.0, 0.0],
    ];
    let (same, keep) = modify_components(&adj, &t, 2.0, None, Policy::Single).unwrap();
    assert_eq!(same, adj);
    assert!(keep.iter().all(|&k| k));
    assert!(modify_components(&adj, &t, 3.0, None, Policy::Single)
        .unwrap_err()
        .contains("coords"));
    assert!(modify_components(&adj, &t, 3.0, Some(&coords), Policy::Keep).is_err());
    assert!(
        modify_components(&adj, &t, 9.0, Some(&coords), Policy::Single)
            .unwrap_err()
            .contains("No feasible")
    );
    let (single, _) = modify_components(&adj, &t, 3.0, Some(&coords), Policy::Single).unwrap();
    assert_eq!(single.components().0, 1);
    assert!(single.nb[3].contains(&4), "nearest pair is 3-4");
    let (multi, _) = modify_components(&adj, &t, 3.0, Some(&coords), Policy::Multiple).unwrap();
    assert!(multi.nb[3].contains(&4) && multi.nb[3].contains(&5));
    let (dropped, keep) = modify_components(&adj, &t, 3.0, None, Policy::Drop).unwrap();
    assert_eq!(keep, vec![true, true, true, true, false, false]);
    assert_eq!(dropped.n(), 4);
}

// -- random regions -----------------------------------------------------------------------

mod random_regions {
    use super::super::random::{random_region, random_regions, RandomOpts, RandomOut};
    use super::{lattice, non_contiguous_regions};
    use crate::perm::replicate_rng;

    fn opts(n: usize) -> RandomOpts<'static> {
        RandomOpts {
            n,
            adj: None,
            num_regions: None,
            cardinality: None,
            maxiter: 1000,
            compact: false,
            max_swaps: 1_000_000,
        }
    }

    fn run(o: &RandomOpts, seed: u64) -> RandomOut {
        random_region(o, &mut replicate_rng(seed, 0)).unwrap()
    }

    fn labels(out: &RandomOut, n: usize) -> Vec<u32> {
        let mut l = vec![u32::MAX; n];
        for (r, members) in out.regions.iter().enumerate() {
            for &a in members {
                assert_eq!(l[a as usize], u32::MAX, "area {a} assigned twice");
                l[a as usize] = r as u32;
            }
        }
        assert!(l.iter().all(|&x| x != u32::MAX), "unassigned area");
        l
    }

    #[test]
    fn unconstrained_partitions_every_area_once_in_two_to_n_minus_one_regions() {
        for seed in 0..50 {
            let out = run(&opts(20), seed);
            assert!(out.feasible);
            labels(&out, 20);
            assert!((2..=19).contains(&out.regions.len()));
            assert!(out.regions.iter().all(|r| !r.is_empty()));
        }
    }

    #[test]
    fn region_count_is_exact_and_last_region_keeps_spopts_two_area_minimum() {
        let mut o = opts(12);
        o.num_regions = Some(5);
        for seed in 0..100 {
            let out = run(&o, seed);
            assert_eq!(out.regions.len(), 5);
            labels(&out, 12);
            // spopt draws breaks from 1..n-2, so the last slice always holds two areas or more.
            assert!(out.regions.last().unwrap().len() >= 2);
        }
    }

    #[test]
    fn cardinality_is_honored_in_order_without_a_graph() {
        let mut o = opts(10);
        o.cardinality = Some(vec![5, 1, 4]);
        let out = run(&o, 3);
        let sizes: Vec<usize> = out.regions.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![5, 1, 4]);
        labels(&out, 10);
    }

    #[test]
    fn contiguous_cardinality_regions_are_connected_with_exact_sizes() {
        let adj = lattice(6, 6);
        for compact in [false, true] {
            for seed in 0..20 {
                let mut o = opts(36);
                o.adj = Some(&adj);
                o.cardinality = Some(vec![18, 9, 6, 3]);
                o.compact = compact;
                let out = run(&o, seed);
                assert!(out.feasible, "compact={compact} seed={seed}");
                let l = labels(&out, 36);
                assert!(non_contiguous_regions(&adj, &l).is_empty());
                let mut sizes: Vec<usize> = out.regions.iter().map(Vec::len).collect();
                sizes.sort_unstable();
                assert_eq!(sizes, vec![3, 6, 9, 18]);
            }
        }
    }

    #[test]
    fn contiguous_with_number_of_regions_or_nothing_yields_connected_regions() {
        let adj = lattice(5, 5);
        let mut feasible_seen = 0;
        for seed in 0..30 {
            for k in [None, Some(4)] {
                let mut o = opts(25);
                o.adj = Some(&adj);
                o.num_regions = k;
                o.maxiter = 200;
                let out = run(&o, seed);
                if out.feasible {
                    feasible_seen += 1;
                    let l = labels(&out, 25);
                    assert!(non_contiguous_regions(&adj, &l).is_empty());
                    if let Some(k) = k {
                        assert_eq!(out.regions.len(), k);
                    }
                } else {
                    assert!(out.regions.is_empty());
                }
            }
        }
        assert!(feasible_seen > 10, "only {feasible_seen} feasible runs");
    }

    #[test]
    fn impossible_cardinality_on_split_graph_is_reported_infeasible() {
        // two components of three areas cannot hold a region of four
        let adj = super::Adj::from_edges(6, &[(0, 1), (1, 2), (3, 4), (4, 5)]);
        let mut o = opts(6);
        o.adj = Some(&adj);
        o.cardinality = Some(vec![4, 2]);
        o.maxiter = 5;
        o.max_swaps = 50;
        let out = run(&o, 1);
        assert!(!out.feasible);
        assert!(out.regions.is_empty());
        assert_eq!(out.attempts, 5);
    }

    #[test]
    fn a_seed_reproduces_and_seeds_differ() {
        let adj = lattice(6, 6);
        let mut o = opts(36);
        o.adj = Some(&adj);
        o.cardinality = Some(vec![12, 12, 12]);
        assert_eq!(run(&o, 9), run(&o, 9));
        assert_ne!(run(&o, 9).regions, run(&o, 10).regions);
    }

    #[test]
    fn invalid_requests_are_errors() {
        let mut o = opts(10);
        o.cardinality = Some(vec![5, 4]);
        assert!(random_region(&o, &mut replicate_rng(0, 0))
            .unwrap_err()
            .contains("cardinality"));
        let mut o = opts(10);
        o.num_regions = Some(3);
        o.cardinality = Some(vec![5, 5]);
        assert!(random_region(&o, &mut replicate_rng(0, 0)).is_err());
        let mut o = opts(10);
        o.num_regions = Some(10);
        assert!(random_region(&o, &mut replicate_rng(0, 0)).is_err());
        assert!(random_region(&opts(2), &mut replicate_rng(0, 0)).is_err());
        let adj = lattice(2, 2);
        let mut o = opts(5);
        o.adj = Some(&adj);
        assert!(random_region(&o, &mut replicate_rng(0, 0)).is_err());
        let mut o = opts(10);
        o.cardinality = Some(vec![10, 0]);
        assert!(random_region(&o, &mut replicate_rng(0, 0)).is_err());
    }

    #[test]
    fn permutations_are_independent_of_thread_count_and_start_with_stream_zero() {
        let adj = lattice(5, 5);
        let mut o = opts(25);
        o.adj = Some(&adj);
        o.cardinality = Some(vec![10, 8, 7]);
        let pool = |t| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .unwrap()
        };
        let a = pool(1).install(|| random_regions(&o, 12, 77).unwrap());
        let b = pool(4).install(|| random_regions(&o, 12, 77).unwrap());
        assert_eq!(a, b);
        assert_eq!(a.len(), 12);
        assert_eq!(a[0], run(&o, 77));
        assert_ne!(a[0].regions, a[1].regions);
    }
}

// -- SA3 -------------------------------------------------------------------------------------

fn path_values(v: &[f64]) -> (Adj, Vec<f64>) {
    let edges: Vec<(u32, u32)> = (0..v.len() as u32 - 1).map(|i| (i, i + 1)).collect();
    (Adj::from_edges(v.len(), &edges), v.to_vec())
}

#[test]
fn sa3_splits_two_tight_groups() {
    let (adj, v) = path_values(&[0.0, 0.1, 0.2, 10.0, 10.1, 10.2]);
    let x = Mat::new(&v, 6, 1).unwrap();
    for ext in [Extraction::Eom, Extraction::Leaf] {
        let out = sa3(&adj, &x, 3, ext).unwrap();
        // cluster ids follow fast_hdbscan's node numbering (verified against spopt 0.8.0)
        assert_eq!(out.labels, vec![1, 1, 1, 0, 0, 0], "{ext:?}");
        assert_eq!(out.n_components, 1);
        assert_eq!(out.trees.len(), 1);
        let link = &out.trees[0].1;
        assert_eq!(link.len(), 5);
        assert_eq!(link[4].3, 6);
        assert!(link.windows(2).all(|w| w[0].2 <= w[1].2));
    }
}

#[test]
fn sa3_labels_small_components_as_noise_and_numbers_clusters_globally() {
    // component A (3 areas) is not larger than min_cluster_size: all noise; B and C cluster
    let (adj_b, vb) = path_values(&[0.0, 0.1, 0.2, 10.0, 10.1, 10.2]);
    let mut edges: Vec<(u32, u32)> = vec![(0, 1), (1, 2)];
    let mut v = vec![5.0, 5.1, 5.2];
    for (i, row) in adj_b.nb.iter().enumerate() {
        for &j in row.iter().filter(|&&j| j as usize > i) {
            edges.push((i as u32 + 3, j + 3));
        }
    }
    v.extend(vb.iter().copied());
    let adj = Adj::from_edges(9, &edges);
    let x = Mat::new(&v, 9, 1).unwrap();
    let out = sa3(&adj, &x, 3, Extraction::Eom).unwrap();
    assert_eq!(out.labels, vec![-1, -1, -1, 1, 1, 1, 0, 0, 0]);
    assert_eq!(out.n_components, 2);
    assert_eq!(out.trees.len(), 1);
    assert_eq!(out.trees[0].0, vec![3, 4, 5, 6, 7, 8]);
}

#[test]
fn sa3_everything_is_noise_when_no_group_reaches_min_cluster_size() {
    let (adj, v) = path_values(&[0.0, 0.1, 0.2, 10.0, 10.1, 10.2]);
    let x = Mat::new(&v, 6, 1).unwrap();
    for ext in [Extraction::Eom, Extraction::Leaf] {
        let out = sa3(&adj, &x, 6, ext).unwrap();
        assert_eq!(out.labels, vec![-1; 6]);
    }
}

#[test]
fn sa3_isolated_areas_are_noise() {
    let adj = Adj::from_edges(3, &[]);
    let v = [1.0, 2.0, 3.0];
    let x = Mat::new(&v, 3, 1).unwrap();
    let out = sa3(&adj, &x, 2, Extraction::Eom).unwrap();
    assert_eq!(out.labels, vec![-1, -1, -1]);
    assert_eq!(out.n_components, 3);
}

#[test]
fn sa3_rejects_a_min_cluster_size_below_two() {
    let (adj, v) = path_values(&[0.0, 1.0, 2.0]);
    let x = Mat::new(&v, 3, 1).unwrap();
    assert!(sa3(&adj, &x, 1, Extraction::Eom)
        .unwrap_err()
        .contains("min_cluster_size"));
}

#[test]
fn extract_clusters_reads_a_linkage_matrix() {
    // scipy layout: leaves 0..=5, then nodes 6.. ; two triples joined at distance 9
    let link = [
        (0, 1, 0.1, 2),
        (6, 2, 0.2, 3),
        (3, 4, 0.1, 2),
        (8, 5, 0.2, 3),
        (7, 9, 9.0, 6),
    ];
    assert_eq!(
        extract_clusters(&link, 3, Extraction::Eom),
        vec![0, 0, 0, 1, 1, 1]
    );
    assert_eq!(
        extract_clusters(&link, 3, Extraction::Leaf),
        vec![0, 0, 0, 1, 1, 1]
    );
    assert_eq!(extract_clusters(&link, 4, Extraction::Eom), vec![-1; 6]);
    // a root that cannot split is a leaf: leaf extraction keeps it as one cluster
    assert_eq!(extract_clusters(&link, 4, Extraction::Leaf), vec![0; 6]);
}

#[test]
fn monotone_fix_shifts_a_drop_by_one_hundredth_of_the_previous_distance() {
    let mut d = vec![1.0, 0.5, 0.9, 3.0];
    fix_monotone(&mut d).unwrap();
    assert_eq!(d, vec![1.0, 1.01, 1.02, 3.0]);
    let mut e = vec![1.0, 2.0];
    fix_monotone(&mut e).unwrap();
    assert_eq!(e, vec![1.0, 2.0]);
}
