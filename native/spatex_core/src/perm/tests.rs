use super::*;
use rand_core::Rng;

fn rows_to_bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn shuffle_is_a_permutation_and_deterministic() {
    let mut a: Vec<u32> = (0..50).collect();
    let mut b = a.clone();
    shuffle(&mut a, &mut replicate_rng(42, 3));
    shuffle(&mut b, &mut replicate_rng(42, 3));
    assert_eq!(a, b);
    let mut sorted = a.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..50).collect::<Vec<u32>>());
    assert_ne!(a, (0..50).collect::<Vec<u32>>());
}

#[test]
fn different_seeds_and_streams_give_different_draws() {
    let draw = |seed, r| replicate_rng(seed, r).next_u64();
    assert_ne!(draw(1, 0), draw(2, 0));
    assert_ne!(draw(1, 0), draw(1, 1));
    assert_eq!(draw(1, 5), draw(1, 5));
}

#[test]
fn uniform_below_stays_in_range_and_is_roughly_uniform() {
    let mut rng = replicate_rng(7, 0);
    let mut counts = [0usize; 6];
    for _ in 0..60_000 {
        let v = uniform_below(&mut rng, 6);
        assert!(v < 6);
        counts[v as usize] += 1;
    }
    for c in counts {
        assert!((9_000..11_000).contains(&c), "counts {counts:?}");
    }
    assert_eq!(uniform_below(&mut rng, 1), 0);
}

#[test]
fn shuffle_positions_are_roughly_uniform() {
    // Where does element 0 land after shuffling 4 items? Each slot ~25%.
    let mut counts = [0usize; 4];
    for r in 0..20_000u64 {
        let mut v = [0u8, 1, 2, 3];
        shuffle(&mut v, &mut replicate_rng(99, r));
        counts[v.iter().position(|&x| x == 0).unwrap()] += 1;
    }
    for c in counts {
        assert!((4_500..5_500).contains(&c), "counts {counts:?}");
    }
}

fn weighted_stat(v: &[f64]) -> f64 {
    v.iter()
        .enumerate()
        .map(|(i, x)| (i as f64 + 1.0) * x)
        .sum()
}

#[test]
fn simulate_is_identical_for_one_and_many_threads() {
    let base: Vec<f64> = (0..37).map(|i| (i as f64).sin() * 10.0).collect();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| simulate(&base, 777, 2024, weighted_stat))
    };
    let one = rows_to_bits(&run(1));
    for threads in [2, 4, 7] {
        assert_eq!(one, rows_to_bits(&run(threads)), "threads = {threads}");
    }
}

#[test]
fn simulate_is_deterministic_and_seed_sensitive() {
    let base: Vec<f64> = (0..20).map(f64::from).collect();
    let a = simulate(&base, 50, 1, weighted_stat);
    assert_eq!(
        rows_to_bits(&a),
        rows_to_bits(&simulate(&base, 50, 1, weighted_stat))
    );
    assert_ne!(
        rows_to_bits(&a),
        rows_to_bits(&simulate(&base, 50, 2, weighted_stat))
    );
}

#[test]
fn simulate_replicates_do_not_depend_on_the_total_count() {
    let base: Vec<f64> = (0..20).map(f64::from).collect();
    let short = simulate(&base, 5, 9, weighted_stat);
    let long = simulate(&base, 25, 9, weighted_stat);
    assert_eq!(rows_to_bits(&short), rows_to_bits(&long[..5]));
}

#[test]
fn simulate_permutes_the_input_values() {
    let base = [1.0, 2.0, 3.0, 4.0, 5.0];
    // Sum of squares is permutation invariant; position-weighted sum is not.
    let sums = simulate(&base, 30, 3, |v| v.iter().map(|x| x * x).sum());
    assert!(sums.iter().all(|&s| s == 55.0));
    let mixed = simulate(&base, 30, 3, weighted_stat);
    assert!(mixed.iter().any(|&s| s != weighted_stat(&base)));
}

#[test]
fn simulate_with_zero_permutations_is_empty() {
    assert!(simulate(&[1.0, 2.0], 0, 1, weighted_stat).is_empty());
}

#[test]
fn folded_p_matches_esda() {
    let sim = [1.0, 2.0, 3.0, 4.0, 5.0];
    // larger = #(sim >= 3) = 3; P - larger = 2 < 3, so larger = 2; (2 + 1) / 6.
    assert_eq!(folded_p(3.0, &sim), 0.5);
    // larger = 1 (only 5); P - larger = 4 >= 1.
    assert_eq!(folded_p(5.0, &sim), 2.0 / 6.0);
    // Everything is >= -10: larger = 5, flipped to 0.
    assert_eq!(folded_p(-10.0, &sim), 1.0 / 6.0);
    // NaN observed: nothing is >= NaN, larger = 0.
    assert_eq!(folded_p(f64::NAN, &sim), 1.0 / 6.0);
}

const R11: [f64; 11] = [0.5, -1.2, 0.3, 2.2, 0.3, -0.4, 1.1, 0.9, -2.0, 0.0, 0.7];

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-12
}

#[test]
fn significance_alternatives_match_esda() {
    use Alternative::*;
    // (observed, directed, two_sided, greater, lesser, folded) from esda.significance.
    let table = [
        (-2.5, 1.0 / 12.0, 0.25, 1.0, 1.0 / 12.0, 1.0 / 12.0),
        (-1.2, 2.0 / 12.0, 5.0 / 12.0, 11.0 / 12.0, 0.25, 4.0 / 12.0),
        (0.0, 4.0 / 12.0, 0.75, 0.75, 5.0 / 12.0, 10.0 / 12.0),
        (0.3, 5.0 / 12.0, 1.0, 8.0 / 12.0, 7.0 / 12.0, 1.0),
        (0.9, 4.0 / 12.0, 5.0 / 12.0, 4.0 / 12.0, 10.0 / 12.0, 0.5),
        (2.2, 2.0 / 12.0, 0.25, 2.0 / 12.0, 1.0, 0.25),
        (3.0, 1.0 / 12.0, 0.25, 1.0 / 12.0, 1.0, 1.0 / 12.0),
    ];
    for (t, d, two, g, l, f) in table {
        assert!(close(significance(Directed, t, &R11), d), "directed {t}");
        assert!(close(significance(TwoSided, t, &R11), two), "two-sided {t}");
        assert!(close(significance(Greater, t, &R11), g), "greater {t}");
        assert!(close(significance(Lesser, t, &R11), l), "lesser {t}");
        assert!(close(significance(Folded, t, &R11), f), "folded {t}");
    }
    let r = [1.0, 2.0, 3.0, 4.0, 5.0];
    assert!(close(significance(TwoSided, 3.0, &r), 5.0 / 6.0));
    assert!(close(significance(TwoSided, 2.5, &r), 5.0 / 6.0));
    assert!(close(significance(Lesser, 2.5, &r), 0.5));
    assert!(close(significance(Folded, 2.5, &r), 5.0 / 6.0));
}

#[test]
fn two_sided_can_exceed_one_when_the_statistic_sits_at_the_median() {
    // R = 1..=5 repeated structure: t = 0.3 above gives exactly 1.0; craft > 1 with ties at the median.
    let r = [1.0, 2.0, 2.0, 2.0, 3.0];
    // pct = 100 * mean(r <= 2) = 80 -> p_low = 20; low = high-side mirror 80th percentile.
    let p = significance(Alternative::TwoSided, 2.0, &r);
    assert!(p > 0.0);
    let r = [5.0, 5.0, 5.0, 5.0];
    // low == high == 5, every value counted twice: (4 + 4 + 1) / 5.
    assert!(close(
        significance(Alternative::TwoSided, 5.0, &r),
        9.0 / 5.0
    ));
}

#[test]
fn percentile_uses_numpy_linear_interpolation() {
    let s = [1.0, 2.0, 4.0, 8.0];
    assert_eq!(percentile(&s, 0.0), 1.0);
    assert_eq!(percentile(&s, 100.0), 8.0);
    assert!(close(percentile(&s, 50.0), 3.0));
    assert!(close(percentile(&s, 25.0), 1.75));
    assert!(close(percentile(&s, 90.0), 6.8));
}

#[test]
fn alternative_parses_names() {
    assert_eq!(Alternative::parse("directed"), Ok(Alternative::Directed));
    assert_eq!(Alternative::parse("two_sided"), Ok(Alternative::TwoSided));
    assert_eq!(Alternative::parse("greater"), Ok(Alternative::Greater));
    assert_eq!(Alternative::parse("lesser"), Ok(Alternative::Lesser));
    assert_eq!(Alternative::parse("folded"), Ok(Alternative::Folded));
    assert!(Alternative::parse("bogus").is_err());
}

#[test]
fn normal_cdf_and_sf_are_accurate() {
    assert!(close(norm_cdf(0.0), 0.5));
    assert!(close(norm_sf(0.0), 0.5));
    assert!(close(norm_cdf(1.96), 0.9750021048517795));
    assert!(close(norm_cdf(-1.96), 0.024997895148220435));
    assert!((norm_sf(5.630312787738188) / 8.994154957945022e-9 - 1.0).abs() < 1e-9);
    assert!((erfc(0.5) / 0.4795001221869535 - 1.0).abs() < 1e-14);
    assert!((erfc(2.5) / 4.069520174449589e-4 - 1.0).abs() < 1e-13);
    // Far tail: sf must not lose precision to 1 - cdf.
    let sf = norm_sf(8.0);
    assert!((sf / 6.220960574271785e-16 - 1.0).abs() < 1e-9, "{sf}");
    assert!(norm_cdf(f64::NAN).is_nan());
    assert!(norm_sf(f64::NAN).is_nan());
}

#[test]
fn summarize_matches_numpy_conventions() {
    let sim = [1.0, 2.0, 3.0, 4.0];
    let s = summarize(5.0, &sim);
    assert!(close(s.ei_sim, 2.5));
    // population std (ddof = 0): sqrt(1.25)
    assert!(close(s.se_sim, 1.25_f64.sqrt()));
    assert!(close(s.vi_sim, 1.25));
    assert!(close(s.z_sim, (5.0 - 2.5) / 1.25_f64.sqrt()));
    // positive z: one-tailed survival.
    assert!(close(s.p_z_sim, norm_sf(s.z_sim)));
    let low = summarize(0.0, &sim);
    assert!(low.z_sim < 0.0);
    assert!(close(low.p_z_sim, norm_cdf(low.z_sim)));
}

#[test]
fn p_value_bounds_hold_for_random_inputs() {
    for seed in 0..20 {
        let base: Vec<f64> = (0..25).map(|i| ((i * 7 + seed) % 13) as f64).collect();
        let sim = simulate(&base, 99, seed as u64, weighted_stat);
        let obs = weighted_stat(&base);
        let p = folded_p(obs, &sim);
        assert!((1.0 / 100.0..=1.0).contains(&p), "{p}");
    }
}

#[test]
fn permutation_is_a_valid_reproducible_shuffle() {
    let p = permutation(50, 7, 3);
    let mut sorted = p.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..50).collect::<Vec<u32>>());
    assert_eq!(p, permutation(50, 7, 3));
    assert_ne!(
        p,
        permutation(50, 7, 4),
        "replicates use independent streams"
    );
    assert_ne!(p, permutation(50, 8, 3), "seeds differ");
    assert_ne!(p, (0..50).collect::<Vec<u32>>());
}

#[test]
fn permutation_of_nothing_or_one_is_trivial() {
    assert!(permutation(0, 1, 0).is_empty());
    assert_eq!(permutation(1, 1, 0), vec![0]);
}

#[test]
fn permutation_is_roughly_uniform_on_three_elements() {
    let mut counts = std::collections::HashMap::new();
    for r in 0..6000 {
        *counts.entry(permutation(3, 5, r)).or_insert(0_u32) += 1;
    }
    assert_eq!(counts.len(), 6);
    for c in counts.values() {
        assert!((800..1200).contains(c), "{counts:?}");
    }
}
