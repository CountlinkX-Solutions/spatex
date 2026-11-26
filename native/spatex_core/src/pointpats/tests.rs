use super::centrography::*;
use super::distance::*;
use super::hull::*;
use super::quadrat::*;
use super::random::*;
use super::*;

fn approx(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} != {b}");
}

const P10: [Pt; 10] = [
    [66.22, 32.54],
    [22.52, 38.53],
    [31.01, 81.21],
    [9.47, 31.02],
    [30.78, 60.10],
    [75.21, 58.44],
    [79.26, 85.36],
    [8.23, 39.93],
    [98.73, 77.17],
    [89.78, 42.53],
];

#[test]
fn linspace_matches_numpy_arithmetic() {
    assert_eq!(
        linspace(0.0, 30.0, 7),
        vec![0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    );
    assert_eq!(linspace(1.0, 2.0, 1), vec![1.0]);
    assert_eq!(*linspace(0.0, 0.3, 4).last().unwrap(), 0.3);
}

#[test]
fn std_distance_is_the_raw_moment_form() {
    approx(std_distance(&P10).unwrap(), 38.00298369865186, 1e-12);
}

#[test]
fn centrography_probed_values() {
    let c = mean_center(&P10).unwrap();
    approx(c[0], 51.121, 1e-12);
    let (maj, min, ang) = ellipse(&P10, None, EllipseMethod::Crimestat).unwrap();
    approx(maj, 53.612878559714375, 1e-12);
    approx(min, 27.133488119689254, 1e-12);
    approx(ang, 0.33520089366330835, 1e-12);
    let m = euclidean_median(&P10).unwrap();
    approx(m[0], 49.03534437711376, 1e-6);
    approx(m[1], 54.35466949626994, 1e-6);
}

#[test]
fn euclidean_median_handles_a_data_point_optimum() {
    // the median of an odd collinear-ish cluster sits at the central data point
    let p = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 0.0], [1.0, 0.0]];
    let m = euclidean_median(&p).unwrap();
    approx(m[0], 1.0, 1e-9);
    approx(m[1], 0.0, 1e-9);
    assert_eq!(euclidean_median(&[[3.0, 4.0]]).unwrap(), [3.0, 4.0]);
}

#[test]
fn hull_is_ccw_from_the_lowest_x() {
    let h = convex_hull(&[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1.0, 1.0], [1.0, 0.5]]).unwrap();
    assert_eq!(h, vec![[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]]);
    assert!(convex_hull(&[[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]).is_err());
}

#[test]
fn rotated_rectangle_of_a_square_and_circle_of_a_triangle() {
    let r = minimum_rotated_rectangle(&[[0.0, 0.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]]).unwrap();
    approx(r.area, 1.0, 1e-12);
    approx(r.angle, 0.0, 1e-12);
    let (c, rad) = minimum_bounding_circle(&[[0.0, 0.0], [4.0, 0.0], [2.0, 1.0]]).unwrap();
    approx(c[0], 2.0, 1e-12);
    approx(c[1], 0.0, 1e-12);
    approx(rad, 2.0, 1e-12);
    let (c, rad) = minimum_bounding_circle(&P10).unwrap();
    approx(c[0], 54.1, 1e-9);
    approx(c[1], 54.095, 1e-9);
    approx(rad, 50.242337973068096, 1e-9);
}

#[test]
fn g_counts_follow_numpy_histogram_semantics() {
    // NN distances: 1, 1, 2, 2 -> support [0,1,2]: bins [0,1) and [1,2] (closed)
    let p = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [5.0, 0.0]];
    let (s, g) = super::distance::g(&p, &SupportSpec::Explicit(vec![0.0, 1.0, 2.0])).unwrap();
    assert_eq!(s, vec![0.0, 1.0, 2.0]);
    assert_eq!(g, vec![0.0, 0.0, 1.0]);
}

#[test]
fn k_counts_pairs_strictly_below() {
    let p = [[0.0, 0.0], [1.0, 0.0], [3.0, 0.0], [0.0, 2.0]];
    // bbox area 6, n = 4, intensity 2/3; pairs at distances 1, 2, 2, 2.236.., 3, 3.6..
    let (_, k) = super::distance::k(&p, &SupportSpec::Explicit(vec![0.0, 1.0, 2.0, 2.5])).unwrap();
    // strict: d = 1 and 2 are not below 1 and 2
    assert_eq!(k[0], 0.0);
    assert_eq!(k[1], 0.0);
    approx(k[2], 1.0 * 2.0 / 4.0 / (4.0 / 6.0), 1e-12);
    approx(k[3], 4.0 * 2.0 / 4.0 / (4.0 / 6.0), 1e-12);
}

#[test]
fn duplicate_points_have_zero_nearest_distance() {
    let p = [[0.0, 0.0], [0.0, 0.0], [4.0, 0.0]];
    let (_, g) = super::distance::g(&p, &SupportSpec::Explicit(vec![0.0, 2.0, 4.0])).unwrap();
    // distances 0, 0, 4 -> bins [0,2): 2, [2,4]: 1
    assert_eq!(g[0], 0.0);
    approx(g[1], 2.0 / 3.0, 1e-15);
    assert_eq!(g[2], 1.0);
}

fn opts(kind: Kind) -> TestOptions {
    TestOptions {
        kind,
        spec: SupportSpec::Int(8),
        window: Window::Bbox([0.0, 0.0, 10.0, 10.0]),
        n_simulations: 49,
        seed: 3,
        keep_simulations: true,
        truncate: true,
        linearized: false,
        n_empty: 200,
    }
}

fn pattern() -> Vec<Pt> {
    poisson(&Window::Bbox([0.0, 0.0, 10.0, 10.0]), 40, 1, 99)
        .unwrap()
        .remove(0)
}

#[test]
fn simulations_do_not_depend_on_the_thread_count() {
    let p = pattern();
    for kind in [Kind::G, Kind::F, Kind::J, Kind::K, Kind::L] {
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| ripley_test(&p, &opts(kind)).unwrap())
        };
        let (a, b) = (run(1), run(4));
        assert_eq!(a.simulations, b.simulations, "{kind:?}");
        assert_eq!(a.pvalue, b.pvalue, "{kind:?}");
    }
}

#[test]
fn keeping_simulations_does_not_change_the_summary() {
    let p = pattern();
    let kept = ripley_test(&p, &opts(Kind::K)).unwrap();
    let mut o = opts(Kind::K);
    o.keep_simulations = false;
    let lean = ripley_test(&p, &o).unwrap();
    assert!(lean.simulations.is_none());
    assert_eq!(kept.pvalue, lean.pvalue);
    assert_eq!(kept.envelope_low, lean.envelope_low);
    assert_eq!(kept.envelope_high, lean.envelope_high);
}

#[test]
fn pvalue_folds_and_is_zero_at_distance_zero() {
    let r = ripley_test(&pattern(), &opts(Kind::G)).unwrap();
    assert_eq!(r.pvalue[0], 0.0);
    assert!(r.pvalue.iter().all(|p| (0.0..=0.5).contains(p)));
}

#[test]
fn poisson_is_in_window_and_seeded() {
    let w = Window::Convex(vec![[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]]);
    let a = poisson(&w, 100, 3, 5).unwrap();
    assert_eq!(a, poisson(&w, 100, 3, 5).unwrap());
    assert!(a.iter().flatten().all(|q| q[0] + q[1] <= 4.0 + 1e-12));
    assert!(poisson(&Window::Bbox([0.0, 0.0, 0.0, 1.0]), 3, 1, 1).is_err());
}

#[test]
fn size_and_intensity_parsing() {
    assert_eq!(
        parse_size_and_intensity(50.0, Some(2.0), None, None).unwrap(),
        (100, 1, 2.0)
    );
    assert_eq!(
        parse_size_and_intensity(50.0, None, Some(20), Some(3)).unwrap(),
        (20, 3, 0.4)
    );
    assert_eq!(
        parse_size_and_intensity(50.0, None, None, None).unwrap(),
        (100, 1, 2.0)
    );
    assert!(parse_size_and_intensity(50.0, Some(1.0), Some(2), Some(2)).is_err());
}

#[test]
fn python_floor_division() {
    assert_eq!(py_floordiv_for_test(7.0, 2.0), 3.0);
    assert_eq!(py_floordiv_for_test(0.6, 0.2), 2.0); // 0.6 % 0.2 is almost 0.2 in IEEE
    assert_eq!(py_floordiv_for_test(1.0, 0.1), 9.0); // CPython: 1.0 // 0.1 == 9.0
}

fn py_floordiv_for_test(a: f64, b: f64) -> f64 {
    super::quadrat::py_floordiv(a, b)
}

#[test]
fn rectangle_grid_puts_max_coordinates_in_the_last_cell() {
    let p = [[0.0, 0.0], [3.0, 3.0], [1.5, 1.5], [0.2, 2.9]];
    let g = grid_counts(
        &p,
        Shape::Rectangle {
            nx: 3,
            ny: 3,
            width: 0.0,
            height: 0.0,
        },
        crate::limits::DEFAULT_MAX_CELLS,
    )
    .unwrap();
    assert_eq!(g.counts, vec![1, 0, 0, 0, 1, 0, 1, 0, 1]);
}

#[test]
fn chisquare_uses_the_uniform_expectation() {
    let (chi2, p, df, c) = chisquare(&[4, 3, 3, 4, 3, 3, 3, 6, 1]);
    approx(chi2, 4.2, 1e-12);
    assert_eq!(df, 8);
    approx(p, 0.8386428488905466, 1e-9);
    assert_eq!(c.len(), 9);
}

#[test]
fn quadrat_realizations_are_seeded() {
    let p = pattern();
    let shape = Shape::Rectangle {
        nx: 3,
        ny: 3,
        width: 0.0,
        height: 0.0,
    };
    let a = quadrat_test(&p, shape, 30, 2, crate::limits::DEFAULT_MAX_CELLS).unwrap();
    let b = quadrat_test(&p, shape, 30, 2, crate::limits::DEFAULT_MAX_CELLS).unwrap();
    assert_eq!(a.chi2_realizations, b.chi2_realizations);
    assert!(a.chi2_r_pvalue.unwrap() > 0.0);
}

#[test]
fn cluster_chunks_follow_array_split() {
    let w = Window::Bbox([0.0, 0.0, 10.0, 10.0]);
    let sims = cluster_poisson(&w, 11, 2, 3, &[0.4], 1).unwrap();
    assert_eq!(sims.len(), 2);
    assert!(sims.iter().all(|s| s.len() == 11));
    // sizes 4, 4, 3: the seeds sit at chunk starts
    let s = &sims[0];
    for (start, size) in [(0usize, 4usize), (4, 4), (8, 3)] {
        let seed = s[start];
        for q in &s[start..start + size] {
            assert!((q[0] - seed[0]).hypot(q[1] - seed[1]) <= 0.4 + 1e-12);
        }
    }
    assert!(cluster_poisson(&w, 2, 1, 3, &[0.4], 1).is_err());
}

fn rect(nx: usize, ny: usize, width: f64, height: f64) -> Shape {
    Shape::Rectangle {
        nx,
        ny,
        width,
        height,
    }
}

#[test]
fn rectangle_grid_over_max_cells_errors_with_the_cell_count() {
    let err = grid_counts(&P10, rect(4, 4, 0.0, 0.0), 10).unwrap_err();
    assert!(
        err.contains("16 cells") && err.contains("max_cells"),
        "{err}"
    );
    assert!(grid_counts(&P10, rect(4, 4, 0.0, 0.0), 16).is_ok());
}

#[test]
fn rectangle_grid_multiplication_overflow_is_an_error_not_a_wrap() {
    let err = grid_counts(&P10, rect(usize::MAX, 2, 0.0, 0.0), usize::MAX).unwrap_err();
    assert!(err.contains("overflow"), "{err}");
}

#[test]
fn rectangle_cell_size_grid_is_guarded_before_allocating() {
    let p = [[0.0, 0.0], [1.0e6, 1.0e6]];
    let err = grid_counts(&p, rect(3, 3, 1e-3, 1e-3), 1_000_000).unwrap_err();
    assert!(err.contains("cells") && err.contains("max_cells"), "{err}");
}

#[test]
fn hexagon_grid_over_max_cells_errors_before_allocating() {
    let p = [[0.0, 0.0], [1.0e6, 1.0e6]];
    let err = grid_counts(&p, Shape::Hexagon { lh: 1e-3 }, 1_000_000).unwrap_err();
    assert!(err.contains("cells") && err.contains("max_cells"), "{err}");
}

#[test]
fn quadrat_test_forwards_max_cells() {
    let err = quadrat_test(&pattern(), rect(5, 5, 0.0, 0.0), 0, 1, 20).unwrap_err();
    assert!(err.contains("25 cells"), "{err}");
}

#[test]
fn simulated_point_budget_is_checked_before_sampling() {
    let err = parse_size_and_intensity(1.0, None, Some(60_000_000), Some(1)).unwrap_err();
    assert!(err.contains("points") && err.contains("60000000"), "{err}");
    let err = parse_size_and_intensity(1.0, None, Some(usize::MAX), Some(2)).unwrap_err();
    assert!(err.contains("overflow"), "{err}");
    let err = parse_size_and_intensity(1.0, Some(1e30), None, Some(1)).unwrap_err();
    assert!(err.contains("points"), "{err}");
    assert!(parse_size_and_intensity(1.0, None, Some(1_000), Some(10)).is_ok());
}

fn mean_cov(p: &[Pt]) -> ([f64; 2], [f64; 3]) {
    let n = p.len() as f64;
    let mx = p.iter().map(|q| q[0]).sum::<f64>() / n;
    let my = p.iter().map(|q| q[1]).sum::<f64>() / n;
    let (mut sxx, mut syy, mut sxy) = (0.0, 0.0, 0.0);
    for q in p {
        sxx += (q[0] - mx).powi(2);
        syy += (q[1] - my).powi(2);
        sxy += (q[0] - mx) * (q[1] - my);
    }
    ([mx, my], [sxx / n, syy / n, sxy / n])
}

#[test]
fn normal_matches_its_moments_inside_a_wide_window() {
    // A window many standard deviations wide: truncation is negligible.
    let w = Window::Bbox([-50.0, -50.0, 50.0, 50.0]);
    let cov = [4.0, 1.2, 1.2, 1.0];
    let sims = normal(&w, 20_000, 1, Some([3.0, -2.0]), Some(cov), 11).unwrap();
    let (m, c) = mean_cov(&sims[0]);
    approx(m[0], 3.0, 0.05);
    approx(m[1], -2.0, 0.05);
    approx(c[0], 4.0, 0.15);
    approx(c[1], 1.0, 0.05);
    approx(c[2], 1.2, 0.07);
}

#[test]
fn normal_defaults_follow_pointpats() {
    // Default centre is the window centre; default covariance is (max(w, h) / 2)^2 * I,
    // so the standard deviation is half the longer side.
    let w = Window::Bbox([0.0, 0.0, 1000.0, 400.0]);
    let sims = normal(&w, 30_000, 1, None, None, 2).unwrap();
    assert!(sims[0].iter().all(|q| w.contains(q[0], q[1])));
    let (m, _) = mean_cov(&sims[0]);
    approx(m[0], 500.0, 6.0);
    approx(m[1], 200.0, 6.0);
}

#[test]
fn normal_is_seeded_contained_and_validated() {
    let w = Window::Convex(vec![[0.0, 0.0], [10.0, 0.0], [5.0, 8.0]]);
    let a = normal(&w, 50, 3, None, None, 5).unwrap();
    assert_eq!(a, normal(&w, 50, 3, None, None, 5).unwrap());
    assert_ne!(a, normal(&w, 50, 3, None, None, 6).unwrap());
    assert!(a.iter().flatten().all(|q| w.contains(q[0], q[1])));
    assert_ne!(a[0], a[1]);
    assert!(normal(&w, 5, 1, None, Some([1.0, 0.0, 0.0, -1.0]), 1).is_err());
    assert!(normal(&w, 5, 1, None, Some([1.0, 0.5, 0.2, 1.0]), 1).is_err());
    assert!(normal(&w, 5, 1, None, Some([1.0, 2.0, 2.0, 1.0]), 1).is_err());
    // a centre that never lands in the window fails instead of looping forever
    let far = normal(&w, 5, 1, Some([1e6, 1e6]), Some([1.0, 0.0, 0.0, 1.0]), 1);
    assert!(far.is_err());
}

#[test]
fn cluster_normal_clusters_and_counts() {
    let w = Window::Bbox([0.0, 0.0, 100.0, 100.0]);
    let cov = Some([4.0, 0.0, 0.0, 4.0]);
    let sims = cluster_normal(&w, 203, 4, 3, cov, 7).unwrap();
    assert_eq!(sims.len(), 4);
    assert!(sims
        .iter()
        .all(|s| s.len() == 203 && s.iter().all(|q| w.contains(q[0], q[1]))));
    assert_eq!(sims, cluster_normal(&w, 203, 4, 3, cov, 7).unwrap());
    // clusters hold 68, 68 and 67 points, each starting with its centre
    let s = &sims[0];
    for (start, len) in [(0, 68), (68, 68), (136, 67)] {
        let c = s[start];
        let near = s[start..start + len]
            .iter()
            .filter(|q| (q[0] - c[0]).hypot(q[1] - c[1]) < 10.0)
            .count();
        assert!(near >= len - 2, "cluster at {start}: {near}/{len}");
    }
    // clustered patterns are tighter than CSR
    let csr = poisson(&w, 203, 1, 7).unwrap();
    assert!(nn_mean(&s[..]) < nn_mean(&csr[0]));
}

fn nn_mean(p: &[Pt]) -> f64 {
    let mut total = 0.0;
    for (i, a) in p.iter().enumerate() {
        let mut m = f64::INFINITY;
        for (j, b) in p.iter().enumerate() {
            if i != j {
                m = m.min((a[0] - b[0]).hypot(a[1] - b[1]));
            }
        }
        total += m;
    }
    total / p.len() as f64
}

#[test]
fn cluster_normal_default_cov_and_errors() {
    let w = Window::Bbox([0.0, 0.0, 10.0, 10.0]);
    let a = cluster_normal(&w, 30, 2, 3, None, 1).unwrap();
    assert_eq!(a, cluster_normal(&w, 30, 2, 3, None, 1).unwrap());
    assert!(cluster_normal(&w, 2, 1, 3, None, 1).is_err());
    // the default covariance needs two seeds to measure a distance between
    assert!(cluster_normal(&w, 10, 1, 1, None, 1).is_err());
    assert!(cluster_normal(&w, 10, 1, 1, Some([1.0, 0.0, 0.0, 1.0]), 1).is_ok());
}

fn on_one_thread<T: Send>(f: impl FnOnce() -> T + Send) -> T {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(f)
}

#[test]
fn normal_processes_do_not_depend_on_the_thread_count() {
    let w = Window::Bbox([0.0, 0.0, 10.0, 10.0]);
    let run = || {
        (
            normal(&w, 30, 8, None, Some([2.0, 0.5, 0.5, 1.0]), 3).unwrap(),
            cluster_normal(&w, 30, 8, 3, None, 3).unwrap(),
        )
    };
    assert_eq!(run(), on_one_thread(run));
}

fn close_pairs(p: &[Pt], r: f64) -> usize {
    let mut k = 0;
    for i in 0..p.len() {
        for j in (i + 1)..p.len() {
            if (p[i][0] - p[j][0]).hypot(p[i][1] - p[j][1]) <= r {
                k += 1;
            }
        }
    }
    k
}

#[test]
fn strauss_inhibits_close_pairs() {
    let w = Window::Bbox([0.0, 0.0, 1.0, 1.0]);
    let r = 0.08;
    let mean_pairs = |gamma: f64| {
        let sims = strauss(&w, 60, 12, gamma, r, 4000, 10, 3).unwrap();
        assert!(sims.iter().all(|s| s.len() == 60));
        assert!(sims.iter().flatten().all(|q| w.contains(q[0], q[1])));
        sims.iter().map(|s| close_pairs(s, r)).sum::<usize>() as f64 / 12.0
    };
    let (g1, g_low, g_zero) = (mean_pairs(1.0), mean_pairs(0.05), mean_pairs(0.001));
    assert!(g_low < 0.6 * g1, "{g_low} vs {g1}");
    assert!(g_zero <= g_low + 0.5, "{g_zero} vs {g_low}");
    // gamma = 1 is binomial: the expected number of close pairs is C(60, 2) * pi r^2 (less edge)
    let csr = (60.0 * 59.0 / 2.0) * std::f64::consts::PI * r * r;
    assert!(g1 > 0.6 * csr && g1 < 1.2 * csr, "{g1} vs {csr}");
}

#[test]
fn strauss_is_seeded_and_validated() {
    let w = Window::Bbox([0.0, 0.0, 1.0, 1.0]);
    let a = strauss(&w, 20, 3, 0.2, 0.05, 500, 10, 9).unwrap();
    assert_eq!(a, strauss(&w, 20, 3, 0.2, 0.05, 500, 10, 9).unwrap());
    assert_ne!(a, strauss(&w, 20, 3, 0.2, 0.05, 500, 10, 10).unwrap());
    assert_ne!(a[0], a[1]);
    assert!(strauss(&w, 20, 1, -0.1, 0.05, 500, 10, 1).is_err());
    assert!(strauss(&w, 20, 1, f64::NAN, 0.05, 500, 10, 1).is_err());
    assert!(strauss(&w, 20, 1, 0.5, 0.0, 500, 10, 1).is_err());
    assert!(strauss(&w, 20, 1, 0.5, 0.05, 500, 0, 1).is_err());
    // gamma = 0 with a death move over a close pair is the pointpats RuntimeError
    assert!(strauss(&w, 50, 1, 0.0, 0.5, 2000, 2, 1).is_err());
}

#[test]
fn strauss_does_not_depend_on_the_thread_count() {
    let w = Window::Bbox([0.0, 0.0, 1.0, 1.0]);
    let run = || strauss(&w, 25, 6, 0.3, 0.06, 800, 10, 4).unwrap();
    assert_eq!(run(), on_one_thread(run));
}
