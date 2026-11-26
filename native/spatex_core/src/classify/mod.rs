//! mapclassify-compatible map classifiers.
//!
//! Every classifier mirrors `mapclassify` 2.11.0: the bins are upper bounds, class `i` holds
//! values in `(bins[i-1], bins[i]]` (the first class is open below), and the fit summaries
//! `tss`, `adcm` and `gadf` are computed from the resulting classes.

pub mod bins;
pub mod maxp;
pub mod mt19937;
pub mod optimal;
pub mod stats;

use crate::limits;

/// A classifier and its parameters.
#[derive(Debug, Clone, PartialEq)]
pub enum Method {
    EqualInterval {
        k: usize,
    },
    Quantiles {
        k: usize,
    },
    BoxPlot {
        hinge: f64,
    },
    StdMean {
        multiples: Vec<f64>,
        anchor: bool,
    },
    Percentiles {
        pct: Vec<f64>,
    },
    MaximumBreaks {
        k: usize,
    },
    HeadTailBreaks,
    UserDefined {
        bins: Vec<f64>,
        lowest: Option<f64>,
    },
    FisherJenks {
        k: usize,
    },
    FisherJenksSampled {
        k: usize,
        pct: f64,
        truncate: bool,
        seed: u64,
    },
    JenksCaspall {
        k: usize,
    },
    JenksCaspallSampled {
        k: usize,
        pct: f64,
        seed: u64,
    },
    JenksCaspallForced {
        k: usize,
    },
    NaturalBreaks {
        k: usize,
        initial: usize,
        seed: u64,
    },
    PrettyBreaks {
        k: usize,
    },
    MaxP {
        k: usize,
        initial: usize,
        seed1: u32,
        seed2: u32,
    },
}

/// The result of classifying a vector.
#[derive(Debug, Clone, PartialEq)]
pub struct Classification {
    /// Upper bound of each class.
    pub bins: Vec<f64>,
    /// mapclassify's `k` attribute (the requested `k` for EqualInterval and MaximumBreaks;
    /// the number of bins otherwise, including Quantiles, which deviates from mapclassify here).
    pub k: usize,
    /// Class id of each observation.
    pub yb: Vec<u32>,
    /// Observations per bin.
    pub counts: Vec<u32>,
    /// Total sum of squares around class means.
    pub tss: f64,
    /// Absolute deviation around class medians.
    pub adcm: f64,
    /// Goodness of absolute deviation of fit.
    pub gadf: f64,
    /// Lower edge of the first class: the user's `lowest` or `min(y)`.
    pub lowest: f64,
    /// Interquartile range (BoxPlot only).
    pub iqr: Option<f64>,
    /// Iterations of the Jenks-Caspall family.
    pub iterations: Option<usize>,
    /// Non-fatal issues mapclassify reports as warnings.
    pub warnings: Vec<String>,
}

fn min_max(y: &[f64]) -> (f64, f64) {
    y.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| {
            (lo.min(*v), hi.max(*v))
        })
}

/// Classifies `y` with `method`.
pub fn classify(y: &[f64], method: &Method) -> Result<Classification, String> {
    if y.is_empty() {
        return Err("y must not be empty".into());
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err("y must contain only finite values".into());
    }
    let (lo, hi) = min_max(y);
    let mut warnings = Vec::new();
    let mut iqr = None;
    let mut iterations = None;
    let mut lowest = lo;
    let (bins, k) = match method {
        Method::EqualInterval { k } => {
            check_k(*k)?;
            if lo == hi {
                return Err(all_equal(*k));
            }
            (bins::equal_interval_bins(y, *k), *k)
        }
        Method::Quantiles { k } => {
            check_k(*k)?;
            let (b, short) = bins::quantile_bins(y, *k);
            if short {
                warnings.push(format!(
                    "Not enough unique values in array to form {k} classes. Setting k to {}.",
                    b.len()
                ));
            }
            let n = b.len();
            (b, n)
        }
        Method::BoxPlot { hinge } => {
            let (b, q) = bins::box_plot_bins(y, *hinge);
            iqr = Some(q);
            let n = b.len();
            (b, n)
        }
        Method::StdMean { multiples, anchor } => {
            let b = bins::std_mean_bins(y, multiples, *anchor)?;
            let n = b.len();
            (b, n)
        }
        Method::Percentiles { pct } => {
            let b = bins::percentile_bins(y, pct)?;
            let n = b.len();
            (b, n)
        }
        Method::MaximumBreaks { k } => {
            check_k(*k)?;
            if lo == hi {
                return Err(all_equal(*k));
            }
            let (b, insufficient) = bins::maximum_breaks_bins(y, *k)?;
            if insufficient {
                warnings.push("Insufficient number of unique diffs. Breaks are random.".into());
            }
            (b, *k)
        }
        Method::HeadTailBreaks => {
            let b = bins::head_tail_bins(y);
            let n = b.len();
            (b, n)
        }
        Method::UserDefined {
            bins: user,
            lowest: l,
        } => {
            if user.is_empty() {
                return Err("bins must not be empty".into());
            }
            let mut b = user.clone();
            if b[b.len() - 1] < hi {
                b.push(hi);
            }
            if let Some(l) = l {
                lowest = *l;
            }
            let n = b.len();
            (b, n)
        }
        Method::FisherJenks { k } => {
            check_k(*k)?;
            (fisher_jenks(y, *k)?, *k)
        }
        Method::FisherJenksSampled {
            k,
            pct,
            truncate,
            seed,
        } => {
            check_k(*k)?;
            let mut yr = optimal::sample(y, *pct, *truncate, *seed)?;
            let last = yr.len() - 1;
            yr[last] = hi;
            yr[0] = lo;
            (fisher_jenks(&yr, *k)?, *k)
        }
        Method::JenksCaspall { k } => {
            check_k(*k)?;
            quantile_warning(y, *k, &mut warnings);
            let (b, it) = optimal::jenks_caspall_bins(y, *k);
            iterations = Some(it);
            (b, *k)
        }
        Method::JenksCaspallSampled { k, pct, seed } => {
            check_k(*k)?;
            let mut yr = optimal::sample(y, *pct, true, *seed)?;
            yr[0] = hi;
            let (b, it) = optimal::jenks_caspall_bins(&yr, *k);
            iterations = Some(it);
            (b, *k)
        }
        Method::JenksCaspallForced { k } => {
            check_k(*k)?;
            if lo == hi {
                return Err(all_equal(*k));
            }
            quantile_warning(y, *k, &mut warnings);
            let (b, it) = optimal::jenks_caspall_forced_bins(y, *k)?;
            iterations = Some(it);
            (b, *k)
        }
        Method::NaturalBreaks { k, initial, seed } => {
            check_k(*k)?;
            let (b, short) = optimal::natural_breaks_bins(y, *k, *initial, *seed);
            let n = b.len();
            if short {
                warnings.push(format!(
                    "Not enough unique values in array to form {k} classes. Setting k to {n}."
                ));
            }
            (b, n)
        }
        Method::PrettyBreaks { k } => {
            if *k < 2 {
                // mapclassify divides the range by k - 1.
                return Err(format!("PrettyBreaks needs k >= 2, got {k}"));
            }
            if lo == hi {
                return Err(all_equal(*k));
            }
            (pretty_bins(lo, hi, *k), *k)
        }
        Method::MaxP {
            k,
            initial,
            seed1,
            seed2,
        } => {
            check_k(*k)?;
            if *initial == 0 {
                return Err("initial must be at least 1".into());
            }
            if lo == hi {
                return Err(all_equal(*k));
            }
            (maxp::max_p_bins(y, *k, *initial, *seed1, *seed2)?, *k)
        }
    };
    let (yb, counts) = bins::bin1d(y, &bins);
    let nclasses = k.max(counts.len());
    // mapclassify sums the fit over classes `0..self.k`; PrettyBreaks keeps the requested k even
    // when the rounded breaks give more bins, so those extra classes are left out as there.
    let considered = match method {
        Method::PrettyBreaks { k } => *k,
        _ => nclasses,
    };
    let (tss, adcm, gadf) = fit_statistics(y, &yb, nclasses, considered);
    Ok(Classification {
        bins,
        k,
        yb,
        counts,
        tss,
        adcm,
        gadf,
        lowest,
        iqr,
        iterations,
        warnings,
    })
}

/// mapclassify's `_pretty_number`: `x` rounded to 1, 2, 5 or 10 times a power of ten.
fn pretty_number(x: f64, rounded: bool) -> f64 {
    let exp = x.log10().floor();
    let f = x / 10f64.powf(exp);
    let nf = if rounded {
        if f < 1.5 {
            1.0
        } else if f < 3.0 {
            2.0
        } else if f < 7.0 {
            5.0
        } else {
            10.0
        }
    } else if f <= 1.0 {
        1.0
    } else if f <= 2.0 {
        2.0
    } else if f <= 5.0 {
        5.0
    } else {
        10.0
    };
    nf * 10f64.powf(exp)
}

/// `PrettyBreaks` bins: the numpy `arange` of round steps covering `[lo, hi]`, minus its start.
fn pretty_bins(lo: f64, hi: f64, k: usize) -> Vec<f64> {
    let rg = pretty_number(hi - lo, false);
    let d = pretty_number(rg / (k - 1) as f64, true);
    let miny = (lo / d).floor() * d;
    let maxy = (hi / d).ceil() * d;
    let mut b = stats::arange(miny, maxy + 0.5 * d, d);
    b.remove(0);
    b
}

fn fisher_jenks(y: &[f64], k: usize) -> Result<Vec<f64>, String> {
    let cells = limits::checked_product("Fisher-Jenks table", y.len(), k)?;
    limits::ensure_within(
        "Fisher-Jenks table (n * k)",
        cells,
        limits::MAX_DP_CELLS,
        "use fewer classes or FisherJenksSampled",
    )?;
    let x = stats::sorted(y);
    let mut uv = x.clone();
    uv.dedup();
    if uv.len() < k {
        return Err(format!(
            "Fewer unique values ({}) than specified classes ({k}).",
            uv.len()
        ));
    }
    Ok(optimal::fisher_jenks_bins(&x, k))
}

/// Jenks-Caspall starts from `Quantiles`, which warns when ties leave fewer than `k` bins.
fn quantile_warning(y: &[f64], k: usize, warnings: &mut Vec<String>) {
    let (b, short) = bins::quantile_bins(y, k);
    if short {
        warnings.push(format!(
            "Not enough unique values in array to form {k} classes. Setting k to {}.",
            b.len()
        ));
    }
}

fn all_equal(k: usize) -> String {
    format!("Not enough unique values in array to form {k} classes. All values in `y` are equal.")
}

fn check_k(k: usize) -> Result<(), String> {
    if k < 1 {
        Err("k must be at least 1".into())
    } else {
        Ok(())
    }
}

/// `get_tss`, `get_adcm` and `get_gadf` of `MapClassifier`. Per-class sums run sequentially in
/// observation order (Python's `sum`), the class mean and the global deviation use pairwise sums.
fn fit_statistics(y: &[f64], yb: &[u32], nclasses: usize, considered: usize) -> (f64, f64, f64) {
    let mut classes: Vec<Vec<f64>> = vec![Vec::new(); nclasses];
    for (v, c) in y.iter().zip(yb) {
        classes[*c as usize].push(*v);
    }
    let mut tss = 0.0;
    let mut adcm = 0.0;
    for yc in classes.iter().take(considered).filter(|c| !c.is_empty()) {
        let m = stats::mean(yc);
        let mut css = 0.0;
        for v in yc {
            let d = *v - m;
            css += d * d;
        }
        tss += css;
        let med = stats::median(yc);
        let mut dev = 0.0;
        for v in yc {
            dev += (*v - med).abs();
        }
        adcm += dev;
    }
    let med = stats::median(y);
    let abs_dev: Vec<f64> = y.iter().map(|v| (*v - med).abs()).collect();
    let adam = stats::pairwise_sum(&abs_dev);
    let gadf = if adam == 0.0 { 1.0 } else { 1.0 - adcm / adam };
    (tss, adcm, gadf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(y: &[f64], m: Method) -> Classification {
        classify(y, &m).unwrap()
    }

    #[test]
    fn equal_interval_matches_mapclassify_doc_example() {
        // mapclassify docs: y = [1, 7, 4, 12, 4, 31] -> bins [7.0, 13.0, 19.0, 25.0, 31.0]... width 5
        let c = run(
            &[1.0, 7.0, 4.0, 12.0, 4.0, 31.0],
            Method::EqualInterval { k: 5 },
        );
        assert_eq!(c.bins, vec![7.0, 13.0, 19.0, 25.0, 31.0]);
        assert_eq!(c.yb, vec![0, 0, 0, 1, 0, 4]);
        assert_eq!(c.counts, vec![4, 1, 0, 0, 1]);
        assert_eq!(c.k, 5);
    }

    #[test]
    fn equal_interval_constant_errors() {
        let e = classify(&[2.0; 4], &Method::EqualInterval { k: 3 }).unwrap_err();
        assert!(e.contains("All values"), "{e}");
    }

    #[test]
    fn quantiles_dedups_with_warning() {
        let c = run(&[1.0, 1.0, 1.0, 1.0, 2.0, 5.0], Method::Quantiles { k: 4 });
        assert!(c.bins.len() < 4);
        assert_eq!(c.k, c.bins.len());
        assert_eq!(c.warnings.len(), 1);
        assert_eq!(c.counts.iter().sum::<u32>(), 6);
    }

    #[test]
    fn quantiles_arange_doc() {
        let y: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let c = run(&y, Method::Quantiles { k: 4 });
        assert_eq!(c.bins, vec![249.75, 499.5, 749.25, 999.0]);
        let c3 = run(&y, Method::Quantiles { k: 3 });
        assert_eq!(c3.bins.len(), 3);
        assert_eq!(c3.bins[2], 999.0);
    }

    #[test]
    fn quantiles_top_bin_is_max_even_when_percentile_drifts() {
        // mapclassify gives 45.99999999999999 as the top bin for k = 23, leaving 46 in class 0.
        let y: Vec<f64> = (1..=46).map(|i| i as f64).collect();
        let c = run(&y, Method::Quantiles { k: 23 });
        assert_eq!(*c.bins.last().unwrap(), 46.0);
        assert_eq!(c.yb[45] as usize, c.bins.len() - 1);
    }

    #[test]
    fn box_plot_has_fences_and_outliers() {
        let y = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 100.0];
        let c = run(&y, Method::BoxPlot { hinge: 1.5 });
        assert_eq!(c.k, 6);
        assert_eq!(c.bins.len(), 6);
        assert_eq!(*c.bins.last().unwrap(), 100.0);
        assert_eq!(c.yb[7], 5);
    }

    #[test]
    fn std_mean_appends_max() {
        let y = [1.0, 2.0, 3.0, 4.0, 50.0];
        let c = run(
            &y,
            Method::StdMean {
                multiples: vec![-1.0, 1.0],
                anchor: false,
            },
        );
        assert_eq!(c.bins.len(), 3);
        assert_eq!(*c.bins.last().unwrap(), 50.0);
    }

    #[test]
    fn percentiles_default() {
        let y: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let c = run(
            &y,
            Method::Percentiles {
                pct: vec![1.0, 10.0, 50.0, 90.0, 99.0, 100.0],
            },
        );
        assert_eq!(c.bins.len(), 6);
        assert_eq!(c.bins[5], 100.0);
        assert_eq!(c.counts.iter().sum::<u32>(), 100);
    }

    #[test]
    fn maximum_breaks_picks_largest_gaps() {
        let y = [1.0, 2.0, 3.0, 10.0, 11.0, 30.0, 31.0];
        let c = run(&y, Method::MaximumBreaks { k: 3 });
        assert_eq!(c.bins, vec![6.5, 20.5, 31.0]);
    }

    #[test]
    fn head_tail_breaks_recurses_on_head() {
        let y = [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 10.0, 50.0];
        let c = run(&y, Method::HeadTailBreaks);
        assert_eq!(c.bins[0], 8.5);
        assert_eq!(*c.bins.last().unwrap(), 50.0);
        assert_eq!(c.k, c.bins.len());
    }

    #[test]
    fn user_defined_appends_max_and_sets_lowest() {
        let c = run(
            &[1.0, 5.0, 20.0],
            Method::UserDefined {
                bins: vec![2.0, 10.0],
                lowest: None,
            },
        );
        assert_eq!(c.bins, vec![2.0, 10.0, 20.0]);
        assert_eq!(c.k, 3);
        assert_eq!(c.yb, vec![0, 1, 2]);
        assert_eq!(c.lowest, 1.0);
    }

    #[test]
    fn values_above_last_bin_fall_in_class_zero() {
        // mapclassify bin1d quirk: no interval contains the value, so it stays 0.
        let (yb, counts) = bins::bin1d(&[1.0, 9.0], &[2.0, 5.0]);
        assert_eq!(yb, vec![0, 0]);
        assert_eq!(counts, vec![2, 0]);
    }

    #[test]
    fn find_bin_clamps_like_digitize_right() {
        let b = [2.0, 5.0, 9.0];
        assert_eq!(
            bins::find_bin(&b, &[-1.0, 2.0, 2.5, 5.0, 9.0, 100.0]),
            vec![0, 0, 1, 1, 2, 2]
        );
    }

    #[test]
    fn fit_statistics_on_two_classes() {
        let c = run(
            &[1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
            Method::UserDefined {
                bins: vec![5.0, 12.0],
                lowest: None,
            },
        );
        // class means 2 and 11: tss = 2 + 2
        assert!((c.tss - 4.0).abs() < 1e-12);
        // class medians 2 and 11: adcm = 2 + 2
        assert!((c.adcm - 4.0).abs() < 1e-12);
        // global median 6.5: sum |y-6.5| = 5.5+4.5+3.5+3.5+4.5+5.5 = 27
        assert!((c.gadf - (1.0 - 4.0 / 27.0)).abs() < 1e-12);
    }

    const CLUSTERS: [f64; 8] = [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 30.0, 31.0];

    fn brute_min_tss(y: &[f64], k: usize) -> f64 {
        let mut x = y.to_vec();
        x.sort_by(|a, b| a.total_cmp(b));
        fn ss(v: &[f64]) -> f64 {
            let m = v.iter().sum::<f64>() / v.len() as f64;
            v.iter().map(|a| (a - m) * (a - m)).sum()
        }
        fn go(x: &[f64], k: usize) -> f64 {
            if k == 1 {
                return ss(x);
            }
            (1..=x.len() - k + 1)
                .map(|i| ss(&x[..i]) + go(&x[i..], k - 1))
                .fold(f64::INFINITY, f64::min)
        }
        go(&x, k)
    }

    #[test]
    fn fisher_jenks_finds_the_obvious_clusters() {
        let c = run(&CLUSTERS, Method::FisherJenks { k: 3 });
        assert_eq!(c.bins, vec![3.0, 12.0, 31.0]);
        assert_eq!(c.counts, vec![3, 3, 2]);
        assert_eq!(c.k, 3);
    }

    #[test]
    fn fisher_jenks_table_is_guarded_before_allocating() {
        let y: Vec<f64> = (0..10_000).map(f64::from).collect();
        let e = classify(&y, &Method::FisherJenks { k: 10_000 }).unwrap_err();
        assert!(e.contains("table") && e.contains("100000000"), "{e}");
    }

    #[test]
    fn fisher_jenks_needs_enough_unique_values() {
        let e = classify(&[1.0, 1.0, 2.0], &Method::FisherJenks { k: 3 }).unwrap_err();
        assert!(e.contains("Fewer unique values"), "{e}");
    }

    #[test]
    fn fisher_jenks_is_optimal_on_small_integer_inputs() {
        let y = [4.0, 9.0, 1.0, 17.0, 3.0, 22.0, 8.0, 15.0, 30.0, 2.0];
        for k in 2..=5 {
            let c = run(&y, Method::FisherJenks { k });
            assert_eq!(c.bins.len(), k);
            assert!((c.tss - brute_min_tss(&y, k)).abs() < 1e-9, "k={k}");
        }
    }

    #[test]
    fn fisher_jenks_single_class_is_the_maximum() {
        let c = run(&CLUSTERS, Method::FisherJenks { k: 1 });
        assert_eq!(c.bins, vec![31.0]);
    }

    #[test]
    fn jenks_caspall_converges_to_the_clusters() {
        let c = run(&CLUSTERS, Method::JenksCaspall { k: 3 });
        assert_eq!(c.bins, vec![3.0, 12.0, 31.0]);
        assert!(c.iterations.unwrap() >= 1);
    }

    #[test]
    fn jenks_caspall_forced_returns_k_classes() {
        let c = run(&CLUSTERS, Method::JenksCaspallForced { k: 3 });
        assert_eq!(c.bins.len(), 3);
        assert_eq!(*c.bins.last().unwrap(), 31.0);
        let q = run(&CLUSTERS, Method::Quantiles { k: 3 });
        assert!(c.tss <= q.tss + 1e-12);
    }

    // Regression test: forced moves are by position in `y`, so classes need not be contiguous
    // in value and mapclassify returns class maxima out of order (it even yields an empty
    // class). Spatex returns the cuts sorted so `bins` stays a valid ascending break vector.
    #[test]
    fn jenks_caspall_forced_returns_sorted_bins_for_noncontiguous_classes() {
        let y = [
            0.0,
            0.0,
            1_000_000.0,
            1_000_000.0,
            24291.9921875,
            250_000.0,
            -887329.1015625,
            861099.2431640625,
            875061.03515625,
            321594.23828125,
            0.0,
            -214843.75,
        ];
        let c = run(&y, Method::JenksCaspallForced { k: 6 });
        assert_eq!(
            c.bins,
            vec![
                -887329.1015625,
                -214843.75,
                24291.9921875,
                321594.23828125,
                875061.03515625,
                1_000_000.0
            ]
        );
        assert_eq!(c.counts.iter().sum::<u32>() as usize, y.len());
    }

    #[test]
    fn natural_breaks_matches_fisher_jenks_on_separated_clusters() {
        let m = Method::NaturalBreaks {
            k: 3,
            initial: 10,
            seed: 7,
        };
        let c = run(&CLUSTERS, m.clone());
        assert_eq!(c.bins, vec![3.0, 12.0, 31.0]);
        assert_eq!(run(&CLUSTERS, m.clone()).bins, c.bins);
    }

    #[test]
    fn natural_breaks_with_few_unique_values_uses_every_value() {
        let c = run(
            &[1.0, 1.0, 2.0, 2.0, 5.0],
            Method::NaturalBreaks {
                k: 5,
                initial: 10,
                seed: 0,
            },
        );
        assert_eq!(c.bins, vec![1.0, 2.0, 5.0]);
        assert_eq!(c.k, 3);
        assert_eq!(c.warnings.len(), 1);
    }

    #[test]
    fn natural_breaks_reaches_the_optimum_on_skewed_data() {
        let y: Vec<f64> = (1..=60).map(|i| (i as f64).powi(2)).collect();
        let nb = run(
            &y,
            Method::NaturalBreaks {
                k: 4,
                initial: 10,
                seed: 3,
            },
        );
        let fj = run(&y, Method::FisherJenks { k: 4 });
        assert!(nb.tss <= fj.tss * (1.0 + 1e-6));
    }

    #[test]
    fn sampled_variants_are_seeded_and_keep_the_maximum() {
        let y: Vec<f64> = (0..400)
            .map(|i| ((i * 37) % 101) as f64 + i as f64 / 7.0)
            .collect();
        let m = Method::FisherJenksSampled {
            k: 4,
            pct: 0.2,
            truncate: true,
            seed: 5,
        };
        let a = run(&y, m.clone());
        assert_eq!(a.bins, run(&y, m).bins);
        assert_eq!(a.bins.len(), 4);
        let max = y.iter().copied().fold(f64::MIN, f64::max);
        assert_eq!(*a.bins.last().unwrap(), max);
        let j = run(
            &y,
            Method::JenksCaspallSampled {
                k: 4,
                pct: 0.2,
                seed: 5,
            },
        );
        assert_eq!(*j.bins.last().unwrap(), max);
        assert_eq!(j.counts.iter().sum::<u32>(), 400);
    }
}
