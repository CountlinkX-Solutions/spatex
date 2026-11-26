//! Local spatial heteroscedasticity (`esda.LOSH`, Ord and Getis 2012).
//!
//! The weights are used as given (`Hi` and its variance are invariant to row scaling).
//! `ylag = Wy / rowsum`, `e = |y - ylag|^a`, `Hi = (We) / (mean(e) rowsum)`.
//!
//! Islands (row sum 0): esda's `ylag` is NaN there, which turns `mean(e)` and so EVERY `Hi`
//! into NaN. Here the statistic is computed over the non-island sites only (as if the islands
//! were removed from the data and the graph; `n` in the variance is the number of non-islands)
//! and the islands' own values are NaN. A site whose neighbor is an island (asymmetric graphs)
//! is NaN too, since that neighbor has no residual.

use crate::graph::Csr;

/// LOSH statistics per site.
#[derive(Clone, Debug)]
pub struct Losh {
    /// Local heteroscedasticity `Hi`.
    pub hi: Vec<f64>,
    /// Spatial mean of the neighbors.
    pub ylag: Vec<f64>,
    /// Residuals `|y - ylag|^a`.
    pub yresid: Vec<f64>,
    /// Variance of `Hi`.
    pub var_hi: Vec<f64>,
}

/// LOSH of `y` over `g` with residual exponent `a` (2 for a variance measure, 1 for absolute
/// deviations).
pub fn losh(y: &[f64], g: &Csr, a: f64) -> Result<Losh, String> {
    if y.len() != g.n {
        return Err(format!(
            "y has length {} but the graph has {} nodes (length mismatch)",
            y.len(),
            g.n
        ));
    }
    if let Some(i) = y.iter().position(|v| !v.is_finite()) {
        return Err(format!("y: non-finite value at index {i}"));
    }
    if !(a.is_finite() && a > 0.0) {
        return Err("a must be a positive number".into());
    }
    let rowsum: Vec<f64> = (0..g.n)
        .map(|i| g.row(i).map(|k| g.weights[k]).sum())
        .collect();
    let live: Vec<bool> = rowsum.iter().map(|&s| s != 0.0).collect();
    let m = live.iter().filter(|&&l| l).count();
    if m < 2 {
        return Err("LOSH needs at least 2 sites with neighbors".into());
    }
    let lag = g.lag(y)?;
    let ylag: Vec<f64> = (0..g.n)
        .map(|i| {
            if live[i] {
                lag[i] / rowsum[i]
            } else {
                f64::NAN
            }
        })
        .collect();
    let yresid: Vec<f64> = (0..g.n)
        .map(|i| {
            let d = (y[i] - ylag[i]).abs();
            if a == 2.0 {
                d * d
            } else {
                d.powf(a)
            }
        })
        .collect();
    let mf = m as f64;
    let mean_e = yresid.iter().filter(|e| !e.is_nan()).sum::<f64>() / mf;
    let sum_e2 = yresid
        .iter()
        .filter(|e| !e.is_nan())
        .map(|e| e * e)
        .sum::<f64>();
    let term3 = sum_e2 / mf - mean_e * mean_e;
    let (mut hi, mut var_hi) = (vec![f64::NAN; g.n], vec![f64::NAN; g.n]);
    for i in (0..g.n).filter(|&i| live[i]) {
        let (mut num, mut sq) = (0.0, 0.0);
        for k in g.row(i).filter(|&k| g.weights[k] != 0.0) {
            num += g.weights[k] * yresid[g.neighbors[k] as usize];
            sq += g.weights[k] * g.weights[k];
        }
        let denom = mean_e * rowsum[i];
        hi[i] = num / denom;
        var_hi[i] = (1.0 / (mf - 1.0)) * denom.powi(-2) * term3 * (mf * sq - rowsum[i] * rowsum[i]);
    }
    Ok(Losh {
        hi,
        ylag,
        yresid,
        var_hi,
    })
}

/// Chi-square based p-values of esda: `dof = 2 / VarHi`, `Z = 2 Hi / VarHi`,
/// `pval = P(chi2_dof > Z)`; NaN where either input is NaN or the degrees of freedom are not
/// positive.
pub fn chi2_pvalues(hi: &[f64], var_hi: &[f64]) -> Vec<f64> {
    hi.iter()
        .zip(var_hi)
        .map(|(&h, &v)| chi2_sf(2.0 * h / v, 2.0 / v))
        .collect()
}

/// Survival function of the chi-square distribution: `Q(dof / 2, x / 2)`.
pub fn chi2_sf(x: f64, dof: f64) -> f64 {
    if x.is_nan() || dof.is_nan() || dof <= 0.0 || dof.is_infinite() {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    upper_gamma_q(dof / 2.0, x / 2.0)
}

/// `ln Gamma(x)` for `x > 0` (Lanczos, g = 7, n = 9; relative error ~1e-15).
fn ln_gamma(x: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        return (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let sum = C[0]
        + C.iter()
            .enumerate()
            .skip(1)
            .map(|(i, c)| c / (x + i as f64))
            .sum::<f64>();
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + sum.ln()
}

/// Regularized upper incomplete gamma `Q(a, x)`: series for `x < a + 1`, Lentz continued
/// fraction otherwise (Numerical Recipes 6.2).
fn upper_gamma_q(a: f64, x: f64) -> f64 {
    let prefactor = (a * x.ln() - x - ln_gamma(a)).exp();
    if x < a + 1.0 {
        let (mut term, mut sum, mut ap) = (1.0 / a, 1.0 / a, a);
        for _ in 0..10_000 {
            ap += 1.0;
            term *= x / ap;
            sum += term;
            if term.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        return 1.0 - sum * prefactor;
    }
    const TINY: f64 = 1e-300;
    let mut b = x + 1.0 - a;
    let mut c = 1.0 / TINY;
    let mut d = 1.0 / b;
    let mut h = d;
    for i in 1..10_000 {
        let an = -(i as f64) * (i as f64 - a);
        b += 2.0;
        d = an * d + b;
        if d.abs() < TINY {
            d = TINY;
        }
        c = b + an / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    prefactor * h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * 1.0_f64.max(b.abs())
    }

    fn path(n: usize, islands: usize) -> Csr {
        let mut focal = Vec::new();
        let mut neighbor = Vec::new();
        for i in 0..n as u32 - 1 {
            focal.extend([i, i + 1]);
            neighbor.extend([i + 1, i]);
        }
        let w = vec![1.0; focal.len()];
        Csr::from_arrays(n + islands, &focal, &neighbor, &w).unwrap()
    }

    #[test]
    fn matches_hand_computation_on_a_three_node_path() {
        let l = losh(&[1.0, 2.0, 4.0], &path(3, 0), 2.0).unwrap();
        assert_eq!(l.ylag, vec![2.0, 2.5, 2.0]);
        assert_eq!(l.yresid, vec![1.0, 0.25, 4.0]);
        let hi = [0.25 / 1.75, 5.0 / 3.5, 0.25 / 1.75];
        let var = [2.625 / 3.0625, 2.625 / 12.25, 2.625 / 3.0625];
        for i in 0..3 {
            assert!(close(l.hi[i], hi[i], 1e-12), "{:?}", l.hi);
            assert!(close(l.var_hi[i], var[i], 1e-12), "{:?}", l.var_hi);
        }
    }

    #[test]
    fn row_scaling_does_not_change_hi() {
        let g = path(4, 0);
        let y = [3.0, 1.0, 4.0, 1.5];
        let a = losh(&y, &g, 2.0).unwrap();
        let b = losh(&y, &g.transform(crate::graph::Transform::R), 2.0).unwrap();
        for i in 0..4 {
            assert!(close(a.hi[i], b.hi[i], 1e-12));
            assert!(close(a.var_hi[i], b.var_hi[i], 1e-12));
        }
    }

    #[test]
    fn islands_are_nan_and_the_rest_equals_the_island_free_graph() {
        let y = [1.0, 5.0, 2.0, 8.0, 3.0, 4.0];
        let with = losh(&y, &path(5, 1), 2.0).unwrap();
        let without = losh(&y[..5], &path(5, 0), 2.0).unwrap();
        for i in 0..5 {
            assert!(close(with.hi[i], without.hi[i], 1e-12));
            assert!(close(with.ylag[i], without.ylag[i], 1e-12));
            assert!(close(with.yresid[i], without.yresid[i], 1e-12));
            assert!(close(with.var_hi[i], without.var_hi[i], 1e-12));
        }
        assert!(with.hi[5].is_nan() && with.ylag[5].is_nan());
        assert!(with.yresid[5].is_nan() && with.var_hi[5].is_nan());
    }

    #[test]
    fn rejects_bad_input() {
        let g = path(3, 0);
        assert!(losh(&[1.0, 2.0], &g, 2.0).is_err());
        assert!(losh(&[1.0, f64::NAN, 2.0], &g, 2.0).is_err());
        assert!(losh(&[1.0, 2.0, 3.0], &g, 0.0).is_err());
        let empty = Csr::from_arrays(3, &[], &[], &[]).unwrap();
        assert!(losh(&[1.0, 2.0, 3.0], &empty, 2.0).is_err());
    }

    #[test]
    fn chi_square_survival_matches_closed_forms() {
        assert!(close(chi2_sf(2.0, 2.0), (-1.0_f64).exp(), 1e-13));
        assert!(close(chi2_sf(10.0, 4.0), 6.0 * (-5.0_f64).exp(), 1e-13));
        assert!(close(chi2_sf(3.841458820694124, 1.0), 0.05, 1e-9));
        assert!(close(chi2_sf(5.991464547107979, 2.0), 0.05, 1e-12));
        assert!(close(chi2_sf(0.0, 3.0), 1.0, 1e-15));
        assert!(chi2_sf(1.0, f64::NAN).is_nan() && chi2_sf(1.0, -1.0).is_nan());
    }

    #[test]
    fn chi_square_pvalues_use_dof_2_over_var_and_z_2hi_over_var() {
        let p = chi2_pvalues(&[0.5, f64::NAN], &[2.0, 1.0]);
        // dof = 1, z = 0.5: P(chi2_1 > 0.5) = erfc(sqrt(0.25)).
        assert!(close(p[0], crate::perm::erfc(0.5), 1e-12));
        assert!(p[1].is_nan());
    }
}
