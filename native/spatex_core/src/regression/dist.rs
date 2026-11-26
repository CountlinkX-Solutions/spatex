//! Survival functions of the t, F and chi-square distributions (plus the normal, re-exported
//! from `perm`), accurate in the far tail.
//!
//! Everything reduces to the regularized incomplete beta `I_x(a, b)` and gamma `Q(a, x)`.
//! Each is evaluated on the side where it is small (series or continued fraction), so a
//! survival probability such as `9.2e-19` keeps full relative accuracy; `1 - cdf` is never used.

const EPS: f64 = 1e-16;
const TINY: f64 = 1e-300;
const MAX_ITER: usize = 20_000;

/// `ln Beta(a, b)`.
fn ln_beta(a: f64, b: f64) -> f64 {
    libm::lgamma(a) + libm::lgamma(b) - libm::lgamma(a + b)
}

/// Continued fraction for the incomplete beta (modified Lentz).
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=MAX_ITER {
        let m = m as f64;
        let m2 = 2.0 * m;
        let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Regularized incomplete beta `I_x(a, b)`, evaluated on the side where the continued
/// fraction converges quickly; small results keep full relative accuracy.
fn betai(a: f64, b: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || b.is_nan() || a <= 0.0 || b <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let ln_front = a * x.ln() + b * (-x).ln_1p() - ln_beta(a, b);
    if x < (a + 1.0) / (a + b + 2.0) {
        ln_front.exp() * beta_cf(a, b, x) / a
    } else {
        1.0 - ln_front.exp() * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Regularized upper incomplete gamma `Q(a, x)`.
fn gammaq(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let ln_front = -x + a * x.ln() - libm::lgamma(a);
    if x < a + 1.0 {
        // Series for P(a, x); here Q is not small, so 1 - P keeps relative accuracy.
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..MAX_ITER {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * EPS {
                break;
            }
        }
        1.0 - ln_front.exp() * sum
    } else {
        // Continued fraction for Q (modified Lentz).
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..=MAX_ITER {
            let i = i as f64;
            let an = -i * (i - a);
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
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < EPS {
                break;
            }
        }
        ln_front.exp() * h
    }
}

/// Student t two-sided p-value `2 * P(T > |t|)` with `df` degrees of freedom.
pub fn t_two_sided(t: f64, df: f64) -> f64 {
    if t.is_nan() || df.is_nan() || df <= 0.0 {
        return f64::NAN;
    }
    if t.is_infinite() {
        return 0.0;
    }
    betai(df / 2.0, 0.5, df / (df + t * t))
}

/// F survival function `P(F > x)` with `(d1, d2)` degrees of freedom.
pub fn f_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x.is_nan() || d1.is_nan() || d2.is_nan() || d1 <= 0.0 || d2 <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x.is_infinite() {
        return 0.0;
    }
    betai(d2 / 2.0, d1 / 2.0, d2 / (d2 + d1 * x))
}

/// Chi-square survival function `P(X > x)` with `df` degrees of freedom.
pub fn chi2_sf(x: f64, df: f64) -> f64 {
    gammaq(df / 2.0, x / 2.0)
}

/// Two-sided normal p-value `2 * P(Z > |z|)`.
pub fn norm_two_sided(z: f64) -> f64 {
    2.0 * crate::perm::norm_sf(z.abs())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, e: f64, rel: f64) {
        assert!(
            (a - e).abs() <= rel * e.abs().max(f64::MIN_POSITIVE),
            "got {a:e}, expected {e:e}"
        );
    }

    // Expected values from scipy.stats (t.sf * 2, f.sf, chi2.sf, norm.sf * 2).
    const T: [(f64, f64, f64); 8] = [
        (14.49037314368909, 46.0, 9.210889988917443e-19),
        (-4.780496191296567, 46.0, 1.8289595070843706e-05),
        (0.3, 3.0, 0.783763292039919),
        (2.5, 1.0, 0.24223788318168682),
        (0.0, 10.0, 1.0),
        (40.0, 196.0, 3.1561482714254195e-96),
        (8.0, 5.0, 0.0004929066605724442),
        (1.2, 500.0, 0.23070761116864938),
    ];
    const F: [(f64, f64, f64, f64); 6] = [
        (28.385629224694835, 2.0, 46.0, 9.340747100510886e-09),
        (0.5, 1.0, 1.0, 0.6081734479693928),
        (3.3, 4.0, 195.0, 0.012098247546830585),
        (100.0, 3.0, 7.0, 4.130422101383355e-06),
        (1.0, 10.0, 10.0, 0.5),
        (0.001, 2.0, 30.0, 0.9990005331319123),
    ];
    const CHI2: [(f64, f64, f64); 9] = [
        (1.8357525200758478, 2.0, 0.3993662912487855),
        (7.900441675960222, 2.0, 0.019250450075424905),
        (19.94600823990323, 5.0, 0.001279222817392593),
        (0.01, 1.0, 0.920344325445942),
        (50.0, 1.0, 1.537459794428033e-12),
        (120.0, 8.0, 3.3153025398645396e-22),
        (3.0, 40.0, 0.9999999999999997),
        (0.5, 200.0, 1.0),
        (5.8148, 1.0, 0.015891860498097467),
    ];
    const NORM: [(f64, f64); 4] = [
        (2.9367861608440484, 0.003316326943384204),
        (0.1, 0.920344325445942),
        (6.5, 8.032001167718176e-11),
        (12.0, 3.552964224155306e-33),
    ];

    #[test]
    fn t_matches_scipy() {
        for (x, df, e) in T {
            close(t_two_sided(x, df), e, 1e-11);
        }
    }

    #[test]
    fn f_matches_scipy() {
        for (x, d1, d2, e) in F {
            close(f_sf(x, d1, d2), e, 1e-11);
        }
    }

    #[test]
    fn chi2_matches_scipy() {
        for (x, df, e) in CHI2 {
            close(chi2_sf(x, df), e, 1e-11);
        }
    }

    #[test]
    fn normal_matches_scipy() {
        for (z, e) in NORM {
            close(norm_two_sided(z), e, 1e-11);
        }
    }

    #[test]
    fn degenerate_arguments() {
        assert_eq!(chi2_sf(-1.0, 2.0), 1.0);
        assert_eq!(chi2_sf(0.0, 2.0), 1.0);
        assert!(chi2_sf(f64::NAN, 2.0).is_nan());
        assert!(t_two_sided(1.0, 0.0).is_nan());
        assert_eq!(f_sf(-1.0, 2.0, 3.0), 1.0);
    }
}
