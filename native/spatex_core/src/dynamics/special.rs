//! Special functions behind the homogeneity tests.

use std::f64::consts::PI;

/// Natural log of the gamma function (Lanczos, `g = 7`), accurate to about 1e-15.
pub fn ln_gamma(x: f64) -> f64 {
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
        return PI.ln() - (PI * x).sin().ln() - ln_gamma(1.0 - x);
    }
    let x = x - 1.0;
    let t = x + 7.5;
    let mut a = C[0];
    for (i, c) in C.iter().enumerate().skip(1) {
        a += c / (x + i as f64);
    }
    0.5 * (2.0 * PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
}

const FPMIN: f64 = 1e-300;

/// Regularized upper incomplete gamma function `Q(a, x)`.
pub fn gamma_q(a: f64, x: f64) -> f64 {
    if x.is_nan() || a.is_nan() || a <= 0.0 {
        return f64::NAN;
    }
    if x <= 0.0 {
        return 1.0;
    }
    let prefactor = (-x + a * x.ln() - ln_gamma(a)).exp();
    if x < a + 1.0 {
        // series for P(a, x)
        let mut ap = a;
        let mut del = 1.0 / a;
        let mut sum = del;
        for _ in 0..100_000 {
            ap += 1.0;
            del *= x / ap;
            sum += del;
            if del.abs() < sum.abs() * 1e-17 {
                break;
            }
        }
        1.0 - sum * prefactor
    } else {
        // continued fraction for Q(a, x) (modified Lentz)
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / FPMIN;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..100_000 {
            let an = -f64::from(i) * (f64::from(i) - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < FPMIN {
                d = FPMIN;
            }
            c = b + an / c;
            if c.abs() < FPMIN {
                c = FPMIN;
            }
            d = 1.0 / d;
            let del = d * c;
            h *= del;
            if (del - 1.0).abs() < 1e-16 {
                break;
            }
        }
        prefactor * h
    }
}

/// Survival function of the chi-squared distribution, `1 - chi2.cdf(x, dof)`.
///
/// `dof <= 0` gives `NaN` like scipy.
pub fn chi2_sf(x: f64, dof: f64) -> f64 {
    if dof <= 0.0 || dof.is_nan() {
        return f64::NAN;
    }
    gamma_q(dof / 2.0, x / 2.0)
}
