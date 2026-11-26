//! Lorenz-curve based measures (`inequality` 1.2.0): Atkinson, Schutz, Wolfson.
//!
//! Arithmetic follows numpy's (sequential sums, `linspace` as `i * step`, `interp` as a
//! two-point line) so results agree with the Python package to rounding.

use super::gini;

/// Atkinson index and equally distributed equivalent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Atkinson {
    /// Atkinson index.
    pub a: f64,
    /// Equally distributed equivalent income, `mean * (1 - a)`.
    pub ede: f64,
}

/// Atkinson index of strictly positive values with inequality aversion `epsilon >= 0`.
///
/// `epsilon == 1` uses the geometric mean. `Err` for empty input, a value `<= 0` or a negative
/// `epsilon`.
pub fn atkinson(y: &[f64], epsilon: f64) -> Result<Atkinson, String> {
    if y.is_empty() {
        return Err("y must not be empty".to_string());
    }
    if y.iter().any(|&v| v <= 0.0) {
        return Err("all values in y must be positive".to_string());
    }
    if epsilon < 0.0 {
        return Err("epsilon must be non-negative".to_string());
    }
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let a = if epsilon == 1.0 {
        let geometric = (y.iter().map(|v| v.ln()).sum::<f64>() / n).exp();
        1.0 - geometric / mean
    } else {
        let p = 1.0 - epsilon;
        let mean_p = y.iter().map(|v| v.powf(p)).sum::<f64>() / n;
        1.0 - mean_p.powf(1.0 / p) / mean
    };
    Ok(Atkinson {
        a,
        ede: mean * (1.0 - a),
    })
}

/// Schutz measures.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Schutz {
    /// Maximum distance between the line of equality and the Lorenz curve.
    pub distance: f64,
    /// Cumulative population share at which `distance` is first reached.
    pub intersection_point: f64,
    /// Original Schutz coefficient.
    pub coefficient: f64,
}

/// Schutz distance, intersection point and coefficient.
///
/// With the values sorted ascending, `distance` is the largest `cum_population - cum_share`,
/// `intersection_point` the cumulative population share of its first occurrence, and
/// `coefficient` the sum of `10 * (share / population_share - 1)` over the positive terms (the
/// factor 10 is the Python package's). `Err` for empty input or values that sum to zero (the
/// Python class raises `IndexError` there).
pub fn schutz(y: &[f64]) -> Result<Schutz, String> {
    let total: f64 = y.iter().sum();
    if y.is_empty() || total == 0.0 {
        return Err("y must not be empty or sum to zero".to_string());
    }
    let mut sorted = y.to_vec();
    sorted.sort_by(f64::total_cmp);
    let upct = 1.0 / sorted.len() as f64;
    let (mut ucpct, mut ycpct) = (0.0, 0.0);
    let mut best = Schutz {
        distance: f64::NEG_INFINITY,
        intersection_point: f64::NAN,
        coefficient: 0.0,
    };
    for v in sorted {
        let ypct = v / total;
        ucpct += upct;
        ycpct += ypct;
        let distance = ucpct - ycpct;
        if distance > best.distance {
            best.distance = distance;
            best.intersection_point = ucpct;
        }
        let coefficient = 10.0 * (ypct / upct - 1.0);
        if coefficient > 0.0 {
            best.coefficient += coefficient;
        }
    }
    Ok(best)
}

/// Lorenz curve: cumulative population share and cumulative value share, `n + 1` points each.
///
/// The population axis is `numpy.linspace(0, 1, n + 1)`. The value share is `NaN` when the values
/// sum to zero.
pub fn lorenz_curve(y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = y.len();
    let mut sorted = y.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut share = Vec::with_capacity(n + 1);
    share.push(0.0);
    let mut run = 0.0;
    for v in sorted {
        run += v;
        share.push(run);
    }
    let total = share[n];
    for v in &mut share {
        *v /= total;
    }
    let step = 1.0 / n as f64;
    let mut pop: Vec<f64> = (0..=n).map(|i| i as f64 * step).collect();
    pop[n] = 1.0;
    (pop, share)
}

/// `numpy.interp(x, xp, fp)` for increasing `xp` and `x` inside its range.
fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    let last = xp.len() - 1;
    let j = xp.partition_point(|&v| v <= x).saturating_sub(1);
    if j >= last || xp[j] == x {
        return fp[j];
    }
    let slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j]);
    slope * (x - xp[j]) + fp[j]
}

/// Median as `numpy.median`: the mean of the two middle values for an even count.
fn median(y: &[f64]) -> f64 {
    let mut sorted = y.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// Wolfson bipolarization index.
///
/// `W = (2 D50 - G) mean / median` with `D50 = 0.5 - L(0.5)` read off the Lorenz curve by linear
/// interpolation and `G` the Gini coefficient. `Err` for empty input.
pub fn wolfson(y: &[f64]) -> Result<f64, String> {
    if y.is_empty() {
        return Err("y must not be empty".to_string());
    }
    let (pop, share) = lorenz_curve(y);
    let d50 = 0.5 - interp(0.5, &pop, &share);
    let ratio = y.iter().sum::<f64>() / y.len() as f64 / median(y);
    Ok((2.0 * d50 - gini(y)) * ratio)
}
