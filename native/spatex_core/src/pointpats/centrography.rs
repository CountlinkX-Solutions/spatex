//! Centrography (`pointpats/centrography.py`): centers, spread, standard deviational ellipse.

use super::Pt;

fn require(p: &[Pt]) -> Result<(), String> {
    if p.is_empty() {
        Err("points must not be empty".into())
    } else {
        Ok(())
    }
}

/// Arithmetic mean of the coordinates.
pub fn mean_center(p: &[Pt]) -> Result<Pt, String> {
    require(p)?;
    let n = p.len() as f64;
    let (sx, sy) = p.iter().fold((0.0, 0.0), |a, q| (a.0 + q[0], a.1 + q[1]));
    Ok([sx / n, sy / n])
}

/// Mean weighted by `w / sum(w)`.
pub fn weighted_mean_center(p: &[Pt], w: &[f64]) -> Result<Pt, String> {
    require(p)?;
    if w.len() != p.len() {
        return Err(format!(
            "weights length {} does not match {} points",
            w.len(),
            p.len()
        ));
    }
    let total: f64 = w.iter().sum();
    if total == 0.0 || !total.is_finite() {
        return Err("weights must have a non-zero finite sum".into());
    }
    let (mut x, mut y) = (0.0, 0.0);
    for (q, wi) in p.iter().zip(w) {
        let wn = wi / total;
        x += wn * q[0];
        y += wn * q[1];
    }
    Ok([x, y])
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// Coordinate-wise median (not unique for even `n`; the mean of the two middle values).
pub fn manhattan_median(p: &[Pt]) -> Result<Pt, String> {
    require(p)?;
    Ok([
        median(p.iter().map(|q| q[0]).collect()),
        median(p.iter().map(|q| q[1]).collect()),
    ])
}

/// Standard distance in pointpats' raw-moment form `sqrt(E[x^2] - mx^2 + E[y^2] - my^2)`
/// (loses digits for large coordinate offsets, as pointpats does).
pub fn std_distance(p: &[Pt]) -> Result<f64, String> {
    let m = mean_center(p)?;
    let n = p.len() as f64;
    let (s2x, s2y) = p
        .iter()
        .fold((0.0, 0.0), |a, q| (a.0 + q[0] * q[0], a.1 + q[1] * q[1]));
    Ok((s2x / n - m[0] * m[0] + (s2y / n - m[1] * m[1])).sqrt())
}

/// Sum of Euclidean distances from `c` to every point.
pub fn dtot(c: Pt, p: &[Pt]) -> f64 {
    p.iter()
        .map(|q| ((q[0] - c[0]) * (q[0] - c[0]) + (q[1] - c[1]) * (q[1] - c[1])).sqrt())
        .sum()
}

/// Euclidean (geometric) median by Weiszfeld iteration with the Vardi-Zhang modification, from
/// the mean center. pointpats runs scipy's BFGS with `gtol = 1e-5`, which stops short of the
/// optimum (recorded as a deviation).
pub fn euclidean_median(p: &[Pt]) -> Result<Pt, String> {
    let mut x = mean_center(p)?;
    for _ in 0..10_000 {
        let (mut nx, mut ny, mut den) = (0.0, 0.0, 0.0);
        let (mut rx, mut ry, mut eta) = (0.0, 0.0, 0.0);
        for q in p {
            let d = ((q[0] - x[0]).powi(2) + (q[1] - x[1]).powi(2)).sqrt();
            if d == 0.0 {
                eta += 1.0;
            } else {
                nx += q[0] / d;
                ny += q[1] / d;
                den += 1.0 / d;
                rx += (q[0] - x[0]) / d;
                ry += (q[1] - x[1]) / d;
            }
        }
        if den == 0.0 {
            return Ok(x);
        }
        let t = [nx / den, ny / den];
        let next = if eta == 0.0 {
            t
        } else {
            let r = rx.hypot(ry);
            if r <= eta {
                // the coincident data point is optimal
                return Ok(x);
            }
            let g = eta / r;
            [(1.0 - g) * t[0] + g * x[0], (1.0 - g) * t[1] + g * x[1]]
        };
        let step = (next[0] - x[0]).abs().max((next[1] - x[1]).abs());
        let scale = x[0].abs().max(x[1].abs()).max(1.0);
        x = next;
        if step < 1e-12 * scale {
            break;
        }
    }
    Ok(x)
}

/// Plain per-axis extent `[min_x, min_y, max_x, max_y]`.
pub fn mbr(p: &[Pt]) -> Result<[f64; 4], String> {
    require(p)?;
    Ok(super::bounds(p))
}

/// Ellipse axis correction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EllipseMethod {
    /// `sqrt(2) sqrt(n) / sqrt(n - 2)`.
    Crimestat,
    /// `sqrt(2)` (if `crimestat_corr`) and `sqrt(n) / sqrt(n - 2)` (if `degfreed_corr`).
    Yuill {
        /// Apply the `sqrt(2)` factor.
        crimestat_corr: bool,
        /// Apply the degrees-of-freedom factor.
        degfreed_corr: bool,
    },
}

/// Standard deviational ellipse `(major, minor, angle)`; `angle` is the direction of the major
/// axis in radians counter-clockwise from +x. The correction uses `n = len(p)` even when
/// weighted, as pointpats does.
pub fn ellipse(
    p: &[Pt],
    weights: Option<&[f64]>,
    method: EllipseMethod,
) -> Result<(f64, f64, f64), String> {
    let n = p.len();
    if n < 3 {
        return Err("ellipse needs at least 3 points".into());
    }
    let ones;
    let w = match weights {
        Some(w) if w.len() == n => w,
        Some(_) => return Err("weights must have the same length as points".into()),
        None => {
            ones = vec![1.0; n];
            &ones
        }
    };
    let sumw: f64 = w.iter().sum();
    let meanx = p.iter().zip(w).map(|(q, wi)| q[0] * wi).sum::<f64>() / sumw;
    let meany = p.iter().zip(w).map(|(q, wi)| q[1] * wi).sum::<f64>() / sumw;
    let (mut xyw, mut x2w, mut y2w) = (0.0, 0.0, 0.0);
    for (q, wi) in p.iter().zip(w) {
        let (xm, ym) = (q[0] - meanx, q[1] - meany);
        xyw += xm * ym * wi;
        x2w += xm * xm * wi;
        y2w += ym * ym * wi;
    }
    let den = 2.0 * xyw;
    let left = x2w - y2w;
    let right = (left * left + 4.0 * xyw * xyw).sqrt();
    let (theta1, theta2) = if den == 0.0 {
        (0.0, std::f64::consts::FRAC_PI_2)
    } else {
        (
            (-(left + right) / den).atan(),
            (-(left - right) / den).atan(),
        )
    };
    let term = |theta: f64| -> f64 {
        let (s, c) = theta.sin_cos();
        p.iter()
            .zip(w)
            .map(|(q, wi)| {
                let (xm, ym) = (q[0] - meanx, q[1] - meany);
                wi * (ym * c - xm * s).powi(2)
            })
            .sum()
    };
    let mut sx = (term(theta1) / sumw).sqrt();
    let mut sy = (term(theta2) / sumw).sqrt();
    let nf = n as f64;
    match method {
        EllipseMethod::Crimestat => {
            let corr = (2f64.sqrt() * nf.sqrt()) / (nf - 2.0).sqrt();
            sx *= corr;
            sy *= corr;
        }
        EllipseMethod::Yuill {
            crimestat_corr,
            degfreed_corr,
        } => {
            if crimestat_corr {
                sx *= 2f64.sqrt();
                sy *= 2f64.sqrt();
            }
            if degfreed_corr {
                sx *= nf.sqrt() / (nf - 2.0).sqrt();
                sy *= nf.sqrt() / (nf - 2.0).sqrt();
            }
        }
    }
    Ok(if sy > sx {
        (sy, sx, theta1)
    } else {
        (sx, sy, theta2)
    })
}
