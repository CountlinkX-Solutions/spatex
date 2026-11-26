//! Classic Markov chain estimation and the homogeneity tests (`giddy.markov`).

use super::special::chi2_sf;
use crate::limits;

/// Transition counts of an `n x t` panel of class indices in `0..k` (row-major).
/// More than [`limits::MAX_CLASSES`] classes is an error raised before the `k x k` table exists.
pub fn count_transitions(ids: &[u32], n: usize, t: usize, k: usize) -> Result<Vec<f64>, String> {
    check_classes(k)?;
    if ids.len() != n * t {
        return Err(format!("expected {n} x {t} class ids, got {}", ids.len()));
    }
    if let Some(bad) = ids.iter().find(|&&c| c as usize >= k) {
        return Err(format!("class index {bad} is out of range for {k} classes"));
    }
    let mut counts = vec![0.0; k * k];
    for i in 0..n {
        for s in 0..t.saturating_sub(1) {
            let a = ids[i * t + s] as usize;
            let b = ids[i * t + s + 1] as usize;
            counts[a * k + b] += 1.0;
        }
    }
    Ok(counts)
}

/// Fails when `k` classes exceed [`limits::MAX_CLASSES`].
pub fn check_classes(k: usize) -> Result<(), String> {
    if k > limits::MAX_CLASSES {
        Err(format!(
            "{k} classes exceed the limit of {} classes; use fewer classes (or bin the data)",
            limits::MAX_CLASSES
        ))
    } else {
        Ok(())
    }
}

/// Row-normalizes counts; all-zero rows stay zero (`diag(1 / (rs + (rs == 0))) @ T`).
pub fn row_normalize(counts: &[f64], k: usize) -> Vec<f64> {
    let mut p = counts.to_vec();
    for i in 0..k {
        let rs: f64 = counts[i * k..(i + 1) * k].iter().sum();
        let inv = 1.0 / (rs + f64::from(u8::from(rs == 0.0)));
        for v in &mut p[i * k..(i + 1) * k] {
            *v *= inv;
        }
    }
    p
}

fn row_sums(m: &[f64], k: usize) -> Vec<f64> {
    (0..k).map(|i| m[i * k..(i + 1) * k].iter().sum()).collect()
}

/// Chi-squared test of the counts `t1` against the probabilities implied by `t2`
/// (`giddy.markov.chi2`). Returns `(statistic, p-value, degrees of freedom)`.
pub fn chi2_test(t1: &[f64], t2: &[f64], k: usize) -> (f64, f64, f64) {
    let rs1 = row_sums(t1, k);
    let rs2 = row_sums(t2, k);
    let dof1 = rs1.iter().filter(|&&v| v > 0.0).count() as f64;
    let dof2 = rs2.iter().filter(|&&v| v > 0.0).count() as f64;
    let dof = (dof1 - 1.0) * (dof2 - 1.0);
    let mut stat = 0.0;
    for i in 0..k {
        let denom = rs2[i] + f64::from(u8::from(rs2[i] == 0.0));
        let inv = 1.0 / denom;
        for j in 0..k {
            let e = rs1[i] * (inv * t2[i * k + j]);
            let num = (t1[i * k + j] - e) * (t1[i * k + j] - e);
            let e = e + f64::from(u8::from(e == 0.0));
            stat += num / e;
        }
    }
    (stat, chi2_sf(stat, dof), dof)
}

/// Homogeneity test of several transition count matrices (Bickenbach-Bode).
#[derive(Clone, Debug)]
pub struct Homogeneity {
    /// Pearson-type statistic.
    pub q: f64,
    /// p-value of `q`.
    pub q_p: f64,
    /// Likelihood-ratio statistic.
    pub lr: f64,
    /// p-value of `lr`.
    pub lr_p: f64,
    /// Degrees of freedom (`sum_i (b_i - 1)(A_i - 1)`).
    pub dof: i64,
}

/// Homogeneity of `m` stacked `k x k` count matrices (`giddy.markov.homogeneity`).
pub fn homogeneity(matrices: &[Vec<f64>], k: usize) -> Homogeneity {
    let mut total = vec![0.0; k * k];
    for m in matrices {
        for (t, v) in total.iter_mut().zip(m) {
            *t += v;
        }
    }
    let n_i = row_sums(&total, k);
    let a_i: Vec<i64> = (0..k)
        .map(|i| {
            total[i * k..(i + 1) * k]
                .iter()
                .filter(|&&v| v > 0.0)
                .count() as i64
        })
        .collect();
    let mut p0 = vec![0.0; k * k];
    for i in 0..k {
        let inv = 1.0 / (n_i[i] + f64::from(u8::from(n_i[i] == 0.0)));
        for j in 0..k {
            p0[i * k + j] = inv * total[i * k + j];
        }
    }
    let mut b_i = vec![0_i64; k];
    let (mut q, mut lr) = (0.0, 0.0);
    for m in matrices {
        let nim = row_sums(m, k);
        for i in 0..k {
            if nim[i] > 0.0 {
                b_i[i] += 1;
            }
            let inv = 1.0 / (nim[i] + f64::from(u8::from(nim[i] == 0.0)));
            for j in 0..k {
                let pijm = inv * m[i * k + j];
                let pij = p0[i * k + j];
                let den = pij + f64::from(u8::from(pij == 0.0));
                q += nim[i] * ((pijm - pij) * (pijm - pij) / den);
                if m[i * k + j] > 0.0 && pij > 0.0 {
                    lr += m[i * k + j] * (pijm / pij).ln();
                }
            }
        }
    }
    let lr = lr * 2.0;
    let dof: i64 = b_i.iter().zip(&a_i).map(|(b, a)| (b - 1) * (a - 1)).sum();
    Homogeneity {
        q,
        q_p: chi2_sf(q, dof as f64),
        lr,
        lr_p: chi2_sf(lr, dof as f64),
        dof,
    }
}

/// Kullback information test of Markov homogeneity across strata.
#[derive(Clone, Copy, Debug)]
pub struct Kullback {
    /// Conditional homogeneity statistic.
    pub statistic: f64,
    /// Degrees of freedom, `r (s - 1) (r - 1)`.
    pub dof: i64,
    /// `1 - chi2.cdf(statistic, dof)` (`NaN` for zero degrees of freedom).
    pub p_value: f64,
}

/// `2 sum v ln v` with `0 ln 0 = 0`, as giddy's `v * log(v + (v == 0))`.
fn two_x_ln_x<I: Iterator<Item = f64>>(values: I) -> f64 {
    2.0 * values
        .map(|v| v * (v + f64::from(u8::from(v == 0.0))).ln())
        .sum::<f64>()
}

/// Kullback's conditional homogeneity test of `s` stacked `r x r` transition count matrices
/// (`giddy.markov.kullback`).
///
/// With `f_sij` the transitions from class `i` to `j` in stratum `s`, the statistic is
/// `2 [sum f_sij ln f_sij - sum f_si. ln f_si. - sum f_.ij ln f_.ij + sum f_.i. ln f_.i.]`.
/// giddy sums the second term over the wrong axis (the column totals `f_s.j`), which agrees only
/// when every stratum has equal row and column totals; `giddy_totals` reproduces that. `Err` for
/// a wrong length, no strata, or a negative or non-finite count.
pub fn kullback(f: &[f64], s: usize, r: usize, giddy_totals: bool) -> Result<Kullback, String> {
    if s == 0 || r == 0 || f.len() != s * r * r {
        return Err(format!("expected {s} stacked {r} x {r} count matrices"));
    }
    if f.iter().any(|&v| !(v.is_finite() && v >= 0.0)) {
        return Err("counts must be finite and non-negative".to_string());
    }
    let rr = r * r;
    let mut pooled = vec![0.0; rr];
    for stratum in f.chunks(rr) {
        for (p, v) in pooled.iter_mut().zip(stratum) {
            *p += v;
        }
    }
    let pooled_rows: Vec<f64> = pooled.chunks(r).map(|row| row.iter().sum()).collect();
    let mut totals = Vec::with_capacity(s * r);
    for stratum in f.chunks(rr) {
        for a in 0..r {
            totals.push(if giddy_totals {
                (0..r).map(|i| stratum[i * r + a]).sum::<f64>()
            } else {
                stratum[a * r..(a + 1) * r].iter().sum::<f64>()
            });
        }
    }
    let statistic = two_x_ln_x(f.iter().copied())
        - two_x_ln_x(totals.into_iter())
        - two_x_ln_x(pooled.into_iter())
        + two_x_ln_x(pooled_rows.into_iter());
    let dof = (r * (s - 1) * (r - 1)) as i64;
    Ok(Kullback {
        statistic,
        dof,
        p_value: chi2_sf(statistic, dof as f64),
    })
}
