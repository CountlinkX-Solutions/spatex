//! Global Moran's I: univariate, bivariate and for rates (`esda.moran`, `esda.smoothing`).
//!
//! The graph passed in must already carry the requested transformation (PySAL's default is
//! row-standardisation, `"r"`); `S0`, `S1`, `S2` are taken from it as is.

use crate::graph::Csr;
use crate::perm::{self, norm_cdf, norm_sf, PermTest};

/// Analytic Moran's I and its normality / randomization inference.
#[derive(Clone, Copy, Debug)]
pub struct MoranAnalytic {
    /// Number of observations (islands included).
    pub n: usize,
    /// Moran's I.
    pub i: f64,
    /// Expectation `-1 / (n - 1)`.
    pub ei: f64,
    /// Variance under normality.
    pub vi_norm: f64,
    /// `sqrt(vi_norm)`.
    pub se_i_norm: f64,
    /// `(I - EI) / se_i_norm`.
    pub z_norm: f64,
    /// Normal p-value (doubled when two-tailed).
    pub p_norm: f64,
    /// Variance under randomization (NaN for `n < 4`).
    pub vi_rand: f64,
    /// `sqrt(vi_rand)`.
    pub se_i_rand: f64,
    /// `(I - EI) / se_i_rand`.
    pub z_rand: f64,
    /// Randomization p-value (doubled when two-tailed).
    pub p_rand: f64,
}

pub(crate) fn check_finite(values: &[f64], name: &str) -> Result<(), String> {
    match values.iter().position(|v| !v.is_finite()) {
        Some(i) => Err(format!("{name}: non-finite value at index {i}")),
        None => Ok(()),
    }
}

pub(crate) fn check_graph(len: usize, g: &Csr) -> Result<(), String> {
    if len != g.n {
        return Err(format!(
            "y has length {len} but the graph has {} nodes (length mismatch)",
            g.n
        ));
    }
    if len < 2 {
        return Err("at least 2 observations are required".into());
    }
    Ok(())
}

pub(crate) fn is_constant(v: &[f64]) -> bool {
    v.iter().all(|x| *x == v[0])
}

/// `sum_i left_i * (W right)_i`, accumulated in row order.
pub(crate) fn lag_dot(g: &Csr, left: &[f64], right: &[f64]) -> f64 {
    let mut acc = 0.0;
    for (i, &l) in left.iter().enumerate() {
        let mut lag = 0.0;
        for k in g.offsets[i]..g.offsets[i + 1] {
            lag += g.weights[k] * right[g.neighbors[k] as usize];
        }
        acc += l * lag;
    }
    acc
}

/// Observed-statistic context for univariate Moran: the centred values and the constants of
/// `I = n / S0 * sum_i z_i (Wz)_i / sum z^2`. Observed and permuted statistics share
/// [`MoranStat::statistic`], so an identity permutation reproduces `I` bit for bit.
pub struct MoranStat<'a> {
    g: &'a Csr,
    /// `y - mean(y)`.
    pub z: Vec<f64>,
    n: f64,
    s0: f64,
    z2ss: f64,
}

impl<'a> MoranStat<'a> {
    /// Validates `y` against the graph and centres it.
    pub fn new(y: &[f64], g: &'a Csr) -> Result<Self, String> {
        check_graph(y.len(), g)?;
        check_finite(y, "y")?;
        if is_constant(y) {
            return Err("y is constant: Moran's I is undefined".into());
        }
        let s0 = g.s0();
        if s0 == 0.0 {
            return Err("the graph has no weights (every node is an island)".into());
        }
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let z: Vec<f64> = y.iter().map(|v| v - mean).collect();
        let z2ss = z.iter().map(|v| v * v).sum();
        Ok(MoranStat { g, z, n, s0, z2ss })
    }

    /// Moran's I of the (possibly permuted) centred values `z`.
    pub fn statistic(&self, z: &[f64]) -> f64 {
        let inum = lag_dot(self.g, z, z);
        self.n / self.s0 * inum / self.z2ss
    }
}

fn tail_p(z_norm: f64, z: f64, two_tailed: bool) -> f64 {
    // The tail is chosen by the sign of z_norm for both inferences, as in esda.
    let p = if z_norm > 0.0 {
        norm_sf(z)
    } else {
        norm_cdf(z)
    };
    if two_tailed {
        2.0 * p
    } else {
        p
    }
}

/// Moran's I with analytic inference. `two_tailed` doubles `p_norm` and `p_rand`.
pub fn moran(y: &[f64], g: &Csr, two_tailed: bool) -> Result<MoranAnalytic, String> {
    let ctx = MoranStat::new(y, g)?;
    let i = ctx.statistic(&ctx.z);
    let n = ctx.n;
    let (s0, s1, s2) = (ctx.s0, g.s1(), g.s2());
    let s02 = s0 * s0;
    let ei = -1.0 / (n - 1.0);

    let vi_norm = (n * n * s1 - n * s2 + 3.0 * s02) / ((n - 1.0) * (n + 1.0) * s02)
        - (1.0 / (n - 1.0)) * (1.0 / (n - 1.0));
    let se_i_norm = vi_norm.sqrt();

    let vi_rand = if y.len() < 4 {
        f64::NAN
    } else {
        let m4 = ctx.z.iter().map(|v| v.powi(4)).sum::<f64>() / n;
        let m2 = ctx.z2ss / n;
        let k = m4 / (m2 * m2);
        let a = n * ((n * n - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02);
        let b = k * ((n * n - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
        (a - b) / ((n - 1.0) * (n - 2.0) * (n - 3.0) * s02) - ei * ei
    };
    let se_i_rand = vi_rand.sqrt();

    let z_norm = (i - ei) / se_i_norm;
    let z_rand = (i - ei) / se_i_rand;
    Ok(MoranAnalytic {
        n: y.len(),
        i,
        ei,
        vi_norm,
        se_i_norm,
        z_norm,
        p_norm: tail_p(z_norm, z_norm, two_tailed),
        vi_rand,
        se_i_rand,
        z_rand,
        p_rand: tail_p(z_norm, z_rand, two_tailed),
    })
}

pub(crate) fn check_permutations(permutations: usize) -> Result<(), String> {
    if permutations == 0 {
        Err("permutations must be positive".into())
    } else {
        Ok(())
    }
}

/// Permutation test of Moran's I: `permutations` shuffles of the centred values.
pub fn moran_permutations(
    y: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
) -> Result<PermTest, String> {
    check_permutations(permutations)?;
    let ctx = MoranStat::new(y, g)?;
    let observed = ctx.statistic(&ctx.z);
    let sim = perm::simulate(&ctx.z, permutations, seed, |z| ctx.statistic(z));
    Ok(PermTest::new(observed, sim))
}

/// Standardises with the sample standard deviation (ddof = 1), as `Moran_BV` does.
fn standardize_sample(v: &[f64], name: &str) -> Result<Vec<f64>, String> {
    check_finite(v, name)?;
    if is_constant(v) {
        return Err(format!("{name} is constant: bivariate Moran is undefined"));
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|a| (a - mean) * (a - mean)).sum::<f64>() / (n - 1.0);
    let sd = var.sqrt();
    Ok(v.iter().map(|a| (a - mean) / sd).collect())
}

fn bv_prepare(x: &[f64], y: &[f64], g: &Csr) -> Result<(Vec<f64>, Vec<f64>), String> {
    check_graph(y.len(), g)?;
    if x.len() != y.len() {
        return Err(format!(
            "x has length {} but y has length {} (length mismatch)",
            x.len(),
            y.len()
        ));
    }
    Ok((standardize_sample(x, "x")?, standardize_sample(y, "y")?))
}

/// Bivariate Moran's I: `sum_i zx_i (W zy)_i / (n - 1)` with sample-standardised variables.
pub fn moran_bv(x: &[f64], y: &[f64], g: &Csr) -> Result<f64, String> {
    let (zx, zy) = bv_prepare(x, y, g)?;
    Ok(lag_dot(g, &zx, &zy) / (x.len() as f64 - 1.0))
}

/// Permutation test of bivariate Moran's I: `y` is shuffled, `x` stays fixed.
pub fn moran_bv_permutations(
    x: &[f64],
    y: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
) -> Result<PermTest, String> {
    check_permutations(permutations)?;
    let (zx, zy) = bv_prepare(x, y, g)?;
    let den = x.len() as f64 - 1.0;
    let observed = lag_dot(g, &zx, &zy) / den;
    let sim = perm::simulate(&zy, permutations, seed, |z| lag_dot(g, &zx, z) / den);
    Ok(PermTest::new(observed, sim))
}

fn check_rate_inputs(events: &[f64], base: &[f64]) -> Result<(), String> {
    if events.len() != base.len() {
        return Err(format!(
            "events has length {} but base has length {} (length mismatch)",
            events.len(),
            base.len()
        ));
    }
    if events.is_empty() {
        return Err("events and base must not be empty".into());
    }
    check_finite(events, "events")?;
    check_finite(base, "base")?;
    if let Some(i) = events.iter().position(|&e| e < 0.0) {
        return Err(format!("events must be non-negative (index {i})"));
    }
    if let Some(i) = base.iter().position(|&b| b <= 0.0) {
        return Err(format!("base must be positive (index {i})"));
    }
    Ok(())
}

/// Rates `events / base`, empirical-Bayes standardised when `adjusted`.
pub fn rate(events: &[f64], base: &[f64], adjusted: bool) -> Result<Vec<f64>, String> {
    if adjusted {
        assuncao_rate(events, base)
    } else {
        check_rate_inputs(events, base)?;
        Ok(events.iter().zip(base).map(|(e, b)| e / b).collect())
    }
}

/// Empirical-Bayes standardised rates of Assuncao and Reis (1999), `esda.smoothing.assuncao_rate`.
pub fn assuncao_rate(events: &[f64], base: &[f64]) -> Result<Vec<f64>, String> {
    check_rate_inputs(events, base)?;
    let n = events.len() as f64;
    let sum_e: f64 = events.iter().sum();
    let sum_b: f64 = base.iter().sum();
    let rates: Vec<f64> = events.iter().zip(base).map(|(e, b)| e / b).collect();
    let ebi_b = sum_e / sum_b;
    let s2 = rates
        .iter()
        .zip(base)
        .map(|(p, b)| b * (p - ebi_b) * (p - ebi_b))
        .sum::<f64>()
        / sum_b;
    let ebi_a = s2 - ebi_b / (sum_b / n);
    Ok(rates
        .iter()
        .zip(base)
        .map(|(p, b)| {
            let mut v = ebi_a + ebi_b / b;
            if v < 0.0 {
                v = ebi_b / b;
            }
            (p - ebi_b) / v.sqrt()
        })
        .collect())
}

#[cfg(test)]
mod tests;
