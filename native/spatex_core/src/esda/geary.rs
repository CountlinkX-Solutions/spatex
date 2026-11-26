//! Global Geary's C (`esda.Geary`).
//!
//! The graph passed in must already carry the requested transformation (PySAL's default is
//! row-standardisation, `"r"`); `S0`, `S1`, `S2` are taken from it as is. Inference is
//! one-tailed, as in esda: the tail follows the sign of `C - 1`.

use crate::esda::moran::{check_finite, check_graph, check_permutations, is_constant};
use crate::graph::Csr;
use crate::perm::{self, norm_cdf, norm_sf, PermTest};

/// Geary's C and its normality / randomization inference.
#[derive(Clone, Copy, Debug)]
pub struct GearyAnalytic {
    /// Number of observations (islands included).
    pub n: usize,
    /// Geary's C.
    pub c: f64,
    /// Expectation, always 1.
    pub ec: f64,
    /// Variance under normality.
    pub vc_norm: f64,
    /// `sqrt(vc_norm)`.
    pub se_c_norm: f64,
    /// `(C - 1) / se_c_norm`.
    pub z_norm: f64,
    /// One-tailed normal p-value.
    pub p_norm: f64,
    /// Variance under randomization (NaN for `n < 4`).
    pub vc_rand: f64,
    /// `sqrt(vc_rand)`.
    pub se_c_rand: f64,
    /// `(C - 1) / se_c_rand`.
    pub z_rand: f64,
    /// One-tailed randomization p-value.
    pub p_rand: f64,
}

/// Observed-statistic context: the constants of `C = (n-1) sum_ij w_ij (y_i - y_j)^2 / den`.
/// Observed and permuted statistics share [`GearyStat::statistic`].
pub struct GearyStat<'a> {
    g: &'a Csr,
    n: f64,
    s0: f64,
    den: f64,
    /// `y - mean(y)`.
    centred: Vec<f64>,
}

impl<'a> GearyStat<'a> {
    /// Validates `y` against the graph.
    pub fn new(y: &[f64], g: &'a Csr) -> Result<Self, String> {
        check_graph(y.len(), g)?;
        check_finite(y, "y")?;
        if is_constant(y) {
            return Err("y is constant: Geary's C is undefined".into());
        }
        let s0 = g.s0();
        if s0 == 0.0 {
            return Err("the graph has no weights (every node is an island)".into());
        }
        let n = y.len() as f64;
        let mean = y.iter().sum::<f64>() / n;
        let centred: Vec<f64> = y.iter().map(|v| v - mean).collect();
        let yss: f64 = centred.iter().map(|v| v * v).sum();
        Ok(GearyStat {
            g,
            n,
            s0,
            den: yss * s0 * 2.0,
            centred,
        })
    }

    /// Geary's C of the (possibly permuted) raw values `y`.
    pub fn statistic(&self, y: &[f64]) -> f64 {
        let mut num = 0.0;
        for (i, &yi) in y.iter().enumerate() {
            for k in self.g.offsets[i]..self.g.offsets[i + 1] {
                let d = yi - y[self.g.neighbors[k] as usize];
                num += self.g.weights[k] * d * d;
            }
        }
        (self.n - 1.0) * num / self.den
    }
}

fn one_tailed(de: f64, z: f64) -> f64 {
    if de > 0.0 {
        norm_sf(z)
    } else {
        norm_cdf(z)
    }
}

/// Geary's C with analytic inference.
pub fn geary(y: &[f64], g: &Csr) -> Result<GearyAnalytic, String> {
    let ctx = GearyStat::new(y, g)?;
    let c = ctx.statistic(y);
    let n = ctx.n;
    let (s0, s1, s2) = (ctx.s0, g.s1(), g.s2());
    let s02 = s0 * s0;
    let n2 = n * n;

    let vc_norm = (1.0 / (2.0 * (n + 1.0) * s02)) * ((2.0 * s1 + s2) * (n - 1.0) - 4.0 * s02);
    let vc_rand = if y.len() < 4 {
        f64::NAN
    } else {
        let m4 = ctx.centred.iter().map(|v| v.powi(4)).sum::<f64>() / n;
        let m2 = ctx.centred.iter().map(|v| v * v).sum::<f64>() / n;
        let k = m4 / (m2 * m2);
        let a = (n - 1.0) * s1 * (n2 - 3.0 * n + 3.0 - (n - 1.0) * k);
        let b = 0.25 * ((n - 1.0) * s2 * (n2 + 3.0 * n - 6.0 - (n2 - n + 2.0) * k));
        let cc = s02 * (n2 - 3.0 - (n - 1.0) * (n - 1.0) * k);
        (a - b + cc) / (n * (n - 2.0) * (n - 3.0) * s02)
    };
    let (se_c_norm, se_c_rand) = (vc_norm.sqrt(), vc_rand.sqrt());
    let de = c - 1.0;
    let (z_norm, z_rand) = (de / se_c_norm, de / se_c_rand);
    Ok(GearyAnalytic {
        n: y.len(),
        c,
        ec: 1.0,
        vc_norm,
        se_c_norm,
        z_norm,
        p_norm: one_tailed(de, z_norm),
        vc_rand,
        se_c_rand,
        z_rand,
        p_rand: one_tailed(de, z_rand),
    })
}

/// Permutation test of Geary's C: `permutations` shuffles of the raw values. Geary reports
/// `p_z_sim` as the one-tailed survival of `|z_sim|`.
pub fn geary_permutations(
    y: &[f64],
    g: &Csr,
    permutations: usize,
    seed: u64,
) -> Result<PermTest, String> {
    check_permutations(permutations)?;
    let ctx = GearyStat::new(y, g)?;
    let observed = ctx.statistic(y);
    let sim = perm::simulate(y, permutations, seed, |v| ctx.statistic(v));
    let mut t = PermTest::new(observed, sim);
    t.summary.p_z_sim = norm_sf(t.summary.z_sim.abs());
    Ok(t)
}

#[cfg(test)]
mod tests;
