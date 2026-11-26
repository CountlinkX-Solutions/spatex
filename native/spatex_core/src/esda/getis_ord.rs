//! Global Getis-Ord General G (`esda.G`).
//!
//! esda forces binary weights (`w.transform("B")`) with no user option; so does this module,
//! whatever transformation the graph carries. Intended for non-negative variables, which this
//! implementation enforces (esda does not).

use crate::esda::moran::{check_finite, check_graph, check_permutations, lag_dot};
use crate::graph::{Csr, Transform};
use crate::perm::{self, norm_cdf, PermTest};

/// General G and its normality inference.
#[derive(Clone, Copy, Debug)]
pub struct GAnalytic {
    /// Number of observations (islands included).
    pub n: usize,
    /// General G.
    pub g: f64,
    /// Expectation `S0 / (n (n - 1))`.
    pub eg: f64,
    /// Variance (NaN for `n < 4`).
    pub vg: f64,
    /// `sqrt(vg)`.
    pub se_g: f64,
    /// `(G - EG) / sqrt(VG)`; NaN when `VG` is not positive.
    pub z_norm: f64,
    /// `1 - Phi(|z_norm|)`, as esda computes it.
    pub p_norm: f64,
    /// `(sum y)^2 - sum y^2`.
    pub den_sum: f64,
}

/// Observed-statistic context: `G = sum_i y_i (W y)_i / den_sum` on the binarized graph.
pub struct GStat {
    g: Csr,
    /// `(sum y)^2 - sum y^2`.
    pub den_sum: f64,
}

impl GStat {
    /// Validates `y` (finite, non-negative, not degenerate) and binarizes the graph.
    pub fn new(y: &[f64], graph: &Csr) -> Result<Self, String> {
        check_graph(y.len(), graph)?;
        check_finite(y, "y")?;
        if let Some(i) = y.iter().position(|&v| v < 0.0) {
            return Err(format!(
                "y must be non-negative for Getis-Ord G (index {i})"
            ));
        }
        let g = graph.transform(Transform::B);
        if g.s0() == 0.0 {
            return Err("the graph has no weights (every node is an island)".into());
        }
        let sum: f64 = y.iter().sum();
        let den_sum = sum * sum - y.iter().map(|v| v * v).sum::<f64>();
        if den_sum == 0.0 {
            return Err("Getis-Ord G is undefined: fewer than two non-zero values in y".into());
        }
        Ok(GStat { g, den_sum })
    }

    /// G of the (possibly permuted) values `y`.
    pub fn statistic(&self, y: &[f64]) -> f64 {
        lag_dot(&self.g, y, y) / self.den_sum
    }
}

/// General G with analytic inference.
pub fn general_g(y: &[f64], graph: &Csr) -> Result<GAnalytic, String> {
    let ctx = GStat::new(y, graph)?;
    let g_stat = ctx.statistic(y);
    let n = y.len() as f64;
    let n2 = n * n;
    let (s0, s1, s2) = (ctx.g.s0(), ctx.g.s1(), ctx.g.s2());
    let s02 = s0 * s0;
    let eg = s0 / (n * (n - 1.0));

    let vg = if y.len() < 4 {
        f64::NAN
    } else {
        let b0 = (n2 - 3.0 * n + 3.0) * s1 - n * s2 + 3.0 * s02;
        let b1 = -((n2 - n) * s1 - 2.0 * n * s2 + 6.0 * s02);
        let b2 = -(2.0 * n * s1 - (n + 3.0) * s2 + 6.0 * s02);
        let b3 = 4.0 * (n - 1.0) * s1 - 2.0 * (n + 1.0) * s2 + 8.0 * s02;
        let b4 = s1 - s2 + s02;
        let pow_sum = |p: i32| y.iter().map(|v| v.powi(p)).sum::<f64>();
        let (m1, m2, m3, m4) = (pow_sum(1), pow_sum(2), pow_sum(3), pow_sum(4));
        let num = b0 * (m2 * m2) + b1 * m4 + b2 * (m1 * m1) * m2 + b3 * m1 * m3 + b4 * m1.powi(4);
        let den = ((m1 * m1 - m2) * (m1 * m1 - m2)) * n * (n - 1.0) * (n - 2.0) * (n - 3.0);
        let eg2 = num / den;
        eg2 - eg * eg
    };
    // A variance at or below rounding noise (complete graphs give about -2e-16) has no
    // meaningful square root; esda yields NaN there as well.
    let degenerate = vg.is_nan() || vg <= 4.0 * f64::EPSILON * (eg * eg).max(f64::MIN_POSITIVE);
    let (se_g, z_norm) = if degenerate {
        (f64::NAN, f64::NAN)
    } else {
        (vg.sqrt(), (g_stat - eg) / vg.sqrt())
    };
    Ok(GAnalytic {
        n: y.len(),
        g: g_stat,
        eg,
        vg,
        se_g,
        z_norm,
        p_norm: 1.0 - norm_cdf(z_norm.abs()),
        den_sum: ctx.den_sum,
    })
}

/// Permutation test of G: `permutations` shuffles of `y`. G reports `p_z_sim` as
/// `1 - Phi(|z_sim|)`.
pub fn general_g_permutations(
    y: &[f64],
    graph: &Csr,
    permutations: usize,
    seed: u64,
) -> Result<PermTest, String> {
    check_permutations(permutations)?;
    let ctx = GStat::new(y, graph)?;
    let observed = ctx.statistic(y);
    let sim = perm::simulate(y, permutations, seed, |v| ctx.statistic(v));
    let mut t = PermTest::new(observed, sim);
    t.summary.p_z_sim = 1.0 - norm_cdf(t.summary.z_sim.abs());
    Ok(t)
}

#[cfg(test)]
mod tests;
