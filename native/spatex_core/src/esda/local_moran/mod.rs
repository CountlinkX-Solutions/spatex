//! Local Moran's I (LISA) and its bivariate form (`esda.Moran_Local`, `esda.Moran_Local_BV`).
//!
//! The graph passed in must already carry the requested transformation (PySAL defaults to
//! row-standardisation). Self weights (diagonal entries) are part of `W` and enter both the
//! statistic and its analytic moments, exactly as in esda. Inference is the conditional
//! randomization of [`crate::perm::crand`].

use crate::graph::Csr;
use crate::perm::crand::{self, Crand, Settings};

/// Analytic moments of the univariate statistic (esda has none for the bivariate one).
#[derive(Clone, Debug)]
pub struct LocalMoments {
    /// Total randomization expectation `-wi / (n - 1)`.
    pub ei: Vec<f64>,
    /// Total randomization variance (Sokal et al. 1998, A3/A4*).
    pub vi: Vec<f64>,
    /// Conditional randomization expectation (Sokal et al. 1998, A7).
    pub eic: Vec<f64>,
    /// Conditional randomization variance (Sokal et al. 1998, A8).
    pub vic: Vec<f64>,
}

/// Observed local Moran statistics.
#[derive(Clone, Debug)]
pub struct LocalMoran {
    /// Number of sites.
    pub n: usize,
    /// Standardised values (`x` for the bivariate statistic), ddof 0.
    pub z: Vec<f64>,
    /// Standardised `y` of the bivariate statistic.
    pub zy: Option<Vec<f64>>,
    /// Spatial lag `W z` (`W zy` for the bivariate statistic).
    pub lag: Vec<f64>,
    /// `(n - 1) z_i lag_i / sum(z^2)`.
    pub is: Vec<f64>,
    /// Quadrant 1 HH, 2 LH, 3 LL, 4 HL (LH and LL swapped with `geoda_quads`).
    pub q: Vec<u8>,
    /// Analytic moments (univariate only).
    pub moments: Option<LocalMoments>,
}

fn standardize(v: &[f64], name: &str) -> Result<Vec<f64>, String> {
    if let Some(i) = v.iter().position(|x| !x.is_finite()) {
        return Err(format!("{name}: non-finite value at index {i}"));
    }
    if v.iter().all(|x| *x == v[0]) {
        return Err(format!("{name} is constant: local Moran's I is undefined"));
    }
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let sd = (v.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n).sqrt();
    Ok(v.iter().map(|x| (x - mean) / sd).collect())
}

fn check(len: usize, g: &Csr, name: &str) -> Result<(), String> {
    if len != g.n {
        return Err(format!(
            "{name} has length {len} but the graph has {} nodes (length mismatch)",
            g.n
        ));
    }
    if len < 3 {
        return Err("at least 3 observations are required".into());
    }
    Ok(())
}

fn quadrants(focal: &[f64], lag: &[f64], geoda_quads: bool) -> Vec<u8> {
    let (lh, ll) = if geoda_quads { (3, 2) } else { (2, 3) };
    focal
        .iter()
        .zip(lag)
        .map(|(&z, &l)| match (z > 0.0, l > 0.0) {
            (true, true) => 1,
            (false, true) => lh,
            (false, false) => ll,
            (true, false) => 4,
        })
        .collect()
}

fn moments(g: &Csr, z: &[f64]) -> LocalMoments {
    let n = z.len() as f64;
    let m2 = z.iter().map(|v| v * v).sum::<f64>() / n;
    let m4 = z.iter().map(|v| v.powi(4)).sum::<f64>() / n;
    let b2 = m4 / (m2 * m2);
    let n1 = n - 1.0;
    let mut out = LocalMoments {
        ei: Vec::with_capacity(z.len()),
        vi: Vec::with_capacity(z.len()),
        eic: Vec::with_capacity(z.len()),
        vic: Vec::with_capacity(z.len()),
    };
    for (i, &zi) in z.iter().enumerate() {
        let row = g.offsets[i]..g.offsets[i + 1];
        let wi: f64 = g.weights[row.clone()].iter().sum();
        let wi2: f64 = g.weights[row].iter().map(|w| w * w).sum();
        out.eic.push(-(zi * zi * wi) / (n1 * m2));
        let term1 = (zi / m2) * (zi / m2);
        let term2 = n / (n - 2.0);
        let term3 = wi2 - wi * wi / n1;
        let term4 = m2 - zi * zi / n1;
        out.vic.push(term1 * term2 * term3 * term4);
        out.ei.push(-wi / n1);
        let mut vi = wi2 * (n - b2) / n1;
        vi += (wi * wi - wi2) * (2.0 * b2 - n) / (n1 * (n - 2.0));
        vi -= (-wi / n1) * (-wi / n1);
        out.vi.push(vi);
    }
    out
}

fn statistics(focal: &[f64], lag: Vec<f64>, geoda_quads: bool) -> (Vec<f64>, Vec<u8>, Vec<f64>) {
    let den: f64 = focal.iter().map(|z| z * z).sum();
    let n1 = focal.len() as f64 - 1.0;
    let is = focal
        .iter()
        .zip(&lag)
        .map(|(z, l)| n1 * z * l / den)
        .collect();
    let q = quadrants(focal, &lag, geoda_quads);
    (is, q, lag)
}

/// Local Moran's I of `y` with its quadrants and analytic moments.
pub fn local_moran(y: &[f64], g: &Csr, geoda_quads: bool) -> Result<LocalMoran, String> {
    check(y.len(), g, "y")?;
    let z = standardize(y, "y")?;
    let (is, q, lag) = statistics(&z, g.lag(&z)?, geoda_quads);
    let moments = Some(moments(g, &z));
    Ok(LocalMoran {
        n: y.len(),
        z,
        zy: None,
        lag,
        is,
        q,
        moments,
    })
}

/// Bivariate local Moran's I: `x` at the site against the spatial lag of `y`.
pub fn local_moran_bv(
    x: &[f64],
    y: &[f64],
    g: &Csr,
    geoda_quads: bool,
) -> Result<LocalMoran, String> {
    check(x.len(), g, "x")?;
    check(y.len(), g, "y")?;
    let zx = standardize(x, "x")?;
    let zy = standardize(y, "y")?;
    let (is, q, lag) = statistics(&zx, g.lag(&zy)?, geoda_quads);
    Ok(LocalMoran {
        n: x.len(),
        z: zx,
        zy: Some(zy),
        lag,
        is,
        q,
        moments: None,
    })
}

/// Conditional randomization of the univariate statistic.
///
/// p-values are taken against the observed statistic re-evaluated with the replicate formula
/// ([`crand::observed_with`]), which equals `Is` up to rounding but ties exactly with a
/// replicate that redraws the observed neighbors.
///
/// esda's `Moran_Local` never forwards `island_weight`, so `settings.island_weight` is
/// ignored here and an island's replicates are all zero.
pub fn local_moran_permutations(y: &[f64], g: &Csr, settings: &Settings) -> Result<Crand, String> {
    let m = local_moran(y, g, false)?;
    let scaling = (m.n as f64 - 1.0) / m.z.iter().map(|z| z * z).sum::<f64>();
    let settings = Settings {
        island_weight: 0.0,
        ..*settings
    };
    let z = &m.z;
    let stat = |i: usize, self_w: f64, other: &[f64], draw: &[f64]| {
        let lag: f64 = other.iter().zip(draw).map(|(w, d)| w * d).sum();
        z[i] * (lag + self_w * z[i]) * scaling
    };
    let observed = crand::observed_with(g, z, 0.0, stat);
    crand::crand(g, z, &observed, &settings, stat)
}

/// Conditional randomization of the bivariate statistic (neighbors drawn from `zy`).
pub fn local_moran_bv_permutations(
    x: &[f64],
    y: &[f64],
    g: &Csr,
    settings: &Settings,
) -> Result<Crand, String> {
    let m = local_moran_bv(x, y, g, false)?;
    let zy = m.zy.as_ref().expect("bivariate result carries zy");
    let scaling = (m.n as f64 - 1.0) / m.z.iter().map(|z| z * z).sum::<f64>();
    let settings = Settings {
        island_weight: 0.0,
        ..*settings
    };
    let zx = &m.z;
    let stat = |i: usize, self_w: f64, other: &[f64], draw: &[f64]| {
        let lag: f64 = other.iter().zip(draw).map(|(w, d)| w * d).sum();
        zx[i] * (lag + self_w * zy[i]) * scaling
    };
    let observed = crand::observed_with(g, zy, 0.0, stat);
    crand::crand(g, zy, &observed, &settings, stat)
}

/// LISA cluster labels (`esda`'s `get_cluster_labels`): the quadrant name when `p_sim < crit`
/// (strict), otherwise `"Insignificant"`. Labels follow `q` as given, so with `geoda_quads`
/// the labels of quadrants 2 and 3 are swapped, as in esda.
pub fn cluster_labels(q: &[u8], p_sim: &[f64], crit: f64) -> Vec<&'static str> {
    q.iter()
        .zip(p_sim)
        .map(|(&q, &p)| {
            if p < crit {
                match q {
                    1 => "High-High",
                    2 => "Low-High",
                    3 => "Low-Low",
                    _ => "High-Low",
                }
            } else {
                "Insignificant"
            }
        })
        .collect()
}

#[cfg(test)]
mod tests;
