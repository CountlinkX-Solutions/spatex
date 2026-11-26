//! Local Geary (`esda.Geary_Local`, Anselin 1995).
//!
//! The graph passed in must already be row-standardised (esda forces `transform("r")`).
//! `localG_i = sum_j w_ij (z_i - z_j)^2` over standardised values (ddof 0). Inference is the
//! conditional randomization of [`crate::perm::crand`]. Unlike esda, islands keep their place in
//! the output (`localG = 0`); esda drops them and misaligns the result.

use crate::graph::Csr;
use crate::perm::crand::{self, Crand, Settings};

/// Observed local Geary statistics.
#[derive(Clone, Debug)]
pub struct GearyLocal {
    /// Standardised values, ddof 0.
    pub z: Vec<f64>,
    /// `localG` per site (0 for islands).
    pub local_g: Vec<f64>,
}

pub(super) fn standardize(x: &[f64], g: &Csr) -> Result<Vec<f64>, String> {
    if x.len() != g.n {
        return Err(format!(
            "x has length {} but the graph has {} nodes (length mismatch)",
            x.len(),
            g.n
        ));
    }
    if x.len() < 3 {
        return Err("at least 3 observations are required".into());
    }
    if let Some(i) = x.iter().position(|v| !v.is_finite()) {
        return Err(format!("x: non-finite value at index {i}"));
    }
    if x.iter().all(|v| *v == x[0]) {
        return Err("x is constant: local Geary is undefined".into());
    }
    let n = x.len() as f64;
    let mean = x.iter().sum::<f64>() / n;
    let sd = (x.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n).sqrt();
    Ok(x.iter().map(|v| (v - mean) / sd).collect())
}

/// Local Geary `localG` of `x` over the (row-standardised) graph `g`.
pub fn geary_local(x: &[f64], g: &Csr) -> Result<GearyLocal, String> {
    let z = standardize(x, g)?;
    let local_g = (0..g.n)
        .map(|i| {
            g.row(i)
                .map(|k| {
                    let d = z[i] - z[g.neighbors[k] as usize];
                    g.weights[k] * d * d
                })
                .sum()
        })
        .collect();
    Ok(GearyLocal { z, local_g })
}

/// Conditional randomization of `localG`: `rstat = sum_c w_c (z_i - draw_c)^2` over the
/// non-self neighbors. `settings.island_weight` is honored as in esda: an island simulates
/// against one pseudo-neighbor of that weight, while its observed `localG` stays 0.
pub fn geary_local_permutations(x: &[f64], g: &Csr, settings: &Settings) -> Result<Crand, String> {
    let observed = geary_local(x, g)?.local_g;
    let z = standardize(x, g)?;
    let stat = |i: usize, _self_w: f64, other: &[f64], draw: &[f64]| {
        other
            .iter()
            .zip(draw)
            .map(|(w, d)| w * (z[i] - d) * (z[i] - d))
            .sum::<f64>()
    };
    crand::crand(g, &z, &observed, settings, stat)
}

/// Cluster/outlier labels of esda (`1` outlier, `2` cluster, `3` other, `4` not significant,
/// NaN where none applies), assigned in that order so later rules overwrite earlier ones.
pub fn labels(local_g: &[f64], x: &[f64], p_sim: &[f64], sig: f64) -> Vec<f64> {
    let n = local_g.len() as f64;
    let eij_mean = local_g.iter().sum::<f64>() / n;
    let x_mean = x.iter().sum::<f64>() / n;
    local_g
        .iter()
        .zip(x)
        .zip(p_sim)
        .map(|((&g, &x), &p)| {
            let mut lab = f64::NAN;
            let significant = p <= sig;
            if g < eij_mean && x > x_mean && significant {
                lab = 1.0;
            }
            if g < eij_mean && x < x_mean && significant {
                lab = 2.0;
            }
            if g > eij_mean && significant {
                lab = 3.0;
            }
            if p > sig {
                lab = 4.0;
            }
            lab
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Transform;
    use crate::perm::crand::Settings;
    use crate::perm::Alternative;

    /// Path 0-1-2 plus island 3, row-standardised.
    fn path_island() -> Csr {
        let focal = [0, 1, 1, 2];
        let neighbor = [1, 0, 2, 1];
        Csr::from_arrays(4, &focal, &neighbor, &[1.0; 4])
            .unwrap()
            .transform(Transform::R)
    }

    fn settings(permutations: usize, keep: bool) -> Settings {
        Settings {
            permutations,
            seed: 7,
            alternative: Alternative::Directed,
            island_weight: 0.0,
            keep,
        }
    }

    #[test]
    fn local_g_matches_hand_computation_and_zeroes_islands() {
        // x = [1, 2, 3, 2]: mean 2, sd 1/sqrt(2), z = [-sqrt2, 0, sqrt2, 0].
        let m = geary_local(&[1.0, 2.0, 3.0, 2.0], &path_island()).unwrap();
        for g in &m.local_g[..3] {
            assert!((g - 2.0).abs() < 1e-12, "{:?}", m.local_g);
        }
        assert_eq!(m.local_g[3], 0.0);
        assert_eq!(m.z.len(), 4);
    }

    #[test]
    fn rejects_constant_and_non_finite_input() {
        let g = path_island();
        assert!(geary_local(&[2.0; 4], &g).unwrap_err().contains("constant"));
        assert!(geary_local(&[1.0, f64::NAN, 3.0, 4.0], &g).is_err());
        assert!(geary_local(&[1.0, 2.0, 3.0], &g)
            .unwrap_err()
            .contains("length"));
    }

    #[test]
    fn labels_follow_esda_assignment_order() {
        // Eij_mean = 1.8, x_mean = 4.2.
        let local_g = [0.5, 0.5, 3.0, 3.0, 2.0];
        let x = [10.0, 1.0, 5.0, 5.0, 5.0];
        let p = [0.01, 0.01, 0.01, 0.2, 0.01];
        assert_eq!(
            labels(&local_g, &x, &p, 0.05),
            vec![1.0, 2.0, 3.0, 4.0, 3.0]
        );
        // A tie with the mean and p <= sig stays unlabelled (NaN), as in esda.
        let l = labels(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0], &[0.01, 0.01, 0.2], 0.05);
        assert_eq!(l[0], 2.0);
        assert!(l[1].is_nan());
        assert_eq!(l[2], 4.0);
    }

    #[test]
    fn permutations_are_valid_deterministic_and_island_is_minimal() {
        let x = [1.0, 2.0, 3.5, 2.0];
        let g = path_island();
        let a = geary_local_permutations(&x, &g, &settings(99, true)).unwrap();
        let b = geary_local_permutations(&x, &g, &settings(99, false)).unwrap();
        assert_eq!(a.p_sim, b.p_sim);
        assert!(b.sims.is_none());
        assert_eq!(a.sims.as_ref().unwrap().len(), 4 * 99);
        assert!(a.p_sim.iter().all(|p| (0.01..=1.0).contains(p)));
        assert_eq!(a.p_sim[3], 0.01);
    }
}
