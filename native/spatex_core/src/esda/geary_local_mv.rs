//! Multivariate local Geary (`esda.Geary_Local_MV`, Anselin 2019).
//!
//! The graph passed in must already be row-standardised (esda forces `transform("r")`).
//! `localG_i = sum_v sum_j w_ij (z_vi - z_vj)^2 / k` over the `k` variables, each standardised
//! with ddof 0. Islands get 0. Inference is the conditional randomization of
//! [`crate::perm::crand`] on neighbor *indices*: one draw relabels the neighbors of a site for
//! every variable at once, as esda's own loop does. esda's draws are unseeded numpy ones; the
//! folded pseudo p-value is that of esda (the directed alternative).

use super::geary_local::standardize;
use crate::graph::Csr;
use crate::perm::crand::{self, Crand, Settings};

fn standardized(vars: &[Vec<f64>], g: &Csr) -> Result<Vec<Vec<f64>>, String> {
    if vars.len() < 2 {
        return Err(format!(
            "at least two variables are required, got {}",
            vars.len()
        ));
    }
    vars.iter()
        .enumerate()
        .map(|(v, x)| standardize(x, g).map_err(|e| format!("variable {v}: {e}")))
        .collect()
}

fn local_g(z: &[Vec<f64>], i: usize, weights: &[f64], neighbors: impl Fn(usize) -> usize) -> f64 {
    let k = z.len() as f64;
    let mut total = 0.0;
    for zv in z {
        for (c, w) in weights.iter().enumerate() {
            let d = zv[i] - zv[neighbors(c)];
            total += w * d * d;
        }
    }
    total / k
}

/// Observed multivariate `localG` of `vars` (one vector per variable) over `g`.
pub fn geary_local_mv(vars: &[Vec<f64>], g: &Csr) -> Result<Vec<f64>, String> {
    let z = standardized(vars, g)?;
    Ok((0..g.n)
        .map(|i| {
            let row: Vec<usize> = g.row(i).collect();
            let weights: Vec<f64> = row.iter().map(|&e| g.weights[e]).collect();
            local_g(&z, i, &weights, |c| g.neighbors[row[c]] as usize)
        })
        .collect())
}

/// Conditional randomization of the multivariate `localG`.
pub fn geary_local_mv_permutations(
    vars: &[Vec<f64>],
    g: &Csr,
    settings: &Settings,
) -> Result<Crand, String> {
    let observed = geary_local_mv(vars, g)?;
    let z = standardized(vars, g)?;
    // crand permutes these values; they are the site indices, exact in f64.
    let ids: Vec<f64> = (0..g.n).map(|i| i as f64).collect();
    let stat = |i: usize, _self_w: f64, other: &[f64], draw: &[f64]| {
        local_g(&z, i, other, |c| draw[c] as usize)
    };
    crand::crand(g, &ids, &observed, settings, stat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::esda::geary_local::geary_local;
    use crate::perm::Alternative;

    /// Row-standardised path 0-1-2-3 plus island 4.
    fn graph() -> Csr {
        let focal = [0, 1, 1, 2, 2, 3];
        let neighbor = [1, 0, 2, 1, 3, 2];
        let weight = [1.0, 0.5, 0.5, 0.5, 0.5, 1.0];
        Csr::from_arrays(5, &focal, &neighbor, &weight).unwrap()
    }

    #[test]
    fn localg_is_the_mean_of_the_univariate_statistics() {
        let a = vec![1.0, 4.0, 2.0, 8.0, 5.0];
        let b = vec![2.0, 2.5, 9.0, 1.0, 4.0];
        let mv = geary_local_mv(&[a.clone(), b.clone()], &graph()).unwrap();
        let (ga, gb) = (
            geary_local(&a, &graph()).unwrap().local_g,
            geary_local(&b, &graph()).unwrap().local_g,
        );
        for i in 0..5 {
            assert!((mv[i] - (ga[i] + gb[i]) / 2.0).abs() < 1e-12);
        }
        assert_eq!(mv[4], 0.0);
    }

    #[test]
    fn permutations_are_seeded_and_islands_take_the_smallest_p() {
        let vars = vec![vec![1.0, 4.0, 2.0, 8.0, 5.0], vec![2.0, 2.5, 9.0, 1.0, 4.0]];
        let s = Settings {
            permutations: 99,
            seed: 5,
            alternative: Alternative::Directed,
            island_weight: 0.0,
            keep: true,
        };
        let a = geary_local_mv_permutations(&vars, &graph(), &s).unwrap();
        let b = geary_local_mv_permutations(&vars, &graph(), &s).unwrap();
        assert_eq!(a.p_sim, b.p_sim);
        assert_eq!(a.p_sim[4], 1.0 / 100.0);
        assert!(a.p_sim.iter().all(|p| *p > 0.0 && *p <= 0.5 + 1e-12));
        assert!(geary_local_mv(&vars[..1], &graph()).is_err());
    }
}
