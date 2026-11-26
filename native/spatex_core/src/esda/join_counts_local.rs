//! Local join counts (`esda.Join_Counts_Local`, Anselin and Li 2019): univariate, and the
//! bivariate `BJC` case of `esda.Join_Counts_Local_BV`. The bivariate `CLC` case and
//! `esda.Join_Counts_Local_MV` are the univariate counts of the product of the variables.
//!
//! `LJC_i` counts the neighbors of a site with `y = 1` when the site itself has `y = 1`. Self
//! weights are ignored and weights are treated as binary (any non-zero weight is a join), which
//! is what esda's `assign_self_weight(0).eliminate_zeros().transform("b")` amounts to. Islands
//! have `LJC = 0` (esda crashes on a graph with islands).

use crate::graph::Csr;
use crate::perm::crand::{self, Crand, Settings};

fn check(y: &[f64], g: &Csr) -> Result<(), String> {
    if y.len() != g.n {
        return Err(format!(
            "y has length {} but the graph has {} nodes (length mismatch)",
            y.len(),
            g.n
        ));
    }
    if y.iter().any(|&v| v != 0.0 && v != 1.0) {
        return Err("y must be binary (every value 0 or 1)".into());
    }
    Ok(())
}

/// Number of joined (`y = 1`) neighbors of every `y = 1` site; 0 for `y = 0` sites and islands.
pub fn join_counts_local(y: &[f64], g: &Csr) -> Result<Vec<f64>, String> {
    check(y, g)?;
    Ok((0..g.n)
        .map(|i| {
            if y[i] != 1.0 {
                return 0.0;
            }
            g.row(i)
                .filter(|&k| g.neighbors[k] as usize != i && g.weights[k] != 0.0)
                .filter(|&k| y[g.neighbors[k] as usize] == 1.0)
                .count() as f64
        })
        .collect())
}

/// The simulated joins of site `i` over its drawn neighbor values, with binary weights: 1 for a
/// neighbor (non-zero weight), `island_weight` for an island's single pseudo-neighbor. esda
/// weighs its draws the same way (`transform("b")`, then `zrand @ weights`).
fn drawn_joins(g: &Csr, island_weight: f64) -> impl Fn(usize, &[f64], &[f64]) -> f64 + Sync {
    // Islands: no neighbor other than the site itself (crand then draws one pseudo-neighbor).
    let island: Vec<bool> = (0..g.n)
        .map(|i| g.row(i).all(|k| g.neighbors[k] as usize == i))
        .collect();
    move |i, other, draw| {
        if island[i] {
            island_weight * draw.iter().sum::<f64>()
        } else {
            draw.iter()
                .zip(other)
                .filter(|(_, w)| **w != 0.0)
                .map(|(d, _)| d)
                .sum()
        }
    }
}

fn nan_where_zero(mut out: Crand, ljc: &[f64]) -> Crand {
    for (p, l) in out.p_sim.iter_mut().zip(ljc) {
        if *l == 0.0 {
            *p = f64::NAN;
        }
    }
    out
}

/// Conditional randomization of the counts (`rstat = y_i * joins(draw)`, see [`drawn_joins`]);
/// `p_sim` is NaN at the sites whose observed count is 0, as in esda.
pub fn join_counts_local_permutations(
    y: &[f64],
    g: &Csr,
    settings: &Settings,
) -> Result<Crand, String> {
    let ljc = join_counts_local(y, g)?;
    let joins = drawn_joins(g, settings.island_weight);
    let stat = |i: usize, _self_w: f64, other: &[f64], draw: &[f64]| y[i] * joins(i, other, draw);
    let observed = crand::observed_with(g, y, settings.island_weight, stat);
    let out = crand::crand(g, y, &observed, settings, stat)?;
    Ok(nan_where_zero(out, &ljc))
}

/// Bivariate local join counts, esda's `case="BJC"`: at every site with `x = 1, z = 0`, the
/// number of neighbors with `x = 0, z = 1` (0 elsewhere and at islands).
pub fn join_counts_local_bjc(x: &[f64], z: &[f64], g: &Csr) -> Result<Vec<f64>, String> {
    check(x, g)?;
    check(z, g)?;
    Ok((0..g.n)
        .map(|i| {
            if !(x[i] == 1.0 && z[i] == 0.0) {
                return 0.0;
            }
            g.row(i)
                .filter(|&k| g.neighbors[k] as usize != i && g.weights[k] != 0.0)
                .filter(|&k| {
                    let j = g.neighbors[k] as usize;
                    x[j] == 0.0 && z[j] == 1.0
                })
                .count() as f64
        })
        .collect())
}

/// Conditional randomization of the BJC counts. esda simulates `x_i * joins(z draws)` against
/// the observed BJC count (its `_ljc_bv_case1`), which ignores the `1 - z_i` and `1 - x_j`
/// factors of the observation; that quirk is kept. `p_sim` is NaN where the count is 0.
pub fn join_counts_local_bjc_permutations(
    x: &[f64],
    z: &[f64],
    g: &Csr,
    settings: &Settings,
) -> Result<Crand, String> {
    let ljc = join_counts_local_bjc(x, z, g)?;
    let joins = drawn_joins(g, settings.island_weight);
    let stat = |i: usize, _self_w: f64, other: &[f64], draw: &[f64]| x[i] * joins(i, other, draw);
    let out = crand::crand(g, z, &ljc, settings, stat)?;
    Ok(nan_where_zero(out, &ljc))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perm::crand::Settings;
    use crate::perm::Alternative;

    /// Path 0-1-2-3 plus island 4, with a self weight on node 0 (ignored by the statistic).
    fn graph() -> Csr {
        let focal = [0, 0, 1, 1, 2, 2, 3];
        let neighbor = [0, 1, 0, 2, 1, 3, 2];
        let weight = [5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        Csr::from_arrays(5, &focal, &neighbor, &weight).unwrap()
    }

    const Y: [f64; 5] = [1.0, 1.0, 0.0, 1.0, 1.0];

    fn settings(permutations: usize, keep: bool) -> Settings {
        Settings {
            permutations,
            seed: 3,
            alternative: Alternative::Directed,
            island_weight: 0.0,
            keep,
        }
    }

    #[test]
    fn counts_joined_neighbors_and_ignores_self_weights_and_islands() {
        assert_eq!(
            join_counts_local(&Y, &graph()).unwrap(),
            vec![1.0, 1.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn rejects_non_binary_input() {
        assert!(join_counts_local(&[0.0, 1.0, 2.0, 0.0, 1.0], &graph())
            .unwrap_err()
            .contains("binary"));
        assert!(join_counts_local(&[0.0, 1.0], &graph()).is_err());
    }

    #[test]
    fn p_sim_is_nan_exactly_where_the_count_is_zero() {
        let c = join_counts_local_permutations(&Y, &graph(), &settings(99, true)).unwrap();
        let ljc = join_counts_local(&Y, &graph()).unwrap();
        for (p, l) in c.p_sim.iter().zip(&ljc) {
            assert_eq!(p.is_nan(), *l == 0.0, "{:?}", c.p_sim);
            if *l > 0.0 {
                assert!((0.01..=1.0).contains(p));
            }
        }
        let k = join_counts_local_permutations(&Y, &graph(), &settings(99, false)).unwrap();
        assert!(k.sims.is_none());
        for (a, b) in c.p_sim.iter().zip(&k.p_sim) {
            assert!(a == b || (a.is_nan() && b.is_nan()));
        }
    }
}
