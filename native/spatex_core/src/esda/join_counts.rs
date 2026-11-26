//! Binary join counts (`esda.Join_Counts`).
//!
//! The graph is binarized and every stored directed row counts (both directions of each edge),
//! each tallied half, exactly like esda. Rows of island nodes are dropped when `drop_islands`
//! (the default); kept, an island's zero-weight self-loop counts as a same-colour join, which is
//! esda's behaviour too.
//!
//! One deliberate deviation: esda's `sim_autocorr_chi2` divides both squared deviations by the
//! square of the mean simulated positive autocorrelation, which is not a chi-square statistic.
//! [`autocorr_chi2_p`] computes the Pearson statistic the attribute was meant to hold.

use rayon::prelude::*;

use crate::esda::moran::{check_finite, check_graph, check_permutations};
use crate::graph::{Csr, Transform};
use crate::perm::{self, erfc};

/// Observed join counts and the contingency-table test.
#[derive(Clone, Copy, Debug)]
pub struct JoinCounts {
    /// Number of observations (islands included).
    pub n: usize,
    /// Number of joins, `S0 / 2` of the binary graph.
    pub j: f64,
    /// Black-black joins.
    pub bb: f64,
    /// White-white joins.
    pub ww: f64,
    /// Black-white joins (`bw + wb` of the table).
    pub bw: f64,
    /// Yates-corrected chi-square of the 2x2 table.
    pub chi2: f64,
    /// Survival function of `chi2` with one degree of freedom.
    pub chi2_p: f64,
    /// Degrees of freedom, always 1.
    pub chi2_dof: f64,
    /// `[[ww, wb], [bw, bb]]` flattened row-major (focal rows, neighbor columns).
    pub crosstab: [f64; 4],
    /// Expected table under independence, same layout.
    pub expected: [f64; 4],
    /// `bb + ww`.
    pub autocorr_pos: f64,
    /// `bw`.
    pub autocorr_neg: f64,
}

/// `scipy.stats.chi2_contingency` for a 2x2 table `[a, b, c, d]` with its default Yates
/// correction. Returns `(chi2, p, dof, expected)`; errors when an expected cell is zero.
pub fn chi2_contingency(table: [f64; 4]) -> Result<(f64, f64, f64, [f64; 4]), String> {
    let [a, b, c, d] = table;
    let (r0, r1, c0, c1) = (a + b, c + d, a + c, b + d);
    let total = r0 + r1;
    let expected = [
        r0 * c0 / total,
        r0 * c1 / total,
        r1 * c0 / total,
        r1 * c1 / total,
    ];
    if expected.iter().any(|&e| e.is_nan() || e <= 0.0) {
        return Err(
            "the table of expected join frequencies has a zero element (y is constant or no \
             node has a neighbor of the other colour class)"
                .into(),
        );
    }
    let mut chi2 = 0.0;
    for (o, e) in table.iter().zip(&expected) {
        let diff = e - o;
        let corrected = o + diff.signum() * diff.abs().min(0.5);
        chi2 += (corrected - e) * (corrected - e) / e;
    }
    Ok((chi2, erfc((chi2 / 2.0).sqrt()), 1.0, expected))
}

/// P-value of esda's `sim_autocorr_chi2` with the Pearson statistic
/// `(pos - mean_pos)^2 / mean_pos + (neg - mean_neg)^2 / mean_neg` and one degree of freedom.
/// NaN when a simulated mean is not positive.
pub fn autocorr_chi2_p(pos: f64, neg: f64, mean_pos: f64, mean_neg: f64) -> f64 {
    if !(mean_pos > 0.0 && mean_neg > 0.0) {
        return f64::NAN;
    }
    let stat = (pos - mean_pos).powi(2) / mean_pos + (neg - mean_neg).powi(2) / mean_neg;
    erfc((stat / 2.0).sqrt())
}

/// The directed (focal, neighbor) rows that join counts tallies, plus validated colours.
pub struct JoinStat {
    focal: Vec<u32>,
    neighbor: Vec<u32>,
    j: f64,
    n: usize,
}

/// `[bb, ww, bw, wb]` directed row counts.
type Tally = [usize; 4];

impl JoinStat {
    /// Validates `y` as 0/1 and collects the rows of the binarized graph.
    pub fn new(y: &[f64], graph: &Csr, drop_islands: bool) -> Result<Self, String> {
        check_graph(y.len(), graph)?;
        check_finite(y, "y")?;
        if let Some(i) = y.iter().position(|&v| v != 0.0 && v != 1.0) {
            return Err(format!(
                "y must be binary (0 or 1), found {} at index {i}",
                y[i]
            ));
        }
        let g = graph.transform(Transform::B);
        let isolated: Vec<bool> = g.cardinalities().iter().map(|&c| c == 0).collect();
        let (mut focal, mut neighbor) = (Vec::new(), Vec::new());
        for (i, &island) in isolated.iter().enumerate() {
            if drop_islands && island {
                continue;
            }
            for &j in &g.neighbors[g.offsets[i]..g.offsets[i + 1]] {
                focal.push(i as u32);
                neighbor.push(j);
            }
        }
        Ok(JoinStat {
            focal,
            neighbor,
            j: g.s0() / 2.0,
            n: y.len(),
        })
    }

    fn tally(&self, z: &[f64]) -> Tally {
        let mut t = [0usize; 4];
        for (&f, &nb) in self.focal.iter().zip(&self.neighbor) {
            let (zf, zn) = (z[f as usize] == 1.0, z[nb as usize] == 1.0);
            let slot = match (zf, zn) {
                (true, true) => 0,
                (false, false) => 1,
                (true, false) => 2,
                (false, true) => 3,
            };
            t[slot] += 1;
        }
        t
    }

    /// `(bb, ww, bw + wb, chi2, table)` for colours `z`, or an error for a degenerate table.
    fn counts(&self, z: &[f64]) -> Result<Counts, String> {
        let t = self.tally(z);
        let (bb, ww, bw, wb) = (
            t[0] as f64 / 2.0,
            t[1] as f64 / 2.0,
            t[2] as f64 / 2.0,
            t[3] as f64 / 2.0,
        );
        let table = [ww, wb, bw, bb];
        let (chi2, chi2_p, dof, expected) = chi2_contingency(table)?;
        Ok(Counts {
            bb,
            ww,
            bw: bw + wb,
            chi2,
            chi2_p,
            dof,
            table,
            expected,
        })
    }
}

struct Counts {
    bb: f64,
    ww: f64,
    bw: f64,
    chi2: f64,
    chi2_p: f64,
    dof: f64,
    table: [f64; 4],
    expected: [f64; 4],
}

/// Observed join counts of binary `y`.
pub fn join_counts(y: &[f64], graph: &Csr, drop_islands: bool) -> Result<JoinCounts, String> {
    let ctx = JoinStat::new(y, graph, drop_islands)?;
    let c = ctx.counts(y)?;
    Ok(JoinCounts {
        n: ctx.n,
        j: ctx.j,
        bb: c.bb,
        ww: c.ww,
        bw: c.bw,
        chi2: c.chi2,
        chi2_p: c.chi2_p,
        chi2_dof: c.dof,
        crosstab: c.table,
        expected: c.expected,
        autocorr_pos: c.bb + c.ww,
        autocorr_neg: c.bw,
    })
}

/// Reference distributions and upper-tail pseudo p-values of a join counts permutation test.
#[derive(Clone, Debug)]
pub struct JoinPermTest {
    /// Simulated black-black joins, in replicate order.
    pub sim_bb: Vec<f64>,
    /// Simulated white-white joins.
    pub sim_ww: Vec<f64>,
    /// Simulated black-white joins.
    pub sim_bw: Vec<f64>,
    /// Simulated chi-square statistics.
    pub sim_chi2: Vec<f64>,
    /// `min(sim_bb)`.
    pub min_bb: f64,
    /// `mean(sim_bb)`.
    pub mean_bb: f64,
    /// `max(sim_bb)`.
    pub max_bb: f64,
    /// `min(sim_bw)`.
    pub min_bw: f64,
    /// `mean(sim_bw)`.
    pub mean_bw: f64,
    /// `max(sim_bw)`.
    pub max_bw: f64,
    /// Mean simulated `bb + ww`.
    pub mean_autocorr_pos: f64,
    /// Mean simulated `bw`.
    pub mean_autocorr_neg: f64,
    /// `(#{sim_bb >= bb} + 1) / (P + 1)`.
    pub p_sim_bb: f64,
    /// Same for `bw`.
    pub p_sim_bw: f64,
    /// Same for `chi2`.
    pub p_sim_chi2: f64,
    /// Same for `bb + ww`.
    pub p_sim_autocorr_pos: f64,
    /// Same for `bw`.
    pub p_sim_autocorr_neg: f64,
    /// P-value of the Pearson autocorrelation statistic (see [`autocorr_chi2_p`]).
    pub sim_autocorr_chi2: f64,
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn upper_p<I: IntoIterator<Item = f64>>(sim: I, observed: f64, permutations: usize) -> f64 {
    let larger = sim.into_iter().filter(|&s| s >= observed).count();
    (larger as f64 + 1.0) / (permutations as f64 + 1.0)
}

/// Permutation test of the join counts: `permutations` shuffles of the colours. A shuffle whose
/// table has a zero expected cell is discarded and redrawn from the same replicate stream, as
/// esda does.
pub fn join_counts_permutations(
    y: &[f64],
    graph: &Csr,
    drop_islands: bool,
    permutations: usize,
    seed: u64,
) -> Result<JoinPermTest, String> {
    check_permutations(permutations)?;
    let ctx = JoinStat::new(y, graph, drop_islands)?;
    let obs = ctx.counts(y)?;
    let draws: Vec<[f64; 4]> = (0..permutations)
        .into_par_iter()
        .map_init(
            || y.to_vec(),
            |buf, r| {
                let mut rng = perm::replicate_rng(seed, r as u64);
                loop {
                    buf.copy_from_slice(y);
                    perm::shuffle(buf, &mut rng);
                    if let Ok(c) = ctx.counts(buf) {
                        return [c.bb, c.ww, c.bw, c.chi2];
                    }
                }
            },
        )
        .collect();
    let col = |k: usize| -> Vec<f64> { draws.iter().map(|d| d[k]).collect() };
    let (sim_bb, sim_ww, sim_bw, sim_chi2) = (col(0), col(1), col(2), col(3));
    let pos: Vec<f64> = sim_bb.iter().zip(&sim_ww).map(|(a, b)| a + b).collect();
    let fold = |v: &[f64], f: fn(f64, f64) -> f64, init: f64| v.iter().copied().fold(init, f);
    let (mean_pos, mean_neg) = (mean(&pos), mean(&sim_bw));
    Ok(JoinPermTest {
        min_bb: fold(&sim_bb, f64::min, f64::INFINITY),
        mean_bb: mean(&sim_bb),
        max_bb: fold(&sim_bb, f64::max, f64::NEG_INFINITY),
        min_bw: fold(&sim_bw, f64::min, f64::INFINITY),
        mean_bw: mean_neg,
        max_bw: fold(&sim_bw, f64::max, f64::NEG_INFINITY),
        mean_autocorr_pos: mean_pos,
        mean_autocorr_neg: mean_neg,
        p_sim_bb: upper_p(sim_bb.iter().copied(), obs.bb, permutations),
        p_sim_bw: upper_p(sim_bw.iter().copied(), obs.bw, permutations),
        p_sim_chi2: upper_p(sim_chi2.iter().copied(), obs.chi2, permutations),
        p_sim_autocorr_pos: upper_p(pos.iter().copied(), obs.bb + obs.ww, permutations),
        p_sim_autocorr_neg: upper_p(sim_bw.iter().copied(), obs.bw, permutations),
        sim_autocorr_chi2: autocorr_chi2_p(obs.bb + obs.ww, obs.bw, mean_pos, mean_neg),
        sim_bb,
        sim_ww,
        sim_bw,
        sim_chi2,
    })
}

#[cfg(test)]
mod tests;
