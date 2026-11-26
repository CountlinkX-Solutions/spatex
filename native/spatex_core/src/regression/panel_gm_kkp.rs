//! The Kapoor-Kelejian-Prucha GM estimator of the random-effects spatial error panel
//! (`spreg.GM_ErrorRE`, spreg 1.9.1 `panel.py`; `spreg.GM_KKP` in the deprecated `sp_panels.py` is
//! the same estimator).
//!
//! `y = X b + mu_i + u`, `u = lambda W u + e`, data stacked by period (`y[s * n + i]` is area `i`
//! in period `s`), `x` including the constant. The estimator has three stages:
//!
//! 1. OLS; the three moment equations of the within transformation `Q0` give a first `lambda`
//!    and `sigma_v^2` (`optim_moments`), and `sigma_1^2` follows from the between transformation.
//! 2. All six moments (within and between, `trace_w2` shared) are minimized over
//!    `(lambda, sigma_v^2, sigma_1^2)`, weighted by `Xi = diag(sigma_v^4 / (t - 1), sigma_1^4) (x)
//!    Tau` (`full_weights`) or `Tau = I`.
//! 3. A feasible GLS: `theta = 1 - sigma_v / sigma_1`, both sides filtered with `I - lambda W` and
//!    quasi-demeaned by `I - theta Q1`, OLS on the result (`vm = s2 (X*' X*)^-1`, `s2` with
//!    `nt - k`).
//!
//! Deviation (`known_deviations.exs`): spreg minimizes with L-BFGS-B, which stops early. This port
//! profiles the variance parameters out in closed form (a non-negative least-squares problem in
//! `sigma_v^2`, `sigma_1^2`) and minimizes over `lambda` on `[-0.99, 0.99]` exactly, starting the
//! downhill walk from the same point as spreg (0, then the first-step `lambda`). Every formula
//! agrees with spreg to `1e-9` when handed spreg's own parameters ([`gm_kkp`] with `at`).

use faer::Mat;

use super::gm_common::{cholesky, dot, minimize_lambda_from, Moments, Sp};
use super::linalg::{fit, inverse, row_major};
use super::ml_common::pearson_r;
use super::ols::Entries;
use super::panel::{fitted, lag_nt, quasi_demean, z_stats};
use crate::graph::Csr;

/// spreg's two optimizer outcomes, to bypass the optimizer: `[lambda_1, sigma_v^2 (first step),
/// lambda, sigma_v^2, sigma_1^2]`.
pub type KkpAt = [f64; 5];

/// Area means of `v`, replicated over the periods (`Q1 v`).
fn between(v: &[f64], n: usize, t: usize) -> Vec<f64> {
    let means: Vec<f64> = (0..n)
        .map(|i| (0..t).map(|s| v[s * n + i]).sum::<f64>() / t as f64)
        .collect();
    (0..v.len()).map(|e| means[e % n]).collect()
}

/// `G` and `g` of the within (`between_q = false`, 3 x 3) or between (3 x 4) transformation
/// (`_moments_kkp`), and `trace_w2` (the sum of the squared weights, shared between the two).
fn moments_kkp(
    w: &Csr,
    u: &[f64],
    t: usize,
    between_q: bool,
    trace_w2: Option<f64>,
) -> Result<(Moments, f64), String> {
    let n = w.n;
    let ub = lag_nt(w, u, n)?;
    let ubb = lag_nt(w, &ub, n)?;
    let q = |v: &[f64]| -> Vec<f64> {
        let m = between(v, n, t);
        if between_q {
            m
        } else {
            v.iter().zip(&m).map(|(a, b)| a - b).collect()
        }
    };
    let (qu, qub, qubb) = (q(u), q(&ub), q(&ubb));
    let g11 = 2.0 * dot(u, &qub);
    let g12 = -dot(&ub, &qub);
    let g21 = 2.0 * dot(&ubb, &qub);
    let g22 = -dot(&ubb, &qubb);
    let g31 = dot(u, &qubb) + dot(&ub, &qub);
    let g32 = -dot(&ub, &qubb);
    let trace_w2 = trace_w2.unwrap_or_else(|| w.weights.iter().map(|v| v * v).sum());
    let nf = n as f64;
    let d = if between_q { nf } else { nf * (t as f64 - 1.0) };
    let g_mat = if between_q {
        let rows = [
            [g11, g12, 0.0, nf],
            [g21, g22, 0.0, trace_w2],
            [g31, g32, 0.0, 0.0],
        ];
        Mat::from_fn(3, 4, |i, j| rows[i][j] / d)
    } else {
        let rows = [
            [g11, g12, nf * (t as f64 - 1.0)],
            [g21, g22, (t as f64 - 1.0) * trace_w2],
            [g31, g32, 0.0],
        ];
        Mat::from_fn(3, 3, |i, j| rows[i][j] / d)
    };
    let g = vec![dot(u, &qu) / d, dot(&ub, &qub) / d, dot(u, &qub) / d];
    Ok((Moments { g_mat, g }, trace_w2))
}

/// `Tau` of Kapoor, Kelejian and Prucha (`_get_Tau`).
fn tau(w: &Csr, trace_w2: f64) -> Mat<f64> {
    let n = w.n as f64;
    let ws = Sp::from_csr(w);
    let wt = ws.transpose();
    let wtw = wt.matmul(&ws);
    let t12 = 2.0 * trace_w2 / n;
    let t22: f64 = wtw.entries().map(|(_, _, v)| v * v).sum();
    let t23: f64 = wtw
        .entries()
        .map(|(i, j, v)| v * (ws.get(j, i) + ws.get(i, j)))
        .sum();
    let t33: f64 = ws.entries().map(|(i, j, v)| v * ws.get(j, i)).sum::<f64>()
        + wtw.diag().iter().sum::<f64>();
    let rows = [[2.0 * n, t12, 0.0], [t12, t22, t23], [0.0, t23, t33]];
    Mat::from_fn(3, 3, |i, j| rows[i][j] / n)
}

/// Least squares of `r` on the columns `c` with non-negative coefficients (at most two): the
/// best of every active set, as `(coefficients, sum of squares)`.
fn nnls(c: &[Vec<f64>], r: &[f64]) -> (Vec<f64>, f64) {
    let q = c.len();
    let rss = |s: &[f64]| -> f64 {
        (0..r.len())
            .map(|i| {
                let f: f64 = (0..q).map(|j| c[j][i] * s[j]).sum();
                (r[i] - f) * (r[i] - f)
            })
            .sum()
    };
    let mut best = (vec![0.0; q], rss(&vec![0.0; q]));
    for mask in 1..(1usize << q) {
        let idx: Vec<usize> = (0..q).filter(|j| mask >> j & 1 == 1).collect();
        let sol: Option<Vec<f64>> = if idx.len() == 1 {
            let cc = dot(&c[idx[0]], &c[idx[0]]);
            (cc > 0.0).then(|| vec![dot(&c[idx[0]], r) / cc])
        } else {
            let (a, b) = (&c[idx[0]], &c[idx[1]]);
            let (aa, ab, bb) = (dot(a, a), dot(a, b), dot(b, b));
            let det = aa * bb - ab * ab;
            (det.abs() > 1e-300 * (aa * bb).max(f64::MIN_POSITIVE)).then(|| {
                let (ra, rb) = (dot(a, r), dot(b, r));
                vec![(bb * ra - ab * rb) / det, (aa * rb - ab * ra) / det]
            })
        };
        let Some(sol) = sol else { continue };
        if sol.iter().any(|v| *v < 0.0 || !v.is_finite()) {
            continue;
        }
        let mut s = vec![0.0; q];
        for (k, j) in idx.iter().enumerate() {
            s[*j] = sol[k];
        }
        let obj = rss(&s);
        if obj < best.1 {
            best = (s, obj);
        }
    }
    best
}

/// `lambda` and the variance parameters minimizing `|Ec (g - G v)|^2` with
/// `v = [lambda, lambda^2, sigma...]`, `sigma >= 0`, `lambda` in `[-0.99, 0.99]`
/// (`optim_moments(all_par=True)`), `Ec = chol(inv(vc))'` when `vc` is given. The walk over
/// `lambda` starts at `start`.
fn optimize(mom: &Moments, vc: Option<&Mat<f64>>, start: f64) -> Result<(f64, Vec<f64>), String> {
    let m = mom.g.len();
    let cols = mom.g_mat.ncols();
    let mut gm = mom.g_mat.clone();
    let mut g = Mat::from_fn(m, 1, |i, _| mom.g[i]);
    if let Some(vc) = vc {
        let ec = cholesky(&inverse(vc)?)?.transpose().to_owned();
        gm = &ec * &gm;
        g = &ec * &g;
    }
    let scale = (0..m)
        .flat_map(|i| (0..cols).map(move |j| (i, j)))
        .map(|ij| gm[ij])
        .chain((0..m).map(|i| g[(i, 0)]))
        .fold(f64::INFINITY, f64::min);
    if !scale.is_finite() || scale == 0.0 {
        return Err(
            "the moment equations have a zero scale; the GM estimator is not defined".into(),
        );
    }
    let gv: Vec<f64> = (0..m).map(|i| g[(i, 0)] / scale).collect();
    let col = |j: usize| -> Vec<f64> { (0..m).map(|i| gm[(i, j)] / scale).collect() };
    let (a, b) = (col(0), col(1));
    let sigma_cols: Vec<Vec<f64>> = (2..cols).map(col).collect();
    let profile = |lam: f64| -> (Vec<f64>, f64) {
        let r: Vec<f64> = (0..m)
            .map(|i| gv[i] - a[i] * lam - b[i] * lam * lam)
            .collect();
        nnls(&sigma_cols, &r)
    };
    let lam = minimize_lambda_from(|lam| profile(lam).1, start);
    Ok((lam, profile(lam).0))
}

fn check(y: &[f64], x: &[Vec<f64>], w: &Csr, t: usize) -> Result<usize, String> {
    let n = w.n;
    if t < 2 {
        return Err("a panel needs at least two periods".into());
    }
    if y.len() != n * t {
        return Err(format!(
            "y has {} values but the panel has {n} areas x {t} periods (length mismatch)",
            y.len()
        ));
    }
    if x.is_empty() || x.iter().any(|c| c.len() != n * t) {
        return Err("every x column needs n * t values (length mismatch)".into());
    }
    if n * t <= x.len() {
        return Err("the panel has no degrees of freedom".into());
    }
    Ok(n)
}

/// `GM_ErrorRE`: see the module documentation. `at` replaces the optimizer (test hook).
pub fn gm_kkp(
    y: &[f64],
    x: &[Vec<f64>],
    w: &Csr,
    t: usize,
    full_weights: bool,
    at: Option<KkpAt>,
) -> Result<Entries, String> {
    let n = check(y, x, w, t)?;
    let (nf, tf, nt, k) = (n as f64, t as f64, n * t, x.len());

    // 1. OLS and the within moments.
    let ols = fit(y, x)?;
    let (m0, trace_w2) = moments_kkp(w, &ols.u, t, false, None)?;
    let (lam1, sig_v) = match at {
        Some(a) => (a[0], a[1]),
        None => {
            let (lam, s) = optimize(&m0, None, 0.0)?;
            (lam, s[0])
        }
    };
    let ub = lag_nt(w, &ols.u, n)?;
    let ulu: Vec<f64> = ols.u.iter().zip(&ub).map(|(a, b)| a - lam1 * b).collect();
    let q1ulu = between(&ulu, n, t);
    let sig_1 = dot(&ulu, &q1ulu) / nf;

    // 2. All six moments, weighted by Xi.
    let tau = if full_weights {
        tau(w, trace_w2)
    } else {
        Mat::from_fn(3, 3, |i, j| f64::from(u8::from(i == j)))
    };
    let (a_v, a_1) = (sig_v * sig_v / (tf - 1.0), sig_1 * sig_1);
    let xi = Mat::from_fn(6, 6, |i, j| {
        if i / 3 == j / 3 {
            [a_v, a_1][i / 3] * tau[(i % 3, j % 3)]
        } else {
            0.0
        }
    });
    let (mb, _) = moments_kkp(w, &ols.u, t, true, Some(trace_w2))?;
    let g6 = Moments {
        g_mat: Mat::from_fn(6, 4, |i, j| {
            if i < 3 {
                if j < 3 {
                    m0.g_mat[(i, j)]
                } else {
                    0.0
                }
            } else {
                mb.g_mat[(i - 3, j)]
            }
        }),
        g: m0.g.iter().chain(&mb.g).copied().collect(),
    };
    let (lam, sig_vb, sig_1b) = match at {
        Some(a) => (a[2], a[3], a[4]),
        None => {
            let (lam, s) = optimize(&g6, Some(&xi), lam1)?;
            (lam, s[0], s[1])
        }
    };

    // 3. Feasible GLS.
    let theta = 1.0 - sig_vb.sqrt() / sig_1b.sqrt();
    let transform = |v: &[f64]| -> Result<Vec<f64>, String> {
        let wv = lag_nt(w, v, n)?;
        let filtered: Vec<f64> = v.iter().zip(&wv).map(|(a, b)| a - lam * b).collect();
        Ok(quasi_demean(&filtered, n, t, 1.0 - theta))
    };
    let xs: Vec<Vec<f64>> = x.iter().map(|c| transform(c)).collect::<Result<_, _>>()?;
    let ys = transform(y)?;
    let gls = fit(&ys, &xs)?;
    let s2 = gls.utu / (nt - k) as f64;
    let vm = Mat::from_fn(k, k, |i, j| s2 * gls.xtxi[(i, j)]);
    let predy = fitted(x, &gls.betas);
    let u: Vec<f64> = y.iter().zip(&predy).map(|(a, b)| a - b).collect();
    let wu = lag_nt(w, &u, n)?;
    let e_filtered: Vec<f64> = u.iter().zip(&wu).map(|(a, b)| a - lam * b).collect();
    let std_err: Vec<f64> = (0..k).map(|i| vm[(i, i)].sqrt()).collect();
    let mut betas = gls.betas.clone();
    betas.extend([lam, sig_vb, sig_1b]);

    let mut out = Entries::new();
    out.insert("lam", vec![lam]);
    out.insert("sig2_v", vec![sig_vb]);
    out.insert("sig2_1", vec![sig_1b]);
    out.insert("theta", vec![theta]);
    out.insert("first_step", vec![lam1, sig_v, sig_1]);
    out.insert("vm", row_major(&vm));
    out.insert("z_stat", z_stats(&gls.betas, &std_err));
    out.insert("std_err", std_err);
    out.insert("pr2", vec![pearson_r(y, &predy).powi(2)]);
    out.insert("u", u);
    out.insert("predy", predy);
    out.insert("e_filtered", e_filtered);
    out.insert("betas", betas);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn csr(n: usize, edges: &[(u32, u32, f64)]) -> Csr {
        let f: Vec<u32> = edges.iter().map(|e| e.0).collect();
        let nb: Vec<u32> = edges.iter().map(|e| e.1).collect();
        let wt: Vec<f64> = edges.iter().map(|e| e.2).collect();
        Csr::from_arrays(n, &f, &nb, &wt).unwrap()
    }

    fn path3() -> Csr {
        csr(3, &[(0, 1, 1.0), (1, 0, 0.5), (1, 2, 0.5), (2, 1, 1.0)])
    }

    fn ring(n: usize) -> Csr {
        let mut e = Vec::new();
        for i in 0..n as u32 {
            e.push((i, (i + 1) % n as u32, 0.5));
            e.push((i, (i + n as u32 - 1) % n as u32, 0.5));
        }
        csr(n, &e)
    }

    /// Deterministic pseudo-random numbers in `[-0.5, 0.5)`.
    fn noise(seed: u64) -> impl FnMut() -> f64 {
        let mut s = seed;
        move || {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f64) / (1u64 << 31) as f64 - 0.5
        }
    }

    /// A random-effects spatial error panel on a ring: returns `(y, [1, x1])`.
    fn panel(n: usize, t: usize, lam: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
        let mut next = noise(11);
        let w = ring(n);
        let x1: Vec<f64> = (0..n * t).map(|_| next() * 4.0).collect();
        let mu: Vec<f64> = (0..n).map(|_| next() * 3.0).collect();
        let e: Vec<f64> = (0..n * t).map(|_| next() * 2.0).collect();
        // u = (I - lam W)^-1 e by the power series.
        let mut u = e.clone();
        let mut term = e;
        for _ in 0..200 {
            term = lag_nt(&w, &term, n)
                .unwrap()
                .into_iter()
                .map(|v| v * lam)
                .collect();
            for (a, b) in u.iter_mut().zip(&term) {
                *a += b;
            }
        }
        let y: Vec<f64> = (0..n * t)
            .map(|r| 1.0 + 2.0 * x1[r] + mu[r % n] + u[r])
            .collect();
        (y, vec![vec![1.0; n * t], x1])
    }

    #[test]
    fn moments_of_an_empty_graph_are_the_variance_equations() {
        let w = csr(3, &[]);
        let u = [1.0, 2.0, 3.0, 2.0, 0.0, 5.0];
        let (within, trace) = moments_kkp(&w, &u, 2, false, None).unwrap();
        assert_eq!(trace, 0.0);
        // Q0 u = [-.5, 1, -1, .5, -1, 1]: u'Q0u = 4.5, N (T - 1) = 3.
        assert_eq!(within.g, vec![1.5, 0.0, 0.0]);
        assert_eq!((within.g_mat.nrows(), within.g_mat.ncols()), (3, 3));
        for (i, j, v) in [(0, 2, 1.0), (0, 0, 0.0), (1, 2, 0.0), (2, 2, 0.0)] {
            assert!((within.g_mat[(i, j)] - v).abs() < 1e-15, "G[{i},{j}]");
        }
        // Q1 u = area means [1.5, 1, 4] twice: u'Q1u = 38.5, divided by N = 3.
        let (bet, _) = moments_kkp(&w, &u, 2, true, Some(trace)).unwrap();
        assert!((bet.g[0] - 38.5 / 3.0).abs() < 1e-14);
        assert_eq!((bet.g_mat.nrows(), bet.g_mat.ncols()), (3, 4));
        assert_eq!(bet.g_mat[(0, 3)], 1.0);
        assert_eq!(bet.g_mat[(0, 2)], 0.0);
    }

    #[test]
    fn within_moments_ignore_area_effects_and_trace_is_the_squared_weights() {
        let w = path3();
        let u = [1.0, 2.0, 3.0, 2.0, 0.0, 5.0, -1.0, 1.0, 0.5];
        let shifted: Vec<f64> = u
            .iter()
            .enumerate()
            .map(|(e, v)| v + [3.0, -2.0, 7.0][e % 3])
            .collect();
        let (a, trace) = moments_kkp(&w, &u, 3, false, None).unwrap();
        let (b, _) = moments_kkp(&w, &shifted, 3, false, None).unwrap();
        assert!((trace - 2.5).abs() < 1e-15);
        for i in 0..3 {
            assert!((a.g[i] - b.g[i]).abs() < 1e-12, "g[{i}]");
            for j in 0..3 {
                assert!(
                    (a.g_mat[(i, j)] - b.g_mat[(i, j)]).abs() < 1e-12,
                    "G[{i},{j}]"
                );
            }
        }
        assert!((a.g_mat[(1, 2)] - 2.5 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn tau_of_a_path_is_the_hand_computed_matrix() {
        // W = [[0,1,0],[.5,0,.5],[0,1,0]]: sum w^2 = 2.5, W'W = [[.25,0,.25],[0,2,0],[.25,0,.25]],
        // sum (W'W)^2 = 4.25, sum (W'W)(W' + W) = 0, sum w_ij w_ji + tr(W'W) = 2 + 2.5.
        let t = tau(&path3(), 2.5);
        let want = [
            [2.0, 5.0 / 9.0, 0.0],
            [5.0 / 9.0, 4.25 / 3.0, 0.0],
            [0.0, 0.0, 1.5],
        ];
        for i in 0..3 {
            for j in 0..3 {
                assert!((t[(i, j)] - want[i][j]).abs() < 1e-14, "Tau[{i},{j}]");
            }
        }
    }

    /// `|g - G v|^2` for `v = [lambda, lambda^2, sigma...]`.
    fn objective(mom: &Moments, lam: f64, sigma: &[f64]) -> f64 {
        (0..mom.g.len())
            .map(|i| {
                let fit = mom.g_mat[(i, 0)] * lam
                    + mom.g_mat[(i, 1)] * lam * lam
                    + sigma
                        .iter()
                        .enumerate()
                        .map(|(j, s)| mom.g_mat[(i, 2 + j)] * s)
                        .sum::<f64>();
                (mom.g[i] - fit) * (mom.g[i] - fit)
            })
            .sum()
    }

    #[test]
    fn the_optimizer_beats_a_grid_search_and_keeps_variances_non_negative() {
        let n = 12;
        let t = 4;
        let (y, x) = panel(n, t, 0.4);
        let w = ring(n);
        let u = fit(&y, &x).unwrap().u;
        let (m0, trace) = moments_kkp(&w, &u, t, false, None).unwrap();
        let (lam1, s1) = optimize(&m0, None, 0.0).unwrap();
        assert!(s1.len() == 1 && s1[0] >= 0.0 && lam1.abs() <= 0.99);
        let (mb, _) = moments_kkp(&w, &u, t, true, Some(trace)).unwrap();
        let g6 = Moments {
            g_mat: Mat::from_fn(6, 4, |i, j| match (i < 3, j < 3) {
                (true, true) => m0.g_mat[(i, j)],
                (true, false) => 0.0,
                _ => mb.g_mat[(i - 3, j)],
            }),
            g: m0.g.iter().chain(&mb.g).copied().collect(),
        };
        let (lam2, s2) = optimize(&g6, None, lam1).unwrap();
        assert!(s2.len() == 2 && s2.iter().all(|v| *v >= 0.0) && lam2.abs() <= 0.99);
        // Every grid point is feasible, so none can beat the optimizer.
        let best = objective(&g6, lam2, &s2);
        for l in -19..=19 {
            for a in 0..=100 {
                for b in 0..=100 {
                    let grid = objective(
                        &g6,
                        f64::from(l) * 0.05,
                        &[f64::from(a) * 0.3, f64::from(b) * 0.3],
                    );
                    assert!(best <= grid + 1e-12, "{best} > {grid} at {l} {a} {b}");
                }
            }
        }
    }

    #[test]
    fn spreg_parameters_with_an_empty_graph_give_the_quasi_demeaned_ols() {
        let (n, t) = (6, 4);
        let (y, x) = panel(n, t, 0.0);
        let w = csr(n, &[]);
        let at: KkpAt = [0.0, 2.0, 0.0, 2.0, 5.0];
        let r = gm_kkp(&y, &x, &w, t, false, Some(at)).unwrap();
        let theta = 1.0 - (2.0f64).sqrt() / (5.0f64).sqrt();
        assert!((r["theta"][0] - theta).abs() < 1e-14);
        let ys = quasi_demean(&y, n, t, 1.0 - theta);
        let xs: Vec<Vec<f64>> = x
            .iter()
            .map(|c| quasi_demean(c, n, t, 1.0 - theta))
            .collect();
        let direct = fit(&ys, &xs).unwrap();
        for (a, b) in r["betas"].iter().zip(&direct.betas) {
            assert!((a - b).abs() < 1e-10, "{a} {b}");
        }
        assert_eq!(&r["betas"][2..], &[0.0, 2.0, 5.0]);
        let s2 = direct.utu / (n * t - 2) as f64;
        assert!((r["vm"][0] - s2 * direct.xtxi[(0, 0)]).abs() < 1e-10);
        assert_eq!(r["vm"].len(), 4);
        assert_eq!(r["u"].len(), n * t);
    }

    #[test]
    fn a_fit_reports_consistent_spreg_quantities_for_both_weightings() {
        let (n, t) = (30, 4);
        let (y, x) = panel(n, t, 0.5);
        let w = ring(n);
        let plain = gm_kkp(&y, &x, &w, t, false, None).unwrap();
        let full = gm_kkp(&y, &x, &w, t, true, None).unwrap();
        for r in [&plain, &full] {
            let (lam, sv, s1) = (r["lam"][0], r["sig2_v"][0], r["sig2_1"][0]);
            assert!(lam.abs() <= 0.99 && sv >= 0.0 && s1 >= 0.0);
            assert_eq!(r["betas"][2..], [lam, sv, s1]);
            assert_eq!(r["first_step"].len(), 3);
            assert_eq!(r["std_err"].len(), 2);
            assert_eq!(r["z_stat"].len(), 4);
            assert!((0.0..=1.0).contains(&r["pr2"][0]));
            // u is y - X b on the raw data and e_filtered its spatial filter.
            let b = &r["betas"][..2];
            for e in 0..n * t {
                assert!((r["u"][e] - (y[e] - b[0] - b[1] * x[1][e])).abs() < 1e-10);
            }
            assert!((r["theta"][0] - (1.0 - (sv / s1).sqrt())).abs() < 1e-14);
        }
        assert!(
            (plain["lam"][0] - full["lam"][0]).abs() > 1e-9,
            "weights must matter"
        );
    }

    #[test]
    fn a_zero_moment_scale_is_an_error() {
        // y = 0 gives zero residuals: the smallest entry of the moment system is exactly zero.
        let w = ring(5);
        let y = vec![0.0; 15];
        let x = vec![vec![1.0; 15]];
        let err = gm_kkp(&y, &x, &w, 3, false, None).unwrap_err();
        assert!(err.contains("zero scale"), "{err}");
    }

    #[test]
    fn inputs_are_validated() {
        let w = ring(5);
        let (y, x) = panel(5, 3, 0.1);
        assert!(gm_kkp(&y, &x, &w, 1, false, None)
            .unwrap_err()
            .contains("two periods"));
        assert!(gm_kkp(&y[1..], &x, &w, 3, false, None)
            .unwrap_err()
            .contains("length mismatch"));
        assert!(gm_kkp(&y, &[], &w, 3, false, None)
            .unwrap_err()
            .contains("length mismatch"));
    }
}
