//! Pieces shared by the maximum-likelihood spatial lag and error models: the log-Jacobian
//! `ln|I - rho W|` for each estimation method, the trace terms of the asymptotic variance, the
//! power-series inverse product, Pearson correlation and the spatial impact multipliers.
//!
//! Everything mirrors spreg 1.9.1 (`ml_lag.py`, `ml_error.py`, `utils.power_expansion`,
//! `sputils.spmultiplier`) except where a documented deviation applies (see `ml_lag`).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::factor::{lu_in_place, lu_in_place_scratch};
use faer::{Mat, Par};

use super::linalg::{entry, inverse, mean, row_range, symmetric_eigenvalues};
use crate::graph::Csr;

/// Default cap on `n` for the estimators that build dense `n x n` matrices.
pub const DEFAULT_MAX_DENSE_N: usize = 5_000;

/// How the log-Jacobian is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Dense LU determinant at every evaluation (spreg `'full'`).
    Full,
    /// Ord (1975) eigenvalue method: eigenvalues once, `O(n)` per evaluation.
    Ord,
    /// spreg's sparse LU. Spatex computes it exactly like [`Method::Full`] (same optimum to
    /// rounding; only the sign check of the determinant differs).
    Lu,
}

impl Method {
    /// Decodes the NIF code (0 full, 1 ord, 2 lu).
    pub fn from_code(code: u8) -> Result<Method, String> {
        match code {
            0 => Ok(Method::Full),
            1 => Ok(Method::Ord),
            2 => Ok(Method::Lu),
            other => Err(format!("unknown method code {other}")),
        }
    }
}

/// Which spatial multipliers to compute (`spat_impacts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Impacts {
    None,
    Simple,
    Full,
    Power,
    All,
}

impl Impacts {
    /// Decodes the NIF code (0 none, 1 simple, 2 full, 3 power, 4 all).
    pub fn from_code(code: u8) -> Result<Impacts, String> {
        match code {
            0 => Ok(Impacts::None),
            1 => Ok(Impacts::Simple),
            2 => Ok(Impacts::Full),
            3 => Ok(Impacts::Power),
            4 => Ok(Impacts::All),
            other => Err(format!("unknown impacts code {other}")),
        }
    }

    fn methods(self) -> &'static [&'static str] {
        match self {
            Impacts::None => &[],
            Impacts::Simple => &["simple"],
            Impacts::Full => &["full"],
            Impacts::Power => &["power"],
            Impacts::All => &["simple", "full", "power"],
        }
    }
}

/// Checks the weights against the data and the dense-size guard shared by all methods.
pub fn check_inputs(w: &Csr, n: usize, epsilon: f64, max_dense_n: usize) -> Result<(), String> {
    if w.n != n {
        return Err(format!("graph has {} nodes but y has {n} rows", w.n));
    }
    if epsilon.is_nan() || epsilon <= 0.0 {
        return Err("epsilon must be positive".to_string());
    }
    if n > max_dense_n {
        return Err(format!(
            "n = {n} exceeds max_dense_n = {max_dense_n}: the ML estimators build dense n x n \
             matrices (about {} MiB each); raise :max_dense_n if you have the memory",
            n * n * 8 / (1 << 20)
        ));
    }
    for i in 0..n {
        for k in row_range(w, i) {
            if w.neighbors[k] as usize == i && w.weights[k] != 0.0 {
                return Err("the weights must have a zero diagonal".to_string());
            }
        }
    }
    Ok(())
}

/// The weights as a dense matrix.
pub fn dense(w: &Csr) -> Mat<f64> {
    let mut m = Mat::zeros(w.n, w.n);
    for i in 0..w.n {
        for k in row_range(w, i) {
            m[(i, w.neighbors[k] as usize)] += w.weights[k];
        }
    }
    m
}

/// Eigenvalues of `W`.
#[derive(Clone, Debug)]
pub enum Spectrum {
    /// All real (symmetric case): `ln(1 - rho * lambda)` is NaN when its argument is negative.
    Real(Vec<f64>),
    /// Some complex pairs: the real part of the complex logarithm is `ln|1 - rho * lambda|`.
    Complex(Vec<(f64, f64)>),
}

/// Whether every nonzero weight equals `1 / card(i)` and the pattern is symmetric: the structure
/// for which `D^{-1/2} B D^{-1/2}` is a symmetric matrix similar to `W` (spreg's `symmetrize`).
fn row_standardized_symmetric(w: &Csr) -> bool {
    for i in 0..w.n {
        let card = row_range(w, i).filter(|&k| w.weights[k] != 0.0).count();
        for k in row_range(w, i) {
            let wij = w.weights[k];
            if wij == 0.0 {
                continue;
            }
            let j = w.neighbors[k] as usize;
            if (wij - 1.0 / card as f64).abs() > 1e-12 || entry(w, j, i) == 0.0 {
                return false;
            }
        }
    }
    true
}

/// The eigenvalues Ord's method needs.
///
/// A row-standardized symmetric pattern goes through the real symmetric similar matrix (as spreg
/// does); anything else uses the general eigenvalues of `W` itself. Unlike spreg, the symmetrized
/// matrix is only used for its eigenvalues, and only when it is genuinely similar to `W`.
pub fn spectrum(w: &Csr) -> Result<Spectrum, String> {
    if row_standardized_symmetric(w) {
        let card: Vec<f64> = (0..w.n)
            .map(|i| row_range(w, i).filter(|&k| w.weights[k] != 0.0).count() as f64)
            .collect();
        let mut s = Mat::zeros(w.n, w.n);
        for i in 0..w.n {
            for k in row_range(w, i) {
                if w.weights[k] != 0.0 {
                    let j = w.neighbors[k] as usize;
                    s[(i, j)] = 1.0 / (card[i] * card[j]).sqrt();
                }
            }
        }
        return Ok(Spectrum::Real(symmetric_eigenvalues(&s)?));
    }
    let ev = dense(w)
        .eigenvalues()
        .map_err(|e| format!("eigenvalue decomposition failed: {e:?}"))?;
    if ev.iter().all(|z| z.im == 0.0) {
        Ok(Spectrum::Real(ev.iter().map(|z| z.re).collect()))
    } else {
        Ok(Spectrum::Complex(ev.iter().map(|z| (z.re, z.im)).collect()))
    }
}

/// Log-Jacobian evaluator for one weights matrix and method.
pub enum Jacobian {
    Dense(Mat<f64>),
    Eigen(Spectrum),
}

impl Jacobian {
    pub fn new(w: &Csr, method: Method) -> Result<Jacobian, String> {
        match method {
            Method::Full | Method::Lu => Ok(Jacobian::Dense(dense(w))),
            Method::Ord => Ok(Jacobian::Eigen(spectrum(w)?)),
        }
    }

    /// `ln|det(I - rho W)|`, spreg's SUR-error Jacobian (`sum(log(abs(diag(U))))`): finite for a
    /// negative determinant, `-inf` for a singular matrix.
    pub fn log_abs_det(&self, rho: f64) -> f64 {
        match self {
            Jacobian::Dense(w) => {
                let n = w.nrows();
                let a = Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { -rho * w[(i, j)] });
                log_abs_det_lu(a).0
            }
            Jacobian::Eigen(Spectrum::Real(ev)) => {
                ev.iter().map(|l| (1.0 - rho * l).abs().ln()).sum()
            }
            Jacobian::Eigen(Spectrum::Complex(ev)) => ev
                .iter()
                .map(|(re, im)| (1.0 - rho * re).hypot(-rho * im).ln())
                .sum(),
        }
    }

    /// `ln det(I - rho W)`. Like `np.log(np.linalg.det(.))` it is NaN for a negative determinant
    /// and `-inf` for a singular matrix; unlike numpy it is a sum of logs, so it cannot overflow.
    pub fn log_det(&self, rho: f64) -> f64 {
        match self {
            Jacobian::Dense(w) => {
                let n = w.nrows();
                let a = Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { -rho * w[(i, j)] });
                log_det_lu(a)
            }
            Jacobian::Eigen(Spectrum::Real(ev)) => ev.iter().map(|l| (1.0 - rho * l).ln()).sum(),
            Jacobian::Eigen(Spectrum::Complex(ev)) => ev
                .iter()
                .map(|(re, im)| (1.0 - rho * re).hypot(-rho * im).ln())
                .sum(),
        }
    }
}

/// `ln det(A)` by partial-pivot LU (serial, so results do not depend on the thread count).
fn log_det_lu(a: Mat<f64>) -> f64 {
    let (sum, negative) = log_abs_det_lu(a);
    if negative {
        f64::NAN
    } else {
        sum
    }
}

/// `ln|det(A)|` by partial-pivot LU (the sum of `ln|u_ii|`, `-inf` when singular) and whether the
/// determinant is negative.
pub fn log_abs_det_lu(mut a: Mat<f64>) -> (f64, bool) {
    let n = a.nrows();
    if n == 0 {
        return (0.0, false);
    }
    let mut fwd = vec![0usize; n];
    let mut bwd = vec![0usize; n];
    let par = Par::Seq;
    let mut buffer = MemBuffer::new(lu_in_place_scratch::<usize, f64>(
        n,
        n,
        par,
        Default::default(),
    ));
    lu_in_place(
        a.as_mut(),
        &mut fwd,
        &mut bwd,
        par,
        MemStack::new(&mut buffer),
        Default::default(),
    );
    // Sign of the row permutation: (-1)^(n - cycles).
    let mut seen = vec![false; n];
    let mut cycles = 0usize;
    for start in 0..n {
        if !seen[start] {
            cycles += 1;
            let mut j = start;
            while !seen[j] {
                seen[j] = true;
                j = fwd[j];
            }
        }
    }
    let mut negative = (n - cycles) % 2 == 1;
    let mut sum = 0.0;
    for i in 0..n {
        let d = a[(i, i)];
        if d == 0.0 {
            return (f64::NEG_INFINITY, false);
        }
        negative ^= d < 0.0;
        sum += d.abs().ln();
    }
    (sum, negative)
}

/// Trace terms of the asymptotic variance with `WAi = W (I - rho W)^{-1}`.
#[derive(Clone, Copy, Debug)]
pub struct Traces {
    pub tr1: f64,
    pub tr2: f64,
    pub tr3: f64,
}

/// `tr(WAi)`, `tr(WAi WAi)` and `tr(WAi' WAi)`, the last two as `sum_ij WAi_ij WAi_ji` and
/// `sum_ij WAi_ij^2` (no `n^3` products).
pub fn traces(w: &Csr, rho: f64) -> Result<Traces, String> {
    let n = w.n;
    let wai = wai(w, rho)?;
    let mut tr1 = 0.0;
    let mut tr2 = 0.0;
    let mut tr3 = 0.0;
    for i in 0..n {
        tr1 += wai[(i, i)];
        for j in 0..n {
            let v = wai[(i, j)];
            tr3 += v * v;
            tr2 += v * wai[(j, i)];
        }
    }
    Ok(Traces { tr1, tr2, tr3 })
}

/// `W (I - rho W)^{-1}` as a dense matrix.
pub fn wai(w: &Csr, rho: f64) -> Result<Mat<f64>, String> {
    let n = w.n;
    let a = Mat::from_fn(n, n, |i, j| if i == j { 1.0 } else { 0.0 });
    let mut a = a;
    for i in 0..n {
        for k in row_range(w, i) {
            a[(i, w.neighbors[k] as usize)] -= rho * w.weights[k];
        }
    }
    let ai = inverse(&a)?;
    drop(a);
    Ok(sparse_times_dense(w, &ai))
}

/// `W M` for a CSR `W` and a dense `M`, column by column.
fn sparse_times_dense(w: &Csr, m: &Mat<f64>) -> Mat<f64> {
    let n = w.n;
    let mut out = Mat::zeros(n, m.ncols());
    for j in 0..m.ncols() {
        for i in 0..n {
            let mut s = 0.0;
            for k in row_range(w, i) {
                s += w.weights[k] * m[(w.neighbors[k] as usize, j)];
            }
            out[(i, j)] = s;
        }
    }
    out
}

/// `(I - scalar W)^{-1} data` by the power series, literally as `utils.power_expansion` does:
/// stop when the Euclidean norm of the increment is `<= threshold`, fail when it grows.
pub fn power_expansion(
    w: &Csr,
    data: &[f64],
    scalar: f64,
    threshold: f64,
) -> Result<Vec<f64>, String> {
    let mut total = data.to_vec();
    let mut inc = data.to_vec();
    let mut test = 1.0e7_f64;
    let mut count = 1u64;
    while test > threshold && count <= 10_000_000 {
        inc = w.lag(&inc)?.into_iter().map(|v| v * scalar).collect();
        for (t, d) in total.iter_mut().zip(&inc) {
            *t += d;
        }
        let test_old = test;
        test = inc.iter().map(|v| v * v).sum::<f64>().sqrt();
        if test > test_old {
            return Err("power expansion will not converge, check model specification and that weights are less than 1".to_string());
        }
        count += 1;
    }
    Ok(total)
}

/// Pearson correlation, as `scipy.stats.pearsonr` (centered vectors normalized before the dot
/// product, clipped to `[-1, 1]`).
pub fn pearson_r(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let ca: Vec<f64> = a.iter().map(|v| v - ma).collect();
    let cb: Vec<f64> = b.iter().map(|v| v - mb).collect();
    let na = ca.iter().map(|v| v * v).sum::<f64>().sqrt();
    let nb = cb.iter().map(|v| v * v).sum::<f64>().sqrt();
    let r: f64 = ca.iter().zip(&cb).map(|(p, q)| (p / na) * (q / nb)).sum();
    r.clamp(-1.0, 1.0)
}

/// Sample standard deviation (`ddof = 1`).
pub fn std_dev(v: &[f64]) -> f64 {
    let m = mean(v);
    (v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (v.len() as f64 - 1.0)).sqrt()
}

/// Average direct, indirect and total impact multipliers (`sputils.spmultiplier`).
#[derive(Clone, Copy, Debug)]
pub struct Multipliers {
    pub adi: f64,
    pub aii: f64,
    pub ati: f64,
    /// Powers used by the power approximation (0 otherwise).
    pub pow: usize,
}

/// Multipliers for one method name (`"simple"`, `"full"` or `"power"`).
pub fn multipliers(w: &Csr, rho: f64, method: &str) -> Result<Multipliers, String> {
    let n = w.n as f64;
    let ati = 1.0 / (1.0 - rho);
    let mut adi = 1.0;
    let mut pow = 0usize;
    match method {
        "simple" => {}
        "full" => {
            let mut a = dense(w);
            for i in 0..w.n {
                for j in 0..w.n {
                    a[(i, j)] = f64::from(u8::from(i == j)) - rho * a[(i, j)];
                }
            }
            let inv = inverse(&a)?;
            adi = (0..w.n).map(|i| inv[(i, i)]).sum::<f64>() / n;
        }
        "power" => {
            let mut ww = dense(w);
            let mut rhop = rho;
            pow = 1;
            let mut diff = 100.0_f64;
            while diff > 1e-8 {
                pow += 1;
                ww = sparse_times_dense(w, &ww);
                let trw: f64 = (0..w.n).map(|i| ww[(i, i)]).sum();
                rhop *= rho;
                diff = rhop * trw / n;
                adi += diff;
            }
        }
        other => return Err(format!("unknown impacts method {other}")),
    }
    Ok(Multipliers {
        adi,
        aii: ati - adi,
        ati,
        pow,
    })
}

/// Names of the multiplier methods an [`Impacts`] choice computes.
pub fn impact_methods(impacts: Impacts) -> &'static [&'static str] {
    impacts.methods()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 4-cycle, row-standardized: symmetric pattern, eigenvalues 1, 0, 0, -1.
    fn cycle4() -> Csr {
        let f = [0u32, 0, 1, 1, 2, 2, 3, 3];
        let t = [1u32, 3, 0, 2, 1, 3, 2, 0];
        Csr::from_arrays(4, &f, &t, &[0.5; 8]).unwrap()
    }

    /// Directed 3-cycle: a non-symmetric pattern with complex eigenvalues.
    fn directed3() -> Csr {
        Csr::from_arrays(3, &[0, 1, 2], &[1, 2, 0], &[1.0, 1.0, 1.0]).unwrap()
    }

    #[test]
    fn log_det_matches_eigenvalue_formula() {
        let w = cycle4();
        for rho in [-0.7, 0.0, 0.3, 0.9] {
            let full = Jacobian::new(&w, Method::Full).unwrap().log_det(rho);
            let ord = Jacobian::new(&w, Method::Ord).unwrap().log_det(rho);
            // det(I - rho W) = (1 - rho)(1 + rho) for eigenvalues 1, 0, 0, -1.
            let exact = ((1.0 - rho) * (1.0 + rho)).ln();
            assert!((full - exact).abs() < 1e-13, "{rho}: {full} vs {exact}");
            assert!((ord - exact).abs() < 1e-13, "{rho}: {ord} vs {exact}");
        }
    }

    #[test]
    fn complex_spectrum_matches_dense_determinant() {
        let w = directed3();
        // det(I - rho W) = 1 - rho^3 for the directed 3-cycle.
        assert!(matches!(spectrum(&w).unwrap(), Spectrum::Complex(_)));
        for rho in [-0.8_f64, 0.4] {
            let exact = (1.0 - rho * rho * rho).ln();
            let full = Jacobian::new(&w, Method::Full).unwrap().log_det(rho);
            let ord = Jacobian::new(&w, Method::Ord).unwrap().log_det(rho);
            assert!((full - exact).abs() < 1e-13 && (ord - exact).abs() < 1e-13);
        }
    }

    #[test]
    fn negative_determinant_is_nan_and_singular_is_neg_infinity() {
        let w = cycle4();
        let full = Jacobian::new(&w, Method::Full).unwrap();
        assert!(full.log_det(2.0).is_nan()); // (1 - 2)(1 + 2) < 0
                                             // rho = 1 makes I - W singular: the pivot is zero or rounding noise.
        assert!(full.log_det(1.0) < -30.0);
    }

    #[test]
    fn non_binary_symmetric_weights_use_the_general_spectrum() {
        // Symmetric pattern, weights not 1/card: PySAL would symmetrize the binary pattern.
        let w = Csr::from_arrays(3, &[0, 0, 1, 2], &[1, 2, 0, 0], &[0.8, 0.2, 1.0, 1.0]).unwrap();
        assert!(!row_standardized_symmetric(&w));
        let full = Jacobian::new(&w, Method::Full).unwrap().log_det(0.4);
        let ord = Jacobian::new(&w, Method::Ord).unwrap().log_det(0.4);
        assert!((full - ord).abs() < 1e-12);
    }

    #[test]
    fn traces_match_closed_form_on_a_two_node_graph() {
        // W = [[0, 1], [1, 0]]: WAi = W (I - rho W)^-1 has diagonal rho/(1 - rho^2).
        let w = Csr::from_arrays(2, &[0, 1], &[1, 0], &[1.0, 1.0]).unwrap();
        let rho = 0.5;
        let t = traces(&w, rho).unwrap();
        let d = rho / (1.0 - rho * rho);
        let o = 1.0 / (1.0 - rho * rho);
        assert!((t.tr1 - 2.0 * d).abs() < 1e-14);
        assert!((t.tr2 - (2.0 * d * d + 2.0 * o * o)).abs() < 1e-13);
        assert!((t.tr3 - (2.0 * d * d + 2.0 * o * o)).abs() < 1e-13);
    }

    #[test]
    fn power_expansion_approximates_the_inverse_and_detects_divergence() {
        let w = cycle4();
        let x = [1.0, 2.0, 3.0, 4.0];
        let got = power_expansion(&w, &x, 0.5, 1e-12).unwrap();
        // (I - 0.5 W)^-1 x solves (I - 0.5 W) z = x.
        let lagged = w.lag(&got).unwrap();
        for i in 0..4 {
            assert!((got[i] - 0.5 * lagged[i] - x[i]).abs() < 1e-9);
        }
        assert!(power_expansion(&w, &x, 1.5, 1e-7)
            .unwrap_err()
            .contains("will not converge"));
    }

    #[test]
    fn pearson_and_std() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson_r(&a, &[2.0, 4.0, 6.0, 8.0]) - 1.0).abs() < 1e-15);
        assert!((pearson_r(&a, &[4.0, 3.0, 2.0, 1.0]) + 1.0).abs() < 1e-15);
        assert!((std_dev(&a) - (5.0_f64 / 3.0).sqrt()).abs() < 1e-15);
    }

    #[test]
    fn multipliers_simple_full_power_agree_for_a_regular_graph() {
        // K4 row-standardized (odd closed walks, so every power contributes).
        let (mut f, mut t) = (vec![], vec![]);
        for i in 0..4u32 {
            for j in 0..4u32 {
                if i != j {
                    f.push(i);
                    t.push(j);
                }
            }
        }
        let w = Csr::from_arrays(4, &f, &t, &[1.0 / 3.0; 12]).unwrap();
        let rho = 0.3;
        let simple = multipliers(&w, rho, "simple").unwrap();
        assert_eq!((simple.adi, simple.pow), (1.0, 0));
        assert!((simple.ati - 1.0 / 0.7).abs() < 1e-15);
        let full = multipliers(&w, rho, "full").unwrap();
        let power = multipliers(&w, rho, "power").unwrap();
        assert!((full.adi - power.adi).abs() < 1e-7);
        assert!(power.pow >= 2);
        assert!((full.adi + full.aii - full.ati).abs() < 1e-15);
        assert!(multipliers(&w, rho, "bogus").is_err());
    }

    #[test]
    fn input_checks() {
        let w = cycle4();
        assert!(check_inputs(&w, 4, 1e-7, 10).is_ok());
        assert!(check_inputs(&w, 5, 1e-7, 10)
            .unwrap_err()
            .contains("4 nodes"));
        assert!(check_inputs(&w, 4, 0.0, 10)
            .unwrap_err()
            .contains("epsilon"));
        assert!(check_inputs(&w, 4, 1e-7, 3)
            .unwrap_err()
            .contains("max_dense_n"));
        let diag = Csr::from_arrays(2, &[0, 1], &[0, 1], &[1.0, 1.0]).unwrap();
        assert!(check_inputs(&diag, 2, 1e-7, 10)
            .unwrap_err()
            .contains("zero diagonal"));
    }
}
