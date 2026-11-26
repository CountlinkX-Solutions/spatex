//! Bandwidth selection (`mgwr.sel_bw.Sel_BW`, `mgwr.search`): golden section and equal-interval
//! searches over a GWR information criterion.

use std::collections::HashMap;

use rayon::prelude::*;

use super::fit::fit_lite;
use super::kernel::distances;
use super::Model;
use crate::regression::brent::minimize_bounded;

/// `minimize_scalar(method='bounded')` defaults: `xatol = 1e-5`, at most 500 evaluations.
const SCIPY_XATOL: f64 = 1e-5;
const SCIPY_MAXFUN: usize = 500;

/// Golden-section step used by mgwr (a literal, not `(3 - sqrt 5) / 2`).
const DELTA: f64 = 0.381_97;

/// Criterion minimized by the search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Criterion {
    Aicc,
    Aic,
    Bic,
    Cv,
}

impl Criterion {
    pub fn parse(name: &str) -> Result<Self, String> {
        match name {
            "aicc" => Ok(Self::Aicc),
            "aic" => Ok(Self::Aic),
            "bic" => Ok(Self::Bic),
            "cv" => Ok(Self::Cv),
            other => Err(format!(
                "unknown criterion {other:?}; expected aicc, aic, bic or cv"
            )),
        }
    }
}

/// Search strategy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Search {
    /// mgwr's `golden_section`.
    Golden,
    /// mgwr's `interval`: evaluate `bw_min, bw_min + step, ...` up to `bw_max`.
    Interval(f64),
    /// `scipy.optimize.minimize_scalar(method='bounded')` over mgwr's default (or given) bounds.
    Scipy,
}

/// Options of [`select_bandwidth`].
#[derive(Debug, Clone, Copy)]
pub struct SelectOptions {
    pub criterion: Criterion,
    pub search: Search,
    pub bw_min: Option<f64>,
    /// `Some(+inf)` additionally evaluates the global model (mgwr's `bw_max=np.inf`).
    pub bw_max: Option<f64>,
    /// Number of columns of `X` as the user gave them (before the constant), used only for the
    /// default adaptive lower bound `40 + 2 * n_vars`.
    pub n_loc: usize,
    pub constant: bool,
    pub tol: f64,
    pub max_iter: usize,
}

/// The selected bandwidth, its criterion value and the full evaluation history.
#[derive(Debug, Clone)]
pub struct Selection {
    pub bw: f64,
    pub score: f64,
    pub history: Vec<(f64, f64)>,
    /// Evaluation count and SciPy status of the `scipy` search (`None` for the others).
    pub optimizer: Option<Optimizer>,
}

/// What mgwr's `_optimize_result` exposes besides `x` and `fun`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Optimizer {
    /// Objective evaluations (`nfev`, equal to `nit` for the bounded method).
    pub nfev: usize,
    /// SciPy's `status`: 0 converged, 1 evaluation limit reached, 2 NaN encountered.
    pub status: u8,
}

/// `np.round(x, 2)`: scaled round-half-to-even.
fn round2(x: f64) -> f64 {
    (x * 100.0).round_ties_even() / 100.0
}

/// Port of `mgwr.search.golden_section` (the `c = inf` branch of the original is dead code).
///
/// `f` is evaluated at most once per exact bandwidth value. With `int_score` the probes are
/// rounded half to even. With `include_global` the model at `bw = +inf` is also evaluated and
/// wins ties.
pub fn golden_section<F>(
    mut a: f64,
    mut c: f64,
    mut f: F,
    tol: f64,
    max_iter: usize,
    int_score: bool,
    include_global: bool,
) -> Result<Selection, String>
where
    F: FnMut(f64) -> Result<f64, String>,
{
    let mut b = a + DELTA * (c - a).abs();
    let mut d = c - DELTA * (c - a).abs();
    let mut opt_val;
    let mut opt_score = f64::INFINITY;
    let mut diff = 1.0e9_f64;
    let mut iters = 0;
    let mut history: Vec<(f64, f64)> = Vec::new();
    let mut memo: HashMap<u64, f64> = HashMap::new();
    let mut best_val = f64::NAN;
    while diff.abs() > tol && iters < max_iter {
        iters += 1;
        if int_score {
            b = b.round_ties_even();
            d = d.round_ties_even();
        }
        let score_b = match memo.get(&b.to_bits()) {
            Some(s) => *s,
            None => {
                let s = f(b)?;
                memo.insert(b.to_bits(), s);
                s
            }
        };
        let score_d = match memo.get(&d.to_bits()) {
            Some(s) => *s,
            None => {
                let s = f(d)?;
                memo.insert(d.to_bits(), s);
                s
            }
        };
        if score_b <= score_d {
            opt_val = b;
            opt_score = score_b;
            c = d;
            d = b;
            b = a + DELTA * (c - a).abs();
        } else {
            opt_val = d;
            opt_score = score_d;
            a = b;
            b = d;
            d = c - DELTA * (c - a).abs();
        }
        history.push((opt_val, opt_score));
        let rounded = round2(opt_val);
        if !history.contains(&(rounded, opt_score)) {
            history.push((rounded, opt_score));
        }
        best_val = rounded;
        diff = score_b - score_d;
    }
    if include_global {
        let s = f(f64::INFINITY)?;
        history.push((f64::INFINITY, s));
        if s <= opt_score {
            opt_score = s;
            best_val = f64::INFINITY;
        }
    }
    Ok(Selection {
        bw: best_val,
        score: opt_score,
        history,
        optimizer: None,
    })
}

/// Port of `mgwr.search.equal_interval`: strict improvement keeps the earlier bandwidth.
pub fn equal_interval<F>(
    lower: f64,
    upper: f64,
    interval: f64,
    mut f: F,
    int_score: bool,
) -> Result<Selection, String>
where
    F: FnMut(f64) -> Result<f64, String>,
{
    let (mut a, mut c, mut b) = (lower, upper, lower + interval);
    if int_score {
        a = a.round_ties_even();
        c = c.round_ties_even();
        b = b.round_ties_even();
    }
    let mut history = Vec::new();
    let score_a = f(a)?;
    history.push((a, score_a));
    let (mut opt_val, mut opt_score) = (a, score_a);
    while b < c {
        let s = f(b)?;
        history.push((b, s));
        if s < opt_score {
            opt_val = b;
            opt_score = s;
        }
        b += interval;
    }
    let s = f(c)?;
    history.push((c, s));
    if s < opt_score {
        opt_val = c;
        opt_score = s;
    }
    Ok(Selection {
        bw: opt_val,
        score: opt_score,
        history,
        optimizer: None,
    })
}

/// Min and max over all pairwise distances (self excluded from the minimum), in parallel.
fn distance_extent(m: &Model) -> (f64, f64) {
    (0..m.n)
        .into_par_iter()
        .map_init(
            || vec![0.0; m.n],
            |row, i| {
                distances(m, i, row);
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for (j, &d) in row.iter().enumerate() {
                    hi = hi.max(d);
                    if j != i {
                        lo = lo.min(d);
                    }
                }
                (lo, hi)
            },
        )
        .reduce(
            || (f64::INFINITY, f64::NEG_INFINITY),
            |p, q| (p.0.min(q.0), p.1.max(q.1)),
        )
}

/// mgwr's `_init_section`: the default search bounds (explicit `bw_min`/`bw_max` override them),
/// and whether `bw_max = +inf` asked for the global model as well.
///
/// Adaptive searches run over `[40 + 2 n_vars, n]` neighbors and fixed ones over
/// `[min_dist / 2, 2 max_dist]`. Unlike mgwr, which crashes inside numpy, an adaptive search
/// whose bounds fall outside `1..=n` is rejected up front.
fn search_bounds(m: &Model, o: &SelectOptions, distinct: bool) -> Result<(f64, f64, bool), String> {
    let ordered = |a: f64, c: f64| if distinct { a < c } else { a <= c };
    let int_score = !m.fixed;
    let n_vars = o.n_loc + usize::from(o.constant);
    let (mut a, mut c) = if int_score {
        ((40 + 2 * n_vars) as f64, m.n as f64)
    } else {
        let (lo, hi) = distance_extent(m);
        (lo / 2.0, hi * 2.0)
    };
    if let Some(v) = o.bw_min {
        a = v;
    }
    let include_global = o.bw_max == Some(f64::INFINITY);
    if let Some(v) = o.bw_max {
        if v != f64::INFINITY {
            c = v;
        }
    }
    if int_score && !(a >= 1.0 && ordered(a, c) && c <= m.n as f64) {
        return Err(format!(
            "adaptive bandwidth search needs n >= 40 + 2k or an explicit bw_min/bw_max \
             (bounds {a}..{c} for n = {})",
            m.n
        ));
    }
    if !int_score && !(a > 0.0 && ordered(a, c)) {
        return Err(format!("invalid bandwidth bounds {a}..{c}"));
    }
    Ok((a, c, include_global))
}

/// `scipy.optimize.minimize_scalar(f, bounds=(a, c), method='bounded')` as mgwr calls it: the
/// float abscissa is returned as is (an adaptive probe is truncated by the fit itself) and no
/// history is kept.
fn scipy_bounded<F>(a: f64, c: f64, mut f: F) -> Result<Selection, String>
where
    F: FnMut(f64) -> Result<f64, String>,
{
    let mut failure: Option<String> = None;
    let min = minimize_bounded(
        |bw| match f(bw) {
            Ok(v) => v,
            Err(e) => {
                failure.get_or_insert(e);
                f64::NAN
            }
        },
        a,
        c,
        SCIPY_XATOL,
        SCIPY_MAXFUN,
    )?;
    if let Some(e) = failure {
        return Err(e);
    }
    Ok(Selection {
        bw: min.x,
        score: min.fun,
        history: Vec::new(),
        optimizer: Some(Optimizer {
            nfev: min.nfev,
            status: min.flag,
        }),
    })
}

/// Selects the bandwidth that minimizes the criterion of a GWR fit (any family of the model).
///
/// Bounds follow mgwr's `_init_section`: adaptive searches run over `[40 + 2 n_vars, n]` neighbors,
/// fixed ones over `[min_dist / 2, 2 max_dist]`, with explicit `bw_min`/`bw_max` overriding them.
/// Unlike mgwr, which crashes inside numpy, an adaptive search whose bounds fall outside
/// `1..=n` is rejected up front.
pub fn select_bandwidth(m: &Model, o: &SelectOptions) -> Result<Selection, String> {
    let score = |bw: f64| -> Result<f64, String> {
        let fit = fit_lite(m, bw)?;
        Ok(match o.criterion {
            Criterion::Aicc => fit.criteria.aicc,
            Criterion::Aic => fit.criteria.aic,
            Criterion::Bic => fit.criteria.bic,
            Criterion::Cv => fit.cv,
        })
    };
    let int_score = !m.fixed;
    match o.search {
        Search::Golden => {
            let (a, c, include_global) = search_bounds(m, o, true)?;
            golden_section(a, c, score, o.tol, o.max_iter, int_score, include_global)
        }
        Search::Scipy => {
            if o.bw_max == Some(f64::INFINITY) {
                return Err(
                    "the scipy search cannot include the global model (bw_max = infinity)"
                        .to_string(),
                );
            }
            let (a, c, _) = search_bounds(m, o, false)?;
            if a == c {
                return Err(
                    "maximum and minimum bandwidth must be distinct for the scipy optimizer"
                        .to_string(),
                );
            }
            scipy_bounded(a, c, score)
        }
        Search::Interval(step) => {
            let (Some(lo), Some(hi)) = (o.bw_min, o.bw_max) else {
                return Err("interval search needs bw_min and bw_max".to_string());
            };
            if !(step > 0.0 && step.is_finite()) {
                return Err("interval search needs a positive interval".to_string());
            }
            if !(lo < hi && hi.is_finite()) {
                return Err(format!("invalid bandwidth bounds {lo}..{hi}"));
            }
            equal_interval(lo, hi, step, score, int_score)
        }
    }
}
