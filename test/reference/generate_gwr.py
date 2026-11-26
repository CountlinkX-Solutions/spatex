"""Generate GWR / bandwidth-selection oracle fixtures (`fixtures/regression/gwr_*.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_gwr.py

Ground truth is mgwr 2.2.1 (`GWR`, `Sel_BW`). Two datasets:

* `gwr_georgia.json` - libpysal's bundled Georgia example (n = 159, PctBach ~ PctRural + PctPov +
  PctBlack, UTM metres). Bandwidth selection for {adaptive, fixed} x {bisquare, gaussian,
  exponential} with AICc, plus AIC/BIC/CV (adaptive bisquare), the interval search and the
  `bw_max = inf` global check; GWR fits at each selected bandwidth with every result attribute.
* `gwr_synthetic.json` - a seeded n = 120 lon/lat sample with known spatially varying
  coefficients: spherical (haversine) fixed gaussian, and a no-constant model.

* `gwr_glm.json` - seeded n = 80 jittered-lattice samples: Poisson with an exposure offset (adaptive
  bisquare AICc/AIC/BIC searches, fixed gaussian, global bandwidth), a Poisson sample with zero
  counts (documents the NaN local deviance of mgwr), and Binomial 0/1 (adaptive and fixed).
  The GLM fits record deviance, null/global deviance, D2, adj D2, local deviance and pDev.
* `gwr_mgwr.json` - MGWR (backfitting + inference) on seeded jittered lattices, standardized as
  mgwr requires: fixed and adaptive bisquare, bounds, RSS score, fixed gaussian, and Georgia.

Search histories are stored in full so a mismatch can be compared evaluation by evaluation.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal as ps  # noqa: E402
import mgwr  # noqa: E402
import numpy as np  # noqa: E402
from mgwr.diagnostics import get_CV  # noqa: E402
from mgwr.gwr import GWR, MGWR  # noqa: E402
from mgwr.sel_bw import Sel_BW  # noqa: E402
from spglm.family import Binomial, Poisson  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def hist(sel):
    return [[float(b), float(np.asarray(s).ravel()[0])] for b, s in sel.sel_hist]


def select(coords, y, x, **kw):
    search = {
        k: kw.pop(k)
        for k in ("criterion", "bw_min", "bw_max", "search_method", "interval")
        if k in kw
    }
    sel = Sel_BW(coords, y, x, n_jobs=1, **kw)
    bw = sel.search(**search)
    return {
        "options": {**kw, **search},
        "bw": float(bw),
        "score": float(np.asarray(sel.bw[1]).ravel()[0]),
        "history": hist(sel),
    }


def fit(coords, y, x, bw, **kw):
    sigma2_v1 = kw.pop("sigma2_v1", True)
    r = GWR(coords, y, x, bw, sigma2_v1=sigma2_v1, n_jobs=1, **kw).fit()
    flat = lambda a: np.asarray(a, dtype=float).reshape(-1)
    return {
        "options": {**kw, "bw": float(bw), "sigma2_v1": sigma2_v1},
        "params": r.params,
        "bse": r.bse,
        "tvalues": r.tvalues,
        "local_r2": flat(r.localR2),
        "predy": flat(r.predy),
        "resid": flat(r.resid_response),
        "influ": flat(r.influ),
        "cooks_d": flat(r.cooksD),
        "std_res": flat(r.std_res),
        "resid_ss": float(r.resid_ss),
        "tr_s": float(r.tr_S),
        "tr_sts": float(r.tr_STS),
        "enp": float(r.ENP),
        "sigma2": float(r.sigma2),
        "scale": float(r.scale),
        "llf": float(r.llf),
        "aic": float(r.aic),
        "aicc": float(r.aicc),
        "bic": float(r.bic),
        "cv": float(get_CV(r)),
        "r2": float(r.R2),
        "adj_r2": float(r.adj_R2),
        "df_resid": float(r.df_resid),
        "adj_alpha": flat(r.adj_alpha),
        "critical_tval": float(r.critical_tval()),
        "critical_tval_05": float(r.critical_tval(alpha=0.05)),
        "filter_tvals": r.filter_tvals(),
        "filter_tvals_05": r.filter_tvals(alpha=0.05),
    }


def georgia():
    d = ps.io.open(ps.examples.get_path("GData_utm.csv"))
    coords = np.array(list(zip(d.by_col("X"), d.by_col("Y"))))
    y = np.array(d.by_col("PctBach")).reshape(-1, 1)
    cols = ["PctRural", "PctPov", "PctBlack"]
    x = np.hstack([np.array(d.by_col(c)).reshape(-1, 1) for c in cols])
    out = {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "coords": coords,
        "y": y.ravel(),
        "x": x.T,
        "selection": {},
        "fits": {},
    }
    for kernel in ("bisquare", "gaussian", "exponential"):
        for fixed in (False, True):
            key = f"{kernel}_{'fixed' if fixed else 'adaptive'}"
            sel = select(coords, y, x, kernel=kernel, fixed=fixed, criterion="AICc")
            out["selection"][key] = sel
            out["fits"][key] = fit(coords, y, x, sel["bw"], kernel=kernel, fixed=fixed)
    for crit in ("AIC", "BIC", "CV"):
        out["selection"][f"bisquare_adaptive_{crit.lower()}"] = select(
            coords, y, x, criterion=crit
        )
    out["selection"]["bisquare_fixed_interval"] = select(
        coords, y, x, fixed=True, search_method="interval",
        bw_min=211001.0, bw_max=211035.0, interval=2,
    )
    sel_inf = Sel_BW(coords, y, x, n_jobs=1)
    bw_inf = sel_inf.search(bw_max=np.inf)
    out["selection"]["bisquare_adaptive_include_global"] = {
        "options": {"bw_max": "inf"}, "bw": float(bw_inf),
        "score": float(np.asarray(sel_inf.bw[1]).ravel()[0]), "history": hist(sel_inf),
    }
    out["selection"]["bisquare_adaptive_bounds"] = select(
        coords, y, x, bw_min=60, bw_max=120
    )
    out["fits"]["bisquare_adaptive_sigma2_v2"] = fit(
        coords, y, x, out["selection"]["bisquare_adaptive"]["bw"], sigma2_v1=False
    )
    out["fits"]["bisquare_adaptive_infinity"] = fit(coords, y, x, np.inf)
    out["fits"]["gaussian_fixed_small"] = fit(
        coords, y, x, 60000.0, kernel="gaussian", fixed=True
    )
    return out


def synthetic():
    rng = np.random.default_rng(20261025)
    n = 120
    lon = rng.uniform(-100.0, -90.0, n)
    lat = rng.uniform(30.0, 40.0, n)
    coords = np.column_stack([lon, lat])
    u = (lon + 100.0) / 10.0
    v = (lat - 30.0) / 10.0
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) + 0.3 * x1
    y = (1.0 + 2.0 * u) + (1.5 - 2.0 * v) * x1 + (0.5 + np.sin(3.0 * u)) * x2
    y = y + rng.normal(scale=0.3, size=n)
    x = np.column_stack([x1, x2])
    yy = y.reshape(-1, 1)
    out = {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "coords": coords, "y": y, "x": x.T, "selection": {}, "fits": {},
    }
    sel = select(coords, yy, x, kernel="gaussian", fixed=True, spherical=True)
    out["selection"]["gaussian_fixed_spherical"] = sel
    out["fits"]["gaussian_fixed_spherical"] = fit(
        coords, yy, x, sel["bw"], kernel="gaussian", fixed=True, spherical=True
    )
    sel = select(coords, yy, x, kernel="bisquare", fixed=False, spherical=True)
    out["selection"]["bisquare_adaptive_spherical"] = sel
    out["fits"]["bisquare_adaptive_spherical"] = fit(
        coords, yy, x, sel["bw"], spherical=True
    )
    # No-constant model: the design is used exactly as given.
    out["selection"]["bisquare_adaptive_no_constant"] = select(
        coords, yy, x, constant=False
    )
    out["fits"]["bisquare_adaptive_no_constant"] = fit(
        coords, yy, x, out["selection"]["bisquare_adaptive_no_constant"]["bw"],
        constant=False,
    )
    # A constant column in X is dropped and replaced by the intercept.
    xc = np.column_stack([x, np.full(n, 3.0)])
    out["fits"]["constant_column_dropped"] = fit(coords, yy, xc, 50)
    out["fits"]["constant_column_dropped"]["input_x_columns"] = 3
    return out


FAMILIES = {"poisson": Poisson, "binomial": Binomial}


def glm_select(coords, y, x, family, offset=None, **kw):
    search = {k: kw.pop(k) for k in ("criterion", "bw_min", "bw_max") if k in kw}
    sel = Sel_BW(coords, y, x, family=FAMILIES[family](), offset=offset, n_jobs=1, **kw)
    bw = sel.search(**search)
    return {
        "options": {**kw, **search, "family": family},
        "bw": float(bw),
        "score": float(np.asarray(sel.bw[1]).ravel()[0]),
        "history": hist(sel),
    }


def glm_fit(coords, y, x, bw, family, offset=None, **kw):
    r = GWR(coords, y, x, bw, family=FAMILIES[family](), offset=offset, n_jobs=1, **kw).fit()
    flat = lambda a: np.asarray(a, dtype=float).reshape(-1)
    return {
        "options": {**kw, "bw": float(bw), "family": family},
        "params": r.params, "bse": r.bse, "tvalues": r.tvalues,
        "predy": flat(r.predy), "resid": flat(r.resid_response), "influ": flat(r.influ),
        "cooks_d": flat(r.cooksD), "std_res": flat(r.std_res),
        "resid_ss": float(r.resid_ss), "tr_s": float(r.tr_S), "tr_sts": float(r.tr_STS),
        "enp": float(r.ENP), "sigma2": float(r.sigma2), "scale": float(r.scale),
        "llf": float(r.llf), "aic": float(r.aic), "aicc": float(r.aicc), "bic": float(r.bic),
        "cv": float(get_CV(r)), "d2": float(r.D2), "adj_d2": float(r.adj_D2),
        "null_deviance": float(r.null_deviance), "global_deviance": float(r.global_deviance),
        "deviance": flat(r.deviance), "resid_deviance": flat(r.resid_deviance),
        "p_dev": flat(r.pDev), "df_resid": float(r.df_resid),
        "adj_alpha": flat(r.adj_alpha), "critical_tval": float(r.critical_tval()),
        "filter_tvals": r.filter_tvals(),
    }


def lattice(rng, rows, cols, jitter=0.4):
    gx, gy = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    c = np.c_[gx.ravel(), gy.ravel()]
    return c + rng.uniform(0.0, jitter, size=c.shape)


def glm():
    rng = np.random.default_rng(20261026)
    n = 80
    coords = lattice(rng, 8, 10)
    x = rng.normal(size=(n, 2))
    u, v = coords[:, 0] / 9.0, coords[:, 1] / 7.0
    off = rng.uniform(0.5, 2.0, size=n)
    eta = 1.2 + 0.4 * u + (0.5 + 0.4 * v) * x[:, 0] - (0.3 + 0.5 * u) * x[:, 1]
    counts = rng.poisson(off * np.exp(eta))
    y_pos = np.maximum(counts, 1).astype(float)
    y_zero = counts.astype(float)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 1.2 * (u - 0.5) * 2 * x[:, 0] + 0.8 * x[:, 1] * (v + 0.3))))
    yb = (rng.uniform(size=n) < p).astype(float)
    out = {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "coords": coords, "x": x.T, "offset": off,
        "y_poisson": y_pos, "y_poisson_zeros": y_zero, "y_binomial": yb,
        "selection": {}, "fits": {},
    }
    def tag(entry, ykey, uses_offset):
        entry["y_key"] = ykey
        entry["uses_offset"] = uses_offset
        return entry

    yp = y_pos.reshape(-1, 1)
    ob = off.reshape(-1, 1)
    yz = y_zero.reshape(-1, 1)
    yb2 = yb.reshape(-1, 1)
    sel, fits = out["selection"], out["fits"]
    sel["poisson_adaptive_aicc"] = tag(glm_select(coords, yp, x, "poisson", ob), "y_poisson", True)
    bw = sel["poisson_adaptive_aicc"]["bw"]
    fits["poisson_adaptive"] = tag(glm_fit(coords, yp, x, bw, "poisson", ob), "y_poisson", True)
    for crit in ("AIC", "BIC"):
        sel[f"poisson_adaptive_{crit.lower()}"] = tag(
            glm_select(coords, yp, x, "poisson", ob, criterion=crit), "y_poisson", True)
    sel["poisson_fixed_gaussian"] = tag(
        glm_select(coords, yp, x, "poisson", ob, kernel="gaussian", fixed=True),
        "y_poisson", True)
    fits["poisson_fixed_gaussian"] = tag(glm_fit(
        coords, yp, x, sel["poisson_fixed_gaussian"]["bw"], "poisson", ob,
        kernel="gaussian", fixed=True), "y_poisson", True)
    fits["poisson_infinity"] = tag(
        glm_fit(coords, yp, x, np.inf, "poisson", ob), "y_poisson", True)
    # No offset argument at all: mgwr defaults to ones.
    fits["poisson_no_offset"] = tag(
        glm_fit(coords, yp, x, 55, "poisson", None), "y_poisson", False)
    fits["poisson_zeros"] = tag(
        glm_fit(coords, yz, x, 55, "poisson", ob), "y_poisson_zeros", True)
    sel["binomial_adaptive_aicc"] = tag(glm_select(coords, yb2, x, "binomial"), "y_binomial", False)
    fits["binomial_adaptive"] = tag(glm_fit(
        coords, yb2, x, sel["binomial_adaptive_aicc"]["bw"], "binomial"), "y_binomial", False)
    sel["binomial_fixed_gaussian"] = tag(
        glm_select(coords, yb2, x, "binomial", kernel="gaussian", fixed=True),
        "y_binomial", False)
    fits["binomial_fixed_gaussian"] = tag(glm_fit(
        coords, yb2, x, sel["binomial_fixed_gaussian"]["bw"], "binomial",
        kernel="gaussian", fixed=True), "y_binomial", False)
    fits["binomial_infinity"] = tag(
        glm_fit(coords, yb2, x, np.inf, "binomial"), "y_binomial", False)
    return out


def mgwr_case(coords, y, x, standardize=True, **kw):
    """One MGWR run: the selector's backfitting, then inference, with every reported attribute."""
    if standardize:
        y = (y - y.mean()) / y.std()
        x = (x - x.mean(0)) / x.std(0)
    yy = y.reshape(-1, 1)
    model = {k: kw.pop(k) for k in ("kernel", "fixed") if k in kw}
    sel = Sel_BW(coords, yy, x, multi=True, n_jobs=1, **model)
    sel.search(verbose=False, **kw)
    m = MGWR(coords, yy, x, sel, n_jobs=1, **model).fit()
    flat = lambda a: np.asarray(a, dtype=float).reshape(-1)
    return {
        "options": {**model, **{k: v for k, v in kw.items()}},
        "y": y, "x": x.T,
        "bws": sel.bw[0], "bws_history": sel.bw[1], "scores": sel.bw[2],
        "bw_init": float(sel.bw_init),
        "params": m.params, "bse": m.bse, "tvalues": m.tvalues,
        "predy": flat(m.predy), "resid": flat(m.resid_response),
        "enp_j": m.ENP_j, "tr_s": float(m.tr_S), "sigma2": float(m.sigma2),
        "resid_ss": float(m.resid_ss), "llf": float(m.llf), "aic": float(m.aic),
        "aicc": float(m.aicc), "bic": float(m.bic), "r2": float(m.R2),
        "adj_r2": float(m.adj_R2), "adj_alpha_j": m.adj_alpha_j,
        "critical_tval": m.critical_tval(), "cct": m.CCT,
        "filter_tvals": m.filter_tvals(),
    }


def mgwr_cases():
    rng = np.random.default_rng(7)
    coords = lattice(rng, 7, 7, 0.0) + rng.normal(scale=0.05, size=(49, 2))
    x = rng.normal(size=(49, 2))
    y = 1 + (0.5 + 0.3 * coords[:, 0]) * x[:, 0] + 0.8 * x[:, 1] + rng.normal(size=49) * 0.3
    out = {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "coords": coords, "cases": {},
    }
    c = out["cases"]
    c["lattice49_fixed"] = mgwr_case(coords, y, x, fixed=True)
    c["lattice49_adaptive"] = mgwr_case(coords, y, x, fixed=False)
    c["lattice49_fixed_gaussian_bic"] = mgwr_case(
        coords, y, x, fixed=True, kernel="gaussian", criterion="BIC", multi_bw_min=[1.0])
    c["lattice49_fixed_bounds"] = mgwr_case(
        coords, y, x, fixed=True, multi_bw_min=[2.0, 1.0, 3.0], multi_bw_max=[5.0, 6.0, 12.0])
    c["lattice49_fixed_rss_score"] = mgwr_case(coords, y, x, fixed=True, rss_score=True)
    # A second, larger sample with a smooth and a nearly global effect.
    rng = np.random.default_rng(99)
    coords2 = lattice(rng, 10, 10, 0.5)
    x2 = rng.normal(size=(100, 2))
    y2 = (1 + 0.1 * coords2[:, 0] + (0.5 + 0.25 * coords2[:, 1]) * x2[:, 0]
          + 0.7 * x2[:, 1] + rng.normal(size=100) * 0.4)
    out["coords2"] = coords2
    c["synthetic100_adaptive"] = mgwr_case(coords2, y2, x2, fixed=False)
    d = ps.io.open(ps.examples.get_path("GData_utm.csv"))
    gc = np.array(list(zip(d.by_col("X"), d.by_col("Y"))))
    gy = np.array(d.by_col("PctBach"))
    gx = np.column_stack([np.array(d.by_col(col)) for col in ("PctRural", "PctPov", "PctBlack")])
    out["georgia_coords"] = gc
    c["georgia_adaptive"] = mgwr_case(gc, gy, gx, fixed=False)
    return out



if __name__ == "__main__":
    which = sys.argv[1:] or ["gwr", "glm", "mgwr"]
    if "gwr" in which:
        dump_json(FIXTURES / "regression" / "gwr_georgia.json", georgia())
        dump_json(FIXTURES / "regression" / "gwr_synthetic.json", synthetic())
    if "glm" in which:
        dump_json(FIXTURES / "regression" / "gwr_glm.json", glm())
    if "mgwr" in which:
        dump_json(FIXTURES / "regression" / "gwr_mgwr.json", mgwr_cases())
