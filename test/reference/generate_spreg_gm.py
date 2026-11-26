"""Generate spreg 2SLS / GM oracle fixtures (`fixtures/regression/gm_*.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_gm.py

Covers `TSLS`, `GM_Lag`, `GM_Error`, `GM_Endog_Error`, `GM_Error_Het`, `GM_Endog_Error_Het`,
`GM_Error_Hom`, `GM_Endog_Error_Hom` and `GM_Combo`, `GM_Combo_Het`, `GM_Combo_Hom`.

spreg minimizes the GMM moment objective with L-BFGS-B (early stop, ~1e-6 accuracy), so every
case stores the lambdas spreg's `optim_moments` returned, in call order (`lambdas`). Spatex's
`*_at` test hooks are fed these values, which verifies every formula at 1e-9. The exact
optimizer's own result is compared with a looser tolerance.

Each case stores the data (`y`, `x` columns without the constant, optional `yend` / `q`), the
exact sparse weights spreg consumed, and the attributes of the fitted model. The instrument
`DISCBD` (not part of `datasets/columbus.json`) is read from libpysal's Columbus data.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from libpysal.weights import Kernel, W  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

LAMBDAS = []


def _abs_without_clamp(x):
    """`abs` for `error_sp_het`: a length-1 array (`tsls.betas[-1]`) reads as 0, so the clamp of
    the last endogenous coefficient to +-0.9 never fires; scalars behave normally."""
    if isinstance(x, np.ndarray) and x.shape == (1,):
        return 0.0
    return abs(x)


def record_optim_moments():
    """Wrap every reference to `optim_moments` so each returned lambda is recorded in order."""
    import spreg.utils as utils

    original = utils.optim_moments

    def wrapped(*args, **kwargs):
        out = original(*args, **kwargs)
        LAMBDAS.append(float(np.asarray(out).ravel()[0]))
        return out

    for name, mod in list(sys.modules.items()):
        if name.startswith("spreg") and getattr(mod, "optim_moments", None) is original:
            mod.optim_moments = wrapped


record_optim_moments()


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    db = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    assert np.allclose(db["CRIME"].to_numpy(), attrs["CRIME"]), "dataset order differs"
    attrs["DISCBD"] = db["DISCBD"].to_numpy(dtype=float)
    return geoms, attrs


def sparse_export(w):
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    return {
        "n": int(w.n),
        "focal": coo.row[order].astype(int).tolist(),
        "neighbor": coo.col[order].astype(int).tolist(),
        "weight": coo.data[order].astype(float).tolist(),
    }


def rebuild(spec):
    n = spec["n"]
    neighbors = {i: [] for i in range(n)}
    weights = {i: [] for i in range(n)}
    for f, t, w in zip(spec["focal"], spec["neighbor"], spec["weight"]):
        neighbors[f].append(t)
        weights[f].append(w)
    return W(neighbors, weights, silence_warnings=True)


def transformed_w(graph, transform):
    g = graph.transform(transform) if transform != "o" else graph
    return g.to_W()


def vec(a):
    return np.asarray(a, dtype=float).ravel()


def mat(a):
    return np.asarray(a, dtype=float).tolist()


def pair(t):
    return [float(t[0]), float(t[1])]


def attrs_of(reg):
    out = {
        "n": int(reg.n),
        "k": int(reg.k),
        "betas": vec(reg.betas),
        "vm": mat(reg.vm),
        "std_err": vec(reg.std_err),
        "z_stat": [pair(z) for z in reg.z_stat],
        "u": vec(reg.u),
        "predy": vec(reg.predy),
        "pr2": float(reg.pr2),
        "name_x": list(reg.name_x),
    }
    for key in ("sig2", "varb"):
        v = getattr(reg, key, None)
        if v is not None:
            out[key] = float(np.asarray(v).ravel()[0]) if key == "sig2" else mat(v)
    if getattr(reg, "e_filtered", None) is not None:
        out["e_filtered"] = vec(reg.e_filtered)
    if getattr(reg, "predy_e", None) is not None and hasattr(reg, "rho"):
        out["rho"] = float(np.asarray(reg.rho).ravel()[0])
        out["predy_e"] = vec(reg.predy_e)
        out["e_pred"] = vec(reg.e_pred)
        out["pr2_e"] = float(reg.pr2_e)
    if hasattr(reg, "iteration"):
        out["iteration"] = int(reg.iteration)
        out["iter_stop"] = str(reg.iter_stop)
    if getattr(reg, "ak_test", None) is not None:
        out["ak_test"] = pair(reg.ak_test)
    if getattr(reg, "sp_multipliers", None):
        out["sp_multipliers"] = {m: [float(v) for v in t] for m, t in reg.sp_multipliers.items()}
    if getattr(reg, "name_yend", None):
        out["name_yend"] = list(reg.name_yend)
    if getattr(reg, "name_z", None):
        out["name_z"] = list(reg.name_z)
    return out


def case(label, model, y, cols, names, w_spec, *, yend=None, q=None, yend_names=None, q_names=None,
         gwk=None, no_clamp=False, **opts):
    """Fit one spreg model; `opts` are the spreg keyword arguments (and are stored verbatim)."""
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    yd = None if yend is None else np.column_stack([np.asarray(c, dtype=float) for c in yend])
    qq = None if q is None else np.column_stack([np.asarray(c, dtype=float) for c in q])
    w = rebuild(w_spec)
    LAMBDAS.clear()
    kw = dict(opts)
    if gwk is not None:
        kw["gwk"] = gwk
    common = dict(w=w, name_x=list(names), name_y="y")
    if yend_names:
        common["name_yend"] = list(yend_names)
    if q_names:
        common["name_q"] = list(q_names)
    yv = y.reshape(-1, 1)
    cls = {
        "tsls": lambda: spreg.TSLS(yv, x, yd, qq, nonspat_diag=False, **kw, **common),
        "gm_lag": lambda: spreg.GM_Lag(yv, x, yend=yd, q=qq, **kw, **common),
        "gm_error": lambda: spreg.GM_Error(yv, x, **kw, **common),
        "gm_endog_error": lambda: spreg.GM_Endog_Error(yv, x, yd, qq, **kw, **common),
        "gm_error_het": lambda: spreg.GM_Error_Het(yv, x, **kw, **common),
        "gm_endog_error_het": lambda: spreg.GM_Endog_Error_Het(yv, x, yd, qq, **kw, **common),
        "gm_error_hom": lambda: spreg.GM_Error_Hom(yv, x, **kw, **common),
        "gm_endog_error_hom": lambda: spreg.GM_Endog_Error_Hom(yv, x, yd, qq, **kw, **common),
        "gm_combo": lambda: spreg.GM_Combo(yv, x, yend=yd, q=qq, **kw, **common),
        "gm_combo_het": lambda: spreg.GM_Combo_Het(yv, x, yend=yd, q=qq, **kw, **common),
        "gm_combo_hom": lambda: spreg.GM_Combo_Hom(yv, x, yend=yd, q=qq, **kw, **common),
    }[model]
    if no_clamp:
        import spreg.error_sp_het as het_mod

        het_mod.abs = _abs_without_clamp
        try:
            reg = cls()
        finally:
            del het_mod.abs
    else:
        reg = cls()
    kw.pop("gwk", None)
    out = {
        "label": label,
        "model": model,
        "opts": kw,
        "y": y,
        "x": [np.asarray(c, dtype=float) for c in cols],
        "name_x": list(names),
        "w": w_spec,
        "lambdas": list(LAMBDAS),
        "expected": attrs_of(reg),
    }
    if yend is not None:
        out["yend"] = [np.asarray(c, dtype=float) for c in yend]
        out["yend_names"] = list(yend_names)
        out["q"] = [np.asarray(c, dtype=float) for c in q]
        out["q_names"] = list(q_names)
    if gwk is not None:
        out["gwk"] = sparse_export(gwk)
    return out


def header(name, description):
    m = meta()
    m["spreg"] = spreg.__version__
    return {"name": name, "description": description, "meta": m}


def columbus_cases():
    geoms, attrs = columbus()
    queen = Graph.build_contiguity(geoms, rook=False)
    rook = Graph.build_contiguity(geoms, rook=True)
    queen_r = sparse_export(transformed_w(queen, "r"))
    rook_r = sparse_export(transformed_w(rook, "r"))
    rook_b = sparse_export(transformed_w(rook, "o"))
    cx, cy = geoms.centroid.x.to_numpy(), geoms.centroid.y.to_numpy()
    gwk = Kernel(np.column_stack([cx, cy]), k=15, function="triangular", fixed=False, diagonal=True)
    gwk.transform = "o"

    inc, crime, hoval, disc = attrs["INC"], attrs["CRIME"], attrs["HOVAL"], attrs["DISCBD"]
    hx, hn = [inc, crime], ["INC", "CRIME"]
    cx2, cn = [inc, hoval], ["INC", "HOVAL"]
    ex = dict(yend=[hoval], q=[disc], yend_names=["HOVAL"], q_names=["DISCBD"])

    cases = [
        # 2SLS
        case("tsls/crime~inc|hoval|discbd", "tsls", crime, [inc], ["INC"], queen_r, **ex),
        case("tsls/crime~inc|hoval|discbd/white", "tsls", crime, [inc], ["INC"], queen_r,
             robust="white", **ex),
        case("tsls/crime~inc|hoval|discbd/white_sig2n_k", "tsls", crime, [inc], ["INC"], queen_r,
             robust="white", sig2n_k=True, **ex),
        case("tsls/crime~inc|hoval|discbd/hac", "tsls", crime, [inc], ["INC"], queen_r,
             robust="hac", gwk=gwk, **ex),
        # spatial lag
        case("gm_lag/hoval~inc+crime/queen_r", "gm_lag", hoval, hx, hn, queen_r),
        case("gm_lag/hoval~inc+crime/queen_r/lags2", "gm_lag", hoval, hx, hn, queen_r, w_lags=2),
        case("gm_lag/hoval~inc+crime/queen_r/white", "gm_lag", hoval, hx, hn, queen_r,
             robust="white"),
        case("gm_lag/hoval~inc+crime/queen_r/hac", "gm_lag", hoval, hx, hn, queen_r,
             robust="hac", gwk=gwk),
        case("gm_lag/hoval~inc+crime/queen_r/sig2n_k", "gm_lag", hoval, hx, hn, queen_r,
             sig2n_k=True, spat_impacts=["simple", "full", "power"]),
        case("gm_lag/crime~inc+hoval/queen_r", "gm_lag", crime, cx2, cn, queen_r),
        case("gm_lag/crime~inc+hoval/rook_binary", "gm_lag", crime, cx2, cn, rook_b),
        case("gm_lag/crime~inc|hoval|discbd/queen_r", "gm_lag", crime, [inc], ["INC"], queen_r, **ex),
        case("gm_lag/crime~inc|hoval|discbd/queen_r/no_lag_q", "gm_lag", crime, [inc], ["INC"],
             queen_r, lag_q=False, **ex),
        # KP98 error
        case("gm_error/hoval~inc+crime/queen_r", "gm_error", hoval, hx, hn, queen_r),
        case("gm_error/crime~inc+hoval/queen_r", "gm_error", crime, cx2, cn, queen_r),
        case("gm_endog_error/crime~inc|hoval|discbd/queen_r", "gm_endog_error", crime, [inc], ["INC"],
             queen_r, **ex),
        # Het
        case("gm_error_het/hoval~inc+crime/queen_r", "gm_error_het", hoval, hx, hn, queen_r),
        case("gm_error_het/crime~inc+hoval/queen_r/step1c5", "gm_error_het", crime, cx2, cn,
             queen_r, step1c=True, max_iter=5),
        case("gm_error_het/crime~inc+hoval/rook_binary", "gm_error_het", crime, cx2, cn, rook_b),
        # Hom
        case("gm_error_hom/hoval~inc+crime/queen_r", "gm_error_hom", hoval, hx, hn, queen_r),
        case("gm_error_hom/crime~inc+hoval/queen_r/hom3", "gm_error_hom", crime, cx2, cn, queen_r,
             A1="hom", max_iter=3),
        case("gm_error_hom/crime~inc+hoval/queen_r/het", "gm_error_hom", crime, cx2, cn, queen_r,
             A1="het"),
        case("gm_endog_error_hom/crime~inc|hoval|discbd/queen_r", "gm_endog_error_hom", crime,
             [inc], ["INC"], queen_r, **ex),
        case("gm_endog_error_hom/crime~inc|hoval|discbd/queen_r/hom3", "gm_endog_error_hom", crime,
             [inc], ["INC"], queen_r, A1="hom", max_iter=3, **ex),
        # Combo
        case("gm_combo/crime~inc/queen_r", "gm_combo", crime, [inc], ["INC"], queen_r),
        case("gm_combo/hoval~inc+crime/queen_r", "gm_combo", hoval, hx, hn, queen_r, w_lags=2),
        case("gm_combo/crime~inc|hoval|discbd/queen_r", "gm_combo", crime, [inc], ["INC"], queen_r,
             **ex),
        case("gm_combo_het/crime~inc/queen_r", "gm_combo_het", crime, [inc], ["INC"], queen_r),
        case("gm_combo_het/crime~inc/queen_r/step1c3", "gm_combo_het", crime, [inc], ["INC"],
             queen_r, step1c=True, max_iter=3),
        case("gm_combo_het/crime~inc|hoval|discbd/queen_r", "gm_combo_het", crime, [inc], ["INC"],
             queen_r, **ex),
        case("gm_combo_hom/crime~inc/queen_r", "gm_combo_hom", crime, [inc], ["INC"], queen_r),
        case("gm_combo_hom/crime~inc/queen_r/hom3", "gm_combo_hom", crime, [inc], ["INC"], queen_r,
             A1="hom", max_iter=3),
        case("gm_combo_hom/crime~inc|hoval|discbd/queen_r", "gm_combo_hom", crime, [inc], ["INC"],
             queen_r, **ex),
    ]
    return cases


def synthetic_cases():
    rng = np.random.default_rng(20)
    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(10) for c in range(20)])
    lat = transformed_w(Graph.build_contiguity(cells, rook=True), "r")
    spec = sparse_export(lat)
    wd = lat.full()[0]
    n = 200
    x1 = rng.normal(size=n).round(6)
    x2 = rng.normal(size=n).round(6)
    inst = rng.normal(size=n).round(6)
    eps = rng.normal(size=n) * 2.0
    yend = (0.8 * inst + 0.5 * x1 + 0.6 * eps + rng.normal(size=n)).round(6)
    mu = 1.0 + 1.5 * x1 - 0.75 * x2 + 0.5 * yend
    y_lag = np.linalg.solve(np.eye(n) - 0.45 * wd, mu + eps).round(6)
    y_err = (mu + np.linalg.solve(np.eye(n) - 0.55 * wd, eps)).round(6)
    nm = ["X1", "X2"]
    ex = dict(yend=[yend], q=[inst], yend_names=["ENDOG"], q_names=["INST"])
    xs = [x1, x2]
    return [
        case("synthetic200/gm_lag", "gm_lag", y_lag, xs, nm, spec),
        case("synthetic200/gm_lag/endog", "gm_lag", y_lag, xs, nm, spec, **ex),
        case("synthetic200/gm_error", "gm_error", y_err, xs, nm, spec),
        case("synthetic200/gm_endog_error", "gm_endog_error", y_err, xs, nm, spec, **ex),
        case("synthetic200/gm_error_het", "gm_error_het", y_err, xs, nm, spec, step1c=True,
             max_iter=4),
        case("synthetic200/gm_error_hom", "gm_error_hom", y_err, xs, nm, spec, max_iter=3),
        case("synthetic200/gm_combo", "gm_combo", y_lag, xs, nm, spec),
        case("synthetic200/gm_combo_het", "gm_combo_het", y_lag, xs, nm, spec, max_iter=2),
        case("synthetic200/gm_combo_hom", "gm_combo_hom", y_lag, xs, nm, spec, max_iter=2),
        case("synthetic200/gm_combo_hom/endog", "gm_combo_hom", y_lag, xs, nm, spec, **ex),
    ]


def first_step_curve(c):
    """The first-step moment objective on the lambda grid spreg's L-BFGS-B searches.

    `[lam, F]` pairs with spreg's scaling, for the 3-equation (KP98, sigma^2 profiled out) or the
    2-equation (Het) objective of the first stage.
    """
    import copy

    import spreg.error_sp as sp_mod
    import spreg.utils as utils

    w = rebuild(c["w"])
    y = np.array(c["y"]).reshape(-1, 1)
    x = np.column_stack([np.ones(len(y))] + [np.array(a) for a in c["x"]])
    if "yend" in c:
        yd = np.column_stack([np.array(a) for a in c["yend"]])
        q = np.column_stack([np.array(a) for a in c["q"]])
        u = spreg.twosls.BaseTSLS(y, x, yd, q).u
    else:
        u = spreg.ols.BaseOLS(y, x).u
    if c["model"].endswith("het"):
        g_mat, g = utils._moments2eqs(utils.get_A1_het(w.sparse), w.sparse, u)
        three = False
    else:
        g_mat, g = sp_mod._momentsGM_Error(w.sparse, u)
        three = True
    scale = np.min([[np.min(g_mat), np.min(g)]])
    g_mat, g = copy.deepcopy(g_mat) / scale, copy.deepcopy(g) / scale

    def objective(lam):
        r = g - g_mat[:, :2] @ np.array([[lam, lam**2]]).T
        if three:
            col = g_mat[:, 2:3]
            s2 = max(0.0, float(((col.T @ r) / (col.T @ col)).item()))
            r = r - col * s2
        return float(np.sum(r**2))

    from scipy.optimize import minimize_scalar

    curve = [[float(lam), objective(lam)] for lam in np.linspace(-0.99, 0.99, 199)]
    best = min(curve, key=lambda p: p[1])[0]
    res = minimize_scalar(objective, bounds=(max(best - 0.02, -0.99), min(best + 0.02, 0.99)),
                          method="bounded", options={"xatol": 1e-12})
    return {
        "curve": curve,
        "lam_exact": float(res.x),
        "f_exact": float(res.fun),
        "f_pysal": objective(c["lambdas"][0]),
    }


def optimizer_cases():
    """Cases where spreg's L-BFGS-B ends away from the exact minimizer of its own objective.

    * binary rook weights, KP98: the objective is flat near the minimum, so the search stops
      4e-4 early;
    * `GM_Endog_Error_Het`: the first step ends at 0.8227 although the objective is increasing
      there (minimum at 0.3665): an abnormal line-search exit.
    """
    geoms, attrs = columbus()
    rook_b = sparse_export(transformed_w(Graph.build_contiguity(geoms, rook=True), "o"))
    queen_r = sparse_export(transformed_w(Graph.build_contiguity(geoms, rook=False), "r"))
    inc, crime, hoval, disc = attrs["INC"], attrs["CRIME"], attrs["HOVAL"], attrs["DISCBD"]
    ex = dict(yend=[hoval], q=[disc], yend_names=["HOVAL"], q_names=["DISCBD"])
    cases = [
        case("gm_error/crime~inc+hoval/rook_binary", "gm_error", crime, [inc, hoval],
             ["INC", "HOVAL"], rook_b),
        case("gm_endog_error_het/crime~inc|hoval|discbd/queen_r", "gm_endog_error_het", crime,
             [inc], ["INC"], queen_r, **ex),
    ]
    for c in cases:
        c["first_step"] = first_step_curve(c)
    # Flat objective with binary weights: the weighted second step stops 1.6e-5 early.
    cases.append(case("gm_error_hom/crime~inc+hoval/rook_binary", "gm_error_hom", crime,
                      [inc, hoval], ["INC", "HOVAL"], rook_b))
    return cases


def clamp_cases():
    """The two +-0.9 clamps of spreg's heteroskedastic IV models.

    * `GM_Combo_Het` clamps the coefficient of `W y` (replicated by Spatex): strong dependence
      (rho = 0.93) makes the first-stage estimate exceed 0.9.
    * `GM_Endog_Error_Het` (no `W y`) clamps whichever endogenous coefficient is last (a spreg
      bug, not replicated). Each deviation case has a twin ("/unclamped") run with the clamp
      disabled, which is what Spatex computes.
    """
    geoms, attrs = columbus()
    queen_r = sparse_export(transformed_w(Graph.build_contiguity(geoms, rook=False), "r"))
    inc, crime, hoval, disc = attrs["INC"], attrs["CRIME"], attrs["HOVAL"], attrs["DISCBD"]
    ex = dict(yend=[hoval], q=[disc], yend_names=["HOVAL"], q_names=["DISCBD"])
    out = []
    for no_clamp in (False, True):
        out.append(
            case("gm_endog_error_het/step1c3" + ("/unclamped" if no_clamp else ""),
                 "gm_endog_error_het", crime, [inc], ["INC"], queen_r, step1c=True, max_iter=3,
                 no_clamp=no_clamp, **ex)
        )
    rng = np.random.default_rng(5)
    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(10) for c in range(20)])
    lat = transformed_w(Graph.build_contiguity(cells, rook=True), "r")
    spec = sparse_export(lat)
    n = 200
    x1 = rng.normal(size=n).round(6)
    eps = rng.normal(size=n)
    y = np.linalg.solve(np.eye(n) - 0.93 * lat.full()[0], 1.0 + x1 + eps).round(6)
    out.append(case("synthetic200/gm_combo_het/rho_clamp", "gm_combo_het", y, [x1], ["X1"], spec))
    out.append(case("synthetic200/gm_combo_het/rho_clamp/step1c", "gm_combo_het", y, [x1], ["X1"],
                    spec, step1c=True, max_iter=3))
    return out


def main():
    out = FIXTURES / "regression"
    fx = header("gm_optimizer", "Cases where spreg's L-BFGS-B stops away from the exact minimizer.")
    fx["cases"] = optimizer_cases()
    dump_json(out / "gm_optimizer.json", fx)
    fx = header("gm_clamp", "spreg's +-0.9 coefficient clamps in GM_Combo_Het / GM_Endog_Error_Het.")
    fx["cases"] = clamp_cases()
    dump_json(out / "gm_clamp.json", fx)
    fx = header("gm_columbus", "spreg TSLS and GM models on the Columbus data (queen/rook, HAC kernel).")
    fx["cases"] = columbus_cases()
    dump_json(out / "gm_columbus.json", fx)
    fx = header("gm_synthetic", "spreg GM models on a 10 x 20 rook lattice (n = 200).")
    fx["cases"] = synthetic_cases()
    dump_json(out / "gm_synthetic.json", fx)


if __name__ == "__main__":
    main()
