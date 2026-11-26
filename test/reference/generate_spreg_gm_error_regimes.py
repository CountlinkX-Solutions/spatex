"""Generate spreg GM_Error_Regimes / GM_Endog_Error_Regimes / GM_Combo_Regimes oracle fixtures
(`fixtures/regression/gm_error_regimes.json`), KP98 estimator.

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_gm_error_regimes.py

Columbus (libpysal's bundled ``columbus.shp``) as in ``generate_spreg_gm_lag_regimes.py``: y =
CRIME, exogenous INC / OPEN, optional endogenous HOVAL instrumented by DISCBD and PLUMB, a queen
(row-standardized) or rook (binary) contiguity matrix, regimes NSA (2), NSA * 2 + EW (4) and
terciles of X (3).

Each case stores the class, options, layout (`kr`, `kf`, `kryd`, `nr`), names, betas (lambda last,
`rho` before it for combos), covariance (without lambda), standard errors, z statistics,
residuals, filtered residuals, predictions, pseudo R-squares and two sets of Chow tests: `chow_*`
with the restriction matrix on the real coefficient layout (spreg's `wald_test` on a corrected R)
and `spreg_chow_*` as spreg's `Chow` returns them (they differ when exogenous columns are global
and endogenous ones, here also `W y`, vary by regime, flagged by `chow_layout_bug`; the script
asserts they agree otherwise). Runs with `regime_err_sep` / `regime_lag_sep` store each regime's
attributes under `multi`.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from libpysal.graph import Graph  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402
from regimes_common import corrected_chow, flat, pair  # noqa: E402,F401

warnings.simplefilter("ignore")

OPTIM = []


def optimizer_record(moments, lam):
    """spreg's lambda next to the exact minimizer of its own KP98 moment objective (sigma^2
    profiled out, spreg's scaling): spreg's L-BFGS-B stops early, and far from the minimizer
    when the objective is flat (binary weights), see docs/parity.md "spreg GM notes"."""
    from scipy.optimize import minimize_scalar

    g_mat, g = np.asarray(moments[0], dtype=float), np.asarray(moments[1], dtype=float)
    scale = np.min([[np.min(g_mat), np.min(g)]])
    g_mat, g = g_mat / scale, g / scale

    def objective(x):
        r = g - g_mat[:, :2] @ np.array([[x, x**2]]).T
        col = g_mat[:, 2:3]
        s2 = max(0.0, float(((col.T @ r) / (col.T @ col)).item()))
        return float(np.sum((r - col * s2) ** 2))

    grid = np.linspace(-0.99, 0.99, 199)
    # the basin spreg starts in: walk downhill on the grid from 0
    i = 99
    vals = [objective(x) for x in grid]
    while True:
        left = vals[i - 1] if i > 0 else np.inf
        right = vals[i + 1] if i < 198 else np.inf
        if left < vals[i] and left <= right:
            i -= 1
        elif right < vals[i]:
            i += 1
        else:
            break
    res = minimize_scalar(objective, bounds=(max(grid[i] - 0.01, -0.99), min(grid[i] + 0.01, 0.99)),
                          method="bounded", options={"xatol": 1e-12})
    return {"lam": float(lam), "f": objective(lam), "lam_exact": float(res.x),
            "f_exact": float(res.fun)}


def record_optim_moments():
    """Wrap every reference to `optim_moments` so each call stores `optimizer_record`."""
    import spreg.utils as utils

    original = utils.optim_moments

    def wrapped(moments, *args, **kwargs):
        out = original(moments, *args, **kwargs)
        OPTIM.append(optimizer_record(moments, np.asarray(out).ravel()[0]))
        return out

    for name, mod in list(sys.modules.items()):
        if name.startswith("spreg") and getattr(mod, "optim_moments", None) is original:
            mod.optim_moments = wrapped


record_optim_moments()


def attrs(reg):
    out = {"betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u),
           "predy": flat(reg.predy), "e_filtered": flat(reg.e_filtered),
           "name_x": list(reg.name_x)}
    for key in ("name_yend", "name_q", "name_z"):
        if hasattr(reg, key):
            out[key] = list(getattr(reg, key))
    if hasattr(reg, "z_stat"):
        out["std_err"] = flat(reg.std_err)
        out["z_stat"] = [pair(z) for z in reg.z_stat]
        out["sig2"] = float(np.asarray(reg.sig2).ravel()[0])
        out["pr2"] = float(reg.pr2)
    if getattr(reg, "rho", None) is not None:
        out["rho"] = float(np.asarray(reg.rho).ravel()[0])
        out["predy_e"] = flat(reg.predy_e)
        out["e_pred"] = flat(reg.e_pred)
        out["pr2_e"] = float(reg.pr2_e) if hasattr(reg, "pr2_e") else None  # rho outside (-1, 1)
    return out


def case(db, regimes, graphs, label, klass, x, yend, q, rkey, wkey="queen", **opts):
    cols = lambda names: np.column_stack([db[n].to_numpy(dtype=float) for n in names])  # noqa: E731
    y = db["CRIME"].to_numpy(dtype=float).reshape(-1, 1)
    w = graphs[wkey]
    OPTIM.clear()
    names = dict(name_y="CRIME", name_x=list(x))
    if klass == "error":
        reg = spreg.GM_Error_Regimes(y, cols(x), regimes[rkey], w, **names, **opts)
    elif klass == "endog":
        reg = spreg.GM_Endog_Error_Regimes(
            y, cols(x), cols(yend), cols(q), regimes[rkey], w, name_yend=list(yend),
            name_q=list(q), **names, **opts)
    else:
        reg = spreg.GM_Combo_Regimes(
            y, cols(x), regimes[rkey], yend=cols(yend) if yend else None,
            q=cols(q) if q else None, w=w, name_yend=list(yend), name_q=list(q), **names, **opts)
    optim = list(OPTIM)
    kx = len(x)
    c2r = opts.get("cols2regi", "all")
    c2r = [True] * (kx + len(yend)) if c2r == "all" else list(c2r)
    sep = hasattr(reg, "multi")
    out = {"label": label, "class": klass, "regimes": rkey, "w": wkey, "x": list(x),
           "yend": list(yend), "q": list(q), "opts": opts,
           "kr": int(reg.kr), "kf": int(reg.kf), "kryd": int(reg.kryd), "nr": int(reg.nr),
           "regime_err_sep": bool(reg.regime_err_sep),
           "regime_lag_sep": bool(getattr(reg, "regime_lag_sep", False)),
           "warning": str(getattr(reg, "warning", None) or ""), "multi": None,
           "optim": optim}
    if sep:
        out["chow_layout_bug"] = False
        out["chow_joint"], out["chow_regi"] = flat(reg.chow.joint), np.asarray(reg.chow.regi).tolist()
        out["spreg_chow_joint"], out["spreg_chow_regi"] = out["chow_joint"], out["chow_regi"]
        out["multi"] = {str(r): attrs(m) for r, m in reg.multi.items()}
        out.update({"betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u),
                    "predy": flat(reg.predy), "e_filtered": flat(reg.e_filtered),
                    "name_x": list(reg.name_x)})
        for key in ("name_yend", "name_q", "name_z"):
            if hasattr(reg, key):
                out[key] = list(getattr(reg, key))
        if klass == "combo":
            out["predy_e"], out["e_pred"] = flat(reg.predy_e), flat(reg.e_pred)
    else:
        x_flags = [opts.get("constant_regi", "many") == "many"] + c2r[:kx]
        y_flags = c2r[kx:] + ([False] if klass == "combo" else [])
        joint, regi = corrected_chow(reg, x_flags, y_flags)
        affected = (len(x_flags) - sum(x_flags)) > 0 and sum(y_flags) > 0
        if not affected:
            assert np.allclose(joint, flat(reg.chow.joint), rtol=1e-8, atol=1e-12), label
            assert np.allclose(regi, reg.chow.regi, rtol=1e-8, atol=1e-12), label
        out.update({"chow_layout_bug": bool(affected), "chow_joint": joint, "chow_regi": regi,
                    "spreg_chow_joint": flat(reg.chow.joint),
                    "spreg_chow_regi": np.asarray(reg.chow.regi).tolist()})
        out.update(attrs(reg))
    return out


def main():
    info = meta()
    info["spreg"] = spreg.__version__
    db = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    nsa = [int(v) for v in db["NSA"]]
    four = [int(a) * 2 + int(b) for a, b in zip(db["NSA"], db["EW"])]
    xc = db["X"].to_numpy(dtype=float)
    cuts = np.quantile(xc, [1 / 3, 2 / 3])
    three = [int(v) for v in np.searchsorted(cuts, xc)]
    regimes = {"nsa": nsa, "four": four, "three": three}

    geoms = db.geometry
    queen_g = Graph.build_contiguity(geoms, rook=False)
    rook_g = Graph.build_contiguity(geoms, rook=True)
    queen = queen_g.to_W()
    queen.transform = "r"
    rook = rook_g.to_W()
    graphs = {"queen": queen, "rook_binary": rook}

    plain = dict(x=["INC", "OPEN"], yend=[], q=[])
    endog = dict(x=["INC", "OPEN"], yend=["HOVAL"], q=["DISCBD", "PLUMB"])

    def c(label, klass, rkey, base, **kw):
        return case(db, regimes, graphs, label, klass, rkey=rkey, **{**base, **kw})

    cases = [
        # GM_Error_Regimes
        c("error_nsa_pooled", "error", "nsa", plain),
        c("error_nsa_one_constant", "error", "nsa", plain, constant_regi="one"),
        c("error_nsa_mixed", "error", "nsa", plain, cols2regi=[True, False]),
        c("error_nsa_global_x", "error", "nsa", plain, cols2regi=[False, False]),
        c("error_nsa_rook_binary", "error", "nsa", plain, wkey="rook_binary"),
        c("error_four_pooled", "error", "four", plain),
        c("error_three_one_constant", "error", "three", plain, constant_regi="one",
          cols2regi=[True, False]),
        c("error_nsa_sep", "error", "nsa", plain, regime_err_sep=True),
        c("error_nsa_sep_rook_binary", "error", "nsa", plain, wkey="rook_binary",
          regime_err_sep=True),
        c("error_three_sep", "error", "three", plain, regime_err_sep=True),
        c("error_four_sep", "error", "four", plain, regime_err_sep=True),
        # GM_Endog_Error_Regimes
        c("endog_nsa_pooled", "endog", "nsa", endog),
        c("endog_nsa_one_constant", "endog", "nsa", endog, constant_regi="one"),
        c("endog_nsa_mixed", "endog", "nsa", endog, cols2regi=[True, False, True]),
        c("endog_nsa_global_yend", "endog", "nsa", endog, cols2regi=[True, True, False]),
        c("endog_nsa_rook_binary", "endog", "nsa", endog, wkey="rook_binary"),
        c("endog_four_pooled", "endog", "four", endog),
        c("endog_three_one_constant", "endog", "three", endog, constant_regi="one"),
        c("endog_nsa_sep", "endog", "nsa", endog, regime_err_sep=True),
        c("endog_three_sep", "endog", "three", endog, regime_err_sep=True),
        c("endog_nsa_sep_rook_binary", "endog", "nsa", endog, wkey="rook_binary",
          regime_err_sep=True),
        # GM_Combo_Regimes
        c("combo_nsa_pooled", "combo", "nsa", plain),
        c("combo_nsa_pooled_lags2", "combo", "nsa", plain, w_lags=2),
        c("combo_nsa_pooled_no_lag_q", "combo", "nsa", endog, lag_q=False),
        c("combo_nsa_endog_pooled", "combo", "nsa", endog),
        c("combo_nsa_endog_one_constant", "combo", "nsa", endog, constant_regi="one"),
        c("combo_nsa_one_constant", "combo", "nsa", plain, constant_regi="one"),
        c("combo_nsa_mixed", "combo", "nsa", endog, cols2regi=[True, False, True]),
        c("combo_nsa_global_x", "combo", "nsa", plain, cols2regi=[False, True]),
        c("combo_nsa_rook_binary", "combo", "nsa", plain, wkey="rook_binary"),
        c("combo_four_pooled", "combo", "four", plain),
        c("combo_three_endog_pooled", "combo", "three", endog),
        c("combo_nsa_err_sep_only", "combo", "nsa", plain, regime_err_sep=True),
        c("combo_nsa_lagsep", "combo", "nsa", plain, regime_lag_sep=True),
        c("combo_nsa_lagsep_lags2", "combo", "nsa", plain, regime_lag_sep=True, w_lags=2),
        c("combo_nsa_lagsep_endog", "combo", "nsa", endog, regime_lag_sep=True),
        c("combo_nsa_lagsep_no_lag_q", "combo", "nsa", endog, regime_lag_sep=True, lag_q=False),
        c("combo_nsa_lagsep_rook_binary", "combo", "nsa", plain, wkey="rook_binary",
          regime_lag_sep=True),
        c("combo_nsa_lagsep_mixed", "combo", "nsa", plain, regime_lag_sep=True,
          cols2regi=[True, False]),
        c("combo_three_lagsep", "combo", "three", plain, regime_lag_sep=True),
        c("combo_four_lagsep", "combo", "four", plain, regime_lag_sep=True),
    ]
    dump_json(FIXTURES / "regression" / "gm_error_regimes.json", {
        "name": "gm_error_regimes",
        "description": "spreg GM_Error_Regimes / GM_Endog_Error_Regimes / GM_Combo_Regimes (KP98) on columbus (CRIME ~ INC + OPEN [| HOVAL <- DISCBD + PLUMB] [+ W CRIME])",
        "meta": info,
        "y": flat(db["CRIME"]),
        "columns": {n: flat(db[n]) for n in ("INC", "OPEN", "EW", "HOVAL", "DISCBD", "PLUMB")},
        "regimes": regimes,
        "graphs": {"queen": sparse_export(queen_g.to_W()), "rook_binary": sparse_export(rook_g.to_W())},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
