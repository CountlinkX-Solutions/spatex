"""Generate spreg GM_Lag_Regimes oracle fixtures (`fixtures/regression/gm_lag_regimes.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_gm_lag_regimes.py

Columbus (libpysal's bundled ``columbus.shp``): y = CRIME, exogenous INC / OPEN (/ EW), optional
endogenous HOVAL instrumented by DISCBD and PLUMB, and a queen (row-standardized) or rook
(binary) contiguity matrix built from the same polygons. Regimes: NSA (2), NSA * 2 + EW (4) and
terciles of the X coordinate (3).

Each case stores the options, the layout (`kr`, `kf`, `kryd`, `nr`), names, betas, covariance,
standard errors, z statistics, residuals, the reduced-form predictions, the Anselin-Kelejian test,
the pseudo R-squares, the impact multipliers and impacts, and two sets of Chow tests: `chow_*`
with the restriction matrix on the real coefficient layout (spreg's `wald_test` on a corrected R)
and `spreg_chow_*` as spreg's `Chow` returns them (they differ when exogenous columns are global
and endogenous ones, here also `W y`, vary by regime, flagged by `chow_layout_bug`; the script
asserts they agree otherwise). `regime_lag_sep` fits that run one regression per regime store each
regime's attributes under `multi`.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import spreg  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from libpysal.weights import Kernel  # noqa: E402
from spreg.regimes import wald_test  # noqa: E402
from spreg.sputils import _sp_effects, spmultiplier  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def pair(t):
    return [float(t[0]), float(t[1])]


def impacts_of(reg, w, methods):
    """spreg's `_summary_impacts` effects for the regime-specific / global `x` coefficients (the
    non-constant `x` columns, in coefficient order)."""
    idx = [i for i, n in enumerate(reg.name_x) if not n.endswith("_CONSTANT")]
    variables = pd.DataFrame({"var_names": [reg.name_x[i] for i in idx]}, index=idx)
    rho = float(np.asarray(reg.rho).ravel()[0])
    if abs(rho) >= 1:
        return None
    out = {}
    for method in methods:
        spmult = spmultiplier(w, rho, method=method)
        btot, bdir, bind = _sp_effects(reg, variables, spmult, 0)
        out[spmult["method"]] = [
            {"name": n, "direct": float(d[0]), "indirect": float(i[0]), "total": float(t[0])}
            for n, d, i, t in zip(variables["var_names"], bdir, bind, btot)
        ]
    return out


def attrs(reg):
    out = {
        "name_x": list(reg.name_x), "name_yend": list(reg.name_yend), "name_q": list(reg.name_q),
        "betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u), "predy": flat(reg.predy),
        "predy_e": flat(reg.predy_e), "e_pred": flat(reg.e_pred),
        "rho": float(np.asarray(reg.rho).ravel()[0]),
    }
    if hasattr(reg, "z_stat"):
        out["std_err"] = flat(reg.std_err)
        out["z_stat"] = [pair(z) for z in reg.z_stat]
        out["sig2"] = float(np.asarray(reg.sig2).ravel()[0])
        out["utu"] = float(np.asarray(reg.utu).ravel()[0])
        out["pr2"] = float(reg.pr2)
        out["varb"] = flat(reg.varb)
    if getattr(reg, "pr2_e", None) is not None:
        out["pr2_e"] = float(reg.pr2_e)
    if getattr(reg, "ak_test", None) is not None:
        out["ak_test"] = pair(reg.ak_test)
    if getattr(reg, "sp_multipliers", None):
        out["sp_multipliers"] = {m: [float(v) for v in t] for m, t in reg.sp_multipliers.items()}
    return out


def corrected_chow(reg, x_flags, y_flags):
    """Chow tests with the restriction matrix on the real coefficient order (x regime blocks, x
    globals, yend regime blocks, yend globals), tested with spreg's own `wald_test`."""
    nr = int(reg.nr)
    kxr, kyr = sum(x_flags), sum(y_flags)
    kxf, kyf = len(x_flags) - kxr, len(y_flags) - kyr
    total = nr * (kxr + kyr) + kxf + kyf
    betas = np.asarray(reg.betas, dtype=float).reshape(-1, 1)
    rows_by_var = []
    for base, stride, count in ((0, kxr, kxr), (nr * kxr + kxf, kyr, kyr)):
        for v in range(count):
            block = np.zeros((nr - 1, total))
            for j in range(nr - 1):
                block[j, base + v] = 1.0
                block[j, base + v + (j + 1) * stride] = -1.0
            rows_by_var.append(block)
    regi = np.array([wald_test(betas, r, np.zeros((r.shape[0], 1)), reg.vm) for r in rows_by_var])
    r_all = np.vstack(rows_by_var)
    joint = wald_test(betas, r_all, np.zeros((r_all.shape[0], 1)), reg.vm)
    return [float(joint[0]), float(joint[1])], regi.tolist()


def case(db, regimes, graphs, label, x, yend, q, rkey, wkey="queen", gwk=None, **opts):
    cols = lambda names: np.column_stack([db[n].to_numpy(dtype=float) for n in names])  # noqa: E731
    kw = dict(opts)
    if gwk is not None:
        kw["gwk"] = gwk
    w = graphs[wkey]
    reg = spreg.GM_Lag_Regimes(
        db["CRIME"].to_numpy(dtype=float).reshape(-1, 1), cols(x),
        regimes[rkey], yend=cols(yend) if yend else None, q=cols(q) if q else None, w=w,
        name_y="CRIME", name_x=list(x), name_yend=list(yend), name_q=list(q), **kw)
    multi = hasattr(reg, "multi")
    methods = kw.get("spat_impacts", "simple")
    methods = ["simple", "full", "power"] if methods == "all" else (
        [methods] if isinstance(methods, str) else list(methods or []))
    out = {"label": label, "regimes": rkey, "w": wkey, "x": list(x), "yend": list(yend),
           "q": list(q), "opts": opts, "gwk": gwk is not None,
           "kr": int(reg.kr), "kf": int(reg.kf), "kryd": int(reg.kryd), "nr": int(reg.nr),
           "regime_lag_sep": bool(reg.regime_lag_sep), "regime_err_sep": bool(reg.regime_err_sep),
           "warning": str(getattr(reg, "warning", None) or ""), "multi": None}
    kx = len(x)
    c2r = kw.get("cols2regi", "all")
    c2r = [True] * (kx + len(yend)) if c2r == "all" else list(c2r)
    if multi:
        out["chow_layout_bug"] = False
        out["chow_joint"], out["chow_regi"] = flat(reg.chow.joint), np.asarray(reg.chow.regi).tolist()
        out["spreg_chow_joint"], out["spreg_chow_regi"] = out["chow_joint"], out["chow_regi"]
        out["multi"] = {}
        for r, m in reg.multi.items():
            a = attrs(m)
            if methods and getattr(m, "sp_multipliers", None):
                a["impacts"] = impacts_of(m, m.w, methods)
            out["multi"][str(r)] = a
        out.update({"betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u),
                    "predy": flat(reg.predy), "predy_e": flat(reg.predy_e),
                    "e_pred": flat(reg.e_pred), "name_x": list(reg.name_x),
                    "name_yend": list(reg.name_yend), "name_q": list(reg.name_q)})
    else:
        lag_sep = bool(reg.regime_lag_sep)
        x_flags = [kw.get("constant_regi", "many") == "many"] + c2r[:kx]
        y_flags = c2r[kx:] + [lag_sep]
        joint, regi = corrected_chow(reg, x_flags, y_flags)
        affected = (len(x_flags) - sum(x_flags)) > 0 and sum(y_flags) > 0
        if not affected:
            assert np.allclose(joint, flat(reg.chow.joint), rtol=1e-8, atol=1e-12), label
            assert np.allclose(regi, reg.chow.regi, rtol=1e-8, atol=1e-12), label
        out.update({"chow_layout_bug": bool(affected), "chow_joint": joint, "chow_regi": regi,
                    "spreg_chow_joint": flat(reg.chow.joint),
                    "spreg_chow_regi": np.asarray(reg.chow.regi).tolist()})
        out.update(attrs(reg))
        if methods and getattr(reg, "sp_multipliers", None):
            # spreg multiplies with the (union) W it used; the stored multipliers carry the result.
            idx = [i for i, n in enumerate(reg.name_x) if not n.endswith("_CONSTANT")]
            out["impacts"] = {
                method: [
                    {"name": reg.name_x[i], "direct": float(adi * reg.betas[i][0]),
                     "indirect": float(aii * reg.betas[i][0]),
                     "total": float(ati * reg.betas[i][0])}
                    for i in idx
                ]
                for method, (adi, aii, ati) in reg.sp_multipliers.items()
            }
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
    queen_raw, rook_raw = queen_g.to_W(), rook_g.to_W()

    cent = np.column_stack([geoms.centroid.x, geoms.centroid.y])
    gwk = Kernel(cent, k=15, function="triangular", fixed=False, diagonal=True)
    gwk.transform = "o"

    plain = dict(x=["INC", "OPEN"], yend=[], q=[])
    endog = dict(x=["INC", "OPEN"], yend=["HOVAL"], q=["DISCBD", "PLUMB"])

    def c(label, rkey, base, **kw):
        return case(db, regimes, graphs, label, rkey=rkey, **{**base, **kw})

    cases = [
        c("nsa_pooled", "nsa", plain, robust=None),
        c("nsa_pooled_default_white", "nsa", plain),
        c("nsa_pooled_sig2n", "nsa", plain, robust=None, sig2n_k=True),
        c("nsa_pooled_hac", "nsa", plain, robust="hac", gwk=gwk),
        c("nsa_pooled_lags2", "nsa", plain, robust=None, w_lags=2),
        c("nsa_pooled_no_lag_q", "nsa", endog, robust=None, lag_q=False),
        c("nsa_pooled_lags2_impacts_all", "nsa", endog, robust=None, w_lags=2, spat_impacts="all"),
        c("nsa_pooled_no_diag", "nsa", plain, robust=None, spat_diag=False, spat_impacts=None),
        c("nsa_pooled_rook_binary", "nsa", plain, wkey="rook_binary", robust=None),
        c("nsa_endog_pooled", "nsa", endog, robust=None),
        c("nsa_endog_pooled_white", "nsa", endog, robust="white"),
        c("nsa_endog_one_constant", "nsa", endog, robust=None, constant_regi="one"),
        c("nsa_endog_mixed", "nsa", endog, robust=None, cols2regi=[True, False, True]),
        c("nsa_endog_global_yend", "nsa", endog, robust=None, cols2regi=[True, True, False]),
        c("nsa_endog_err_sep_only", "nsa", endog, robust=None, regime_err_sep=True),
        c("four_pooled", "four", plain, robust=None),
        c("three_endog_pooled_white", "three", endog, robust="white"),
        # regime_lag_sep: one regression per regime when everything varies by regime
        c("nsa_lagsep", "nsa", plain, robust=None, regime_lag_sep=True),
        c("nsa_lagsep_default_white", "nsa", plain, regime_lag_sep=True),
        c("nsa_lagsep_sig2n_lags2", "nsa", plain, robust=None, regime_lag_sep=True,
          sig2n_k=True, w_lags=2, spat_impacts="all"),
        c("nsa_lagsep_endog", "nsa", endog, robust=None, regime_lag_sep=True),
        c("nsa_lagsep_endog_no_lag_q", "nsa", endog, robust="white", regime_lag_sep=True,
          lag_q=False),
        c("nsa_lagsep_rook_binary", "nsa", plain, wkey="rook_binary", robust=None,
          regime_lag_sep=True),
        c("nsa_lagsep_no_diag", "nsa", plain, robust=None, regime_lag_sep=True, spat_diag=False,
          spat_impacts=None),
        c("three_lagsep", "three", plain, robust=None, regime_lag_sep=True),
        c("four_lagsep", "four", plain, robust=None, regime_lag_sep=True),
        # regime_lag_sep that cannot run separate regressions: pooled on the union of subgraphs
        c("nsa_lagsep_one_constant", "nsa", plain, robust=None, regime_lag_sep=True,
          constant_regi="one"),
        c("nsa_lagsep_endog_one_constant", "nsa", endog, robust=None, regime_lag_sep=True,
          constant_regi="one"),
        c("nsa_lagsep_mixed", "nsa", endog, robust=None, regime_lag_sep=True,
          cols2regi=[True, False, True]),
        c("nsa_lagsep_hac", "nsa", plain, robust="hac", gwk=gwk, regime_lag_sep=True),
        c("nsa_lagsep_mixed_rook_binary", "nsa", plain, wkey="rook_binary", robust="white",
          regime_lag_sep=True, cols2regi=[False, True]),
        c("four_lagsep_mixed", "four", plain, robust=None, regime_lag_sep=True,
          cols2regi=[True, False]),
    ]
    dump_json(FIXTURES / "regression" / "gm_lag_regimes.json", {
        "name": "gm_lag_regimes",
        "description": "spreg GM_Lag_Regimes on columbus (CRIME ~ INC + OPEN [| HOVAL <- DISCBD + PLUMB] + W CRIME)",
        "meta": info,
        "y": flat(db["CRIME"]),
        "columns": {n: flat(db[n]) for n in ("INC", "OPEN", "EW", "HOVAL", "DISCBD", "PLUMB")},
        "regimes": regimes,
        "graphs": {"queen": sparse_export(queen_raw), "rook_binary": sparse_export(rook_raw)},
        "gwk": sparse_export(gwk), "cases": cases,
    })


if __name__ == "__main__":
    main()
