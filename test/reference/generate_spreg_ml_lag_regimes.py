"""Generate spreg ML_Lag_Regimes oracle fixtures (`fixtures/regression/ml_lag_regimes.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_ml_lag_regimes.py

Columbus (libpysal's bundled ``columbus.shp``): y = CRIME, exogenous INC / OPEN (/ HOVAL), a queen
(row-standardized) or rook (binary) contiguity matrix built from the same polygons. Regimes: NSA
(2), NSA * 2 + EW (4) and terciles of the X coordinate (3).

Each case stores the options, the layout (`kr`, `kf`, `nr`), names, betas, covariance, standard
errors, z statistics, the likelihood-based diagnostics, residuals, the reduced-form predictions,
the pseudo R-squares, the impact multipliers and impacts, and the Chow tests. `ML_Lag_Regimes`
has two shapes: one ML fit on the regime design with a single `rho` (`regime_lag_sep=False`; the
top-level object carries everything), or, with `regime_lag_sep=True` (which needs every column by
regime and a constant per regime), one `BaseML_Lag` fit per regime on the regime's own subgraph;
there the top level only carries the stacked betas, the block-diagonal `vm`, the residuals and
predictions and the Chow tests, and every regime's attributes are stored under `multi`.

Chow: spreg's `Chow` puts the global columns last, and `rho` is always the last global (pooled) or
a regime-specific column (separate), so unlike the 2SLS regimes models the restrictions are on the
real coefficient layout; the script asserts that spreg's Chow equals one built on the real layout
(`regimes_common.corrected_chow`) in every case.

`method='ord'`: spreg's variance substitutes the symmetrized matrix for W whenever the binary
pattern is symmetric (known deviation, see `docs/parity.md`). Every `ord` case therefore also
stores the covariance-derived quantities of the same case run with `method='full'` as `full_ref`.
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
from regimes_common import corrected_chow, flat, pair  # noqa: E402

warnings.simplefilter("ignore")


def make_w(graph, transform):
    w = graph.to_W()
    w.transform = transform
    return w


def attrs(reg):
    """Attributes of one ML fit (pooled top-level object or one regime of the separate fits)."""
    out = {
        "betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u), "predy": flat(reg.predy),
        "predy_e": flat(reg.predy_e), "e_pred": flat(reg.e_pred),
        "name_x": list(reg.name_x),
    }
    if hasattr(reg, "z_stat"):
        out["std_err"] = flat(reg.std_err)
        out["z_stat"] = [pair(z) for z in reg.z_stat]
        out["vm1"] = flat(reg.vm1)
        out["rho"] = float(np.asarray(reg.rho).ravel()[0])
        for key in ("logll", "aic", "schwarz", "sig2", "utu", "pr2", "mean_y", "std_y"):
            out[key] = float(np.asarray(getattr(reg, key)).ravel()[0])
        out["k"] = int(reg.k)
        out["n"] = int(reg.n)
        if getattr(reg, "pr2_e", None) is not None:
            out["pr2_e"] = float(reg.pr2_e)
    if getattr(reg, "sp_multipliers", None):
        out["sp_multipliers"] = {m: [float(v) for v in t] for m, t in reg.sp_multipliers.items()}
    return out


def impacts_of(reg, names):
    """spreg's `_summary_impacts` effects: multiplier times the slope of every non-constant `x`
    coefficient (names ending `CONSTANT` and `rho` are skipped), in coefficient order."""
    if not getattr(reg, "sp_multipliers", None):
        return None
    idx = [i for i, n in enumerate(names[:-1]) if not n.endswith("CONSTANT")]
    return {
        method: [
            {"name": names[i], "direct": float(adi * reg.betas[i][0]),
             "indirect": float(aii * reg.betas[i][0]), "total": float(ati * reg.betas[i][0])}
            for i in idx
        ]
        for method, (adi, aii, ati) in reg.sp_multipliers.items()
    }


def chow_of(reg, x_flags, y_flags):
    """spreg's Chow tests, asserted equal to the ones on the real coefficient layout."""
    joint, regi = corrected_chow(reg, x_flags, y_flags)
    assert np.allclose(joint, flat(reg.chow.joint), rtol=1e-8, atol=1e-12)
    assert np.allclose(regi, reg.chow.regi, rtol=1e-8, atol=1e-12)
    return flat(reg.chow.joint), np.asarray(reg.chow.regi).tolist()


def fit(db, regimes, w, x, **kw):
    cols = np.column_stack([db[n].to_numpy(dtype=float) for n in x])
    return spreg.ML_Lag_Regimes(
        db["CRIME"].to_numpy(dtype=float).reshape(-1, 1), cols, regimes, w=w,
        name_y="CRIME", name_x=list(x), **kw)


def variance_part(out, multi):
    keys = ("vm", "vm1", "std_err", "z_stat")
    part = {k: out[k] for k in ("vm", "chow_joint", "chow_regi") if k in out}
    if multi:
        part["multi"] = {r: {k: m[k] for k in keys} for r, m in out["multi"].items()}
    else:
        part.update({k: out[k] for k in keys})
    return part


def case(db, regimes, graphs, label, rkey, x=("INC", "OPEN"), wkey="queen", **opts):
    kw = dict(opts)
    transform = "o" if wkey == "rook_binary" else "r"
    reg = fit(db, regimes[rkey], make_w(graphs[wkey], transform), list(x), **kw)
    multi = hasattr(reg, "multi")
    methods = kw.get("spat_impacts", "simple")
    kx = len(x)
    c2r = kw.get("cols2regi", "all")
    c2r = [True] * kx if c2r == "all" else list(c2r)
    out = {"label": label, "regimes": rkey, "w": wkey, "x": list(x), "opts": opts,
           "kr": int(reg.kr), "kf": int(reg.kf), "nr": int(reg.nr),
           "regime_lag_sep": bool(reg.regime_lag_sep),
           "warning": str(getattr(reg, "warning", None) or ""), "multi": None}
    if multi:
        out["chow_joint"], out["chow_regi"] = chow_of(reg, [True] * int(reg.kr), [])
        out["multi"] = {}
        for r, m in reg.multi.items():
            a = attrs(m)
            if methods:
                a["impacts"] = impacts_of(m, list(m.name_x))
            out["multi"][str(r)] = a
        out.update({
            "betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u),
            "predy": flat(reg.predy), "predy_e": flat(reg.predy_e), "e_pred": flat(reg.e_pred),
            "name_x": list(reg.name_x), "utu": float(np.asarray(reg.utu).ravel()[0]),
            "mean_y": float(reg.mean_y), "std_y": float(reg.std_y),
        })
    else:
        x_flags = [kw.get("constant_regi", "many") == "many"] + c2r
        out["chow_joint"], out["chow_regi"] = chow_of(reg, x_flags, [False])
        out.update(attrs(reg))
        if methods:
            out["impacts"] = impacts_of(reg, list(reg.name_x))
    if kw.get("method") == "ord":
        full = {**opts, "method": "full"}
        ref = fit(db, regimes[rkey], make_w(graphs[wkey], transform), list(x), **full)
        ref_out = {"multi": {}} if multi else {}
        if multi:
            ref_out["vm"] = flat(ref.vm)
            ref_out["chow_joint"], ref_out["chow_regi"] = chow_of(ref, [True] * int(ref.kr), [])
            for r, m in ref.multi.items():
                ref_out["multi"][str(r)] = attrs(m)
        else:
            ref_out.update(attrs(ref))
            ref_out["chow_joint"], ref_out["chow_regi"] = chow_of(ref, x_flags, [False])
        out["full_ref"] = variance_part(ref_out, multi)
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
    graphs = {"queen": queen_g, "rook_binary": rook_g}

    def c(label, rkey, **kw):
        return case(db, regimes, graphs, label, rkey, **kw)

    three_x = ("INC", "OPEN", "HOVAL")
    cases = [
        # one fit on the regime design, a single rho
        c("nsa_pooled", "nsa"),
        c("nsa_pooled_impacts_all", "nsa", spat_impacts="all"),
        c("nsa_pooled_impacts_full", "nsa", spat_impacts="full"),
        c("nsa_pooled_no_impacts", "nsa", spat_impacts=None),
        c("nsa_pooled_ord", "nsa", method="ord"),
        c("nsa_pooled_lu", "nsa", method="lu"),
        c("nsa_pooled_epsilon", "nsa", epsilon=1e-5),
        c("nsa_pooled_one_constant", "nsa", constant_regi="one"),
        c("nsa_pooled_mixed", "nsa", cols2regi=[True, False]),
        c("nsa_pooled_mixed_open", "nsa", cols2regi=[False, True]),
        c("nsa_pooled_one_constant_mixed", "nsa", constant_regi="one", cols2regi=[True, False]),
        c("nsa_pooled_global_slopes", "nsa", cols2regi=[False, False]),
        c("nsa_pooled_rook_binary", "nsa", wkey="rook_binary"),
        c("nsa_pooled_three_x", "nsa", x=three_x),
        c("nsa_pooled_three_x_mixed", "nsa", x=three_x, cols2regi=[True, False, True],
          spat_impacts="all"),
        c("four_pooled", "four"),
        c("three_pooled_ord", "three", method="ord"),
        c("three_pooled_mixed", "three", cols2regi=[False, True]),
        # regime_lag_sep: one ML fit per regime on the regime's subgraph
        c("nsa_lagsep", "nsa", regime_lag_sep=True),
        c("nsa_lagsep_impacts_all", "nsa", regime_lag_sep=True, spat_impacts="all"),
        c("nsa_lagsep_no_impacts", "nsa", regime_lag_sep=True, spat_impacts=None),
        c("nsa_lagsep_ord", "nsa", regime_lag_sep=True, method="ord"),
        c("nsa_lagsep_lu", "nsa", regime_lag_sep=True, method="lu"),
        c("nsa_lagsep_epsilon", "nsa", regime_lag_sep=True, epsilon=1e-5),
        c("nsa_lagsep_rook_binary", "nsa", wkey="rook_binary", regime_lag_sep=True),
        c("nsa_lagsep_three_x", "nsa", x=three_x, regime_lag_sep=True),
        c("three_lagsep", "three", regime_lag_sep=True),
        c("three_lagsep_ord", "three", regime_lag_sep=True, method="ord"),
        c("four_lagsep", "four", regime_lag_sep=True),
    ]
    dump_json(FIXTURES / "regression" / "ml_lag_regimes.json", {
        "name": "ml_lag_regimes",
        "description": "spreg ML_Lag_Regimes on columbus (CRIME ~ INC + OPEN [+ HOVAL] + W CRIME)",
        "meta": info,
        "y": flat(db["CRIME"]),
        "columns": {n: flat(db[n]) for n in ("INC", "OPEN", "HOVAL")},
        "regimes": regimes,
        "graphs": {"queen": sparse_export(make_w(queen_g, "o")),
                   "rook_binary": sparse_export(make_w(rook_g, "o"))},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
