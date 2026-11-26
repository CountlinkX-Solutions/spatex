"""Generate spreg GM_{Error,Endog_Error,Combo}_{Het,Hom}_Regimes oracle fixtures
(`fixtures/regression/gm_error_het_hom_regimes.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I \
        test/reference/generate_spreg_gm_error_het_hom_regimes.py

Same data and weights as ``generate_spreg_gm_error_regimes.py`` (Columbus: y = CRIME, exogenous
INC / OPEN, optional endogenous HOVAL instrumented by DISCBD and PLUMB, queen row-standardized or
rook binary contiguity, regimes NSA (2), NSA * 2 + EW (4) and terciles of X (3)) for the six
classes of ``error_sp_het_regimes.py`` and ``error_sp_hom_regimes.py``.

Each case stores the class, estimator, options, layout (``kr``, ``kf``, ``kryd``, ``nr``), names,
betas (lambda last, ``rho`` before it for combos), covariance (with lambda), standard errors,
z statistics, residuals, predictions and the Chow tests in two forms: ``chow_*`` with the
restriction matrix on the real coefficient layout (spreg's ``wald_test`` on a corrected R) and
``spreg_chow_*`` as spreg's ``Chow`` returns them (they differ when exogenous columns are global
and endogenous ones, here also ``W y``, vary by regime, flagged by ``chow_layout_bug``; the
script asserts they agree otherwise). Runs with ``regime_err_sep`` / ``regime_lag_sep`` store each
regime's attributes under ``multi``.

Separate ``GM_Endog_Error_Het`` fits clamp the coefficient of the last endogenous column to +-0.9
(a spreg defect, ``docs/parity.md`` "spreg GM notes"); every such case has a twin
``<label>/unclamped`` run with the clamp disabled, flagged ``clamped`` on the original, which is
what Spatex reproduces. (For the combos the clamped coefficient is ``rho``, replicated.)

``lambdas`` lists every lambda spreg's ``optim_moments`` returned, in call order (for the
separate fits also ``lambdas_by_regime``): Spatex's internal ``*_at`` hook consumes them and then
agrees with spreg at 1e-9 whatever the optimizer did. ``optim`` stores, per call, spreg's lambda,
the exact minimizer of the objective spreg minimizes (weights and scale included) and both
objective values: L-BFGS-B stops early on flat objectives (binary weights) and can end where the
objective is increasing.
"""

import copy
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import numpy.linalg as la  # noqa: E402
import spreg  # noqa: E402
from libpysal.graph import Graph  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402
from regimes_common import corrected_chow, flat, pair  # noqa: E402

warnings.simplefilter("ignore")

OPTIM = []


def _abs_without_clamp(x):
    """`abs` for `error_sp_het` (as in generate_spreg_gm.py): a length-1 array reads as 0, so the
    clamp of the last endogenous coefficient to +-0.9 never fires; scalars behave normally."""
    if isinstance(x, np.ndarray) and x.shape == (1,):
        return 0.0
    return abs(x)


def optimizer_record(moments_in, vcx, lam):
    """spreg's lambda next to the exact minimizer of the objective `optim_moments` minimizes."""
    from scipy.optimize import minimize_scalar

    moments = copy.deepcopy(moments_in)
    if np.asarray(vcx).any():
        ec = np.transpose(la.cholesky(la.inv(vcx)))
        moments[0] = np.dot(ec, moments_in[0])
        moments[1] = np.dot(ec, moments_in[1])
    scale = np.min([[np.min(moments[0]), np.min(moments[1])]])
    g_mat, g = moments[0] / scale, moments[1] / scale

    def objective(x):
        r = g - g_mat[:, :2] @ np.array([[x, x**2]]).T
        return float(np.sum(r**2))

    grid = np.linspace(-0.99, 0.99, 199)
    vals = [objective(x) for x in grid]
    i = 99  # the basin spreg starts in: walk downhill on the grid from 0
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

    def wrapped(moments, vcx=np.array([0]), *args, **kwargs):
        out = original(moments, vcx, *args, **kwargs)
        OPTIM.append(optimizer_record(moments, vcx, np.asarray(out).ravel()[0]))
        return out

    for name, mod in list(sys.modules.items()):
        if name.startswith("spreg") and getattr(mod, "optim_moments", None) is original:
            mod.optim_moments = wrapped


record_optim_moments()


def attrs(reg):
    out = {"betas": flat(reg.betas), "vm": flat(reg.vm), "u": flat(reg.u),
           "predy": flat(reg.predy), "e_filtered": flat(reg.e_filtered),
           "name_x": list(reg.name_x), "iteration": int(reg.iteration)}
    for key in ("name_yend", "name_q", "name_z"):
        if hasattr(reg, key):
            out[key] = list(getattr(reg, key))
    if hasattr(reg, "z_stat"):
        out["std_err"] = flat(reg.std_err)
        out["z_stat"] = [pair(z) for z in reg.z_stat]
    out["sig2"] = float(np.asarray(reg.sig2).ravel()[0]) if hasattr(reg, "sig2") else None
    out["pr2"] = float(reg.pr2) if hasattr(reg, "pr2") else None
    if getattr(reg, "rho", None) is not None:
        out["rho"] = float(np.asarray(reg.rho).ravel()[0])
        out["predy_e"] = flat(reg.predy_e)
        out["e_pred"] = flat(reg.e_pred)
        out["pr2_e"] = float(reg.pr2_e) if hasattr(reg, "pr2_e") else None  # rho outside (-1, 1)
    return out


def build(est, klass, y, cols, x, yend, q, regimes, w, opts):
    names = dict(name_y="CRIME", name_x=list(x))
    het = est == "het"
    if klass == "error":
        cls = spreg.GM_Error_Het_Regimes if het else spreg.GM_Error_Hom_Regimes
        return cls(y, cols(x), regimes, w, **names, **opts)
    if klass == "endog":
        cls = spreg.GM_Endog_Error_Het_Regimes if het else spreg.GM_Endog_Error_Hom_Regimes
        return cls(y, cols(x), cols(yend), cols(q), regimes, w, name_yend=list(yend),
                   name_q=list(q), **names, **opts)
    cls = spreg.GM_Combo_Het_Regimes if het else spreg.GM_Combo_Hom_Regimes
    return cls(y, cols(x), regimes, yend=cols(yend) if yend else None,
               q=cols(q) if q else None, w=w, name_yend=list(yend), name_q=list(q),
               **names, **opts)


def split_by_regime(est, reg, optim):
    """The `optim` calls of every regime's fit, in regime order."""
    pieces, at = {}, 0
    for r, m in reg.multi.items():
        count = int(m.iteration) + (2 if est == "het" and reg.step1c else 1)
        pieces[str(r)] = optim[at:at + count]
        at += count
    assert at == len(optim)
    return pieces


def clamped(klass, est, reg):
    """spreg's separate `BaseGM_Endog_Error_Het` fits clamp their last endogenous coefficient to
    +-0.9 (for a combo that is rho, which Spatex replicates; for `yend` it is a spreg bug)."""
    if klass != "endog" or est != "het" or not hasattr(reg, "multi"):
        return False
    return any(abs(abs(float(np.asarray(m.betas).ravel()[-2])) - 0.9) < 1e-12
               for m in reg.multi.values())


def case(db, regimes, graphs, label, est, klass, x, yend, q, rkey, wkey="queen", no_clamp=False,
         **opts):
    cols = lambda names: np.column_stack([db[n].to_numpy(dtype=float) for n in names])  # noqa: E731
    y = db["CRIME"].to_numpy(dtype=float).reshape(-1, 1)
    OPTIM.clear()
    if no_clamp:
        import spreg.error_sp_het as het_mod

        het_mod.abs = _abs_without_clamp
        try:
            reg = build(est, klass, y, cols, x, yend, q, regimes[rkey], graphs[wkey], opts)
        finally:
            del het_mod.abs
    else:
        reg = build(est, klass, y, cols, x, yend, q, regimes[rkey], graphs[wkey], opts)
    optim = list(OPTIM)
    kx = len(x)
    c2r = opts.get("cols2regi", "all")
    c2r = [True] * (kx + len(yend)) if c2r == "all" else list(c2r)
    sep = hasattr(reg, "multi")
    out = {"label": label, "class": klass, "estimator": est, "regimes": rkey, "w": wkey,
           "x": list(x), "yend": list(yend), "q": list(q), "opts": opts,
           "kr": int(reg.kr), "kf": int(reg.kf), "kryd": int(reg.kryd), "nr": int(reg.nr),
           "regime_err_sep": bool(reg.regime_err_sep),
           "regime_lag_sep": bool(getattr(reg, "regime_lag_sep", False)),
           "no_clamp": no_clamp, "clamped": clamped(klass, est, reg),
           "warning": str(getattr(reg, "warning", None) or ""), "multi": None,
           "optim": optim, "lambdas": [o["lam"] for o in optim]}
    if sep:
        out["chow_layout_bug"] = False
        out["chow_joint"], out["chow_regi"] = flat(reg.chow.joint), np.asarray(reg.chow.regi).tolist()
        out["spreg_chow_joint"], out["spreg_chow_regi"] = out["chow_joint"], out["chow_regi"]
        out["multi"] = {str(r): attrs(m) for r, m in reg.multi.items()}
        out["optim_by_regime"] = split_by_regime(est, reg, optim)
        out["lambdas_by_regime"] = {r: [o["lam"] for o in v]
                                    for r, v in out["optim_by_regime"].items()}
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

    cases = []

    def c(est, label, klass, rkey, base, **kw):
        fit = case(db, regimes, graphs, f"{est}_{label}", est, klass, rkey=rkey, **{**base, **kw})
        cases.append(fit)
        if fit["clamped"]:  # the twin with the clamp disabled, which is what Spatex computes
            cases.append(case(db, regimes, graphs, f"{est}_{label}/unclamped", est, klass,
                              rkey=rkey, no_clamp=True, **{**base, **kw}))

    for est in ("het", "hom"):
        # estimator-specific options: Het has step1c, Hom has A1 (spreg's default is 'het')
        extra = {"het": dict(step1c=True), "hom": dict(A1="hom_sc")}[est]
        extra2 = {"het": dict(step1c=True, max_iter=3), "hom": dict(A1="hom", max_iter=3)}[est]
        # GM_Error_*_Regimes
        c(est, "error_nsa_pooled", "error", "nsa", plain)
        c(est, "error_nsa_extra", "error", "nsa", plain, **extra)
        c(est, "error_nsa_iter3", "error", "nsa", plain, max_iter=3, epsilon=1e-12)
        c(est, "error_nsa_extra_iter3", "error", "nsa", plain, **extra2)
        c(est, "error_nsa_loose_epsilon", "error", "nsa", plain, max_iter=20, epsilon=0.05)
        c(est, "error_nsa_one_constant", "error", "nsa", plain, constant_regi="one")
        c(est, "error_nsa_mixed", "error", "nsa", plain, cols2regi=[True, False])
        c(est, "error_nsa_global_x", "error", "nsa", plain, cols2regi=[False, False])
        c(est, "error_nsa_rook_binary", "error", "nsa", plain, wkey="rook_binary")
        c(est, "error_four_pooled", "error", "four", plain)
        c(est, "error_three_one_constant", "error", "three", plain, constant_regi="one",
          cols2regi=[True, False])
        c(est, "error_nsa_sep", "error", "nsa", plain, regime_err_sep=True)
        c(est, "error_nsa_sep_extra_iter3", "error", "nsa", plain, regime_err_sep=True, **extra2)
        c(est, "error_nsa_sep_rook_binary", "error", "nsa", plain, wkey="rook_binary",
          regime_err_sep=True)
        c(est, "error_three_sep", "error", "three", plain, regime_err_sep=True)
        c(est, "error_four_sep", "error", "four", plain, regime_err_sep=True)
        # GM_Endog_Error_*_Regimes
        c(est, "endog_nsa_pooled", "endog", "nsa", endog)
        c(est, "endog_nsa_extra", "endog", "nsa", endog, **extra)
        c(est, "endog_nsa_iter3", "endog", "nsa", endog, max_iter=3, epsilon=1e-12)
        c(est, "endog_nsa_extra_iter3", "endog", "nsa", endog, **extra2)
        c(est, "endog_nsa_one_constant", "endog", "nsa", endog, constant_regi="one")
        c(est, "endog_nsa_mixed", "endog", "nsa", endog, cols2regi=[True, False, True])
        c(est, "endog_nsa_global_yend", "endog", "nsa", endog, cols2regi=[True, True, False])
        c(est, "endog_nsa_rook_binary", "endog", "nsa", endog, wkey="rook_binary")
        c(est, "endog_four_pooled", "endog", "four", endog)
        c(est, "endog_three_one_constant", "endog", "three", endog, constant_regi="one")
        c(est, "endog_nsa_sep", "endog", "nsa", endog, regime_err_sep=True)
        c(est, "endog_nsa_sep_extra_iter3", "endog", "nsa", endog, regime_err_sep=True, **extra2)
        c(est, "endog_three_sep", "endog", "three", endog, regime_err_sep=True)
        c(est, "endog_nsa_sep_rook_binary", "endog", "nsa", endog, wkey="rook_binary",
          regime_err_sep=True)
        # GM_Combo_*_Regimes
        c(est, "combo_nsa_pooled", "combo", "nsa", plain)
        c(est, "combo_nsa_extra", "combo", "nsa", plain, **extra)
        c(est, "combo_nsa_iter3", "combo", "nsa", plain, max_iter=3, epsilon=1e-12)
        c(est, "combo_nsa_extra_iter3", "combo", "nsa", plain, **extra2)
        c(est, "combo_nsa_pooled_lags2", "combo", "nsa", plain, w_lags=2)
        c(est, "combo_nsa_pooled_no_lag_q", "combo", "nsa", endog, lag_q=False)
        c(est, "combo_nsa_endog_pooled", "combo", "nsa", endog)
        c(est, "combo_nsa_endog_one_constant", "combo", "nsa", endog, constant_regi="one")
        c(est, "combo_nsa_one_constant", "combo", "nsa", plain, constant_regi="one")
        c(est, "combo_nsa_mixed", "combo", "nsa", endog, cols2regi=[True, False, True])
        c(est, "combo_nsa_global_x", "combo", "nsa", plain, cols2regi=[False, True])
        c(est, "combo_nsa_rook_binary", "combo", "nsa", plain, wkey="rook_binary")
        c(est, "combo_four_pooled", "combo", "four", plain)
        c(est, "combo_three_endog_pooled", "combo", "three", endog)
        c(est, "combo_nsa_err_sep_only", "combo", "nsa", plain, regime_err_sep=True)
        c(est, "combo_nsa_lagsep", "combo", "nsa", plain, regime_lag_sep=True)
        c(est, "combo_nsa_lagsep_extra_iter3", "combo", "nsa", plain, regime_lag_sep=True,
          **extra2)
        c(est, "combo_nsa_lagsep_lags2", "combo", "nsa", plain, regime_lag_sep=True, w_lags=2)
        c(est, "combo_nsa_lagsep_endog", "combo", "nsa", endog, regime_lag_sep=True)
        c(est, "combo_nsa_lagsep_no_lag_q", "combo", "nsa", endog, regime_lag_sep=True,
          lag_q=False)
        c(est, "combo_nsa_lagsep_rook_binary", "combo", "nsa", plain, wkey="rook_binary",
          regime_lag_sep=True)
        c(est, "combo_nsa_lagsep_mixed", "combo", "nsa", plain, regime_lag_sep=True,
          cols2regi=[True, False])
        c(est, "combo_three_lagsep", "combo", "three", plain, regime_lag_sep=True)
        c(est, "combo_four_lagsep", "combo", "four", plain, regime_lag_sep=True)

    dump_json(FIXTURES / "regression" / "gm_error_het_hom_regimes.json", {
        "name": "gm_error_het_hom_regimes",
        "description": "spreg GM_{Error,Endog_Error,Combo}_{Het,Hom}_Regimes on columbus (CRIME ~ INC + OPEN [| HOVAL <- DISCBD + PLUMB] [+ W CRIME])",
        "meta": info,
        "y": flat(db["CRIME"]),
        "columns": {n: flat(db[n]) for n in ("INC", "OPEN", "EW", "HOVAL", "DISCBD", "PLUMB")},
        "regimes": regimes,
        "graphs": {"queen": sparse_export(queen_g.to_W()), "rook_binary": sparse_export(rook_g.to_W())},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
