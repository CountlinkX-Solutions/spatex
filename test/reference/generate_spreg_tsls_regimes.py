"""Generate spreg TSLS_Regimes oracle fixtures (`fixtures/regression/tsls_regimes.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_tsls_regimes.py

Columbus (libpysal's bundled ``columbus.shp``): y = CRIME, exogenous INC / OPEN (/ EW), endogenous
HOVAL instrumented by DISCBD and PLUMB. Regimes: NSA (2), NSA * 2 + EW (4) and terciles of the X
coordinate (3). Each case stores the options, the layout (`kr`, `kf`, `kryd`, `nr`), names,
betas, covariance, standard errors, z statistics (pooled fits), the residuals and two sets of
joint and per-variable Chow tests: `chow_*` with the restriction matrix on the real coefficient
layout (computed here with spreg's `wald_test`) and `spreg_chow_*` as spreg's `Chow` returns them
(they differ when exogenous columns are global and endogenous ones vary by regime, flagged by
`chow_layout_bug`; the script asserts they agree otherwise). Separate-regressions fits
(`regime_err_sep` with every column by regime and `constant_regi='many'`) also store every regime's `TSLS` attributes under `multi`.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from spreg.regimes import wald_test  # noqa: E402
from libpysal.weights import Kernel  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def attrs(reg, with_pooled):
    out = {"name_x": list(reg.name_x), "name_yend": list(reg.name_yend),
           "name_q": list(reg.name_q), "betas": flat(reg.betas), "vm": flat(reg.vm),
           "u": flat(reg.u), "predy": flat(reg.predy)}
    if with_pooled or hasattr(reg, "z_stat"):
        out["std_err"] = flat(reg.std_err)
        out["z_stat"] = [[float(z), float(p)] for z, p in reg.z_stat]
        out["sig2"] = float(np.asarray(reg.sig2).ravel()[0])
        out["utu"] = float(np.asarray(reg.utu).ravel()[0])
        out["pr2"] = float(reg.pr2)
        out["varb"] = flat(reg.varb)
    return out


def corrected_chow(reg, x_flags, y_flags):
    """Chow tests with the restriction matrix laid out on the real coefficient order (x regime
    blocks, x globals, yend regime blocks, yend globals), tested with spreg's own `wald_test`.
    spreg's `buildR1var` appends the global columns last, which is wrong when exogenous globals
    precede regime-specific endogenous columns."""
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


def case(db, regimes, label, x, yend, q, rkey, gwk=None, **opts):
    cols = lambda names: np.column_stack([db[n].to_numpy(dtype=float) for n in names])  # noqa: E731
    kw = dict(opts)
    if gwk is not None:
        kw["gwk"] = gwk
    reg = spreg.TSLS_Regimes(
        db["CRIME"].to_numpy(dtype=float).reshape(-1, 1), cols(x), cols(yend), cols(q),
        regimes[rkey], name_y="CRIME", name_x=list(x), name_yend=list(yend), name_q=list(q), **kw)
    pooled = not hasattr(reg, "multi")
    if pooled:
        kx = len(x)
        c2r = kw.get("cols2regi", "all")
        c2r = [True] * (kx + len(yend)) if c2r == "all" else list(c2r)
        x_flags = [opts.get("constant_regi", "many") == "many"] + c2r[:kx]
        joint, regi = corrected_chow(reg, x_flags, c2r[kx:])
        affected = (len(x_flags) - sum(x_flags)) > 0 and sum(c2r[kx:]) > 0
    else:
        joint, regi = flat(reg.chow.joint), np.asarray(reg.chow.regi).tolist()
        affected = False
    if not affected:
        assert np.allclose(joint, flat(reg.chow.joint), rtol=1e-8, atol=1e-12), label
        assert np.allclose(regi, reg.chow.regi, rtol=1e-8, atol=1e-12), label
    out = {"label": label, "chow_layout_bug": bool(affected),
           "spreg_chow_joint": flat(reg.chow.joint),
           "spreg_chow_regi": np.asarray(reg.chow.regi).tolist(), "regimes": rkey, "x": list(x), "yend": list(yend), "q": list(q),
           "opts": opts, "gwk": gwk is not None, "kr": int(reg.kr), "kf": int(reg.kf),
           "kryd": int(reg.kryd), "nr": int(reg.nr),
           "chow_joint": joint, "chow_regi": regi,
           "warning": str(getattr(reg, "warning", None) or ""), "multi": None}
    out.update(attrs(reg, with_pooled=not hasattr(reg, "multi")))
    if hasattr(reg, "multi"):
        out["multi"] = {str(r): attrs(m, True) for r, m in reg.multi.items()}
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

    cent = np.column_stack([db.geometry.centroid.x, db.geometry.centroid.y])
    gwk = Kernel(cent, k=15, function="triangular", fixed=False, diagonal=True)
    gwk.transform = "o"

    base = dict(x=["INC", "OPEN"], yend=["HOVAL"], q=["DISCBD", "PLUMB"])
    c = lambda label, rkey, **kw: case(db, regimes, label, rkey=rkey, **{**base, **kw})  # noqa: E731
    cases = [
        c("nsa_separate", "nsa", regime_err_sep=True),
        c("nsa_separate_sig2n", "nsa", regime_err_sep=True, sig2n_k=False),
        c("nsa_separate_white", "nsa", regime_err_sep=True, robust="white"),
        c("nsa_pooled", "nsa"),
        c("nsa_pooled_sig2n", "nsa", sig2n_k=False),
        c("nsa_pooled_white", "nsa", robust="white"),
        c("nsa_pooled_hac", "nsa", robust="hac", gwk=gwk),
        c("nsa_separate_hac_disabled", "nsa", regime_err_sep=True, robust="hac", gwk=gwk),
        c("nsa_one_constant", "nsa", constant_regi="one"),
        c("nsa_one_constant_err_sep", "nsa", constant_regi="one", regime_err_sep=True),
        c("nsa_mixed_global_open", "nsa", cols2regi=[True, False, True]),
        c("nsa_mixed_err_sep", "nsa", cols2regi=[True, False, True], regime_err_sep=True),
        c("nsa_global_inc_yend", "nsa", cols2regi=[False, True, False]),
        c("nsa_global_yend", "nsa", cols2regi=[True, True, False], robust="white"),
        c("four_separate", "four", regime_err_sep=True),
        c("four_pooled", "four"),
        c("four_dropped_ew", "four", x=["INC", "EW"], regime_err_sep=True),
        c("three_separate", "three", regime_err_sep=True),
        c("three_pooled_white", "three", robust="white"),
        c("three_one_constant", "three", constant_regi="one", cols2regi=[True, False, True]),
    ]
    dump_json(FIXTURES / "regression" / "tsls_regimes.json", {
        "name": "tsls_regimes",
        "description": "spreg TSLS_Regimes on columbus (CRIME ~ INC + OPEN | HOVAL <- DISCBD + PLUMB)",
        "meta": info,
        "y": flat(db["CRIME"]),
        "columns": {n: flat(db[n]) for n in ("INC", "OPEN", "EW", "HOVAL", "DISCBD", "PLUMB")},
        "regimes": regimes, "gwk": sparse_export(gwk), "cases": cases,
    })


if __name__ == "__main__":
    main()
