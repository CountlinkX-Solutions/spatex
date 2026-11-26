"""Generate spreg OLS_Regimes oracle fixtures (`fixtures/regression/ols_regimes.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_regimes.py

Columbus from libpysal's bundled ``columbus.dbf``: y = CRIME, x = [INC, HOVAL]; regimes NSA
(2 regimes) and NSA * 2 + EW (4 regimes). Each case stores the options, coefficient names,
betas, covariance, the joint and per-variable Chow tests and, for the separate-regressions
path (``regime_err_sep`` with every column by regime), every regime's OLS betas / vm / sig2 /
r2 / names.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def case(y, x, regimes, label, **opts):
    reg = spreg.OLS_Regimes(y, x, regimes, name_y="CRIME", name_x=["INC", "HOVAL"],
                            nonspat_diag=False, **opts)
    out = {"label": label, "opts": opts, "name_x": list(reg.name_x), "betas": flat(reg.betas),
           "vm": flat(reg.vm), "kr": int(reg.kr), "kf": int(reg.kf), "nr": int(reg.nr),
           "chow_joint": flat(reg.chow.joint), "chow_regi": np.asarray(reg.chow.regi).tolist(),
           "u": flat(reg.u), "predy": flat(reg.predy)}
    if hasattr(reg, "multi"):
        out["multi"] = {str(r): {"name_x": list(m.name_x), "betas": flat(m.betas),
                                 "vm": flat(m.vm), "sig2": float(m.sig2), "r2": float(m.r2)}
                        for r, m in reg.multi.items()}
    return out


def main():
    db = libpysal.io.open(libpysal.examples.get_path("columbus.dbf"), "r")
    y = np.array(db.by_col("CRIME"), dtype=float).reshape(-1, 1)
    x = np.column_stack([db.by_col("INC"), db.by_col("HOVAL")]).astype(float)
    nsa = [int(v) for v in db.by_col("NSA")]
    four = [int(a) * 2 + int(b) for a, b in zip(db.by_col("NSA"), db.by_col("EW"))]
    cases = [
        case(y, x, nsa, "nsa_separate"),
        case(y, x, nsa, "nsa_pooled", regime_err_sep=False),
        case(y, x, nsa, "nsa_one_constant", regime_err_sep=False, constant_regi="one"),
        case(y, x, nsa, "nsa_global_hoval_groupwise", cols2regi=[True, False]),
        case(y, x, nsa, "nsa_global_hoval_pooled", cols2regi=[True, False], regime_err_sep=False),
        case(y, x, nsa, "nsa_white", robust="white"),
        case(y, x, nsa, "nsa_pooled_white", robust="white", regime_err_sep=False),
        case(y, x, four, "four_separate"),
        case(y, x, four, "four_pooled", regime_err_sep=False),
    ]
    dump_json(FIXTURES / "regression" / "ols_regimes.json", {
        "name": "ols_regimes", "description": "spreg OLS_Regimes on columbus (CRIME ~ INC + HOVAL)",
        "meta": meta(), "y": flat(y), "x": [flat(x[:, 0]), flat(x[:, 1])],
        "regimes": {"nsa": nsa, "four": four}, "cases": cases,
    })


if __name__ == "__main__":
    main()
