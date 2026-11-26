"""Generate spreg SUR oracle fixtures (`fixtures/regression/sur.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_sur.py

libpysal's bundled columbus.dbf, equations built with ``spreg.sur_dictxy`` (which adds the
constant). Cases: two equations with the same regressor count (so the cross-equation Chow tests
exist), the iterated estimator, and three equations of different size with spatial diagnostics
on the row-standardized queen weights.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from libpysal.weights import Queen  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def case(db, w, label, y_vars, x_vars, **opts):
    bigy, bigX, bigyvars, bigXvars = spreg.sur_dictxy(db, y_vars, x_vars)
    reg = spreg.SUR(bigy, bigX, w=w, name_bigy=bigyvars, name_bigX=bigXvars, **opts)
    n_eq = reg.n_eq
    out = {"label": label, "y_vars": y_vars, "x_vars": x_vars, "opts": opts,
           "name_bigy": [bigyvars[r] for r in range(n_eq)],
           "name_bigX": [list(bigXvars[r]) for r in range(n_eq)],
           "bOLS": [flat(reg.bOLS[r]) for r in range(n_eq)],
           "bSUR": [flat(reg.bSUR[r]) for r in range(n_eq)],
           "varb": flat(reg.varb), "sig": flat(reg.sig), "sig_ols": flat(reg.sig_ols),
           "corr": flat(reg.corr), "llik": float(reg.llik), "niter": int(reg.niter),
           "ldetS0": float(reg.ldetS0), "ldetS1": float(reg.ldetS1),
           "sur_inf": [np.asarray(reg.sur_inf[r]).tolist() for r in range(n_eq)],
           "lrtest": flat(reg.lrtest), "lmtest": flat(reg.lmtest),
           "surchow": [flat(t) for t in reg.surchow] if reg.surchow else None,
           "bigE": [flat(reg.bigE[:, r]) for r in range(n_eq)]}
    if opts.get("spat_diag"):
        out["lmEtest"] = flat(reg.lmEtest)
        out["lmlagtest"] = flat(reg.lmlagtest)
    return out


def main():
    path = libpysal.examples.get_path("columbus.dbf")
    db = libpysal.io.open(path, "r")
    w = Queen.from_shapefile(libpysal.examples.get_path("columbus.shp"))
    w.transform = "r"
    two_y, two_x = ["CRIME", "HOVAL"], [["INC", "DISCBD"], ["INC", "PLUMB"]]
    three_y = ["CRIME", "HOVAL", "INC"]
    three_x = [["INC", "HOVAL"], ["INC", "DISCBD", "PLUMB"], ["DISCBD"]]
    cases = [
        case(db, w, "two", two_y, two_x),
        case(db, w, "two_iter", two_y, two_x, iter=True),
        case(db, w, "three_spatial", three_y, three_x, spat_diag=True),
    ]
    columns = sorted({v for v in two_y + three_y + sum(two_x + three_x, [])})
    data = {c: [float(v) for v in db.by_col(c)] for c in columns}
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    dump_json(FIXTURES / "regression" / "sur.json", {
        "name": "sur", "description": "spreg SUR on columbus", "meta": meta(), "data": data,
        "graph": {"n": int(w.n), "focal": coo.row[order].tolist(),
                  "neighbor": coo.col[order].tolist(), "weight": coo.data[order].tolist()},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
