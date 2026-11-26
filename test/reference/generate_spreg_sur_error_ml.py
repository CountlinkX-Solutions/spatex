"""Generate spreg SURerrorML oracle fixtures (`fixtures/regression/sur_error_ml.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_sur_error_ml.py

libpysal's bundled columbus (equations built with ``spreg.sur_dictxy``, which adds the constant) with
row-standardized queen and rook weights: two equations, and three equations of different and of the
same size. Every case is fitted with ``spat_diag=True`` and ``vm=True`` so that all the tests spreg
reports are stored (they do not change the estimates). spreg alternates the SUR estimation with an
L-BFGS-B search of the lambdas (see docs/parity.md, "spreg spatial SUR notes"); the fixture records
the starting lambdas followed by the lambdas each search returned (``lam_path``: the reported ``bSUR``, ``varb``, ``sig`` and ``bigE``
follow the one before the last, while ``lamsur`` is the last), the number of searches,
and the same model refitted with ``epsilon = 1e-10`` (key ``tight``), whose estimates are the best
available reference for the converged ML solution. Binary weights are not included: spreg stops
with ``RuntimeError: Factor is exactly singular`` for them.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
import spreg.sur_error as sur_error  # noqa: E402
from libpysal.weights import Queen, Rook  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402

warnings.simplefilter("ignore")

CALLS = []
_minimize = sur_error.minimize


def _record(fun, x0, **kwargs):
    res = _minimize(fun, x0, **kwargs)
    CALLS.append({"x0": np.asarray(x0).flatten().tolist(), "x": np.asarray(res.x).tolist(),
                  "fun": float(res.fun), "nit": int(res.nit)})
    return res


sur_error.minimize = _record


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def pair(t):
    return None if t is None else [float(t[0]), float(t[1]), float(t[2])]


def fit(db, w, y_vars, x_vars, epsilon):
    del CALLS[:]
    bigy, bigX, bigyvars, bigXvars = spreg.sur_dictxy(db, y_vars, x_vars)
    reg = spreg.SURerrorML(bigy, bigX, w=w, name_bigy=bigyvars, name_bigX=bigXvars,
                           epsilon=epsilon, spat_diag=True, vm=True)
    n_eq = reg.n_eq
    return reg, {
        "lamsur": flat(reg.lamsur), "lam_path": [CALLS[0]["x0"]] + [c["x"] for c in CALLS], "searches": len(CALLS),
        "bSUR": [flat(reg.bSUR[r]) for r in range(n_eq)],
        "varb": flat(reg.varb), "sig": flat(reg.sig), "corr": flat(reg.corr),
        "bigE": [flat(reg.bigE[:, r]) for r in range(n_eq)],
        "cliksurerr": float(reg.cliksurerr), "surerrllik": float(reg.surerrllik),
    }


def case(db, graphs, wkey, label, y_vars, x_vars):
    reg, out = fit(db, graphs[wkey], y_vars, x_vars, 1e-7)
    n_eq = reg.n_eq
    _, tight = fit(db, graphs[wkey], y_vars, x_vars, 1e-10)
    out.update({
        "label": label, "weights": wkey, "y_vars": y_vars, "x_vars": x_vars,
        "name_bigy": [reg.name_bigy[r] for r in range(n_eq)],
        "name_bigX": [list(reg.name_bigX[r]) for r in range(n_eq)],
        "bigK": flat(reg.bigK),
        "lamols": flat(reg.lamols), "clikerr": float(reg.clikerr), "errllik": float(reg.errllik),
        "llik": float(reg.llik), "bSUR0": [flat(reg.bSUR0[r]) for r in range(n_eq)],
        "sur_inf": [np.asarray(reg.sur_inf[r]).tolist() for r in range(n_eq)],
        "surchow": [flat(t) for t in reg.surchow] if reg.surchow else None,
        "lrtest": pair(reg.lrtest), "likrlambda": pair(reg.likrlambda),
        "vm": flat(reg.vm), "vm_size": int(reg.vm.shape[0]),
        "lamsetp": [flat(a) for a in reg.lamsetp],
        "lamtest": pair(reg.lamtest), "joinlam": pair(reg.joinlam),
        "tight": tight,
    })
    return out


def main():
    path = libpysal.examples.get_path("columbus.dbf")
    db = libpysal.io.open(path, "r")
    shp = libpysal.examples.get_path("columbus.shp")
    queen, rook = Queen.from_shapefile(shp), Rook.from_shapefile(shp)
    queen.transform = "r"
    rook.transform = "r"
    graphs = {"queen": queen, "rook": rook}
    two_y, two_x = ["CRIME", "HOVAL"], [["INC", "DISCBD"], ["INC", "PLUMB"]]
    three_y = ["CRIME", "HOVAL", "INC"]
    unequal_x = [["DISCBD", "OPEN"], ["PLUMB"], ["DISCBD", "PLUMB", "OPEN"]]
    equal_x = [["DISCBD"], ["PLUMB"], ["OPEN"]]
    cases = [
        case(db, graphs, "queen", "two_queen", two_y, two_x),
        case(db, graphs, "rook", "two_rook", two_y, two_x),
        case(db, graphs, "queen", "three_unequal_queen", three_y, unequal_x),
        case(db, graphs, "queen", "three_equal_queen", three_y, equal_x),
        case(db, graphs, "rook", "three_equal_rook", three_y, equal_x),
    ]
    columns = sorted({v for v in two_y + three_y + sum(two_x + unequal_x + equal_x, [])})
    data = {c: [float(v) for v in db.by_col(c)] for c in columns}
    dump_json(FIXTURES / "regression" / "sur_error_ml.json", {
        "name": "sur_error_ml", "description": "spreg SURerrorML on columbus",
        "meta": meta(), "data": data,
        "graphs": {k: sparse_export(w) for k, w in graphs.items()},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
