"""Generate spreg SURerrorGM oracle fixtures (`fixtures/regression/sur_error_gm.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_sur_error_gm.py

libpysal's bundled columbus (equations built with ``spreg.sur_dictxy``, which adds the constant) with
row-standardized queen and rook weights and binary rook weights (flat moment objective). Cases: two
equations with the same regressor count (so the cross-equation Chow tests exist), three equations
of different and of equal size. Every optimizer call is recorded next to the exact minimizer of its
own objective (see docs/parity.md, "spreg GM notes").
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from libpysal.weights import Queen, Rook  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402
from generate_spreg_gm_error_regimes import OPTIM  # noqa: E402  (also wraps optim_moments)

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def case(db, graphs, wkey, label, y_vars, x_vars):
    del OPTIM[:]
    bigy, bigX, bigyvars, bigXvars = spreg.sur_dictxy(db, y_vars, x_vars)
    reg = spreg.SURerrorGM(bigy, bigX, w=graphs[wkey], name_bigy=bigyvars, name_bigX=bigXvars)
    n_eq = reg.n_eq
    return {
        "label": label, "weights": wkey, "y_vars": y_vars, "x_vars": x_vars,
        "name_bigy": [bigyvars[r] for r in range(n_eq)],
        "name_bigX": [list(bigXvars[r]) for r in range(n_eq)],
        "bigK": flat(reg.bigK),
        "lamsur": flat(reg.lamsur), "optim": [dict(o) for o in OPTIM],
        "bSUR": [flat(reg.bSUR[r]) for r in range(n_eq)],
        "varb": flat(reg.varb), "sig": flat(reg.sig), "corr": flat(reg.corr),
        "sur_inf": [np.asarray(reg.sur_inf[r]).tolist() for r in range(n_eq)],
        "surchow": [flat(t) for t in reg.surchow] if reg.surchow else None,
        "bigE": [flat(reg.bigE[:, r]) for r in range(n_eq)],
    }


def main():
    path = libpysal.examples.get_path("columbus.dbf")
    db = libpysal.io.open(path, "r")
    shp = libpysal.examples.get_path("columbus.shp")
    queen, rook, binary = Queen.from_shapefile(shp), Rook.from_shapefile(shp), Rook.from_shapefile(shp)
    queen.transform = "r"
    rook.transform = "r"
    graphs = {"queen": queen, "rook": rook, "rook_binary": binary}
    two_y, two_x = ["CRIME", "HOVAL"], [["INC", "DISCBD"], ["INC", "PLUMB"]]
    three_y = ["CRIME", "HOVAL", "INC"]
    three_x = [["INC", "HOVAL"], ["INC", "DISCBD", "PLUMB"], ["DISCBD"]]
    equal_x = [["DISCBD"], ["PLUMB"], ["OPEN"]]
    cases = [
        case(db, graphs, "queen", "two_queen", two_y, two_x),
        case(db, graphs, "rook", "two_rook", two_y, two_x),
        case(db, graphs, "queen", "three_unequal_queen", three_y, three_x),
        case(db, graphs, "rook", "three_equal_rook", three_y, equal_x),
        case(db, graphs, "rook_binary", "two_rook_binary", two_y, two_x),
        case(db, graphs, "rook_binary", "three_equal_rook_binary", three_y, equal_x),
    ]
    columns = sorted({v for v in two_y + three_y + sum(two_x + three_x + equal_x, [])})
    data = {c: [float(v) for v in db.by_col(c)] for c in columns}
    dump_json(FIXTURES / "regression" / "sur_error_gm.json", {
        "name": "sur_error_gm", "description": "spreg SURerrorGM on columbus",
        "meta": meta(), "data": data,
        "graphs": {k: sparse_export(w) for k, w in graphs.items()},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
