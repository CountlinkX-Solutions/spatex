"""Generate spreg SURlagIV oracle fixtures (`fixtures/regression/sur_lag_iv.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_sur_lag_iv.py

libpysal's bundled columbus (frames passed with ``df=``) with row-standardized queen and rook
weights. Cases: two and three equations, ``w_lags`` 1 and 2, ``lag_q`` true and false, and systems
with user endogenous variables and instruments. ``spat_diag=True`` adds the joint test on the
spatial coefficients (``joinrho``); spreg reports no spatial impacts for this class.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import spreg  # noqa: E402
from libpysal.weights import Queen, Rook  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_spreg_gm import sparse_export  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def case(df, graphs, wkey, label, y_vars, x_vars, yend_vars=None, q_vars=None, w_lags=1, lag_q=True):
    reg = spreg.SURlagIV(y_vars, x_vars, yend_vars, q_vars, w=graphs[wkey], df=df,
                         w_lags=w_lags, lag_q=lag_q, spat_diag=True)
    n_eq = reg.n_eq
    return {
        "label": label, "weights": wkey, "y_vars": y_vars, "x_vars": x_vars,
        "yend_vars": yend_vars, "q_vars": q_vars, "w_lags": w_lags, "lag_q": lag_q,
        "name_bigy": [reg.name_bigy[r] for r in range(n_eq)],
        "name_bigX": [list(reg.name_bigX[r]) for r in range(n_eq)],
        "name_bigyend": [list(reg.name_bigyend[r]) for r in range(n_eq)],
        "name_bigq": [list(reg.name_bigq[r]) for r in range(n_eq)],
        "bigK": flat(reg.bigK),
        "b2SLS": [flat(reg.b2SLS[r]) for r in range(n_eq)],
        "tslsE": [flat(reg.tslsE[:, r]) for r in range(n_eq)],
        "b3SLS": [flat(reg.b3SLS[r]) for r in range(n_eq)],
        "varb": flat(reg.varb), "sig": flat(reg.sig), "corr": flat(reg.corr),
        "bigE": [flat(reg.bigE[:, r]) for r in range(n_eq)],
        "tsls_inf": [np.asarray(reg.tsls_inf[r]).tolist() for r in range(n_eq)],
        "surchow": [flat(t) for t in reg.surchow] if reg.surchow else None,
        "joinrho": flat(reg.joinrho),
    }


def main():
    path = libpysal.examples.get_path("columbus.dbf")
    db = libpysal.io.open(path, "r")
    shp = libpysal.examples.get_path("columbus.shp")
    queen, rook = Queen.from_shapefile(shp), Rook.from_shapefile(shp)
    queen.transform = "r"
    rook.transform = "r"
    graphs = {"queen": queen, "rook": rook}
    cols = ["CRIME", "HOVAL", "INC", "DISCBD", "PLUMB", "OPEN", "NSA", "NSB"]
    cb = pd.DataFrame({c: [float(v) for v in db.by_col(c)] for c in cols})
    two_y, two_x = ["CRIME", "HOVAL"], [["INC", "DISCBD"], ["INC", "PLUMB"]]
    three_y = ["CRIME", "HOVAL", "INC"]
    three_x = [["DISCBD", "PLUMB"], ["OPEN"], ["NSA"]]
    equal_x = [["DISCBD"], ["PLUMB"], ["OPEN"]]
    yend, qv = [["INC"], ["INC"]], [["DISCBD", "OPEN"], ["DISCBD", "NSB"]]
    ux = [["PLUMB"], ["NSA"]]
    cases = [
        case(cb, graphs, "queen", "two_lag1", two_y, two_x),
        case(cb, graphs, "queen", "two_lag2", two_y, two_x, w_lags=2),
        case(cb, graphs, "rook", "three_unequal_lag1", three_y, three_x),
        case(cb, graphs, "rook", "three_equal_lag2", three_y, equal_x, w_lags=2),
        case(cb, graphs, "queen", "user_lag1_lagq", two_y, ux, yend, qv),
        case(cb, graphs, "queen", "user_lag1_nolagq", two_y, ux, yend, qv, lag_q=False),
        case(cb, graphs, "queen", "user_lag2_lagq", two_y, ux, yend, qv, w_lags=2),
        case(cb, graphs, "rook", "user_lag2_nolagq", two_y, ux, yend, qv, w_lags=2, lag_q=False),
    ]
    data = {c: cb[c].tolist() for c in cols}
    dump_json(FIXTURES / "regression" / "sur_lag_iv.json", {
        "name": "sur_lag_iv", "description": "spreg SURlagIV on columbus",
        "meta": meta(), "data": data,
        "graphs": {k: sparse_export(w) for k, w in graphs.items()},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
