"""Generate spreg ThreeSLS oracle fixtures (`fixtures/regression/three_sls.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_three_sls.py

libpysal's bundled columbus.dbf (equations built with ``spreg.sur_dictxy`` / ``sur_dictZ``) and
seeded synthetic systems. Cases: two equations with the same number of coefficients (so the
cross-equation Chow tests exist), three equations of different size, three equations of equal size,
and synthetic systems with several endogenous variables and more instruments than regressors.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import spreg  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def case(df, label, y_vars, x_vars, yend_vars, q_vars):
    reg = spreg.ThreeSLS(y_vars, x_vars, yend_vars, q_vars, df=df)
    n_eq = reg.n_eq
    return {
        "label": label, "y_vars": y_vars, "x_vars": x_vars, "yend_vars": yend_vars,
        "q_vars": q_vars,
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
    }


def synthetic(seed, n, n_eq, spec):
    """spec: per equation (n_exog, n_endog, n_extra_instruments)."""
    rng = np.random.default_rng(seed)
    cols, y_vars, x_vars, yend_vars, q_vars = {}, [], [], [], []
    shocks = rng.normal(size=(n, n_eq))
    shocks[:, 1:] += 0.6 * shocks[:, :1]  # correlated disturbances across equations
    for r, (kx, ke, kq) in enumerate(spec):
        xs = [f"x{r}_{i}" for i in range(kx)]
        es = [f"e{r}_{i}" for i in range(ke)]
        qs = [f"q{r}_{i}" for i in range(ke + kq)]
        for c in xs + qs:
            cols[c] = rng.normal(size=n)
        for i, c in enumerate(es):
            cols[c] = sum(0.8 * cols[q] for q in qs) / len(qs) + 0.5 * shocks[:, r] \
                + rng.normal(scale=0.7, size=n)
        y = sum((i + 1) * 0.5 * cols[c] for i, c in enumerate(xs)) \
            + sum(0.7 * cols[c] for c in es) + 1.0 + shocks[:, r]
        cols[f"y{r}"] = y
        y_vars.append(f"y{r}")
        x_vars.append(xs)
        yend_vars.append(es)
        q_vars.append(qs)
    return pd.DataFrame(cols), y_vars, x_vars, yend_vars, q_vars


def main():
    path = libpysal.examples.get_path("columbus.dbf")
    db = libpysal.io.open(path, "r")
    cb = pd.DataFrame({c: [float(v) for v in db.by_col(c)]
                       for c in ["CRIME", "HOVAL", "INC", "DISCBD", "PLUMB", "OPEN", "NSA", "NSB"]})
    cases = [
        case(cb, "columbus_two", ["CRIME", "HOVAL"], [["DISCBD"], ["PLUMB"]],
             [["INC"], ["INC"]], [["OPEN", "PLUMB"], ["OPEN", "NSA"]]),
        case(cb, "columbus_three_unequal", ["CRIME", "HOVAL", "INC"],
             [["DISCBD", "PLUMB"], ["OPEN"], ["NSA"]],
             [["HOVAL"], ["INC", "DISCBD"], ["CRIME"]],
             [["OPEN", "NSB"], ["NSA", "NSB", "PLUMB"], ["OPEN", "PLUMB", "NSB"]]),
        case(cb, "columbus_three_equal", ["CRIME", "HOVAL", "INC"],
             [["DISCBD"], ["PLUMB"], ["OPEN"]], [["INC"], ["CRIME"], ["HOVAL"]],
             [["NSA", "NSB"], ["NSA", "NSB"], ["NSA", "PLUMB"]]),
    ]
    synth = {}
    for label, seed, n, spec in [
        ("synthetic_two", 11, 80, [(2, 2, 1), (1, 1, 2)]),
        ("synthetic_three", 12, 120, [(1, 1, 0), (2, 2, 1), (3, 1, 2)]),
    ]:
        df, yv, xv, ev, qv = synthetic(seed, n, len(spec), spec)
        cases.append(case(df, label, yv, xv, ev, qv))
        synth[label] = {c: df[c].tolist() for c in df.columns}
    dump_json(FIXTURES / "regression" / "three_sls.json", {
        "name": "three_sls", "description": "spreg ThreeSLS on columbus and synthetic systems",
        "meta": meta(),
        "data": {"columbus": cb.to_dict(orient="list"), **synth},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
