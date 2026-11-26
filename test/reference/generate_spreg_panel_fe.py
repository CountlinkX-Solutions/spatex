"""Generate spreg ML_LagFE / ML_ErrorFE oracle fixtures (`fixtures/regression/panel_fe.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_panel_fe.py

A seeded synthetic panel on the columbus queen graph (row-standardized, n = 49, T = 5): two
regressors, individual effects, a spatial lag DGP for y_lag and a spatial error DGP for y_err.
Data are stored wide (n x T per variable). Each case records betas, vm, rho / lam, logll, aic,
schwarz, sig2, u, the fixed effects mu_i and, for the lag model, spreg's simple impacts.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from libpysal.weights import Queen  # noqa: E402
from shapely import wkb  # noqa: E402
from spreg.sputils import spmultiplier  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def record(reg, model, params):
    out = {"model": model, "params": params, "name_x": list(reg.name_x), "betas": flat(reg.betas),
           "vm": flat(reg.vm), "logll": float(np.asarray(reg.logll).flatten()[0]),
           "aic": float(reg.aic), "schwarz": float(reg.schwarz),
           "sig2": float(np.asarray(reg.sig2).flatten()[0]), "u": flat(reg.u),
           "mu_i": flat(reg.mu_i)}
    for attr in ("rho", "lam"):
        if hasattr(reg, attr):
            out[attr] = float(np.asarray(getattr(reg, attr)).flatten()[0])
    return out


def main():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    w = Queen.from_iterable(gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]]))
    w.transform = "r"
    n, t = w.n, 5
    rng = np.random.default_rng(61)
    x1, x2 = rng.normal(size=(n, t)), rng.normal(2.0, 1.0, size=(n, t))
    mu = rng.normal(0.0, 2.0, size=(n, 1))
    eps = rng.normal(0.0, 1.0, size=(n, t))
    a_inv = np.linalg.inv(np.eye(n) - 0.4 * w.full()[0])
    y_lag = a_inv @ (1.5 * x1 - 0.8 * x2 + mu + eps)
    y_err = 1.5 * x1 - 0.8 * x2 + mu + np.linalg.inv(np.eye(n) - 0.5 * w.full()[0]) @ eps
    x = np.hstack((x1, x2))
    names = {"name_y": "y", "name_x": ["x1", "x2"]}
    cases = []
    for te in (False, True):
        reg = spreg.ML_LagFE(y_lag, x, w, time_effects=te, spat_impacts="simple", **names)
        c = record(reg, "lag", {"time_effects": te})
        mult = spmultiplier(w, reg.rho, method="simple")
        c["multipliers"] = {k: float(v) for k, v in mult.items() if k in ("ati", "adi", "aii")}
        cases.append(c)
        reg = spreg.ML_ErrorFE(y_err, x, w, time_effects=te, **names)
        cases.append(record(reg, "error", {"time_effects": te}))
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    dump_json(FIXTURES / "regression" / "panel_fe.json", {
        "name": "panel_fe", "description": "spreg ML_LagFE / ML_ErrorFE, synthetic columbus panel",
        "meta": meta(), "n": n, "t": t, "y_lag": y_lag.tolist(), "y_err": y_err.tolist(),
        "x1": x1.tolist(), "x2": x2.tolist(),
        "graph": {"focal": coo.row[order].tolist(), "neighbor": coo.col[order].tolist(),
                  "weight": coo.data[order].tolist()},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
