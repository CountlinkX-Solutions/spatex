"""Generate spreg ML_LagRE oracle fixtures (`fixtures/regression/panel_re.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_panel_re.py

A seeded synthetic random-effects panel on the columbus queen graph (row-standardized,
n = 49, T = 5) with a spatial lag DGP. Data are stored wide (n x T per variable). Each case
records betas (slopes with the constant, rho, phi), vm, logll, aic, schwarz, sig2 and u.
ML_ErrorRE has its own generator (`generate_spreg_ml_error_re.py`).
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


def main():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    w = Queen.from_iterable(gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]]))
    w.transform = "r"
    n, t = w.n, 5
    rng = np.random.default_rng(62)
    x1, x2 = rng.normal(size=(n, t)), rng.normal(2.0, 1.0, size=(n, t))
    mu = rng.normal(0.0, 1.0, size=(n, 1))
    eps = rng.normal(0.0, 1.0, size=(n, t))
    a_inv = np.linalg.inv(np.eye(n) - 0.4 * w.full()[0])
    y = a_inv @ (3.0 + 1.5 * x1 - 0.8 * x2 + mu + eps)
    x = np.hstack((x1, x2))
    cases = []
    for te in (False, True):
        reg = spreg.ML_LagRE(y, x, w, time_effects=te, spat_impacts="simple", name_y="y",
                             name_x=["x1", "x2"])
        mult = spmultiplier(w, reg.rho, method="simple")
        cases.append({"params": {"time_effects": te}, "name_x": list(reg.name_x),
                      "betas": flat(reg.betas), "vm": flat(reg.vm),
                      "rho": float(reg.rho), "phi": float(reg.phi),
                      "logll": float(np.asarray(reg.logll).flatten()[0]), "aic": float(reg.aic),
                      "schwarz": float(reg.schwarz),
                      "sig2": float(np.asarray(reg.sig2).flatten()[0]), "u": flat(reg.u),
                      "multipliers": {k: float(v) for k, v in mult.items()
                                      if k in ("ati", "adi", "aii")}})
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    dump_json(FIXTURES / "regression" / "panel_re.json", {
        "name": "panel_re", "description": "spreg ML_LagRE, synthetic columbus panel",
        "meta": meta(), "n": n, "t": t, "y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist(),
        "graph": {"focal": coo.row[order].tolist(), "neighbor": coo.col[order].tolist(),
                  "weight": coo.data[order].tolist()},
        "cases": cases,
    })


if __name__ == "__main__":
    main()
