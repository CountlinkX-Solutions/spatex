"""Generate spreg ``slx_lags`` oracle fixtures (`fixtures/regression/slx.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_slx.py

Columbus: y = CRIME, x = [INC, HOVAL], queen contiguity row-standardized (spreg uses W as
given). Every case records the model, its ``slx_lags`` / ``slx_vars``, the coefficient names,
betas, covariance and the model-specific scalars; ML_Lag also records spreg's spatial
multipliers and its Durbin impacts, computed with ``_sp_effects``' rule (total =
ati (beta + theta), direct = adi beta, indirect = total - direct) because spreg overwrites the
``output`` table it would read them from.
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


def base(reg, model, params):
    out = {"model": model, "params": params, "name_x": list(reg.name_x),
           "betas": flat(reg.betas), "vm": flat(reg.vm), "std_err": flat(reg.std_err)}
    for attr in ("rho", "lam", "logll", "sig2", "utu", "pr2", "aic", "schwarz"):
        if hasattr(reg, attr) and getattr(reg, attr) is not None:
            v = getattr(reg, attr)
            out[attr] = float(np.asarray(v).flatten()[0])
    return out


def main():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    a = d["attributes"]
    y = np.asarray(a["CRIME"], dtype=float).reshape(-1, 1)
    x = np.column_stack([a["INC"], a["HOVAL"]]).astype(float)
    w = Queen.from_iterable(geoms)
    w.transform = "r"
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    graph = {"n": int(w.n), "focal": coo.row[order].tolist(), "neighbor": coo.col[order].tolist(),
             "weight": coo.data[order].tolist()}
    names = {"name_y": "CRIME", "name_x": ["INC", "HOVAL"]}
    cases = []
    for params in ({"slx_lags": 1}, {"slx_lags": 2}, {"slx_lags": 1, "slx_vars": [True, False]}):
        reg = spreg.OLS(y, x, w=w, **params, **names)
        cases.append(base(reg, "ols", params))
    for params in ({"slx_lags": 1}, {"slx_lags": 1, "slx_vars": [False, True]}):
        reg = spreg.ML_Lag(y, x, w, method="full", spat_impacts="simple", spat_diag=False,
                           **params, **names)
        c = base(reg, "ml_lag", params)
        mult = spmultiplier(w, reg.rho, method="simple")
        c["multipliers"] = {k: float(v) for k, v in mult.items() if k in ("ati", "adi", "aii")}
        # _sp_effects for one SLX lag: total = ati (beta + theta), direct = adi beta.
        betas = np.asarray(reg.betas).flatten()
        lagged = params.get("slx_vars", [True, True])
        beta = betas[1:3]
        theta = np.zeros(2)
        theta[np.array(lagged)] = betas[3:3 + sum(lagged)]
        total = mult["ati"] * (beta + theta)
        direct = mult["adi"] * beta
        c["impacts"] = {"names": names["name_x"], "total": flat(total),
                        "direct": flat(direct), "indirect": flat(total - direct)}
        cases.append(c)
    reg = spreg.ML_Error(y, x, w, method="full", slx_lags=1, **names)
    cases.append(base(reg, "ml_error", {"slx_lags": 1}))
    for est, cls in (("kp98", spreg.GM_Error), ("het", spreg.GM_Error_Het),
                     ("hom", spreg.GM_Error_Hom)):
        reg = cls(y, x, w, slx_lags=1, **names)
        cases.append(base(reg, f"gm_error_{est}", {"slx_lags": 1}))
    for params in ({"slx_lags": 1}, {"slx_lags": 1, "w_lags": 2}, {"slx_lags": 2}):
        reg = spreg.GM_Lag(y, x, w=w, spat_diag=False, **params, **names)
        cases.append(base(reg, "gm_lag", params))
    dump_json(FIXTURES / "regression" / "slx.json", {
        "name": "slx", "description": "spreg slx_lags on columbus (CRIME ~ INC + HOVAL, queen r)",
        "meta": meta(), "y": flat(y), "x": [flat(x[:, 0]), flat(x[:, 1])], "graph": graph,
        "cases": cases,
    })


if __name__ == "__main__":
    main()
