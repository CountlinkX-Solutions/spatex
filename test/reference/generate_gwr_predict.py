"""Generate GWR prediction and local-collinearity oracle fixtures
(`fixtures/regression/gwr_predict.json`, `gwr_collinearity.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_gwr_predict.py

Ground truth is mgwr 2.2.1 (`GWR.predict`, `GWRResults.local_collinearity`,
`MGWRResults.local_collinearity`).

* `gwr_predict.json` - `GWR(...).predict(points, P)` on libpysal's Georgia example (the canonical
  mgwr test setup, where the ten predicted counties are also in the calibration sample, and a
  held-out split that trains on the first 139 counties and predicts the last 20), on a seeded
  spherical sample (including a model without a constant) and on seeded Poisson (with an exposure
  offset) and Binomial samples. Every case records `params`, `bse`, `tvalues` and `predictions`
  and, for the Gaussian family, `local_r2`; two cases pass `exog_scale`/`exog_resid` explicitly.
* `gwr_collinearity.json` - `GWRResults.local_collinearity()` (Georgia adaptive and fixed, the
  global bandwidth, the spherical no-constant sample) and `MGWRResults.local_collinearity()`
  (seeded lattice and Georgia, standardized as mgwr requires).
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal as ps  # noqa: E402
import mgwr  # noqa: E402
import numpy as np  # noqa: E402
from mgwr.gwr import GWR, MGWR  # noqa: E402
from mgwr.sel_bw import Sel_BW  # noqa: E402
from spglm.family import Binomial, Poisson  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

FAMILIES = {"gaussian": None, "poisson": Poisson, "binomial": Binomial}


def flat(a):
    return np.asarray(a, dtype=float).reshape(-1)


def georgia():
    d = ps.io.open(ps.examples.get_path("GData_utm.csv"))
    coords = np.array(list(zip(d.by_col("X"), d.by_col("Y"))))
    y = np.array(d.by_col("PctBach"))
    cols = ["PctRural", "PctPov", "PctBlack"]
    x = np.column_stack([np.array(d.by_col(c)) for c in cols])
    return coords, y, x


def lattice(rng, rows, cols, jitter=0.4):
    gx, gy = np.meshgrid(np.arange(cols, dtype=float), np.arange(rows, dtype=float))
    c = np.c_[gx.ravel(), gy.ravel()]
    return c + rng.uniform(0.0, jitter, size=c.shape)


def synthetic():
    rng = np.random.default_rng(20261025)
    n = 120
    lon = rng.uniform(-100.0, -90.0, n)
    lat = rng.uniform(30.0, 40.0, n)
    coords = np.column_stack([lon, lat])
    u, v = (lon + 100.0) / 10.0, (lat - 30.0) / 10.0
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n) + 0.3 * x1
    y = (1.0 + 2.0 * u) + (1.5 - 2.0 * v) * x1 + (0.5 + np.sin(3.0 * u)) * x2
    y = y + rng.normal(scale=0.3, size=n)
    return coords, y, np.column_stack([x1, x2])


def glm_data():
    rng = np.random.default_rng(20261026)
    n = 80
    coords = lattice(rng, 8, 10)
    x = rng.normal(size=(n, 2))
    u, v = coords[:, 0] / 9.0, coords[:, 1] / 7.0
    off = rng.uniform(0.5, 2.0, size=n)
    eta = 1.2 + 0.4 * u + (0.5 + 0.4 * v) * x[:, 0] - (0.3 + 0.5 * u) * x[:, 1]
    counts = np.maximum(rng.poisson(off * np.exp(eta)), 1).astype(float)
    p = 1.0 / (1.0 + np.exp(-(0.2 + 1.2 * (u - 0.5) * 2 * x[:, 0] + 0.8 * x[:, 1] * (v + 0.3))))
    yb = (rng.uniform(size=n) < p).astype(float)
    return coords, x, off, counts, yb


def predict_case(data, train, test, bw, family="gaussian", offset=None, exog=False, **kw):
    """One `GWR.predict` run. `train`/`test` are index lists into the dataset."""
    coords, y, x = data
    sigma2_v1 = kw.pop("sigma2_v1", True)
    fam = FAMILIES[family]
    extra = {} if fam is None else {"family": fam()}
    off = None if offset is None else offset[train].reshape(-1, 1)
    model = GWR(coords[train], y[train].reshape(-1, 1), x[train], bw,
                sigma2_v1=sigma2_v1, offset=off, n_jobs=1, **extra, **kw)
    scale = resid = None
    if exog:
        # Deliberately not the model's own values: the arguments must be honoured.
        scale = 0.75
        resid = np.linspace(-1.0, 1.0, len(train)).reshape(-1, 1)
        r = model.predict(coords[test], x[test], exog_scale=scale, exog_resid=resid)
    else:
        r = model.predict(coords[test], x[test])
    out = {
        "options": {**kw, "bw": float(bw), "family": family, "sigma2_v1": sigma2_v1},
        "train": list(map(int, train)),
        "test": list(map(int, test)),
        "params": r.params, "bse": r.bse, "tvalues": r.tvalues,
        "predictions": flat(r.predictions),
    }
    if family == "gaussian":
        out["local_r2"] = flat(r.localR2)
    if exog:
        out["exog_scale"] = scale
        out["exog_resid"] = flat(resid)
    return out


def predict_fixture():
    gc, gy, gx = georgia()
    sc, sy, sx = synthetic()
    pc, px, off, counts, yb = glm_data()
    n = len(gy)
    all_idx = np.arange(n)
    out = {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "datasets": {
            "georgia": {"coords": gc, "y": gy, "x": gx.T},
            "synthetic": {"coords": sc, "y": sy, "x": sx.T},
            "poisson": {"coords": pc, "y": counts, "x": px.T, "offset": off},
            "binomial": {"coords": pc, "y": yb, "x": px.T},
        },
        "cases": {},
    }
    c = out["cases"]
    geo = (gc, gy, gx)
    # mgwr's own test: the whole sample calibrates, the last ten counties are predicted.
    c["georgia_mgwr_test"] = {
        "dataset": "georgia",
        **predict_case(geo, all_idx, all_idx[-10:], 93, sigma2_v1=False),
    }
    held, test = all_idx[:139], all_idx[139:]
    c["georgia_holdout_adaptive"] = {
        "dataset": "georgia", **predict_case(geo, held, test, 70)}
    c["georgia_holdout_gaussian_fixed"] = {
        "dataset": "georgia",
        **predict_case(geo, held, test, 90000.0, kernel="gaussian", fixed=True)}
    c["georgia_holdout_exponential_sigma2_v2"] = {
        "dataset": "georgia",
        **predict_case(geo, held, test, 60, kernel="exponential", sigma2_v1=False)}
    c["georgia_holdout_global"] = {
        "dataset": "georgia", **predict_case(geo, held, test, np.inf)}
    c["georgia_holdout_exog"] = {
        "dataset": "georgia", **predict_case(geo, held, test, 70, exog=True)}
    syn = (sc, sy, sx)
    sidx = np.arange(len(sy))
    c["synthetic_spherical"] = {
        "dataset": "synthetic",
        **predict_case(syn, sidx[:100], sidx[100:], 400.0, kernel="gaussian",
                       fixed=True, spherical=True)}
    c["synthetic_no_constant"] = {
        "dataset": "synthetic",
        **predict_case(syn, sidx[:100], sidx[100:], 45, constant=False, spherical=True)}
    pidx = np.arange(len(counts))
    c["poisson_offset"] = {
        "dataset": "poisson",
        **predict_case((pc, counts, px), pidx[:65], pidx[65:], 40, family="poisson",
                       offset=off)}
    c["binomial"] = {
        "dataset": "binomial",
        **predict_case((pc, yb, px), pidx[:65], pidx[65:], 40, family="binomial")}
    c["binomial_fixed_gaussian_exog"] = {
        "dataset": "binomial",
        **predict_case((pc, yb, px), pidx[:65], pidx[65:], 3.0, family="binomial",
                       kernel="gaussian", fixed=True, exog=True)}
    return out


def collinearity_gwr(coords, y, x, bw, **kw):
    r = GWR(coords, y.reshape(-1, 1), x, bw, n_jobs=1, **kw).fit()
    corr, vif, cn, vdp = r.local_collinearity()
    return {
        "options": {**kw, "bw": float(bw)},
        "corr": corr, "vif": vif, "condition_number": flat(cn), "vdp": vdp,
    }


def collinearity_mgwr(coords, y, x, standardize=True, **kw):
    if standardize:
        y = (y - y.mean()) / y.std()
        x = (x - x.mean(0)) / x.std(0)
    yy = y.reshape(-1, 1)
    model = {k: kw.pop(k) for k in ("kernel", "fixed") if k in kw}
    sel = Sel_BW(coords, yy, x, multi=True, n_jobs=1, **model)
    sel.search(verbose=False, **kw)
    r = MGWR(coords, yy, x, sel, n_jobs=1, **model).fit()
    cn, vdp = r.local_collinearity()
    return {
        "options": model, "y": y, "x": x.T,
        "bws": [float(b) for b in sel.bw[0]],
        "condition_number": flat(cn), "vdp": vdp,
    }


def collinearity_fixture():
    gc, gy, gx = georgia()
    sc, sy, sx = synthetic()
    out = {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "datasets": {
            "georgia": {"coords": gc, "y": gy, "x": gx.T},
            "synthetic": {"coords": sc, "y": sy, "x": sx.T},
        },
        "gwr": {}, "mgwr": {},
    }
    g = out["gwr"]
    g["georgia_adaptive"] = {"dataset": "georgia", **collinearity_gwr(gc, gy, gx, 93)}
    g["georgia_gaussian_fixed"] = {
        "dataset": "georgia",
        **collinearity_gwr(gc, gy, gx, 60000.0, kernel="gaussian", fixed=True)}
    g["georgia_exponential_adaptive"] = {
        "dataset": "georgia", **collinearity_gwr(gc, gy, gx, 50, kernel="exponential")}
    g["georgia_global"] = {"dataset": "georgia", **collinearity_gwr(gc, gy, gx, np.inf)}
    g["synthetic_no_constant"] = {
        "dataset": "synthetic",
        **collinearity_gwr(sc, sy, sx, 45, constant=False, spherical=True)}
    rng = np.random.default_rng(7)
    coords = lattice(rng, 7, 7, 0.0) + rng.normal(scale=0.05, size=(49, 2))
    x = rng.normal(size=(49, 2))
    y = 1 + (0.5 + 0.3 * coords[:, 0]) * x[:, 0] + 0.8 * x[:, 1] + rng.normal(size=49) * 0.3
    out["mgwr_coords"] = coords
    out["mgwr_georgia_coords"] = gc
    m = out["mgwr"]
    m["lattice49_fixed"] = collinearity_mgwr(coords, y, x, fixed=True)
    m["lattice49_adaptive"] = collinearity_mgwr(coords, y, x, fixed=False)
    m["georgia_adaptive"] = collinearity_mgwr(gc, gy, gx, fixed=False)
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["predict", "collinearity"]
    if "predict" in which:
        dump_json(FIXTURES / "regression" / "gwr_predict.json", predict_fixture())
    if "collinearity" in which:
        dump_json(FIXTURES / "regression" / "gwr_collinearity.json", collinearity_fixture())
