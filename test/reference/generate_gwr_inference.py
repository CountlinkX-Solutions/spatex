"""Generate the GWR/MGWR bandwidth-interval and spatial-variability oracle fixture
(`fixtures/regression/gwr_bws.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_gwr_inference.py

Ground truth is mgwr 2.2.1.

* `intervals` - `GWRResults.get_bws_intervals(selector, level)` for golden-section and interval
  searches, with the selector's search history (`sel_hist`).
* `mgwr_intervals` - `MGWRResults.get_bws_intervals(selector, level)`, with the selector's
  per-search histories (`sel_hist`, one list per searched covariate and backfitting iteration).
* `variability` - `GWRResults.spatial_variability(selector, n_iters, seed)`: the observed
  `np.std(params, axis=0)` and the standard deviations of mgwr's own Monte Carlo draws (the
  loop of `spatial_variability` repeated so the draws can be kept; its p-values are asserted
  equal to mgwr's method). mgwr uses numpy's generator, so Spatex is compared statistically.
"""

import copy
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import mgwr  # noqa: E402
from mgwr.gwr import GWR, MGWR  # noqa: E402
from mgwr.sel_bw import Sel_BW  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_gwr_predict import georgia, lattice  # noqa: E402

warnings.simplefilter("ignore")

LEVELS = [0.95, 0.9, 0.5, 0.99]


def hist(sel):
    return [[float(b), float(np.asarray(s).ravel()[0])] for b, s in sel.sel_hist]


def interval_case(dataset, coords, y, x, **kw):
    search = {k: kw.pop(k) for k in ("criterion", "bw_min", "bw_max", "search_method", "interval")
              if k in kw}
    sel = Sel_BW(coords, y.reshape(-1, 1), x, n_jobs=1, **kw)
    bw = sel.search(**search)
    r = GWR(coords, y.reshape(-1, 1), x, bw, n_jobs=1, **kw).fit()
    return {
        "dataset": dataset,
        "options": {**kw, **search},
        "bw": float(bw),
        "history": hist(sel),
        "intervals": {str(lv): [float(v) for v in r.get_bws_intervals(sel, level=lv)]
                      for lv in LEVELS},
    }


def mgwr_interval_case(dataset, coords, y, x, **kw):
    y = (y - y.mean()) / y.std()
    x = (x - x.mean(0)) / x.std(0)
    yy = y.reshape(-1, 1)
    model = {k: kw.pop(k) for k in ("kernel", "fixed") if k in kw}
    sel = Sel_BW(coords, yy, x, multi=True, n_jobs=1, **model)
    sel.search(verbose=False, **kw)
    m = MGWR(coords, yy, x, sel, n_jobs=1, **model).fit()
    return {
        "dataset": dataset,
        "options": {**model, **kw},
        "y": y, "x": x.T,
        "bws": [float(b) for b in sel.bw[0]],
        "sel_hist": [[[float(b), float(np.asarray(s).ravel()[0])] for b, s in h]
                     for h in sel.sel_hist],
        "intervals": {str(lv): [[float(v) for v in iv] for iv in m.get_bws_intervals(sel, level=lv)]
                      for lv in LEVELS},
    }


def variability_case(dataset, coords, y, x, n_iters, seed, **kw):
    search = {k: kw.pop(k) for k in ("criterion", "bw_min", "bw_max", "search_method", "interval")
              if k in kw}
    yy = y.reshape(-1, 1)
    sel = Sel_BW(coords, yy, x, n_jobs=1, **kw)
    bw = sel.search(**search)
    r = GWR(coords, yy, x, bw, n_jobs=1, **kw).fit()
    p_mgwr = r.spatial_variability(sel, n_iters=n_iters, seed=seed)

    # The same loop, keeping the draws.
    temp_sel = copy.deepcopy(sel)
    temp_gwr = copy.deepcopy(r.model)
    np.random.seed(seed)
    init_sd = np.std(r.params, axis=0)
    sds = []
    for _ in range(n_iters):
        tc = np.random.permutation(r.model.coords)
        temp_sel.coords = tc
        temp_gwr.bw = temp_sel.search(**temp_sel.search_params)
        temp_gwr.coords = tc
        sds.append(np.std(temp_gwr.fit(**temp_gwr.fit_params).params, axis=0))
    sds = np.array(sds)
    p = np.sum(sds > init_sd, axis=0) / float(n_iters)
    assert np.allclose(p, p_mgwr), (p, p_mgwr)
    return {
        "dataset": dataset,
        "options": {**kw, **search},
        "bw": float(bw),
        "n_iters": n_iters,
        "seed": seed,
        "observed_sd": init_sd,
        "p_values": p_mgwr,
        "sds": sds,
    }


def lattice49():
    rng = np.random.default_rng(7)
    coords = lattice(rng, 7, 7, 0.0) + rng.normal(scale=0.05, size=(49, 2))
    x = rng.normal(size=(49, 2))
    y = 1 + (0.5 + 0.3 * coords[:, 0]) * x[:, 0] + 0.8 * x[:, 1] + rng.normal(size=49) * 0.3
    return coords, y, x


def lattice100():
    rng = np.random.default_rng(99)
    coords = lattice(rng, 10, 10, 0.5)
    x = rng.normal(size=(100, 2))
    y = (1 + 0.1 * coords[:, 0] + (0.5 + 0.25 * coords[:, 1]) * x[:, 0]
         + 0.7 * x[:, 1] + rng.normal(size=100) * 0.4)
    return coords, y, x


def fixture():
    gc, gy, gx = georgia()
    lc, ly, lx = lattice49()
    mc, my, mx = lattice100()
    datasets = {
        "georgia": {"coords": gc, "y": gy, "x": gx.T},
        "lattice49": {"coords": lc, "y": ly, "x": lx.T},
        "lattice100": {"coords": mc, "y": my, "x": mx.T},
    }
    iv = {}
    iv["georgia_adaptive"] = interval_case("georgia", gc, gy, gx)
    iv["georgia_adaptive_bic"] = interval_case("georgia", gc, gy, gx, criterion="BIC")
    iv["georgia_fixed_gaussian"] = interval_case("georgia", gc, gy, gx, fixed=True, kernel="gaussian")
    iv["georgia_fixed_interval"] = interval_case(
        "georgia", gc, gy, gx, fixed=True, search_method="interval",
        bw_min=211001.0, bw_max=211035.0, interval=2)
    iv["georgia_adaptive_interval"] = interval_case(
        "georgia", gc, gy, gx, search_method="interval", bw_min=50, bw_max=110, interval=5)
    iv["lattice49_fixed"] = interval_case("lattice49", lc, ly, lx, fixed=True)
    iv["lattice100_adaptive"] = interval_case("lattice100", mc, my, mx)
    mi = {}
    mi["lattice49_fixed"] = mgwr_interval_case("lattice49", lc, ly, lx, fixed=True)
    mi["lattice49_adaptive"] = mgwr_interval_case("lattice49", lc, ly, lx, fixed=False)
    mi["lattice100_adaptive"] = mgwr_interval_case("lattice100", mc, my, mx, fixed=False)
    mi["georgia_adaptive"] = mgwr_interval_case("georgia", gc, gy, gx, fixed=False)
    var = {}
    var["lattice49_fixed"] = variability_case("lattice49", lc, ly, lx, 300, 11, fixed=True)
    var["lattice49_adaptive"] = variability_case("lattice49", lc, ly, lx, 300, 12)
    var["lattice100_fixed_gaussian"] = variability_case(
        "lattice100", mc, my, mx, 200, 13, fixed=True, kernel="gaussian")
    var["georgia_adaptive"] = variability_case("georgia", gc, gy, gx, 100, 14)
    return {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "datasets": datasets,
        "intervals": iv,
        "mgwr_intervals": mi,
        "variability": var,
    }


if __name__ == "__main__":
    dump_json(FIXTURES / "regression" / "gwr_bws.json", fixture())
