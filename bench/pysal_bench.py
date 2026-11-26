"""PySAL timings matching the Benchee scripts in this directory.

Run (the repository convention caps address space at 2 GiB):

    prlimit --as=2147483648 <venv>/bin/python -I bench/pysal_bench.py [case ...]

Writes bench/results/pysal.json. Data has the same shape as the Elixir scripts (same sizes, same
lattice, uniform random points, normal covariates); random values come from numpy, so they are not
identical numbers, which does not matter for timing. Library versions are recorded in the output.

Threads: nothing is pinned. numpy/scipy use their BLAS defaults and numba uses its default
threading layer; esda's `n_jobs` is 1 (its default) unless a case says otherwise.
"""

import json
import os
import sys
import time

import numpy as np
import shapely
from shapely.geometry import box

import esda
import geopandas
import libpysal
import mapclassify
import mgwr
import spreg
from libpysal.graph import Graph
from mgwr.sel_bw import Sel_BW

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "pysal.json")
COLUMBUS = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "test",
    "reference",
    "fixtures",
    "datasets",
    "columbus.json",
)


def lattice_polygons(rows, cols):
    return [box(c, r, c + 1.0, r + 1.0) for r in range(rows) for c in range(cols)]


def lattice_graph(side):
    g = Graph.build_contiguity(geopandas.GeoSeries(lattice_polygons(side, side)), rook=True)
    return g.transform("r")


def timed(fn, reps):
    """First call (includes JIT/compile warmup), then `reps` timed calls."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return {
        "median_ms": times[len(times) // 2] * 1e3,
        "mean_ms": sum(times) / len(times) * 1e3,
        "min_ms": times[0] * 1e3,
        "samples": len(times),
        "first_call_ms": first * 1e3,
    }


RESULTS = {}


def suffix():
    n = os.environ.get("OPENBLAS_NUM_THREADS")
    return f" [OPENBLAS_NUM_THREADS={n}]" if n else ""


def run(key, fn, reps):
    """Time `fn`; a MemoryError under the address-space cap is recorded, not fatal."""
    try:
        record(key, timed(fn, reps))
    except MemoryError as exc:
        record(key, {"error": f"MemoryError under the 2 GiB address-space cap ({exc})"})


def record(key, entry):
    key = key + suffix()
    RESULTS[key] = entry
    if "error" in entry:
        print(f"{key}: {entry['error']}", flush=True)
    else:
        print(f"{key}: median {entry['median_ms']:.3f} ms (first call {entry['first_call_ms']:.1f} ms)", flush=True)
    existing = {}
    if os.path.exists(OUT):
        with open(OUT) as f:
            existing = json.load(f)
    existing.update(RESULTS)
    existing["_versions"] = {
        "libpysal": libpysal.__version__,
        "esda": esda.__version__,
        "spreg": spreg.__version__,
        "mapclassify": mapclassify.__version__,
        "mgwr": mgwr.__version__,
        "numpy": np.__version__,
        "shapely": shapely.__version__,
        "python": sys.version.split()[0],
    }
    with open(OUT, "w") as f:
        json.dump(existing, f, indent=2)


def case_contiguity():
    with open(COLUMBUS) as f:
        data = json.load(f)
    geoms = geopandas.GeoSeries([shapely.from_wkb(bytes.fromhex(h)) for h in data["wkb_hex"]])
    run("build_contiguity columbus (n=49, rook)", lambda: Graph.build_contiguity(geoms, rook=True), 20)
    lat = geopandas.GeoSeries(lattice_polygons(100, 100))
    run("build_contiguity lattice (n=10000, rook)", lambda: Graph.build_contiguity(lat, rook=True), 5)


def case_knn():
    for n, reps in [(10_000, 5), (100_000, 3)]:
        rng = np.random.default_rng(1)
        pts = rng.random((n, 2))
        run(f"build_knn (n={n}, k=8)", lambda: Graph.build_knn(pts, 8), reps)


def case_moran():
    g = lattice_graph(100)
    y = np.random.default_rng(2).standard_normal(10_000)
    run("moran (n=10000, 999 perms)", lambda: esda.Moran(y, g, permutations=999), 5)


def case_moran_local():
    g = lattice_graph(100)
    y = np.random.default_rng(2).standard_normal(10_000)
    run("moran_local (n=10000, 999 perms)", lambda: esda.Moran_Local(y, g, permutations=999, seed=0), 3)
    run("moran_local (n=10000, 999 perms, no sims kept)", lambda: esda.Moran_Local(y, g, permutations=999, seed=0, keep_simulations=False), 3)
    run("moran_local (n=10000, 999 perms, n_jobs=-1)", lambda: esda.Moran_Local(y, g, permutations=999, seed=0, n_jobs=-1), 3)


def case_fisher_jenks():
    y = np.exp(np.random.default_rng(3).standard_normal(5_000))
    run("fisher_jenks (n=5000, k=5)", lambda: mapclassify.FisherJenks(y, k=5), 5)


def case_ols():
    n = 10_000
    rng = np.random.default_rng(4)
    X = rng.standard_normal((n, 3))
    y = (1.0 + X @ np.array([0.5, -0.3, 0.2]) + rng.standard_normal(n)).reshape(-1, 1)
    w = lattice_graph(100).to_W()
    run("ols + spatial diagnostics (n=10000, k=4)", lambda: spreg.OLS(y, X, w=w, spat_diag=True, moran=True), 3)


def case_ml_lag():
    n = 2_500
    rng = np.random.default_rng(8)
    X = rng.standard_normal((n, 2))
    y = (1.0 + X @ np.array([0.5, -0.3]) + rng.standard_normal(n)).reshape(-1, 1)
    w = lattice_graph(50).to_W()
    run("ml_lag ord (n=2500)", lambda: spreg.ML_Lag(y, X, w, method="ord"), 3)


def case_gwr():
    n = 1_000
    rng = np.random.default_rng(11)
    pts = rng.random((n, 2))
    X = rng.standard_normal((n, 2))
    y = (1.0 + (1.0 + pts[:, 0]) * X[:, 0] + (1.0 - pts[:, 1]) * X[:, 1] + 0.3 * rng.standard_normal(n)).reshape(-1, 1)
    run("gwr select_bandwidth (n=1000, k=3)", lambda: Sel_BW(pts, y, X, kernel="bisquare", fixed=False).search(criterion="AICc"), 3)


CASES = {
    "contiguity": case_contiguity,
    "knn": case_knn,
    "moran": case_moran,
    "moran_local": case_moran_local,
    "fisher_jenks": case_fisher_jenks,
    "ols": case_ols,
    "ml_lag": case_ml_lag,
    "gwr": case_gwr,
}

if __name__ == "__main__":
    for name in sys.argv[1:] or CASES:
        CASES[name]()
