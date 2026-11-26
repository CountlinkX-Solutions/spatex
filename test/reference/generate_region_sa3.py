"""Generate spopt 0.8.0 oracle fixtures for SA3 (spatially constrained agglomerative clustering
with HDBSCAN-style extraction).

Run from the repository root under a memory cap (see test/reference/README.md):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_region_sa3.py

Writes test/reference/fixtures/region/sa3.json.

SA3 is deterministic: labels are compared exactly and the per-component linkage matrices
(scikit-learn children and distances, after spopt's monotonicity correction) to 1e-12. The
linkage of every component SA3 clusters is captured by wrapping ``SA3._get_tree`` (raw tree)
and ``spopt.region.sa3.extract_clusters`` (tree after the correction). Every ``W`` is rebuilt
with sorted neighbor lists and ids ``0..n-1``.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import spopt  # noqa: E402
import spopt.region.sa3 as sa3_mod  # noqa: E402
from spopt.region import SA3  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

OUT = FIXTURES / "region" / "sa3.json"
warnings.simplefilter("ignore")
ATTRS = ["CRIME", "INC", "HOVAL"]


def sorted_w(neighbors):
    n = len(neighbors)
    return libpysal.weights.W(
        {i: sorted(int(j) for j in neighbors[i]) for i in range(n)},
        ids=list(range(n)),
    )


def lattice(r, c, queen=False):
    w = libpysal.weights.lat2W(r, c, rook=not queen)
    return sorted_w({i: w.neighbors[i] for i in range(w.n)})


def disjoint(*ws):
    nb, off = {}, 0
    for w in ws:
        for i in range(w.n):
            nb[i + off] = [j + off for j in w.neighbors[i]]
        off += w.n
    return sorted_w(nb)


def graph_json(w):
    focal, neighbor = [], []
    for i in range(w.n):
        for j in sorted(w.neighbors[i]):
            focal.append(i)
            neighbor.append(int(j))
    return {"n": w.n, "focal": focal, "neighbor": neighbor}


def run(w, x, min_cluster_size, extraction):
    """Run SA3 and capture the raw and corrected linkage of every clustered component."""
    raw, fixed = [], []
    orig_get_tree = SA3._get_tree
    orig_extract = sa3_mod.extract_clusters

    def get_tree(self, *a, **k):
        t = orig_get_tree(self, *a, **k)
        raw.append(t.copy())
        return t

    def extract(tree, *a, **k):
        fixed.append(np.array(tree, copy=True))
        return orig_extract(tree, *a, **k)

    SA3._get_tree = get_tree
    sa3_mod.extract_clusters = extract
    try:
        gdf = pd.DataFrame(x, columns=[f"a{j}" for j in range(x.shape[1])])
        model = SA3(
            gdf, w, list(gdf.columns), min_cluster_size=min_cluster_size, extraction=extraction
        )
        model.solve()
    finally:
        SA3._get_tree = orig_get_tree
        sa3_mod.extract_clusters = orig_extract
    labels = [int(v) for v in np.asarray(model.labels_)]
    linkage = [
        {
            "children": [[int(a), int(b)] for a, b in t[:, :2]],
            "distances": [float(d) for d in t[:, 2]],
            "sizes": [int(s) for s in t[:, 3]],
            "raw_distances": [float(d) for d in r[:, 2]],
            "corrected": bool(not np.array_equal(t[:, 2], r[:, 2])),
        }
        for t, r in zip(fixed, raw)
    ]
    return {
        "min_cluster_size": min_cluster_size,
        "extraction": extraction,
        "labels": labels,
        "linkage": linkage,
    }


def dataset(name, w, x, sizes):
    cases = []
    for m in sizes:
        for ext in ("eom", "leaf"):
            cases.append(run(w, x, m, ext))
    return {"name": name, "graph": graph_json(w), "x": x.tolist(), "cases": cases}


def main():
    gdf = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    w0 = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    wc = sorted_w({i: w0.neighbors[i] for i in range(w0.n)})
    xc = gdf[ATTRS].to_numpy(float)

    datasets = [dataset("columbus", wc, xc, (2, 3, 5, 8, 15, 30))]

    # Disconnected: two lattices, a small component (<= min_cluster_size) and an island.
    wd = disjoint(lattice(6, 6), lattice(4, 5, queen=True), lattice(2, 2), sorted_w({0: []}))
    rng = np.random.default_rng(21)
    xd = np.round(rng.normal(size=(wd.n, 2)) * 3, 3)
    datasets.append(dataset("disconnected", wd, xd, (2, 4, 6, 15)))

    # Ties: integer attributes on a lattice give many equal Ward distances.
    wt = lattice(8, 8)
    xt = np.random.default_rng(3).integers(0, 3, size=(wt.n, 2)).astype(float)
    datasets.append(dataset("lattice_ties", wt, xt, (2, 3, 5, 10)))

    # Larger lattice with two attribute blobs.
    wb = lattice(15, 12, queen=True)
    rng = np.random.default_rng(8)
    xb = np.round(rng.normal(size=(wb.n, 3)) * 2 + (np.arange(wb.n) % 12 > 5)[:, None] * 5, 3)
    datasets.append(dataset("lattice180", wb, xb, (3, 8, 20)))

    m = meta()
    m.update(
        spopt=spopt.__version__,
        scikit_learn=sklearn.__version__,
        scipy=scipy.__version__,
    )
    import fast_hdbscan

    m["fast_hdbscan"] = fast_hdbscan.__version__ if hasattr(fast_hdbscan, "__version__") else "0.3.4"
    dump_json(OUT, {"meta": m, "datasets": datasets})
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
