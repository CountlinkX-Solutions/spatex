"""Generate libpysal.graph triangulation oracle fixtures.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_triangulation.py

Reads fixtures/datasets/columbus.json; writes fixtures/weights/triangulation_*.json.

Layout is that of ``generate_weights_knn.py``: ``points`` plus ``cases`` with ``params``
(``method``, ``kernel``, ``bandwidth``, ``taper``, ``decay``, ``coincident``) and the positional
``adjacency``. All point sets are in general position (no four cocircular points), where the
Delaunay triangulation is unique and Qhull and Spatex agree; cocircular input (e.g. a square)
is implementation-defined and deliberately avoided. Voronoi is not covered (unsupported).

The ``triangulation_dropped_node`` fixture records libpysal's behavior when a finite bandwidth
tapers away every edge of a node: the node vanishes from the graph (``pysal_ids`` lists the
survivors). Spatex keeps it as an island (known deviation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from libpysal.graph import Graph  # noqa: E402

from common import FIXTURES, dump_json  # noqa: E402
from generate_weights_knn import case, fixture  # noqa: E402

METHODS = ("delaunay", "gabriel", "relative_neighborhood")


def tcase(points, method, kernel="boxcar", bandwidth=np.inf, taper=True, decay=False, coincident="raise", **extra):
    params = {
        "method": method,
        "kernel": kernel,
        "bandwidth": bandwidth,
        "taper": taper,
        "decay": decay,
        "coincident": coincident,
    }
    graph = Graph.build_triangulation(
        points, method=method, kernel=kernel, bandwidth=bandwidth, taper=taper, decay=decay, coplanar=coincident
    )
    tag = "_".join(
        [method, kernel]
        + ([] if np.isinf(bandwidth) else [f"bw{bandwidth}"])
        + ([] if taper else ["notaper"])
        + (["decay"] if decay else [])
        + ([coincident] if coincident != "raise" else [])
    )
    return case(tag, params, graph, **extra)


def standard(points, bandwidth):
    cases = [tcase(points, m) for m in METHODS]
    for m in METHODS:
        cases.append(tcase(points, m, kernel="triangular", bandwidth=bandwidth))
        cases.append(tcase(points, m, kernel="gaussian", bandwidth=bandwidth, taper=False))
    cases.append(tcase(points, "delaunay", kernel="gaussian", bandwidth=bandwidth, decay=True))
    cases.append(tcase(points, "delaunay", kernel="exponential"))
    cases.append(tcase(points, "gabriel", kernel="identity"))
    return cases


def main():
    out = FIXTURES / "weights"

    columbus = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = shapely.from_wkb([bytes.fromhex(h) for h in columbus["wkb_hex"]])
    centroids = shapely.get_coordinates(shapely.centroid(geoms))
    random_pts = np.random.default_rng(20261025).uniform(0.0, 10.0, size=(200, 2))
    # Seven points from the spec; (1, 1) is cocircular with (0,0), (1,0), (0,1), so nudge it.
    small = np.array(
        [[0, 0], [1, 0], [0, 1], [1.1, 0.95], [0.5, 0.5], [2, 0.2], [1.7, 1.6]], dtype=float
    )

    dump_json(out / "triangulation_columbus.json", fixture(
        "triangulation_columbus", "Triangulation graphs on the 49 columbus polygon centroids.",
        centroids, standard(centroids, 8.0)))
    dump_json(out / "triangulation_random200.json", fixture(
        "triangulation_random200", "Triangulation graphs on 200 seeded uniform points (seed 20261025).",
        random_pts, standard(random_pts, 1.2)))
    dump_json(out / "triangulation_small.json", fixture(
        "triangulation_small", "Seven hand-placed points in general position.",
        small, standard(small, 1.3)))

    dup = np.vstack([small, small[4], small[2]])  # points 7 and 8 duplicate points 4 and 2
    dup_cases = [tcase(dup, m, coincident="clique") for m in METHODS]
    dup_cases.append(tcase(dup, "delaunay", kernel="triangular", bandwidth=1.3, coincident="clique"))
    dump_json(out / "triangulation_coincident.json", fixture(
        "triangulation_coincident", "Seven points plus two duplicates, coincident='clique'.",
        dup, dup_cases))

    # A finite bandwidth below the shortest edge of one node: libpysal drops it from the graph.
    g = Graph.build_triangulation(small, method="delaunay", bandwidth=0.9, kernel="boxcar")
    pysal_ids = [int(i) for i in g.unique_ids]
    assert len(pysal_ids) < len(small), "expected a dropped node"
    params = {"method": "delaunay", "kernel": "boxcar", "bandwidth": 0.9, "taper": True,
              "decay": False, "coincident": "raise"}
    c = case("delaunay_boxcar_bw0.9", params, g, quirk="drops_tapered_node", pysal_ids=pysal_ids)
    dump_json(out / "triangulation_dropped_node.json", fixture(
        "triangulation_dropped_node",
        "libpysal drops a node whose edges are all tapered away; Spatex keeps it as an island.",
        small, [c]))


if __name__ == "__main__":
    main()
