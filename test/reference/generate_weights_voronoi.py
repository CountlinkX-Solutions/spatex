"""Generate libpysal.graph Voronoi-triangulation oracle fixtures.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_voronoi.py

Writes fixtures/weights/voronoi.json: ``build_triangulation(points, method="voronoi")`` (rook
contiguity of the GEOS Voronoi cells clipped to ``clip``) on seeded random point sets and a
square lattice (cocircular quadruples: zero-length Voronoi edges). Layout as
``generate_weights_knn.py``: ``points`` plus ``cases`` with ``params`` and the positional
``adjacency``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from libpysal.graph import Graph  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402


def case(name, points, clip):
    g = Graph.build_triangulation(points, method="voronoi", clip=clip)
    ids = list(g.unique_ids)
    out = {
        "name": name,
        "params": {"clip": clip},
        "isolates": [ids.index(i) for i in g.isolates],
        "cardinalities": g.cardinalities.reindex(ids).to_numpy(int),
        "adjacency": export_graph(g),
    }
    del out["adjacency"]["ids"]
    return out


def main():
    rng = np.random.default_rng(56)
    sets = {
        "random40": rng.uniform(0, 100, (40, 2)),
        "random200": rng.normal(0, 10, (200, 2)),
        "lattice": np.array([[x, y] for y in range(5) for x in range(6)], dtype=float),
    }
    fixtures = []
    for name, pts in sets.items():
        for clip in ("bounding_box", "convex_hull", None):
            fixtures.append({"set": name, "points": pts, **case(f"{name}_{clip}", pts, clip)})
    dump_json(FIXTURES / "weights" / "voronoi.json", {
        "name": "voronoi", "description": "build_triangulation(method='voronoi') rook cells",
        "meta": meta(), "cases": fixtures,
    })


if __name__ == "__main__":
    main()
