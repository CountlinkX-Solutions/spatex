"""Generate libpysal.graph KNN and distance-band oracle fixtures.

Run from the repository root (see test/reference/README.md):

    <venv>/bin/python -I test/reference/generate_weights_knn.py

Reads fixtures/datasets/columbus.json; writes fixtures/weights/knn_*.json and
fixtures/weights/distance_band_*.json.

Each fixture holds ``points`` (n x 2) and a list of ``cases``. A case records the builder
``params`` and the resulting graph in positional form (``common.export_graph``): ``focal``,
``neighbor`` and ``weight`` mirror ``Graph.adjacency`` row order, plus ``isolates`` and
``cardinalities``. Cases with ``"ties": true`` come from point sets with exactly equidistant
neighbors; libpysal resolves those ties in scikit-learn KDTree order, Spatex by
``(distance, index)``, so tests compare per-row neighbor *distances* instead of ids there.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from scipy.spatial import KDTree  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402


def case(name, params, graph, **extra):
    ids = list(graph.unique_ids)
    out = {
        "name": name,
        "params": params,
        "isolates": [ids.index(i) for i in graph.isolates],
        "cardinalities": graph.cardinalities.reindex(ids).to_numpy(int),
        "adjacency": export_graph(graph),
        **extra,
    }
    del out["adjacency"]["ids"]
    return out


def knn_cases(points, ks, coincident="raise", **extra):
    return [
        case(
            f"k{k}" if coincident == "raise" else f"{coincident}_k{k}",
            {"k": k, "coincident": coincident},
            Graph.build_knn(points, k=k, coplanar=coincident),
            **extra,
        )
        for k in ks
    ]


def band_cases(points, thresholds):
    cases = []
    for t in thresholds:
        cases.append(case(f"t{t!r}_binary", {"threshold": t, "binary": True}, Graph.build_distance_band(points, t)))
        for alpha in (-1.0, -2.0, 1.0):
            cases.append(
                case(
                    f"t{t!r}_alpha{alpha!r}",
                    {"threshold": t, "binary": False, "alpha": alpha},
                    Graph.build_distance_band(points, t, binary=False, alpha=alpha),
                )
            )
    return cases


def fixture(name, description, points, cases):
    return {
        "name": name,
        "description": description,
        "meta": meta(),
        "n": len(points),
        "points": np.asarray(points, dtype=float),
        "cases": cases,
    }


def min_threshold(points):
    d, _ = KDTree(points).query(points, k=2)
    return float(d[:, 1].max())


def main():
    out = FIXTURES / "weights"

    columbus = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = shapely.from_wkb([bytes.fromhex(h) for h in columbus["wkb_hex"]])
    centroids = shapely.get_coordinates(shapely.centroid(geoms))

    rng = np.random.default_rng(20261025)
    random_pts = rng.uniform(0.0, 10.0, size=(200, 2))

    grid = np.array([[float(c), float(r)] for r in range(6) for c in range(6)])

    coincident_small = np.array([[0, 0], [0, 0], [1, 0], [3, 0], [3, 1]], dtype=float)
    triple = np.array([[0, 0], [2, 0], [0, 0], [2, 1], [0, 0], [5, 5], [1.1, 4]], dtype=float)
    mixed = rng.uniform(0.0, 5.0, size=(30, 2))
    mixed[7] = mixed[3]
    mixed[19] = mixed[3]
    mixed[25] = mixed[11]
    mixed[28] = mixed[2]
    mixed[29] = mixed[2]

    t_col = min_threshold(centroids)
    line = np.array([[0.0, 0], [1, 0], [2, 0], [3.5, 0], [10, 0]])

    dump_json(out / "knn_columbus.json", fixture(
        "knn_columbus", "KNN on the 49 columbus polygon centroids (general position).",
        centroids, knn_cases(centroids, (1, 4, 8))))
    dump_json(out / "knn_random200.json", fixture(
        "knn_random200", "KNN on 200 seeded uniform points in [0, 10)^2 (seed 20261025).",
        random_pts, knn_cases(random_pts, (1, 5, 10))))
    dump_json(out / "knn_grid_ties.json", fixture(
        "knn_grid_ties", "KNN on a 6x6 unit lattice: exact distance ties. Compare distances only.",
        grid, knn_cases(grid, (1, 3, 4, 8), ties=True)))
    dump_json(out / "knn_coincident.json", fixture(
        "knn_coincident", "KNN with coincident points, coincident='clique'.",
        coincident_small, knn_cases(coincident_small, (1, 2), "clique")))
    dump_json(out / "knn_coincident_triple.json", fixture(
        "knn_coincident_triple", "Three points on one site plus uniques, coincident='clique'.",
        triple, knn_cases(triple, (1, 2, 3), "clique")))
    dump_json(out / "knn_coincident_mixed.json", fixture(
        "knn_coincident_mixed", "30 random points with a triple, a pair and another triple site.",
        mixed, knn_cases(mixed, (1, 4), "clique")))

    dump_json(out / "distance_band_columbus.json", fixture(
        "distance_band_columbus",
        "Distance band on columbus centroids: the minimum island-free threshold and 0.8 of it (islands).",
        centroids, band_cases(centroids, (t_col, 0.8 * t_col))))
    dump_json(out / "distance_band_random200.json", fixture(
        "distance_band_random200", "Distance band on 200 seeded uniform points.",
        random_pts, band_cases(random_pts, (0.5, 1.2))))
    dump_json(out / "distance_band_grid.json", fixture(
        "distance_band_grid",
        "Distance band on a 6x6 lattice; thresholds sit exactly on lattice distances (inclusive).",
        grid, band_cases(grid, (1.0, 2.0 ** 0.5, 2.0))))
    dump_json(out / "distance_band_line.json", fixture(
        "distance_band_line", "Line x = [0, 1, 2, 3.5, 10]; exact-threshold pair and islands.",
        line, band_cases(line, (1.0, 1.5))))
    dump_json(out / "distance_band_coincident.json", fixture(
        "distance_band_coincident", "Binary distance band keeps coincident pairs.",
        coincident_small, [case("t1_binary", {"threshold": 1.0, "binary": True},
                                Graph.build_distance_band(coincident_small, 1.0))]))


if __name__ == "__main__":
    main()
