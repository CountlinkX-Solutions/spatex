"""Generate libpysal.graph haversine-metric oracle fixtures.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_haversine.py

Writes fixtures/weights/haversine_europe.json and haversine_antimeridian.json.

Layout as ``generate_weights_knn.py``; ``points`` are ``[lon, lat]`` degrees. libpysal 4.15.1
leaves haversine KNN distances in radians (its ``* 6371`` is a no-op), so only unit-free
outputs are recorded: KNN neighbor sets and kernel weights whose bandwidth comes from the data
(``d / h`` cancels the unit). Spatex reports kilometres; fixed bandwidths are not comparable.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas  # noqa: E402
import numpy as np  # noqa: E402
from libpysal.graph import Graph  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

KERNELS = ("triangular", "gaussian", "bisquare")


def case(name, params, graph):
    ids = list(graph.unique_ids)
    out = {
        "name": name,
        "params": params,
        "isolates": [ids.index(i) for i in graph.isolates],
        "cardinalities": graph.cardinalities.reindex(ids).to_numpy(int),
        "adjacency": export_graph(graph),
    }
    del out["adjacency"]["ids"]
    return out


def cases(points):
    geoms = geopandas.GeoSeries(geopandas.points_from_xy(points[:, 0], points[:, 1]))
    out = [
        case(f"knn_k{k}", {"k": k}, Graph.build_knn(geoms, k=k, metric="haversine"))
        for k in (1, 2, 3, 5)
    ]
    for kernel in KERNELS:
        for bandwidth in (None, "adaptive"):
            for k in (3, 6):
                params = {"kernel": kernel, "k": k, "bandwidth": bandwidth}
                g = Graph.build_kernel(
                    geoms,
                    kernel=kernel,
                    k=k,
                    bandwidth=bandwidth,
                    metric="haversine",
                )
                out.append(case(f"{kernel}_{bandwidth or 'default'}_k{k}", params, g))
    return out


def fixture(name, description, points):
    return {
        "name": name,
        "description": description,
        "meta": meta(),
        "n": len(points),
        "points": points,
        "cases": cases(points),
    }


def main():
    rng = np.random.default_rng(57)
    europe = np.column_stack((rng.uniform(-10, 25, 60), rng.uniform(36, 60, 60)))
    antimeridian = np.column_stack(
        (np.concatenate((rng.uniform(170, 179.9, 15), rng.uniform(-179.9, -170, 15))),
         rng.uniform(-30, 30, 30))
    )
    out = FIXTURES / "weights"
    dump_json(
        out / "haversine_europe.json",
        fixture("haversine_europe", "60 random lon/lat points over Europe", europe),
    )
    dump_json(
        out / "haversine_antimeridian.json",
        fixture(
            "haversine_antimeridian",
            "30 random lon/lat points on both sides of the antimeridian",
            antimeridian,
        ),
    )


if __name__ == "__main__":
    main()
