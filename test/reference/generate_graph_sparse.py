"""Generate libpysal ``Graph.sparse`` oracle fixtures.

Run from the repository root (see test/reference/README.md):

    <venv>/bin/python -I test/reference/generate_graph_sparse.py

Writes test/reference/fixtures/graph/sparse_*.json.

Per fixture: the graph as positional adjacency (``common.export_graph``, original weights) and,
keyed by transform (O, R), the ``Graph.sparse`` CSR of that transform: ``indptr``, ``indices``,
``data`` and ``shape``. libpysal stores every adjacency row, so island self-loops and explicit
zero weights are explicit zeros in the CSR; the generator asserts that and that the indices
are sorted within each row.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely.geometry import Point, box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402


def sparse_export(graph):
    s = graph.sparse
    assert s.format == "csr" and s.has_sorted_indices
    assert s.nnz == len(graph.adjacency)  # nothing dropped, explicit zeros kept
    return {
        "indptr": s.indptr.tolist(),
        "indices": s.indices.tolist(),
        "data": s.data.astype(float).tolist(),
        "shape": list(s.shape),
    }


def fixture(name, description, graph):
    adjacency = export_graph(graph)
    ids = adjacency.pop("ids")
    return {
        "name": name,
        "description": description,
        "meta": meta(),
        "n": len(ids),
        "ids": ids,
        "adjacency": adjacency,
        "sparse": {"O": sparse_export(graph), "R": sparse_export(graph.transform("r"))},
    }


def main():
    out = FIXTURES / "graph"

    cells = [box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)]
    cells.append(box(10, 10, 11, 11))  # island
    queen = Graph.build_contiguity(gpd.GeoSeries(cells), rook=False)
    assert len(queen.isolates) == 1
    dump_json(
        out / "sparse_queen_island.json",
        fixture(
            "sparse_queen_island",
            "Queen contiguity of a 3x3 grid plus one distant square (position 9 is an island).",
            queen,
        ),
    )

    rng = np.random.default_rng(98)
    pts = rng.random((12, 2)) * 10
    pts_gs = gpd.GeoSeries([Point(x, y) for x, y in pts])
    kernel = Graph.build_kernel(pts_gs, kernel="gaussian", bandwidth=4.0)
    dump_json(
        out / "sparse_kernel.json",
        fixture("sparse_kernel", "Gaussian kernel graph, bandwidth 4, 12 random points.", kernel),
    )

    band = Graph.build_distance_band(pts_gs, threshold=2.5, binary=False, alpha=-1.0)
    dump_json(
        out / "sparse_distance_band.json",
        fixture(
            "sparse_distance_band",
            "Inverse-distance band graph (threshold 2.5, alpha -1), 12 random points; may hold islands.",
            band,
        ),
    )

    hand = Graph.from_arrays(
        ["a", "a", "b", "c", "d", "d", "e", "f"],
        ["b", "c", "a", "d", "c", "a", "e", "a"],
        [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.0, 0.0],
    )
    dump_json(
        out / "sparse_explicit_zero.json",
        fixture(
            "sparse_explicit_zero",
            "Hand-written a..f: e is an island (self-loop 0.0) and f -> a carries an explicit 0.0 weight.",
            hand,
        ),
    )


if __name__ == "__main__":
    main()
