"""Generate libpysal.graph.Graph oracle fixtures.

Run from the repository root (see test/reference/README.md):

    <venv>/bin/python -I test/reference/generate_graph.py

Writes test/reference/fixtures/graph/*.json and datasets/columbus.json.

Per graph fixture:
  n, ids, symmetric (``Graph.asymmetry()`` is empty), isolates (positions), cardinalities
  (positional, ``Graph.cardinalities``), variable {name, values} and ``transforms`` keyed by
  O (original), R, B, D, V. Each transform holds the transformed adjacency (positional, see
  common.export_graph), s0/s1/s2 and ``lag`` = ``transformed_graph.lag(variable)``.

s0/s1/s2 come from common.graph_stats (dense-matrix definitions on the transformed weights),
asserted equal to ``transformed.to_W().s0/s1/s2`` at generation time.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from libpysal import examples  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, graph_stats, meta  # noqa: E402

TRANSFORMS = ["R", "B", "D", "V"]


def graph_fixture(name, description, graph, var_name, values):
    values = np.asarray(values, dtype=float)
    ids = list(graph.unique_ids)
    transforms = {}
    for label, g in [("O", graph)] + [(t, graph.transform(t)) for t in TRANSFORMS]:
        transforms[label] = {
            "adjacency": export_graph(g),
            **graph_stats(g),
            "lag": g.lag(values),
        }
        # ids are identical across transforms; keep them once at the top level.
        del transforms[label]["adjacency"]["ids"]
    return {
        "name": name,
        "description": description,
        "meta": meta(),
        "n": len(ids),
        "ids": ids,
        "symmetric": bool(graph.asymmetry().empty),
        "isolates": [ids.index(i) for i in graph.isolates],
        "cardinalities": graph.cardinalities.reindex(ids).to_numpy(int),
        "variable": {"name": var_name, "values": values},
        "transforms": transforms,
    }


def columbus():
    df = gpd.read_file(examples.get_path("columbus.shp"))
    return df


def main():
    out = FIXTURES / "graph"

    df = columbus()
    dump_json(
        FIXTURES / "datasets" / "columbus.json",
        {
            "name": "columbus",
            "description": "libpysal built-in columbus.shp: 49 polygons, WKB hex in row order.",
            "meta": meta(),
            "n": len(df),
            "wkb_hex": [g.wkb_hex.lower() for g in df.geometry],
            "attributes": {c: df[c].to_numpy(float) for c in ("CRIME", "INC", "HOVAL")},
        },
    )

    queen = Graph.build_contiguity(df, rook=False)
    dump_json(
        out / "columbus_queen.json",
        graph_fixture(
            "columbus_queen",
            "Queen contiguity of the columbus polygons; lag variable is CRIME.",
            queen,
            "CRIME",
            df["CRIME"],
        ),
    )

    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(4) for c in range(4)])
    rook = Graph.build_contiguity(cells, rook=True)
    dump_json(
        out / "lattice4x4_rook.json",
        graph_fixture(
            "lattice4x4_rook",
            "Rook contiguity of a 4x4 grid of unit squares (row-major ids 0..15); lag variable x_i = i + 1.",
            rook,
            "x",
            np.arange(1, 17, dtype=float),
        ),
    )

    # Hand-written: asymmetric weights, node "e" is an island.
    hand = Graph.from_arrays(
        ["a", "a", "b", "c", "d", "d", "e"],
        ["b", "c", "a", "d", "c", "a", "e"],
        [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.0],
    )
    dump_json(
        out / "asymmetric_island.json",
        graph_fixture(
            "asymmetric_island",
            "Hand-written graph a..e (positions 0..4) with asymmetric weights; e is an island "
            "(zero-weight self-loop). Lag variable x = [1, 2, 3, 4, 5].",
            hand,
            "x",
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ),
    )


if __name__ == "__main__":
    main()
