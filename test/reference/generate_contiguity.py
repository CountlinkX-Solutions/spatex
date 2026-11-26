"""Generate libpysal.graph contiguity oracle fixtures (queen, rook, block).

Run from the repository root (see test/reference/README.md):

    <venv>/bin/python -I test/reference/generate_contiguity.py

Writes test/reference/fixtures/weights/contiguity_*.json and block_contiguity.json.

Contiguity fixture: ``wkb_hex`` (input geometries, lowercase hex), ``ids`` and for each of
``queen`` / ``rook`` the exported adjacency (positions into ``ids``, ``Graph.adjacency`` order,
islands as zero-weight self-loops) plus ``isolates`` (positions). Only bundled libpysal
datasets are used, so generation needs no network access.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import shapely  # noqa: E402
from libpysal import examples  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely.geometry import MultiPolygon, Polygon, box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

OUT = FIXTURES / "weights"


def export(graph):
    adj = export_graph(graph)
    del adj["ids"]
    ids = list(graph.unique_ids)
    adj["isolates"] = [ids.index(i) for i in graph.isolates]
    return adj


def contiguity_fixture(name, description, geoms, ids=None):
    series = gpd.GeoSeries(list(geoms), index=ids)
    graphs = {
        "queen": Graph.build_contiguity(series, rook=False),
        "rook": Graph.build_contiguity(series, rook=True),
    }
    out_ids = list(series.index)
    for g in graphs.values():
        assert list(g.unique_ids) == out_ids, "libpysal reordered ids"
    return {
        "name": name,
        "description": description,
        "meta": meta(),
        "n": len(series),
        "ids": out_ids,
        "wkb_hex": [g.wkb_hex.lower() for g in series],
        "queen": export(graphs["queen"]),
        "rook": export(graphs["rook"]),
    }


def dataset(name, shp):
    df = gpd.read_file(examples.get_path(shp))
    return contiguity_fixture(
        f"contiguity_{name}", f"Queen and rook contiguity of libpysal's bundled {name} polygons.", df.geometry
    )


def synthetic_mix():
    geoms = []
    # -0.0 vs 0.0: both squares share the edge x = 0, one spelled with -0.0.
    geoms.append(Polygon([(-1, -1), (0.0, -1), (0.0, 0.0), (-1, 0.0)]))
    geoms.append(Polygon([(-0.0, -1), (1, -1), (1, -0.0), (-0.0, -0.0)]))
    # Hole: ring with a hole, a polygon filling the hole, an edge neighbor and a corner neighbor.
    big = Polygon(box(10, 10, 13, 13).exterior.coords, [box(11, 11, 12, 12).exterior.coords])
    geoms.append(big)  # 2
    geoms.append(box(11, 11, 12, 12))  # 3 fills the hole
    geoms.append(box(13, 10, 14, 13))  # 4 shares an edge with the big ring
    geoms.append(box(13, 13, 14, 14))  # 5 corner only
    # T-junction: the long edge of 6 spans the edges of 7 and 8.
    geoms.append(box(20, 0, 22, 1))  # 6
    geoms.append(box(20, 1, 21, 2))  # 7
    geoms.append(box(21, 1, 22, 2))  # 8
    # Multipolygon whose parts touch the same neighbor on both sides.
    geoms.append(MultiPolygon([box(30, 0, 31, 1), box(32, 0, 33, 1)]))  # 9
    geoms.append(box(31, 0, 32, 1))  # 10
    geoms.append(box(33, 0, 34, 1))  # 11 touches only the second part
    # Multipolygon with a holed part and an island.
    geoms.append(
        MultiPolygon(
            [
                Polygon(box(40, 0, 43, 3).exterior.coords, [box(41, 1, 42, 2).exterior.coords]),
                box(50, 0, 51, 1),
            ]
        )
    )  # 12
    geoms.append(box(41, 1, 42, 2))  # 13 fills the hole of 12
    geoms.append(box(100, 100, 101, 101))  # 14 island
    assert b"\x00\x00\x00\x00\x00\x00\x00\x80" in shapely.to_wkb(geoms[1]), "no -0.0 in WKB"
    return contiguity_fixture(
        "contiguity_synthetic",
        "Synthetic polygons: -0.0 vs 0.0 vertices, holes, T-junction, multipolygons, island.",
        geoms,
    )


def islands():
    return contiguity_fixture(
        "contiguity_islands",
        "Three pairwise disjoint squares: every node is an island under queen and rook.",
        [box(0, 0, 1, 1), box(5, 5, 6, 6), box(9, 0, 10, 1)],
    )


def labeled():
    df = gpd.read_file(examples.get_path("columbus.shp")).iloc[:12]
    return contiguity_fixture(
        "contiguity_labeled",
        "First 12 columbus polygons with string ids (ids option / first-appearance order).",
        df.geometry,
        ids=[f"tract-{i:02d}" for i in range(12)],
    )


def block_fixture(name, description, regimes, ids=None):
    series = gpd.pd.Series(list(regimes), index=ids)
    g = Graph.build_block_contiguity(series)
    adj = g.adjacency.reset_index()
    neighbors = {}
    for i in series.index:
        neighbors[str(i)] = sorted(
            (str(n) for f, n, w in zip(adj["focal"], adj["neighbor"], adj["weight"]) if f == i and w != 0),
            key=str,
        )
    return {
        "name": name,
        "description": description,
        "meta": meta(),
        "ids": [str(i) for i in series.index],
        "regimes": [str(r) for r in series],
        "pysal_unique_ids": [str(i) for i in g.unique_ids],
        "neighbors": neighbors,
        "isolates": [str(i) for i in g.isolates],
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, shp in [("columbus", "columbus.shp"), ("sids2", "sids2.shp"), ("mexico", "mexicojoin.shp")]:
        dump_json(OUT / f"contiguity_{name}.json", dataset(name, shp))
    for fx in (synthetic_mix(), islands(), labeled()):
        dump_json(OUT / f"{fx['name']}.json", fx)

    cols = gpd.read_file(examples.get_path("columbus.shp"))
    dump_json(
        OUT / "block_contiguity.json",
        {
            "meta": meta(),
            "cases": [
                block_fixture("abacb", "Interleaved regimes a,b,a,c,b; c is an island.", list("abacb")),
                block_fixture(
                    "columbus_nsa",
                    "Columbus NSA (north-south indicator) regimes.",
                    cols["NSA"].astype(int).tolist(),
                ),
                block_fixture(
                    "labeled",
                    "String ids with integer regimes.",
                    [1, 2, 1, 1, 3, 2],
                    ids=["u", "v", "w", "x", "y", "z"],
                ),
            ],
        },
    )


if __name__ == "__main__":
    main()
