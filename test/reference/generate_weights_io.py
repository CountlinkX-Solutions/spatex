"""Generate GAL/GWT interop fixtures from libpysal.graph.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_io.py

Reads fixtures/datasets/columbus.json; writes ``fixtures/io/<name>.gal``, ``<name>.gwt`` (the
files written by ``Graph.to_gal`` / ``Graph.to_gwt``) and ``fixtures/io/<name>.json``.

The JSON holds the source graph (``ids`` plus positional ``focal``/``neighbor``/``weight``,
``Graph.adjacency`` order, islands as zero-weight self-loops) and ``gal_roundtrip`` /
``gwt_roundtrip``: the adjacency libpysal itself reads back from the files it wrote.
Cases: columbus queen (integer ids), a gaussian kernel graph on columbus centroids (non-binary
weights), and a small graph with string ids and an island.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import geopandas as gpd  # noqa: E402
import shapely  # noqa: E402
from libpysal.graph import Graph, read_gal, read_gwt  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

OUT = FIXTURES / "io"


def adj(graph):
    out = export_graph(graph)
    del out["ids"]
    return out


def case(name, description, graph):
    OUT.mkdir(parents=True, exist_ok=True)
    gal, gwt = OUT / f"{name}.gal", OUT / f"{name}.gwt"
    graph.to_gal(gal)
    graph.to_gwt(gwt)
    back_gal, back_gwt = read_gal(gal), read_gwt(gwt)
    ids = list(graph.unique_ids)
    for g in (back_gal, back_gwt):
        assert sorted(map(str, g.unique_ids)) == sorted(map(str, ids))
    dump_json(
        OUT / f"{name}.json",
        {
            "name": name,
            "description": description,
            "meta": meta(),
            "ids": ids,
            "isolates": [ids.index(i) for i in graph.isolates],
            **adj(graph),
            "gal_roundtrip": {"ids": list(back_gal.unique_ids), **adj(back_gal)},
            "gwt_roundtrip": {"ids": list(back_gwt.unique_ids), **adj(back_gwt)},
        },
    )


def main():
    columbus = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries(shapely.from_wkb([bytes.fromhex(h) for h in columbus["wkb_hex"]]))
    case("columbus_queen", "Queen contiguity of columbus (integer ids, binary weights).",
         Graph.build_contiguity(geoms, rook=False))
    pts = geoms.centroid
    case("columbus_kernel", "Gaussian kernel graph on columbus centroids (non-binary weights).",
         Graph.build_kernel(gpd.GeoDataFrame(geometry=pts), kernel="gaussian"))
    case("labeled_island", "String ids and an island node.",
         Graph.from_arrays(["a", "a", "b", "c", "c", "d"], ["b", "c", "a", "a", "b", "d"],
                           [1.0, 0.5, 0.25, 2.0, 0.125, 0.0]))


if __name__ == "__main__":
    main()
