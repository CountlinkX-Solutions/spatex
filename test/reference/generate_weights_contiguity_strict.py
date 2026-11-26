"""Generate libpysal.graph strict / perimeter / line contiguity oracle fixtures.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_contiguity_strict.py

Writes fixtures/weights/contiguity_strict.json. Each geometry set is stored as WKB hex; every
case records ``build_contiguity(geoms, rook=, strict=, by_perimeter=)`` as positional adjacency
(weights are shared boundary lengths with ``by_perimeter``) or the error message.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import LineString, MultiLineString, Polygon, box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    return [wkb.loads(h, hex=True) for h in d["wkb_hex"]]


def synthetic():
    """T-junction, corner contact, a hole filled by an island, a sliver and an overlap-free row."""
    return [
        box(0, 0, 2, 1),
        box(0, 1, 1, 2),
        box(1, 1, 2, 2),
        box(2, 2, 3, 3),  # corner contact with 2
        Polygon([(4, 0), (8, 0), (8, 4), (4, 4)], [[(5, 1), (7, 1), (7, 3), (5, 3)]]),
        box(5, 1, 7, 3),  # fills the hole
        box(8, 0, 9, 2),  # shares half of 4's right edge (T-junction on a long edge)
        box(10, 10, 11, 11),  # island
    ]


def overlapping():
    return [box(0, 0, 2, 2), box(1, 1, 3, 3), box(3, 0, 4, 1)]


def lines():
    return [
        LineString([(0, 0), (1, 0), (2, 0)]),
        LineString([(2, 0), (2, 1)]),
        LineString([(1, 0), (2, 0)]),
        MultiLineString([[(5, 5), (6, 6)], [(2, 1), (3, 1)]]),
        LineString([(9, 9), (10, 10)]),
    ]


def case(geoms, rook, strict, by_perimeter):
    params = {"rook": rook, "strict": strict, "by_perimeter": by_perimeter}
    try:
        g = Graph.build_contiguity(gpd.GeoSeries(geoms), rook=rook, strict=strict,
                                   by_perimeter=by_perimeter)
    except Exception as exc:  # noqa: BLE001 - the message is the fixture
        return {"params": params, "error": f"{type(exc).__name__}: {exc}"}
    ids = list(g.unique_ids)
    out = {"params": params, "cardinalities": g.cardinalities.reindex(ids).to_numpy(int),
           "adjacency": export_graph(g)}
    del out["adjacency"]["ids"]
    return out


def main():
    sets = {"columbus": columbus(), "synthetic": synthetic(), "overlapping": overlapping(),
            "lines": lines()}
    fixtures = []
    for name, geoms in sets.items():
        cases = []
        for rook in (True, False):
            for strict in (False, True):
                for by_perimeter in (False, True):
                    if name == "lines" and (strict or by_perimeter):
                        continue
                    cases.append(case(geoms, rook, strict, by_perimeter))
        fixtures.append({"set": name, "wkb_hex": [wkb.dumps(g, hex=True) for g in geoms],
                         "cases": cases})
    dump_json(FIXTURES / "weights" / "contiguity_strict.json", {
        "name": "contiguity_strict", "description": "strict, by_perimeter and line contiguity",
        "meta": meta(), "sets": fixtures,
    })


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
