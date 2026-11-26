"""Generate libpysal.graph fuzzy contiguity oracle fixtures.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_fuzzy_contiguity.py

Writes fixtures/weights/contiguity_fuzzy.json. Every set stores its geometries as WKB hex and a
list of cases ``Graph.build_fuzzy_contiguity(geoms, tolerance=, buffer=, predicate=)``, each as
the directed positional neighbor pairs (self-loops of islands removed) and the isolate
positions, or the error message. The ``near_threshold`` set is made of independent pairs of
polygons whose distance is ``ratio * 2 * buffer``; it records, per pair, whether libpysal (GEOS
round buffers of 16 segments per quadrant) found the buffers intersecting.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from libpysal import examples  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely.geometry import MultiPolygon, Point, Polygon, box  # noqa: E402
from shapely.affinity import translate  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

PREDICATES = [
    "intersects", "touches", "overlaps", "within", "contains", "covers", "covered_by",
    "contains_properly", "crosses", None,
]


def run(geoms, **params):
    try:
        g = Graph.build_fuzzy_contiguity(gpd.GeoSeries(list(geoms)), **params)
    except Exception as exc:  # noqa: BLE001 - the message is the fixture
        return {"params": params, "error": f"{type(exc).__name__}: {exc}"}
    ids = list(g.unique_ids)
    assert ids == list(range(len(geoms))), "libpysal reordered ids"
    adj = g.adjacency.reset_index()
    pairs = [[int(f), int(n)] for f, n, w in zip(adj["focal"], adj["neighbor"], adj["weight"])
             if f != n or w != 0]
    pairs = [p for p in pairs if p[0] != p[1]]
    return {"params": params, "pairs": pairs, "isolates": [int(i) for i in g.isolates]}


def synthetic():
    parts = []

    def at(offset, geoms):
        parts.extend(translate(g, xoff=offset) for g in geoms)

    # Cluster 0: edge, corner, overlap, nesting, equality, a gap and an island.
    at(0, [
        box(0, 0, 4, 4),        # 0
        box(4, 0, 6, 4),        # 1 shares an edge with 0
        box(6, 4, 8, 6),        # 2 corner contact with 1
        box(3, 3, 7, 7),        # 3 overlaps 0, 1, 2
        box(1, 1, 2, 2),        # 4 strictly inside 0
        box(0, 0, 4, 4),        # 5 equal to 0
        box(6.5, 0, 8, 2),      # 6 gap 0.5 to 1
        box(0, 1, 2, 3),        # 7 inside 0, sharing the left edge
    ])
    # Cluster 1: a holed polygon, a polygon in the hole (with gaps) and one touching the hole.
    ring = Polygon(box(0, 0, 10, 10).exterior.coords, [box(3, 3, 7, 7).exterior.coords])
    at(100, [ring, box(4, 4, 6, 6), box(3, 3, 5, 5), box(3.8, 3.8, 6.2, 6.2)])  # 8 9 10 11
    # Cluster 2: multipolygons; one part touches, one is apart.
    at(200, [
        MultiPolygon([box(0, 0, 1, 1), box(5, 0, 6, 1)]),   # 12
        box(1, 0, 2, 1),                                    # 13 touches part one
        box(3, 0, 4, 1),                                    # 14 gap 1 to 13 and 1 to part two
        MultiPolygon([box(8, 0, 9, 1), box(8, 2, 9, 3)]),   # 15 gap 2 to 12's second part
    ])
    # Cluster 3: a polygon nested in a polygon nested in a polygon, plus an island.
    at(300, [box(0, 0, 9, 9), box(1, 1, 8, 8), box(2, 2, 7, 7), box(500, 500, 501, 501)])
    return parts


def near_threshold():
    ratios = [0.99, 0.995, 0.997, 0.998, 0.999, 0.9995, 0.9999, 1.0001, 1.0005, 1.001, 1.002,
              1.003, 1.005, 1.01]
    geoms, meta_rows = [], []
    row = 0

    def add(a, b, kind, angle, ratio):
        nonlocal row
        dy = 1000.0 * row
        geoms.extend([translate(a, yoff=dy), translate(b, yoff=dy)])
        meta_rows.append({"kind": kind, "angle": angle, "ratio": ratio})
        row += 1

    unit = box(0, 0, 1, 1)
    for ratio in ratios + [1.0]:
        d = ratio * 2.0
        add(unit, translate(unit, xoff=1 + d), "edge", 0.0, ratio)
    for angle in (2.8125, 11.25, 45.0, 84.0):
        for ratio in ratios:
            d = ratio * 2.0
            dx, dy = d * math.cos(math.radians(angle)), d * math.sin(math.radians(angle))
            add(unit, translate(unit, xoff=1 + dx, yoff=1 + dy), "corner", angle, ratio)
    tri = Polygon([(0, 0), (1, 0), (0.5, 1)])
    for ratio in ratios:
        d = ratio * 2.0   # tip of a triangle against the flat edge of a square
        add(tri, box(-1, 1 + d, 2, 3 + d), "tip", 0.0, ratio)
    disc = Point(0, 0).buffer(1.0, 16)
    for angle in (3.0, 40.0):
        for ratio in ratios:
            d = ratio * 2.0   # discs (16 segments per quadrant) with unit radii
            c = 2.0 + d
            b = translate(disc, xoff=c * math.cos(math.radians(angle)),
                          yoff=c * math.sin(math.radians(angle)))
            add(disc, b, "disc", angle, ratio)
    return geoms, meta_rows


def real_cases(geoms):
    series = gpd.GeoSeries(list(geoms))
    minx, miny, maxx, maxy = series.total_bounds
    side = min(maxx - minx, maxy - miny)
    cases = [run(geoms)]
    for tol in (0.005, 0.02, 0.1):
        cases.append(run(geoms, tolerance=tol))
    for frac in (0.001, 0.01):
        cases.append(run(geoms, buffer=frac * side))
    cases.append(run(geoms, buffer=0.0))
    cases.append(run(geoms, tolerance=0.02, predicate=None))
    for p in PREDICATES:
        cases.append(run(geoms, predicate=p))
    cases.append(run(geoms, tolerance=0.1, buffer=1.0))
    return cases


def main():
    sets = []

    def add_set(name, geoms, cases, **extra):
        sets.append({"set": name, "wkb_hex": [g.wkb_hex.lower() for g in geoms], "cases": cases,
                     **extra})

    for name, shp in (("columbus", "columbus.shp"), ("sids2", "sids2.shp")):
        geoms = list(gpd.read_file(examples.get_path(shp)).geometry)
        add_set(name, geoms, real_cases(geoms))

    syn = synthetic()
    cases = []
    for b in (None, 0.0, 0.4, 0.5, 0.6, 1.0, 2.0, 50.0):
        cases.append(run(syn, buffer=b) if b is not None else run(syn))
    for tol in (0.0, 0.001, 0.01):
        cases.append(run(syn, tolerance=tol))
    for p in PREDICATES:
        cases.append(run(syn, predicate=p))
    cases.append(run(syn, buffer=0.7, predicate=None))
    add_set("synthetic", syn, cases)

    geoms, rows = near_threshold()
    case = run(geoms, buffer=1.0)
    pairs = {tuple(p) for p in case["pairs"]}
    for k, r in enumerate(rows):
        r["pysal"] = (2 * k, 2 * k + 1) in pairs
        r["exact"] = r["ratio"] <= 1.0
    cases = [case, run(geoms, buffer=1.0, predicate=None)]
    add_set("near_threshold", geoms, cases, pairs=rows)

    dump_json(FIXTURES / "weights" / "contiguity_fuzzy.json", {
        "name": "contiguity_fuzzy", "description": "libpysal Graph.build_fuzzy_contiguity",
        "meta": meta(), "sets": sets,
    })
    diff = [r for r in rows if r["pysal"] != r["exact"]]
    print("near-threshold pairs:", len(rows), "differences from exact:", len(diff))
    for r in diff:
        print("  ", r)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
