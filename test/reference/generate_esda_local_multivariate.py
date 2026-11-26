"""Generate esda oracle fixtures for Geary_Local_MV, Join_Counts_Local_BV/MV and island_weight.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_esda_local_multivariate.py

Writes fixtures/esda/local_multivariate_columbus.json and local_multivariate_islands.json.

Observed statistics are deterministic. ``p_sim`` comes from numba / numpy draws that Spatex does
not replicate, so it is compared statistically. Geary_Local_MV draws from numpy's unseeded global
generator; it is seeded here with ``np.random.seed`` for a reproducible fixture.

esda quirks recorded by the cases:
* Geary_Local_MV on a Graph with islands gives them ``localG = 0`` with either ``drop_islands``.
* Join_Counts_Local_BV ``case="BJC"`` observes ``x_i (1 - z_i) sum_j (1 - x_j) z_j`` but simulates
  ``x_i sum_j z_j``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from esda.geary_local import Geary_Local  # noqa: E402
from esda.geary_local_mv import Geary_Local_MV  # noqa: E402
from esda.join_counts_local_bv import Join_Counts_Local_BV  # noqa: E402
from esda.join_counts_local_mv import Join_Counts_Local_MV  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

PERMS = 999
SEED = 20240601


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    return Graph.build_contiguity(geoms, rook=False), attrs


def islands_graph():
    cells = [box(c, r, c + 1, r + 1) for r in range(5) for c in range(5)]
    cells += [box(10, 10, 11, 11), box(20, 20, 21, 21)]
    return Graph.build_contiguity(gpd.GeoSeries(cells), rook=True)


def floats(a):
    return [None if np.isnan(v) else float(v) for v in np.asarray(a, dtype=float)]


def geary_mv(label, variables, g, drop_islands=True):
    np.random.seed(SEED % 2**32)
    m = Geary_Local_MV(connectivity=g, permutations=PERMS, drop_islands=drop_islands).fit(variables)
    return {"label": label, "variables": [floats(v) for v in variables],
            "drop_islands": drop_islands, "permutations": PERMS,
            "local_g": floats(m.localG), "p_sim": floats(m.p_sim)}


def joins_bv(label, x, z, g, case):
    m = Join_Counts_Local_BV(connectivity=g, seed=SEED, alternative="directed").fit(
        x, z, case=case, permutations=PERMS)
    return {"label": label, "x": floats(x), "z": floats(z), "case": case,
            "alternative": "directed", "permutations": PERMS,
            "ljc": floats(m.LJC), "p_sim": floats(m.p_sim)}


def joins_mv(label, variables, g):
    m = Join_Counts_Local_MV(connectivity=g, seed=SEED, alternative="directed").fit(
        variables, permutations=PERMS)
    return {"label": label, "variables": [floats(v) for v in variables],
            "alternative": "directed", "permutations": PERMS,
            "ljc": floats(m.LJC), "p_sim": floats(m.p_sim)}


def main():
    out = FIXTURES / "esda"
    queen, a = columbus()
    crime, hoval, inc = a["CRIME"], a["HOVAL"], a["INC"]
    x = (crime > crime.mean()).astype(float)
    z = (hoval < hoval.mean()).astype(float)
    u = (inc > np.percentile(inc, 25)).astype(float)
    fx = {"name": "local_multivariate_columbus",
          "description": "Geary_Local_MV and Join_Counts_Local_BV/MV on columbus queen contiguity.",
          "meta": meta(), "graph": export_graph(queen),
          "geary_mv": [geary_mv("crime_hoval", [crime, hoval], queen),
                       geary_mv("crime_hoval_inc", [crime, hoval, inc], queen)],
          "joins_bv": [joins_bv("clc", x, z, queen, "CLC"), joins_bv("bjc", x, z, queen, "BJC")],
          "joins_mv": [joins_mv("three", [x, z, u], queen), joins_mv("two", [x, z], queen)]}
    dump_json(out / "local_multivariate_columbus.json", fx)

    g = islands_graph()
    rng = np.random.default_rng(58)
    v1, v2 = rng.normal(size=27), rng.normal(size=27)
    gl = Geary_Local(connectivity=g, permutations=PERMS, seed=SEED, island_weight=1.0).fit(v1)
    fx = {"name": "local_multivariate_islands",
          "description": "5x5 rook lattice plus two islands: Geary_Local_MV (drop_islands=False) "
                         "and island_weight=1 for Geary_Local (Graph). Join_Counts_Local cannot "
                         "fit islands without counting their self pairs, so it is not recorded.",
          "meta": meta(), "graph": export_graph(g),
          "geary_mv": [geary_mv("two_vars_kept_islands", [v1, v2], g, drop_islands=False)],
          "geary_mv_dropped": {"local_g": floats(Geary_Local_MV(
              connectivity=g, permutations=0).fit([v1, v2]).localG)},
          "island_weight": {"x": floats(v1), "island_weight": 1.0, "permutations": PERMS,
                            "geary_p_sim": floats(gl.p_sim)}}
    dump_json(out / "local_multivariate_islands.json", fx)


if __name__ == "__main__":
    main()
