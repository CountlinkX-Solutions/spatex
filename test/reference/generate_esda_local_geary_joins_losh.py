"""Generate esda oracle fixtures for Geary_Local, Join_Counts_Local and LOSH.

Run from the repository root (memory-capped):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_esda_local_geary_joins_losh.py

Writes test/reference/fixtures/esda/{local_geary,local_joins,losh}_*.json. Every fixture embeds
the graph it was computed on (see common.export_graph).

Analytic quantities (localG, LJC, Hi, ylag, yresid, VarHi, pval) are compared to 1e-9;
permutation quantities come from PySAL with ``permutations=9999`` and ``seed=SEED`` and are
compared statistically (numba's RNG stream is not reproduced).

PySAL bugs that Spatex deliberately does not replicate are probed here and stored under
``pysal`` keys so tests can document the behavior (see known_deviations.exs):

* Geary_Local with islands: ``localG`` has n - #islands entries while ``p_sim`` has n.
* Join_Counts_Local on a Graph with islands: IndexError; ``fit`` ignores the constructor's
  ``permutations``.
* LOSH with any island: every Hi is NaN.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from esda.geary_local import Geary_Local  # noqa: E402
from esda.join_counts_local import Join_Counts_Local  # noqa: E402
from esda.losh import LOSH  # noqa: E402
from libpysal import weights  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

PERMS = 9999
SEED = 12345


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    return Graph.build_contiguity(geoms, rook=False), attrs


def header(name, description, graph):
    return {"name": name, "description": description, "meta": meta(), "graph": export_graph(graph)}


def lattice_graph(extra_cells=()):
    cells = [box(c, r, c + 1, r + 1) for r in range(5) for c in range(5)]
    cells += list(extra_cells)
    return Graph.build_contiguity(gpd.GeoSeries(cells), rook=True)


def geary_case(label, x, g, alt="directed", sig=0.05, drop_islands=True):
    x = np.asarray(x, float)
    lg = Geary_Local(connectivity=g, labels=True, sig=sig, permutations=PERMS, seed=SEED,
                     alternative=alt, drop_islands=drop_islands).fit(x)
    return {
        "label": label,
        "x": x,
        "alternative": alt,
        "sig": sig,
        "permutations": PERMS,
        "seed": SEED,
        "local_g": np.asarray(lg.localG, float),
        "p_sim": np.asarray(lg.p_sim, float),
        "labs": np.asarray(lg.labs, float),
    }


def losh_case(label, y, g, a=2, inference="chi-square"):
    y = np.asarray(y, float)
    ls = LOSH(connectivity=g, inference=inference).fit(y, a=a)
    out = {
        "label": label,
        "y": y,
        "a": a,
        "inference": inference,
        "hi": np.asarray(ls.Hi, float),
        "ylag": np.asarray(ls.ylag, float),
        "yresid": np.asarray(ls.yresid, float),
        "var_hi": np.asarray(ls.VarHi, float),
    }
    if hasattr(ls, "pval"):
        out["pval"] = np.asarray(ls.pval, float)
    return out


def main():
    out = FIXTURES / "esda"
    queen, attrs = columbus()
    rng = np.random.default_rng(11)

    # ---- Local Geary ------------------------------------------------------------------
    fx = header("local_geary_columbus_queen", "Geary_Local on columbus queen contiguity.", queen)
    fx["cases"] = [
        geary_case("CRIME", attrs["CRIME"], queen),
        geary_case("HOVAL/greater", attrs["HOVAL"], queen, alt="greater"),
        geary_case("INC/lesser/sig0.1", attrs["INC"], queen, alt="lesser", sig=0.1),
    ]
    dump_json(out / "local_geary_columbus_queen.json", fx)

    lat = lattice_graph()
    yl = rng.normal(size=25).round(6)
    fx = header("local_geary_lattice", "Geary_Local on a 5x5 rook lattice.", lat)
    fx["cases"] = [geary_case("lattice", yl, lat), geary_case("lattice/two_sided", yl, lat, "two-sided")]
    dump_json(out / "local_geary_lattice.json", fx)

    # Lattice plus two islands. PySAL's default (drop_islands=True) misaligns localG and
    # p_sim; with drop_islands=False the island's localG is 0 and p_sim comes from crand.
    latis = lattice_graph([box(20, 20, 21, 21), box(30, 30, 31, 31)])
    yi = rng.normal(size=27).round(6)
    fx = header("local_geary_islands", "Geary_Local on a lattice with two islands.", latis)
    fx["cases"] = [geary_case("lattice+2islands/keep", yi, latis, drop_islands=False)]
    default = Geary_Local(connectivity=latis, permutations=99, seed=SEED).fit(yi)
    fx["pysal_default"] = {
        "x": yi,
        "local_g_length": len(default.localG),
        "p_sim_length": len(default.p_sim),
    }
    dump_json(out / "local_geary_islands.json", fx)

    # ---- Local join counts --------------------------------------------------------------
    crime = attrs["CRIME"]
    yb = (crime > crime.mean()).astype(float)
    pys = Join_Counts_Local(connectivity=queen, seed=SEED, alternative="directed").fit(
        yb, permutations=PERMS)
    fx = header("local_joins_columbus_queen", "Join_Counts_Local on columbus: CRIME > mean.", queen)
    fx["cases"] = [{
        "label": "CRIME>mean",
        "y": yb,
        "alternative": "directed",
        "permutations": PERMS,
        "seed": SEED,
        "ljc": np.asarray(pys.LJC, float),
        "p_sim": np.asarray(pys.p_sim, float),
    }]
    dump_json(out / "local_joins_columbus_queen.json", fx)

    yl_b = (yl > np.median(yl)).astype(float)
    pys = Join_Counts_Local(connectivity=lat, seed=SEED, alternative="two-sided").fit(
        yl_b, permutations=PERMS)
    fx = header("local_joins_lattice", "Join_Counts_Local on a 5x5 rook lattice.", lat)
    fx["cases"] = [{
        "label": "lattice/two_sided",
        "y": yl_b,
        "alternative": "two-sided",
        "permutations": PERMS,
        "seed": SEED,
        "ljc": np.asarray(pys.LJC, float),
        "p_sim": np.asarray(pys.p_sim, float),
    }]
    dump_json(out / "local_joins_lattice.json", fx)

    # Islands: PySAL crashes on a Graph. The W equivalent (island as an empty neighbor list)
    # with drop_islands=False computes the answer Spatex returns.
    neighbors = {}
    for r in range(5):
        for c in range(5):
            i = r * 5 + c
            nb = []
            if c > 0: nb.append(i - 1)
            if c < 4: nb.append(i + 1)
            if r > 0: nb.append(i - 5)
            if r < 4: nb.append(i + 5)
            neighbors[i] = nb
    neighbors[25] = []
    neighbors[26] = []
    wis = weights.W(neighbors)
    yis = np.concatenate([yl_b, [0.0, 0.0]])
    pys = Join_Counts_Local(connectivity=wis, seed=SEED, alternative="directed",
                            drop_islands=False).fit(yis, permutations=PERMS)
    # to_adjlist keeps an island as a (self, self) pair and LJC ignores weights, so an island
    # with y = 1 counts itself as a joined neighbor (LJC 1). Spatex gives 0.
    yself = np.concatenate([yl_b, [1.0, 1.0]])
    selfpair = Join_Counts_Local(connectivity=wis, seed=SEED, drop_islands=False).fit(
        yself, permutations=99)
    crash = None
    try:
        Join_Counts_Local(connectivity=latis, seed=SEED).fit(yis, permutations=99)
    except Exception as e:  # noqa: BLE001 - the probe records whatever PySAL raises
        crash = type(e).__name__
    fx = header("local_joins_islands", "Join_Counts_Local on a lattice with two islands (W).", latis)
    fx["cases"] = [{
        "label": "lattice+2islands",
        "y": yis,
        "alternative": "directed",
        "permutations": PERMS,
        "seed": SEED,
        "ljc": np.asarray(pys.LJC, float),
        "p_sim": np.asarray(pys.p_sim, float),
    }]
    fx["pysal_graph_crash"] = crash
    fx["pysal_island_selfpair"] = {"y": yself, "ljc": np.asarray(selfpair.LJC, float)}
    # fit() ignores the constructor's permutations (uses its own default 999).
    p = Join_Counts_Local(connectivity=lat, seed=SEED, permutations=49).fit(yl_b)
    fx["pysal_constructor_permutations"] = {"constructor": 49, "permutations_used": len(p.rjoins[0])}
    dump_json(out / "local_joins_islands.json", fx)

    # ---- LOSH ---------------------------------------------------------------------------
    fx = header("losh_columbus_queen", "LOSH on columbus queen (O weights; row-scaling invariant).", queen)
    fx["cases"] = [
        losh_case("CRIME/a2/chi2", crime, queen),
        losh_case("CRIME/a1/chi2", crime, queen, a=1),
        losh_case("HOVAL/a2/none", attrs["HOVAL"], queen, inference=None),
        losh_case("CRIME/a2/chi2/R", crime, queen.transform("r")),
    ]
    fx["pysal_a1_has_pval"] = False
    dump_json(out / "losh_columbus_queen.json", fx)

    fx = header("losh_lattice", "LOSH on a 5x5 rook lattice.", lat)
    fx["cases"] = [losh_case("lattice/a2/chi2", yl, lat)]
    dump_json(out / "losh_lattice.json", fx)

    # Islands: PySAL gives all-NaN Hi. Spatex computes LOSH over the non-islands, which equals
    # PySAL on the island-free subgraph (path 0-1-2-3-4) for those sites.
    path = Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4], [1, 0, 2, 1, 3, 2, 4, 3], [1.0] * 8)
    y5 = np.array([1.0, 5.0, 2.0, 8.0, 3.0])
    y6 = np.append(y5, 4.0)
    path6 = Graph.from_arrays([0, 1, 1, 2, 2, 3, 3, 4, 5], [1, 0, 2, 1, 3, 2, 4, 3, 5],
                              [1.0] * 8 + [0.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        isl = LOSH(connectivity=path6, inference="chi-square").fit(y6)
    fx = header("losh_islands", "LOSH with an island: PySAL all-NaN vs the island-free subgraph.", path6)
    fx["y"] = y6
    fx["pysal_islands"] = {"hi": np.asarray(isl.Hi, float), "pval": np.asarray(isl.pval, float)}
    fx["subgraph_case"] = losh_case("path0-4", y5, path)
    dump_json(out / "losh_islands.json", fx)


if __name__ == "__main__":
    main()
