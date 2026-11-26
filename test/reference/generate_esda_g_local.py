"""Generate esda oracle fixtures for the local Getis-Ord statistics (G_Local, G*).

Run from the repository root (memory-capped):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_esda_g_local.py

Writes test/reference/fixtures/esda/g_local_*.json. Every fixture embeds the graph it was
computed on (see common.export_graph) in its ORIGINAL (untransformed) form; ``input_transform``
names the transformation applied to it before it is handed to ``G_Local``, because esda's
``star=True`` rule depends on it.

Per case: ``analytic`` quantities (Gs, EGs, VGs, Zs, p_norm; compared to 1e-9) and ``perm``
quantities from PySAL with ``permutations=9999`` and ``seed=SEED`` (compared statistically:
numba's RNG stream is not reproduced).
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from esda.getisord import G_Local  # noqa: E402
from esda.util import fdr  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

PERMS = 9999
SEED = 12345

ANALYTIC = ["Gs", "EGs", "VGs", "Zs", "p_norm"]
PERM = ["p_sim", "EG_sim", "seG_sim", "z_sim", "p_z_sim"]


def arrays(obj, names):
    return {n.lower(): np.asarray(getattr(obj, n), dtype=float) for n in names}


def case(label, y, w, input_transform="o", transform="r", star=False, alt="directed",
         comparable=True, island_weight=0.0):
    """One G_Local run. ``star``: False | True | None | float (esda's own encoding)."""
    y = np.asarray(y, float)
    g = w if input_transform == "o" else w.transform(input_transform)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the star=True / row-standardized UserWarning
        lg = G_Local(y, g, transform=transform, permutations=PERMS, star=star, seed=SEED,
                     alternative=alt, island_weight=island_weight, n_jobs=1)
    return {
        "label": label,
        "y": y,
        "input_transform": input_transform,
        "transform": transform,
        "star_arg": "infer" if star is None else star,
        "star": bool(lg.star),
        "alternative": alt,
        "island_weight": island_weight,
        "perm_comparable": comparable,
        "analytic": arrays(lg, ANALYTIC),
        "permutations": PERMS,
        "seed": SEED,
        "perm": arrays(lg, PERM),
    }


def header(name, description, graph):
    return {"name": name, "description": description, "meta": meta(), "graph": export_graph(graph)}


def main():
    out = FIXTURES / "esda"
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    queen = Graph.build_contiguity(geoms, rook=False)
    pts = np.column_stack([geoms.centroid.x.to_numpy(), geoms.centroid.y.to_numpy()])
    band = Graph.build_distance_band(pts, threshold=1.0, binary=True)
    assert len(band.isolates) == 0

    crime = attrs["CRIME"]
    fx = header("g_local_columbus_queen", "G_Local on columbus queen contiguity.", queen)
    fx["cases"] = [
        case("CRIME/r/star=False", crime, queen),
        case("CRIME/r/star=True", crime, queen, star=True),
        case("CRIME/r/star=0.5", crime, queen, star=0.5),
        case("CRIME/b/star=False", crime, queen, transform="b"),
        case("CRIME/b/star=True", crime, queen, transform="b", star=True),
        case("HOVAL/r/star=False", attrs["HOVAL"], queen),
        case("INC/r/star=True/greater", attrs["INC"], queen, star=True, alt="greater"),
        case("CRIME/r/star=False/lesser", crime, queen, alt="lesser"),
        case("CRIME/r/star=False/two_sided", crime, queen, alt="two-sided"),
        case("CRIME/r/star=True/folded", crime, queen, star=True, alt="folded"),
        # esda assigns self weight 1 when the input graph is binary, whatever the transform.
        case("CRIME/b-input/r/star=True", crime, queen, input_transform="b", star=True),
        # Row-standardized input: the self weight is the (smaller) row maximum.
        case("CRIME/r-input/r/star=True", crime, queen, input_transform="r", star=True),
        case("CRIME/r-input/b/star=True", crime, queen, input_transform="r", transform="b",
             star=True),
    ]
    dump_json(out / "g_local_columbus_queen.json", fx)

    fx = header("g_local_columbus_band", "G_Local on a binary distance band (1.0) of columbus centroids.", band)
    fx["cases"] = [
        case("CRIME/r/star=False", crime, band),
        case("CRIME/r/star=True", crime, band, star=True),
        case("CRIME/b/star=True", crime, band, transform="b", star=True),
        case("HOVAL/b/star=False", attrs["HOVAL"], band, transform="b"),
        case("INC/r/star=0.5", attrs["INC"], band, star=0.5),
    ]
    dump_json(out / "g_local_columbus_band.json", fx)

    # Self-weighted graph: star=None infers G* from the nonzero diagonal; False strips it.
    selfw = queen.assign_self_weight(1)
    fx = header("g_local_columbus_selfweight", "G_Local on queen with self weights 1.", selfw)
    fx["cases"] = [
        case("CRIME/r/star=None", crime, selfw, star=None),
        case("CRIME/r/star=False", crime, selfw, star=False),
        case("CRIME/r/star=True", crime, selfw, star=True),
        case("CRIME/b/star=None", crime, selfw, transform="b", star=None),
    ]
    dump_json(out / "g_local_columbus_selfweight.json", fx)

    # Path 0-4 plus island 5 (positive y), and a 5x5 rook lattice plus two islands.
    path = Graph.from_arrays(
        [0, 1, 1, 2, 2, 3, 3, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5],
        [1.0] * 8 + [0.0],
    )
    y6 = [1.0, 5.0, 2.0, 8.0, 3.0, 4.0]
    fx = header("g_local_islands", "G_Local with an island node (path 0-4 plus 5).", path)
    fx["cases"] = [
        case("path+island/r/star=False", y6, path, comparable=False),
        case("path+island/r/star=True", y6, path, star=True, comparable=False),
        case("path+island/b/star=True", y6, path, transform="b", star=True, comparable=False),
    ]
    dump_json(out / "g_local_islands.json", fx)

    cells = [box(c, r, c + 1, r + 1) for r in range(5) for c in range(5)]
    cells += [box(20, 20, 21, 21), box(30, 30, 31, 31)]
    lattice = Graph.build_contiguity(gpd.GeoSeries(cells), rook=True)
    rng = np.random.default_rng(3)
    yl = (rng.gamma(3.0, 2.0, size=27)).round(6)
    fx = header("g_local_lattice_islands", "5x5 rook lattice plus two islands.", lattice)
    fx["cases"] = [
        case("lattice+2islands/r/star=False", yl, lattice),
        case("lattice+2islands/r/star=True", yl, lattice, star=True),
        case("lattice+2islands/b/star=False", yl, lattice, transform="b"),
        case("lattice+2islands/r/star=False/island_weight=1", yl, lattice, island_weight=1.0),
    ]
    dump_json(out / "g_local_lattice_islands.json", fx)

    fdr_cases = {
        "typical": [0.001, 0.01, 0.02, 0.04, 0.3, 0.5],
        "none significant": [0.5, 0.7, 0.9, 0.99],
        "all significant": [0.001, 0.002, 0.003, 0.004],
        "ties": [0.01, 0.01, 0.9, 0.9, 0.03, 0.03],
        "single": [0.01],
        "above one": [0.0001, 1.3, 0.02],
        "unsorted": [0.3, 0.001, 0.5, 0.04, 0.01, 0.02],
    }
    lm = G_Local(crime, queen, seed=SEED, permutations=999, n_jobs=1)
    fdr_cases["columbus p_sim"] = np.asarray(lm.p_sim, float).tolist()
    fdr_cases["columbus p_norm"] = np.asarray(lm.p_norm, float).tolist()
    rows = []
    for label, p in fdr_cases.items():
        for alpha in (0.05, 0.1, 0.01):
            rows.append({"label": label, "p": p, "alpha": alpha, "cutoff": float(fdr(np.asarray(p), alpha))})
    dump_json(out / "fdr.json", {"name": "fdr", "meta": meta(), "cases": rows})


if __name__ == "__main__":
    main()
