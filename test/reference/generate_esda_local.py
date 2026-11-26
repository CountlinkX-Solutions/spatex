"""Generate esda oracle fixtures for the local Moran family (Moran_Local, Moran_Local_BV).

Run from the repository root (memory-capped):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_esda_local.py

Writes test/reference/fixtures/esda/local_moran_*.json. Every fixture embeds the graph it was
computed on (see common.export_graph).

Per case: ``analytic`` quantities (Is, quadrants, z, lag, EI, VI, EIc, VIc; compared to 1e-9)
and ``perm`` quantities from PySAL with ``permutations=9999`` and ``seed=SEED`` (compared
statistically: numba's RNG stream is not reproduced), plus PySAL's cluster labels at 0.05.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from esda.moran import Moran_Local, Moran_Local_BV  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

PERMS = 9999
SEED = 12345

UNI_ANALYTIC = ["EI", "VI", "EIc", "VIc"]
PERM = ["EI_sim", "seI_sim", "VI_sim", "z_sim", "p_sim", "p_z_sim"]


def arrays(obj, names):
    return {n.lower(): np.asarray(getattr(obj, n), dtype=float) for n in names}


def case(label, lm, w, transform, geoda, alt, comparable=True, **inputs):
    lag = np.asarray(w.sparse @ (lm.zy if hasattr(lm, "zy") else lm.z), dtype=float)
    analytic = {
        "is": np.asarray(lm.Is, float),
        "q": np.asarray(lm.q, int),
        "z": np.asarray(lm.zx if hasattr(lm, "zx") else lm.z, float),
        "lag": lag,
    }
    if hasattr(lm, "EI"):
        analytic.update(arrays(lm, UNI_ANALYTIC))
    return {
        "label": label,
        "transform": transform,
        "geoda_quads": geoda,
        "alternative": alt,
        # False on graphs so small that the reference distribution is a handful of discrete
        # values: p_sim then hinges on floating-point ties (see the test) and is not comparable.
        "perm_comparable": comparable,
        **{k: np.asarray(v, float) for k, v in inputs.items()},
        "analytic": analytic,
        "permutations": PERMS,
        "seed": SEED,
        "perm": arrays(lm, PERM),
        "labels": [str(s) for s in lm.get_cluster_labels(0.05)],
    }


def uni(label, y, w, transform="r", geoda=False, alt="directed", comparable=True):
    y = np.asarray(y, float)
    lm = Moran_Local(y, w, transformation=transform, permutations=PERMS, geoda_quads=geoda,
                     seed=SEED, alternative=alt)
    return case(label, lm, lm.w, transform, geoda, alt, comparable, y=y)


def bv(label, x, y, w, transform="r", geoda=False, alt="directed", comparable=True):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    lm = Moran_Local_BV(x, y, w, transformation=transform, permutations=PERMS,
                        geoda_quads=geoda, seed=SEED, alternative=alt)
    return case(label, lm, lm.w, transform, geoda, alt, comparable, x=x, y=y)


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    return Graph.build_contiguity(geoms, rook=False), attrs


def header(name, description, graph):
    return {"name": name, "description": description, "meta": meta(), "graph": export_graph(graph)}


def main():
    out = FIXTURES / "esda"
    queen, attrs = columbus()

    fx = header("local_moran_columbus_queen", "Moran_Local on columbus queen contiguity.", queen)
    fx["cases"] = [
        uni("CRIME/r", attrs["CRIME"], queen),
        uni("CRIME/b/geoda", attrs["CRIME"], queen, "b", True),
        uni("HOVAL/r", attrs["HOVAL"], queen),
        uni("CRIME/r/greater", attrs["CRIME"], queen, alt="greater"),
        uni("INC/r/lesser", attrs["INC"], queen, alt="lesser"),
    ]
    dump_json(out / "local_moran_columbus_queen.json", fx)

    fx = header("local_moran_bv_columbus", "Moran_Local_BV on columbus queen: x = INC, y = CRIME.", queen)
    fx["cases"] = [
        bv("INC~CRIME/r", attrs["INC"], attrs["CRIME"], queen),
        bv("CRIME~HOVAL/b/geoda", attrs["CRIME"], attrs["HOVAL"], queen, "b", True),
        bv("INC~CRIME/r/greater", attrs["INC"], attrs["CRIME"], queen, alt="greater"),
    ]
    dump_json(out / "local_moran_bv_columbus.json", fx)

    # Path 0-1-2-3-4 plus island 5 (spec probe), and a second graph with two islands.
    path = Graph.from_arrays(
        [0, 1, 1, 2, 2, 3, 3, 4, 5],
        [1, 0, 2, 1, 3, 2, 4, 3, 5],
        [1.0] * 8 + [0.0],
    )
    y6 = [1.0, 5.0, 2.0, 8.0, 3.0, 4.0]
    x6 = [2.0, 7.0, 1.0, 3.0, 9.0, 5.0]
    fx = header("local_moran_islands", "Local Moran with an island node (path 0-4 plus 5).", path)
    fx["cases"] = [
        uni("path+island/r", y6, path, comparable=False),
        uni("path+island/b/geoda", y6, path, "b", True, comparable=False),
        bv("path+island bv/r", x6, y6, path, comparable=False),
    ]
    dump_json(out / "local_moran_islands.json", fx)

    two = Graph.from_arrays(
        [0, 1, 1, 2, 3, 4, 5, 6, 7],
        [1, 0, 2, 1, 3, 4, 5, 6, 7],
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )
    y8 = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    fx = header("local_moran_islands_many", "Path 0-1-2 with five island nodes.", two)
    fx["cases"] = [uni("path+5islands/r", y8, two, comparable=False)]
    dump_json(out / "local_moran_islands_many.json", fx)

    # A 5x5 rook lattice plus two far-away cells (islands): continuous data, enough neighbors
    # for a non-degenerate reference distribution, islands still present.
    cells = [box(c, r, c + 1, r + 1) for r in range(5) for c in range(5)]
    cells += [box(20, 20, 21, 21), box(30, 30, 31, 31)]
    lattice = Graph.build_contiguity(gpd.GeoSeries(cells), rook=True)
    rng = np.random.default_rng(3)
    yl = rng.normal(size=27).round(6)
    xl = (0.5 * yl + rng.normal(size=27)).round(6)
    fx = header("local_moran_lattice_islands", "5x5 rook lattice plus two islands.", lattice)
    fx["cases"] = [
        uni("lattice+2islands/r", yl, lattice),
        uni("lattice+2islands/b/geoda", yl, lattice, "b", True),
        bv("lattice+2islands bv/r", xl, yl, lattice),
    ]
    dump_json(out / "local_moran_lattice_islands.json", fx)

    # Documented PySAL behavior that Spatex does not replicate: a constant y gives NaN
    # statistics (Spatex raises ArgumentError instead; see known_deviations.exs).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        const = Moran_Local(np.full(49, 3.0), queen, permutations=99, seed=SEED,
                            alternative="directed")
    fx = header("local_moran_constant", "PySAL Moran_Local on a constant y: NaN statistics.", queen)
    fx["pysal"] = {"is": np.asarray(const.Is, float), "z": np.asarray(const.z, float)}
    fx["y"] = np.full(49, 3.0)
    dump_json(out / "local_moran_constant.json", fx)


if __name__ == "__main__":
    main()
