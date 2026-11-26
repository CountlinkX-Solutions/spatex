"""Generate esda oracle fixtures for the global Moran family (Moran, Moran_BV, Moran_Rate).

Run from the repository root:

    <venv>/bin/python -I test/reference/generate_esda_global.py

Writes test/reference/fixtures/esda/{moran,geary,g,join_counts}_*.json. Every fixture embeds the graph it was
computed on (original weights, positional adjacency, see common.export_graph) so Spatex can
rebuild it with ``Graph.from_arrays`` without any contiguity code.

Per case: ``analytic`` quantities (compared to 1e-9) and ``perm`` quantities from PySAL with
``permutations=9999`` after ``np.random.seed(SEED)`` (compared statistically: numpy's RNG
stream is not reproduced).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
from esda.geary import Geary  # noqa: E402
from esda.getisord import G  # noqa: E402
from esda.join_counts import Join_Counts  # noqa: E402
from esda.moran import Moran, Moran_BV, Moran_Rate  # noqa: E402
from esda.smoothing import assuncao_rate  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, export_graph, meta  # noqa: E402

PERMS = 9999
SEED = 12345

MORAN_ANALYTIC = [
    "I", "EI", "VI_norm", "seI_norm", "z_norm", "p_norm",
    "VI_rand", "seI_rand", "z_rand", "p_rand",
]
MORAN_PERM = ["EI_sim", "seI_sim", "VI_sim", "z_sim", "p_sim", "p_z_sim"]


def pick(obj, names):
    return {n.lower(): float(getattr(obj, n)) for n in names}


def moran_case(label, y, graph, transform, two_tailed):
    y = np.asarray(y, dtype=float)
    analytic = Moran(y, graph, transformation=transform, permutations=0, two_tailed=two_tailed)
    np.random.seed(SEED)
    sim = Moran(y, graph, transformation=transform, permutations=PERMS, two_tailed=two_tailed)
    return {
        "label": label,
        "y": y,
        "transform": transform,
        "two_tailed": two_tailed,
        "analytic": pick(analytic, MORAN_ANALYTIC),
        "permutations": PERMS,
        "perm": pick(sim, MORAN_PERM),
    }


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

    cases = []
    for var in ("CRIME", "HOVAL", "INC"):
        for transform in ("r", "b"):
            cases.append(moran_case(f"{var}/{transform}", attrs[var], queen, transform, True))
    cases.append(moran_case("CRIME/r/one_tailed", attrs["CRIME"], queen, "r", False))
    fx = header("moran_columbus_queen", "Moran on columbus queen contiguity (CRIME, HOVAL, INC).", queen)
    fx["cases"] = cases
    dump_json(out / "moran_columbus_queen.json", fx)

    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(4) for c in range(4)])
    lattice = Graph.build_contiguity(cells, rook=True)
    rng = np.random.default_rng(7)
    checker = np.array([(r + c) % 2 for r in range(4) for c in range(4)], dtype=float)
    ys = {
        "ramp": np.arange(1, 17, dtype=float),
        "checkerboard": checker * 3.0 + 1.0,
        "random": rng.normal(size=16).round(6),
    }
    fx = header("moran_lattice", "Moran on a 4x4 rook lattice.", lattice)
    fx["cases"] = [
        moran_case(f"{k}/{t}", v, lattice, t, True) for k, v in ys.items() for t in ("r", "b")
    ]
    dump_json(out / "moran_lattice.json", fx)

    # Islands: path 0-1-2 plus isolated node 3 (spec probe), and a longer path with two islands.
    path = Graph.from_arrays([0, 1, 1, 2, 3], [1, 0, 2, 1, 3], [1.0, 1.0, 1.0, 1.0, 0.0])
    fx = header("moran_islands", "Moran with island nodes (zero-weight self-loops).", path)
    fx["cases"] = [moran_case("path+island/r", [1, 2, 4, 7], path, "r", True),
                   moran_case("path+island/b", [1, 2, 4, 7], path, "b", True)]
    dump_json(out / "moran_islands.json", fx)

    many = Graph.from_arrays(
        [0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
        [1, 0, 2, 1, 3, 2, 4, 5, 6, 7],
        [1.0] * 6 + [0.0] * 4,
    )
    y8 = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    fx = header("moran_islands_many", "Path 0-1-2-3 with four island nodes (4..7).", many)
    fx["cases"] = [moran_case("path+4islands/r", y8, many, "r", True)]
    dump_json(out / "moran_islands_many.json", fx)

    # Existing asymmetric graph fixture (5 nodes, node 4 an island).
    asym = Graph.from_arrays(
        ["a", "a", "b", "c", "d", "d", "e"],
        ["b", "c", "a", "d", "c", "a", "e"],
        [1.0, 2.0, 3.0, 0.5, 1.5, 2.5, 0.0],
    )
    fx = header("moran_asymmetric", "Moran on the asymmetric 5-node graph with an island.", asym)
    fx["cases"] = [moran_case("asym/r", [1.0, 2.0, 3.0, 4.0, 5.0], asym, "r", True),
                   moran_case("asym/b", [1.0, 2.0, 3.0, 4.0, 5.0], asym, "b", True)]
    dump_json(out / "moran_asymmetric.json", fx)

    # Bivariate.
    def bv_case(label, x, y, graph, transform):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        np.random.seed(SEED)
        m = Moran_BV(x, y, graph, transformation=transform, permutations=PERMS)
        return {
            "label": label,
            "x": x,
            "y": y,
            "transform": transform,
            "analytic": {"i": float(m.I)},
            "permutations": PERMS,
            "perm": pick(m, ["EI_sim", "seI_sim", "VI_sim", "z_sim", "p_sim", "p_z_sim"]),
        }

    fx = header("moran_bv_columbus", "Moran_BV on columbus queen: x = INC, y = CRIME.", queen)
    fx["cases"] = [
        bv_case("INC~CRIME/r", attrs["INC"], attrs["CRIME"], queen, "r"),
        bv_case("INC~CRIME/b", attrs["INC"], attrs["CRIME"], queen, "b"),
        bv_case("CRIME~HOVAL/r", attrs["CRIME"], attrs["HOVAL"], queen, "r"),
    ]
    dump_json(out / "moran_bv_columbus.json", fx)

    fx = header("moran_bv_lattice", "Moran_BV on the 4x4 rook lattice.", lattice)
    fx["cases"] = [bv_case("ramp~random/r", ys["ramp"], ys["random"], lattice, "r"),
                   bv_case("checker~ramp/r", ys["checkerboard"], ys["ramp"], lattice, "r")]
    dump_json(out / "moran_bv_lattice.json", fx)

    # Rates: synthetic events/base on columbus (spec probe) and a seeded random set.
    def rate_case(label, e, b, graph, adjusted):
        e = np.asarray(e, float)
        b = np.asarray(b, float)
        np.random.seed(SEED)
        m = Moran_Rate(e, b, graph, adjusted=adjusted, transformation="r", permutations=PERMS)
        analytic = pick(m, MORAN_ANALYTIC)
        return {
            "label": label,
            "e": e,
            "b": b,
            "adjusted": adjusted,
            "y": np.asarray(m.y, float),
            "analytic": analytic,
            "permutations": PERMS,
            "perm": pick(m, MORAN_PERM),
        }

    e_c = np.round(attrs["CRIME"])
    b_c = attrs["HOVAL"] * 10.0
    rng = np.random.default_rng(11)
    b_r = rng.integers(50, 500, size=49).astype(float)
    e_r = rng.poisson(b_r * 0.07).astype(float)
    fx = header("moran_rate_columbus", "Moran_Rate on columbus queen contiguity.", queen)
    fx["cases"] = [
        rate_case("crime/hoval adjusted", e_c, b_c, queen, True),
        rate_case("crime/hoval raw", e_c, b_c, queen, False),
        rate_case("poisson adjusted", e_r, b_r, queen, True),
        rate_case("poisson raw", e_r, b_r, queen, False),
    ]
    # Direct EB-standardization oracle (esda.smoothing.assuncao_rate), no graph involved.
    fx["assuncao"] = [
        {"e": e_r, "b": b_r, "y": np.asarray(assuncao_rate(e_r, b_r), float)},
        {"e": e_c, "b": b_c, "y": np.asarray(assuncao_rate(e_c, b_c), float)},
    ]
    dump_json(out / "moran_rate_columbus.json", fx)

    # Documented PySAL behavior that Spatex does not replicate: a constant y gives NaN
    # statistics (Spatex raises ArgumentError instead; see known_deviations.exs).
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(SEED)
        const = Moran(np.full(49, 3.0), queen, permutations=99)
    fx = header("moran_constant", "PySAL Moran on a constant y: NaN statistics.", queen)
    fx["pysal"] = {"i": float(const.I), "z_norm": float(const.z_norm), "p_sim": float(const.p_sim)}
    fx["y"] = np.full(49, 3.0)
    dump_json(out / "moran_constant.json", fx)

    esda_geary_g_joins(out)


GEARY_ANALYTIC = [
    "C", "EC", "VC_norm", "seC_norm", "z_norm", "p_norm",
    "VC_rand", "seC_rand", "z_rand", "p_rand",
]
GEARY_PERM = ["EC_sim", "seC_sim", "VC_sim", "z_sim", "p_sim", "p_z_sim"]
G_ANALYTIC = ["G", "EG", "VG", "z_norm", "p_norm", "den_sum"]
G_PERM = ["EG_sim", "seG_sim", "VG_sim", "z_sim", "p_sim", "p_z_sim"]


def geary_case(label, y, graph, transform):
    y = np.asarray(y, dtype=float)
    analytic = Geary(y, graph, transformation=transform, permutations=0)
    np.random.seed(SEED)
    sim = Geary(y, graph, transformation=transform, permutations=PERMS)
    return {
        "label": label,
        "y": y,
        "transform": transform,
        "analytic": pick(analytic, GEARY_ANALYTIC),
        "permutations": PERMS,
        "perm": pick(sim, GEARY_PERM),
    }


def g_case(label, y, graph):
    y = np.asarray(y, dtype=float)
    analytic = G(y, graph, permutations=0)
    np.random.seed(SEED)
    sim = G(y, graph, permutations=PERMS)
    return {
        "label": label,
        "y": y,
        "analytic": pick(analytic, G_ANALYTIC),
        "permutations": PERMS,
        "perm": pick(sim, G_PERM),
    }


JC_ANALYTIC = ["J", "bb", "ww", "bw", "chi2", "chi2_p", "chi2_dof", "autocorr_pos", "autocorr_neg"]
JC_PERM = [
    "min_bb", "mean_bb", "max_bb", "min_bw", "mean_bw", "max_bw",
    "p_sim_bb", "p_sim_bw", "p_sim_chi2", "p_sim_autocorr_pos", "p_sim_autocorr_neg",
    "sim_autocorr_chi2",
]


def jc_case(label, y, graph, drop_islands=True):
    y = np.asarray(y, dtype=float)
    analytic = Join_Counts(y, graph, permutations=0, drop_islands=drop_islands)
    np.random.seed(SEED)
    sim = Join_Counts(y, graph, permutations=PERMS, drop_islands=drop_islands)
    perm = pick(sim, JC_PERM)
    perm["mean_autocorr_pos"] = float(np.mean(sim.sim_autocurr_pos))
    perm["mean_autocorr_neg"] = float(np.mean(sim.sim_autocurr_neg))
    return {
        "label": label,
        "y": y,
        "drop_islands": drop_islands,
        "analytic": pick(analytic, JC_ANALYTIC),
        "crosstab": np.asarray(analytic.crosstab, float),
        "expected": np.asarray(analytic.expected, float),
        "permutations": PERMS,
        "perm": perm,
    }


def centroid_points(geoms):
    return np.column_stack([geoms.centroid.x.to_numpy(), geoms.centroid.y.to_numpy()])


def esda_geary_g_joins(out):
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    queen = Graph.build_contiguity(geoms, rook=False)
    pts = centroid_points(geoms)
    band = Graph.build_distance_band(pts, threshold=1.0, binary=True)
    sparse_band = Graph.build_distance_band(pts, threshold=0.5, binary=True)
    assert len(band.isolates) == 0 and len(sparse_band.isolates) > 0, "adjust the thresholds"

    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(4) for c in range(4)])
    lattice = Graph.build_contiguity(cells, rook=True)
    rng = np.random.default_rng(7)
    checker = np.array([(r + c) % 2 for r in range(4) for c in range(4)], dtype=float)
    ramp = np.arange(1, 17, dtype=float)
    rand16 = np.round(rng.uniform(1, 20, size=16), 4)
    path = Graph.from_arrays([0, 1, 1, 2, 3], [1, 0, 2, 1, 3], [1.0, 1.0, 1.0, 1.0, 0.0])
    many = Graph.from_arrays(
        [0, 1, 1, 2, 2, 3, 4, 5, 6, 7], [1, 0, 2, 1, 3, 2, 4, 5, 6, 7], [1.0] * 6 + [0.0] * 4
    )

    # Geary.
    cases = [geary_case(f"{v}/{t}", attrs[v], queen, t) for v in ("CRIME", "HOVAL", "INC") for t in ("r", "b")]
    fx = header("geary_columbus_queen", "Geary's C on columbus queen contiguity.", queen)
    fx["cases"] = cases
    dump_json(out / "geary_columbus_queen.json", fx)

    fx = header("geary_lattice", "Geary's C on a 4x4 rook lattice.", lattice)
    fx["cases"] = [geary_case(f"{k}/{t}", v, lattice, t)
                   for k, v in (("ramp", ramp), ("checkerboard", checker * 3 + 1), ("random", rand16))
                   for t in ("r", "b")]
    dump_json(out / "geary_lattice.json", fx)

    # esda 2.10.0 crashes (ValueError, shape mismatch) when Geary receives a libpysal Graph with
    # islands: nonzero() drops the zero-weight self-loop but .data keeps it. The oracle values
    # therefore come from the equivalent W object, which stores no entry for an island.
    y8 = [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0]
    try:
        Geary(np.asarray(y8), many, permutations=0)
        crash = None
    except ValueError as exc:
        crash = str(exc)
    fx = header("geary_islands", "Geary's C with island nodes (oracle computed on the W object).", many)
    fx["graph_crash"] = crash
    fx["cases"] = [geary_case("path+4islands/r", y8, many.to_W(), "r"),
                   geary_case("path+4islands/b", y8, many.to_W(), "b")]
    dump_json(out / "geary_islands.json", fx)

    # Getis-Ord G (binary weights are forced by esda).
    fx = header("g_columbus_band", "General G on a binary distance band (1.0) of columbus centroids.", band)
    fx["cases"] = [g_case(v, attrs[v], band) for v in ("CRIME", "HOVAL", "INC")]
    dump_json(out / "g_columbus_band.json", fx)

    fx = header("g_columbus_sparse_band", "General G on a binary distance band (0.5) with islands.", sparse_band)
    fx["cases"] = [g_case(v, attrs[v], sparse_band) for v in ("CRIME", "HOVAL")]
    fx["isolates"] = len(sparse_band.isolates)
    dump_json(out / "g_columbus_sparse_band.json", fx)

    fx = header("g_columbus_queen", "General G on columbus queen contiguity.", queen)
    fx["cases"] = [g_case("CRIME", attrs["CRIME"], queen)]
    dump_json(out / "g_columbus_queen.json", fx)

    fx = header("g_lattice", "General G on a 4x4 rook lattice and islands.", lattice)
    fx["cases"] = [g_case("ramp", ramp, lattice), g_case("random", rand16, lattice)]
    dump_json(out / "g_lattice.json", fx)

    fx = header("g_islands", "General G on path + four islands.", many)
    fx["cases"] = [g_case("path+4islands", [3.0, 1.0, 4.0, 1.5, 5.0, 9.0, 2.0, 6.0], many)]
    dump_json(out / "g_islands.json", fx)

    complete = Graph.build_distance_band(pts, threshold=90.0, binary=True)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        np.random.seed(SEED)
        gc = G(attrs["CRIME"], complete, permutations=99)
    fx = header("g_complete", "General G on a complete graph: VG <= 0, z_norm and p_norm NaN.", complete)
    fx["y"] = attrs["CRIME"]
    fx["pysal"] = {"g": float(gc.G), "vg": float(gc.VG), "z_norm": float(gc.z_norm), "p_norm": float(gc.p_norm)}
    dump_json(out / "g_complete.json", fx)

    # Join counts.
    crime = attrs["CRIME"]
    hi_median = (crime > np.median(crime)).astype(float)
    hi_mean = (crime > crime.mean()).astype(float)
    fx = header("join_counts_columbus", "Join counts on columbus queen (CRIME above median / mean).", queen)
    fx["cases"] = [jc_case("CRIME>median", hi_median, queen), jc_case("CRIME>mean", hi_mean, queen)]
    dump_json(out / "join_counts_columbus.json", fx)

    half = np.array([1.0 if r >= 2 else 0.0 for r in range(4) for c in range(4)])
    fx = header("join_counts_lattice", "Join counts on a 4x4 rook lattice.", lattice)
    fx["cases"] = [jc_case("checkerboard", checker, lattice), jc_case("half", half, lattice)]
    dump_json(out / "join_counts_lattice.json", fx)

    iso_y = np.array([1.0, 0.0, 1.0, 1.0])
    many_y = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    sparse_y = (crime > np.median(crime)).astype(float)
    fx = header("join_counts_islands", "Join counts with islands, drop_islands true and false.", many)
    fx["cases"] = [
        jc_case("path+4islands/drop", many_y, many, True),
        jc_case("path+4islands/keep", many_y, many, False),
    ]
    fx["sparse"] = {
        "graph": export_graph(sparse_band),
        "cases": [
            jc_case("sparse band/drop", sparse_y, sparse_band, True),
            jc_case("sparse band/keep", sparse_y, sparse_band, False),
        ],
    }
    fx["path"] = {
        "graph": export_graph(path),
        "cases": [jc_case("path+island/drop", iso_y, path, True), jc_case("path+island/keep", iso_y, path, False)],
    }
    dump_json(out / "join_counts_islands.json", fx)


if __name__ == "__main__":
    main()
