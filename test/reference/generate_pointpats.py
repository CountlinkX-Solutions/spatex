"""Generate pointpats 2.6.0 oracle fixtures for centrography, distance statistics and quadrats.

Run from the repository root under a memory cap (see test/reference/README.md):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_pointpats.py

Writes test/reference/fixtures/pointpats/{centrography,distance,simulation,quadrat}.json.

* ``centrography.json``: per point set the deterministic centrography results (mean/median
  centers, standard distance, ellipses, hull, bounding boxes/rectangles/circles, dtot).
* ``distance.json``: G, K, L, F, J on stored patterns with stored empty-space points (F and J take
  the empty points' nearest-neighbour distances as ``distances``, which makes them deterministic).
* ``simulation.json``: ``*_test`` runs (n_simulations=999, n_jobs=1). pointpats' generators are
  unseeded, so only distributional summaries (per-support mean/sd/min/max of the simulations and
  the p-values) are stored and compared statistically.
* ``quadrat.json``: RectangleM / HexagonM / QStatistic counts and chi-square results; the
  realization-based p-value is summarized the same way as the simulations.

All random patterns are generated here with numpy's default_rng and STORED, never regenerated in
Elixir.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pointpats  # noqa: E402
from pointpats import centrography as cg  # noqa: E402
from pointpats import distance_statistics as ds  # noqa: E402
from pointpats.quadrat_statistics import HexagonM, QStatistic, RectangleM  # noqa: E402
from scipy.spatial import ConvexHull  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

OUT = FIXTURES / "pointpats"
warnings.simplefilter("ignore")

P10 = np.array(
    [[66.22, 32.54], [22.52, 38.53], [31.01, 81.21], [9.47, 31.02], [30.78, 60.10],
     [75.21, 58.44], [79.26, 85.36], [8.23, 39.93], [98.73, 77.17], [89.78, 42.53]]
)


def rng_points(seed, n, hi=100.0):
    return np.round(np.random.default_rng(seed).uniform(0, hi, (n, 2)), 3)


def clustered(seed, n_clusters=5, per=40, sd=3.0):
    r = np.random.default_rng(seed)
    centers = r.uniform(10, 90, (n_clusters, 2))
    pts = np.concatenate([c + r.normal(0, sd, (per, 2)) for c in centers])
    return np.round(pts, 3)


R30 = rng_points(7, 30)
E50 = rng_points(8, 50)
R100 = rng_points(21, 100)
R500 = rng_points(22, 500)
C200 = clustered(31)
R30D = np.vstack([R30, R30[:1]])  # one duplicated point
CROSS = np.array([[0, 0], [2, 0], [-2, 0], [0, 1], [0, -1]], dtype=float)
DIAG = np.array([[0, 0], [1, 1], [2, 2.1], [3, 2.9], [4, 4.2]], dtype=float)
TRI = np.array([[0, 0], [4, 0], [2, 1]], dtype=float)
LINE = np.array([[0, 0], [1, 1], [2, 2], [3, 3]], dtype=float)  # collinear edge case


def attempt(fn):
    try:
        return fn()
    except Exception as e:  # noqa: BLE001 - recorded as the oracle's behavior
        return {"error": f"{type(e).__name__}: {e}"}


def poly_area(c):
    x, y = c[:, 0], c[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)


def centro(pts):
    n = len(pts)
    w = np.arange(1, n + 1, dtype=float)
    out = {"points": pts.tolist(), "weights": w.tolist(), "n": n}
    out["mean_center"] = cg.mean_center(pts).tolist()
    out["weighted_mean_center"] = cg.weighted_mean_center(pts, w).tolist()
    out["manhattan_median"] = cg.manhattan_median(pts).tolist()
    out["std_distance"] = float(cg.std_distance(pts))
    out["dtot_at_mean"] = float(cg.dtot(cg.mean_center(pts), pts))
    em = cg.euclidean_median(pts)
    out["euclidean_median"] = {"xy": em.tolist(), "dtot": float(cg.dtot(em, pts))}
    out["mbr"] = list(cg.minimum_bounding_rectangle(pts))
    ell = {}
    if n >= 3:
        ell["crimestat"] = list(cg.ellipse(pts))
        ell["crimestat_weighted"] = list(cg.ellipse(pts, weights=w))
        for cc in (True, False):
            for dc in (True, False):
                ell[f"yuill_{cc}_{dc}"] = list(
                    cg.ellipse(pts, method="yuill", crimestatCorr=cc, degfreedCorr=dc)
                )
    out["ellipse"] = ell

    def hull():
        return pts[ConvexHull(pts).vertices].tolist()

    out["hull"] = attempt(hull)

    def mrr():
        corners, angle = cg.minimum_rotated_rectangle(pts, return_angle=True)
        corners = np.asarray(corners)
        return {"corners": corners.tolist(), "angle": float(angle), "area": poly_area(corners)}

    out["mrr"] = attempt(mrr)

    def mbc():
        (cx, cy), r = cg.minimum_bounding_circle(pts)
        return {"center": [float(cx), float(cy)], "radius": float(r)}

    out["mbc"] = attempt(mbc)
    return out


def centrography():
    sets = {"p10": P10, "r30": R30, "r100": R100, "r500": R500, "c200": C200,
            "cross": CROSS, "diag": DIAG, "tri": TRI}
    cases = {k: centro(v) for k, v in sets.items()}
    # collinear: ellipse (den != 0 case) and the geometry that PySAL cannot build
    line = {"points": LINE.tolist(), "mean_center": cg.mean_center(LINE).tolist(),
            "std_distance": float(cg.std_distance(LINE)),
            "ellipse": list(cg.ellipse(LINE)),
            "hull": attempt(lambda: LINE[ConvexHull(LINE).vertices].tolist())}
    cases["line"] = line
    cases["n2_ellipse"] = {"points": [[0, 0], [1, 2]],
                           "ellipse": attempt(lambda: list(cg.ellipse(np.array([[0.0, 0.0], [1, 2]]))))}
    dump_json(OUT / "centrography.json", {"meta": meta_pp(), "cases": cases})


def meta_pp():
    m = meta()
    m["pointpats"] = pointpats.__version__
    return m


SUPPORTS = {
    "default": None,
    "int5": 5,
    "tuple1": (25,),
    "tuple3": (0, 30, 7),
    "array": [0.0, 2.5, 5.0, 10.0, 20.0, 40.0],
}


def distance_case(pts, empty):
    from scipy.spatial import cKDTree

    tree = cKDTree(pts)
    f_dist = tree.query(empty, k=1)[0]
    out = {"points": pts.tolist(), "empty_points": empty.tolist(), "supports": {}}
    for name, sup in SUPPORTS.items():
        s = {}
        gs, gv = ds.g(pts, support=sup)
        s["g"] = {"support": gs.tolist(), "statistic": gv.tolist()}
        ks, kv = ds.k(pts, support=sup)
        s["k"] = {"support": ks.tolist(), "statistic": kv.tolist()}
        ls, lv = ds.l(pts, support=sup)
        s["l"] = {"support": ls.tolist(), "statistic": lv.tolist()}
        ls2, lv2 = ds.l(pts, support=sup, linearized=True)
        s["l_linearized"] = {"support": ls2.tolist(), "statistic": lv2.tolist()}
        fs, fv = ds.f(pts, support=sup, distances=f_dist)
        s["f"] = {"support": fs.tolist(), "statistic": fv.tolist()}
        g_dist = tree.query(pts, k=2)[0][:, 1]
        for trunc in (True, False):
            js, jv = ds.j(pts, support=sup, distances=(g_dist, f_dist), truncate=trunc)
            s["j_truncated" if trunc else "j_untruncated"] = {
                "support": js.tolist(), "statistic": jv.tolist()}
        out["supports"][name] = s
    return out


def distance():
    cases = {"r30": distance_case(R30, E50), "r30_dup": distance_case(R30D, E50),
             "r100": distance_case(R100, rng_points(9, 200)),
             "c200": distance_case(C200, rng_points(10, 200))}
    dump_json(OUT / "distance.json", {"meta": meta_pp(), "cases": cases})


def summarize(res, keep):
    sims = res.simulations
    out = {"support": res.support.tolist(), "statistic": res.statistic.tolist(),
           "pvalue": res.pvalue.tolist()}
    if sims is not None:
        out["sim_mean"] = sims.mean(axis=0).tolist()
        out["sim_sd"] = sims.std(axis=0).tolist()
        out["sim_min"] = sims.min(axis=0).tolist()
        out["sim_max"] = sims.max(axis=0).tolist()
        out["n_simulations"] = int(sims.shape[0])
    return out


def simulation():
    np.random.seed(0)
    n_sims = 999
    cases = {}
    for name, pts in (("r100", R100), ("c200", C200)):
        c = {"points": pts.tolist(), "n_simulations": n_sims}
        kw = dict(n_simulations=n_sims, keep_simulations=True, n_jobs=1)
        c["g"] = summarize(ds.g_test(pts, **kw), True)
        c["k"] = summarize(ds.k_test(pts, **kw), True)
        c["l"] = summarize(ds.l_test(pts, **kw), True)
        c["f"] = summarize(ds.f_test(pts, **kw), True)
        c["j_untruncated"] = summarize(ds.j_test(pts, truncate=False, **kw), True)
        cases[name] = c
    dump_json(OUT / "simulation.json", {"meta": meta_pp(), "cases": cases})


EDGE = np.array([[x, y] for x in range(0, 7) for y in range(0, 5)], dtype=float)
HEXEDGE = np.array([[0, 0], [3, 0], [1.5, 2.598076211353316], [6, 3], [4.5, 0.2], [0.5, 1],
                    [2.9, 1.1], [3.1, 1.1], [5.2, 2.2], [0.2, 2.5]])


def quadrat_case(pts, **kw):
    q = QStatistic(pts, **kw)
    out = {"kwargs": {k: v for k, v in kw.items()}, "cell_ids": [int(i) for i in q.cell_ids],
           "counts": [int(c) for c in q.mr.point_location_sta().values()],
           "chi2": float(q.chi2), "chi2_pvalue": float(q.chi2_pvalue), "df": int(q.df),
           "chi2_contrib": q.chi2_contrib.tolist(), "mbb": [float(v) for v in q.mbb],
           "pysal_num": int(q.mr.num)}
    if hasattr(q.mr, "count_row_even"):
        out.update(count_column=q.mr.count_column, count_row_even=q.mr.count_row_even,
                   count_row_odd=q.mr.count_row_odd)
    else:
        out.update(count_column=q.mr.count_column, count_row=q.mr.count_row,
                   rectangle_width=q.mr.rectangle_width, rectangle_height=q.mr.rectangle_height)
    return out


def quadrat():
    cases = []
    for name, pts in (("r30", R30), ("r100", R100), ("edge", EDGE), ("hexedge", HEXEDGE)):
        specs = [dict(shape="rectangle", nx=3, ny=3), dict(shape="rectangle", nx=2, ny=4),
                 dict(shape="rectangle", rectangle_width=30, rectangle_height=40),
                 dict(shape="hexagon", lh=20), dict(shape="hexagon", lh=35),
                 dict(shape="hexagon", lh=1.5)]
        if name in ("edge", "hexedge"):
            specs = [dict(shape="rectangle", nx=3, ny=2), dict(shape="rectangle", nx=6, ny=4),
                     dict(shape="hexagon", lh=1.0), dict(shape="hexagon", lh=2.0)]
        for spec in specs:
            case = quadrat_case(pts, **spec)
            case["points_name"] = name
            case["points"] = pts.tolist()
            cases.append(case)
    # realization based p-values: distribution summaries only
    sims = []
    np.random.seed(0)
    for name, pts, spec in (("r100", R100, dict(shape="rectangle", nx=4, ny=4)),
                            ("r100", R100, dict(shape="hexagon", lh=15)),
                            ("c200", C200, dict(shape="rectangle", nx=4, ny=4))):
        q = QStatistic(pts, realizations=499, rng=np.random.default_rng(5), **spec)
        r = np.asarray(q.chi2_realizations)
        sims.append({"points_name": name, "points": pts.tolist(), "kwargs": spec,
                     "chi2": float(q.chi2), "realizations": 499, "r_mean": float(r.mean()),
                     "r_sd": float(r.std()), "r_pvalue": float(q.chi2_r_pvalue)})
    dump_json(OUT / "quadrat.json", {"meta": meta_pp(), "cases": cases, "realizations": sims})


if __name__ == "__main__":
    centrography()
    distance()
    simulation()
    quadrat()
    print("done")
