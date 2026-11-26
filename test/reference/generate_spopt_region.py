"""Generate spopt 0.8.0 oracle fixtures for regionalization (SKATER, spatial Ward, Region-K-Means,
objective functions, AZP and Max-p).

Run from the repository root under a memory cap (see test/reference/README.md):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spopt_region.py

Writes test/reference/fixtures/region/{skater,ward,region_kmeans,objective,azp,maxp}.json.

* ``skater.json`` / ``ward.json`` / ``region_kmeans.json`` are deterministic ports: labels are
  compared exactly (Region-K-Means with the seeds numpy drew for PySAL, stored here).
* ``azp.json`` / ``maxp.json`` hold PySAL objective distributions and brute-force optima for
  tiny lattices; the heuristics are compared by validity and objective, never by labels.

Every ``W`` is rebuilt with SORTED neighbor lists (ids ``0..n-1``): Spatex CSR rows are sorted
while ``Queen.from_dataframe`` is not, and Region-K-Means tie-breaking depends on the order.
"""

import itertools
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
import sklearn.metrics.pairwise as skm  # noqa: E402
import spopt  # noqa: E402
from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from spopt.region import AZP, MaxPHeuristic, RegionKMeansHeuristic  # noqa: E402
from spopt.region.maxp import _assign_enclave, _grow_cluster_for_poly  # noqa: E402
from spopt.region.objective_function import (  # noqa: E402
    ObjectiveFunctionCenter,
    ObjectiveFunctionPairwise,
)
from spopt.region.skater import SpanningForest  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

OUT = FIXTURES / "region"
warnings.simplefilter("ignore")
ATTRS = ["CRIME", "INC", "HOVAL"]


def region_meta():
    m = meta()
    m.update(
        spopt=spopt.__version__,
        scikit_learn=sklearn.__version__,
        scipy=scipy.__version__,
    )
    return m


def sorted_w(neighbors):
    n = len(neighbors)
    return libpysal.weights.W(
        {i: sorted(int(j) for j in neighbors[i]) for i in range(n)},
        ids=list(range(n)),
    )


def lattice(r, c, queen=False):
    w = libpysal.weights.lat2W(r, c, rook=not queen)
    return sorted_w({i: w.neighbors[i] for i in range(w.n)})


def disjoint(*ws):
    """Disjoint union of W objects (islands allowed)."""
    nb, off = {}, 0
    for w in ws:
        for i in range(w.n):
            nb[i + off] = [j + off for j in w.neighbors[i]]
        off += w.n
    return sorted_w(nb)


def graph_json(w):
    focal, neighbor = [], []
    for i in range(w.n):
        for j in sorted(w.neighbors[i]):
            focal.append(i)
            neighbor.append(int(j))
    return {"n": w.n, "focal": focal, "neighbor": neighbor}


def columbus():
    gdf = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    w0 = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    w = sorted_w({i: w0.neighbors[i] for i in range(w0.n)})
    return gdf, w, gdf[ATTRS].to_numpy(float)


def scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


def labels_list(a):
    return [int(v) for v in a]


# --- SKATER -------------------------------------------------------------------------------

def run_skater(w, x, k, floor=-np.inf, islands="increase", trace=False, metric=None):
    sf = SpanningForest(dissimilarity=metric or skm.manhattan_distances)
    caught = []
    err = None
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        try:
            sf.fit(k, w, data=x, quorum=floor, trace=trace, islands=islands)
        except ValueError as e:
            err = str(e)
        caught = [str(m.message) for m in ws]
    out = {"k": k, "floor": None if floor == -np.inf else floor, "islands": islands,
           "warnings": caught}
    if err is not None:
        out["error"] = err
        return out
    labels = sf.current_labels_
    out["labels"] = labels_list(labels)
    out["score"] = float(sf.score(x, labels=labels))
    msf = sf.minimum_spanning_forest_.tocoo()
    out["msf_edges"] = sorted([int(a), int(b)] for a, b in zip(msf.row, msf.col))
    if trace:
        out["trace"] = [
            {"labels": labels_list(lab), "in": None if np.isnan(d.in_node) else int(d.in_node),
             "out": None if np.isnan(d.out_node) else int(d.out_node),
             "score": "inf" if np.isinf(d.score) else float(d.score)}
            for lab, d in sf._trace
        ]
    return out


def skater_fixture():
    gdf, w, x = columbus()
    cases = []
    for k in (2, 5, 10):
        c = run_skater(w, x, k, trace=(k == 5))
        c.update(name=f"columbus_k{k}", x=None)
        cases.append(c)
    c = run_skater(w, x, 5, floor=8)
    c["name"] = "columbus_k5_floor8"
    cases.append(c)
    c = run_skater(w, x, 20, floor=5)
    c["name"] = "columbus_k20_floor5_early_stop"
    cases.append(c)
    c = run_skater(w, x, 5, metric=skm.euclidean_distances)
    c["name"] = "columbus_k5_euclidean"
    c["dissimilarity"] = "euclidean"
    cases.append(c)
    for c in cases:
        c.pop("x", None)

    # duplicates: PySAL drops zero-dissimilarity edges (QUIRK Z)
    wl = lattice(3, 3)
    xd = np.array([[1.0], [1.0], [2.0], [3.0], [3.0], [5.0], [8.0], [8.0], [9.0]])
    dup = run_skater(wl, xd, 3)
    dup["name"] = "lattice3_duplicates"

    # islands: two lattices and an isolated area
    wi = disjoint(lattice(3, 3), lattice(3, 2), libpysal.weights.W({0: []}, ids=[0]))
    rng = np.random.default_rng(3)
    xi = np.round(rng.normal(size=(wi.n, 2)) * 5, 3)
    isl = []
    for islands in ("increase", "ignore"):
        c = run_skater(wi, xi, 3, islands=islands)
        c["name"] = f"islands_{islands}_k3"
        isl.append(c)
    c = run_skater(wi, xi, 3, floor=2)
    c["name"] = "islands_floor_larger_than_island"
    isl.append(c)
    return {
        "meta": region_meta(),
        "columbus": {"graph": graph_json(w), "x": x.tolist(), "attrs": ATTRS},
        "cases": cases,
        "duplicates": {"graph": graph_json(wl), "x": xd.tolist(), "case": dup},
        "islands": {"graph": graph_json(wi), "x": xi.tolist(), "cases": isl},
    }


# --- Ward ---------------------------------------------------------------------------------

def ward(w, x, k, distances=False):
    full = k < max(100, 0.02 * len(x))
    a = AgglomerativeClustering(
        n_clusters=k, connectivity=w.sparse, linkage="ward", compute_distances=distances
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a.fit(x)
    out = {"k": k, "labels": labels_list(a.labels_), "n_connected_components": int(a.n_connected_components_),
           "full_tree": bool(full), "children": a.children_.tolist()}
    if distances:
        out["distances"] = [float(v) for v in a.distances_]
    return out


def ward_fixture():
    gdf, w, x = columbus()
    cases = [dict(ward(w, x, k, distances=True), name=f"columbus_k{k}") for k in (1, 2, 5, 10)]
    wd = disjoint(lattice(4, 4), lattice(3, 3))
    rng = np.random.default_rng(11)
    xd = np.round(rng.normal(size=(wd.n, 2)) * 3, 3)
    disc = [dict(ward(wd, xd, k, distances=True), name=f"disconnected_k{k}") for k in (2, 3)]
    wb = lattice(25, 10)
    xb = np.round(np.random.default_rng(5).normal(size=(wb.n, 2)) * 4, 3)
    big = dict(ward(wb, xb, 120), name="lattice250_k120_partial")
    big.pop("children")
    small = dict(ward(wb, xb, 8, distances=True), name="lattice250_k8")
    return {
        "meta": region_meta(),
        "columbus": {"graph": graph_json(w), "x": x.tolist()},
        "cases": cases,
        "disconnected": {"graph": graph_json(wd), "x": xd.tolist(), "cases": disc},
        "lattice250": {"graph": graph_json(wb), "x": xb.tolist(), "cases": [big, small]},
    }


# --- Region-K-Means -----------------------------------------------------------------------

def rkm_fixture():
    gdf, w, x = columbus()
    cases = []
    for s in (0, 1, 2):
        np.random.seed(s)
        seeds = np.random.choice(np.arange(w.n), size=5, replace=False)
        r = RegionKMeansHeuristic(x, 5, w, seed=s)
        r.solve()
        cases.append({"seed": s, "seeds": labels_list(seeds), "labels": labels_list(r.labels_),
                      "centroids": np.asarray(r.centroids_).tolist(), "iters": int(r.iters_)})
    return {"meta": region_meta(), "graph": graph_json(w), "x": x.tolist(), "cases": cases}


# --- objective functions ------------------------------------------------------------------

def objective_fixture():
    gdf, w, x = columbus()
    sf = SpanningForest()
    sf.fit(5, w, data=x)
    wardl = AgglomerativeClustering(n_clusters=5, connectivity=w.sparse, linkage="ward").fit(x).labels_
    np.random.seed(0)
    seeds = np.random.choice(np.arange(w.n), size=5, replace=False)
    rk = RegionKMeansHeuristic(x, 5, w, seed=0)
    rk.solve()
    cases = []
    for name, lab in (("skater", sf.current_labels_), ("ward", wardl), ("region_kmeans", rk.labels_)):
        lab = np.asarray(lab)
        entry = {"name": name, "labels": labels_list(lab)}
        for mname, metric in (("manhattan", skm.manhattan_distances), ("euclidean", skm.euclidean_distances)):
            entry[f"pairwise_{mname}"] = scalar(ObjectiveFunctionPairwise(metric)(lab, x))
            entry[f"center_{mname}"] = scalar(ObjectiveFunctionCenter(metric)(lab, x))
        ssd = 0.0
        pair_sq = 0.0
        for r in np.unique(lab):
            z = x[lab == r]
            ssd += float(((z - z.mean(axis=0)) ** 2).sum())
            for i, j in itertools.combinations(range(len(z)), 2):
                pair_sq += float(((z[i] - z[j]) ** 2).sum())
        entry["center_sqeuclidean"] = ssd
        entry["pairwise_sqeuclidean"] = pair_sq
        cases.append(entry)
    return {"meta": region_meta(), "x": x.tolist(), "cases": cases}


# --- brute force (tiny lattices) ----------------------------------------------------------

def connected(nb, nodes):
    nodes = set(nodes)
    stack = [min(nodes)]
    seen = {stack[0]}
    while stack:
        a = stack.pop()
        for b in nb[a]:
            if b in nodes and b not in seen:
                seen.add(b)
                stack.append(b)
    return len(seen) == len(nodes)


def partitions(items):
    if not items:
        yield []
        return
    first = items[0]
    for p in partitions(items[1:]):
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]
        yield [[first]] + p


def pairwise_l1(x, part):
    return sum(abs(x[i] - x[j]) for reg in part for i, j in itertools.combinations(reg, 2))


def brute_force(w, x, pops=None, threshold=None):
    nb = {i: list(w.neighbors[i]) for i in range(w.n)}
    best = {}
    for part in partitions(list(range(w.n))):
        if not all(connected(nb, reg) for reg in part):
            continue
        if pops is not None and not all(sum(pops[i] for i in reg) >= threshold for reg in part):
            continue
        k = len(part)
        o = pairwise_l1(x, part)
        if k not in best or o < best[k]:
            best[k] = o
    return {str(k): v for k, v in sorted(best.items())}


def obj_of(x, labels):
    labels = np.asarray(labels)
    return float(pairwise_l1(x, [list(np.where(labels == r)[0]) for r in np.unique(labels)]))


def azp_fixture():
    w = lattice(3, 4)
    rng = np.random.default_rng(7)
    x = np.round(rng.normal(size=w.n) * 10, 3)
    df = pd.DataFrame({"x": x})
    brute = brute_force(w, x)
    py = []
    for s in range(30):
        a = AZP(df, w, ["x"], n_clusters=3, random_state=s)
        a.solve()
        py.append(obj_of(x, a.labels_))
    gdf, wc, xc = columbus()
    pyc = []
    for s in range(30):
        a = AZP(gdf, wc, ATTRS, n_clusters=5, random_state=s)
        a.solve()
        pyc.append(scalar(ObjectiveFunctionPairwise()(np.asarray(a.labels_).astype(int), xc)))
    return {
        "meta": region_meta(),
        "lattice": {"graph": graph_json(w), "x": x.tolist(), "k": 3,
                    "brute_force_optimum": brute["3"], "brute_force_by_k": brute,
                    "pysal_objectives": py},
        "columbus": {"graph": graph_json(wc), "x": xc.tolist(), "k": 5, "metric": "euclidean",
                     "pysal_objectives": pyc},
    }


def maxp_fixture():
    w = lattice(3, 4)
    rng = np.random.default_rng(7)
    x = np.round(rng.normal(size=w.n) * 10, 3)
    pop = rng.integers(1, 6, w.n).astype(float)
    df = pd.DataFrame({"x": x, "pop": pop})
    brute = brute_force(w, x, pops=pop, threshold=8.0)
    py = []
    for s in range(10):
        np.random.seed(s)
        m = MaxPHeuristic(df, w, ["x"], "pop", 8, max_iterations_construction=20)
        m.solve()
        lab = np.asarray(m.labels_) - 1
        py.append({"seed": s, "p": int(m.p), "objective": obj_of(x, lab)})

    gdf, wc, xc = columbus()
    gdf = gdf.copy()
    gdf["cnt"] = 1.0
    pyc = []
    for s in range(10):
        np.random.seed(s)
        m = MaxPHeuristic(gdf, wc, ATTRS, "cnt", 8, top_n=2, max_iterations_construction=99)
        m.solve()
        lab = np.asarray(m.labels_) - 1
        pyc.append({"seed": s, "p": int(m.p),
                    "objective": scalar(ObjectiveFunctionPairwise(skm.manhattan_distances)(lab.astype(int), xc))})

    # deterministic construction (top_n=1 makes the enclave pick deterministic) for given orders
    dist = skm.manhattan_distances(xc)
    thr_arr = np.ones(wc.n)
    construct = []
    for s in range(5):
        order = np.random.default_rng(100 + s).permutation(wc.n)
        labels = [0] * wc.n
        c = 0
        enclave, region_list, region_attr = [], {}, {}
        for p in order:
            if labels[p] != 0:
                continue
            c += 1
            ids, tot = _grow_cluster_for_poly(labels, thr_arr, int(p), list(wc.neighbors[int(p)]), c, wc, 8)
            if tot < 8:
                c -= 1
                enclave.extend(ids)
            else:
                region_list[c] = ids
                region_attr[c] = tot
        p_count = len(region_list)
        lab, _, _ = _assign_enclave(enclave, labels, region_list, region_attr, thr_arr, wc, dist,
                                    random_assign=1)
        construct.append({"order": labels_list(order), "p": p_count,
                          "labels": [int(v) - 1 for v in lab]})
    return {
        "meta": region_meta(),
        "lattice": {"graph": graph_json(w), "x": x.tolist(), "pop": pop.tolist(), "threshold": 8.0,
                    "brute_force_by_p": brute, "pysal": py},
        "columbus": {"graph": graph_json(wc), "x": xc.tolist(), "threshold": 8.0, "pysal": pyc,
                     "construction": construct},
    }


def main():
    dump_json(OUT / "skater.json", skater_fixture())
    dump_json(OUT / "ward.json", ward_fixture())
    dump_json(OUT / "region_kmeans.json", rkm_fixture())
    dump_json(OUT / "objective.json", objective_fixture())
    dump_json(OUT / "azp.json", azp_fixture())
    dump_json(OUT / "maxp.json", maxp_fixture())
    print("wrote", OUT)


if __name__ == "__main__":
    main()
