"""Generate libpysal 4.15.1 alpha-shape fixtures (`libpysal.cg.alpha_shape`, `alpha_shape_auto`).

Run from the repository root under a memory cap (see test/reference/README.md):

    OPENBLAS_NUM_THREADS=1 prlimit --as=2147483648 <venv>/bin/python -I \\
        test/reference/generate_alpha_shapes.py

Writes test/reference/fixtures/pointpats/alpha_shape.json (fixed alpha) and alpha_shape_auto.json
(automatic alpha, ``step``, radius and bounding circles).

Geometry is stored in a canonical form so Spatex can be compared without GEOS: every polygon is a
shell (clockwise, as GEOS's ``polygonize`` returns it) plus holes (counter-clockwise), each ring
open (no closing vertex) and starting at its lexicographically smallest vertex; polygons (and
holes) are sorted lexicographically by their vertices (touching polygons can share a first vertex). Areas are shapely's.

Point sets are in general position (random), so Qhull and Spatex's triangulation agree. Each alpha
is ``1 / threshold`` with the threshold half-way between two consecutive sorted circumradii, so no
triangle sits on the ``radius < 1 / alpha`` boundary.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pointpats  # noqa: E402
import scipy.spatial as spat  # noqa: E402
import shapely  # noqa: E402
import itertools  # noqa: E402
from libpysal.cg import alpha_shape, alpha_shape_auto  # noqa: E402
from libpysal.cg import alpha_shapes as lps  # noqa: E402
from shapely.geometry import Polygon  # noqa: E402
from shapely.geometry.polygon import orient  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

OUT = FIXTURES / "pointpats"
warnings.simplefilter("ignore")


def meta_as():
    m = meta()
    m["pointpats"] = pointpats.__version__
    m["shapely"] = shapely.__version__
    m["scipy"] = __import__("scipy").__version__
    return m


def ring_out(coords):
    """Open ring starting at the lexicographically smallest vertex (orientation kept)."""
    pts = [tuple(map(float, p)) for p in list(coords)[:-1]]
    k = pts.index(min(pts))
    return [list(p) for p in pts[k:] + pts[:k]]


def poly_out(p):
    """Canonical polygon: clockwise shell, counter-clockwise holes."""
    p = orient(p, sign=-1.0)  # exterior clockwise, interiors counter-clockwise
    holes = sorted(ring_out(r.coords) for r in p.interiors)  # lexicographic, like Spatex
    return {"shell": ring_out(p.exterior.coords), "holes": holes, "area": float(p.area)}


def polys_out(geoms):
    polys = [poly_out(g) for g in geoms if not g.is_empty]
    return sorted(polys, key=lambda q: q["shell"])


def radii_of(xys):
    tri = spat.Delaunay(xys)
    t = xys[tri.simplices]
    return np.sort(lps.r_circumcircle_triangle(t[:, 0, :], t[:, 1, :], t[:, 2, :]))


def thresholds(xys, quantiles):
    """Thresholds half-way between consecutive sorted radii at the given quantiles."""
    r = radii_of(xys)
    out = []
    for q in quantiles:
        k = min(max(int(q * (len(r) - 1)), 0), len(r) - 2)
        out.append(float((r[k] + r[k + 1]) / 2))
    return out, float(r[0]), float(r[-1])


def pointsets():
    rng = np.random.default_rng(108)
    sets = {}
    sets["uniform60"] = rng.random((60, 2))
    sets["uniform200_scaled"] = rng.random((200, 2)) * 100.0 + np.array([500.0, -300.0])
    centers = np.array([[0.0, 0.0], [6.0, 1.0], [2.0, 7.0]])
    sets["clusters150"] = np.vstack([c + rng.normal(0, 0.8, (50, 2)) for c in centers])
    th = rng.uniform(0, 2 * np.pi, 220)
    r = rng.uniform(3.0, 5.0, 220)
    sets["annulus"] = np.c_[r * np.cos(th), r * np.sin(th)]
    th = rng.uniform(0, 2 * np.pi, 220)
    r = rng.uniform(5.0, 7.0, 220)
    ring = np.c_[r * np.cos(th), r * np.sin(th)]
    core = rng.normal(0, 0.5, (30, 2))
    sets["island_in_hole"] = np.vstack([ring, core])
    sets["two_blobs"] = np.vstack(
        [rng.normal(0, 1.0, (60, 2)), rng.normal(0, 1.0, (60, 2)) + np.array([15.0, 3.0])]
    )
    sets["c_shape"] = _c_shape(rng)
    return sets


def _c_shape(rng):
    th = rng.uniform(0.35 * np.pi, 1.65 * np.pi, 200)
    r = rng.uniform(4.0, 6.0, 200)
    return np.c_[r * np.cos(th), r * np.sin(th)]


def fixed():
    cases = []
    quantiles = [0.1, 0.4, 0.7, 0.9, 0.97]
    for name, xys in pointsets().items():
        ths, rmin, rmax = thresholds(xys, quantiles)
        # too small (empty), the quantiles, and larger than every radius (convex hull)
        for label, thr in [("tiny", rmin / 2)] + [(f"q{q}", t) for q, t in zip(quantiles, ths)] + [
            ("hull", rmax * 2)
        ]:
            alpha = 1.0 / thr
            geoms = alpha_shape(xys, alpha)
            cases.append(
                {
                    "name": name,
                    "label": label,
                    "points": xys.tolist(),
                    "alpha": alpha,
                    "polygons": polys_out(geoms),
                }
            )
    # fewer than four points: libpysal returns the convex hull whatever alpha is
    small = {
        "three": np.array([[0.0, 0.0], [4.0, 0.5], [1.0, 3.0]]),
        "three_collinear": np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]),
        "two": np.array([[0.0, 0.0], [1.0, 1.0]]),
        "one": np.array([[0.0, 0.0]]),
    }
    for name, xys in small.items():
        for alpha in (0.01, 5.0):
            g = alpha_shape(xys, alpha)
            geoms = [g] if isinstance(g, Polygon) else list(g.geoms) if hasattr(g, "geoms") else list(g)
            cases.append(
                {
                    "name": name,
                    "label": f"alpha{alpha}",
                    "points": xys.tolist(),
                    "alpha": alpha,
                    "polygons": polys_out(geoms),
                }
            )
    dump_json(OUT / "alpha_shape.json", {"meta": meta_as(), "cases": cases})
    return cases


def circle_centers(poly, radius):
    """libpysal's bounding-circle centres, rotated to start at the smallest shell vertex."""
    assert not poly.exterior.is_ccw  # the sign rule of ``_construct_centers`` assumes clockwise
    centers = lps.construct_bounding_circles(poly, radius)
    pts = [tuple(map(float, p)) for p in list(poly.exterior.coords)[:-1]]
    k = pts.index(min(pts))
    return [list(map(float, c)) for c in centers[k:] + centers[:k]]


def auto_variants(xys, step):
    """Results of ``alpha_shape_auto`` when the circumradii are computed with each of the six
    vertex orders of a triangle.

    The search compares ``radius < 1 / (1 / radius - EPS)``, which depends on the last bit of the
    radius, and Qhull's vertex order is arbitrary, so Spatex (which gets the triangles from spade
    in another order) can only be expected to equal one of these. Distinct results are kept, the
    first being the default order.
    """
    original = lps.r_circumcircle_triangle
    out, seen = [], set()
    try:
        for perm in itertools.permutations(range(3)):

            def patched(a, b, c, perm=perm):
                v = (a, b, c)
                return original(v[perm[0]], v[perm[1]], v[perm[2]])

            lps.r_circumcircle_triangle = patched
            try:
                poly, radius, _ = alpha_shape_auto(xys, step=step, return_circles=True)
            except UnboundLocalError:
                res = {"error": "no valid hull"}
            else:
                shell = poly_out(poly)["shell"]
                res = {
                    "polygon": poly_out(poly),
                    "radius": float(radius),
                    "centers": circle_centers(poly, radius),
                }
                assert not poly.interiors
            key = repr(res.get("polygon", {}).get("shell", res.get("error")))
            if key not in seen:
                seen.add(key)
                out.append(res)
    finally:
        lps.r_circumcircle_triangle = original
    return out


def auto_pointsets():
    sets = {k: (v, (1, 3, 10)) for k, v in pointsets().items()}
    rng = np.random.default_rng(1080)
    for i in range(4):
        sets[f"uniform_seed{i}"] = (rng.random((int(rng.integers(30, 90)), 2)), (1, 2))
    sets["tiny_scale"] = (rng.random((50, 2)) * 0.01, (1, 5))
    sets["five"] = (np.array([[0.0, 0.0], [4.0, 0.3], [4.2, 4.0], [0.1, 3.9], [2.1, 1.9]]), (1,))
    sets["three"] = (np.array([[0.0, 0.0], [4.0, 0.5], [1.0, 3.0]]), (1,))
    sets["step_beyond_end"] = (rng.random((40, 2)), (1000,))
    return sets


def auto():
    cases = []
    for name, (xys, steps) in auto_pointsets().items():
        for step in steps:
            cases.append({"name": name, "step": step, "points": xys.tolist(),
                          "variants": auto_variants(xys, step)})
    dump_json(OUT / "alpha_shape_auto.json", {"meta": meta_as(), "cases": cases})
    return cases


if __name__ == "__main__":
    fixed()
    auto()
    print("done")
