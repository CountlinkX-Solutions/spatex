"""Generate libpysal.graph kernel-weight oracle fixtures.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_weights_kernel.py

Reads fixtures/datasets/columbus.json; writes fixtures/weights/kernel_*.json.

Fixture layout is that of ``generate_weights_knn.py`` (``points`` plus ``cases`` with ``params``
and the positional ``adjacency``). Kernel ``params`` hold ``kernel``, ``bandwidth`` (a number,
``null`` for libpysal's default rule or ``"adaptive"``), ``k`` (or ``null``), ``taper``,
``decay`` and, for the band fixture, ``threshold``.

Points are in general position (no ties), so KNN candidate sets are unambiguous. The
``kernel_coincident`` fixture holds libpysal's fixed-bandwidth output on duplicated points:
its sparse fast path silently drops coincident pairs, which Spatex keeps (known deviation).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import json  # noqa: E402

import numpy as np  # noqa: E402
import shapely  # noqa: E402
from libpysal.graph import Graph  # noqa: E402

from common import FIXTURES, dump_json  # noqa: E402
from generate_weights_knn import case, fixture, min_threshold  # noqa: E402

KERNELS = (
    "gaussian",
    "triangular",
    "parabolic",
    "bisquare",
    "tricube",
    "cosine",
    "boxcar",
    "exponential",
    "identity",
)


def kcase(points, kernel, bandwidth=None, k=None, taper=True, decay=False, coincident="raise", **extra):
    params = {
        "kernel": kernel,
        "bandwidth": bandwidth,
        "k": k,
        "taper": taper,
        "decay": decay,
        "coincident": coincident,
    }
    graph = Graph.build_kernel(
        points, kernel=kernel, bandwidth=bandwidth, k=k, taper=taper, decay=decay, coplanar=coincident
    )
    tag = "_".join(
        [kernel]
        + [f"bw{bandwidth}" if bandwidth is not None else "bwdefault"]
        + ([f"k{k}"] if k else [])
        + ([] if taper else ["notaper"])
        + (["decay"] if decay else [])
    )
    return case(tag, params, graph, **extra)


def matrix(points, fixed_bw, kernels=KERNELS, *, default_bw=True, taper_off=(), k=5):
    """Fixed, default, global-k and adaptive bandwidths, decay, and optional taper-off cases."""
    cases = []
    for kernel in kernels:
        cases.append(kcase(points, kernel, bandwidth=fixed_bw))
        if default_bw:
            cases.append(kcase(points, kernel))
        cases.append(kcase(points, kernel, k=k))
        cases.append(kcase(points, kernel, bandwidth="adaptive", k=k))
        if kernel != "identity":
            cases.append(kcase(points, kernel, bandwidth=fixed_bw, decay=True))
            cases.append(kcase(points, kernel, bandwidth="adaptive", k=k, decay=True))
        if kernel in taper_off:
            cases.append(kcase(points, kernel, bandwidth=fixed_bw, taper=False))
            if default_bw:
                cases.append(kcase(points, kernel, taper=False))
            cases.append(kcase(points, kernel, bandwidth="adaptive", k=k, taper=False))
            cases.append(kcase(points, kernel, k=k, taper=False))
    return cases


def band_cases(points, threshold):
    cases = []
    for kernel in KERNELS:
        for bandwidth in (threshold * 1.5, None):
            params = {
                "threshold": threshold,
                "kernel": kernel,
                "bandwidth": bandwidth,
                "taper": True,
                "decay": False,
            }
            graph = Graph.build_distance_band(
                points, threshold, binary=False, kernel=kernel, bandwidth=bandwidth
            )
            cases.append(case(f"band_{kernel}_bw{bandwidth}", params, graph))
        params = {"threshold": threshold, "kernel": kernel, "bandwidth": threshold, "taper": True, "decay": True}
        if kernel != "identity":
            graph = Graph.build_distance_band(
                points, threshold, binary=False, kernel=kernel, bandwidth=threshold, decay=True
            )
            cases.append(case(f"band_{kernel}_decay", params, graph))
    return cases


def main():
    out = FIXTURES / "weights"

    columbus = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = shapely.from_wkb([bytes.fromhex(h) for h in columbus["wkb_hex"]])
    centroids = shapely.get_coordinates(shapely.centroid(geoms))
    t_col = min_threshold(centroids)

    rng = np.random.default_rng(20261025)
    random_pts = rng.uniform(0.0, 10.0, size=(200, 2))
    random60 = np.random.default_rng(60).uniform(0.0, 10.0, size=(60, 2))
    line = np.array([[0.0, 0], [1, 0], [2, 0], [3.5, 0], [10, 0]])

    dump_json(out / "kernel_line.json", fixture(
        "kernel_line", "Line x = [0, 1, 2, 3.5, 10]: every kernel, fixed/default/k/adaptive, decay, taper off.",
        line, matrix(line, 1.5, taper_off=KERNELS, k=2)))
    dump_json(out / "kernel_columbus.json", fixture(
        "kernel_columbus", "Kernels on the 49 columbus centroids (general position), taper off for all.",
        centroids, matrix(centroids, 1.5 * t_col, taper_off=KERNELS)))
    dump_json(out / "kernel_random60.json", fixture(
        "kernel_random60", "Kernels on 60 seeded points (seed 60): default bandwidth, taper off for gaussian.",
        random60, matrix(random60, 2.0, taper_off=("gaussian", "exponential"))))
    dump_json(out / "kernel_random200.json", fixture(
        "kernel_random200", "Kernels on 200 seeded points (seed 20261025): fixed bandwidth 1.0, k = 5.",
        random_pts, matrix(random_pts, 1.0, default_bw=False)))
    dump_json(out / "kernel_band.json", fixture(
        "kernel_band", "Distance band with kernel weights on columbus centroids (1.2 x minimal threshold).",
        centroids, band_cases(centroids, 1.2 * t_col)))

    coincident = np.array([[0, 0], [0, 0], [1, 0], [3, 0], [3, 1], [0.5, 2.0]], dtype=float)
    cases = [
        kcase(coincident, "triangular", bandwidth=2.5, quirk="fast_path_drops_coincident"),
        kcase(coincident, "gaussian", bandwidth=2.5, taper=False),
        kcase(coincident, "triangular"),
        kcase(coincident, "triangular", bandwidth="adaptive", k=2, coincident="clique"),
        kcase(coincident, "gaussian", k=2, coincident="clique"),
    ]
    dump_json(out / "kernel_coincident.json", fixture(
        "kernel_coincident",
        "Duplicated points: libpysal's fixed-bandwidth sparse path drops the coincident pair, "
        "the default-bandwidth and clique paths keep it.",
        coincident, cases))


if __name__ == "__main__":
    main()
