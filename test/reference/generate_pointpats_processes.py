"""Generate pointpats 2.6.0 reference summaries for the normal, cluster_normal and strauss processes.

Run from the repository root under a memory cap (see test/reference/README.md):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_pointpats_processes.py

Writes test/reference/fixtures/pointpats/processes.json.

pointpats' generators draw from numpy's default_rng, so there is no exact oracle. Each case
stores distributional summaries over many pointpats realizations (seeded here with ``rng=``,
which makes the fixture reproducible): the mean and covariance of the points of ``normal``, the
mean nearest-neighbour distance of ``cluster_normal`` (with a CSR baseline) and the mean number of
pairs within ``r`` and the mean nearest-neighbour distance of ``strauss``. The Elixir tests compare
Spatex's own summaries with these inside Monte Carlo bands.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pointpats  # noqa: E402
from pointpats import random as pr  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

from common import FIXTURES, dump_json  # noqa: E402

OUT = FIXTURES / "pointpats"
warnings.simplefilter("ignore")


def nn_mean(p):
    d, _ = cKDTree(p).query(p, k=2)
    return float(d[:, 1].mean())


def close_pairs(p, r):
    return len(cKDTree(p).query_pairs(r))


def moments(reps):
    pts = reps.reshape(-1, 2)
    c = np.cov(pts.T, bias=True)
    return {"mean": pts.mean(axis=0).tolist(), "var": [float(c[0, 0]), float(c[1, 1])],
            "cov_xy": float(c[0, 1])}


def summary(values):
    v = np.asarray(values, float)
    return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)), "n": int(len(v))}


def normal_cases():
    box = np.array([0.0, 0.0, 10.0, 5.0])
    cases = []
    specs = [
        ("scalar_cov", dict(center=np.array([7.0, 3.0]), cov=2.0)),
        ("matrix_cov", dict(center=np.array([5.0, 2.5]), cov=np.array([[2.0, 0.8], [0.8, 2.0]]))),
        ("defaults", dict()),
    ]
    for name, kw in specs:
        reps = pr.normal(box, size=(200, 300), rng=np.random.default_rng(11), **kw)
        cases.append({"name": name, "window": box.tolist(), "n": 200, "realizations": 300,
                      "center": None if "center" not in kw else kw["center"].tolist(),
                      "cov": None if "cov" not in kw else np.asarray(kw["cov"]).tolist(),
                      **moments(reps)})
    return cases


def cluster_normal_cases():
    box = np.array([0.0, 0.0, 100.0, 100.0])
    cases = []
    for n_seeds, cov in [(3, 25.0), (5, 9.0), (4, None)]:
        reps = pr.cluster_normal(box, cov=cov, size=(150, 200), n_seeds=n_seeds,
                                 rng=np.random.default_rng(21))
        csr = pr.poisson(box, size=(150, 200), rng=np.random.default_rng(22))
        cases.append({"window": box.tolist(), "n": 150, "realizations": 200, "n_seeds": n_seeds,
                      "cov": cov, "nn": summary([nn_mean(r) for r in reps]),
                      "csr_nn": summary([nn_mean(r) for r in csr])})
    return cases


def strauss_cases():
    box = np.array([0.0, 0.0, 1.0, 1.0])
    cases = []
    for gamma in [1.0, 0.2, 0.01]:
        reps = pr.strauss(box, size=(40, 20), gamma=gamma, r=0.1, n_iter=2000,
                          rng=np.random.default_rng(31))
        cases.append({"window": box.tolist(), "n": 40, "realizations": 20, "gamma": gamma,
                      "r": 0.1, "n_iter": 2000,
                      "pairs": summary([close_pairs(p, 0.1) for p in reps]),
                      "nn": summary([nn_mean(p) for p in reps])})
    return cases


if __name__ == "__main__":
    dump_json(OUT / "processes.json", {
        "meta": {"pointpats": pointpats.__version__, "numpy": np.__version__},
        "normal": normal_cases(), "cluster_normal": cluster_normal_cases(),
        "strauss": strauss_cases()})
    print("done")
