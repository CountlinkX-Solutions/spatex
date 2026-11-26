"""Generate spopt 0.8.0 oracle fixtures for Spenc (spatially encouraged spectral clustering).

Run from the repository root under a memory cap (see test/reference/README.md):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_region_spenc.py

Writes test/reference/fixtures/region/spenc.json.

What is captured (all from spopt's own code, by wrapping it):

* ``affinity``: ``SPENC.affinity_matrix_`` (attribute affinity multiplied elementwise by ``W``).
* ``embedding`` / ``eigenvalues``: ``SPENC.embedding_`` (``n x k``, per-vector sign normalised by
  spopt) and the ``k`` eigenvalues ``scipy.sparse.linalg.eigsh`` returned for ``-L`` (ascending).
* ``labels`` of the deterministic ``assign_labels="hierarchical"`` cases. spopt 0.8.0 cannot run
  that mode (``TypeError``: it passes ``floor=`` to ``_make_hierarchical_cut``); the oracle is its
  code with that keyword accepted and ignored.
* ``by_start_row``: ``sklearn.cluster._spectral.discretize`` is a pure function of the one row it
  draws with ``random_state.randint(n)``; the labels for every possible row are captured by
  feeding it a ``RandomState`` that returns that row.
* ``planted`` / ``columbus_runs``: label sets of ``Spenc.solve`` over many ``random_state`` values
  (discretize and kmeans) for the statistical tests, with the planted partition.
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402
import sklearn  # noqa: E402
import spopt  # noqa: E402
import spopt.region.spenclib.abstracts as abstracts  # noqa: E402
from sklearn.cluster._spectral import discretize  # noqa: E402
from spopt.region import Spenc  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

OUT = FIXTURES / "region" / "spenc.json"
warnings.simplefilter("ignore")
ATTRS = ["CRIME", "INC", "HOVAL"]


def sorted_w(neighbors, weights=None):
    n = len(neighbors)
    nb = {i: sorted(int(j) for j in neighbors[i]) for i in range(n)}
    wt = None
    if weights is not None:
        wt = {i: [weights(i, j) for j in nb[i]] for i in range(n)}
    return libpysal.weights.W(nb, weights=wt, ids=list(range(n)))


def lattice(r, c):
    w = libpysal.weights.lat2W(r, c, rook=True)
    return sorted_w({i: w.neighbors[i] for i in range(w.n)})


def graph_json(w):
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    return {
        "n": w.n,
        "focal": [int(v) for v in coo.row[order]],
        "neighbor": [int(v) for v in coo.col[order]],
        "weight": [float(v) for v in coo.data[order]],
    }


class FixedRow(np.random.RandomState):
    """A RandomState whose ``randint`` always returns the same row."""

    def __init__(self, row):
        super().__init__(0)
        self.row = row

    def randint(self, *args, **kwargs):
        return self.row


def _patched_cut(orig):
    """spopt 0.8.0's hierarchical mode passes ``floor=`` to ``_make_hierarchical_cut``, which does
    not accept it: every hierarchical fit raises ``TypeError``. The oracle for that mode is spopt's
    own code with that one keyword accepted and ignored (the function never used it)."""

    def cut(self, *a, floor=0, **k):
        return orig(self, *a, **k)

    return cut


def fit(w, x, fit_kwargs=None, **params):
    """Run ``Spenc.solve`` capturing the SPENC model and the eigsh eigenvalues."""
    ev_seen = []
    orig = spla.eigsh

    def eigsh(*a, **k):
        res = orig(*a, **k)
        ev_seen.append(np.asarray(res[0]).copy())
        return res

    spla.eigsh = eigsh
    created = []
    orig_init = abstracts.SPENC.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        created.append(self)

    abstracts.SPENC.__init__ = init
    orig_cut = abstracts.SPENC._make_hierarchical_cut
    abstracts.SPENC._make_hierarchical_cut = _patched_cut(orig_cut)
    try:
        gdf = pd.DataFrame(x, columns=[f"a{j}" for j in range(x.shape[1])])
        model = Spenc(gdf, w, list(gdf.columns), **params)
        model.solve(fit_kwargs or {})
    finally:
        spla.eigsh = orig
        abstracts.SPENC.__init__ = orig_init
        abstracts.SPENC._make_hierarchical_cut = orig_cut
    return model, created[0], ev_seen


def case(name, w, x, fit_kwargs=None, **params):
    shown = {k: ("inf" if v == np.inf else v) for k, v in params.items()}
    try:
        model, spenc, ev_seen = fit(w, x, fit_kwargs, random_state=0, **params)
    except Exception as e:  # spopt itself fails (recorded: Spatex must raise too)
        return {
            "name": name,
            "params": shown,
            "fit_kwargs": fit_kwargs or {},
            "error": type(e).__name__,
        }
    out = {
        "name": name,
        "params": shown,
        "fit_kwargs": fit_kwargs or {},
        "labels": [int(v) for v in model.labels_],
    }
    hier = params.get("assign_labels") == "hierarchical" or np.isinf(params["n_clusters"])
    if not hier:
        aff = spenc.affinity_matrix_
        aff = aff.toarray() if hasattr(aff, "toarray") else np.asarray(aff)
        out["affinity"] = aff.tolist()
        out["embedding"] = spenc.embedding_.tolist()
        out["eigenvalues"] = [float(v) for v in ev_seen[0]]
        if params.get("assign_labels", "discretize") == "discretize":
            out["by_start_row"] = [
                [int(v) for v in discretize(spenc.embedding_.copy(), random_state=FixedRow(r))]
                for r in range(x.shape[0])
            ]
    return out


def runs(w, x, n_runs, **params):
    """Label sets of repeated runs, as ``Spenc.solve`` returns them."""
    out = []
    for seed in range(n_runs):
        model, _, _ = fit(w, x, None, random_state=seed, **params)
        out.append([int(v) for v in model.labels_])
    return out


def main():
    gdf = gpd.read_file(libpysal.examples.get_path("columbus.shp"))
    w0 = libpysal.weights.Queen.from_dataframe(gdf, use_index=False)
    wc = sorted_w({i: w0.neighbors[i] for i in range(w0.n)})
    xc = gdf[ATTRS].to_numpy(float)
    zc = (xc - xc.mean(axis=0)) / xc.std(axis=0)
    pos = xc / xc.max(axis=0)

    # Symmetric non-binary weights: the elementwise product with W must use them.
    ww = sorted_w({i: w0.neighbors[i] for i in range(w0.n)}, lambda i, j: 1.0 + (i + j) % 3)

    cases = [
        case("rbf_k3", wc, zc, n_clusters=3, gamma=0.5),
        case("rbf_k4", wc, zc, n_clusters=4, gamma=0.5),
        case("rbf_k5", wc, zc, n_clusters=5, gamma=1.0),
        case("rbf_weighted_k4", ww, zc, n_clusters=4, gamma=0.5),
        case("laplacian_k4", wc, zc, n_clusters=4, affinity="laplacian", gamma=0.5),
        case("linear_k3", wc, pos, n_clusters=3, affinity="linear"),
        case(
            "polynomial_k3",
            wc,
            pos,
            n_clusters=3,
            affinity="polynomial",
            gamma=0.5,
            degree=2,
            coef0=1,
        ),
        case("sigmoid_k3", wc, pos, n_clusters=3, affinity="sigmoid", gamma=0.5, coef0=1),
        case("cosine_k4", wc, pos, n_clusters=4, affinity="cosine"),
        case(
            "nearest_neighbors_k4",
            wc,
            zc,
            n_clusters=4,
            affinity="nearest_neighbors",
            n_neighbors=6,
        ),
        case("rbf_kmeans_k4", wc, zc, n_clusters=4, gamma=0.5, assign_labels="kmeans"),
        case("hierarchical_k3", wc, zc, n_clusters=3, gamma=0.5, assign_labels="hierarchical"),
        case("hierarchical_k4", wc, zc, n_clusters=4, gamma=0.5, assign_labels="hierarchical"),
        case("hierarchical_k6", wc, zc, n_clusters=6, gamma=0.5, assign_labels="hierarchical"),
        case(
            "hierarchical_inf_floor5",
            wc,
            zc,
            fit_kwargs={"floor": 5},
            n_clusters=np.inf,
            gamma=0.5,
        ),
        # spopt raises here (eigsh with k >= n on a one-area cut; no accepted cut at all).
        case("hierarchical_k8_error", wc, zc, n_clusters=8, gamma=0.5, assign_labels="hierarchical"),
        case(
            "hierarchical_inf_floor8_error",
            wc,
            zc,
            fit_kwargs={"floor": 8},
            n_clusters=np.inf,
            gamma=0.5,
        ),
        case(
            "hierarchical_k5_grid20",
            wc,
            zc,
            fit_kwargs={"grid_resolution": 20},
            n_clusters=5,
            gamma=1.0,
            assign_labels="hierarchical",
        ),
    ]
    # Precomputed: the "attributes" are the affinity itself (W is not applied).
    d2 = ((zc[:, None, :] - zc[None, :, :]) ** 2).sum(axis=2)
    pre = np.exp(-0.5 * d2) * wc.sparse.toarray()
    cases.append(case("precomputed_k3", wc, pre, n_clusters=3, affinity="precomputed"))

    # Planted partition: four 4x4 blocks on an 8x8 rook lattice, well separated in attributes.
    wp = lattice(8, 8)
    block = np.array([(i // 8 // 4) * 2 + (i % 8) // 4 for i in range(64)])
    rng = np.random.default_rng(5)
    centers = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0], [4.0, 4.0]])
    xp = centers[block] + rng.normal(scale=0.4, size=(64, 2))
    planted = {
        "graph": graph_json(wp),
        "x": xp.tolist(),
        "planted": [int(b) for b in block],
        "n_clusters": 4,
        "gamma": 0.5,
        "discretize_runs": runs(wp, xp, 100, n_clusters=4, gamma=0.5),
        "kmeans_runs": runs(wp, xp, 100, n_clusters=4, gamma=0.5, assign_labels="kmeans"),
    }

    # Columbus label sets for the distribution comparison (no planted partition).
    columbus_runs = {
        "n_clusters": 4,
        "gamma": 0.5,
        "discretize_runs": runs(wc, zc, 100, n_clusters=4, gamma=0.5),
        "kmeans_runs": runs(wc, zc, 200, n_clusters=4, gamma=0.5, assign_labels="kmeans"),
    }

    m = meta()
    m.update(spopt=spopt.__version__, scikit_learn=sklearn.__version__, scipy=scipy.__version__)
    dump_json(
        OUT,
        {
            "meta": m,
            "columbus": {
                "graph": graph_json(wc),
                "weighted_graph": graph_json(ww),
                "x": zc.tolist(),
                "x_positive": pos.tolist(),
                "precomputed": pre.tolist(),
            },
            "cases": cases,
            "planted": planted,
            "columbus_runs": columbus_runs,
        },
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
