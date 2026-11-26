"""Generate the GWR hat-matrix, scipy-search and `X_glob` oracle fixtures
(`fixtures/regression/gwr_hat.json`, `gwr_search.json`).

Run from the repository root (under a memory cap):

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_gwr_search.py

Ground truth is mgwr 2.2.1.

* `gwr_hat.json` - `GWR(..., hat_matrix=True).fit().S` (the full n x n hat matrix) for Georgia
  (adaptive bisquare, fixed gaussian, global bandwidth), the seeded spherical sample (with and
  without constant) and the seeded Poisson (offset) and Binomial samples, each with `tr_s`.
* `gwr_search.json` - `Sel_BW.search(search_method='scipy')` (`scipy.optimize.minimize_scalar`,
  bounded) with the selected bandwidth, criterion value and the `OptimizeResult` fields, and
  `Sel_BW(..., X_glob=...)`, which only changes the default search bounds (`_init_section`).
"""

import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import mgwr  # noqa: E402
from mgwr import sel_bw  # noqa: E402
from mgwr.gwr import GWR  # noqa: E402
from mgwr.sel_bw import Sel_BW  # noqa: E402
from spglm.family import Binomial, Poisson  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402
from generate_gwr_predict import FAMILIES, georgia, glm_data, synthetic  # noqa: E402

warnings.simplefilter("ignore")


def _scalar(fn):
    """mgwr's criteria return shape-(1,) arrays. With NumPy 2.5 `int(array([x]))` raises, so
    mgwr's scipy search only runs for `CV` (a scalar). Returning plain floats restores the
    behavior of older NumPy, where such arrays were converted by `int(bw)`; the arithmetic of
    the search is identical."""
    return lambda *a, **k: float(np.asarray(fn(*a, **k)).ravel()[0])


@contextmanager
def scalar_criteria():
    saved = dict(sel_bw.getDiag)
    sel_bw.getDiag.update({k: _scalar(v) for k, v in saved.items()})
    try:
        yield
    finally:
        sel_bw.getDiag.update(saved)


def hat_case(coords, y, x, bw, family="gaussian", offset=None, **kw):
    fam = FAMILIES[family]
    extra = {} if fam is None else {"family": fam()}
    off = None if offset is None else offset.reshape(-1, 1)
    r = GWR(coords, y.reshape(-1, 1), x, bw, hat_matrix=True, offset=off,
            n_jobs=1, **extra, **kw).fit()
    return {
        "options": {**kw, "bw": float(bw), "family": family},
        "s": np.asarray(r.S, dtype=float),
        "tr_s": float(r.tr_S),
        "influ": np.asarray(r.influ, dtype=float).reshape(-1),
    }


def hat_fixture():
    gc, gy, gx = georgia()
    sc, sy, sx = synthetic()
    pc, px, off, counts, yb = glm_data()
    c = {}
    c["georgia_adaptive"] = {"dataset": "georgia", **hat_case(gc, gy, gx, 60)}
    c["georgia_fixed_gaussian"] = {
        "dataset": "georgia",
        **hat_case(gc, gy, gx, 90000.0, kernel="gaussian", fixed=True)}
    c["georgia_global"] = {"dataset": "georgia", **hat_case(gc, gy, gx, np.inf)}
    c["synthetic_spherical"] = {
        "dataset": "synthetic",
        **hat_case(sc, sy, sx, 400.0, kernel="gaussian", fixed=True, spherical=True)}
    c["synthetic_no_constant"] = {
        "dataset": "synthetic", **hat_case(sc, sy, sx, 50, constant=False)}
    c["poisson_offset"] = {
        "dataset": "poisson", **hat_case(pc, counts, px, 30, family="poisson", offset=off)}
    c["binomial_fixed"] = {
        "dataset": "binomial",
        **hat_case(pc, yb, px, 4.0, family="binomial", kernel="gaussian", fixed=True)}
    return {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "datasets": {
            "georgia": {"coords": gc, "y": gy, "x": gx.T},
            "synthetic": {"coords": sc, "y": sy, "x": sx.T},
            "poisson": {"coords": pc, "y": counts, "x": px.T, "offset": off},
            "binomial": {"coords": pc, "y": yb, "x": px.T},
        },
        "cases": c,
    }


def search_case(dataset, coords, y, x, x_glob=None, family="gaussian", offset=None, **kw):
    search = {k: kw.pop(k) for k in ("criterion", "bw_min", "bw_max", "search_method")
              if k in kw}
    fam = FAMILIES[family]
    extra = {} if fam is None else {"family": fam()}
    off = None if offset is None else offset.reshape(-1, 1)
    sel = Sel_BW(coords, y.reshape(-1, 1), x, X_glob=x_glob, offset=off, n_jobs=1,
                 **extra, **kw)
    if search.get("search_method") == "scipy":
        with scalar_criteria():
            bw = sel.search(**search)
    else:
        bw = sel.search(**search)
    out = {
        "dataset": dataset,
        "options": {**kw, **search, "family": family},
        "bw": float(bw),
        "score": float(np.asarray(sel.bw[1]).ravel()[0]),
        "history": [[float(b), float(np.asarray(s).ravel()[0])] for b, s in sel.sel_hist],
        "bw_min": float(sel.bw_min) if sel.bw_min is not None else None,
        "bw_max": float(sel.bw_max) if sel.bw_max is not None else None,
    }
    r = getattr(sel, "_optimize_result", None)
    if r is not None:
        out["nfev"] = int(r.nfev)
        out["nit"] = int(r.nit)
        out["success"] = bool(r.success)
        out["status"] = int(r.status)
    return out


def search_fixture():
    gc, gy, gx = georgia()
    sc, sy, sx = synthetic()
    pc, px, off, counts, yb = glm_data()
    rng = np.random.default_rng(20261112)
    glob1 = rng.normal(size=(len(gy), 1))
    glob3 = rng.normal(size=(len(gy), 3))
    scipy = {}
    scipy["georgia_adaptive_aicc"] = search_case("georgia", gc, gy, gx, search_method="scipy")
    for crit in ("AIC", "BIC", "CV"):
        scipy[f"georgia_adaptive_{crit.lower()}"] = search_case("georgia", gc, gy, gx, search_method="scipy", criterion=crit)
    scipy["georgia_adaptive_exponential"] = search_case("georgia", gc, gy, gx, search_method="scipy", kernel="exponential")
    scipy["georgia_fixed_bisquare"] = search_case("georgia", gc, gy, gx, search_method="scipy", fixed=True)
    scipy["georgia_fixed_gaussian"] = search_case("georgia", gc, gy, gx, search_method="scipy", fixed=True, kernel="gaussian")
    scipy["georgia_adaptive_bounds"] = search_case("georgia", gc, gy, gx, search_method="scipy", bw_min=60, bw_max=120)
    scipy["georgia_fixed_bounds"] = search_case("georgia", gc, gy, gx, search_method="scipy", fixed=True, kernel="gaussian",
        bw_min=30000.0, bw_max=300000.0)
    scipy["synthetic_spherical_fixed"] = search_case("synthetic", sc, sy, sx, search_method="scipy", fixed=True, kernel="gaussian", spherical=True)
    scipy["synthetic_no_constant"] = search_case("synthetic", sc, sy, sx, search_method="scipy", constant=False)
    scipy["poisson_adaptive"] = search_case("poisson", pc, counts, px, search_method="scipy", family="poisson", offset=off)
    scipy["binomial_fixed"] = search_case("binomial", pc, yb, px, search_method="scipy", family="binomial", fixed=True,
        kernel="gaussian")
    xglob = {}
    xglob["georgia_none"] = search_case("georgia", gc, gy, gx)
    xglob["georgia_one_column"] = search_case("georgia", gc, gy, gx, x_glob=glob1)
    xglob["georgia_three_columns"] = search_case("georgia", gc, gy, gx, x_glob=glob3)
    xglob["georgia_one_column_scipy"] = search_case("georgia", gc, gy, gx, x_glob=glob1, search_method="scipy")
    xglob["georgia_one_column_no_constant"] = search_case("georgia", gc, gy, gx, x_glob=glob1, constant=False)
    xglob["georgia_one_column_bounds"] = search_case("georgia", gc, gy, gx, x_glob=glob1, bw_min=60, bw_max=120)
    xglob["georgia_fixed_one_column"] = search_case("georgia", gc, gy, gx, x_glob=glob1, fixed=True, kernel="gaussian")
    return {
        "meta": {**meta(), "mgwr": mgwr.__version__},
        "datasets": {
            "georgia": {"coords": gc, "y": gy, "x": gx.T},
            "synthetic": {"coords": sc, "y": sy, "x": sx.T},
            "poisson": {"coords": pc, "y": counts, "x": px.T, "offset": off},
            "binomial": {"coords": pc, "y": yb, "x": px.T},
        },
        "x_glob": {"one": glob1.T, "three": glob3.T},
        "scipy": scipy,
        "x_glob_cases": xglob,
    }


if __name__ == "__main__":
    which = sys.argv[1:] or ["hat", "search"]
    if "hat" in which:
        dump_json(FIXTURES / "regression" / "gwr_hat.json", hat_fixture())
    if "search" in which:
        dump_json(FIXTURES / "regression" / "gwr_search.json", search_fixture())
