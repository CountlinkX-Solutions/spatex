"""Generate spreg ML_ErrorRE oracle fixtures (`fixtures/regression/ml_error_re.json`).

Run from the repository root (under a memory cap; the first run downloads NCOVR through libpysal):

    prlimit --as=4294967296 <venv>/bin/python -I test/reference/generate_spreg_ml_error_re.py

Cases (panels stored wide, n x T per variable):

* `ncovr`: spreg's own test setup (NCOVR counties of Kansas, Missouri, Oklahoma and Arkansas,
  row-standardized queen weights, HR70/80/90 on RD and PS), without and with time effects.
* `rook4x6`: a seeded synthetic panel on the binary rook 4 x 6 lattice (symmetric W, simple
  spectrum, lam in the valid range |lam| < 1 / lambda_max).
* `columbus`: a seeded synthetic panel on the row-standardized Columbus queen graph (non-symmetric
  W, real simple spectrum).
* `rook5x5`: the binary rook 5 x 5 lattice (symmetric W with repeated eigenvalues). spreg's
  `la.eig` returns a non-orthogonal basis inside a degenerate eigenspace, so its objective is not
  the model's likelihood; the case stores spreg's values next to the exact optimum of the exact
  (eigh) objective.

Every case stores spreg's results, spreg's concentrated objective at three probe points, and the
exact optimum of spreg's own objective (Nelder-Mead polish of spreg's L-BFGS-B result). The script
asserts that spreg's objective is invariant to eigenvector sign flips on every simple-spectrum
case (spreg's `ML_ErrorRE` only depends on `V diag(1/c) V'`).

ML_ErrorRE is the maintained class (`spreg/panel.py`); `spreg/panel_re.py::Panel_RE_Error` is
deprecated and its symmetric branch uses `symmetrize(w)`, which mixes the binary and the
row-standardized matrix, so it is not ported.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import scipy.linalg as la  # noqa: E402
import spreg  # noqa: E402
from libpysal.weights import Queen, lat2W, w_subset  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from shapely import wkb  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def sc(v):
    return float(np.real(np.asarray(v)).reshape(-1)[0])


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def graph_of(w):
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    return {"focal": coo.row[order].tolist(), "neighbor": coo.col[order].tolist(),
            "weight": coo.data[order].tolist()}


def objective_of(reg, w):
    """spreg's concentrated objective (`err_c_loglik_ord`) with the data of a fitted model."""
    n, t, k = w.n, reg.t, reg.x.shape[1]
    big_w, wsp = w.full()[0], w.sparse
    evals, evecs = la.eig(big_w)
    yr = reg.y.reshape(t, n)
    ylag = (yr @ wsp.T).reshape(-1, 1)
    xr = reg.x.reshape(t, n, k)
    xlag = np.zeros_like(reg.x)
    for i in range(k):
        xlag[:, i] = (xr[:, :, i] @ wsp.T).reshape(-1)
    ym, xm = yr.mean(axis=0).reshape(n, 1), xr.mean(axis=0)
    ident = np.identity(n)

    def f(lam, ph, vecs=evecs, ev=evals):
        val = reg.err_c_loglik_ord((lam, ph), ev, vecs, n, t, reg.y, reg.x, ylag, xlag, ym, xm,
                                   ident, wsp)
        return float(np.real(np.asarray(val)).reshape(-1)[0])

    return f, evals, evecs


def case_of(w, y, x, name_x, te, label):
    reg = spreg.ML_ErrorRE(y, x, w, time_effects=te, name_y="y", name_x=name_x)
    big_w = w.full()[0]
    f, evals, evecs = objective_of(reg, w)
    n, t = w.n, reg.t
    ph = float(np.sqrt(sc(reg.phi)))
    lam = sc(reg.lam)
    gaps = np.diff(np.sort(evals.real))
    simple = bool(gaps.min() > 1e-8 and np.abs(evals.imag).max() < 1e-12)
    orth = float(np.abs(evecs.conj().T @ evecs - np.eye(n)).max())
    probes = []
    rng = np.random.default_rng(5)
    for pl, pp in ((0.05, 0.5), (-0.05, 1.5), (0.12, 0.2)):
        rec = {"lam": pl, "phi_opt": pp, "objective": f(pl, pp)}
        if simple:
            flipped = evecs * rng.choice([-1.0, 1.0], size=n)
            assert abs(f(pl, pp, flipped) - rec["objective"]) < 1e-8 * max(1.0, abs(rec["objective"])), (
                label, "objective is not invariant to eigenvector signs")
        probes.append(rec)
    # Exact optimum: polish spreg's L-BFGS-B result with spreg's own objective. With a repeated
    # eigenvalue spreg's la.eig basis is not orthogonal and its objective is not the likelihood,
    # so the exact objective uses the orthonormal eigh basis instead (`exact_objective`).
    symmetric = bool(np.allclose(big_w, big_w.T))
    exact_objective = "la.eig"
    obj = f
    if not simple:
        assert symmetric
        ev_h, vec_h = la.eigh(big_w)
        obj = lambda a, b: f(a, b, vec_h, ev_h)  # noqa: E731
        exact_objective = "eigh"
    polish = minimize(lambda p: obj(p[0], p[1]), [lam, ph], method="Nelder-Mead",
                      bounds=((-0.999, 0.999), (1e-8, 1e9)),
                      options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 400, "maxfev": 800})
    fun_spreg = -(sc(reg.logll) + n * t / 2.0 * np.log(2 * np.pi) + n * t / 2.0)
    return {
        "label": label, "params": {"time_effects": te}, "name_x": list(reg.name_x),
        "betas": flat(np.real(reg.betas)), "vm": flat(reg.vm), "lam": lam, "phi": sc(reg.phi),
        "phi_opt": ph, "sig2": sc(reg.sig2),
        "sig2_u": sc(reg.sig2_u), "logll": sc(reg.logll), "aic": sc(reg.aic),
        "schwarz": sc(reg.schwarz), "u": flat(np.real(reg.u)),
        "objective_at_spreg": fun_spreg, "probes": probes,
        "exact": {"objective_basis": exact_objective, "lam": float(polish.x[0]), "phi_opt": float(polish.x[1]), "objective": float(polish.fun)},
        "spectrum": {"simple": simple, "min_gap": float(gaps.min()),
                     "max_imag": float(np.abs(evals.imag).max()),
                     "eigenvector_orthogonality_error": orth,
                     "symmetric": symmetric},
    }


def synthetic(w, n, t, lam, seed, sigma_u):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(n, t)), rng.normal(2.0, 1.0, size=(n, t))
    mu = rng.normal(0.0, sigma_u, size=(n, 1))
    eps = rng.normal(0.0, 1.0, size=(n, t))
    u = np.linalg.inv(np.eye(n) - lam * w.full()[0]) @ eps
    y = 3.0 + 1.5 * x1 - 0.8 * x2 + mu + u
    return y, x1, x2


def synthetic_fixture(label, w, t, lam, seed, sigma_u, time_effects):
    n = w.n
    y, x1, x2 = synthetic(w, n, t, lam, seed, sigma_u)
    x = np.hstack((x1, x2))
    return {"n": n, "t": t, "y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist(),
            "graph": graph_of(w),
            "cases": [case_of(w, y, x, ["x1", "x2"], te, f"{label} te={te}") for te in time_effects]}


def main():
    out = {"name": "ml_error_re", "description": "spreg ML_ErrorRE (panel.py)", "meta": meta()}

    nat = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    w_full = Queen.from_dataframe(nat, use_index=False)
    sub = np.where(nat.STATE_NAME.isin(["Kansas", "Missouri", "Oklahoma", "Arkansas"]))[0].tolist()
    w = w_subset(w_full, sub)
    w.transform = "r"
    assert w.id_order == sub
    y = nat[["HR70", "HR80", "HR90"]].values[sub]
    rd = nat[["RD70", "RD80", "RD90"]].values[sub]
    ps = nat[["PS70", "PS80", "PS90"]].values[sub]
    out["ncovr"] = {"n": w.n, "t": 3, "y": y.tolist(), "rd": rd.tolist(), "ps": ps.tolist(),
                    "graph": graph_of(w),
                    "cases": [case_of(w, y, np.hstack((rd, ps)), ["RD", "PS"], te, f"ncovr te={te}")
                              for te in (False, True)]}

    rook = lat2W(4, 6, rook=True)
    out["rook4x6"] = synthetic_fixture("rook4x6", rook, 5, 0.15, 41, 1.0, (False, True))

    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    wc = Queen.from_iterable(gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]]))
    wc.transform = "r"
    out["columbus"] = synthetic_fixture("columbus", wc, 5, 0.3, 43, 1.0, (False,))

    rook55 = lat2W(5, 5, rook=True)
    out["rook5x5"] = synthetic_fixture("rook5x5", rook55, 5, 0.1, 47, 1.0, (False,))

    dump_json(FIXTURES / "regression" / "ml_error_re.json", out)


if __name__ == "__main__":
    main()
