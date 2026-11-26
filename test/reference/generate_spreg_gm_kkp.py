"""Generate spreg GM_ErrorRE / GM_KKP oracle fixtures (`fixtures/regression/gm_kkp.json`).

Run from the repository root (under a memory cap; the first run downloads NCOVR through libpysal):

    prlimit --as=4294967296 <venv>/bin/python -I test/reference/generate_spreg_gm_kkp.py

`GM_KKP` (`spreg/sp_panels.py`) is deprecated in spreg 1.9.1 in favor of `GM_ErrorRE`
(`spreg/panel.py`): same estimator (the moments are the same equations, computed as a matrix
product in one and as a T x N array in the other), `GM_ErrorRE` adds `time_effects` and `slx_lags`.
The target class is `GM_ErrorRE`; the script asserts that `GM_KKP` agrees with it on every case
without time effects (to the optimizer noise, below 1e-5) and records the largest difference.

Cases (panels stored wide, n x T per variable):

* `ncovr`: spreg's own test setup (NCOVR counties of Kansas, Missouri, Oklahoma and Arkansas, a
  subset of the 3085 counties spreg's tests use; row-standardized queen weights, HR70/80/90 on RD
  and PS), `full_weights` False and True, without and with time effects.
* `columbus`: a seeded synthetic panel on the row-standardized Columbus queen graph.
* `rook5x5`: a seeded synthetic panel on the binary rook 5 x 5 lattice (the GM estimator never
  uses the spectrum, so repeated eigenvalues and binary weights are fine).

Every case stores spreg's results and the outcome of each of its two `optim_moments` calls (the
first-step `(lambda, sigma_v)` and the weighted `(lambda, sigma_v, sigma_1)`), so the formulas can
be checked at spreg's own parameters. spreg minimizes with L-BFGS-B, which stops early; the
`exact` block repeats the whole fit with every optimizer result replaced by the exact minimizer of
the same objective in spreg's basin (variances profiled out by non-negative least squares, bounded
Brent on lambda around spreg's result), which is what Spatex computes. The script asserts that this
minimizer is also the global minimum of the profile on every case (a 0.001 grid over [-0.99, 0.99]).
"""

import copy
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import libpysal  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
import spreg.panel as spanel  # noqa: E402
import spreg.sp_panels as sp_panels  # noqa: E402
from libpysal.weights import Queen, lat2W, w_subset  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from shapely import wkb  # noqa: E402
from spreg.utils import foptim_par  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")
ORIGINAL = spanel.optim_moments


def flat(a):
    return np.asarray(a, dtype=float).flatten().tolist()


def graph_of(w):
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    return {"focal": coo.row[order].tolist(), "neighbor": coo.col[order].tolist(),
            "weight": coo.data[order].tolist()}


def objective_of(moments_in, vcx):
    """spreg's own objective in `optim_moments` (weighting and scaling included)."""
    moments = copy.deepcopy(moments_in)
    if vcx.any():
        ec = np.transpose(np.linalg.cholesky(np.linalg.inv(vcx)))
        moments[0], moments[1] = ec @ moments_in[0], ec @ moments_in[1]
    scale = np.min([[np.min(moments[0]), np.min(moments[1])]])
    moments[0], moments[1] = moments[0] / scale, moments[1] / scale
    four = moments[0].shape[1] == 4

    def f(p):
        par = [p[0], p[0] ** 2.0] + list(p[1:])
        return float(np.asarray(foptim_par(np.array([par]).T, moments)).reshape(-1)[0])

    bounds = [(-0.99, 0.99), (0.0, None)] + ([(0.0, None)] if four else [])
    return f, bounds


def profile_of(moments_in, vcx):
    """`lambda -> (objective, variances)` of spreg's objective with the variance parameters
    profiled out by non-negative least squares."""
    from scipy.optimize import nnls

    moments = copy.deepcopy(moments_in)
    if vcx.any():
        ec = np.transpose(np.linalg.cholesky(np.linalg.inv(vcx)))
        moments[0], moments[1] = ec @ moments_in[0], ec @ moments_in[1]
    scale = np.min([[np.min(moments[0]), np.min(moments[1])]])
    gm, g = moments[0] / scale, (moments[1] / scale).reshape(-1)

    def profile(lam):
        r = g - gm[:, 0] * lam - gm[:, 1] * lam * lam
        s, rnorm = nnls(gm[:, 2:], r)
        return rnorm ** 2, s

    return profile


def polish_local(moments_in, vcx, point):
    """The exact minimizer next to spreg's result: bounded Brent on the profile within 0.02 of
    spreg's lambda (the result must be interior)."""
    from scipy.optimize import minimize_scalar

    profile = profile_of(moments_in, vcx)
    lo, hi = max(point[0] - 0.02, -0.99), min(point[0] + 0.02, 0.99)
    res = minimize_scalar(lambda x: profile(x)[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-13})
    assert lo < res.x < hi, "the local minimum is not interior"
    return np.array([res.x] + profile(res.x)[1].tolist())


def global_min(moments_in, vcx):
    """The minimum over lambda of the profile: a 0.001 grid, then a local refinement."""
    from scipy.optimize import minimize_scalar

    profile = profile_of(moments_in, vcx)
    grid = np.arange(-990, 991) / 1000.0
    i = int(np.argmin([profile(x)[0] for x in grid]))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda x: profile(x)[0], bounds=(lo, hi), method="bounded",
                          options={"xatol": 1e-13})
    return {"lam": float(res.x), "sigma": profile(res.x)[1].tolist(), "objective": float(res.fun)}


def run(y, x, w, full_weights, te, polish):
    """Fits `GM_ErrorRE` with `optim_moments` recording (and optionally polishing) its results."""
    calls = []

    def recorded(moments_in, vcX=np.array([0]), all_par=False, start=None, hard_bound=False):
        point = ORIGINAL(moments_in, vcX=vcX, all_par=True, start=start, hard_bound=hard_bound)
        f, bounds = objective_of(moments_in, vcX)
        rec = {"spreg": point.tolist(), "objective_at_spreg": f(point)}
        if polish:
            point = polish_local(moments_in, vcX, point)
            rec["exact"] = point.tolist()
            rec["objective_at_exact"] = f(point)
            # The profile has a single basin on every case here: the local minimizer is global.
            assert abs(global_min(moments_in, vcX)["lam"] - point[0]) < 1e-6, "second basin"
        calls.append(rec)

        return point if all_par else point[0]

    spanel.optim_moments = recorded
    try:
        reg = spreg.GM_ErrorRE(y, x, w, full_weights=full_weights, time_effects=te,
                               name_y="y", name_x=NAMES)
    finally:
        spanel.optim_moments = ORIGINAL
    return reg, calls


def result_of(reg, k):
    b = np.asarray(reg.betas).flatten()
    return {"betas": b.tolist(), "vm": flat(reg.vm), "std_err": [float(np.sqrt(reg.vm[i, i])) for i in range(k)],
            "z_stat": [[float(a), float(c)] for a, c in reg.z_stat],
            "pr2": float(reg.pr2), "lam": float(b[k]), "sig2_v": float(b[k + 1]),
            "sig2_1": float(b[k + 2])}


def residuals_of(reg):
    return {"u": flat(reg.u), "e_filtered": flat(reg.e_filtered)}


def case_of(w, y, x, te, full_weights, label):
    global NAMES
    reg, calls = run(y, x, w, full_weights, te, polish=False)
    exact_reg, exact_calls = run(y, x, w, full_weights, te, polish=True)
    k = len(reg.name_x) - 3
    rec = {"label": label, "params": {"time_effects": te, "full_weights": full_weights},
           "name_x": [s.strip() for s in reg.name_x], "k": k,
           "spreg": {**result_of(reg, k), **residuals_of(reg)}, "calls": calls,
           "exact": result_of(exact_reg, k), "exact_calls": exact_calls}
    # GM_KKP is the deprecated twin of GM_ErrorRE (no time effects there).
    if not te:
        kkp = sp_panels.GM_KKP(y, x, w, full_weights=full_weights, name_y=["y"], name_x=NAMES)
        diff = float(np.abs(np.asarray(kkp.betas) - np.asarray(reg.betas)).max())
        assert diff < 1e-5, (label, "GM_KKP differs from GM_ErrorRE", diff)
        rec["kkp_max_diff"] = diff
    return rec


def synthetic(w, n, t, lam, seed, sigma_u):
    rng = np.random.default_rng(seed)
    x1, x2 = rng.normal(size=(n, t)), rng.normal(2.0, 1.0, size=(n, t))
    mu = rng.normal(0.0, sigma_u, size=(n, 1))
    eps = rng.normal(0.0, 1.0, size=(n, t))
    u = np.linalg.inv(np.eye(n) - lam * w.full()[0]) @ eps
    y = 3.0 + 1.5 * x1 - 0.8 * x2 + mu + u
    return y, x1, x2


def synthetic_fixture(label, w, t, lam, seed, sigma_u, weights):
    n = w.n
    y, x1, x2 = synthetic(w, n, t, lam, seed, sigma_u)
    x = np.hstack((x1, x2))
    global NAMES
    NAMES = ["x1", "x2"]
    return {"n": n, "t": t, "y": y.tolist(), "x1": x1.tolist(), "x2": x2.tolist(),
            "graph": graph_of(w),
            "cases": [case_of(w, y, x, False, fw, f"{label} full_weights={fw}") for fw in weights]}


NAMES = None


def main():
    global NAMES
    out = {"name": "gm_kkp", "description": "spreg GM_ErrorRE (panel.py) = GM_KKP (sp_panels.py)",
           "meta": meta()}

    nat = gpd.read_file(libpysal.examples.get_path("NAT.shp"))
    w_full = Queen.from_dataframe(nat, use_index=False)
    sub = np.where(nat.STATE_NAME.isin(["Kansas", "Missouri", "Oklahoma", "Arkansas"]))[0].tolist()
    w = w_subset(w_full, sub)
    w.transform = "r"
    assert w.id_order == sub
    y = nat[["HR70", "HR80", "HR90"]].values[sub]
    rd = nat[["RD70", "RD80", "RD90"]].values[sub]
    ps = nat[["PS70", "PS80", "PS90"]].values[sub]
    NAMES = ["RD", "PS"]
    out["ncovr"] = {"n": w.n, "t": 3, "y": y.tolist(), "rd": rd.tolist(), "ps": ps.tolist(),
                    "graph": graph_of(w),
                    "cases": [case_of(w, y, np.hstack((rd, ps)), te, fw,
                                      f"ncovr full_weights={fw} te={te}")
                              for fw in (False, True) for te in (False, True)]}

    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    wc = Queen.from_iterable(gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]]))
    wc.transform = "r"
    out["columbus"] = synthetic_fixture("columbus", wc, 5, 0.3, 43, 1.0, (False, True))

    out["rook5x5"] = synthetic_fixture("rook5x5", lat2W(5, 5, rook=True), 5, 0.15, 47, 1.0,
                                       (False, True))
    dump_json(FIXTURES / "regression" / "gm_kkp.json", out)


if __name__ == "__main__":
    main()
