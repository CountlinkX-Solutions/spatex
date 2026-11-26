"""Generate spreg ML_Lag / ML_Error oracle fixtures (`fixtures/regression/ml_*.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_ml.py

Every case stores the data (`y`, `x` columns without the constant), the exact sparse weights spreg
consumed (read back from ``W.sparse`` before the fit, because ``method='ord'`` mutates the
transform), the attributes of the fitted model and the optimizer's function-evaluation count
(captured by wrapping ``minimize_scalar``). spreg never transforms W implicitly, so each case also
records which transform was applied here.

Sizes stay small (n <= 200): the full method builds dense n x n matrices.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
import spreg.ml_error as ml_error_mod  # noqa: E402
import spreg.ml_lag as ml_lag_mod  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from libpysal.weights import DistanceBand, KNN, W  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")

LAST = {}


def wrap(module):
    original = module.minimize_scalar

    def wrapped(*args, **kwargs):
        res = original(*args, **kwargs)
        LAST["nfev"] = int(res.nfev)
        LAST["status"] = int(res.status)
        return res

    module.minimize_scalar = wrapped


wrap(ml_lag_mod)
wrap(ml_error_mod)


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    return geoms, attrs


def sparse_export(w):
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    return {
        "n": int(w.n),
        "focal": coo.row[order].astype(int).tolist(),
        "neighbor": coo.col[order].astype(int).tolist(),
        "weight": coo.data[order].astype(float).tolist(),
    }


def rebuild(spec):
    """A fresh W with the stored weights (ord mutates the transform of the one it is given)."""
    n = spec["n"]
    neighbors = {i: [] for i in range(n)}
    weights = {i: [] for i in range(n)}
    for f, t, w in zip(spec["focal"], spec["neighbor"], spec["weight"]):
        neighbors[f].append(t)
        weights[f].append(w)
    return W(neighbors, weights, silence_warnings=True)


def mat(a):
    return np.asarray(a, dtype=float).tolist()


def vec(a):
    return np.asarray(a, dtype=float).ravel()


def pair(t):
    return [float(t[0]), float(t[1])]


def lag_attrs(reg):
    out = {
        "n": int(reg.n),
        "k": int(reg.k),
        "betas": vec(reg.betas),
        "rho": float(reg.rho),
        "vm": mat(reg.vm),
        "vm1": mat(reg.vm1),
        "std_err": vec(reg.std_err),
        "z_stat": [pair(z) for z in reg.z_stat],
        "logll": float(reg.logll),
        "aic": float(reg.aic),
        "schwarz": float(reg.schwarz),
        "sig2": float(reg.sig2),
        "utu": float(reg.utu),
        "u": vec(reg.u),
        "predy": vec(reg.predy),
        "predy_e": vec(reg.predy_e),
        "e_pred": vec(reg.e_pred),
        "pr2": float(reg.pr2),
        "pr2_e": float(reg.pr2_e),
        "mean_y": float(reg.mean_y),
        "std_y": float(reg.std_y),
        "name_x": list(reg.name_x),
    }
    if getattr(reg, "sp_multipliers", None):
        out["sp_multipliers"] = {m: [float(v) for v in t] for m, t in reg.sp_multipliers.items()}
    return out


def err_attrs(reg):
    return {
        "n": int(reg.n),
        "k": int(reg.k),
        "betas": vec(reg.betas),
        "lam": float(np.asarray(reg.lam).ravel()[0]),
        "vm": mat(reg.vm),
        "vm1": mat(reg.vm1),
        "std_err": vec(reg.std_err),
        "z_stat": [pair(z) for z in reg.z_stat],
        "logll": float(np.asarray(reg.logll).ravel()[0]),
        "aic": float(reg.aic),
        "schwarz": float(reg.schwarz),
        "sig2": float(np.asarray(reg.sig2).ravel()[0]),
        "u": vec(reg.u),
        "predy": vec(reg.predy),
        "e_filtered": vec(reg.e_filtered),
        "pr2": float(reg.pr2),
        "mean_y": float(reg.mean_y),
        "std_y": float(reg.std_y),
        "name_x": list(reg.name_x),
    }


def case(label, model, y, cols, names, w_spec, w_transform, method, impacts="all"):
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    w = rebuild(w_spec)
    # `ord` symmetrizes the binary pattern when it is symmetric (the source of its two bugs).
    symmetrized = method == "ord" and w.asymmetry(intrinsic=False) == []
    LAST.clear()
    if model == "lag":
        reg = spreg.ML_Lag(
            y.reshape(-1, 1), x, w, method=method, epsilon=1e-7, spat_impacts=impacts,
            name_x=list(names), spat_diag=False,
        )
        expected = lag_attrs(reg)
    else:
        reg = spreg.ML_Error(y.reshape(-1, 1), x, w, method=method, epsilon=1e-7, name_x=list(names))
        expected = err_attrs(reg)
    expected["nfev"] = LAST["nfev"]
    expected["status"] = LAST["status"]
    return {
        "label": label,
        "model": model,
        "method": method,
        "epsilon": 1e-7,
        "y": y,
        "x": [np.asarray(c, dtype=float) for c in cols],
        "name_x": list(names),
        "w": w_spec,
        "w_transform": w_transform,
        "pysal_symmetrized": bool(symmetrized),
        "expected": expected,
    }


def header(name, description):
    m = meta()
    m["spreg"] = spreg.__version__
    return {"name": name, "description": description, "meta": m}


def transformed_w(graph, transform):
    g = graph.transform(transform) if transform != "o" else graph
    return g.to_W()


def main():
    out = FIXTURES / "regression"
    geoms, attrs = columbus()
    queen = Graph.build_contiguity(geoms, rook=False)
    rook = Graph.build_contiguity(geoms, rook=True)
    queen_r = sparse_export(transformed_w(queen, "r"))
    rook_r = sparse_export(transformed_w(rook, "r"))
    rook_b = sparse_export(transformed_w(rook, "o"))
    pts = np.column_stack([geoms.centroid.x.to_numpy(), geoms.centroid.y.to_numpy()])
    knn = KNN.from_array(pts, k=4)
    knn.transform = "R"
    knn_r = sparse_export(knn)

    inc, crime, hoval = attrs["INC"], attrs["CRIME"], attrs["HOVAL"]
    hx = [inc, crime]
    hn = ["INC", "CRIME"]
    cx = [inc, hoval]
    cn = ["INC", "HOVAL"]

    cases = []
    for method in ("full", "ord", "lu"):
        cases.append(case(f"hoval~inc+crime/queen_r/lag/{method}", "lag", hoval, hx, hn, queen_r, "r", method))
    for method in ("full", "ord"):
        cases.append(case(f"hoval~inc+crime/queen_r/error/{method}", "error", hoval, hx, hn, queen_r, "r", method))
    cases += [
        case("crime~inc+hoval/queen_r/lag/full", "lag", crime, cx, cn, queen_r, "r", "full"),
        case("crime~inc+hoval/queen_r/error/full", "error", crime, cx, cn, queen_r, "r", "full"),
        case("crime~inc+hoval/rook_r/lag/full", "lag", crime, cx, cn, rook_r, "r", "full"),
        case("crime~inc+hoval/rook_r/error/full", "error", crime, cx, cn, rook_r, "r", "full"),
        case("crime~inc+hoval/knn4_r/lag/full", "lag", crime, cx, cn, knn_r, "r", "full"),
        case("crime~inc+hoval/knn4_r/lag/ord", "lag", crime, cx, cn, knn_r, "r", "ord"),
        case("crime~inc+hoval/knn4_r/error/full", "error", crime, cx, cn, knn_r, "r", "full"),
        case("crime~inc+hoval/knn4_r/error/ord", "error", crime, cx, cn, knn_r, "r", "ord"),
        case("crime~inc+hoval/rook_binary/lag/full", "lag", crime, cx, cn, rook_b, "o", "full"),
        case("crime~inc+hoval/rook_binary/error/full", "error", crime, cx, cn, rook_b, "o", "full"),
        case("crime~inc/queen_r/lag/full_simple", "lag", crime, [inc], ["INC"], queen_r, "r", "full",
             impacts="simple"),
    ]
    fx = header("ml_columbus", "spreg ML_Lag / ML_Error on the Columbus data (queen, rook, KNN-4, binary).")
    fx["cases"] = cases
    dump_json(out / "ml_columbus.json", fx)

    # Inverse-distance band: symmetric pattern with non-binary weights (documents the `ord` bugs).
    thr = float(np.max(np.sort(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1) ** 0.5, axis=1)[:, 1]))
    idw = DistanceBand(pts, threshold=thr * 1.5, binary=False, alpha=-1.0)
    idw.transform = "R"
    idw_r = sparse_export(idw)
    dev = []
    for model in ("lag", "error"):
        for method in ("full", "ord"):
            dev.append(case(f"crime~inc+hoval/idw_r/{model}/{method}", model, crime, cx, cn, idw_r, "r", method))
    fx = header(
        "ml_ord_deviation",
        "Symmetric inverse-distance W (row-standardized): PySAL's `ord` uses the eigenvalues of the "
        "binary-symmetrized matrix and substitutes it for W in the variance; `full` is correct.",
    )
    fx["cases"] = dev
    dump_json(out / "ml_ord_deviation.json", fx)

    # Synthetic n = 200 on a 10 x 20 rook lattice, and n = 25 with negative dependence.
    rng = np.random.default_rng(19)
    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(10) for c in range(20)])
    lat = transformed_w(Graph.build_contiguity(cells, rook=True), "r")
    lat_spec = sparse_export(lat)
    wd = lat.full()[0]
    n = 200
    xs = [rng.normal(size=n).round(6) for _ in range(2)]
    eps = rng.normal(size=n) * 2.0
    mu = 1.0 + 1.5 * xs[0] - 0.75 * xs[1]
    y_lag = np.linalg.solve(np.eye(n) - 0.45 * wd, mu + eps).round(6)
    y_err = (mu + np.linalg.solve(np.eye(n) - 0.55 * wd, eps)).round(6)
    nm = ["X1", "X2"]
    synthetic = [
        case("synthetic200/lag/full", "lag", y_lag, xs, nm, lat_spec, "r", "full"),
        case("synthetic200/lag/ord", "lag", y_lag, xs, nm, lat_spec, "r", "ord"),
        case("synthetic200/error/full", "error", y_err, xs, nm, lat_spec, "r", "full"),
        case("synthetic200/error/ord", "error", y_err, xs, nm, lat_spec, "r", "ord"),
    ]
    cells5 = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(5) for c in range(5)])
    lat5 = transformed_w(Graph.build_contiguity(cells5, rook=True), "r")
    spec5 = sparse_export(lat5)
    w5 = lat5.full()[0]
    x5 = [rng.normal(size=25).round(6)]
    e5 = rng.normal(size=25)
    y5_lag = np.linalg.solve(np.eye(25) - (-0.5) * w5, 2.0 + 1.0 * x5[0] + e5).round(6)
    y5_err = (2.0 + 1.0 * x5[0] + np.linalg.solve(np.eye(25) - (-0.5) * w5, e5)).round(6)
    synthetic += [
        case("lattice25/negative/lag/full", "lag", y5_lag, x5, ["X1"], spec5, "r", "full"),
        case("lattice25/negative/error/full", "error", y5_err, x5, ["X1"], spec5, "r", "full"),
    ]
    fx = header("ml_synthetic", "spreg ML models on synthetic lattices (n=200 positive, n=25 negative dependence).")
    fx["cases"] = synthetic
    dump_json(out / "ml_synthetic.json", fx)


if __name__ == "__main__":
    main()
