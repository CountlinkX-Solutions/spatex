"""Generate spreg OLS oracle fixtures (`fixtures/regression/ols_*.json`).

Run from the repository root (under a memory cap):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_spreg_ols.py

Every case stores the data (`y`, `x` columns), the exact sparse weights spreg consumed (read
back from ``W.sparse``, so Elixir rebuilds the identical matrix with ``Graph.from_arrays``) and
the attributes spreg attached to the fitted ``spreg.OLS`` object. spreg never transforms W
implicitly, so each case records which transform was applied here (``w_transform``).

The no-constant case cannot go through ``spreg.OLS`` (it always prepends a constant); it uses
``BaseOLS`` plus the ``spreg.diagnostics`` functions that ``OLS`` itself calls.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import geopandas as gpd  # noqa: E402
import numpy as np  # noqa: E402
import spreg  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from libpysal.weights import Kernel, W  # noqa: E402
from shapely import wkb  # noqa: E402
from shapely.geometry import box  # noqa: E402
from spreg import diagnostics  # noqa: E402
from spreg.ols import BaseOLS  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

warnings.simplefilter("ignore")


def columbus():
    d = json.loads((FIXTURES / "datasets" / "columbus.json").read_text())
    geoms = gpd.GeoSeries([wkb.loads(h, hex=True) for h in d["wkb_hex"]])
    attrs = {k: np.asarray(v, dtype=float) for k, v in d["attributes"].items()}
    return geoms, attrs


def sparse_export(w):
    """Triplets of the matrix spreg actually uses (``W.sparse``), positional."""
    coo = w.sparse.tocoo()
    order = np.lexsort((coo.col, coo.row))
    return {
        "n": int(w.n),
        "focal": coo.row[order].astype(int).tolist(),
        "neighbor": coo.col[order].astype(int).tolist(),
        "weight": coo.data[order].astype(float).tolist(),
    }


def make_w(graph, transform):
    g = graph.transform(transform) if transform != "o" else graph
    return g.to_W()


def pair(t):
    return [float(t[0]), float(t[1])]


def mat(a):
    return np.asarray(a, dtype=float).tolist()


def attrs_of(reg, k):
    """Everything OLS computed that the Elixir port reports."""
    out = {
        "n": int(reg.n),
        "k": int(reg.k),
        "betas": np.asarray(reg.betas).ravel(),
        "vm": mat(reg.vm),
        "std_err": np.asarray(reg.std_err).ravel() if hasattr(reg, "std_err") else None,
        "predy": np.asarray(reg.predy).ravel(),
        "u": np.asarray(reg.u).ravel(),
        "utu": float(reg.utu),
        "sig2": float(reg.sig2),
        "sig2n": float(reg.sig2n),
        "sig2n_k": float(reg.sig2n_k),
        "mean_y": float(reg.mean_y),
        "std_y": float(reg.std_y),
        "xtx": mat(reg.xtx),
        "xtxi": mat(reg.xtxi),
    }
    return out


def full_attrs(reg, *, nonspat=True, spat=False, moran=False, white_test=False, vif=False):
    out = attrs_of(reg, reg.k)
    out["t_stat"] = [pair(t) for t in reg.t_stat]
    out["r2"] = float(reg.r2)
    out["ar2"] = float(reg.ar2)
    out["warnings"] = [] if not getattr(reg, "warning", None) else [s for s in reg.warning.strip().split("\n") if s]
    out["name_x"] = list(reg.name_x)
    if nonspat:
        out["f_stat"] = pair(reg.f_stat)
        out["logll"] = float(reg.logll)
        out["aic"] = float(reg.aic)
        out["schwarz"] = float(reg.schwarz)
        out["sig2ml"] = float(reg.sig2ML)
        out["condition_number"] = float(reg.mulColli)
        out["jarque_bera"] = {k: float(v) for k, v in reg.jarque_bera.items()}
        out["breusch_pagan"] = {k: float(v) for k, v in reg.breusch_pagan.items()}
        out["koenker_bassett"] = {k: float(v) for k, v in reg.koenker_bassett.items()}
        if white_test:
            wt = reg.white
            out["white"] = wt if isinstance(wt, str) else {k: float(v) for k, v in wt.items()}
    if vif:
        out["vif"] = [None if v is None else pair(v) for v in reg.vif]
    if spat:
        for name in ("lm_error", "lm_lag", "rlm_error", "rlm_lag", "lm_sarma"):
            out[name] = pair(getattr(reg, name))
        for name in ("lm_wx", "lm_spdurbin", "rlm_wx", "rlm_durlag", "lm_slxerr"):
            out[name] = pair(getattr(reg, name)) if hasattr(reg, name) else None
        if moran:
            out["moran_res"] = [float(v) for v in reg.moran_res]
    return out


def case(label, y, cols, names, *, w=None, gwk=None, robust=None, sig2n_k=True,
         spat_diag=False, moran=False, white_test=False, vif=False, w_transform=None):
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    reg = spreg.OLS(
        y.reshape(-1, 1), x, w=w, gwk=gwk, robust=robust, sig2n_k=sig2n_k,
        nonspat_diag=True, spat_diag=spat_diag, moran=moran, white_test=white_test, vif=vif,
        name_x=list(names),
    )
    if vif:
        reg.vif = diagnostics.vif(reg)
    spat_ok = spat_diag and robust != "hac"
    c = {
        "label": label,
        "y": y,
        "x": [np.asarray(c, dtype=float) for c in cols],
        "name_x": list(names),
        "constant": True,
        "options": {
            "robust": robust, "sig2n_k": sig2n_k, "spat_diag": spat_diag, "moran": moran,
            "white_test": white_test, "vif": vif,
        },
        "w_transform": w_transform,
        "expected": full_attrs(
            reg, spat=spat_ok, moran=moran and spat_ok, white_test=white_test and robust is None, vif=vif
        ),
    }
    if w is not None:
        c["w"] = sparse_export(w)
    if gwk is not None:
        c["gwk"] = sparse_export(gwk)
    return c


def no_constant_case(label, y, cols, names):
    y = np.asarray(y, dtype=float)
    x = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    reg = BaseOLS(y.reshape(-1, 1), x, sig2n_k=True)
    reg.name_x = list(names)
    reg.t_stat = diagnostics.t_stat(reg)
    reg.r2 = diagnostics.r2(reg)
    reg.ar2 = diagnostics.ar2(reg)
    reg.f_stat = diagnostics.f_stat(reg)
    reg.logll = diagnostics.log_likelihood(reg)
    reg.aic = diagnostics.akaike(reg)
    reg.schwarz = diagnostics.schwarz(reg)
    reg.sig2ML = reg.sig2n
    reg.mulColli = diagnostics.condition_index(reg)
    reg.jarque_bera = diagnostics.jarque_bera(reg)
    reg.breusch_pagan = diagnostics.breusch_pagan(reg)
    reg.koenker_bassett = diagnostics.koenker_bassett(reg)
    reg.white = diagnostics.white(reg)
    reg.vif = diagnostics.vif(reg)
    reg.std_err = np.sqrt(np.diag(reg.vm))
    return {
        "label": label,
        "y": y,
        "x": [np.asarray(c, dtype=float) for c in cols],
        "name_x": list(names),
        "constant": False,
        "options": {
            "robust": None, "sig2n_k": True, "spat_diag": False, "moran": False,
            "white_test": True, "vif": True,
        },
        "w_transform": None,
        "expected": full_attrs(reg, white_test=True, vif=True),
    }


def header(name, description):
    m = meta()
    m["spreg"] = spreg.__version__
    return {"name": name, "description": description, "meta": m}


def main():
    out = FIXTURES / "regression"
    geoms, attrs = columbus()
    queen = Graph.build_contiguity(geoms, rook=False)
    rook = Graph.build_contiguity(geoms, rook=True)
    queen_r = make_w(queen, "r")
    rook_r = make_w(rook, "r")
    rook_b = make_w(rook, "o")

    cx, cy = geoms.centroid.x.to_numpy(), geoms.centroid.y.to_numpy()
    gwk = Kernel(np.column_stack([cx, cy]), k=15, function="triangular", fixed=False, diagonal=True)

    inc, crime, hoval = attrs["INC"], attrs["CRIME"], attrs["HOVAL"]
    cases = [
        case("hoval~inc+crime/plain", hoval, [inc, crime], ["INC", "CRIME"]),
        case("hoval~inc+crime/queen_r/spat", hoval, [inc, crime], ["INC", "CRIME"], w=queen_r,
             spat_diag=True, moran=True, white_test=True, vif=True, w_transform="r"),
        case("crime~inc+hoval/rook_r/spat", crime, [inc, hoval], ["INC", "HOVAL"], w=rook_r,
             spat_diag=True, moran=True, white_test=True, vif=True, w_transform="r"),
        case("crime~inc+hoval/rook_binary/spat", crime, [inc, hoval], ["INC", "HOVAL"], w=rook_b,
             spat_diag=True, moran=True, w_transform="o"),
        case("hoval~inc+crime/sig2n", hoval, [inc, crime], ["INC", "CRIME"], sig2n_k=False,
             white_test=True),
        case("hoval~inc+crime/white", hoval, [inc, crime], ["INC", "CRIME"], robust="white"),
        case("hoval~inc+crime/white_sig2n", hoval, [inc, crime], ["INC", "CRIME"], robust="white",
             sig2n_k=False),
        case("hoval~inc+crime/hac", hoval, [inc, crime], ["INC", "CRIME"], robust="hac", gwk=gwk),
    ]
    dummy = (inc > 12).astype(float)
    cases.append(case("crime~inc+hoval+dummy", crime, [inc, hoval, dummy], ["INC", "HOVAL", "D"],
                      white_test=True))
    cases.append(case("crime~inc+hoval+extra_constant", crime, [inc, hoval, np.full(49, 3.0)],
                      ["INC", "HOVAL", "C3"]))
    rng = np.random.default_rng(11)
    x2 = inc + 1e-6 * rng.normal(size=49)
    cases.append(case("crime~inc+x2/collinear", crime, [inc, x2], ["INC", "X2"], white_test=True))
    cases.append(no_constant_case("hoval~inc+crime/no_constant", hoval, [inc, crime], ["INC", "CRIME"]))

    fx = header("ols_columbus", "spreg.OLS on the Columbus data (queen/rook contiguity, HAC kernel).")
    fx["cases"] = cases
    dump_json(out / "ols_columbus.json", fx)

    # Synthetic n = 200, k = 5 (constant + 4 regressors) on a 10 x 20 rook lattice.
    cells = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(10) for c in range(20)])
    lat = Graph.build_contiguity(cells, rook=True)
    lat_r = make_w(lat, "r")
    n = 200
    xs = [rng.normal(size=n).round(6) for _ in range(4)]
    noise = rng.normal(size=n) * (1.0 + 0.5 * np.abs(xs[0]))
    y = (2.0 + xs[0] - 0.5 * xs[1] + 0.25 * xs[2] + noise).round(6)
    fx = header("ols_synthetic", "spreg.OLS on a synthetic n=200, k=5 design over a 10x20 rook lattice.")
    fx["cases"] = [
        case("synthetic/plain", y, xs, ["X1", "X2", "X3", "X4"], w=lat_r, spat_diag=True, moran=True,
             white_test=True, vif=True, w_transform="r"),
        case("synthetic/white", y, xs, ["X1", "X2", "X3", "X4"], robust="white"),
    ]
    dump_json(out / "ols_synthetic.json", fx)

    # 3x3 rook lattice, hand-checkable T and j.
    cells3 = gpd.GeoSeries([box(c, r, c + 1, r + 1) for r in range(3) for c in range(3)])
    g3 = Graph.build_contiguity(cells3, rook=True)
    w3 = make_w(g3, "r")
    y9 = np.array([3.0, 1.5, 4.0, 1.0, 5.5, 9.0, 2.5, 6.0, 5.0])
    x9 = np.array([1.0, 2.0, 0.5, 4.0, 3.0, 2.0, 6.0, 5.0, 7.5])
    fx = header("ols_lattice", "spreg.OLS on a 3x3 rook lattice (n=9).")
    fx["cases"] = [
        case("lattice3/r", y9, [x9], ["X"], w=w3, spat_diag=True, moran=True, w_transform="r"),
    ]
    dump_json(out / "ols_lattice.json", fx)


if __name__ == "__main__":
    main()
