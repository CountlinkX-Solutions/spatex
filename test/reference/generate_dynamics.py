"""Generate giddy / inequality oracle fixtures (Markov family, rank mobility, Gini, Theil, Atkinson,
Schutz, Wolfson, spatial polarization S, Kullback test).

Run from the repository root (memory-capped):

    export NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
    for s in markov spatial_markov lisa_markov rank_mobility inequality inequality_indices polarization kullback; do
        prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_dynamics.py $s; done

Writes test/reference/fixtures/dynamics/*.json. Datasets: libpysal's bundled ``mexico`` example
(32 states, 7 decades 1940-2000, ``mexico.gal`` queen weights, regime columns) and a seeded
synthetic 49 x 10 panel on the Columbus queen graph (committed graph fixture).

Deterministic quantities are compared to 1e-9; ``perm`` quantities come from PySAL with
``np.random.seed(SEED)`` and large ``permutations`` and are compared statistically.
Documented deviations: the categorical lag resolves residual ties at random in libpysal; the
oracle here patches that to "lowest class" which is the rule Spatex implements.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import libpysal  # noqa: E402
import mapclassify as mc  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from giddy import ergodic, rank  # noqa: E402
from giddy.markov import LISA_Markov, Markov, Spatial_Markov, kullback  # noqa: E402
from giddy.markov import chi2 as giddy_chi2  # noqa: E402
from inequality.atkinson import Atkinson, atkinson  # noqa: E402
from inequality.gini import Gini, Gini_Spatial  # noqa: E402
from inequality.polarization import S  # noqa: E402
from inequality.schutz import Schutz  # noqa: E402
from inequality.theil import Theil, TheilD, TheilDSim  # noqa: E402
from inequality.wolfson import lorenz_curve, wolfson  # noqa: E402
from libpysal.graph import Graph  # noqa: E402
from libpysal.weights import spatial_lag  # noqa: E402

from common import FIXTURES, dump_json, export_graph  # noqa: E402

SEED = 20241025
PERMS = 9999
OUT = FIXTURES / "dynamics"
warnings.filterwarnings("ignore")


def meta():
    import giddy
    import inequality

    return {
        "giddy": giddy.__version__,
        "inequality": inequality.__version__,
        "libpysal": libpysal.__version__,
        "numpy": np.__version__,
    }


# libpysal breaks residual categorical-lag ties at random; make it deterministic (lowest class).
_orig_choice = np.random.choice


def _resolve_lowest(idx, normalized_labels, tally, neighbors, method, w):
    (ties,) = np.where(tally == tally.max())
    if len(ties) <= 1:
        return np.argmax(tally).astype(int)
    if method.lower() == "tryself":
        mean_neighbor_value = np.mean(list(neighbors.values()))
        tally[normalized_labels[idx]] += mean_neighbor_value
        (ties,) = np.where(tally == tally.max())
        if len(ties) <= 1:
            return np.argmax(tally).astype(int)
    return ties[0].astype(int)


spatial_lag._resolve_ties = _resolve_lowest


def load_mexico():
    ex = Path(libpysal.__file__).parent / "examples" / "mexico"
    df = pd.read_csv(ex / "mexico.csv")
    years = list(range(1940, 2001, 10))
    y = df[[f"pcgdp{yr}" for yr in years]].to_numpy(float)
    w = libpysal.io.open(str(ex / "mexico.gal")).read()
    g = Graph.from_W(w)
    # relabel to positions 0..n-1 so x[i] indexing in giddy/inequality works
    pos = {i: k for k, i in enumerate(g.unique_ids)}
    adj = g.adjacency.reset_index()
    g = Graph.from_arrays(
        [pos[a] for a in adj["focal"]], [pos[a] for a in adj["neighbor"]], adj["weight"].to_numpy()
    )
    return df, y, g


def load_columbus_graph():
    fx = json.loads((FIXTURES / "graph" / "columbus_queen.json").read_text())
    a = fx["transforms"]["O"]["adjacency"]
    return Graph.from_arrays(a["focal"], a["neighbor"], a["weight"])


def synthetic_panel(g, n=49, t=10, seed=7):
    rng = np.random.default_rng(seed)
    wr = g.transform("r").sparse
    z = rng.normal(size=n)
    z = 0.5 * z + 0.5 * (wr @ z)
    cols = []
    for _ in range(t):
        z = 0.8 * z + 0.3 * (wr @ z) + 0.3 * rng.normal(size=n)
        cols.append(z.copy())
    return np.round(100 * np.exp(np.column_stack(cols)), 6)


def gw(g):
    """Row-standardized W for giddy."""
    return g.transform("r").to_W()


def lst(a):
    return np.asarray(a).tolist()


def markov_out(m):
    cc = sorted(sorted(int(i) for i in c) for c in m.cclasses_indices)
    rc = sorted(sorted(int(i) for i in c) for c in m.rclasses_indices)
    ss = np.asarray(m.steady_state)
    try:
        mf = m.mfpt
        mf_error = None
    except Exception as exc:  # giddy fails on reducible chains with a recurrent cycle
        mf, mf_error = None, f"{type(exc).__name__}: {exc}"
    return {
        "classes": lst(m.classes),
        "k": int(m.k),
        "transitions": lst(m.transitions),
        "p": lst(m.p),
        "steady_state": lst(ss),
        "steady_state_ndim": int(ss.ndim),
        "mfpt": None if mf is None else lst(mf),
        "mfpt_error": mf_error,
        "sojourn_time": lst(m.sojourn_time),
        "cclasses": cc,
        "rclasses": rc,
        "num_tclasses": int(m.num_tclasses),
        "astates": sorted(int(np.ravel(i)[0]) for i in m.astates_indices),
    }


def markov_case(label, ids=None, y=None, k=None, fill=False):
    if y is not None:
        ids = np.array([mc.Quantiles(y[:, j], k=k).yb for j in range(y.shape[1])]).T
    ids = np.asarray(ids)
    m = Markov(ids, fill_empty_classes=fill, summary=False)
    c = {"label": label, "fill_empty_classes": fill, "ids": ids, "out": markov_out(m)}
    if y is not None:
        c.update({"y": y, "k": k})
    return c


def markov():
    df, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    cases = [
        markov_case("mexico q5", y=ymx, k=5),
        markov_case("mexico q3", y=ymx, k=3),
        markov_case("synthetic q5", y=syn, k=5),
        markov_case("synthetic q4 fill", y=syn, k=4, fill=True),
        # class 3 only appears in the last period: an empty row of p.
        markov_case("empty row", ids=[[0, 1, 1], [1, 0, 0], [2, 2, 3], [0, 0, 3], [1, 1, 2]]),
        markov_case("empty row fill", ids=[[0, 1, 1], [1, 0, 0], [2, 2, 3], [0, 0, 3], [1, 1, 2]], fill=True),
        # two recurrent classes plus a transient one.
        markov_case("reducible", ids=[[0, 1, 0, 1], [1, 0, 1, 0], [2, 2, 2, 2], [3, 3, 2, 2], [3, 3, 0, 1]]),
        markov_case("reducible tree", ids=[[0, 0, 1, 1], [1, 1, 2, 2], [0, 1, 1, 2], [2, 2, 2, 2], [0, 0, 0, 1]]),
        markov_case("absorbing", ids=[[0, 1, 2, 2], [1, 0, 1, 2], [0, 0, 1, 1], [2, 2, 2, 2], [1, 2, 2, 2]]),
        markov_case("labels", ids=[[10, 30, 30], [30, 10, 20], [20, 20, 10], [10, 10, 30]]),
    ]
    fx = {"name": "markov", "meta": meta(), "cases": cases}
    dump_json(OUT / "markov.json", fx)

    # Ergodic helper checks on hand matrices from the giddy docs.
    oz = np.array([[0.5, 0.25, 0.25], [0.5, 0, 0.5], [0.25, 0.25, 0.5]])
    red = np.array([[0.5, 0.5, 0], [0.4, 0.6, 0], [0, 0, 1.0]])
    red2 = np.array([[0.5, 0.5, 0], [0.5, 0.5, 0], [0, 0, 1]])
    mats = []
    for label, p in (("oz", oz), ("reducible", red), ("two_rclasses", red2)):
        ss = np.asarray(ergodic.steady_state(p))
        mats.append({
            "label": label, "p": p, "steady_state": ss, "mfpt": ergodic.mfpt(p),
            "steady_state_ndim": int(ss.ndim),
        })
    dump_json(OUT / "ergodic.json", {"name": "ergodic", "meta": meta(), "cases": mats})


def sm_out(sm):
    ht = sm.ht
    return {
        "k": int(sm.k),
        "m": int(sm.m),
        "cutoffs": None if sm.cutoffs is None else lst(sm.cutoffs),
        "lag_cutoffs": None if sm.lag_cutoffs is None else lst(sm.lag_cutoffs),
        "class_ids": lst(sm.class_ids),
        "lclass_ids": lst(sm.lclass_ids),
        "transitions": lst(sm.transitions),
        "p": lst(sm.p),
        "t": lst(sm.T),
        "p_cond": lst(sm.P),
        "s": lst(sm.s),
        "s_cond": [lst(s) for s in sm.S],
        "f": lst(sm.f),
        "f_cond": lst(sm.F),
        "q": float(sm.Q),
        "q_p_value": float(sm.Q_p_value),
        "lr": float(sm.LR),
        "lr_p_value": float(sm.LR_p_value),
        "dof_hom": int(sm.dof_hom),
        "shtest": [[float(a), float(b), int(c)] for a, b, c in sm.shtest],
        "chi2": [[float(a), float(b), int(c)] for a, b, c in sm.chi2],
        "x2": float(sm.x2),
        "x2_pvalue": float(sm.x2_pvalue),
        "x2_dof": int(sm.x2_dof),
    }


def sm_case(label, y, g, perms=0, **kw):
    np.random.seed(SEED)
    sm = Spatial_Markov(y, gw(g), permutations=perms, fill_empty_classes=True, **kw)
    c = {"label": label, "y": y, "opts": {kk: vv for kk, vv in kw.items()}, "fill_empty_classes": True,
         "out": sm_out(sm)}
    if perms:
        c["permutations"] = perms
        c["x2_rpvalue"] = float(sm.x2_rpvalue)
        c["x2_realizations_mean"] = float(np.mean(sm.x2_realizations))
        c["x2_realizations_std"] = float(np.std(sm.x2_realizations))
    return c


def spatial_markov():
    df, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    cls = np.array([mc.Quantiles(syn[:, j], k=3).yb for j in range(syn.shape[1])]).T
    cases = [
        {**sm_case("mexico fixed", ymx, gmx, k=4, m=4), "graph": "mexico"},
        {**sm_case("mexico k3 m3", ymx, gmx, k=3, m=3), "graph": "mexico"},
        {**sm_case("synthetic fixed", syn, gcol, k=5, m=3), "graph": "columbus"},
        {**sm_case("synthetic not fixed", syn, gcol, k=4, m=4, fixed=False), "graph": "columbus"},
        {**sm_case("synthetic cutoffs", syn, gcol, cutoffs=[90.0, 130.0, 180.0], lag_cutoffs=[100.0, 150.0]),
         "graph": "columbus"},
        {**sm_case("synthetic discrete", cls, gcol, discrete=True), "graph": "columbus"},
        {**sm_case("synthetic perms", syn, gcol, perms=PERMS, k=4, m=4), "graph": "columbus"},
        {**sm_case("mexico perms", ymx, gmx, perms=PERMS, k=4, m=4), "graph": "mexico"},
    ]
    # python bools
    for c in cases:
        c["opts"] = {kk: vv for kk, vv in c["opts"].items()}
    fx = {
        "name": "spatial_markov", "meta": meta(), "seed": SEED,
        "graphs": {"mexico": export_graph(gmx), "columbus": export_graph(gcol)},
        "cases": cases,
    }
    dump_json(OUT / "spatial_markov.json", fx)


def lisa_case(label, y, g, perms=0):
    np.random.seed(SEED)
    lm = LISA_Markov(y, gw(g), permutations=perms)
    out = {
        "q": lst(lm.q),
        "transitions": lst(lm.transitions),
        "p": lst(lm.p),
        "steady_state": lst(lm.steady_state),
        "mfpt": lst(lm.mfpt),
        "sojourn_time": lst(lm.sojourn_time),
        "move_types": lst(lm.move_types),
        "chi_2": [float(lm.chi_2[0]), float(lm.chi_2[1]), int(lm.chi_2[2])],
        "expected_t": lst(lm.expected_t),
    }
    # giddy builds Markov(rc) from the (t, n) array, so its "transitions" run between neighboring
    # *locations* instead of periods (a bug); Spatex uses the intended (n, t) orientation.
    ybar = y.mean(axis=1)
    rc = y / ybar[:, None] < 1.0
    ylag = np.array([spatial_lag.lag_spatial(gw(g), y[:, s]) for s in range(y.shape[1])]).T
    rlagc = ylag / ybar[:, None] < 1.0
    kp0 = Markov(rc, summary=False).p
    kp1 = Markov(rlagc, summary=False).p
    amat = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 1, 0, 0]])
    kp = amat.dot(np.kron(kp0, kp1)).dot(amat.T)
    expected = np.diag(lm.transitions.sum(axis=1)).dot(kp)
    c2 = giddy_chi2(lm.transitions, expected)
    out["chi_2_intended"] = [float(c2[0]), float(c2[1]), int(c2[2])]
    out["expected_t_intended"] = lst(expected)
    if perms:
        out["p_values"] = lst(lm.p_values)
        out["significant_moves"] = lst(lm.significant_moves)
    return {"label": label, "y": y, "permutations": perms, "out": out}


def lisa_markov():
    df, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    cases = [
        {**lisa_case("mexico", ymx, gmx), "graph": "mexico"},
        {**lisa_case("synthetic", syn, gcol), "graph": "columbus"},
        {**lisa_case("synthetic perms", syn, gcol, perms=PERMS), "graph": "columbus"},
    ]
    fx = {
        "name": "lisa_markov", "meta": meta(), "seed": SEED,
        "graphs": {"mexico": export_graph(gmx), "columbus": export_graph(gcol)},
        "cases": cases,
    }
    dump_json(OUT / "lisa_markov.json", fx)


def rank_mobility():
    df, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    rng = np.random.default_rng(11)
    xt = rng.integers(0, 6, size=40).astype(float)
    yt = np.round(xt + rng.normal(size=40) * 2).clip(0, 8)
    pairs = [
        ("mexico 1940-2000", ymx[:, 0], ymx[:, -1]),
        ("synthetic 0-9", syn[:, 0], syn[:, 9]),
        ("ties", xt, yt),
        ("perfect", np.arange(1.0, 9.0), np.arange(1.0, 9.0) * 2),
        ("reversed", np.arange(1.0, 9.0), -np.arange(1.0, 9.0)),
    ]
    taus = []
    for label, x, y in pairs:
        t = rank.Tau(x, y)
        taus.append({
            "label": label, "x": x, "y": y, "tau": float(t.tau), "tau_p": float(t.tau_p),
            "concordant": float(t.concordant), "discordant": float(t.discordant),
            "extra_x": float(t.extraX), "extra_y": float(t.extraY),
        })
    dump_json(OUT / "tau.json", {"name": "tau", "meta": meta(), "cases": taus})

    stau = []
    for label, x, y, g, gname in (
        ("mexico 1940-2000", ymx[:, 0], ymx[:, -1], gmx, "mexico"),
        ("mexico 1960-1980", ymx[:, 2], ymx[:, 4], gmx, "mexico"),
        ("synthetic 0-9", syn[:, 0], syn[:, 9], gcol, "columbus"),
        ("ties", np.round(syn[:, 0] / 20), np.round(syn[:, 9] / 20), gcol, "columbus"),
    ):
        np.random.seed(SEED)
        w = g.to_W()
        s = rank.SpatialTau(x, y, w, permutations=PERMS)
        stau.append({
            "label": label, "graph": gname, "x": x, "y": y, "tau": float(s.tau),
            "tau_p": float(s.tau_p), "tau_spatial": float(s.tau_spatial),
            "pairs_spatial": int(s.pairs_spatial), "concordant": float(s.concordant),
            "discordant": float(s.discordant), "extra_x": float(s.extraX), "extra_y": float(s.extraY),
            "concordant_spatial": float(s.concordant_spatial),
            "discordant_spatial": float(s.discordant_spatial),
            "permutations": PERMS, "tau_spatial_psim": float(s.tau_spatial_psim),
            "taus_mean": float(s.taus.mean()), "taus_std": float(s.taus.std()),
        })
    dump_json(OUT / "spatial_tau.json", {
        "name": "spatial_tau", "meta": meta(), "seed": SEED,
        "graphs": {"mexico": export_graph(gmx), "columbus": export_graph(gcol)}, "cases": stau,
    })

    thetas = []
    rng2 = np.random.default_rng(5)
    for label, y, regime in (
        ("mexico hanson03", ymx, df["hanson03"].to_numpy().astype(int)),
        ("mexico inegi", ymx, df["inegi"].to_numpy().astype(int)),
        ("synthetic 4 regimes", syn, rng2.integers(0, 4, size=49)),
    ):
        np.random.seed(SEED)
        th = rank.Theta(y, regime, permutations=PERMS)
        thetas.append({
            "label": label, "y": y, "regime": regime, "theta": lst(th.theta)[0],
            "total": lst(th.total), "max_total": float(th.max_total), "ranks": lst(th.ranks),
            "permutations": PERMS, "pvalue_left": lst(th.pvalue_left),
            "pvalue_right": lst(th.pvalue_right), "z": lst(th.z),
        })
    dump_json(OUT / "theta.json", {"name": "theta", "meta": meta(), "seed": SEED, "cases": thetas})


def inequality():
    df, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    withzero = np.array([0.0, 3.0, 5.0, 0.0, 9.0, 1.0])
    gini = [{"label": lab, "x": x, "g": float(Gini(x).g)} for lab, x in (
        ("mexico 1940", ymx[:, 0]), ("mexico 2000", ymx[:, -1]), ("synthetic 3", syn[:, 3]),
        ("with zeros", withzero), ("equal", np.full(5, 2.0)),
    )]
    theil = [
        {"label": "mexico 1940", "y": ymx[:, 0], "t": [float(Theil(ymx[:, 0]).T)]},
        {"label": "mexico panel", "y": ymx, "t": lst(Theil(ymx).T)},
        {"label": "synthetic panel", "y": syn, "t": lst(Theil(syn).T)},
        {"label": "with zeros", "y": withzero, "t": [float(Theil(withzero).T)]},
    ]
    part = df["hanson03"].to_numpy().astype(int)
    theil_d = []
    for label, y, p in (
        ("mexico panel hanson03", ymx, part),
        ("mexico 2000 inegi", ymx[:, -1], df["inegi"].to_numpy().astype(int)),
    ):
        d = TheilD(y, p)
        theil_d.append({"label": label, "y": y, "partition": p, "t": lst(d.T), "bg": lst(d.bg), "wg": lst(d.wg)})
    np.random.seed(SEED)
    sim = TheilDSim(ymx, part, permutations=PERMS)
    theil_sim = {
        "label": "mexico panel hanson03", "y": ymx, "partition": part, "permutations": PERMS,
        "bg_pvalue": lst(sim.bg_pvalue), "bg": lst(sim.bg[0]), "bg_sim_mean": lst(sim.bg[1:].mean(axis=0)),
    }
    sg = []
    for label, x, g, gname in (
        ("mexico 2000", ymx[:, -1], gmx, "mexico"),
        ("mexico 1940", ymx[:, 0], gmx, "mexico"),
        ("synthetic 9", syn[:, 9], gcol, "columbus"),
    ):
        np.random.seed(SEED)
        s = Gini_Spatial(x, g.to_W(), permutations=PERMS)
        sg.append({
            "label": label, "graph": gname, "x": x, "g": float(s.g), "wg": float(s.wg),
            "wcg": float(s.wcg), "wcg_share": float(s.wcg_share), "dtotal": float(s.dtotal),
            "polarization": float(s.polarization), "permutations": PERMS,
            "p_sim": float(s.p_sim), "e_wcg": float(s.e_wcg), "s_wcg": float(s.s_wcg),
            "z_wcg": float(s.z_wcg), "p_z_sim": float(s.p_z_sim),
            "polarization_p_sim": float(s.polarization_p_sim),
        })
    dump_json(OUT / "inequality.json", {
        "name": "inequality", "meta": meta(), "seed": SEED,
        "graphs": {"mexico": export_graph(gmx), "columbus": export_graph(gcol)},
        "gini": gini, "theil": theil, "theil_d": theil_d, "theil_d_sim": theil_sim, "spatial_gini": sg,
    })


def inequality_indices():
    """Atkinson, Schutz, Wolfson and the Lorenz curve (deterministic, compared to 1e-9)."""
    _, ymx, _ = load_mexico()
    syn = synthetic_panel(load_columbus_graph())
    doc_income = [20000, 25000, 27000, 30000, 35000, 45000, 60000, 75000, 80000, 120000]
    data = [
        ("mexico 1940", ymx[:, 0]), ("mexico 2000", ymx[:, -1]), ("synthetic 3", syn[:, 3]),
        ("docstring incomes", np.array(doc_income, float)),
        ("ties", np.array([5.0, 5.0, 1.0, 1.0, 9.0, 9.0, 5.0])),
        ("two values", np.array([1.0, 4.0])), ("single value", np.array([7.0])),
        ("equal", np.full(5, 2.0)),
    ]
    atk = []
    for label, x in data:
        for eps in (0.0, 0.25, 0.5, 1.0, 2.0, 5.0):
            a = Atkinson(x, eps)
            assert a.A == atkinson(x, eps)
            atk.append({"label": label, "y": x, "epsilon": eps, "a": float(a.A), "ede": float(a.EDE)})
    schutz_data = data + [
        ("with zeros", np.array([0.0, 3.0, 5.0, 0.0, 9.0, 1.0])),
        ("docstring Y", np.array([1000, 2000, 1500, 3000, 2500], float)),
    ]
    sch = []
    for label, x in schutz_data:
        s = Schutz(pd.DataFrame({"Y": x}), "Y")
        sch.append({
            "label": label, "y": x, "distance": float(s.distance),
            "intersection_point": float(s.intersection_point), "coefficient": float(s.coefficient),
        })
    wolf_data = data + [
        ("zeros", np.array([0.0, 3.0, 5.0, 0.0, 9.0, 1.0])),
        ("docstring bipolar", np.array([6, 6, 8, 8, 10, 10, 12, 12], float)),
        ("odd length", np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0])),
    ]
    wol = []
    for label, x in wolf_data:
        pop, share = lorenz_curve(x)
        wol.append({
            "label": label, "y": x, "wolfson": float(wolfson(x)),
            "population": pop, "income_share": share,
        })
    dump_json(OUT / "inequality_indices.json", {
        "name": "inequality_indices", "meta": meta(), "atkinson": atk, "schutz": sch, "wolfson": wol,
    })


def load_graph_fixture(name):
    fx = json.loads((FIXTURES / "graph" / f"{name}.json").read_text())
    a = fx["transforms"]["O"]["adjacency"]
    return Graph.from_arrays(a["focal"], a["neighbor"], a["weight"])


def polarization():
    """Spatial polarization S: exact statistic and labels, permutation summary (statistical)."""
    df0, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    glat = load_graph_fixture("lattice4x4_rook")
    gisl = load_graph_fixture("asymmetric_island")
    mx00 = ymx[:, -1]
    ties = np.array([1.0] * 12 + [2.0] * 8 + list(np.linspace(3, 9, 12)) + [5.0] * 17)
    cases_in = [
        ("mexico 2000 k2", mx00, gmx, "mexico", dict(k=2)),
        ("mexico 2000 k3", mx00, gmx, "mexico", dict(k=3)),
        ("mexico 1940 k4", ymx[:, 0], gmx, "mexico", dict(k=4)),
        ("mexico 2000 k1", mx00, gmx, "mexico", dict(k=1)),
        ("mexico 2000 bins", mx00, gmx, "mexico",
         dict(bins=[float(mx00.min()), 12000.0, 17500.0, 28000.0, float(mx00.max())])),
        ("mexico bins edge at minimum", mx00, gmx, "mexico",
         dict(bins=[float(mx00.min()), 15000.0, float(mx00.max())])),
        ("synthetic 9 k3", syn[:, 9], gcol, "columbus", dict(k=3)),
        ("synthetic 3 k2", syn[:, 3], gcol, "columbus", dict(k=2)),
        ("columbus ties k4 (quantile edges collapse)", np.resize(ties, 49), gcol, "columbus", dict(k=4)),
        ("lattice k2", np.arange(16.0)[::-1] % 7 + np.arange(16.0), glat, "lattice", dict(k=2)),
        ("constant k1", np.full(32, 5.0), gmx, "mexico", dict(k=1)),
        ("constant k2 (every observation without a group)", np.full(32, 5.0), gmx, "mexico", dict(k=2)),
        ("island graph k2", np.array([3.0, 1.0, 4.0, 1.0, 5.0]), gisl, "island", dict(k=2)),
        ("island graph bins", np.array([3.0, 1.0, 4.0, 1.5, 5.0]), gisl, "island", dict(bins=[0.0, 2.0, 6.0])),
    ]
    cases = []
    for label, y, g, gname, kw in cases_in:
        df = pd.DataFrame({"y": y}, index=list(g.unique_ids))
        s = S(df, g, "y", permutations=PERMS, seed=SEED, keep_sim=True, verbose=False, **kw)
        a_labels = [None if pd.isna(v) else int(v) for v in s.labels["a_labels"]]
        cases.append({
            "label": label, "graph": gname, "y": y, "opts": kw, "statistic": float(s.statistic_),
            "n_a_components": int(s.n_a_components), "n_g_components": int(s.n_g_components),
            "n_i_components": int(s.n_i_components), "i_labels": lst(s.labels["i_labels"]),
            "a_labels": a_labels, "g_labels": lst(s.labels["g_labels"]),
            "permutations": PERMS, "p_value": float(s.p_value), "sim_mean": float(s.sim.mean()),
            "sim_std": float(s.sim.std()), "sim_min": float(s.sim.min()), "sim_max": float(s.sim.max()),
        })
    # PySAL's docstring example, deterministic (every sim and the statistic are 1).
    lat = libpysal.weights.lat2W(40, 40)
    ysd = np.arange(1600)
    sd = S(pd.DataFrame({"y": ysd}, index=ysd), Graph.from_W(lat), "y", permutations=99, seed=123, verbose=False)
    docstring = {"statistic": float(sd.statistic_), "p_value": float(sd.p_value),
                 "n_a_components": int(sd.n_a_components), "n_g_components": int(sd.n_g_components),
                 "n_i_components": int(sd.n_i_components)}
    errors = {}
    for name, kw, y in (("k0", dict(k=0), mx00), ("k_above_n", dict(k=99), mx00),
                        ("bins_out_of_range", dict(bins=[9000.0, 20000.0]), mx00)):
        try:
            S(pd.DataFrame({"y": y}, index=list(gmx.unique_ids)), gmx, "y", permutations=0, verbose=False, **kw)
            errors[name] = None
        except Exception as exc:
            errors[name] = type(exc).__name__
    dump_json(OUT / "polarization.json", {
        "name": "polarization", "meta": meta(), "seed": SEED,
        "graphs": {"mexico": export_graph(gmx), "columbus": export_graph(gcol),
                   "lattice": export_graph(glat), "island": export_graph(gisl)},
        "cases": cases, "docstring_lattice": docstring, "errors": errors,
    })


def kullback_intended(f):
    """Kullback's conditional homogeneity statistic with the row totals ``f_si.`` of each stratum.

    giddy sums ``F`` over axis 1 (the column totals ``f_s.j``) where the formula needs axis 2 (the
    row totals), so its value is only right when every stratum has equal row and column totals.
    """
    f = np.asarray(f, float)

    def xlogx(a):
        return float((a * np.log(a + (a == 0))).sum())

    total = f.sum(axis=0)
    return 2 * (xlogx(f) - xlogx(f.sum(axis=2)) - xlogx(total) + xlogx(total.sum(axis=1)))


def kullback_case(label, f, **extra):
    from scipy import stats

    r = kullback(np.asarray(f))
    dof = int(r["Conditional homogeneity dof"])
    intended = kullback_intended(f)
    return {"label": label, "f": lst(f), **extra, "statistic": float(r["Conditional homogeneity"]),
            "dof": dof, "p_value": float(r["Conditional homogeneity pvalue"]),
            "statistic_intended": intended, "p_value_intended": float(1 - stats.chi2.cdf(intended, dof))}


def kullback_fixture():
    """Kullback information test on raw stacked counts and on Spatial_Markov conditional counts."""
    _, ymx, gmx = load_mexico()
    gcol = load_columbus_graph()
    syn = synthetic_panel(gcol)
    s1 = [[22, 11, 24, 2, 2, 7], [5, 23, 15, 3, 42, 6], [4, 21, 190, 25, 20, 34],
          [0, 2, 14, 56, 14, 28], [32, 15, 20, 10, 56, 14], [5, 22, 31, 18, 13, 134]]
    s2 = [[3, 6, 9, 3, 0, 8], [1, 9, 3, 12, 27, 5], [2, 9, 208, 32, 5, 18],
          [0, 14, 32, 108, 40, 40], [22, 14, 9, 26, 224, 14], [1, 5, 13, 53, 13, 116]]
    cases = [
        kullback_case("kullback 1962 table 9.2", [s1, s2]),
        kullback_case("three strata with empty rows and zero cells",
                      [[[5, 0, 2], [0, 0, 0], [1, 4, 6]], [[3, 1, 0], [2, 2, 2], [0, 0, 0]],
                       [[0, 0, 0], [4, 1, 3], [2, 2, 1]]]),
        kullback_case("two by two, two strata", [[[10, 5], [4, 12]], [[6, 9], [8, 3]]]),
        kullback_case("single stratum (zero degrees of freedom)", [[[10, 5], [4, 12]]]),
        kullback_case("identical strata", [[[10, 5], [4, 12]], [[10, 5], [4, 12]]]),
        kullback_case("fractional counts", [[[1.5, 2.5], [0.5, 4.0]], [[2.0, 1.0], [3.5, 0.25]]]),
    ]
    for label, y, g, gname, kw in (
        ("spatial markov mexico k4 m4", ymx, gmx, "mexico", dict(k=4, m=4)),
        ("spatial markov mexico k3 m3", ymx, gmx, "mexico", dict(k=3, m=3)),
        ("spatial markov synthetic k5 m3", syn, gcol, "columbus", dict(k=5, m=3)),
    ):
        sm = Spatial_Markov(y, gw(g), fill_empty_classes=True, **kw)
        cases.append(kullback_case(label, sm.T, y=y, graph=gname, opts=kw))
    dump_json(OUT / "kullback.json", {
        "name": "kullback", "meta": meta(), "graphs": {"mexico": export_graph(gmx), "columbus": export_graph(gcol)},
        "cases": cases,
    })


STEPS = {
    "markov": markov,
    "spatial_markov": spatial_markov,
    "lisa_markov": lisa_markov,
    "rank_mobility": rank_mobility,
    "inequality": inequality,
    "inequality_indices": inequality_indices,
    "polarization": polarization,
    "kullback": kullback_fixture,
}

if __name__ == "__main__":
    # numba JIT-compiles inside Moran_Local; run one step per process (pass names) to stay
    # under the 2 GiB address-space cap, e.g. `generate_dynamics.py lisa_markov`.
    for name in sys.argv[1:] or STEPS:
        STEPS[name]()
