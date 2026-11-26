"""Generate giddy oracle fixtures for FullRank/GeoRank Markov, local tau and LISA spillover.

Run from the repository root (see test/reference/README.md), under a memory cap:

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_dynamics_rank_local.py

Writes test/reference/fixtures/dynamics/rank_markov.json, tau_local.json and spillover.json on
libpysal's bundled ``mexico`` example (32 states, 1940-2000). Deterministic fields are exact;
local tau p-values come from numpy's global generator (seeded here) and are compared
statistically. The spillover fixture records LISA_Markov's ``q`` and ``p_values`` so the
connected-component logic can be checked exactly on those inputs.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402
from giddy.markov import FullRank_Markov, GeoRank_Markov, LISA_Markov  # noqa: E402
from giddy.rank import Tau_Local, Tau_Local_Neighbor, Tau_Local_Neighborhood, Tau_Regional  # noqa: E402

from common import FIXTURES, dump_json, export_graph  # noqa: E402
from generate_dynamics import load_mexico, meta  # noqa: E402

PERMS = 999


def ints(a):
    return np.asarray(a).astype(int).tolist()


def floats(a):
    return [[None if np.isnan(v) else float(v) for v in row] for row in np.atleast_2d(a)]


def geo_ranks(y):
    """GeoRank_Markov's states: per period, the area holding each (ordinal) rank."""
    ranks = np.array([rankdata(col, method="ordinal") for col in y.T]).T
    return np.argsort(ranks, axis=0) + 1


def chain(m):
    return {"transitions": floats(m.transitions), "p": floats(m.p), "classes": ints(m.classes)}


def main():
    df, y, g = load_mexico()
    w = g.to_W()
    out = FIXTURES / "dynamics"
    tied = np.array([[1, 3, 3], [2, 2, 1], [2, 1, 2], [5, 3, 3]], dtype=float)

    fr, gr = FullRank_Markov(y, summary=False), GeoRank_Markov(y, summary=False)
    frt, grt = FullRank_Markov(tied, summary=False), GeoRank_Markov(tied, summary=False)
    dump_json(out / "rank_markov.json", {
        "name": "rank_markov", "description": "FullRank_Markov and GeoRank_Markov (mexico, tied)",
        "meta": meta(), "y": floats(y), "tied": floats(tied),
        "full_rank": {"ranks": ints(fr.ranks), **chain(fr)},
        "geo_rank": {"ranks": ints(geo_ranks(y)), **chain(gr)},
        "full_rank_tied": {"ranks": ints(frt.ranks), **chain(frt)},
        "geo_rank_tied": chain(grt),
    })

    x0, x1 = y[:, 0], y[:, -1]
    regime = df["esquivel99"].to_numpy()
    tl = Tau_Local(x0, x1)
    np.random.seed(62)
    tn = Tau_Local_Neighbor(x0, x1, w, permutations=PERMS)
    np.random.seed(63)
    th = Tau_Local_Neighborhood(x0, x1, w, permutations=PERMS)
    np.random.seed(64)
    tr = Tau_Regional(x0, x1, regime, permutations=PERMS)
    dump_json(out / "tau_local.json", {
        "name": "tau_local", "description": "Tau_Local family on mexico pcgdp1940 vs pcgdp2000",
        "meta": meta(), "graph": export_graph(g), "x": x0.tolist(), "y": x1.tolist(),
        "regime": ints(regime), "permutations": PERMS,
        "tau": float(tl.tau), "tau_local": tl.tau_local.tolist(),
        "neighbor": {"tau_ln": tn.tau_ln.tolist(), "weights": tn.tau_ln_weights.tolist(),
                     "sign": ints(tn.sign), "p": tn.tau_ln_pvalues.tolist()},
        "neighborhood": {"tau_lnhood": th.tau_lnhood.tolist(), "sign": ints(th.sign),
                         "p": th.tau_lnhood_pvalues.tolist()},
        "regional": {"tau_reg": floats(tr.tau_reg), "p": floats(tr.tau_reg_pvalues)},
    })

    np.random.seed(65)
    lm = LISA_Markov(y, w, permutations=99)
    cases = []
    for quadrant in (1, 2, 3, 4):
        for on in (False, True):
            r = lm.spillover(quadrant=quadrant, neighbors_on=on)
            cases.append({"quadrant": quadrant, "neighbors_on": on,
                          "components": ints(r["components"]), "spill_over": ints(r["spill_over"])})
    dump_json(out / "spillover.json", {
        "name": "spillover", "description": "LISA_Markov.spillover on mexico (permutations=99)",
        "meta": meta(), "graph": export_graph(g), "q": ints(lm.q),
        "p_values": floats(lm.p_values), "significance_level": lm.significance_level,
        "cases": cases,
    })


if __name__ == "__main__":
    main()
