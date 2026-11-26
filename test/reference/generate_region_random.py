"""Generate spopt 0.8.0 reference summaries for RandomRegion(s).

Run from the repository root under a memory cap (see test/reference/README.md):

    prlimit --as=2147483648 <venv>/bin/python -I test/reference/generate_region_random.py

Writes test/reference/fixtures/region/random_region.json.

spopt draws from numpy's global generator, so partitions are never compared. Each case records
summary statistics (mean, standard deviation, runs) of per-run scalars over many runs: the
number of regions, region sizes by construction order, the largest region and, for contiguous
cases, the share of feasible runs and a compactness score (the within-region sum of pairwise
Manhattan distances between lattice cells, divided by the number of areas). The Elixir test
compares Spatex's summaries with these within sampling error.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import libpysal
import numpy as np
import spopt
from spopt.region import RandomRegion

warnings.filterwarnings("ignore")

OUT = Path(__file__).resolve().parent / "fixtures" / "region" / "random_region.json"
SEED = 20260514


def summary(values):
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)), "runs": int(a.size)}


def compactness(regions, cols):
    total = 0.0
    for members in regions:
        rc = np.array([(m // cols, m % cols) for m in members])
        for i in range(len(rc)):
            total += np.abs(rc[i + 1 :] - rc[i]).sum()
    return total


def run_case(spec, runs, w, rows, cols):
    n = spec["n"]
    ids = list(range(n))
    stats = {
        k: []
        for k in ["n_regions", "first_size", "last_size", "max_size", "min_size", "compactness"]
    }
    feasible = 0
    for _ in range(runs):
        r = RandomRegion(
            ids,
            num_regions=spec.get("num_regions"),
            cardinality=spec.get("cardinality"),
            contiguity=w if spec["contiguity"] else None,
            maxiter=spec["maxiter"],
            compact=spec.get("compact", False),
            max_swaps=spec.get("max_swaps", 1000000),
        )
        if not r.feasible:
            continue
        feasible += 1
        sizes = [len(x) for x in r.regions]
        assert sorted(a for x in r.regions for a in x) == ids
        stats["n_regions"].append(len(sizes))
        stats["first_size"].append(sizes[0])
        stats["last_size"].append(sizes[-1])
        stats["max_size"].append(max(sizes))
        stats["min_size"].append(min(sizes))
        stats["compactness"].append(compactness(r.regions, cols) / n)
    return {
        "spec": spec,
        "runs": runs,
        "feasible_share": feasible / runs,
        "stats": {k: summary(v) for k, v in stats.items() if len(v) > 1},
    }


def main():
    rows, cols = 6, 6
    w = libpysal.weights.lat2W(rows, cols, rook=True)
    assert w.id_order == list(range(rows * cols))
    cases = {}
    plan = [
        ("free_n20", dict(n=20, contiguity=False, maxiter=1000), 6000),
        ("regions4_n20", dict(n=20, num_regions=4, contiguity=False, maxiter=1000), 6000),
        (
            "cardinality_n20",
            dict(n=20, cardinality=[8, 7, 5], contiguity=False, maxiter=1000),
            2000,
        ),
        (
            "contig_regions4",
            dict(n=36, num_regions=4, contiguity=True, maxiter=200),
            1500,
        ),
        (
            "contig_cardinality",
            dict(n=36, cardinality=[18, 9, 6, 3], contiguity=True, maxiter=200),
            1500,
        ),
        (
            "contig_cardinality_compact",
            dict(n=36, cardinality=[18, 9, 6, 3], contiguity=True, maxiter=200, compact=True),
            1500,
        ),
        (
            "contig_free",
            dict(n=36, contiguity=True, maxiter=20, max_swaps=2000),
            1500,
        ),
    ]
    for name, spec, runs in plan:
        np.random.seed(SEED)
        t = time.time()
        cases[name] = run_case(spec, runs, w, rows, cols)
        print(name, cases[name]["feasible_share"], round(time.time() - t, 1), "s", file=sys.stderr)
    out = {
        "spopt": spopt.__version__,
        "numpy": np.__version__,
        "seed": SEED,
        "lattice": {"rows": rows, "cols": cols},
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
