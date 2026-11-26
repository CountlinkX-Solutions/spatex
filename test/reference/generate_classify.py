"""Generate mapclassify oracle fixtures for the classifiers.

Run from the repository root (see test/reference/README.md):

    <venv>/bin/python -I test/reference/generate_classify.py

Writes test/reference/fixtures/classify/*.json. Each file holds one input vector ``y`` and a list
of ``cases``; a case records the classifier name, its parameters and mapclassify's ``bins``,
``yb``, ``counts``, ``k``, ``tss``, ``adcm``, ``gadf``, ``lowest``, the legend labels
(``get_legend_classes()`` with the default format), a ``find_bin`` probe and the warnings
raised. Classifiers that raise are recorded as ``{"error": <message>}``.

NaturalBreaks is the only non-deterministic classifier (sklearn KMeans without a random_state);
its cases are recorded after ``np.random.seed(0)`` and compared statistically, with the exact
Fisher-Jenks bins of the same ``k`` stored as ``fisher_jenks_bins``. FisherJenks must match
bit for bit (mapclassify runs its dynamic program in float32); ``fisher_jenks_stress.json``
holds 40 seeded normal samples that diverge under a float64 port.

MaximumBreaks relies on ``numpy.argsort`` whose order among equal gaps is unspecified; cases
where the cut-off gap is tied are skipped and listed under ``skipped``.
"""

import json
import resource
import signal
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import mapclassify as mc  # noqa: E402
import numpy as np  # noqa: E402

from common import FIXTURES, dump_json, meta  # noqa: E402

# Some mapclassify iterations never terminate or explode in memory on degenerate inputs (an
# emptied Jenks-Caspall class turns the medians into NaN). Cap memory and wall time per case and
# record such cases as skipped instead of hanging the generator.
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
CASE_SECONDS = 10


def _alarm(_sig, _frame):
    raise TimeoutError("mapclassify did not terminate")


signal.signal(signal.SIGALRM, _alarm)

COLUMBUS = FIXTURES / "datasets" / "columbus.json"


def probes(y, bins):
    """Values around and on every bin edge, plus values beyond both ends."""
    lo, hi = float(np.min(y)), float(np.max(y))
    xs = [lo - 1.0, lo, hi, hi + 1.0, float(np.median(y))]
    for b in bins:
        xs += [float(b), float(np.nextafter(b, -np.inf)), float(np.nextafter(b, np.inf))]
    return xs


def record(y, name, ctor, params):
    case = {"method": name, "params": params}
    np.random.seed(0)
    signal.alarm(CASE_SECONDS)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            c = ctor()
    except (ValueError, IndexError) as exc:
        case["error"] = str(exc)
        return case
    except (TimeoutError, MemoryError, SystemError, ZeroDivisionError) as exc:
        case["skip"] = f"mapclassify {type(exc).__name__}"
        return case
    finally:
        signal.alarm(0)
    xs = probes(y, c.bins)
    case.update(
        {
            "bins": np.asarray(c.bins, dtype=float),
            "k": int(c.k),
            "yb": c.yb,
            "counts": c.counts,
            "tss": float(c.tss),
            "adcm": float(c.adcm),
            "gadf": float(c.gadf),
            "legend": c.get_legend_classes(),
            "find_bin": {"x": xs, "bin": c.find_bin(np.asarray(xs))},
            "warnings": len(caught),
        }
    )
    if name == "quantiles":
        # Known PySAL bug: the last percentile can fall just below 100, leaving max(y) in class 0.
        case["drift"] = bool(c.bins[-1] != np.max(y))
        case["max_class"] = int(c.yb[int(np.argmax(y))])
    if name == "box_plot":
        case["low_outlier_ids"] = c.low_outlier_ids
        case["high_outlier_ids"] = c.high_outlier_ids
        case["iqr"] = float(c.iqr)
    if name == "user_defined":
        case["lowest_param"] = params["lowest"]
    if name in ("jenks_caspall", "jenks_caspall_forced"):
        case["iterations"] = int(c.iterations)
    if name == "natural_breaks" and len(np.unique(y)) >= params["k"]:
        case["fisher_jenks_bins"] = np.asarray(mc.FisherJenks(y, k=params["k"]).bins, dtype=float)
    return case


def max_breaks_tied(y, k):
    d = np.diff(np.sort(y))
    if len(d) < k:
        return False
    s = np.sort(d)[::-1]
    return k - 1 >= 1 and k - 1 < len(s) and s[k - 2] == s[k - 1]


def cases_for(y, quantile_ks=(5, 4, 10)):
    y = np.asarray(y, dtype=float)
    out, skipped = [], []
    for k in (5, 3, 7):
        out.append(record(y, "equal_interval", lambda k=k: mc.EqualInterval(y, k=k), {"k": k}))
    for k in quantile_ks:
        out.append(record(y, "quantiles", lambda k=k: mc.Quantiles(y, k=k), {"k": k}))
    for h in (1.5, 3.0, 0.5):
        out.append(record(y, "box_plot", lambda h=h: mc.BoxPlot(y, hinge=h), {"hinge": h}))
    out.append(record(y, "std_mean", lambda: mc.StdMean(y), {"multiples": [-2, -1, 1, 2], "anchor": False}))
    out.append(record(y, "std_mean", lambda: mc.StdMean(y, multiples=[-1, 1]), {"multiples": [-1, 1], "anchor": False}))
    out.append(record(y, "std_mean", lambda: mc.StdMean(y, anchor=True), {"multiples": [-2, -1, 1, 2], "anchor": True}))
    out.append(record(y, "percentiles", lambda: mc.Percentiles(y), {"pct": [1, 10, 50, 90, 99, 100]}))
    out.append(record(y, "percentiles", lambda: mc.Percentiles(y, pct=[10, 50, 90]), {"pct": [10, 50, 90]}))
    for k in (5, 3, 8):
        if len(np.unique(y)) > 1 and max_breaks_tied(y, k):
            skipped.append({"method": "maximum_breaks", "params": {"k": k}, "reason": "tied cut-off gap"})
            continue
        out.append(record(y, "maximum_breaks", lambda k=k: mc.MaximumBreaks(y, k=k), {"k": k}))
    out.append(record(y, "head_tail_breaks", lambda: mc.HeadTailBreaks(y), {}))
    for k in (5, 3, 7):
        out.append(record(y, "fisher_jenks", lambda k=k: mc.FisherJenks(y, k=k), {"k": k}))
        out.append(record(y, "jenks_caspall", lambda k=k: mc.JenksCaspall(y, k=k), {"k": k}))
        out.append(record(y, "natural_breaks", lambda k=k: mc.NaturalBreaks(y, k=k), {"k": k}))
    for k in (5, 3):
        out.append(record(y, "jenks_caspall_forced", lambda k=k: mc.JenksCaspallForced(y, k=k), {"k": k}))
    for k in (5, 3, 7):
        out.append(record(y, "pretty_breaks", lambda k=k: mc.PrettyBreaks(y, k=k), {"k": k}))
    q = [float(np.percentile(y, p)) for p in (20, 60)]
    out.append(record(y, "user_defined", lambda: mc.UserDefined(y, bins=q), {"bins": q, "lowest": None}))
    lo = float(np.min(y)) - 1.0
    out.append(record(y, "user_defined", lambda: mc.UserDefined(y, bins=q, lowest=lo), {"bins": q, "lowest": lo}))
    for case in [c for c in out if "skip" in c]:
        out.remove(case)
        skipped.append({"method": case["method"], "params": case["params"], "reason": case["skip"]})
    return out, skipped


def write(name, description, y, quantile_ks=(5, 4, 10)):
    cases, skipped = cases_for(y, quantile_ks)
    fx = {
        "name": name,
        "description": description,
        "meta": {**meta(), "mapclassify": mc.__version__},
        "y": np.asarray(y, dtype=float),
        "cases": cases,
        "skipped": skipped,
    }
    dump_json(FIXTURES / "classify" / f"{name}.json", fx)
    print(f"{name}: {len(cases)} cases, {len(skipped)} skipped")


def stress():
    """40 normal(1e5, 3e4, n=120) samples rounded to 0.1: a float64 port of the Fisher-Jenks
    dynamic program diverges from mapclassify on one of them."""
    rng = np.random.default_rng(140)
    samples = []
    for i in range(40):
        y = np.round(rng.normal(1e5, 3e4, 120), 1)
        k = (3, 4, 5, 6)[i % 4]
        samples.append({"k": k, "y": y, "bins": np.asarray(mc.FisherJenks(y, k=k).bins, dtype=float)})
    fx = {
        "name": "fisher_jenks_stress",
        "description": "40 seeded normal samples locking the float32 Fisher-Jenks arithmetic",
        "meta": {**meta(), "mapclassify": mc.__version__},
        "samples": samples,
    }
    dump_json(FIXTURES / "classify" / "fisher_jenks_stress.json", fx)
    print(f"fisher_jenks_stress: {len(samples)} samples")


def write_max_p(name, description, y, ks=(5, 3, 7)):
    """MaxP on an ascending vector: mapclassify indexes ``self.y`` with positions of the sorted
    copy, so its output is only well defined for sorted input (known deviation)."""
    y = np.sort(np.asarray(y, dtype=float))
    cases = [record(y, "max_p", lambda k=k: mc.MaxP(y, k=k), {"k": k, "initial": 1000}) for k in ks]
    fx = {"name": name, "description": description, "meta": {**meta(), "mapclassify": mc.__version__}, "n": len(y), "y": y,
          "cases": cases, "skipped": []}
    dump_json(FIXTURES / "classify" / f"{name}.json", fx)


def write_gadf(vectors):
    """``gadf`` k-search for the deterministic methods (NaturalBreaks is unseeded)."""
    out = []
    for name, y in vectors:
        y = np.asarray(y, dtype=float)
        for method in ("Quantiles", "FisherJenks", "MaximumBreaks"):
            for pct in (0.8, 0.5, 0.95):
                k, cl, g = mc.gadf(y, method=method, maxk=15, pct=pct)
                out.append({"vector": name, "method": method, "pct": pct, "maxk": 15,
                            "k": int(k), "gadf": float(g), "bins": np.asarray(cl.bins, dtype=float)})
    fx = {"name": "gadf", "description": "mapclassify.gadf k-search", "meta": {**meta(), "mapclassify": mc.__version__},
          "vectors": {name: np.asarray(y, dtype=float) for name, y in vectors}, "cases": out}
    dump_json(FIXTURES / "classify" / "gadf.json", fx)


def main():
    cols = json.loads(COLUMBUS.read_text())["attributes"]
    for col in ("CRIME", "HOVAL", "INC"):
        write(f"columbus_{col.lower()}", f"Columbus {col} (49 values)", cols[col])
    rng = np.random.default_rng(14)
    write("lognormal500", "Seeded lognormal sample, default_rng(14), n=500", rng.lognormal(1.0, 1.0, 500))
    ties = np.concatenate([rng.integers(0, 6, 200).astype(float), [40.0, 55.5, 90.0]])
    rng.shuffle(ties)
    write("heavy_ties", "Integers 0..5 with heavy ties plus three outliers", ties)
    write("edge_small", "mapclassify doc example vector", [1, 7, 4, 12, 4, 31])
    write("edge_dups", "Quantiles with duplicate quantile values", [1, 1, 1, 1, 2, 5])
    write("edge_constant", "Constant vector", [3, 3, 3, 3])
    write("quantiles_drift", "1..46: Quantiles k=23 and 36 leave max(y) in class 0 in mapclassify", np.arange(1, 47), (5, 23, 36))
    blobs = np.concatenate([rng.normal(0, 1, 30), rng.normal(100, 1, 30), rng.normal(1000, 5, 20)])
    write("well_separated", "Three well separated normal clusters, default_rng(14) stream", np.round(blobs, 3))
    stress()
    write("edge_two", "Two values", [0.5, 2.5])
    write_max_p("max_p_crime", "Columbus CRIME, sorted (49 values)", cols["CRIME"])
    write_max_p("max_p_lognormal", "lognormal500, sorted", json.loads(
        (FIXTURES / "classify" / "lognormal500.json").read_text())["y"], ks=(5, 4))
    write_gadf([("crime", cols["CRIME"]), ("hoval", cols["HOVAL"]), ("inc", cols["INC"])])


if __name__ == "__main__":
    main()
