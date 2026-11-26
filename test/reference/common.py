"""Shared helpers for the PySAL oracle fixture generators (dev-only; CI never runs Python).

Conventions
-----------
* Output is deterministic JSON: sorted keys, floats written with ``repr`` precision (shortest
  string that round-trips), one scalar array per line so diffs stay reviewable.
* JSON has no NaN/Infinity, so non-finite floats are written as the strings ``"nan"``,
  ``"inf"`` and ``"-inf"``. ``Spatex.Reference.load!/1`` decodes them. No string datum
  (such as an id) may therefore equal one of those markers.
* Every fixture carries a ``meta`` block with the library versions that produced it.
* Graphs are exported with *positional* indices ``0..n-1`` into ``ids`` (the graph's
  ``unique_ids`` order), never with the original labels, so Elixir code needs no id lookup.
"""

import json
import math
from pathlib import Path

import esda
import libpysal
import numpy as np
import pandas as pd

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def meta():
    """Versions of the libraries that generated a fixture."""
    return {
        "libpysal": libpysal.__version__,
        "esda": esda.__version__,
        "numpy": np.__version__,
    }


def _plain(obj):
    """Convert numpy/pandas values to plain JSON-ready Python, encoding non-finite floats."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, (np.ndarray, pd.Index, pd.Series)):
        return [_plain(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return x
    return obj


def _render(obj, indent=0):
    pad = " " * (indent + 1)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [
            f'{pad}{json.dumps(k)}: {_render(obj[k], indent + 1)}' for k in sorted(obj)
        ]
        return "{\n" + ",\n".join(items) + "\n" + " " * indent + "}"
    if isinstance(obj, list):
        if all(not isinstance(v, (dict, list)) for v in obj):
            return json.dumps(obj, separators=(", ", ": "))
        items = [f"{pad}{_render(v, indent + 1)}" for v in obj]
        return "[\n" + ",\n".join(items) + "\n" + " " * indent + "]"
    return json.dumps(obj)


def dump_json(path, obj):
    """Write ``obj`` deterministically (see module docstring) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _render(_plain(obj)) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def export_graph(graph):
    """Export a ``libpysal.graph.Graph``'s adjacency as positional arrays.

    Returns ``{"ids", "focal", "neighbor", "weight"}`` where ``ids`` is
    ``graph.unique_ids`` and ``focal``/``neighbor`` are integer positions into ``ids``.
    Rows keep ``Graph.adjacency`` order (focal-sorted). Weights are floats.

    Islands are represented exactly as ``Graph.adjacency`` does: a single self-loop row
    ``(i, i, 0.0)`` with zero weight. Such a row contributes nothing to sums, lags or
    s0/s1/s2, and the node is also listed under ``isolates``.
    """
    ids = list(graph.unique_ids)
    pos = pd.Index(ids).get_indexer
    adj = graph.adjacency.reset_index()
    return {
        "ids": ids,
        "focal": pos(adj["focal"]).tolist(),
        "neighbor": pos(adj["neighbor"]).tolist(),
        "weight": adj["weight"].astype(float).tolist(),
    }


def graph_stats(graph):
    """s0, s1, s2 of a graph, computed on the graph's own (possibly transformed) weights.

    s0 = sum_ij w_ij; s1 = 1/2 sum_ij (w_ij + w_ji)^2; s2 = sum_i (sum_j w_ij + sum_j w_ji)^2.
    These are the PySAL definitions; the values are cross-checked against
    ``graph.to_W()`` (``W.s0/s1/s2``) and the generator fails if they disagree.
    """
    ids = list(graph.unique_ids)
    n = len(ids)
    pos = pd.Index(ids).get_indexer
    adj = graph.adjacency.reset_index()
    m = np.zeros((n, n))
    m[pos(adj["focal"]), pos(adj["neighbor"])] = adj["weight"].to_numpy(float)
    s0 = m.sum()
    s1 = 0.5 * ((m + m.T) ** 2).sum()
    s2 = ((m.sum(axis=1) + m.sum(axis=0)) ** 2).sum()
    w = graph.to_W()
    for name, mine in (("s0", s0), ("s1", s1), ("s2", s2)):
        theirs = getattr(w, name)
        if not math.isclose(mine, theirs, rel_tol=1e-12, abs_tol=1e-12):
            raise AssertionError(f"{name}: dense {mine!r} != W {theirs!r}")
    return {"s0": s0, "s1": s1, "s2": s2}
