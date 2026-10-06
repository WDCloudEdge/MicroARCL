#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared config & helpers for the motivation analysis scripts."""
import os
import glob
import json
import collections
import numpy as np
import pandas as pd
import matplotlib
from cycler import cycler
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (re-exported for scripts)
from matplotlib.text import Text

# ---- paper font: Times New Roman, vector PDF with embedded fonts ----
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "dejavuserif"
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType (editable/vector text)
matplotlib.rcParams["ps.fonttype"] = 42

# Okabe-Ito-inspired colors for clear separation in print and for color-vision deficiency.
PALETTE = {"1 user": "#0072B2", "3 users": "#009E73", "5 users": "#D55E00"}
COLORS = {
    "trace": "#0072B2",
    "trace_p99": "#005A8D",
    "metric_p50": "#E69F00",
    "metric_p99": "#D55E00",
    "metric_distinct": "#D55E00",
    "cpu": "#0072B2",
    "memory": "#E69F00",
    "coverage": "#009E73",
    "entry": "#333333",
    "middle": "#0072B2",
    "exit": "#D55E00",
    "variance": "#0072B2",
    "reference": "#666666",
}
matplotlib.rcParams["axes.prop_cycle"] = cycler(color=list(PALETTE.values()))

# ---- paths ----
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "analysis", "figures")
TAB = os.path.join(ROOT, "analysis", "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def save_fig(fig, name):
    """Save a figure as vector PDF (paper) and PNG (preview). `name` has no ext."""
    fig.savefig(os.path.join(FIG, name + ".pdf"))          # vector
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=160)  # raster preview


def scale_figure_text(fig, factor=2):
    """Scale all text in one figure without changing other figures."""
    for text in fig.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() * factor)

# ---- datasets (NORMAL runs, 1/3/5 concurrent users, single-replica) ----
# NOTE: loads are 1/3/5 users. The 10-user runs were dropped: 10 users overloads the
# system and its data is distorted (survivorship/partial-timing artifacts), so 3-user
# (a valid load) was collected instead. Trace-based analysis drops only aborted tasks
# (running & total_level<=1) via load_traces.
DATASETS = {
    "1 user":  "data/MDOC/normal/20260904-14.10-14.15-normal-thingo-1user-5min_14.20",
    "3 users": "data/MDOC/normal/20260913-14.40-14.45-normal-thingo-3user-5min-14.50",
    "5 users": "data/MDOC/normal/20260904-14.30-14.35-normal-thingo-5user-5min_14.50",
}
ORDER = list(DATASETS.keys())
SAMPLE_SEC = 5  # metric sampling interval


def mpath(root, fname):
    """Absolute path to a metrics CSV inside a dataset."""
    return os.path.join(ROOT, root, "agent-network", "metrics", fname)


def cv_series(series):
    """Coefficient of variation for a metric COLUMN (needs >=5 valid samples)."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 5:
        return np.nan
    m = s.mean()
    if m == 0 or np.isnan(m):
        return np.nan
    return s.std() / abs(m)


def cv_pos(seq):
    """Coefficient of variation over the positive values of a sequence."""
    a = np.asarray([x for x in seq if x is not None], float)
    a = a[a > 0]
    return float(a.std() / a.mean()) if len(a) else np.nan


def is_exec_span(node):
    """True for an ACTUAL-EXECUTION span (Group/Agent, name contains '/').

    Bare group-name spans (no '/', e.g. 'OCRParserGroup') are pure routing/
    dispatch markers with time≈0 and are excluded from all timing statistics.
    """
    return "/" in node


# Drop only "aborted" tasks: still-running (task_status == 1) chains that never got
# past the planner (total_level <= 1) — these mostly died early (stillborn). Every
# other trace is kept (running chains with real length, and failed/finished tasks).
# task_status is read per-trace from the dataset's call_chains.json.
ABORTED_STATUS = 1


def load_traces(root):
    """Load real call chains from graph/graph_json/*.json for a dataset.

    Returns a list of per-trace dicts with ordered levels and per-node time (s).
    Only actual-execution spans (see ``is_exec_span``) are kept, and aborted tasks
    (running with total_level <= 1, see ``ABORTED_STATUS``) are dropped.
    """
    graph_dir = os.path.join(ROOT, root, "agent-network", "graph")
    status = {}
    cc = os.path.join(graph_dir, "call_chains.json")
    if os.path.exists(cc):
        try:
            for c in json.load(open(cc, encoding="utf-8")):
                status[c.get("trace_id")] = c.get("task_status")
        except Exception:
            status = {}
    out = []
    for f in glob.glob(os.path.join(graph_dir, "graph_json", "*.json")):
        try:
            g = json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        if status.get(g.get("trace_id")) == ABORTED_STATUS and (g.get("total_level") or 0) <= 1:
            continue  # drop aborted (stillborn) running tasks
        details = sorted(g.get("level_details", []) or [], key=lambda d: d.get("level", 0))
        levels = []
        for lvl in details:
            spans = lvl.get("level_spans") or {}
            node_times = {k: sp.get("time") for k, sp in spans.items()
                          if is_exec_span(k) and isinstance(sp.get("time"), (int, float))}
            nodes = [k for k in spans.keys() if is_exec_span(k)]
            if not nodes:
                nodes = [v for v in (lvl.get("level_vertexes") or []) if is_exec_span(v)]
            levels.append({
                "nodes": nodes,
                "time": float(sum(node_times.values())) if node_times else 0.0,
                "node_times": node_times,
            })
        out.append({
            "trace_id": g.get("trace_id"),
            "total_level": g.get("total_level") or len(levels),
            "e2e_time": g.get("time"),
            "token": g.get("token"),
            "levels": levels,
            "path": [n for lv in levels for n in lv["nodes"]],
        })
    return out


def trace_node_times(traces):
    """Aggregate loaded traces into {vertex_group -> list of per-node times (s)}."""
    times = collections.defaultdict(list)
    for tr in traces:
        for lv in tr["levels"]:
            for node, t in lv["node_times"].items():
                times[node.split("/")[0]].append(float(t))
    return times
