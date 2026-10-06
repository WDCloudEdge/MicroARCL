#!/usr/bin/env python
"""Run RCAEval's TORAI method on the MicroARCL agent-network dataset.

TORAI (RCAEval/e2e/torai.py) is a multi-source RCA method: per-modality anomaly
scoring (metrics + logs + traces), Gaussian-Mixture clustering of the per-service
modality-score matrix, then RCD-based within-cluster refinement.

The MicroARCL agent data is not in RCAEval's torai-* layout, so this adapter
builds TORAI's three modality inputs from the raw per-case files:

  metric : agent-network/metrics/svc_metric.csv  ("<service>&<metric>" columns,
           5 s cadence, own `timestamp` column). Renamed to "<service>_<metric>"
           and up-sampled to a 1 s grid so TORAI's internal ``::15`` sub-sample
           lands on a true 15 s cadence, exactly as on the published 1 s data.
  logts  : agent-network/log/*_<service>.log line counts bucketed at 15 s.
  trace  : derived from the *execution graphs* (agent-network/graph/graph_json)
           via execution_graph.py. These carry no per-trace wall-clock timestamp,
           so instead of a time-windowed series we inject a per-service
           failed-vs-success contrast straight into TORAI's GMM feature matrix
           through the ``trace_scores=`` hook added to torai():
             err = number of traces that traverse the service and fail
                   (task_status 3/8, or a vertex error signature)
             lat = max(0, mean vertex execution time on failed traces
                          - mean vertex execution time on successful traces)

inject_time is each case's ``fault_start`` (from the per-service *_label.txt).

Usage:
    python run_torai_agent.py [LOAD_DIR]
Default LOAD_DIR is data/MDOC/abnormal/load-5.

Must run under RCAEval's .venv-torai (py3.8 + patched causal-learn), because the
within-cluster refinement uses RCD's localized PC algorithm.
"""
# --- path bootstrap: shared engine stays at repo root; add it + sibling runner dirs ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.exists(_os.path.join(_r, 'baseline_common.py')):
    _r = _os.path.dirname(_r)
for _p in (_r, _os.path.join(_r, 'experiments', 'microarcl'),
           _os.path.join(_r, 'experiments', 'baselines', 'microrca'),
           _os.path.join(_r, 'experiments', 'baselines', 'torai')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
import os
import re
import sys
import glob
import json
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import execution_graph as EG  # MicroARCL module (pure python)
from RCAEval.e2e.torai import torai

DEFAULT_LOAD_DIR = os.path.join(
    HERE, "data", "MDOC", "abnormal", "load-5")

NS = "agent-network"          # k8s namespace / metrics sub-dir name
BUCKET = 15                   # log/trace bucket seconds, matches TORAI
FAULTS = ["cpu_load", "mem_load", "net_latency", "pod_failure", "pod_kill"]

_LOG_TS_RE = re.compile(r"^(\S+)\s")


# ---------------------------------------------------------------------------
# labels
# ---------------------------------------------------------------------------
def parse_labels(load_dir):
    """Parse every ``<service>_label.txt`` into {case_name: {...}}.

    Each label file holds one block per case, headed ``=== <case_name> ===``,
    with ``root_cause``, ``fault_type`` and the unix ``fault_start`` we use as
    inject_time.
    """
    labels = {}
    for lf in glob.glob(os.path.join(load_dir, "*_label.txt")):
        with open(lf, encoding="utf-8") as fh:
            text = fh.read()
        blocks = re.split(r"^===\s*(.+?)\s*===\s*$", text, flags=re.M)
        # blocks: ['', name1, body1, name2, body2, ...]
        for i in range(1, len(blocks), 2):
            name = blocks[i].strip()
            body = blocks[i + 1]

            def _field(key):
                m = re.search(rf"^{key}:\s*(.+)$", body, flags=re.M)
                return m.group(1).strip() if m else None

            def _unix(key):
                v = _field(key)
                if not v:
                    return None
                m = re.search(r"\((\d+)\)", v)
                return int(m.group(1)) if m else None

            rc = _field("root_cause") or ""
            labels[name] = {
                "service": _field("service"),
                "fault_type": _field("fault_type"),
                "root_cause_service": rc.split("/")[0] if rc else _field("service"),
                "fault_start": _unix("fault_start"),
                "window_start": _unix("window_start"),
                "window_end": _unix("window_end"),
            }
    return labels


# ---------------------------------------------------------------------------
# metric modality
# ---------------------------------------------------------------------------
def build_metric(metrics_dir):
    """svc_metric.csv -> 1 s metric frame with a unix `time` column."""
    df = pd.read_csv(os.path.join(metrics_dir, "svc_metric.csv"))
    tcol = next((c for c in df.columns if c.lower() == "timestamp"), None)
    if tcol is None:
        # fall back to instance.csv timestamps (row-aligned, same cadence)
        inst = pd.read_csv(os.path.join(metrics_dir, "instance.csv"))
        n = min(len(df), len(inst))
        df = df.iloc[:n].reset_index(drop=True)
        ts = inst["timestamp"].iloc[:n]
    else:
        ts = df[tcol]
        df = df.drop(columns=[tcol])

    unix = pd.to_datetime(ts, utc=True).astype("int64") // 10 ** 9
    df.columns = [c.replace("&", "_") for c in df.columns]
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    df["time"] = unix.values

    # up-sample to a full 1 s grid (ffill) so TORAI's ::15 -> real 15 s
    df = df.drop_duplicates(subset=["time"]).set_index("time").sort_index()
    full = pd.RangeIndex(int(df.index.min()), int(df.index.max()) + 1)
    df = df.reindex(full).ffill().bfill()
    df.index.name = "time"
    return df.reset_index()


def _svc_of_col(col):
    """Recover the hyphenated service name that prefixes a metric column.

    Services are hyphen-only (e.g. ``agent-network-pdf-parsing``); the metric
    suffix begins at the first underscore.
    """
    return col.split("_")[0]


def detect_onset(metric, warmup_frac=0.2, min_norm=0.15, min_abn=0.15,
                 k=3.0, persist=3):
    """Unsupervised anomaly onset on the metric frame (has a unix ``time``).

    Fully self-contained: it uses only the metric data, never the label's
    fault_start. Per column it computes a robust z-score (median / MAD over the
    whole window), reduces to the per-timestep *breadth* of anomalous metrics,
    smooths it, then returns the first time this breadth crosses a warmup
    baseline (mean + k*std) and stays above it for ``persist`` steps. Because
    the MAD scale is estimated over the whole (mostly loaded) window, a gradual
    load ramp is absorbed into the baseline and the sharp fault becomes the
    dominant crossing. The onset is clamped into ``[min_norm, 1-min_abn]`` so
    both the normal and abnormal windows TORAI needs stay non-empty; a peak
    fallback is used when nothing crosses.

    Returns the onset unix time (usable directly as inject_time).
    """
    t = metric["time"].to_numpy()
    X = metric.drop(columns=["time"]).to_numpy(dtype=float)
    T = X.shape[0]
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med), axis=0)
    mad[mad == 0] = 1e-9
    z = np.abs((X - med) / (1.4826 * mad))
    breadth = (z > 3).mean(axis=1)
    s = pd.Series(breadth).rolling(15, min_periods=1, center=True).mean().to_numpy()

    w = max(int(warmup_frac * T), 5)
    base_mu, base_sd = s[:w].mean(), s[:w].std() + 1e-9
    thr = base_mu + k * base_sd
    lo, hi = int(min_norm * T), int((1 - min_abn) * T)
    onset = None
    for i in range(max(w, lo), max(hi, lo + 1)):
        if np.all(s[i:i + persist] > thr):
            onset = i
            break
    if onset is None:
        onset = int(np.clip(np.argmax(s), lo, max(hi - 1, lo)))
    return int(t[onset])


def adaptive_inject_time(metrics_dir, ns_dir):
    """inject_time (and window end) from the MAIN-LINE method's own detector,
    run_agent_rca.adaptive_window (Module B.0), so TORAI's normal/abnormal split
    is computed *identically* to the proposed method rather than by TORAI's own
    detector. Replicates run_all_abnormal_overlap_merge_lag.py's setup: align=True
    with the default lag mode. Returns (start_unix, end_unix).
    """
    from Config import Config
    import run_agent_rca as R
    from lag_align import (align_metrics_df, compute_service_lags,
                           compute_chain_lags)

    cfg = Config()
    sr, lat, _qps = R.load_kpi(metrics_dir)
    if getattr(cfg, "lag_mode", "self") == "chain":
        lag_map = compute_chain_lags(metrics_dir, ns_dir, cfg)
    else:
        lag_map = compute_service_lags(metrics_dir, cfg)
    aligned_lat = align_metrics_df(lat, lag_map) if lag_map else lat
    start, end, _ = R.adaptive_window(
        sr, cfg, metrics_dir=metrics_dir, latency=aligned_lat, lag_map=lag_map)
    epoch = pd.Timestamp("1970-01-01")

    def _to_unix(ts):
        return int((pd.Timestamp(ts) - epoch) // pd.Timedelta(seconds=1))

    return _to_unix(start), _to_unix(end)


# ---------------------------------------------------------------------------
# log modality
# ---------------------------------------------------------------------------
def _parse_leading_unix(line):
    m = _LOG_TS_RE.match(line)
    if not m:
        return None
    try:
        ts = pd.Timestamp(m.group(1))
    except Exception:
        return None
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return int(ts.timestamp())


def build_logts(ns_dir, services, window_start, window_end):
    """Per-service log-line counts bucketed at 15 s over the case window."""
    n = int((window_end - window_start) // BUCKET) + 1
    grid = [window_start + i * BUCKET for i in range(n)]
    counts = {svc: [0] * n for svc in services}
    logdir = os.path.join(ns_dir, "log")
    for svc in services:
        for path in glob.glob(os.path.join(logdir, "*_" + svc + ".log")):
            try:
                fh = open(path, encoding="utf-8", errors="ignore")
            except Exception:
                continue
            with fh:
                for line in fh:
                    t = _parse_leading_unix(line)
                    if t is None or not (window_start <= t <= window_end):
                        continue
                    b = int((t - window_start) // BUCKET)
                    if 0 <= b < n:
                        counts[svc][b] += 1
    data = {f"{svc}_logcount": counts[svc] for svc in services}
    data["time"] = grid
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# trace modality (execution graphs, failed-vs-success contrast)
# ---------------------------------------------------------------------------
def build_trace_scores(ns_dir, services):
    graphs = EG.load_execution_graphs(ns_dir, max_traces=None)
    n_failed = defaultdict(int)
    lat_failed = defaultdict(list)
    lat_success = defaultdict(list)

    for g in graphs:
        chain = set(EG._trace_service_chain(g))
        failed_services = (EG._status_failed_services(g)
                           | EG._signature_failed_services(g))
        svc_lat = defaultdict(float)
        for key in g.ordered_vertexes:
            svc = EG.group_to_service(key)
            if svc is None:
                continue
            v = g.vertexes.get(key)
            t = getattr(v, "time", None) if v is not None else None
            if t:
                svc_lat[svc] += float(t)
        for svc in chain:
            failed = svc in failed_services
            if failed:
                n_failed[svc] += 1
                lat_failed[svc].append(svc_lat.get(svc, 0.0))
            else:
                lat_success[svc].append(svc_lat.get(svc, 0.0))

    scores = {}
    for svc in services:
        err = float(n_failed.get(svc, 0))
        mf = float(np.mean(lat_failed[svc])) if lat_failed[svc] else 0.0
        ms = float(np.mean(lat_success[svc])) if lat_success[svc] else 0.0
        lat = max(0.0, mf - ms)
        if err > 0 or lat > 0:
            scores[svc] = {"err": err, "lat": lat}
    return scores, len(graphs)


# ---------------------------------------------------------------------------
# one case
# ---------------------------------------------------------------------------
def run_case(case_dir, label, onset="label"):
    ns_dir = os.path.join(case_dir, NS)
    metrics_dir = os.path.join(ns_dir, "metrics")

    metric = build_metric(metrics_dir)
    services = sorted({_svc_of_col(c) for c in metric.columns if c != "time"})

    if onset == "auto":
        # unsupervised: TORAI is given no fault time; the split and the
        # log/trace window are derived from the data itself.
        inject_time = detect_onset(metric)
        ws, we = int(metric["time"].min()), int(metric["time"].max())
    elif onset == "adaptive":
        # unsupervised, but using the MAIN-LINE method's own window detector so
        # the split is identical to the proposed method (only the RCA algorithm
        # differs). Log/trace buckets still span the full data range.
        inject_time, _ = adaptive_inject_time(metrics_dir, ns_dir)
        ws, we = int(metric["time"].min()), int(metric["time"].max())
    else:
        inject_time = label["fault_start"]
        ws, we = label["window_start"], label["window_end"]

    # Guard against a degenerate split: if the detected/label inject_time falls
    # at (or outside) the metric's time range, the normal or abnormal window is
    # empty and TORAI's preprocess/drop_constant blows up on an empty frame.
    # Clamp so both sides keep a few 15 s points (metric is 1 s here). This only
    # bites pathological onsets (e.g. adaptive_window landing on the first
    # sample); normal splits are well inside and unchanged.
    tmin, tmax = int(metric["time"].min()), int(metric["time"].max())
    margin = 15 * 3
    inject_time = int(min(max(inject_time, tmin + margin), tmax - margin))

    logts = build_logts(ns_dir, services, ws, we)
    trace_scores, n_graphs = build_trace_scores(ns_dir, services)

    data = {
        "metric": metric,
        "logts": logts,
        "tracets_err": pd.DataFrame(),
        "tracets_lat": pd.DataFrame(),
    }
    out = torai(
        data,
        inject_time=inject_time,
        dataset="torai-agent",
        sli=None,
        dk_select_useful=False,
        verbose=False,
        trace_scores=trace_scores,
    )
    ranks = out.get("ranks", [])

    # "<service>_A" -> service, de-duplicated, order preserved
    pred = []
    for r in ranks:
        svc = r.split("_")[0]
        if svc not in pred:
            pred.append(svc)
    meta = {"n_services": len(services), "n_graphs": n_graphs,
            "n_trace_scored": len(trace_scores), "inject_time": inject_time}
    if onset in ("auto", "adaptive") and label.get("fault_start"):
        # +ve => detector fired before the true fault (clean normal window)
        meta["onset_lead"] = label["fault_start"] - inject_time
    return pred, meta


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
CUTOFFS = tuple(range(1, 11))          # ACC@1..10 / AVG@1..10, as in the main line


def _topk_metrics(ranks):
    """ACC@k and AVG@n over 1..10, identical convention to the main-line
    run_all_abnormal._topk_metrics: rank is 1-based, 0 = miss;
    ACC@k = fraction with 0 < rank <= k; AVG@n = mean of ACC@1..n."""
    total = len(ranks)
    if total == 0:
        return {k: 0.0 for k in CUTOFFS}, {n: 0.0 for n in CUTOFFS}
    acc = {k: sum(0 < r <= k for r in ranks) / total for k in CUTOFFS}
    avg = {n: sum(acc[k] for k in range(1, n + 1)) / n for n in CUTOFFS}
    return acc, avg


def _mrr(ranks):
    """Mean reciprocal rank; a miss (rank 0) contributes 0."""
    total = len(ranks)
    return (sum(1.0 / r for r in ranks if r) / total) if total else 0.0


def _discover_load_dirs(paths):
    """Expand each path to the set of load dirs under it. A load dir is any
    directory that directly contains ``*_label.txt``; a passed path that is
    itself a load dir is kept as-is, otherwise it is searched recursively. This
    lets the caller pass a single 'abnormal' root and pick up single/load-3,
    single/load-5, multi/load-3-multi, ... automatically."""
    found = []
    for p in paths:
        p = os.path.abspath(p)
        if glob.glob(os.path.join(p, "*_label.txt")):
            found.append(p)
            continue
        for lf in glob.glob(os.path.join(p, "**", "*_label.txt"), recursive=True):
            found.append(os.path.dirname(lf))
    # de-duplicate, stable order
    return sorted(dict.fromkeys(found))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run TORAI on the MicroARCL agent data")
    ap.add_argument("paths", nargs="*", default=[DEFAULT_LOAD_DIR],
                    help="load-* dirs, or any parent (e.g. .../abnormal, or "
                         ".../abnormal/single); every labeled load dir found "
                         "underneath is discovered and pooled into one "
                         "evaluation, with per-group (single/multi) and per-load "
                         "breakdowns (default: load-5)")
    ap.add_argument("--onset", choices=["label", "auto", "adaptive"],
                    default="label",
                    help="inject_time source: 'label' = ground-truth fault_start "
                         "(supervised, oracle split); 'auto' = TORAI's own "
                         "unsupervised detector on the metrics; 'adaptive' = the "
                         "main-line method's own adaptive_window (Module B.0), so "
                         "the window is computed identically to the proposed "
                         "method and only the RCA algorithm differs.")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of cases (smoke test)")
    args = ap.parse_args()
    load_dirs = _discover_load_dirs(args.paths)
    onset = args.onset
    if not load_dirs:
        raise SystemExit(f"no labeled load dirs (with *_label.txt) found under {args.paths}")

    # (group, load, case_dir, label) across every discovered load dir. group is
    # the load dir's parent name (single / multi); case names repeat across loads
    # (agent-network-image_cpu_load_1 exists in each), so we key on the full path.
    cases = []
    for ld in load_dirs:
        group = os.path.basename(os.path.dirname(ld))
        load = os.path.basename(ld)
        lbls = parse_labels(ld)
        for cd in sorted(glob.glob(os.path.join(ld, "*_*"))):
            if os.path.isdir(cd) and os.path.basename(cd) in lbls:
                cases.append((group, load, cd, lbls[os.path.basename(cd)]))
    if args.limit:
        cases = cases[:args.limit]

    ds_env = 'MARBLEBench' if any('MARBLEBench' in d for d in load_dirs) else 'MDOC'
    os.environ.setdefault('AGENT_DATASET', ds_env)
    from baseline_common import _category, CATEGORIES
    import results_log
    DS = {'MDOC': 'MDOC', 'MARBLEBench': 'MAR'}[ds_env]
    hdr = ['# TORAI on ' + ', '.join(
        f"{os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}"
        for d in load_dirs), f'# onset={onset}', f'# {len(cases)} cases']
    with results_log.open_batch(DS, 'TORAI', 'TORAI', hdr) as (log_root, ts):
        rows = []
        for group, load, cd, lab in cases:
            name = os.path.basename(cd)
            gt = lab["root_cause_service"]
            ft = lab["fault_type"]
            results_log.sample_banner(f"{group}/{load}", name, gt, ft)
            started = time.perf_counter()
            try:
                pred, meta = run_case(cd, lab, onset=onset)
                rank = (pred.index(gt) + 1) if gt in pred else 0
                print(f"top5={pred[:5]} (svc={meta.get('n_services')} "
                      f"graphs={meta.get('n_graphs')} "
                      f"trace_scored={meta.get('n_trace_scored')})")
                results_log.sample_eval(rank)
                rows.append({"group": group, "load": load, "case": name,
                             "fault": ft, "gt": gt, "category": _category(gt),
                             "rank": rank, "top1": pred[0] if pred else None,
                             "status": "success",
                             "timing": {"total": time.perf_counter() - started}})
            except Exception as e:
                traceback.print_exc()
                rows.append({"group": group, "load": load, "case": name,
                             "fault": ft, "gt": gt, "category": _category(gt),
                             "rank": 0, "status": "failed", "error": str(e),
                             "timing": {"total": time.perf_counter() - started}})
        results_log.summarize("TORAI", rows, kind="agent", categories=CATEGORIES)
    results_log.write_sublogs(log_root, ts, "TORAI", rows, kind="agent",
                              categories=CATEGORIES)


if __name__ == "__main__":
    main()
