#!/usr/bin/env python
"""Run RCAEval's TORAI on the RCAEval RE2 datasets (re2ob / re2ss / re2tt), metrics-only, with the same three onset modes as run_torai_agent.py:"""
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
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import re2_adapter as A
from run_torai_agent import (detect_onset, adaptive_inject_time,
                             CUTOFFS, _topk_metrics, _mrr)
from RCAEval.e2e.torai import torai
import results_log

SUITE2DS = {"re2ss": "SS", "re2ob": "OB", "re2tt": "TT"}
RE2_SUITES = ("re2ob", "re2ss", "re2tt")
FAULTS = ("cpu", "mem", "disk", "delay", "loss", "socket")
DEFAULT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "data", "RCAEval")


def build_metric_re2(case_dir):
    """RE2 metrics.json -> a 1 s TORAI metric frame with a unix `time` column.

    Mirrors main.py's torai preprocessing: keep one latency per service
    (latency-90 -> _latency, drop -50/-99), clean inf/na. Column names stay
    `<service>_<kind>`; service is hyphen/no-sep (never '_'), so TORAI's
    split('_')[0] recovers it unambiguously.
    """
    df = A._read_metrics(case_dir)
    df = df.loc[:, ~df.columns.str.endswith("_latency-50")]
    df = df.loc[:, ~df.columns.str.endswith("_latency-99")]
    df = df.rename(columns={c: c.replace("_latency-90", "_latency")
                            for c in df.columns if c.endswith("_latency-90")})
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return df


def run_case(case_dir, onset, out_root):
    metric = build_metric_re2(case_dir)
    tmin, tmax = int(metric["time"].min()), int(metric["time"].max())

    if onset == "auto":
        inject_time = detect_onset(metric)
    elif onset == "adaptive":
        # materialize the RE2 case into the MicroARCL layout, then run the
        # main-line adaptive_window on it (cached across modes / reruns).
        sample_dir, _inj, _svcs = A.materialize_case(case_dir, out_root, ns="re2")
        ns_dir = os.path.join(sample_dir, "re2")
        inject_time, _ = adaptive_inject_time(os.path.join(ns_dir, "metrics"), ns_dir)
    else:
        inject_time = A.read_inject_time(case_dir)

    # keep both normal/abnormal windows non-empty (see run_torai_agent)
    margin = 15 * 3
    inject_time = int(min(max(inject_time, tmin + margin), tmax - margin))

    # metrics-only: a time-only logts keeps TORAI's log modality empty without
    # tripping drop_constant on an empty frame; no trace modality.
    logts = pd.DataFrame({"time": metric["time"].values})
    data = {"metric": metric, "logts": logts,
            "tracets_err": pd.DataFrame(), "tracets_lat": pd.DataFrame()}

    out = torai(data, inject_time=inject_time, dataset="torai-re2", sli=None,
                dk_select_useful=False, verbose=False)
    ranks = out.get("ranks", [])
    pred = []
    for r in ranks:
        svc = r.split("_")[0]
        if svc not in pred:
            pred.append(svc)
    return pred, inject_time


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run TORAI on the RCAEval RE2 datasets")
    ap.add_argument("root", nargs="?", default=DEFAULT_ROOT,
                    help="RCAEval data dir holding re2ob_*/re2ss_*/re2tt_* cases")
    ap.add_argument("--onset", choices=["label", "auto", "adaptive"],
                    default="label")
    ap.add_argument("--suites", default=",".join(RE2_SUITES),
                    help="comma-separated suite prefixes to include")
    ap.add_argument("--out", default=os.path.join("output", "re2_torai_materialized"),
                    help="cache dir for the MicroARCL materialization (adaptive)")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of cases (smoke test)")
    args = ap.parse_args()
    onset = args.onset
    suites = tuple(s.strip() for s in args.suites.split(",") if s.strip())
    os.makedirs(args.out, exist_ok=True)

    cases = [c for c in A.discover_cases(os.path.abspath(args.root))
             if os.path.basename(c).split("_")[0] in suites]
    if args.limit:
        cases = cases[:args.limit]

    for su in suites:
        ds = SUITE2DS.get(su, su)
        su_cases = [c for c in cases if os.path.basename(c).split("_")[0] == su]
        if not su_cases:
            continue
        hdr = [f"TORAI on {ds} (RE2, metrics-only)  onset={onset}  "
               f"{len(su_cases)} cases"]
        with results_log.open_batch(ds, "TORAI", "TORAI", hdr) as (log_root, ts):
            rows = []
            for cd in su_cases:
                name = os.path.basename(cd)
                gt, fault, _inst = A.parse_case_label(cd)
                results_log.sample_banner(fault, name, gt, fault)
                started = time.perf_counter()
                try:
                    pred, _inject = run_case(cd, onset, args.out)
                    rank = (pred.index(gt) + 1) if gt in pred else 0
                    print(f"top5={pred[:5]}")
                    results_log.sample_eval(rank)
                    rows.append({"case": name, "gt": gt, "fault": fault,
                                 "category": fault, "rank": rank,
                                 "top1": pred[0] if pred else None,
                                 "status": "success",
                                 "timing": {"total": time.perf_counter() - started}})
                except Exception as e:
                    traceback.print_exc()
                    rows.append({"case": name, "gt": gt, "fault": fault,
                                 "category": fault, "rank": 0, "status": "failed",
                                 "error": str(e),
                                 "timing": {"total": time.perf_counter() - started}})
            results_log.summarize("TORAI", rows, kind="re2")
        results_log.write_sublogs(log_root, ts, "TORAI", rows, kind="re2")


if __name__ == "__main__":
    main()
