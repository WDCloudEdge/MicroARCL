"""Baseline on RCAEval RE2: Birch detection + Personalized PageRank.

RE2 port of run_all_abnormal_birch_ppr.py (method='ppr', detector='birch'). The
Birch anomaly strength is the PPR personalization vector; a random walk with
restart over the trace-derived call topology (call_chains.json) then ranks the
services. Each RE2 case is materialized into the MicroARCL layout by re2_adapter
(+ a graph.csv the PPR graph builder needs).

PPR requires trace spans with parentSpanID for each case. Cases without trace
data fail explicitly instead of silently creating a placeholder topology.

Usage:
    python run_all_abnormal_RE2_birch_ppr.py --root /path/to/RCAEval/data \
        --suite re2ob [--window adaptive|label] [--align]
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
import sys
import time
import argparse
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

from Config import Config
from run_agent_rca import _service_rank
from baseline_common import _run_ppr
import re2_adapter as A
import results_log
from experiments.microarcl.run_all_abnormal_RE2_birch_align_MicroARCL import (
    NS, _summarize, _fault_cat, _resolve_root)
from run_all_abnormal_RE2_birch_labelwin_MicroARCL import _label_window


def _run_batch(root, out_root, tag, window='adaptive', align=False, pre=180,
               post=180, limit=None, suite=None, force=False, birch_thr=0.03):
    cfg = Config()
    cfg.agent_namespace = NS
    cfg.anomaly_threshold = birch_thr                 # RE2 Birch scale (see align runner)
    cases = A.discover_cases(root)
    if suite:
        cases = [c for c in cases if suite in os.path.basename(os.path.normpath(c))]
    if limit:
        cases = cases[:limit]
    missing = [c for c in cases if A._case_file(c, 'traces') is None]
    if missing:
        raise FileNotFoundError(
            f'{len(missing)} RE2 cases have no traces.csv or traces.parquet; '
            f'first missing case: {missing[0]}')
    print(f'RE2 birch+PPR [{tag}]  root={root}  window={window}  align={align}')
    print(f'Found {len(cases)} cases (suite={suite or "all"}).')
    rows = []
    for cdir in cases:
        case = os.path.basename(os.path.normpath(cdir))
        gt, fault, _ = A.parse_case_label(cdir)
        results_log.sample_banner(_fault_cat(fault), case, gt, fault)
        started = time.perf_counter()
        timing, res, err = {}, None, None
        try:
            m0 = time.perf_counter()
            sample_dir, inject_time, services = A.materialize_case(
                cdir, out_root, ns=NS, force=force)
            A.ensure_graph_csv(sample_dir, ns=NS)     # PPR graph builder input
            timing['materialize'] = time.perf_counter() - m0
            if window == 'label':
                metrics_dir = os.path.join(sample_dir, NS, 'metrics')
                cfg.label_window = _label_window(metrics_dir, inject_time,
                                                 pre, post)
            if gt not in services:
                print(f'[warn] gt {gt!r} not among {len(services)} services.')
            res = _run_ppr(cfg, sample_dir, align, detector='birch')
            timing.update(res.get('timing') or {})
        except Exception as exc:
            err = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        finally:
            cfg.label_window = None
        timing['total'] = time.perf_counter() - started
        base = {'case': case, 'gt': gt, 'fault': fault,
                'category': _fault_cat(fault), 'timing': timing}
        if not res:
            rows.append({**base, 'status': 'failed', 'rank': 0, 'error': err})
            continue
        rank = _service_rank(res['order'], gt)
        rows.append({**base, 'status': 'success', 'rank': rank,
                     'top1': res['order'][0] if res['order'] else None})
        print(f'[eval] gt rank = {rank}')
    results_log.summarize(tag, rows, kind='re2')
    return rows


SUITE2DS = {'re2ss': 'SS', 're2ob': 'OB', 're2tt': 'TT'}
METHOD = 'MicroRCA'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None)
    ap.add_argument('--out', default=os.path.join('output', 're2_materialized'))
    ap.add_argument('--suite', default=None, help='e.g. re2ob / re2tt')
    ap.add_argument('--window', choices=('adaptive', 'label'), default='adaptive',
                    help="detection window: data-driven (default) or inject-time")
    ap.add_argument('--align', action='store_true',
                    help='enable QPS<->metric lag alignment (default off, as in '
                         'the birch_ppr baseline)')
    ap.add_argument('--pre', type=int, default=180)
    ap.add_argument('--post', type=int, default=180)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--birch-thr', type=float, default=0.03)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    root = _resolve_root(args.root)
    os.makedirs(args.out, exist_ok=True)
    tag = 'RE2_birch_ppr' + (f'_{args.window}' if args.window != 'adaptive' else '')
    suites = [args.suite] if args.suite else list(SUITE2DS)
    for su in suites:
        ds = SUITE2DS.get(su, su)
        hdr = [f'MicroRCA (RE2 birch+PPR) on {ds}  root={root}  '
               f'window={args.window}  align={args.align}']
        with results_log.open_batch(ds, METHOD, tag, hdr) as (log_root, ts):
            rows = _run_batch(root, args.out, tag, window=args.window,
                              align=args.align, pre=args.pre, post=args.post,
                              limit=args.limit, suite=su, force=args.force,
                              birch_thr=args.birch_thr)
        results_log.write_sublogs(log_root, ts, tag, rows, kind='re2')


if __name__ == '__main__':
    main()
