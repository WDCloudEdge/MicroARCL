"""MicroARCL on RCAEval RE2 using the LABEL time window for Birch."""
import os
import sys
import time
import argparse
import traceback
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime

import pandas as pd

from Config import Config
from run_agent_rca import _service_rank
from baseline_common import _run_relrrf
from util.utils import time_string_2_timestamp
import re2_adapter as A
# reuse the shared discovery / summary / fault-category helpers
from experiments.microarcl.run_all_abnormal_RE2_birch_align_MicroARCL import (
    NS, _summarize, _fault_cat, _resolve_root)


def _data_span(metrics_dir):
    """(first, last) unix seconds of the materialized timeline."""
    ts = pd.read_csv(os.path.join(metrics_dir, 'success_rate.csv'),
                     usecols=['timestamp'])['timestamp']
    return (time_string_2_timestamp(str(ts.iloc[0])),
            time_string_2_timestamp(str(ts.iloc[-1])))


def _label_window(metrics_dir, inject_time, pre, post):
    """[inject-pre, inject+post] clamped to the data span, as tz-naive UTC
    Timestamps matching the CSV timeline."""
    first, last = _data_span(metrics_dir)
    start = max(first, inject_time - pre)
    end = min(last, inject_time + post)
    if start >= end:                                  # degenerate -> full span
        start, end = first, last
    return pd.to_datetime(start, unit='s'), pd.to_datetime(end, unit='s')


def _run_batch(root, out_root, tag, pre, post, limit=None, suite=None,
               force=False, birch_thr=0.03):
    cfg = Config()
    cfg.agent_namespace = NS
    cfg.anomaly_threshold = birch_thr                 # RE2 scale (see align runner)
    cfg.lat_percentile = 'p90'                         # RE2: p90 latency voter
    cases = A.discover_cases(root)
    if suite:
        cases = [c for c in cases if suite in os.path.basename(os.path.normpath(c))]
    if limit:
        cases = cases[:limit]
    print(f'RE2 MicroARCL (label window) [{tag}]  root={root}')
    print(f'Found {len(cases)} cases (suite={suite or "all"}); '
          f'window=[inject-{pre}s, inject+{post}s].')
    rows = []
    for cdir in cases:
        case = os.path.basename(os.path.normpath(cdir))
        gt, fault, _ = A.parse_case_label(cdir)
        print('\n' + '#' * 70 + f'\n##### {case} (gt={gt}, fault={fault})\n'
              + '#' * 70)
        started = time.perf_counter()
        timing, res, err = {}, None, None
        try:
            m0 = time.perf_counter()
            sample_dir, inject_time, services = A.materialize_case(
                cdir, out_root, ns=NS, force=force)
            timing['materialize'] = time.perf_counter() - m0
            metrics_dir = os.path.join(sample_dir, NS, 'metrics')
            start, end = _label_window(metrics_dir, inject_time, pre, post)
            print(f'[label window] {start} .. {end} (inject={inject_time})')
            cfg.label_window = (start, end)
            if gt not in services:
                print(f'[warn] gt {gt!r} not among {len(services)} services; '
                      f'rank will be a miss.')
            res = _run_relrrf(cfg, sample_dir, align=True, detector='birch',
                              variant='both', k_rrf=60, include_residual=False,
                              k=15, mu=1.0, fuse='wsum')
            if not res.get('order'):                  # Birch still silent -> full set
                print('[fallback] empty Birch candidate set -> full-set '
                      'reliability fusion (k=0)')
                res = _run_relrrf(cfg, sample_dir, align=True, detector='birch',
                                  variant='both', k_rrf=60,
                                  include_residual=False, k=0, mu=1.0,
                                  fuse='wsum')
                res['fallback'] = True
            timing.update(res.get('timing') or {})
        except Exception as exc:
            err = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        finally:
            cfg.label_window = None                    # never leak into next case
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
    _summarize(tag, rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=None,
                    help='directory of RE2 case folders (or a parent of suites)')
    ap.add_argument('--out', default=os.path.join('output', 're2_materialized'),
                    help='cache dir for the materialized MicroARCL layout')
    ap.add_argument('--suite', default=None,
                    help='substring filter on case name, e.g. re2ob / re2tt')
    ap.add_argument('--pre', type=int, default=180,
                    help='seconds of normal segment before inject_time')
    ap.add_argument('--post', type=int, default=180,
                    help='seconds of abnormal segment after inject_time')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--birch-thr', type=float, default=0.03,
                    help='Birch anomaly threshold (RE2 default 0.03)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()

    from run_all_abnormal import _Tee
    root = _resolve_root(args.root)
    os.makedirs(args.out, exist_ok=True)
    tag = 'RE2_birch_labelwin_MicroARCL' + (f'_{args.suite}' if args.suite else '')
    ts = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_path = os.path.join(args.out, f'batch-{tag}_{ts}.log')
    t0 = time.perf_counter()
    with open(log_path, 'w', encoding='utf-8') as lf:
        with redirect_stdout(_Tee(sys.stdout, lf)), \
                redirect_stderr(_Tee(sys.stderr, lf)):
            print(f'Batch start: {datetime.now().isoformat(timespec="seconds")}')
            print(f'Log: {log_path}')
            try:
                _run_batch(root, args.out, tag, args.pre, args.post,
                           limit=args.limit, suite=args.suite, force=args.force,
                           birch_thr=args.birch_thr)
            finally:
                print(f'\nBatch wall-clock: {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
