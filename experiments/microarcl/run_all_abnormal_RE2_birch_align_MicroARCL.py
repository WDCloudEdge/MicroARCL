"""MicroARCL on the RCAEval RE2 datasets (metrics + traces).

Same localizer configuration as run_all_abnormal_birch_align_MicroARCL.py --
Birch detection on the lag-aligned window, reliability-weighted (wsum) fusion of
the raw per-signal voters within the Birch top-15, then the chain-lag penalty
(mu=1.0) -- but run over RE2 cases instead of the agent-network dataset. Each RE2
case is materialized into the MicroARCL on-disk layout by re2_adapter so the
detector, reliability fusion and chain-lag penalty run byte-for-byte unchanged;
only the data/label reading is new.

Usage:
    python run_all_abnormal_RE2_birch_align_MicroARCL.py \
        --root /path/to/RCAEval/data/re2ob [--limit N] [--suite re2ob]

--root defaults to $RE2_ROOT, then to the RCAEval checkout beside this repo.
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
from run_all_abnormal import TOP_K_CUTOFFS, _topk_metrics, _Tee
from run_agent_rca import _service_rank
from baseline_common import _run_relrrf
import re2_adapter as A


NS = 're2'
FAULTS = ('cpu', 'mem', 'disk', 'delay', 'loss', 'socket')

_DEFAULT_ROOTS = [
    os.environ.get('RE2_ROOT', ''),
    os.path.join(os.path.dirname(__file__),
                 '..', '..', '..', '078-WHU', 'researchProject', 'RCAEval', 'data'),
    os.path.join(os.path.dirname(__file__), 'data', 're2'),
]


def _resolve_root(arg):
    for cand in ([arg] if arg else []) + _DEFAULT_ROOTS:
        if cand and os.path.isdir(cand) and A.discover_cases(cand):
            return cand
    raise SystemExit(
        'No RE2 cases found. Pass --root <dir> (a directory of RCAEval RE2 '
        'case folders, each with metrics.json + traces.csv + inject_time.txt).')


def _fault_cat(fault):
    f = (fault or '').lower()
    for k in FAULTS:
        if k in f:
            return k
    return 'other'


def _summarize(tag, rows):
    print('\n\n' + '=' * 96 + f'\nSUMMARY [{tag}]\n' + '=' * 96)
    print(f'{"case":<40}{"fault":<10}{"rank":<6}{"gt":<22}{"top1":<22}')
    n_hit = 0
    for r in rows:
        n_hit += 1 if r.get('rank') == 1 else 0
        print(f'{str(r.get("case",""))[:38]:<40}{str(r.get("fault",""))[:9]:<10}'
              f'{str(r.get("rank", 0)):<6}{str(r.get("gt",""))[:20]:<22}'
              f'{str(r.get("top1") or "")[:20]:<22}')
    tot = len(rows)
    print('-' * 96)
    print(f'Localization accuracy  top1={n_hit}/{tot}')
    print('\nRanking metrics (rank 0/missing = miss)')
    print(f'{"stage":<22}' + ''.join(f'{"ACC@"+str(k):>9}' for k in TOP_K_CUTOFFS)
          + ''.join(f'{"AVG@"+str(n):>9}' for n in TOP_K_CUTOFFS)
          + f'{"MRR":>9}{"n":>5}')

    def _pm(label, subset):
        if not subset:
            return
        acc, avg = _topk_metrics(subset, 'rank')
        mrr = sum(1.0 / r['rank'] for r in subset if r.get('rank')) / len(subset)
        vals = [acc[k] for k in TOP_K_CUTOFFS] + [avg[n] for n in TOP_K_CUTOFFS]
        print(f'{label:<22}' + ''.join(f'{v:>9.3f}' for v in vals)
              + f'{mrr:>9.3f}{len(subset):>5}')

    _pm(tag, rows)
    for cat in FAULTS + ('other',):
        _pm(f'  [{cat}]', [r for r in rows if r.get('category') == cat])
    succ = sum(r.get('status') == 'success' for r in rows)
    print(f'\nBatch  completed={tot} successful={succ} failed={tot - succ}')
    for stage in ('materialize', 'detect', 'localize', 'total'):
        v = [r['timing'][stage] for r in rows if stage in r.get('timing', {})]
        if v:
            print(f'  mean {stage}: {sum(v)/len(v):.3f}s ({len(v)})')


def _run_batch(root, out_root, tag, limit=None, suite=None, force=False,
               birch_thr=0.03):
    cfg = Config()
    cfg.agent_namespace = NS
    # RE2 windows are long (~1440s/case), so Birch's L2 normalization yields much
    # smaller values than on the agent dataset; the native 0.07 radius then never
    # splits (a 40x CPU spike stays a single cluster). 0.03 restores detection.
    cfg.anomaly_threshold = birch_thr
    # RE2 only: the reliability latency voter uses the p90 tail (= latency-90),
    # where delay faults show up, instead of the p50 median used on the agent set.
    cfg.lat_percentile = 'p90'
    cases = A.discover_cases(root)
    if suite:
        cases = [c for c in cases if suite in os.path.basename(os.path.normpath(c))]
    if limit:
        cases = cases[:limit]
    print(f'RE2 MicroARCL [{tag}]  root={root}')
    print(f'Found {len(cases)} cases (suite={suite or "all"}).')
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
            sample_dir, _inj, services = A.materialize_case(
                cdir, out_root, ns=NS, force=force)
            timing['materialize'] = time.perf_counter() - m0
            if gt not in services:
                print(f'[warn] gt {gt!r} not among {len(services)} materialized '
                      f'services; rank will be a miss.')
            res = _run_relrrf(cfg, sample_dir, align=True, detector='birch',
                              variant='both', k_rrf=60, include_residual=False,
                              k=15, mu=1.0, fuse='wsum')
            if not res.get('order'):
                # Birch gate silent (e.g. a mostly-faulty adaptive window, where
                # Birch's L2 normalization compresses the signal under its
                # threshold). Fall back to full-set reliability fusion so the
                # localizer still ranks rather than throwing the case away.
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
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--birch-thr', type=float, default=0.03,
                    help='Birch anomaly threshold (RE2 default 0.03)')
    ap.add_argument('--force', action='store_true',
                    help='re-materialize even if a cached case dir exists')
    args = ap.parse_args()

    root = _resolve_root(args.root)
    os.makedirs(args.out, exist_ok=True)
    tag = 'RE2_birch_align_MicroARCL' + (f'_{args.suite}' if args.suite else '')
    ts = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_path = os.path.join(args.out, f'batch-{tag}_{ts}.log')
    t0 = time.perf_counter()
    with open(log_path, 'w', encoding='utf-8') as lf:
        with redirect_stdout(_Tee(sys.stdout, lf)), \
                redirect_stderr(_Tee(sys.stderr, lf)):
            print(f'Batch start: {datetime.now().isoformat(timespec="seconds")}')
            print(f'Log: {log_path}')
            try:
                _run_batch(root, args.out, tag, limit=args.limit,
                           suite=args.suite, force=args.force,
                           birch_thr=args.birch_thr)
            finally:
                print(f'\nBatch wall-clock: {time.perf_counter() - t0:.1f}s')


if __name__ == '__main__':
    main()
