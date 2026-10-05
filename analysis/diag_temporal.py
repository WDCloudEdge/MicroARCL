"""RQ2 flattened-temporal-ordering diagnostic (paper S5.2.2) over the
MicroASBench failure runs: anomaly onset-order partition (root-first /
near-sync / victim-first), median root onset lead, and the fraction of zero
QPS->resource lags. Onsets come from run_onset_propagation (Birch, alignment
off => raw observed order); lags from lag_align.compute_service_lags.

Input : data/<dataset>/abnormal/<group>/<load>/<sample>, enumerated via
        baseline_common._discover (--dataset MDOC|MARBLEBench,
        --loads <load>..., --from window|load|fault)
Output: console summary
"""
import os
import sys
import argparse
import warnings
from contextlib import redirect_stdout, redirect_stderr

import numpy as np

# This script lives in analysis/ but reuses the pipeline modules at the repo
# root; make them importable regardless of the invocation directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Resolve --dataset BEFORE importing the project modules: run_all_abnormal,
# baseline_common and run_onset_propagation fix the abnormal-sample root from
# AGENT_DATASET at import time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--dataset', choices=('MDOC', 'MARBLEBench'), default='MDOC')
_DATASET = _pre.parse_known_args()[0].dataset
os.environ['AGENT_DATASET'] = _DATASET

from Config import Config                                         # noqa: E402
from run_all_abnormal import parse_labels                          # noqa: E402
from baseline_common import _discover                              # noqa: E402
from lag_align import compute_service_lags                         # noqa: E402
from run_onset_propagation import (_service_onsets, _classify,     # noqa: E402
                                   _onset_floor)

warnings.filterwarnings('ignore')


def _run(loads_filter, since_mode):
    cfg = Config()
    tol = cfg.step
    results = []            # per-sample onset classification + onset std
    lag_total = lag_zero = lag_mid = lag_taumax = 0
    tau_max = getattr(cfg, 'lag_tau_max', 12)

    for group, load, abn in _discover(None):
        if loads_filter and load not in loads_filter:
            continue
        labels = parse_labels(abn)
        samples = [(n, os.path.join(abn, n)) for n in sorted(labels)
                   if os.path.isdir(os.path.join(abn, n))]
        for sname, sdir in samples:
            kv = labels[sname]
            gt = kv.get('service', '')
            tag = f'{group}/{load}'
            try:
                with open(os.devnull, 'w') as dn, \
                        redirect_stdout(dn), redirect_stderr(dn):
                    _service_onsets._kv = kv
                    since_ts = _onset_floor(cfg, kv, since_mode)
                    onsets = _service_onsets(cfg, sdir, align=False,
                                             since_ts=since_ts)
                    metrics_dir = os.path.join(sdir, cfg.agent_namespace,
                                               'metrics')
                    lag_map = compute_service_lags(metrics_dir, cfg)
            except Exception as exc:
                print(f'  {tag:<18} {sname[:40]:<40} FAILED: '
                      f'{type(exc).__name__}: {exc}')
                continue
            # lag stats
            for v in lag_map.values():
                lag_total += 1
                if v == 0:
                    lag_zero += 1
                elif v >= tau_max:
                    lag_taumax += 1
                elif 3 <= v <= 9:
                    lag_mid += 1
            # onset classification + cross-service onset dispersion
            r = _classify(onsets, gt, tol)
            onset_std = (float(np.std(list(onsets.values())))
                         if len(onsets) >= 2 else float('nan'))
            r.update({'group': group, 'load': load, 'name': sname,
                      'onset_std': onset_std})
            results.append(r)
            lead = r['lead_seconds']
            lead_s = 'n/a' if lead is None else f'{lead:+d}s'
            ostd = 'n/a' if not np.isfinite(onset_std) else f'{onset_std:.0f}s'
            print(f'  {tag:<18} {sname[:40]:<40} anom={r["n_anomalous"]:<2} '
                  f'rootdet={str(r["root_detected"]):<5} '
                  f'cat={str(r["category"]):<13} '
                  f'lead={lead_s:<6} onset_std={ostd}')
    return results, (lag_total, lag_zero, lag_mid, lag_taumax, tau_max)


def _pct(k, n):
    return f'{(k / n * 100 if n else 0):5.1f}%  ({k}/{n})'


def _summary(dataset, results, lag):
    lag_total, lag_zero, lag_mid, lag_taumax, tau_max = lag
    evaluable = [r for r in results
                 if r['category'] in ('root_first', 'near_sync',
                                      'victim_first')]
    detected = [r for r in results if r['root_detected']]
    n_eval = len(evaluable)
    leads = [r['lead_seconds'] for r in evaluable
             if r['lead_seconds'] is not None]
    onset_stds = [r['onset_std'] for r in results
                  if np.isfinite(r['onset_std'])]

    print('\n' + '=' * 78)
    print(f'SUMMARY [{dataset}]  (paper: lag-zero ~62%, root-first 15% / '
          f'near-sync 35% / victim-first 50%, median lead -5s)')
    print('=' * 78)
    print(f'Samples processed ........ {len(results)}')
    print(f'Root detected ............ {_pct(len(detected), len(results))}')
    print(f'Order-evaluable .......... {n_eval}')

    print('\n--- (a) ONSET DISPERSION across services (per-sample std, s) ---')
    if onset_stds:
        a = np.asarray(onset_stds)
        print(f'  median={np.median(a):.1f}s  mean={a.mean():.1f}s  '
              f'[{np.percentile(a, 25):.1f}, {np.percentile(a, 75):.1f}]  '
              f'n={a.size}   (sampling interval = {Config().step}s)')

    print('\n--- (b) LAG ESTIMATES (service x metric, compute_service_lags) ---')
    print(f'  total lags ............. {lag_total}')
    print(f'  == 0 (synchronous) ..... {_pct(lag_zero, lag_total)}')
    print(f'  3..9 steps (mid) ....... {_pct(lag_mid, lag_total)}')
    print(f'  >= tau_max={tau_max} (pseudo) ... {_pct(lag_taumax, lag_total)}')

    print('\n--- (c) ONSET-ORDER PARTITION (evaluable) ---')
    for cat in ('root_first', 'near_sync', 'victim_first'):
        k = sum(1 for r in evaluable if r['category'] == cat)
        print(f'  {cat:<13} {_pct(k, n_eval)}')
    rank1 = sum(1 for r in evaluable if r.get('root_rank_by_onset') == 1)
    print(f'  root among earliest (ties counted): {_pct(rank1, n_eval)}')
    if leads:
        print(f'  root onset lead over closest affected (s): '
              f'median={np.median(leads):+.1f}  mean={np.mean(leads):+.1f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('MDOC', 'MARBLEBench'),
                    default='MDOC')
    ap.add_argument('--loads', nargs='*', default=None,
                    help='restrict to these load subsets (default: all)')
    ap.add_argument('--from', dest='since', default='window',
                    choices=('window', 'load', 'fault'),
                    help="onset floor: 'window' (default), 'load', or 'fault' "
                         '(drops pre-fault baseline detections)')
    args = ap.parse_args()
    print(f'Temporal diagnostic on {args.dataset}  '
          f'loads={args.loads or "all"}  from={args.since}\n')
    results, lag = _run(args.loads, args.since)
    if not results:
        print('\nNo samples processed.')
        return
    _summary(args.dataset, results, lag)


if __name__ == '__main__':
    main()
