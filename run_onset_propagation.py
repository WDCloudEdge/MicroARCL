"""Onset-propagation traceability experiment (rebuttal / motivation evidence)."""
import os
import re
import sys
import glob
import argparse
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import pandas as pd


def _mean(xs):
    return sum(xs) / len(xs)


def _median(xs):
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2

import run_agent_rca as R
from run_all_abnormal import ABN, parse_labels, _Tee
from baseline_common import _to_service
from Config import Config
from anomaly_detection import get_anomaly_by_df
from lag_align import (align_metrics_df, compute_service_lags,
                       compute_chain_lags)
from util.utils import time_string_2_timestamp


def _label_window(kv):
    """Epoch (begin, end) from a parsed label block; None if unavailable.

Inputs: kv. Outputs: result."""
    wr = kv.get('window_range', '')
    m = re.findall(r'\d{9,}', wr)
    if len(m) >= 2:
        return int(m[0]), int(m[1])
    begin = re.search(r'\((\d{9,})\)', kv.get('window_start', ''))
    end = re.search(r'\((\d{9,})\)', kv.get('window_end', ''))
    if begin and end:
        return int(begin.group(1)), int(end.group(1))
    return None


def _label_epoch(kv, field):
    """Parenthesised epoch of a label field (e.g.

Inputs: kv, field. Outputs: result."""
    m = re.search(r'\((\d{9,})\)', kv.get(field, ''))
    return int(m.group(1)) if m else None


def _onset_floor(cfg, kv, mode):
    """Earliest timestamp an onset may be credited to, per --from mode.

Inputs: cfg, kv, mode. Outputs: result."""
    if mode == 'load':
        ref = _label_epoch(kv, 'load_start')
        return ref
    if mode == 'fault':
        ref = _label_epoch(kv, 'fault_start')
        return None if ref is None else ref - cfg.step
    return None


def _service_onsets(cfg, sample_dir, align, since_ts=None):
    """Detect anomalies and return {service: onset_epoch} over the labelled (or, as fallback, adaptive) analysis window.

Inputs: cfg, sample_dir, align, since_ts. Outputs: result."""
    ns_dir = os.path.join(sample_dir, cfg.agent_namespace)
    metrics_dir = os.path.join(ns_dir, 'metrics')
    services = [c for c in pd.read_csv(
        os.path.join(metrics_dir, 'success_rate.csv'), nrows=1).columns
        if c != 'timestamp']

    lag_map = {}
    if align:
        lag_map = (compute_chain_lags(metrics_dir, ns_dir, cfg)
                   if getattr(cfg, 'lag_mode', 'self') == 'chain'
                   else compute_service_lags(metrics_dir, cfg))

    win = _label_window(_service_onsets._kv)
    if win is not None:
        start_ts, end_ts = win
    else:
        sr, lat, qps = R.load_kpi(metrics_dir)
        aligned_lat = align_metrics_df(lat, lag_map) if lag_map else lat
        start, end, _ = R.adaptive_window(
            sr, cfg, metrics_dir=metrics_dir, latency=aligned_lat,
            lag_map=lag_map)
        start_ts = int(time_string_2_timestamp(str(start)))
        end_ts = int(time_string_2_timestamp(str(end)))

    _anomalies, ats = get_anomaly_by_df(
        cfg, sample_dir, metrics_dir, 'onset', start_ts, end_ts,
        lag_map=lag_map)

    onsets = {}
    for node, points in ats.items():
        svc = _to_service(node, services)
        if svc is None or not points:
            continue
        ts = [time_string_2_timestamp(str(t)) for t in points]
        if since_ts is not None:
            ts = [t for t in ts if t >= since_ts]
        if not ts:
            continue
        node_onset = min(ts)
        if svc not in onsets or node_onset < onsets[svc]:
            onsets[svc] = node_onset
    return onsets


def _classify(onsets, gt, tol):
    """Per-sample onset-order classification.

Inputs: onsets, gt, tol. Outputs: result."""
    res = {'n_anomalous': len(onsets), 'root_detected': gt in onsets,
           'root_onset': onsets.get(gt), 'category': None,
           'earliest_other_svc': None, 'earliest_other_onset': None,
           'lead_seconds': None, 'root_rank_by_onset': None,
           'deltas': []}
    if gt not in onsets:
        res['category'] = 'root_undetected'
        return res
    t_r = onsets[gt]
    others = {s: t for s, t in onsets.items() if s != gt}

    res['deltas'] = sorted((t - t_r) for t in others.values())

    res['root_rank_by_onset'] = 1 + sum(1 for t in others.values() if t < t_r)
    if not others:
        res['category'] = 'isolated'
        return res
    v, t_v = min(others.items(), key=lambda kv: kv[1])
    lead = t_v - t_r
    res['earliest_other_svc'] = v
    res['earliest_other_onset'] = t_v
    res['lead_seconds'] = lead
    if lead > tol:
        res['category'] = 'root_first'
    elif lead < -tol:
        res['category'] = 'victim_first'
    else:
        res['category'] = 'near_sync'
    return res


def _run_batch(align, tag, since_mode):
    cfg = Config()
    tol = cfg.step
    labels = parse_labels(ABN)
    samples = [(n, os.path.join(ABN, n)) for n in sorted(labels)
               if os.path.isdir(os.path.join(ABN, n))]
    print(f'Onset-propagation experiment [{tag}]  align={align}  '
          f'onset_floor={since_mode}  tol=1 interval={tol}s')
    print(f'Found {len(samples)} labelled samples with data.\n')

    rows = []
    for name, sdir in samples:
        kv = labels[name]
        gt = kv.get('service', '')
        fault = kv.get('fault_type', name)
        print('#' * 70 + f'\n##### {name} (gt={gt}, fault={fault})')
        try:
            _service_onsets._kv = kv
            since_ts = _onset_floor(cfg, kv, since_mode)
            onsets = _service_onsets(cfg, sdir, align, since_ts=since_ts)
        except Exception as exc:
            print(f'  FAILED: {type(exc).__name__}: {exc}')
            traceback.print_exc()
            rows.append({'name': name, 'fault': fault, 'gt': gt,
                         'category': 'failed', 'n_anomalous': 0,
                         'root_detected': False})
            continue
        r = _classify(onsets, gt, tol)
        r.update({'name': name, 'fault': fault, 'gt': gt})
        rows.append(r)
        lead = r['lead_seconds']
        print(f"  anomalous={r['n_anomalous']:>2}  root_detected="
              f"{str(r['root_detected']):<5}  root_onset_rank="
              f"{str(r['root_rank_by_onset']):<4}  "
              f"lead={('n/a' if lead is None else f'{lead:+d}s'):<6}  "
              f"-> {r['category']}")

    _summary(rows, tol, tag)
    _write_csv(rows, tag)
    return rows


def _summary(rows, tol, tag):
    print('\n\n' + '=' * 80 + f'\nSUMMARY [{tag}]\n' + '=' * 80)
    hdr = (f'{"fault":<14}{"gt":<26}{"n_anom":<8}{"root_det":<9}'
           f'{"onset_rank":<11}{"lead(s)":<9}{"category":<14}')
    print(hdr)
    for r in rows:
        lead = r.get('lead_seconds')
        print(f'{str(r.get("fault",""))[:13]:<14}{str(r.get("gt",""))[:25]:<26}'
              f'{r.get("n_anomalous",0):<8}{str(r.get("root_detected",False)):<9}'
              f'{str(r.get("root_rank_by_onset","")):<11}'
              f'{("" if lead is None else f"{lead:+d}"):<9}'
              f'{str(r.get("category","")):<14}')
    print('-' * 80)

    total = len(rows)
    detected = [r for r in rows if r.get('root_detected')]

    evaluable = [r for r in rows
                 if r.get('category') in ('root_first', 'near_sync',
                                          'victim_first')]
    isolated = [r for r in rows if r.get('category') == 'isolated']

    def _pct(k, n):
        return f'{(k / n * 100 if n else 0):5.1f}%  ({k}/{n})'



    root_first_incl_iso = [r for r in detected
                           if r.get('root_rank_by_onset') == 1]
    near_sync = [r for r in evaluable if r.get('category') == 'near_sync']

    print(f'\nSamples ..................... {total}')
    print(f'Root detected as anomalous .. {_pct(len(detected), total)}')
    print(f'Order-evaluable (root+>=1 affected) . {len(evaluable)}'
          f'   isolated(root-only)={len(isolated)}')

    print('\n--- HEAD-LINE METRICS ---')
    print(f'1) Root-first ratio ......... {_pct(len(root_first_incl_iso), total)}'
          f'   [root is earliest anomalous service; over all samples]')
    print(f'   Root-first (evaluable only) {_pct(sum(1 for r in evaluable if r.get("root_rank_by_onset") == 1), len(evaluable))}')
    print(f'2) Near-synchronous ratio ... {_pct(len(near_sync), len(evaluable))}'
          f'   [|t_r - closest affected| <= {tol}s; over evaluable]')

    print('\n--- SAMPLE ONSET-ORDER PARTITION (evaluable) ---')
    for cat in ('root_first', 'near_sync', 'victim_first'):
        k = sum(1 for r in evaluable if r.get('category') == cat)
        print(f'  {cat:<14} {_pct(k, len(evaluable))}')

    leads = [r['lead_seconds'] for r in evaluable
             if r.get('lead_seconds') is not None]
    if leads:
        print(f'\nRoot onset lead over closest affected service (s): '
              f'mean={_mean(leads):+.1f}  median={_median(leads):+.1f}  '
              f'min={min(leads):+d}  max={max(leads):+d}')


    all_d = [d for r in evaluable for d in r.get('deltas', [])]
    if all_d:
        rf = sum(1 for d in all_d if d > tol)
        ns = sum(1 for d in all_d if abs(d) <= tol)
        vf = sum(1 for d in all_d if d < -tol)
        n = len(all_d)
        print(f'\nPair-level Delta_(r,v)=t_v-t_r over {n} (root,affected) pairs:')
        print(f'  root-first  (Delta > {tol}s) . {_pct(rf, n)}')
        print(f'  near-sync   (|Delta|<={tol}s) . {_pct(ns, n)}')
        print(f'  victim-first(Delta < -{tol}s) . {_pct(vf, n)}')


def _write_csv(rows, tag):
    ts = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    path = os.path.join(ABN, f'onset_propagation_{tag}_{ts}.csv')
    cols = ['name', 'fault', 'gt', 'category', 'n_anomalous', 'root_detected',
            'root_onset', 'root_rank_by_onset', 'earliest_other_svc',
            'earliest_other_onset', 'lead_seconds']
    df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
    df.to_csv(path, index=False)
    print(f'\nPer-sample results written: {path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--align', action='store_true',
                    help='apply B.1 lag alignment before detection '
                         '(sensitivity check; off by default)')
    ap.add_argument('--from', dest='since', default='window',
                    choices=('window', 'load', 'fault'),
                    help="onset floor: 'window' full label window (default), "
                         "'load' workload start, 'fault' injection start "
                         '(drops pre-fault baseline detections)')
    args = ap.parse_args()
    tag = ('aligned' if args.align else 'raw') + f'_{args.since}'

    started = datetime.now().astimezone()
    log_path = os.path.join(
        ABN, f'onset_propagation_{tag}_'
             f'{started.strftime("%Y-%m-%d-%H:%M:%S")}.log')
    with open(log_path, 'w', encoding='utf-8') as lf:
        with redirect_stdout(_Tee(sys.stdout, lf)), \
                redirect_stderr(_Tee(sys.stderr, lf)):
            print(f'Batch start: {started.isoformat(timespec="seconds")}')
            print(f'Batch log: {log_path}')
            try:
                _run_batch(args.align, tag, args.since)
            finally:
                print(f'\nBatch log written: {log_path}')


if __name__ == '__main__':
    main()
