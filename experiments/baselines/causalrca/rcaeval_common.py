#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for running CausalRCA (PC / DAG-GNN + PageRank) on the RCAEval
RE2 datasets (re2ob / re2ss / re2tt).

Data layout (one directory per fault case):
    <DATA>/re2<sys>_<service>_<fault>_<run>/
        metrics.json     {metric_name: [[ts, value], ...]}  (metric = <service>_<type>)
        inject_time.txt   unix timestamp of fault injection
        logs.csv, traces.csv (unused here)

To keep the problem tractable and the two methods comparable we follow the
decisions made with the user:
  * keep only latency-90 + cpu + mem metrics (drop socket/diskio/workload/error
    and latency-50);
  * use a LENGTH-minute window centred on inject_time (half before, half after),
    mirroring RCAEval's main.py slicing;
  * drop constant columns, then L2-normalise each column (as in utils_microarcl).

Ranking/metrics are reported for TOTAL, per system (ob/ss/tt) and per fault type,
matching the MicroARCL summaries.
"""

import os
import json
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import preprocessing

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
DATA_DIR = os.environ.get('RCAEVAL_DATA', os.path.join(REPO_ROOT, 'data', 'RCAEval'))
KEEP_SUFFIXES = ('_cpu', '_mem', '_latency-90')
LENGTH = 10   # minutes (half before / half after inject_time)
TDELTA = 0

# re2<sys>_<service>_<fault>_<run>; service may contain hyphens (train-ticket)
_CASE_RE = re.compile(r'^re2(ob|ss|tt)_(.+)_([^_]+)_([^_]+)$')


def iter_cases():
    """Yield (case_dir_name, system, gt_service, fault) for every RE2 case."""
    for name in sorted(os.listdir(DATA_DIR)):
        if not name.startswith('re2'):
            continue
        if not os.path.isdir(os.path.join(DATA_DIR, name)):
            continue
        m = _CASE_RE.match(name)
        if not m:
            continue
        system, service, fault, _run = m.groups()
        yield name, system, service, fault


def service_of_column(col):
    """'checkoutservice_latency-90' -> 'checkoutservice'; 'ts-auth-service_cpu'
    -> 'ts-auth-service'. Metric type (the suffix) never contains '_'."""
    return col.rsplit('_', 1)[0]


def load_case(case):
    """Return a feature DataFrame (constant cols dropped, each column L2-normalised,
    no time column) for the inject-centred window, plus the ordered column names.
    Returns (None, None) if the case has no usable data."""
    case_dir = os.path.join(DATA_DIR, case)
    metrics_path = os.path.join(case_dir, 'metrics.json')
    inject_path = os.path.join(case_dir, 'inject_time.txt')
    if not (os.path.exists(metrics_path) and os.path.exists(inject_path)):
        return None, None

    with open(metrics_path) as f:
        raw = json.load(f)
    with open(inject_path) as f:
        inject = int(f.read().strip()) + TDELTA

    cols = {}
    for k, series in raw.items():
        if not k.endswith(KEEP_SUFFIXES):
            continue
        arr = np.asarray(series, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or arr.shape[0] == 0:
            continue
        cols[k] = pd.Series(arr[:, 1], index=arr[:, 0].astype(np.int64))
    if not cols:
        return None, None

    df = pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    df.index.name = 'time'
    df = df.reset_index()

    half = LENGTH * 60 // 2
    normal = df[df['time'] < inject].tail(half)
    anomal = df[df['time'] >= inject].head(half)
    win = pd.concat([normal, anomal], ignore_index=True)

    feat = win.drop(columns=['time'])
    keep = [c for c in feat.columns if feat[c].nunique() > 1]
    feat = feat[keep]
    if feat.shape[1] < 2 or feat.shape[0] < 2:
        return None, None

    normed = preprocessing.normalize(feat.values, axis=0)
    return pd.DataFrame(normed, columns=feat.columns), list(feat.columns)


# ===================================
# statistics (ACC@K / AVG@K / MRR, rank 0 = miss) -- same defs as baseline_common.py
# reported for TOTAL, per group (system) and per fault type
# ===================================
def write_summary(results, groups, out_root, method, dataset_desc, time_cost,
                  maxk=10, summary_prefix='summary'):
    def topk_metrics(ranks):
        n = len(ranks)
        acc = [(sum(1 for r in ranks if 1 <= r <= k) / n if n else 0.0)
               for k in range(1, maxk + 1)]
        avg = [sum(acc[:k]) / k for k in range(1, maxk + 1)]
        return acc, avg

    def row_stats(rows):
        ranks = [r['rank'] for r in rows]
        acc, avg = topk_metrics(ranks)
        mrr = (sum(1.0 / r for r in ranks if r >= 1) / len(ranks)) if ranks else 0.0
        seconds = sum(r['seconds'] for r in rows)
        return acc, avg, mrr, seconds, seconds / len(rows) if rows else 0.0

    def fmt_row(label, rows, stats=None):
        acc, avg, mrr, seconds, sec_per_case = stats or row_stats(rows)
        return '{:<26s}'.format(label) \
            + ''.join('{:>8.3f}'.format(v) for v in acc) \
            + ''.join('{:>8.3f}'.format(v) for v in avg) \
            + '{:>8.3f}'.format(mrr) + '{:>5d}'.format(len(rows)) \
            + '{:>12.3f}{:>12.3f}'.format(seconds, sec_per_case)

    seen_g, seen_gl = [], []
    for r in results:
        if r['group'] not in seen_g:
            seen_g.append(r['group'])
        gl = (r['group'], r['load'])
        if gl not in seen_gl:
            seen_gl.append(gl)

    all_ranks = [r['rank'] for r in results]
    hits = sum(1 for r in all_ranks if r >= 1)

    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    summary_path = os.path.join(out_root, summary_prefix + '_' + ts + '.log')
    lines = []
    lines.append('=' * 120)
    lines.append('SUMMARY (CausalRCA: ' + method + ')  dataset=' + dataset_desc)
    lines.append('=' * 120)

    lines.append('{:<60s}{:>8s}{:>12s}'.format('case', 'topK', 'TIME(s)'))
    for r in results:
        lines.append('{:<60s}{:>8d}{:>12.3f}'.format(
            r['group'] + '/' + r['load'] + '/' + r['case'], r['rank'], r['seconds']))
    lines.append('-' * 120)

    lines.append('Top-K ranking metrics (rank 0/missing = miss)')
    header = '{:<26s}'.format('subset') \
        + ''.join('{:>8s}'.format('ACC@%d' % k) for k in range(1, maxk + 1)) \
        + ''.join('{:>8s}'.format('AVG@%d' % k) for k in range(1, maxk + 1)) \
        + '{:>8s}'.format('MRR') + '{:>5s}'.format('n') \
        + '{:>12s}{:>12s}'.format('TIME(s)', 's/case')
    lines.append(header)
    lines.append(fmt_row('TOTAL', results))
    system_rows = []
    for g in seen_g:
        gr = [r for r in results if r['group'] == g]
        system_rows.append(gr)
        lines.append(fmt_row('[' + g + ']', gr))
    if system_rows:
        stats = [row_stats(rows) for rows in system_rows]
        macro = ([sum(s[0][k] for s in stats) / len(stats) for k in range(maxk)],
                 [sum(s[1][k] for s in stats) / len(stats) for k in range(maxk)],
                 sum(s[2] for s in stats) / len(stats),
                 sum(s[3] for s in stats),
                 sum(s[4] for s in stats) / len(stats))
        lines.append(fmt_row('MACRO AVG (systems)', results, macro))
    for (g, l) in seen_gl:
        lr = [r for r in results if r['group'] == g and r['load'] == l]
        lines.append(fmt_row('  ' + g + '/' + l, lr))

    lines.append('')
    lines.append('Samples: {}  hits: {}  misses: {}'.format(len(all_ranks), hits, len(all_ranks) - hits))
    lines.append('Time cost: {:.3f}s'.format(sum(r['seconds'] for r in results)))
    lines.append('This invocation wall time: {:.3f}s'.format(time_cost))
    summary_text = '\n'.join(lines)
    print(summary_text)
    with open(summary_path, 'w') as f:
        f.write(summary_text + '\n')
    print('summary written: ' + summary_path)
