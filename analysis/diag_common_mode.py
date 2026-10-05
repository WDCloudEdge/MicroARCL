"""RQ2 common-mode diffusion diagnostic (paper S5.2.1) over the MicroASBench
failure runs: per-sample first-PC variance ratio eta and mean pairwise
correlation rho_bar of the per-service CPU-usage deviation sequences (PCA on
the correlation matrix, so an equal-variance independent reference gives
eta = 1/N).

Input : data/<dataset>/abnormal/<group>/<load>/<sample>, enumerated via
        baseline_common._discover (--dataset MDOC|MARBLEBench, --align optional)
Output: console summary + data/<dataset>/diag_common_mode_{raw,aligned}.csv
"""
import os
import sys
import argparse
import warnings
from contextlib import redirect_stdout, redirect_stderr

import numpy as np
import pandas as pd

# This script lives in analysis/ but reuses the pipeline modules at the repo
# root; make them importable regardless of the invocation directory.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Resolve --dataset BEFORE importing the project modules: run_all_abnormal and
# baseline_common fix the abnormal-sample root from AGENT_DATASET at import time.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--dataset', choices=('MDOC', 'MARBLEBench'), default='MDOC')
_DATASET = _pre.parse_known_args()[0].dataset
os.environ['AGENT_DATASET'] = _DATASET

from Config import Config                       # noqa: E402
from run_all_abnormal import parse_labels        # noqa: E402
from baseline_common import _prep, _discover     # noqa: E402
from reliability_fusion import _read_services     # noqa: E402

warnings.filterwarnings('ignore')

# The five metric families M of the paper's severity sev(s) (Eq. 1):
# (name, file, column-suffix, direction).  direction: 'lower' = success-rate
# decrease, 'upper' = latency/CPU/memory increase, 'abs' = absolute QPS change.
SEV_METRICS = (
    ('failure', 'success_rate.csv', None, 'lower'),
    ('latency', 'latency.csv', '&p50', 'upper'),
    ('cpu', 'svc_metric.csv', '&cpu_usage', 'upper'),
    ('mem', 'svc_metric.csv', '&mem_usage', 'upper'),
    ('qps', 'svc_qps.csv', None, 'abs'),
)
SEV_DELTA = 0.05     # relative scale floor delta
SEV_EPS = 1e-6


def _sev_map(metrics_dir, start_ts, end_ts):
    """Per-service severity sev(s) strictly per paper Eq. 1:

        sev(s) = (1/|M|) sum_m  (1/|W_{s,m}|) sum_{t in W}  Delta_m(x,b) / a_{s,m}

    b_{s,m}, MAD_{s,m} = median / median-abs-deviation of the NORMAL PREFIX
    (t < window start); a_{s,m} = max(MAD, delta*|b|, eps); W = valid
    observations in the anomaly window. A metric with no usable normal prefix
    contributes 0; the sum is always divided by |M| = 5."""
    services = _read_services(metrics_dir)
    terms = {s: 0.0 for s in services}
    for _name, fn, suf, direction in SEV_METRICS:
        df = pd.read_csv(os.path.join(metrics_dir, fn))
        epoch = pd.to_datetime(df['timestamp']).astype('int64') // 10 ** 9
        pre = (epoch < start_ts).to_numpy()
        win = ((epoch >= start_ts) & (epoch <= end_ts)).to_numpy()
        for s in services:
            col = s if suf is None else f'{s}{suf}'
            if col not in df.columns:
                continue
            x = pd.to_numeric(df[col], errors='coerce')
            x = x.where(x != -1).to_numpy(float)
            base = x[pre]
            base = base[np.isfinite(base)]
            cur = x[win]
            cur = cur[np.isfinite(cur)]
            if base.size < 3 or cur.size < 1:
                continue
            b = float(np.median(base))
            mad = float(np.median(np.abs(base - b)))
            a = max(mad, SEV_DELTA * abs(b), SEV_EPS)
            if direction == 'lower':
                d = np.maximum(0.0, b - cur)
            elif direction == 'abs':
                d = np.abs(cur - b)
            else:
                d = np.maximum(0.0, cur - b)
            terms[s] += float(np.mean(d / a))
    return {s: terms[s] / len(SEV_METRICS) for s in services}


def _cpu_dev_matrix(metrics_dir, start_ts, end_ts):
    """Per-service cpu_usage deviation series over the abnormal window.

    Returns (D, services): D is (T x S), column s = d_s(t) on the abn window,
    base = median over the normal segment (ts < start). Services without a
    usable normal/abn segment or with zero variance are dropped."""
    p = os.path.join(metrics_dir, 'svc_metric.csv')
    df = pd.read_csv(p)
    ts = pd.to_datetime(df['timestamp'])
    epoch = ts.astype('int64') // 10 ** 9
    normal = (epoch < start_ts).to_numpy()
    abn = ((epoch >= start_ts) & (epoch <= end_ts)).to_numpy()
    cols = [c for c in df.columns if c.endswith('&cpu_usage')]
    series, names = [], []
    for c in cols:
        svc = c.split('&')[0]
        x = pd.to_numeric(df[c], errors='coerce').to_numpy(dtype=float)
        xn, xa = x[normal], x[abn]
        if np.isfinite(xn).sum() < 3 or np.isfinite(xa).sum() < 3:
            continue
        base = np.nanmedian(xn)
        if not np.isfinite(base) or abs(base) < 1e-9:
            base = max(abs(np.nanmean(xn)), 1e-6)
        dev = (xa - base) / base
        if not np.isfinite(dev).all():
            dev = np.nan_to_num(dev, nan=0.0, posinf=0.0, neginf=0.0)
        if np.nanstd(dev) < 1e-9:            # constant -> no co-movement info
            continue
        series.append(dev)
        names.append(svc)
    if len(series) < 2:
        return None, names
    T = min(len(s) for s in series)
    D = np.column_stack([s[:T] for s in series])
    return D, names


def _eta_rho(D):
    """First-PC variance ratio (correlation-matrix PCA) and mean pairwise
    correlation over the columns (services) of D."""
    R = np.corrcoef(D, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)
    n = R.shape[0]
    iu = np.triu_indices(n, k=1)
    rho_bar = float(np.mean(R[iu]))
    eig = np.linalg.eigvalsh(R)
    eig = np.clip(eig, 0, None)
    eta = float(eig.max() / eig.sum()) if eig.sum() > 0 else float('nan')
    return eta, rho_bar, n


def _cv(values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if v.size == 0 or abs(v.mean()) < 1e-12:
        return float('nan')
    return float(v.std(ddof=0) / v.mean())


def _run(align):
    cfg = Config()
    rows = []
    for group, load, adir in _discover(None):
        labels = parse_labels(adir)
        samples = [(n, os.path.join(adir, n)) for n in sorted(labels)
                   if os.path.isdir(os.path.join(adir, n))]
        for sname, sdir in samples:
            gt = labels[sname].get('service', '')
            tag = f'{group}/{load}'
            try:
                with open(os.devnull, 'w') as dn, \
                        redirect_stdout(dn), redirect_stderr(dn):
                    ctx = _prep(cfg, sdir, align, detector='severity')
                D, names = _cpu_dev_matrix(
                    ctx['metrics_dir'], ctx['start_ts'], ctx['end_ts'])
                if D is None:
                    raise ValueError('too few usable cpu-dev series')
                eta, rho_bar, N = _eta_rho(D)
                scores = list(ctx['score_map'].values())
                top5 = sorted(scores, reverse=True)[:5]
                sev = _sev_map(ctx['metrics_dir'],
                               ctx['start_ts'], ctx['end_ts'])
                sev_vals = [v for v in sev.values() if np.isfinite(v)]
                sev_cv = _cv(sev_vals)
                # does any non-root service match or exceed the root severity?
                sev_root = sev.get(gt)
                ge_root = (int(any(v >= sev_root - 1e-12
                                   for s, v in sev.items() if s != gt))
                           if sev_root is not None and sev_root > 0 else
                           (1 if sev_root is not None else np.nan))
                rows.append({
                    'group': group, 'load': load, 'name': sname, 'gt': gt,
                    'N_svc': N, 'n_cand': len(scores), 'eta': eta,
                    'rho_bar': rho_bar, 'cv_all': _cv(scores),
                    'cv_top5': _cv(top5), 'sev_cv': sev_cv, 'ge_root': ge_root,
                    'eta_ref_1overN': 1.0 / N if N else float('nan')})
                print(f'  {tag:<18} {sname[:42]:<42} N={N:<2} '
                      f'eta={eta:.3f} rho={rho_bar:+.3f} '
                      f'sevCV={sev_cv:.3f} '
                      f'ge_root={"-" if ge_root != ge_root else int(ge_root)}')
            except Exception as exc:
                print(f'  {tag:<18} {sname[:42]:<42} FAILED: '
                      f'{type(exc).__name__}: {exc}')
    return pd.DataFrame(rows)


def _summ(df, col):
    v = df[col].dropna().to_numpy()
    if v.size == 0:
        return 'n/a'
    return (f'median={np.median(v):.3f}  mean={v.mean():.3f}  '
            f'[{np.percentile(v, 25):.3f}, {np.percentile(v, 75):.3f}] '
            f'min={v.min():.3f} max={v.max():.3f}  n={v.size}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('MDOC', 'MARBLEBench'),
                    default='MDOC')
    ap.add_argument('--align', action='store_true',
                    help='apply B.1 lag alignment in _prep (sensitivity)')
    args = ap.parse_args()
    print(f'Common-mode diagnostic on {args.dataset}  align={args.align}\n')
    df = _run(args.align)
    if df.empty:
        print('\nNo samples processed.')
        return
    out = os.path.join('data', args.dataset, f'diag_common_mode_'
                       f'{"aligned" if args.align else "raw"}.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)

    def _ge(g):
        v = g['ge_root'].dropna()
        return f'{int(v.sum())}/{len(v)}' if len(v) else 'n/a'

    print('\n' + '=' * 78)
    print('SUMMARY  (paper ref: eta~0.50/0.47, rho_bar~0.43/0.35, '
          'severity CV 0.872/0.712, >=root 69/120, 51/120)')
    print('=' * 78)
    print(f'Samples processed: {len(df)}   '
          f'services N: median={int(df["N_svc"].median())} '
          f'(1/N ref ~= {1.0 / df["N_svc"].median():.3f})')
    for load, g in df.groupby('load'):
        print(f'\n--- {load}  (n={len(g)}) ---')
        print(f'  eta        : {_summ(g, "eta")}')
        print(f'  rho_bar    : {_summ(g, "rho_bar")}')
        print(f'  severity CV: {_summ(g, "sev_cv")}')
        print(f'  >=root     : {_ge(g)}  (another service matches/exceeds root)')
    print(f'\n--- ALL loads (n={len(df)}) ---')
    print(f'  eta        : {_summ(df, "eta")}')
    print(f'  rho_bar    : {_summ(df, "rho_bar")}')
    print(f'  severity CV: {_summ(df, "sev_cv")}')
    print(f'  >=root     : {_ge(df)}  (another service matches/exceeds root)')
    print(f'\nPer-sample CSV: {out}')


if __name__ == '__main__':
    main()
