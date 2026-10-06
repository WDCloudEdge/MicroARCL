"""
Convert an RCAEval system-version (e.g. re2ob / re2ss / re2tt) into LagRCA's
preprocessed format.

RCAEval layout: data/<sys>_<service>_<fault>_<idx>/metrics.json + inject_time.txt
  metrics.json : {"<service>_<metric>": [[ts, value], ...]}
  inject_time  : unix ts splitting normal (before) / abnormal (after)

Mapping to LagRCA (service-level; no pod/node tier):
  instances  = union of services across the system's samples
  features   = <metric>&<service> for every (service, metric-type), missing -> 0
  normal_data= pre-injection rows pooled across samples (capped & subsampled)
  case_data  = each sample's post-injection rows; label = faulted service;
               case_id = "<sys>/<fault>/<name>"  (enables per-fault breakdown)
  adj        = fully connected by default; --trace-adj for re2ob uses direct
               calls recovered from pre-injection traces.csv

Usage:  python build_rcaeval_dataset.py re2ob
Writes: data/<sys>/{normal_data.pkl, adj.pkl, case_data.pkl, all_enum.json}
"""
import os
import re
import sys
import glob
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import torch

RCAEVAL_ROOT = os.environ.get("RCAEVAL_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "RCAEval"))
OUT_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

NORMAL_ROW_CAP = 3000          # cap pooled pre-injection rows (keep training tractable)

TRACE_COLUMNS = ['traceID', 'spanID', 'parentSpanID', 'serviceName', 'startTimeMillis']
TRACE_ALIASES = {'re2ob': {'frontendservice': 'frontend'}}


def sample_dirs(sysname):
    out = []
    for d in sorted(glob.glob(os.path.join(RCAEVAL_ROOT, sysname + "_*"))):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metrics.json")):
            out.append(d)
    return out


def parse_name(sysname, name):
    """<sys>_<service>_<fault>_<idx> -> (service, fault, idx)."""
    rest = name[len(sysname) + 1:]
    parts = rest.rsplit("_", 2)
    if len(parts) != 3:
        return None, None, None
    return parts[0], parts[1], parts[2]


def load_metrics_df(sample_dir):
    """metrics.json -> wide DataFrame indexed by ts, columns '<metric>&<service>'."""
    with open(os.path.join(sample_dir, "metrics.json")) as f:
        d = json.load(f)
    cols = {}
    for key, pairs in d.items():
        svc, met = key.rsplit("_", 1)
        s = pd.Series({int(t): float(v) for t, v in pairs})
        cols[f"{met}&{svc}"] = s
    df = pd.DataFrame(cols).sort_index()
    return df


def scan_system(dirs, sysname):
    """Union of services and metric types across the system's samples."""
    services, metrics = set(), set()
    for d in dirs:
        with open(os.path.join(d, "metrics.json")) as f:
            keys = json.load(f).keys()
        for k in keys:
            svc, met = k.rsplit("_", 1)
            services.add(svc)
            metrics.add(met)
    return sorted(services), sorted(metrics)


def align_columns(df, services, metrics):
    """Ensure every (service, metric) column exists (missing -> 0), ordered by
    instance then metric so instance blocks are contiguous."""
    ordered = [f"{m}&{s}" for s in services for m in metrics]
    df = df.reindex(columns=ordered)
    df = df.apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)
    return df


def fit_scaler(normal_df, feat):
    mn, rng = {}, {}
    for c in feat:
        col = normal_df[c].values.astype(float)
        col = col[~np.isnan(col)]
        lo = float(np.min(col)) if col.size else 0.0
        hi = float(np.max(col)) if col.size else 1.0
        mn[c], rng[c] = lo, (hi - lo) or 1.0
    return mn, rng


def apply_scaler(df, feat, mn, rng):
    df = df.copy()
    for c in feat:
        df[c] = (df[c].astype(float) - mn[c]) / rng[c]
    return df.fillna(0.0)


def build_trace_adj(dirs, sysname, services):
    """Union direct parent/child service calls before fault injection only."""
    index = {service: i for i, service in enumerate(services)}
    edges = set()
    trace_files = 0
    unmatched = set()
    for d in dirs:
        path = os.path.join(d, 'traces.csv')
        if not os.path.isfile(path):
            continue
        trace_files += 1
        inj_ms = int(open(os.path.join(d, 'inject_time.txt')).read().strip()) * 1000
        spans = pd.read_csv(path, usecols=TRACE_COLUMNS,
                            dtype={c: str for c in TRACE_COLUMNS if c != 'startTimeMillis'})
        spans = spans[pd.to_numeric(spans['startTimeMillis'], errors='coerce') < inj_ms]
        spans = spans.dropna(subset=['traceID', 'spanID', 'parentSpanID', 'serviceName'])
        parents = spans[['traceID', 'spanID', 'serviceName']].rename(
            columns={'spanID': 'parentSpanID', 'serviceName': 'caller'})
        calls = spans.merge(parents, on=['traceID', 'parentSpanID'], how='inner',
                            validate='many_to_one')
        aliases = TRACE_ALIASES.get(sysname, {})
        for caller, callee in calls[['caller', 'serviceName']].drop_duplicates().itertuples(index=False, name=None):
            caller = aliases.get(caller, caller)
            callee = aliases.get(callee, callee)
            if caller == callee:
                continue
            if caller not in index or callee not in index:
                unmatched.update(s for s in (caller, callee) if s not in index)
                continue
            edges.add((caller, callee))
    if not trace_files:
        raise RuntimeError(f'{sysname}: no traces.csv found; cannot build a trace topology')
    if not edges:
        raise RuntimeError(f'{sysname}: no usable pre-injection cross-service calls found')
    if unmatched:
        print(f'  trace services absent from metrics: {sorted(unmatched)}')
    adj = np.eye(len(services), dtype=np.float32)
    for caller, callee in edges:
        i, j = index[caller], index[callee]
        adj[i, j] = adj[j, i] = 1.0
    print(f'  trace topology: {trace_files}/{len(dirs)} files, '
          f'{len(edges)} directed calls, {int((adj.sum()-len(services))/2)} service pairs')
    return torch.tensor(adj)


def save_adj(out_dir, adj):
    """Preserve the old all-ones prior once when replacing it."""
    path = os.path.join(out_dir, 'adj.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as f:
            old = pickle.load(f)
        if bool((old == 1).all()) and not os.path.exists(os.path.join(out_dir, 'adj_full.pkl')):
            shutil.copy2(path, os.path.join(out_dir, 'adj_full.pkl'))
    with open(path, 'wb') as f:
        pickle.dump(adj, f)


def main(sysname, adj_only=False, trace_adj=False):
    dirs = sample_dirs(sysname)
    if not dirs:
        raise SystemExit(f"No samples for {sysname} under {RCAEVAL_ROOT}")
    out_dir = os.path.join(OUT_BASE, sysname)
    if adj_only:
        enum_path = os.path.join(out_dir, 'all_enum.json')
        if not os.path.isfile(enum_path):
            raise RuntimeError(f'{out_dir}: convert the dataset before --adj-only')
        with open(enum_path) as f:
            existing = json.load(f)
        services = list(existing)
        if existing != {s: i for i, s in enumerate(services)}:
            raise RuntimeError(f'{sysname}: invalid existing instance order')
        metrics = []
    else:
        services, metrics = scan_system(dirs, sysname)
    all_enum = {s: i for i, s in enumerate(services)}
    feat = [f"{m}&{s}" for s in services for m in metrics]
    print(f"{sysname}: {len(dirs)} samples | {len(services)} services | "
          f"metrics={metrics} | {len(feat)} feature cols")

    if adj_only:
        if not trace_adj:
            raise ValueError('--adj-only requires --trace-adj')
        adj = build_trace_adj(dirs, sysname, services)
        save_adj(out_dir, adj)
        print(f'  wrote -> {out_dir}/adj.pkl')
        return

    normal_parts, cases = [], []
    skipped = 0
    for d in dirs:
        name = os.path.basename(d)
        svc, fault, idx = parse_name(sysname, name)
        if svc is None or svc not in all_enum:
            print(f"  [skip] {name}: label '{svc}' not a known service")
            skipped += 1
            continue
        inj = int(open(os.path.join(d, "inject_time.txt")).read().strip())
        df = load_metrics_df(d)
        df = align_columns(df, services, metrics)
        pre = df[df.index < inj]
        post = df[df.index >= inj]
        if len(post) < 1:
            print(f"  [skip] {name}: no post-injection rows")
            skipped += 1
            continue
        if len(pre) > 0:
            normal_parts.append(pre)
        case_df = post.reset_index().rename(columns={"index": "timestamp"})
        case_id = f"{sysname}/{fault}/{name}"
        cases.append([case_df, [svc], case_id])

    # pooled normal, capped + evenly subsampled
    normal = pd.concat(normal_parts, axis=0, ignore_index=True)
    if len(normal) > NORMAL_ROW_CAP:
        step = int(np.ceil(len(normal) / NORMAL_ROW_CAP))
        normal = normal.iloc[::step].reset_index(drop=True)
    normal = normal.reset_index().rename(columns={"index": "timestamp"})

    # normalize (fit on normal, apply to normal + cases)
    mn, rng = fit_scaler(normal, feat)
    normal = apply_scaler(normal, feat, mn, rng)
    cases = [[apply_scaler(df, feat, mn, rng), lab, cid] for df, lab, cid in cases]

    adj = (build_trace_adj(dirs, sysname, services) if trace_adj else
           torch.ones((len(services), len(services)), dtype=torch.float32))

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "all_enum.json"), "w") as f:
        json.dump(all_enum, f)
    save_adj(out_dir, adj)
    with open(os.path.join(out_dir, "normal_data.pkl"), "wb") as f:
        pickle.dump(normal, f)
    with open(os.path.join(out_dir, "case_data.pkl"), "wb") as f:
        pickle.dump(cases, f)

    print(f"  normal_data {normal.shape} (cap {NORMAL_ROW_CAP}) | {len(cases)} cases"
          f" | adj {tuple(adj.shape)} | skipped {skipped}")
    print(f"  wrote -> {out_dir}")


if __name__ == "__main__":
    args = [x for x in sys.argv[1:] if x not in ('--adj-only', '--trace-adj')]
    if '--trace-adj' in sys.argv and args != ['re2ob']:
        raise SystemExit('--trace-adj is currently supported only for re2ob')
    for sysname in (args or ["re2ob"]):
        main(sysname, adj_only='--adj-only' in sys.argv,
             trace_adj='--trace-adj' in sys.argv)
