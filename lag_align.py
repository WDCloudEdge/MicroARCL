"""
Module B.1: metric lag alignment (QPS-driven).

In agent services, QPS (workload) leads while CPU/memory/latency respond with a
lag that varies by service / load / deployment (motivation Fig. m2.1: CPU lag
~30/10s, memory ~35/60s). Comparing or clustering raw (unaligned) metrics mixes
a failure's manifestations across different timestamps. We therefore estimate a
per-(service, kind) lag by cross-correlating QPS against each resource metric,
then shift the lagged metrics back onto the QPS/causal timeline. The aligned
matrix feeds adaptive-window detection, severity ranking, Birch (B.2), and the
GNN feature construction.

success_rate is NOT aligned here; it remains a downstream signal contributing
to the adaptive window alongside aligned latency and resource metrics.
Physical-node metrics (node.csv) are left unaligned in v1 (node aggregates many
services, so a single QPS reference is ill-defined).
"""
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def _service_of_pod(name: str) -> str:
    return re.sub(r'-[0-9a-z]{6,10}-[0-9a-z]{5}$', '', name)


def estimate_lag_qps(qps: pd.Series, metric: pd.Series, tau_max: int) -> int:
    """Lag tau in [0, tau_max] maximizing corr(qps[t], metric[t+tau]).
    Returns 0 if no meaningful positive-correlation lag is found."""
    q = pd.to_numeric(qps, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    m = pd.to_numeric(metric, errors='coerce').fillna(0.0).to_numpy(dtype=float)
    n = min(len(q), len(m))
    q, m = q[:n], m[:n]
    best_tau, best_c = 0, 0.0
    for tau in range(0, tau_max + 1):
        if n - tau < 5:
            break
        a, b = q[:n - tau], m[tau:]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c > best_c:      # require positive correlation to accept a lag
            best_c, best_tau = c, tau
    return best_tau


def compute_service_lags(metrics_dir: str, cfg) -> Dict[Tuple[str, str], int]:
    """Per-(service, kind) lag from QPS vs svc-level cpu/mem usage."""
    tau_max = getattr(cfg, 'lag_tau_max', 12)
    try:
        qps = pd.read_csv(os.path.join(metrics_dir, 'svc_qps.csv'))
        svcm = pd.read_csv(os.path.join(metrics_dir, 'svc_metric.csv'))
    except Exception as e:
        print(f'[lag_align] cannot read metrics for lag estimation: {e}')
        return {}
    if 'timestamp' not in qps or 'timestamp' not in svcm:
        return {}
    merged = pd.merge(qps, svcm, on='timestamp', how='inner')
    services = [c for c in qps.columns if c != 'timestamp']
    lag_map: Dict[Tuple[str, str], int] = {}
    for svc in services:
        if svc not in merged:
            continue
        for kind, suffix in (('cpu', '&cpu_usage'), ('mem', '&mem_usage')):
            col = svc + suffix
            if col in merged:
                lag_map[(svc, kind)] = estimate_lag_qps(merged[svc], merged[col], tau_max)
    return lag_map


def _col_service_kind(col: str) -> Tuple[Optional[str], Optional[str]]:
    """Map a metric column to (service, kind). kind in
    {cpu, mem, net, latency, qps/other}. Returns (None, None) to skip."""
    if col == 'timestamp':
        return None, None
    if col.startswith('(node)'):
        return None, None                      # v1: don't align physical-node metrics
    if '&' in col:                             # svc-level: <svc>&<metric>
        svc, rest = col.split('&', 1)
        if rest in ('p50', 'p90', 'p99'):
            return svc, 'latency'
        if 'cpu' in rest:
            return svc, 'cpu'
        if 'mem' in rest:
            return svc, 'mem'
        if 'net' in rest:
            return svc, 'net'
        return svc, 'other'
    if col.endswith('_network'):               # instance.csv: <pod>_network
        pod = col[:-len('_network')]
        return _service_of_pod(pod), 'net'
    if '_' in col:                             # instance.csv: <pod>_cpu / _mem
        base, suf = col.rsplit('_', 1)
        kind = {'cpu': 'cpu', 'mem': 'mem', 'memory': 'mem'}.get(suf)
        if kind is None:
            return None, None
        return _service_of_pod(base), kind
    return None, None                          # plain svc name (qps/success_rate) -> skip


def _lag_for(lag_map: Dict[Tuple[str, str], int], svc: str, kind: str) -> int:
    if kind == 'cpu':
        return lag_map.get((svc, 'cpu'), 0)
    if kind == 'mem':
        return lag_map.get((svc, 'mem'), 0)
    if kind == 'net':
        return lag_map.get((svc, 'cpu'), 0)
    if kind == 'latency':
        # TODO allow per-svc latency lag estimation (currently use max(cpu, mem) lag)
        return max(lag_map.get((svc, 'cpu'), 0), lag_map.get((svc, 'mem'), 0))
    return 0


def align_metrics_df(df: pd.DataFrame, lag_map: Dict[Tuple[str, str], int]) -> pd.DataFrame:
    """Shift each lagged metric column back onto the QPS/causal timeline
    (metric_aligned[t] = metric[t + tau]). Timestamp and unaligned columns
    (qps, success_rate, physical-node) are left untouched."""
    if not lag_map:
        return df
    out = df.copy()
    for col in df.columns:
        svc, kind = _col_service_kind(col)
        if svc is None:
            continue
        tau = _lag_for(lag_map, svc, kind)
        if tau > 0:
            out[col] = df[col].shift(-tau)
    return out


def align_call_metrics_df(
        df: pd.DataFrame,
        lag_map: Dict[Tuple[str, str], int]) -> pd.DataFrame:
    """Align ``call.csv`` latency columns by the callee service's lag.

    Call columns use ``<caller>_<callee>&pXX`` rather than the service-level
    ``<service>&pXX`` convention handled by :func:`align_metrics_df`. The
    observed latency is therefore shifted using the callee's latency lag.
    """
    if not lag_map:
        return df
    out = df.copy()
    for col in df.columns:
        if col == 'timestamp' or '&' not in col:
            continue
        call, percentile = col.split('&', 1)
        if percentile not in ('p50', 'p90', 'p99') or '_' not in call:
            continue
        # In the collected metric schema service names use '-' internally;
        # '_' is the caller/callee delimiter.
        callee = call.rsplit('_', 1)[1]
        tau = _lag_for(lag_map, callee, 'latency')
        if tau > 0:
            out[col] = df[col].shift(-tau)
    return out
