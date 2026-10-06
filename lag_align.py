"""metric lag alignment (QPS-driven)."""
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def _service_of_pod(name: str) -> str:
    return re.sub(r'-[0-9a-z]{6,10}-[0-9a-z]{5}$', '', name)


def estimate_lag_qps(qps: pd.Series, metric: pd.Series, tau_max: int,
                     min_corr: float = 0.3,
                     min_improvement: float = 0.05,
                     min_points: int = 12) -> int:
    """Lag tau in [0, tau_max] maximizing corr(qps[t], metric[t+tau]).
    A non-zero lag is accepted only when its correlation is practically
    significant and improves on the zero-lag correlation. NaN and the explicit
    -1 missing sentinel are excluded from correlation rather than treated as
    measurements. Returns 0 when the evidence is insufficient."""
    q = pd.to_numeric(qps, errors='coerce').to_numpy(dtype=float)
    m = pd.to_numeric(metric, errors='coerce').to_numpy(dtype=float)
    n = min(len(q), len(m))
    q, m = q[:n], m[:n]
    min_points = max(5, int(min_points))
    correlations = {}
    for tau in range(0, tau_max + 1):
        if n - tau < min_points:
            break
        a, b = q[:n - tau], m[tau:]
        valid = np.isfinite(a) & np.isfinite(b) & (a != -1) & (b != -1)
        if int(valid.sum()) < min_points:
            continue
        a, b = a[valid], b[valid]
        if a.std() < 1e-9 or b.std() < 1e-9:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(c):
            correlations[tau] = c
    if not correlations:
        return 0
    best_tau = max(correlations, key=correlations.get)
    best_corr = correlations[best_tau]
    zero_corr = correlations.get(0, 0.0)
    if (best_tau == 0 or best_corr < min_corr
            or best_corr - zero_corr < min_improvement):
        return 0
    return best_tau


def compute_service_lags(metrics_dir: str, cfg) -> Dict[Tuple[str, str], int]:
    """Per-(service, kind) lag from QPS vs CPU/memory/latency/network.

    CPU and memory use their service usage series, latency uses service p50,
    and network uses receive+transmit traffic. Each kind is estimated directly;
    latency/network never borrow the CPU or memory lag.
    """
    tau_max = getattr(cfg, 'lag_tau_max', 12)
    min_corr = getattr(cfg, 'lag_min_corr', 0.3)
    min_improvement = getattr(cfg, 'lag_min_improvement', 0.05)
    min_points = getattr(cfg, 'lag_min_points', 12)
    try:
        qps = pd.read_csv(os.path.join(metrics_dir, 'svc_qps.csv'))
        svcm = pd.read_csv(os.path.join(metrics_dir, 'svc_metric.csv'))
        latency = pd.read_csv(os.path.join(metrics_dir, 'latency.csv'))
    except Exception as e:
        print(f'[lag_align] cannot read metrics for lag estimation: {e}')
        return {}
    if ('timestamp' not in qps or 'timestamp' not in svcm
            or 'timestamp' not in latency):
        return {}
    merged = pd.merge(qps, svcm, on='timestamp', how='inner')
    merged = pd.merge(merged, latency, on='timestamp', how='inner')
    services = [c for c in qps.columns if c != 'timestamp']
    lag_map: Dict[Tuple[str, str], int] = {}
    for svc in services:
        if svc not in merged:
            continue
        candidates = {
            'cpu': [svc + '&cpu_usage'],
            'mem': [svc + '&mem_usage'],
            'latency': [svc + '&p50'],
            'net': [svc + '&net_receive', svc + '&net_trainsmit'],
        }
        for kind, columns in candidates.items():
            present = [col for col in columns if col in merged]
            if not present:
                continue
            metric = (merged[present[0]] if len(present) == 1
                      else merged[present].replace(-1, np.nan).sum(
                          axis=1, min_count=1))
            lag_map[(svc, kind)] = estimate_lag_qps(
                merged[svc], metric, tau_max,
                min_corr=min_corr,
                min_improvement=min_improvement,
                min_points=min_points)
    return lag_map


def compute_chain_lags(metrics_dir: str, ns_dir: str, cfg
                       ) -> Dict[Tuple[str, str], int]:
    """Per-service lag accumulated along REAL execution-graph traces (vs the
    self QPS->resource lag in compute_service_lags). Each trace's ordered service
    path (call_chains.json, folded to service level) is walked hop-by-hop from the
    entry, accumulating per-edge QPS lags (downstream lags upstream; edge lags
    hitting tau_max ~>60s are dropped, clamped to tau_max). A service aggregates
    its cumulative lag over ALL its trace occurrences (frequency-weighted mean —
    frequent chains dominate). Uses the real per-trace paths, NOT the aggregated
    call graph, so no spurious cross-trace paths are created.
    Returns {(svc, kind): tau} for every kind."""
    tau_max = getattr(cfg, 'lag_tau_max', 12)
    min_corr = getattr(cfg, 'lag_min_corr', 0.3)
    min_improvement = getattr(cfg, 'lag_min_improvement', 0.05)
    min_points = getattr(cfg, 'lag_min_points', 12)
    try:
        qps = pd.read_csv(os.path.join(metrics_dir, 'svc_qps.csv'))
    except Exception as e:
        print(f'[lag_align] cannot read qps for chain lag: {e}')
        return {}
    import json
    from agent_gnn import _read_services
    from execution_graph import group_to_service
    cc_path = os.path.join(ns_dir, 'graph', 'call_chains.json')
    if not os.path.exists(cc_path):
        return {}
    with open(cc_path, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    services = _read_services(metrics_dir)

    edge_lag: Dict[Tuple[str, str], int] = {}    # cache per svc->svc edge

    def _edge_lag(u: str, d: str) -> int:
        if (u, d) not in edge_lag:
            t = 0
            if u in qps.columns and d in qps.columns:
                t = estimate_lag_qps(qps[u], qps[d], tau_max, min_corr,
                                     min_improvement, min_points)
                t = t if 0 < t < tau_max else 0  # drop 0 and boundary (~>60s)
            edge_lag[(u, d)] = t
        return edge_lag[(u, d)]

    # accumulate cumulative lag hop-by-hop along each real trace path; every
    # trace occurrence contributes once, so frequent chains weigh more.
    svc_cum: Dict[str, list] = {}
    svc_cache: Dict[str, Optional[str]] = {}

    def _svc(v):
        if v not in svc_cache:
            svc_cache[v] = group_to_service(v, services)
        return svc_cache[v]

    for tr in chains:
        seq = []                                  # folded service chain
        for v in (tr.get('path', []) or []):
            s = _svc(v)
            if s is None:
                continue
            if not seq or seq[-1] != s:
                seq.append(s)
        cum = 0
        for i, s in enumerate(seq):
            if i > 0:
                cum = min(tau_max, cum + _edge_lag(seq[i - 1], s))
            svc_cum.setdefault(s, []).append(cum)

    lag_map: Dict[Tuple[str, str], int] = {}
    # aggregate a service's cumulative lag over its trace occurrences.
    # 'max' = deepest real propagation path; 'mean' = frequency-weighted.
    agg = getattr(cfg, 'chain_lag_agg', 'max')
    if agg == 'mean':
        tau = {svc: int(round(sum(v) / len(v))) for svc, v in svc_cum.items()}
    else:
        tau = {svc: int(max(v)) for svc, v in svc_cum.items()}
    for svc, t in tau.items():
        if t > 0:
            for kind in ('cpu', 'mem', 'latency', 'net'):
                lag_map[(svc, kind)] = int(t)
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
        return lag_map.get((svc, 'net'), 0)
    if kind == 'latency':
        return lag_map.get((svc, 'latency'), 0)
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
