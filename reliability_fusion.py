"""Reliability-aware evidence fusion for root-cause localization.

Replaces the hand-set fusion coefficients (severity's w_res/w_qps and the
localizer's lam/mu) with per-signal weights that are DERIVED FROM THE CURRENT
OBSERVATION, not tuned on a dataset. Each telemetry modality m (failure,
latency, cpu, mem, net, qps, residual) votes a per-service ranking r_m(s); the
votes are combined by reliability-weighted Reciprocal Rank Fusion:

    Score(s) = sum_m  q_hat_m / (k + r_m(s))

The reliability q_m of a signal is q_m = C_m * E_m, the product of two
complementary, observation-dependent properties (both in [0,1]):

  * abnormality confidence  C_m = 1 - exp(-max_s z_m(s))
        -- does the signal significantly leave its NORMAL baseline at all?
  * candidate concentration E_m = 1 - H(softmax z_m)/log N
        -- does the signal point discriminatively at few services, or is it
           diffuse noise?

z_m(s) is a ROBUST (median/MAD) deviation of service s from its own pre-window
normal baseline, with a relative floor on the scale so a near-constant baseline
cannot explode into spurious huge z (the failure mode of naive std-scaling).

Because q_m = q_m(X) is instance-adaptive, a network-delay fault automatically
up-weights latency and down-weights cpu, an availability fault up-weights
failure, etc., WITHOUT any dataset-specific coefficient. The three ablation
variants are exposed via `variant`:
    'concentration' -> q = E_m
    'confidence'    -> q = C_m
    'both'          -> q = C_m * E_m   (default; most complete semantics)
"""
import os
import math
from typing import Dict, List, Callable, Optional

import numpy as np
import pandas as pd

from lag_align import align_metrics_df


def _read_services(metrics_dir: str) -> List[str]:
    df = pd.read_csv(os.path.join(metrics_dir, 'success_rate.csv'), nrows=1)
    return [c for c in df.columns if c != 'timestamp']


def _win_repr(cur: pd.Series, direction: str) -> float:
    """Representative abnormal-window value: sustained peak (upper), worst drop
    (lower), or median (both)."""
    if direction == 'lower':
        return float(cur.quantile(0.1))
    if direction == 'both':
        return float(cur.median())
    return float(cur.quantile(0.9))


def _dev(rep: float, med: float, scale: float, direction: str) -> float:
    if direction == 'lower':
        return max(0.0, (med - rep) / scale)
    if direction == 'both':
        return max(0.0, abs(rep - med) / scale)
    return max(0.0, (rep - med) / scale)


def _robust_z(df: pd.DataFrame, series_of: Callable[[str], Optional[pd.Series]],
              services: List[str], start, end, direction: str,
              min_rel: float = 0.05,
              baseline_free_fallback: bool = False,
              use_post_baseline: bool = False) -> Dict[str, float]:
    """Per-service robust deviation z_m(s) >= 0. Preferred calibration is
    TEMPORAL: current window vs this service's own pre-window normal baseline
    (median/MAD, scale floored at min_rel*|median| so a near-constant baseline
    cannot explode). When a signal has no usable pre-window segment (latency /
    success_rate / qps start at fault onset in this benchmark), we fall back to
    a CROSS-SERVICE calibration: how far each service's window value stands out
    from its peers' median/MAD -- a baseline-free abnormality proxy that keeps
    fault-discriminative voters (e.g. latency for a delay fault) alive."""
    ts = df['timestamp']
    pre_m = ts < start
    post_m = ts > end
    win = (ts >= start) & (ts <= end)
    temporal: Dict[str, float] = {}
    win_repr: Dict[str, float] = {}
    for s in services:
        ser = series_of(s)
        if ser is None:
            continue
        ser = pd.to_numeric(ser, errors='coerce')
        ser = ser.where(ser != -1)              # -1 = no-data sentinel -> missing
        cur = ser[win].dropna()
        if len(cur) < 1:
            continue
        rep = _win_repr(cur, direction)
        win_repr[s] = rep
        # out-of-window baseline. Default: pre-window prefix only. The
        # ._outside_baseline post-window fallback is available but OFF by
        # default: empirically the ~3-4 post-window tail rows sit right after
        # the fault (still elevated/recovering) and revive latency/qps voters
        # with noisy calibration, degrading fusion.
        pre = ser[pre_m].dropna()
        base = (pre if len(pre) >= 3 or not use_post_baseline
                else pd.concat([pre, ser[post_m].dropna()]))
        if len(base) >= 3:
            med = float(base.median())
            scale = max(float((base - med).abs().median()),
                        min_rel * abs(med), 1e-6)
            temporal[s] = _dev(rep, med, scale, direction)
    # enough services have a real normal baseline -> trust temporal calibration
    if len(temporal) >= max(3, len(win_repr) // 2):
        return temporal
    # no usable normal baseline. Either let the voter die (temporal-only, which
    # empirically beats the noisy cross-service latency/qps votes here) or fall
    # back to peer-relative abnormality when explicitly enabled.
    if not baseline_free_fallback or not win_repr:
        return temporal
    vals = np.array(list(win_repr.values()), dtype=float)
    med = float(np.median(vals))
    scale = max(float(np.median(np.abs(vals - med))), min_rel * abs(med), 1e-6)
    return {s: _dev(rep, med, scale, direction) for s, rep in win_repr.items()}


def _reliability(z: Dict[str, float], services: List[str], variant: str):
    """(q_m, rank_m) for one signal over the candidate services. rank is 1-based
    by descending z (ties resolved by stable order); missing/zero services sink
    to the bottom. q_m in [0,1] per the chosen variant."""
    vals = np.array([max(0.0, z.get(s, 0.0)) for s in services], dtype=float)
    n = len(services)
    order = sorted(range(n), key=lambda i: vals[i], reverse=True)
    rank = {services[order[j]]: j + 1 for j in range(n)}
    if n <= 1 or vals.max() <= 0:
        return 0.0, rank
    zmax = float(vals.max())
    # C = abnormality confidence, on the RAW z (carries magnitude only).
    C = 1.0 - math.exp(-zmax)
    # E = candidate concentration, on the SCALE-FREE relative-range-normalized
    # r (carries shape only), so E is decoupled from the absolute z scale and no
    # longer double-counts magnitude with C. relative range (denominator
    # zmax+eps, NOT zmax-zmin) keeps trivially-close candidates near 0 -> low E.
    r = (vals - float(vals.min())) / (zmax + 1e-9)
    e = np.exp(r - r.max())
    p = e / e.sum()
    H = float(-(p * np.log(p + 1e-12)).sum())
    E = 1.0 - H / math.log(n)
    q = {'confidence': C, 'concentration': E}.get(variant, C * E)
    return float(q), rank


def _telemetry_signals(ns_dir: str, lag_map,
                       lat_percentile: str = 'p50') -> Dict[str, tuple]:
    """Build the raw voter table: signal -> (dataframe, series_of, direction).
    Resource/latency frames are lag-aligned like the rest of the pipeline;
    success_rate and qps are left on the raw timeline.

    lat_percentile selects the latency voter column ('p50' default; 'p90' is used
    on RE2, where the tail percentile carries the delay-fault signal)."""
    md = os.path.join(ns_dir, 'metrics')
    services = _read_services(md)
    sig: Dict[str, tuple] = {}

    def add(name, df, series_of, direction):
        df = df.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        sig[name] = (df, series_of, direction)

    # failure (success_rate deficit, lower)
    sr = pd.read_csv(os.path.join(md, 'success_rate.csv'))
    add('failure', sr, (lambda df: lambda s: df[s] if s in df else None)(sr),
        'lower')
    # latency (upper); column = p50 by default, p90 on RE2
    lat = align_metrics_df(pd.read_csv(os.path.join(md, 'latency.csv')), lag_map)
    latcol = {s: f'{s}&{lat_percentile}' for s in services}
    add('latency', lat,
        (lambda df, cm: lambda s: df[cm[s]] if cm.get(s) in df else None)(
            lat, latcol), 'upper')
    # cpu / mem / net from svc_metric (upper)
    sm = align_metrics_df(pd.read_csv(os.path.join(md, 'svc_metric.csv')),
                          lag_map)
    for name, suf in (('cpu', '&cpu_usage'), ('mem', '&mem_usage')):
        cm = {s: f'{s}{suf}' for s in services}
        add(name, sm,
            (lambda df, cm: lambda s: df[cm[s]] if cm.get(s) in df else None)(
                sm, cm), 'upper')

    def net_series(df):
        def f(s):
            cols = [c for c in (f'{s}&net_receive', f'{s}&net_trainsmit')
                    if c in df]
            if not cols:
                return None
            return df[cols].replace(-1, np.nan).sum(axis=1, min_count=1)
        return f
    add('net', sm, net_series(sm), 'upper')
    # qps (bidirectional workload deviation)
    qps = pd.read_csv(os.path.join(md, 'svc_qps.csv'))
    add('qps', qps, (lambda df: lambda s: df[s] if s in df else None)(qps),
        'both')
    return sig


def reliability_rrf(ns_dir: str, start, end, lag_map, variant: str = 'both',
                    k_rrf: int = 60, extra_z: Optional[Dict[str, Dict]] = None,
                    signals: Optional[List[str]] = None, verbose: bool = True,
                    baseline_free_fallback: bool = False,
                    use_post_baseline: bool = False,
                    candidates: Optional[List[str]] = None,
                    return_scores: bool = False, fuse: str = 'rrf',
                    lat_percentile: str = 'p50'):
    """Rank services by reliability-weighted RRF over the telemetry voters (plus
    any pre-computed extra voters in `extra_z`, e.g. the GNN residual).

    candidates=None ranks ALL services (full localizer, can recover buried root
    causes). candidates=<list> restricts fusion to that set (severity top-k), so
    reliability weighting only REORDERS the given candidates -- a pure,
    parameter-free replacement for the lam/mu rerank, directly comparable to the
    residual method (candidate set unchanged, ACC@>=k preserved)."""
    services = (list(candidates) if candidates is not None
                else _read_services(os.path.join(ns_dir, 'metrics')))
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    table = _telemetry_signals(ns_dir, lag_map, lat_percentile=lat_percentile)
    if signals is not None:
        table = {k: v for k, v in table.items() if k in signals}

    z_by_sig: Dict[str, Dict[str, float]] = {}
    for name, (df, series_of, direction) in table.items():
        z_by_sig[name] = _robust_z(df, series_of, services, start, end,
                                   direction,
                                   baseline_free_fallback=baseline_free_fallback,
                                   use_post_baseline=use_post_baseline)
    for name, z in (extra_z or {}).items():          # residual etc.
        z_by_sig[name] = z

    q_raw, ranks = {}, {}
    for name, z in z_by_sig.items():
        q, rank = _reliability(z, services, variant)
        q_raw[name] = q
        ranks[name] = rank
    total = sum(q_raw.values()) or 1.0
    q_hat = {m: q_raw[m] / total for m in q_raw}

    if fuse in ('wsum', 'max', 'mean'):
        # per-signal magnitudes min-max normalized to [0,1] across services.
        znorm = {}
        for m, z in z_by_sig.items():
            lo = min((z.get(s, 0.0) for s in services), default=0.0)
            hi = max((z.get(s, 0.0) for s in services), default=0.0)
            rng = hi - lo
            znorm[m] = {s: ((z.get(s, 0.0) - lo) / rng if rng > 1e-12 else 0.0)
                        for s in services}
        if fuse == 'wsum':          # reliability-weighted sum (keeps magnitude)
            score = {s: sum(q_hat[m] * znorm[m][s] for m in q_hat)
                     for s in services}
        elif fuse == 'max':         # ablation: no weighting, max over voters
            score = {s: max((znorm[m][s] for m in znorm), default=0.0)
                     for s in services}
        else:                       # 'mean': ablation, equal-weight mean
            score = {s: (sum(znorm[m][s] for m in znorm) / len(znorm)
                         if znorm else 0.0) for s in services}
    else:                           # 'rrf': reciprocal-rank fusion
        score = {}
        for s in services:
            score[s] = sum(q_hat[m] / (k_rrf + ranks[m][s]) for m in ranks)
    order = [s for s, _ in sorted(score.items(), key=lambda kv: kv[1],
                                  reverse=True)]
    if verbose:
        qshow = {m: round(q_hat[m], 3) for m in
                 sorted(q_hat, key=q_hat.get, reverse=True)}
        print(f'[relrrf:{variant}] reliability q_hat: {qshow}')
        print(f'[relrrf:{variant}] order top5: {order[:5]}')
    return score if return_scores else order
