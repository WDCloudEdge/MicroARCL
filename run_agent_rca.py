"""Initial end-to-end driver for agent-service root cause localization."""
import os
import re
import numpy as np
import pandas as pd
from lag_align import align_metrics_df


def _svc_of(col: str) -> str:
    return col.split('&')[0]


def load_kpi(ns_metrics_dir: str):
    sr = pd.read_csv(os.path.join(ns_metrics_dir, 'success_rate.csv'))
    lat = pd.read_csv(os.path.join(ns_metrics_dir, 'latency.csv'))
    qps = pd.read_csv(os.path.join(ns_metrics_dir, 'svc_qps.csv'))
    for df in (sr, lat, qps):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return sr, lat, qps


def _sustain_mask(mask, min_len):
    """Keep only True-runs of length >= min_len (drops isolated blips)."""
    a = np.asarray(mask, dtype=bool)
    out = np.zeros_like(a)
    i, n = 0, len(a)
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if j - i >= min_len:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


def _series_spike(x, k=5.0, min_rel=0.3):
    """Robust upper-spike mask for a metric series. Missing / no-request points
    (NaN or the -1 sentinel) are EXCLUDED from the median/MAD/range statistics
    and can never be flagged as a spike. Otherwise a sparse series (e.g. 70%
    no-request latency) would have its baseline zero-filled, dragging the median
    to 0 so a service's NORMAL level is misread as a spike."""
    x = pd.to_numeric(x, errors='coerce')
    valid = x.notna() & (x != -1)
    xv = x[valid]
    if len(xv) < 3:
        return pd.Series(False, index=x.index)
    med = xv.median()
    mad = (xv - med).abs().median()
    rng = float(xv.max() - med)
    if rng <= 0:
        return pd.Series(False, index=x.index)
    thr = med + max(k * mad, min_rel * rng)
    return (x > thr) & valid


def _outside_baseline(df, col, start, end, min_points=3):
    """Return a clean reference series, preferring observations before the
    anomaly window.  Post-window observations are only used when the prefix is
    too short; the anomaly window itself is never allowed into its baseline.
    """
    pre = pd.to_numeric(df.loc[df['timestamp'] < start, col],
                        errors='coerce').dropna()
    if len(pre) >= min_points:
        return pre
    post = pd.to_numeric(df.loc[df['timestamp'] > end, col],
                         errors='coerce').dropna()
    outside = pd.concat([pre, post])
    return outside if len(outside) >= min_points else None


def _baseline_deviation_mask(x, baseline, direction='upper', k=5.0,
                             min_rel=0.3):
    """Detect deviations in ``x`` against an out-of-window baseline.

    ``direction`` is upper for latency/resources, lower for success rate, and
    both for QPS.  A relative floor keeps a nearly constant baseline from
    turning harmless numerical jitter into anomalies.
    """
    x = pd.to_numeric(x, errors='coerce')
    valid = x.notna() & (x != -1)          # no-request / missing never flagged
    ref = pd.to_numeric(baseline, errors='coerce').dropna()
    if ref.empty:
        return pd.Series(False, index=x.index)
    med = float(ref.median())
    mad = float((ref - med).abs().median())
    scale = max(abs(med), float(ref.quantile(0.75) - ref.quantile(0.25)),
                1e-9)
    delta = max(k * mad, min_rel * scale)
    if direction == 'lower':
        m = x < med - delta
    elif direction == 'both':
        m = (x < med - delta) | (x > med + delta)
    else:
        m = x > med + delta
    return m & valid


def _sustained_baseline_deviation(x, baseline, cfg, direction='upper',
                                  min_rel=0.3):
    mask = _baseline_deviation_mask(
        x, baseline, direction=direction,
        k=getattr(cfg, 'metric_spike_k', 5.0), min_rel=min_rel)
    return _sustain_mask(mask, getattr(cfg, 'win_seg_min_len', 2))


def _resource_active_ts(metrics_dir, cfg, lag_map=None):
    """Causal-timeline timestamps with a sustained CPU/memory spike."""
    p = os.path.join(metrics_dir, 'svc_metric.csv')
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if 'timestamp' not in df.columns:
        return None
    if lag_map:
        df = align_metrics_df(df, lag_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    active = pd.Series(False, index=df.index)
    for c in df.columns:
        if c.endswith('&cpu_usage') or c.endswith('&mem_usage'):
            active = active | _series_spike(df[c])
    active = pd.Series(_sustain_mask(active, getattr(cfg, 'win_seg_min_len', 2)), index=df.index)
    return set(df['timestamp'][active])


def _latency_active_ts(lat, cfg):
    """Timestamps with a sustained aligned service-latency spike."""
    active = pd.Series(False, index=lat.index)
    for c in lat.columns:
        if c.endswith('&p50'):
            # -1 = no-data sentinel -> missing (0 is a real value, kept)
            col = pd.to_numeric(lat[c], errors='coerce').where(lambda v: v != -1)
            active = active | _series_spike(col)
    active = pd.Series(
        _sustain_mask(active, getattr(cfg, 'win_seg_min_len', 2)),
        index=lat.index)
    return set(lat.loc[active, 'timestamp'])


def adaptive_window(sr: pd.DataFrame, cfg, metrics_dir=None, latency=None,
                    lag_map=None):
    """Module B.0 (label-free): outer anomaly window.

    The window is the union of raw success-rate failures and aligned latency /
    CPU / memory spikes. This puts every lagged metric on the same causal
    timeline later consumed by severity and the GNN.
    """
    svc_cols = [c for c in sr.columns if c != 'timestamp']
    fail_ind = (sr[svc_cols] < 1.0).sum(axis=1)          # count of failing svcs
    active_ts = list(sr['timestamp'][fail_ind > 0])
    src = 'success_rate'
    if latency is not None:
        lat_ts = _latency_active_ts(latency, cfg)
        if lat_ts:
            active_ts += list(lat_ts)
            src += ' ∪ aligned_latency'
    if metrics_dir is not None:
        res_ts = _resource_active_ts(metrics_dir, cfg, lag_map=lag_map)
        if res_ts:
            active_ts += list(res_ts)
            src += ' ∪ aligned_resource(cpu/mem)'
    if not active_ts:
        return sr['timestamp'].iloc[0], sr['timestamp'].iloc[-1], fail_ind
    guard = pd.Timedelta(seconds=cfg.win_guard)
    onset, last = min(active_ts), max(active_ts)
    start = max(sr['timestamp'].iloc[0], onset - guard)
    end = min(sr['timestamp'].iloc[-1], last + guard)
    print(f'[B.0] window source: {src}')
    return start, end, fail_ind


def _resource_severity_map(metrics_dir, start, end, cfg, lag_map=None):
    """Per-service resource-deviation term from svc_metric cpu/mem within the
    window. The baseline is learned outside the window (preferably before it),
    so a long resource fault cannot become its own normal baseline. Only a
    robust, sustained spike contributes; magnitude is in [0,1)."""
    out = {}
    p = os.path.join(metrics_dir, 'svc_metric.csv')
    if not os.path.exists(p):
        return out
    df = pd.read_csv(p)
    if 'timestamp' not in df.columns:
        return out
    if lag_map:
        df = align_metrics_df(df, lag_map)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    w = df.loc[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    for c in df.columns:
        if not (c.endswith('&cpu_usage') or c.endswith('&mem_usage')):
            continue
        svc = c.split('&')[0]
        kind = 'cpu' if 'cpu' in c else 'mem'
        s = pd.to_numeric(w[c], errors='coerce')   # cpu/mem: no missing; masked downstream
        baseline = _outside_baseline(
            df, c, start, end, getattr(cfg, 'baseline_min_points', 3))
        if len(s) < 3 or baseline is None:
            continue
        spike = _sustained_baseline_deviation(
            s, baseline, cfg, direction='upper',
            min_rel=getattr(cfg, 'metric_min_rel', 0.3))
        if not spike.any():
            continue
        base_med = float(baseline.median())
        mx = float(s[np.asarray(spike, dtype=bool)].max())
        mag = ((mx - base_med) / (abs(mx) + 1e-9)
               if mx > base_med else 0.0)
        prev_mag, prev_det = out.get(svc, (0.0, {}))
        # Keep every resource channel that crossed its own baseline.  The
        # overall resource term remains the maximum so services with both CPU
        # and memory movement do not receive the resource weight twice.
        det = dict(prev_det)
        det[f'{kind}_dev'] = round(mag, 3)
        out[svc] = (max(prev_mag, mag), det)
    return out


def rank_anomalous_services(sr, lat, qps, start, end, metrics_dir=None,
                            cfg=None, lag_map=None):
    """Coarse severity ranking within [start, end].

    Every KPI is compared with an out-of-window baseline, preferring history
    before ``start``. ``lat`` and resource metrics are expected on the shared
    aligned timeline. severity = availability deviation + normalized latency
    spike + resource deviation + QPS deviation. Returns
    list[(svc, sev, detail)].
    """
    w_res = getattr(cfg, 'severity_resource_weight', 0.5) if cfg is not None else 0.5
    w_qps = getattr(cfg, 'severity_qps_weight', 0.1) if cfg is not None else 0.1
    min_points = getattr(cfg, 'baseline_min_points', 3) if cfg is not None else 3
    min_rel = getattr(cfg, 'metric_min_rel', 0.3) if cfg is not None else 0.3
    # no-request masking: only the -1 no-data sentinel is missing for both
    # latency and qps; 0 is a REAL value (kept) -- latency 0 or a qps collapse
    # to 0 are genuine signals, not missing.
    lat = lat.copy()
    _p50 = [c for c in lat.columns if c.endswith('&p50')]
    if _p50:
        lat[_p50] = lat[_p50].where(lat[_p50] != -1)
    qps = qps.copy()
    _qc = [c for c in qps.columns if c != 'timestamp']
    qps[_qc] = qps[_qc].where(qps[_qc] != -1)
    m = (sr['timestamp'] >= start) & (sr['timestamp'] <= end)
    sr_w = sr.loc[m]
    lat_w = lat.loc[(lat['timestamp'] >= start) & (lat['timestamp'] <= end)]
    qps_w = qps.loc[(qps['timestamp'] >= start) & (qps['timestamp'] <= end)]

    # Latency spike per service (anomaly-window peak / outside baseline median).
    lat_spike = {}
    for c in lat_w.columns:
        if c == 'timestamp' or '&p50' not in c:
            continue
        s = pd.to_numeric(lat_w[c], errors='coerce').dropna()
        baseline = _outside_baseline(lat, c, start, end, min_points)
        if len(s) < 3 or baseline is None or baseline.median() <= 0:
            continue
        spike_mask = _sustained_baseline_deviation(
            s, baseline, cfg, direction='upper', min_rel=min_rel)
        if spike_mask.any():
            peak = float(s[np.asarray(spike_mask, dtype=bool)].max())
            lat_spike[_svc_of(c)] = peak / (float(baseline.median()) + 1e-9)

    # Bidirectional workload deviation. QPS is weaker evidence than an actual
    # failure or resource spike, but it must use the same uncontaminated
    # baseline rule as every other KPI.
    qps_dev = {}
    for c in qps_w.columns:
        if c == 'timestamp':
            continue
        s = pd.to_numeric(qps_w[c], errors='coerce').dropna()
        baseline = _outside_baseline(qps, c, start, end, min_points)
        if len(s) < 3 or baseline is None:
            continue
        dev_mask = _sustained_baseline_deviation(
            s, baseline, cfg, direction='both', min_rel=min_rel)
        if not dev_mask.any():
            continue
        base_med = float(baseline.median())
        anomalous = s[np.asarray(dev_mask, dtype=bool)]
        magnitudes = (anomalous - base_med).abs() / np.maximum(
            np.maximum(anomalous.abs(), abs(base_med)), 1e-9)
        qps_dev[_svc_of(c)] = float(magnitudes.max())

    res_map = (_resource_severity_map(
        metrics_dir, start, end, cfg, lag_map=lag_map)
        if metrics_dir else {})

    # success_rate stats per service
    sr_stats = {}
    for c in sr_w.columns:
        if c == 'timestamp':
            continue
        s = pd.to_numeric(sr_w[c], errors='coerce').dropna()
        baseline = _outside_baseline(sr, c, start, end, min_points)
        if s.empty or baseline is None:
            continue
        fail_mask = _sustained_baseline_deviation(
            s, baseline, cfg, direction='lower',
            min_rel=getattr(cfg, 'availability_min_rel', 0.01))
        base_med = float(baseline.median())
        deficit = ((base_med - s).clip(lower=0) /
                   max(abs(base_med), 1e-9))
        fail_rate = float(deficit.where(fail_mask, 0.0).mean())
        sr_stats[c] = (fail_rate, int(np.asarray(fail_mask).sum()))

    services = set(sr_stats) | set(lat_spike) | set(qps_dev) | set(res_map)
    rows = []
    for c in services:
        fail_rate, n_fail = sr_stats.get(c, (0.0, 0))
        spike = lat_spike.get(c, 1.0)
        qps_term = qps_dev.get(c, 0.0)
        res_term, res_det = res_map.get(c, (0.0, {}))
        latency_term = 0.1 * np.log1p(max(spike - 1.0, 0.0))
        if (fail_rate <= 0 and n_fail == 0 and latency_term <= 0
                and qps_term <= 0 and res_term <= 0):
            continue
        severity = (fail_rate + latency_term + w_res * res_term
                    + w_qps * qps_term)
        # Preserve every detected anomaly dimension for Module C.  A service
        # can simultaneously exhibit availability, latency and resource
        # anomalies; reducing them to a single max would discard valid causes.
        type_scores = {}
        if fail_rate > 0:
            type_scores['availability'] = fail_rate
        if latency_term > 0:
            type_scores['latency'] = latency_term
        if qps_term > 0:
            type_scores['qps'] = w_qps * qps_term
        if res_term > 0:
            resource_devs = {
                kind: float(res_det[key])
                for kind, key in (('cpu', 'cpu_dev'), ('memory', 'mem_dev'))
                if key in res_det and float(res_det[key]) > 0
            }
            resource_sum = sum(resource_devs.values())
            for resource_type, dev in resource_devs.items():
                # Split the single resource contribution between all detected
                # channels; their scores still add up to w_res * res_term.
                type_scores[resource_type] = (
                    w_res * res_term * dev / resource_sum)
        detail = {'fail_rate': round(fail_rate, 3), 'n_fail_points': n_fail,
                  'latency_spike': round(spike, 2),
                  'qps_dev': round(qps_term, 3),
                  'res_dev': round(res_term, 3),
                  'anomaly_types': (sorted(type_scores, key=type_scores.get,
                                           reverse=True) or ['unknown']),
                  'anomaly_type_scores': {
                      k: round(float(v), 4) for k, v in type_scores.items()
                  }}
        detail.update(res_det)
        rows.append((c, severity, detail))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def _node_to_service(node: str) -> str:
    """Map a GNN node name (svc / pod / physical node) back to a metric
    service name for comparison with the severity ranking."""
    if node.startswith('node-'):
        return node
    # Strip the deployment-hash suffix from pod names: svc-<replicaset>-<pod>.
    # Only strip when the candidate suffix actually looks like a k8s hash (it
    # carries a digit); a purely alphabetic tail such as the '-network-image' of
    # 'agent-network-image' is part of the service name, not a pod hash, and must
    # be preserved (otherwise the service collapses to 'agent' and is never
    # matched against its own ground-truth label).
    m = re.search(r'-[0-9a-z]{6,10}-[0-9a-z]{5}$', node)
    if m and any(ch.isdigit() for ch in m.group(0)):
        return node[:m.start()]
    return node

def _infer_evaluation_root_cause(sample, services):
    """Infer benchmark ground truth from the sample directory name.

    This is evaluation-only metadata and is never fed into detection, ranking,
    candidate selection, or the LLM prompt.
    """
    sample_name = os.path.basename(os.path.normpath(sample))
    candidates = sorted(set(services), key=len, reverse=True)
    return next((svc for svc in candidates
                 if sample_name == svc or sample_name.startswith(svc + '_')),
                None)


def _service_rank(order, target):
    """One-based rank after normalizing/deduplicating service names; 0=miss."""
    if not target:
        return 0
    seen = []
    for item in order or []:
        service = _node_to_service(str(item))
        if service not in seen:
            seen.append(service)
    return seen.index(target) + 1 if target in seen else 0


def _write_sample_timing_log(sample, gnn_log_path, timestamp,
                             timing_seconds):
    """Persist completed-sample timings immediately in the sample directory."""
    if gnn_log_path and os.path.exists(gnn_log_path):
        log_path = gnn_log_path
    else:
        log_path = os.path.join(sample, f'sample-agent_rca_{timestamp}.log')
    from datetime import datetime
    with open(log_path, 'a', encoding='utf-8') as sample_log:
        print('\n===== Agent RCA stage timing =====', file=sample_log)
        print(f'sample: {sample}', file=sample_log)
        print(f'completed_at: '
              f'{datetime.now().astimezone().isoformat(timespec="seconds")}',
              file=sample_log)
        for stage in ('severity', 'gnn', 'llm', 'total'):
            print(f'{stage}: {timing_seconds.get(stage, 0.0):.3f}s',
                  file=sample_log)
    return log_path
