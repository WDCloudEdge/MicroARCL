"""Weight-free, scale-free severity via TEMPORAL OVERLAP of metric-type anomaly windows (scheme A: sum of pairwise Jaccard)."""
import os
import math
from itertools import combinations
import numpy as np
import pandas as pd

from sklearn.cluster import Birch
from sklearn.metrics import pairwise_distances

from run_agent_rca import (_outside_baseline, _sustained_baseline_deviation,
                           _sustain_mask)
from lag_align import align_metrics_df


# ------------------------- agent-aware detection -------------------------
# Agent services are sparse (requests are intermittent), have LARGE normal
# fluctuation (LLM inference is bursty), and are heterogeneous (incomparable
# scales). Anomaly evidence is therefore defined RELATIVE TO EACH SERVICE'S OWN
# normal noise (robust z vs its pre-window median/MAD), gated to time bins where
# the service actually had requests, and discounted when the request evidence is
# thin. This deflates the busy high-variance sinks that otherwise inflate their
# overlap windows purely by normal jitter.
def _prewin_clean(df, col, start):
    """Cleaned pre-window normal series (drop NaN and the -1 no-request
    sentinel). None if fewer than 3 points."""
    if col not in df.columns:
        return None
    pre = pd.to_numeric(df.loc[df['timestamp'] < start, col], errors='coerce')
    pre = pre[pre.notna() & (pre != -1)]
    return pre if len(pre) >= 3 else None


def _bin_of(ts_values, start, binw):
    return set(((pd.to_datetime(ts_values) - pd.to_datetime(start)
                 ).total_seconds() // binw).astype(int).tolist())


def _anom_bins_peak(df, col, start, end, direction, binw, min_len, gate,
                    mode, z_thresh, q, rel_thresh=0.3):
    """(anomaly-bin set, peak intensity in [0,1)) for one column, vs the
    service's OWN pre-window normal. mode:
      'quantile' : exceed normal empirical quantile (robust, distribution-aware);
      'z'        : robust z=(x-med)/(1.4826 MAD) > z_thresh (variance-normalized);
      'rel'      : ABLATION of variance-normalization -- fixed PERCENT threshold
                   |x-med|/|med| > rel_thresh, ignoring the service's own noise
                   (a normally-bursty service then trips on its own jitter).
    Sustained + optionally request-gated. Intensity = 1-exp(-peak/thresh)."""
    pre = _prewin_clean(df, col, start)
    if pre is None:
        return set(), 0.0
    med = float(pre.median())
    win = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    xf = pd.to_numeric(win[col], errors='coerce').where(lambda v: v != -1)
    valid = xf.notna()
    x, ts = xf[valid].values, win['timestamp'][valid]
    if len(x) < 1:
        return set(), 0.0
    scale = (max(abs(med), 1e-9) if mode == 'rel'          # percent (no var-norm)
             else max(1.4826 * float((pre - med).abs().median()), 1e-9))
    z = ((med - x) if direction == 'lower'
         else np.abs(x - med) if direction == 'both' else (x - med)) / scale
    thr = rel_thresh if mode == 'rel' else z_thresh
    if mode == 'quantile':
        if direction == 'lower':
            raw = x < float(pre.quantile(1 - q))
        elif direction == 'both':
            raw = (x > float(pre.quantile(q))) | (x < float(pre.quantile(1 - q)))
        else:
            raw = x > float(pre.quantile(q))
    else:                                                   # 'z' or 'rel'
        raw = z > thr
    m = _sustain_mask(np.asarray(raw), min_len)
    if not m.any():
        return set(), 0.0
    bins = _bin_of(ts.values[m], start, binw)
    if gate is not None:
        bins &= gate
    if not bins:
        return set(), 0.0
    return bins, 1.0 - math.exp(-float(np.max(z[m])) / thr)


def _anom_level(specs, start, end, binw, min_len, gate, mode, z_thresh, q,
                rel_thresh=0.3):
    """Union anomaly bins and max intensity of a priority level's columns."""
    bins, inten = set(), 0.0
    for df, col, direction in specs:
        b, it = _anom_bins_peak(df, col, start, end, direction, binw, min_len,
                                gate, mode, z_thresh, q, rel_thresh)
        bins |= b
        inten = max(inten, it)
    return bins, inten


def _request_bins(qps, s, start, end, binw):
    """Time bins where service s actually served requests (qps>0). None if qps
    is unavailable (no gating), so resource-only anomalies are unaffected."""
    if s not in qps.columns:
        return None
    win = qps[(qps['timestamp'] >= start) & (qps['timestamp'] <= end)]
    v = pd.to_numeric(win[s], errors='coerce')
    present = win['timestamp'][v.notna() & (v > 0)]
    return _bin_of(present.values, start, binw)


def _read_services(metrics_dir):
    df = pd.read_csv(os.path.join(metrics_dir, 'success_rate.csv'), nrows=1)
    return [c for c in df.columns if c != 'timestamp']


def _col_anom_bins(df, col, start, end, cfg, direction, min_rel, binw):
    """Set of anomalous time-bin ids for one column: sustained deviation from
    the out-of-window baseline, discretized to `binw`-second bins from `start`."""
    if col not in df.columns:
        return set()
    win = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    xf = pd.to_numeric(win[col], errors='coerce')
    valid = xf.notna()
    x, ts = xf[valid], win['timestamp'][valid]
    if len(x) < 3:
        return set()
    baseline = _outside_baseline(df, col, start, end,
                                 getattr(cfg, 'baseline_min_points', 3))
    if baseline is None:
        return set()
    mask = np.asarray(_sustained_baseline_deviation(
        x, baseline, cfg, direction=direction, min_rel=min_rel), dtype=bool)
    if not mask.any():
        return set()
    secs = (pd.to_datetime(ts.values[mask]) - pd.to_datetime(start)
            ).total_seconds()
    return set((np.asarray(secs) // binw).astype(int).tolist())


def _jaccard(a, b):
    u = len(a | b)
    return (len(a & b) / u) if u else 0.0


def _col_peak_rel(df, col, start, end, cfg, direction, min_rel):
    """Self-relative peak deviation of one column vs its own out-of-window
    baseline (|peak-base|/scale in [0,1)); 0 if no sustained anomaly. Scale-free
    (each service judged against its OWN normal), so it points at the service
    that jumped most relative to itself -- unlike duration, it does not reward a
    perpetually-busy sink. Used for the single-active-level severity."""
    if col not in df.columns:
        return 0.0
    win = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    x = pd.to_numeric(win[col], errors='coerce').dropna()
    if len(x) < 3:
        return 0.0
    baseline = _outside_baseline(df, col, start, end,
                                 getattr(cfg, 'baseline_min_points', 3))
    if baseline is None:
        return 0.0
    mask = np.asarray(_sustained_baseline_deviation(
        x, baseline, cfg, direction=direction, min_rel=min_rel), dtype=bool)
    if not mask.any():
        return 0.0
    base = float(baseline.median())
    xa = x.values[mask]
    if direction == 'lower':
        dev = base - xa.min()
    elif direction == 'both':
        dev = np.abs(xa - base).max()
    else:
        dev = xa.max() - base
    return max(0.0, float(dev) / (abs(base) + abs(dev) + 1e-9))     # [0,1)


def _col_peak(df, col, start, end, cfg, direction, min_rel, binw):
    """(peak_bin, peak_rel) for one column: the time bin of its STRONGEST
    sustained deviation and its self-relative intensity in [0,1). (None, 0) if no
    sustained anomaly. Peak TIME (not window duration) is what aligns a service's
    cause and symptom -- a root's resource and latency peak together, while a
    downstream victim's latency peak lags, so a busy victim no longer wins just
    by staying anomalous longer."""
    if col not in df.columns:
        return None, 0.0
    win = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
    xf = pd.to_numeric(win[col], errors='coerce')
    valid = xf.notna()
    x, ts = xf[valid], win['timestamp'][valid]
    if len(x) < 3:
        return None, 0.0
    baseline = _outside_baseline(df, col, start, end,
                                 getattr(cfg, 'baseline_min_points', 3))
    if baseline is None:
        return None, 0.0
    mask = np.asarray(_sustained_baseline_deviation(
        x, baseline, cfg, direction=direction, min_rel=min_rel), dtype=bool)
    if not mask.any():
        return None, 0.0
    base = float(baseline.median())
    xa, ta = x.values[mask], ts.values[mask]
    if direction == 'lower':
        i = int(np.argmin(xa))
        dev = base - xa[i]
    elif direction == 'both':
        d = np.abs(xa - base)
        i = int(np.argmax(d))
        dev = d[i]
    else:
        i = int(np.argmax(xa))
        dev = xa[i] - base
    peak_bin = int((pd.to_datetime(ta[i]) - pd.to_datetime(start)
                    ).total_seconds() // binw)
    return peak_bin, max(0.0, float(dev) / (abs(base) + abs(dev) + 1e-9))


def _level_peak(specs, start, end, cfg, min_rel, binw):
    """Peak (bin, intensity) of a priority LEVEL = the strongest of its columns.
    specs = [(df, col, direction), ...]. (None, 0) if the level is inactive."""
    best_bin, best_int = None, 0.0
    for df, col, direction in specs:
        b, inten = _col_peak(df, col, start, end, cfg, direction, min_rel, binw)
        if b is not None and inten >= best_int:
            best_bin, best_int = b, inten
    return best_bin, best_int


def overlap_rank(metrics_dir, start, end, cfg, lag_map=None):
    """Per-service overlap severity. Returns list[(svc, severity, detail)] sorted
    by (severity, union-window size) desc -- same shape as rank_anomalous_services
    so it drops into _prep as detector='overlap'."""
    services = _read_services(metrics_dir)
    start, end = pd.to_datetime(start), pd.to_datetime(end)
    min_rel = getattr(cfg, 'metric_min_rel', 0.3)
    avail_rel = getattr(cfg, 'availability_min_rel', 0.01)

    def _load(name, align):
        df = pd.read_csv(os.path.join(metrics_dir, name))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return align_metrics_df(df, lag_map) if (align and lag_map) else df

    sr = _load('success_rate.csv', False)     # user-facing base, not aligned
    qps = _load('svc_qps.csv', False)         # workload reference, not aligned
    lat = _load('latency.csv', True)          # symptom, latency-lag aligned
    sm = _load('svc_metric.csv', True)        # resource CAUSE, lag aligned
    # no-request masking: only the -1 no-data sentinel is missing (for both qps
    # and latency); 0 is a REAL value kept as signal (qps collapse / latency 0).
    # (_request_bins uses qps>0, so a genuine 0 still counts as idle for gating.)
    _qc = [c for c in qps.columns if c != 'timestamp']
    qps[_qc] = qps[_qc].where(qps[_qc] != -1)
    _p50 = [c for c in lat.columns if c.endswith('&p50')]
    if _p50:
        lat[_p50] = lat[_p50].where(lat[_p50] != -1)

    d = sr['timestamp'].diff().dt.total_seconds().median()
    binw = float(d) if d and d > 0 else 5.0

    z_thresh = float(getattr(cfg, 'overlap_z', 3.0))
    n0 = float(getattr(cfg, 'overlap_evidence_n0', 2.0))
    mode = getattr(cfg, 'overlap_detect', 'quantile')   # 'z' or 'quantile'
    q = float(getattr(cfg, 'overlap_q', 0.9))
    gate_R = bool(getattr(cfg, 'overlap_gate_R', False))  # opt1 hurt -> off
    min_ev = float(getattr(cfg, 'overlap_min_ev', 0.0))  # opt3: drop ultra-sparse
    min_len = getattr(cfg, 'win_seg_min_len', 2)
    rows = []
    for s in services:
        req = _request_bins(qps, s, start, end, binw)   # request-present bins
        # variance/quantile-normalized, request-gated anomaly windows per level
        S, Si = _anom_level([(sr, s, 'lower')], start, end, binw, min_len,
                            req, mode, z_thresh, q)
        K, Ki = _anom_level([(qps, s, 'both'), (lat, f'{s}&p50', 'upper')],
                            start, end, binw, min_len, req, mode, z_thresh, q)
        R, Ri = _anom_level([(sm, f'{s}&cpu_usage', 'upper'),
                             (sm, f'{s}&mem_usage', 'upper'),
                             (sm, f'{s}&net_receive', 'upper'),
                             (sm, f'{s}&net_trainsmit', 'upper')],
                            start, end, binw, min_len,
                            (req if gate_R else None), mode, z_thresh, q)
        ev = len(req) if req is not None else len(R)
        if ev < min_ev:                                  # opt3: too sparse
            continue
        active = [(nm, w, it) for nm, w, it in
                  (('S', S, Si), ('K', K, Ki), ('R', R, Ri)) if w]
        n = len(active)
        if n == 0:
            continue
        # tiered weight-free severity: primary = #active priority levels (n-1);
        # fractional in [0,1) = temporal overlap (mean pairwise Jaccard) if >=2
        # levels else single-level self-relative peak intensity.
        if n >= 2:
            frac = sum(_jaccard(a, b) for (_, a, _), (_, b, _)
                       in combinations(active, 2)) / (n * (n - 1) / 2)
        else:
            frac = 1.0 - math.exp(-active[0][2])
        # sparsity discount: thin request evidence -> less trustworthy severity.
        ev = len(req) if req is not None else len(R)
        conf = 1.0 - math.exp(-ev / n0)
        sev = ((n - 1) + frac) * conf
        detail = {'overlap_sev': round(sev, 4), 'n_levels': n, 'conf': round(conf, 3),
                  'win_S': len(S), 'win_K': len(K), 'win_R': len(R),
                  'req_bins': (len(req) if req is not None else -1),
                  'J_SK': round(_jaccard(S, K), 3),
                  'J_SR': round(_jaccard(S, R), 3),
                  'J_KR': round(_jaccard(K, R), 3),
                  'anomaly_types': [nm for nm, _, _ in active],
                  'union': len(S | K | R)}
        rows.append((s, sev, detail))
    rows.sort(key=lambda r: (r[1], r[2]['union']), reverse=True)
    return rows


def _active_types(S, K, R):
    t = []
    if R:
        t.append('resource')
    if K:
        t.append('latency')
    if S:
        t.append('availability')
    return t or ['unknown']


# ---------------- agent-aware Birch (detector='abirch') ----------------
# Keep plain Birch's strength -- WITHIN-(adaptive-)window clustering per signal,
# which needs no pre-window baseline and thus uses ALL signals -- and add only
# two light agent-aware tweaks: (2) request-gating and (3) sparsity confidence.
def agent_aware_birch_ranking(metrics_dir, start, end, cfg, lag_map=None):
    """Agent-aware Birch. (1) clusters within the ADAPTIVE anomaly window
    [start,end]; per (service, signal in cpu/mem/net/latency/success_rate/qps)
    runs the plain within-window Birch (L2, no pre-window baseline needed).
    (2) REQUEST-GATING: an anomalous timestamp counts only if the service served
    a request then (qps>0), dropping no-request / -1 false spikes. (3) SPARSITY:
    the service's anomaly evidence is scaled by 1-exp(-#request-bins/n0) so a
    barely-used service is not over-trusted. Ranks by that evidence."""
    from anomaly_detection import anomaly_detection_with_smoothing_series
    thr = float(getattr(cfg, 'abirch_threshold', 0.03))       # L2-space (plain Birch)
    n0 = float(getattr(cfg, 'abirch_n0', 2.0))
    services = _read_services(metrics_dir)
    start, end = pd.to_datetime(start), pd.to_datetime(end)

    def _load(name, align):
        p = os.path.join(metrics_dir, name)
        if not os.path.exists(p):
            return None
        df = pd.read_csv(p)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return align_metrics_df(df, lag_map) if (align and lag_map) else df

    sm = _load('svc_metric.csv', True)
    lat = _load('latency.csv', True)
    sr = _load('success_rate.csv', False)
    qps = _load('svc_qps.csv', False)

    def _req_ns(s):
        """int64-ns set of window timestamps where s served requests (qps>0);
        None if qps unavailable (then no gating)."""
        if qps is None or s not in qps.columns:
            return None
        w = qps[(qps['timestamp'] >= start) & (qps['timestamp'] <= end)]
        v = pd.to_numeric(w[s], errors='coerce')
        return set(w['timestamp'][v.notna() & (v > 0)].astype('int64').tolist())

    rows = []
    for s in services:
        req = _req_ns(s)
        specs = []
        if sm is not None:
            specs += [(sm, f'{s}{suf}') for suf in
                      ('&cpu_usage', '&mem_usage', '&net_receive', '&net_trainsmit')]
        if lat is not None:
            specs.append((lat, f'{s}&p50'))
        if sr is not None:
            specs.append((sr, s))
        if qps is not None:
            specs.append((qps, s))
        pts, hit = 0, []
        for df, col in specs:
            if df is None or col not in df.columns:
                continue
            w = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)][['timestamp', col]]
            v = pd.to_numeric(w[col], errors='coerce').where(lambda z: z != -1)
            w = w.assign(_v=v).dropna(subset=['_v']).reset_index(drop=True)
            if len(w) < 3:
                continue
            is_anom, idxs = anomaly_detection_with_smoothing_series(
                w['_v'], threshold=thr)
            if not is_anom or not idxs:
                continue
            ats = w['timestamp'].to_numpy()[list(idxs)]
            if req is not None:                          # (2) request-gating
                ats = [t for t in ats if int(pd.Timestamp(t).value) in req]
            if ats:
                pts += len(ats)
                hit.append(col.split('&')[-1] if '&' in col else 'kpi')
        if pts > 0:
            ev = len(req) if req is not None else 0
            conf = (1.0 - math.exp(-ev / n0)) if req is not None else 1.0  # (3)
            score = float(pts) * conf
            if score > 0:
                rows.append([s, score, {'abirch_points': pts,
                                        'conf': round(conf, 2),
                                        'anomaly_types': hit or ['unknown']}])
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows
