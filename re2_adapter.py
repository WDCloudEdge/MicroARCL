"""RE2 (RCAEval) dataset adapter for the MicroARCL localizer."""
import os
import re
import json
import glob

import numpy as np
import pandas as pd

from util.utils import timestamp_2_time_string


# ----------------------------------------------------------------------------
# case discovery + labels
# ----------------------------------------------------------------------------
def _case_file(case_dir, kind):
    """Locate metrics/traces/logs in a case dir (Parquet preferred over the
    original json/csv), or None when the case carries none of that modality."""
    names = {'metrics': ('metrics.parquet', 'metrics.json'),
             'traces': ('traces.parquet', 'traces.csv'),
             'logs': ('logs.parquet', 'logs.csv')}[kind]
    for n in names:
        p = os.path.join(case_dir, n)
        if os.path.exists(p):
            return p
    return None


def discover_cases(root):
    """Every RE2 case directory under `root` that has metrics + an inject time.

    A case dir is any directory containing (metrics.json | metrics.parquet) and
    inject_time.txt, searched recursively so `root` may be a single suite
    (e.g. .../data/re2ob) or a parent holding several suites."""
    out = []
    for inj in glob.glob(os.path.join(root, '**', 'inject_time.txt'),
                         recursive=True):
        cdir = os.path.dirname(inj)
        if _case_file(cdir, 'metrics'):
            out.append(cdir)
    return sorted(set(out))


def parse_case_label(case_dir):
    """(service, fault, instance) from the `{suite}_{service}_{fault}_{inst}`
    directory name. Service names use hyphens (train-ticket) or no separator
    (online-boutique), never '_', so the underscore split is unambiguous."""
    name = os.path.basename(os.path.normpath(case_dir))
    parts = name.split('_')
    if len(parts) >= 4:
        return parts[1], parts[2], '_'.join(parts[3:])
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    return name, 'unknown', '0'


def read_inject_time(case_dir):
    with open(os.path.join(case_dir, 'inject_time.txt')) as f:
        return int(f.readline().strip())


# ----------------------------------------------------------------------------
# metric parsing
# ----------------------------------------------------------------------------
# metric-column suffixes, longest first so `_latency-90` wins over `_latency`.
_METRIC_SUFFIXES = [
    ('_latency-50', 'lat50'), ('_latency-90', 'lat90'), ('_latency-99', 'lat99'),
    ('_latency', 'lat90'), ('_lat_90', 'lat90'), ('_lat_50', 'lat50'),
    ('_cpu', 'cpu'), ('_mem', 'mem'), ('_memory', 'mem'),
    ('_net_receive', 'net_rx'), ('_network_receive', 'net_rx'),
    ('_net_transmit', 'net_tx'), ('_network_transmit', 'net_tx'),
    ('_net', 'net_rx'), ('_diskio', 'disk'), ('_disk', 'disk'),
    ('_socket', 'socket'), ('_fd', 'fd'),
    ('_workload', 'workload'), ('_error', 'error'),   # workload = native QPS
]


def _read_metrics(case_dir):
    p = _case_file(case_dir, 'metrics')
    if p.endswith('.parquet'):
        df = pd.read_parquet(p)
    else:
        raw = {k: v for k, v in json.load(open(p)).items() if v}
        union = sorted({pt[0] for v in raw.values() for pt in v})
        pos = {t: i for i, t in enumerate(union)}
        arr = np.full((len(union), len(raw)), np.nan)
        cols = list(raw)
        for j, k in enumerate(cols):
            for ts, val in raw[k]:
                if val is not None:
                    arr[pos[ts], j] = val
        df = pd.DataFrame(arr, columns=cols)
        df.insert(0, 'time', np.asarray(union, dtype=np.int64))
    df = df.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    return df


def _parse_metric_columns(cols):
    """{service: {kind: column}} for every recognized `{service}_{kind}` column."""
    out = {}
    for c in cols:
        if c == 'time':
            continue
        for suf, kind in _METRIC_SUFFIXES:
            if c.endswith(suf):
                svc = c[:-len(suf)]
                if svc:
                    out.setdefault(svc, {})[kind] = c
                break
    return out


# ----------------------------------------------------------------------------
# trace parsing
# ----------------------------------------------------------------------------
def _read_traces(case_dir):
    p = _case_file(case_dir, 'traces')
    if p is None:
        return None
    df = (pd.read_parquet(p) if p.endswith('.parquet')
          else pd.read_csv(p, low_memory=False))
    return df if df is not None and len(df) else None


def _map_trace_services(trace_names, canonical):
    """serviceName (traces) -> canonical metric-service name. Handles the common
    `frontendservice`(trace) vs `frontend`(metric) mismatch and falls back to a
    prefix match; unmapped names return None and are dropped."""
    cl = {c.lower(): c for c in canonical}
    mapping = {}
    for sn in trace_names:
        if sn is None or (isinstance(sn, float) and np.isnan(sn)):
            continue
        s = str(sn)
        low = s.lower()
        hit = None
        if s in canonical:
            hit = s
        elif low in cl:
            hit = cl[low]
        else:
            stripped = low[:-7] if low.endswith('service') else low
            for c in canonical:
                clc = c.lower()
                clc_s = clc[:-7] if clc.endswith('service') else clc
                if stripped == clc or stripped == clc_s or low == clc + 'service':
                    hit = c
                    break
            if hit is None:                              # last resort: prefix
                for c in canonical:
                    if low.startswith(c.lower()) or c.lower().startswith(low):
                        hit = c
                        break
        mapping[s] = hit
    return mapping


def _bin_trace_series(traces, canonical, time_index):
    """From raw spans build per-service, per-second (aligned to `time_index`):
      qps[svc]          request count,
      success_rate[svc] 1 - error-fraction,
    and a list of (caller, callee, second, latency_us) call edges."""
    svc_map = _map_trace_services(traces['serviceName'].unique(), canonical)
    start_us = pd.to_numeric(traces['startTime'], errors='coerce')
    sec = (start_us // 1_000_000).astype('Int64')
    svc = traces['serviceName'].map(svc_map)
    status = pd.to_numeric(traces.get('statusCode'), errors='coerce')
    dur = pd.to_numeric(traces.get('duration'), errors='coerce')
    err = (status.notna()) & (status != 0)

    base = pd.DataFrame({'svc': svc, 'sec': sec, 'err': err})
    base = base.dropna(subset=['svc', 'sec'])
    grp = base.groupby(['svc', 'sec'])
    cnt = grp.size().unstack('svc').reindex(time_index).fillna(0.0)
    errc = grp['err'].sum().unstack('svc').reindex(time_index).fillna(0.0)
    qps = cnt
    with np.errstate(invalid='ignore', divide='ignore'):
        sr = 1.0 - (errc / cnt.replace(0.0, np.nan))
    sr = sr.fillna(1.0)                                   # no traffic -> healthy

    # call edges: parent span's service -> child span's service, child latency.
    span_svc = dict(zip(zip(traces['traceID'].astype(str),
                            traces['spanID'].astype(str)), svc))
    parent = traces.get('parentSpanID')
    edges = []
    if parent is not None:
        for tid, p, cs, se, d in zip(traces['traceID'].astype(str),
                                     parent.astype(str), svc, sec, dur):
            caller = span_svc.get((tid, p))
            if caller and cs and pd.notna(se) and caller != cs:
                edges.append((caller, cs, int(se), float(d) if pd.notna(d) else np.nan))
    return qps, sr, edges, svc_map


def _trace_call_chains(traces, svc_map, max_traces=3000):
    """Service-level root-to-leaf paths using actual parent span IDs.

    Sorting all spans by start time can create false edges between siblings.
    """
    df = pd.DataFrame({
        'tid': traces['traceID'].astype(str),
        'sid': traces['spanID'].astype(str),
        'parent': traces['parentSpanID'].astype(str),
        'svc': traces['serviceName'].map(svc_map),
        'st': pd.to_numeric(traces['startTime'], errors='coerce'),
    }).dropna(subset=['st'])
    chains = []
    for tid, g in df.groupby('tid', sort=False):
        spans = {row.sid: (row.parent, row.svc)
                 for row in g.sort_values('st').itertuples(index=False)}
        parents = {parent for parent, _ in spans.values() if parent in spans}
        for leaf in spans.keys() - parents:
            seq, seen, sid = [], set(), leaf
            while sid in spans and sid not in seen:
                seen.add(sid)
                parent, svc = spans[sid]
                if svc is not None:
                    seq.append(svc)
                sid = parent
            seq.reverse()
            seq = [s for i, s in enumerate(seq) if i == 0 or s != seq[i-1]]
            if len(seq) >= 2:
                chains.append({'trace_id': tid, 'path': seq})
            if len(chains) >= max_traces:
                return chains
    return chains


# ----------------------------------------------------------------------------
# materialization
# ----------------------------------------------------------------------------
def _edge_latency_frame(edges, time_index):
    """{caller}_{callee}&p50/p90/p99 per-second quantiles from call-edge latencies
    (microseconds -> seconds)."""
    if not edges:
        return pd.DataFrame(index=time_index)
    ed = pd.DataFrame(edges, columns=['caller', 'callee', 'sec', 'lat'])
    ed = ed.dropna(subset=['lat'])
    ed['lat'] = ed['lat'] / 1_000_000.0
    ed['edge'] = ed['caller'] + '_' + ed['callee']
    out = {}
    for (edge, sec), v in ed.groupby(['edge', 'sec'])['lat']:
        out.setdefault(f'{edge}&p50', {})[sec] = float(v.quantile(0.5))
        out.setdefault(f'{edge}&p90', {})[sec] = float(v.quantile(0.9))
        out.setdefault(f'{edge}&p99', {})[sec] = float(v.quantile(0.99))
    frame = pd.DataFrame(out).reindex(time_index).fillna(-1.0)
    return frame


def ensure_graph_csv(sample_dir, ns='re2'):
    """Build graph.csv from trace-derived service paths, never placeholder edges."""
    metrics_dir = os.path.join(sample_dir, ns, 'metrics')
    gpath = os.path.join(metrics_dir, 'graph.csv')
    with open(os.path.join(sample_dir, ns, 'graph', 'call_chains.json')) as f:
        chains = json.load(f)
    edges = sorted({(a, b) for chain in chains
                    for a, b in zip(chain.get('path', []),
                                    chain.get('path', [])[1:]) if a != b})
    if not edges:
        raise ValueError(f'No trace-derived service edges for {sample_dir}; '
                         'provide traces.csv or traces.parquet for this case')
    sr = pd.read_csv(os.path.join(metrics_dir, 'success_rate.csv'))
    pd.DataFrame(((ts, a, b) for ts in sr['timestamp'] for a, b in edges),
                 columns=['timestamp', 'source', 'destination']).to_csv(
                     gpath, index=False)
    return gpath


def materialize_case(case_dir, out_root, ns='re2', force=False):
    """Convert one RE2 case into the MicroARCL layout under
    `out_root/<case>/<ns>/`. Returns (sample_dir, inject_time, services).
    Cached: skips work when the metrics dir already exists unless `force`."""
    case_name = os.path.basename(os.path.normpath(case_dir))
    sample_dir = os.path.join(out_root, case_name)
    ns_dir = os.path.join(sample_dir, ns)
    metrics_dir = os.path.join(ns_dir, 'metrics')
    graph_dir = os.path.join(ns_dir, 'graph')
    inject_time = read_inject_time(case_dir)
    done = os.path.join(metrics_dir, '.materialized')
    trace_path = _case_file(case_dir, 'traces')
    chains_path = os.path.join(graph_dir, 'call_chains.json')
    empty_cached_chains = False
    if trace_path is not None and os.path.exists(chains_path):
        with open(chains_path) as f:
            empty_cached_chains = not json.load(f)
    stale_traces = (trace_path is not None and os.path.exists(done)
                    and (os.path.getmtime(trace_path) > os.path.getmtime(done)
                         or empty_cached_chains))
    if os.path.exists(done) and not force and not stale_traces:
        services = [c for c in pd.read_csv(
            os.path.join(metrics_dir, 'success_rate.csv'), nrows=1).columns
            if c != 'timestamp']
        return sample_dir, inject_time, services
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    metrics = _read_metrics(case_dir)
    time_index = pd.Index(metrics['time'].astype(np.int64), name='time')
    ts_str = [timestamp_2_time_string(int(t)) for t in time_index]
    colmap = _parse_metric_columns(metrics.columns)
    services = sorted(s for s, kinds in colmap.items()
                      if {'cpu', 'mem', 'lat50', 'lat90'} & set(kinds))
    metrics = metrics.set_index(metrics['time'].astype(np.int64))

    def _col(svc, kind):
        c = colmap.get(svc, {}).get(kind)
        return metrics[c] if c is not None else None

    # --- svc_metric.csv (cpu / mem / net) ---
    sm = {'timestamp': ts_str}
    for s in services:
        cpu, mem = _col(s, 'cpu'), _col(s, 'mem')
        sm[f'{s}&cpu_usage'] = (cpu.values if cpu is not None
                                else np.zeros(len(time_index)))
        sm[f'{s}&mem_usage'] = (mem.values if mem is not None
                                else np.zeros(len(time_index)))
        rx, tx = _col(s, 'net_rx'), _col(s, 'net_tx')
        if rx is not None:
            sm[f'{s}&net_receive'] = rx.values
        if tx is not None:
            sm[f'{s}&net_trainsmit'] = tx.values     # sic: pipeline spelling
    pd.DataFrame(sm).to_csv(os.path.join(metrics_dir, 'svc_metric.csv'),
                            index=False)

    # --- latency.csv (p50/p90/p99 from metric latency) ---
    lat = {'timestamp': ts_str}
    for s in services:
        p50 = _col(s, 'lat50')
        p90 = _col(s, 'lat90')
        p50 = p50 if p50 is not None else p90
        p90 = p90 if p90 is not None else p50
        if p50 is None:
            continue
        lat[f'{s}&p50'] = p50.values
        lat[f'{s}&p90'] = p90.values
        lat[f'{s}&p99'] = p90.values
    pd.DataFrame(lat).to_csv(os.path.join(metrics_dir, 'latency.csv'),
                             index=False)

    # --- instance.csv (one instance per service, cpu proxy) ---
    inst = {'timestamp': ts_str}
    for s in services:
        cpu = _col(s, 'cpu')
        inst[f'{s}_0'] = (cpu.values if cpu is not None
                          else np.zeros(len(time_index)))
    pd.DataFrame(inst).to_csv(os.path.join(metrics_dir, 'instance.csv'),
                              index=False)

    # --- trace-derived: svc_qps.csv, success_rate.csv, call.csv, call_chains ---
    traces = _read_traces(case_dir)
    if traces is not None and 'serviceName' in traces.columns:
        qps, sr, edges, svc_map = _bin_trace_series(traces, services, time_index)
        chains = _trace_call_chains(traces, svc_map)
        call_frame = _edge_latency_frame(edges, time_index)
    else:
        qps = pd.DataFrame(index=time_index)
        sr = pd.DataFrame(index=time_index)
        chains, call_frame = [], pd.DataFrame(index=time_index)

    qps_out = {'timestamp': ts_str}
    sr_out = {'timestamp': ts_str}
    for s in services:
        wl = _col(s, 'workload')                       # native QPS when present
        if wl is not None:
            qps_out[s] = wl.values
        elif s in qps.columns:                         # else trace request rate
            qps_out[s] = qps[s].values
        else:
            qps_out[s] = np.zeros(len(time_index))
        sr_out[s] = (sr[s].values if s in sr.columns
                     else np.ones(len(time_index)))
    pd.DataFrame(qps_out).to_csv(os.path.join(metrics_dir, 'svc_qps.csv'),
                                 index=False)
    pd.DataFrame(sr_out).to_csv(os.path.join(metrics_dir, 'success_rate.csv'),
                                index=False)

    call_out = call_frame.reset_index(drop=True)
    call_out.insert(0, 'timestamp', ts_str)
    call_out.to_csv(os.path.join(metrics_dir, 'call.csv'), index=False)

    with open(os.path.join(graph_dir, 'call_chains.json'), 'w') as f:
        json.dump(chains, f)

    open(done, 'w').close()
    return sample_dir, inject_time, services
