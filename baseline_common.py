import os
import re
import sys
import glob
import math
import time
import json
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime

import run_agent_rca as R
from run_all_abnormal import ABN, TOP_K_CUTOFFS, parse_labels, _Tee, _topk_metrics
from Config import Config
from anomaly_detection import get_anomaly_by_df
from lag_align import (align_metrics_df, compute_service_lags,
                       compute_chain_lags, _service_of_pod)
from util.utils import time_string_2_timestamp
from ppr_localize import ppr_rank
import residual_localize as RES
from execution_graph import (load_execution_graphs, aggregate_topk_slices,
                             build_log_evidence, graph_failed_for_services)
# from llm_localize import llm_localize


# Root-cause service -> category for per-category breakdown.
CATEGORY_MDOC = {
    'agent-network-pdf-parsing': 'doc',
    'agent-network-word-gen': 'doc',
    'agent-network-image': 'image',
    'agent-network-ocr': 'image',
    'agent-network-planner': 'sched',
    'agent-network-summarizer': 'sched',
}
CATEGORY_MARBLE = {
    'agent-network-planner': 'sched',
    'agent-network-summarizer': 'sched',
    'agent-network-marble-coding': 'coding',
    'agent-network-marble-database': 'db',
    'agent-network-marble-minecraft': 'game',
    'agent-network-marble-research': 'web',
}
CATEGORIES_MDOC = ('doc', 'image', 'sched')
CATEGORIES_MARBLE = ('web', 'coding', 'db', 'game', 'sched')

CATEGORY_BY_DATASET = {
    'MDOC': CATEGORY_MDOC,
    'MARBLEBench': CATEGORY_MARBLE,
}
CATEGORIES = os.environ.get('AGENT_DATASET', 'MDOC') == 'MDOC' and CATEGORIES_MDOC or CATEGORIES_MARBLE
AGENT_DATASET = os.environ.get('AGENT_DATASET', 'MDOC').strip()
CATEGORY = CATEGORY_BY_DATASET.get(AGENT_DATASET, CATEGORY_MDOC)

# Dataset layout: abnormal/<group>/<load>/<sample>, group in {single, multi},
# load e.g. load-3 / load-5 / load-3-multi. Results report a 4-level hierarchy:
# overall (abnormal) + per-group (single/multi) + per-load + per-category.
ABN_BASE = os.path.dirname(ABN)


def _discover(groups=None):
    """[(group, load, load_dir)] for every load dir holding *_label.txt under
    abnormal/<group>/<load>. groups=None -> all groups present."""
    out = []
    grps = groups or sorted(d for d in os.listdir(ABN_BASE)
                            if os.path.isdir(os.path.join(ABN_BASE, d)))
    for g in grps:
        gdir = os.path.join(ABN_BASE, g)
        if not os.path.isdir(gdir):
            continue
        for ld in sorted(os.listdir(gdir)):
            adir = os.path.join(gdir, ld)
            if os.path.isdir(adir) and glob.glob(os.path.join(adir, '*_label.txt')):
                out.append((g, ld, adir))
    return out


def _category(gt):
    return CATEGORY.get(gt, 'other')


def _rc_names_gt(rc, gt):
    """Whether the free-text root cause names exactly the ground-truth service.

    Uses a service-name boundary (not raw substring) so a sibling like
    'agent-network-image-gen' does NOT count as a hit for 'agent-network-image'.
    """
    if not gt or not rc:
        return False
    return re.search(r'(?<![-a-z0-9])' + re.escape(gt) + r'(?![-a-z0-9])',
                     rc) is not None


def _to_service(node, services):
    """Map a Birch anomaly node (svc name or instance/pod prefix) to a known
    metric service name; None if it maps to nothing we track."""
    if node in services:
        return node
    s = _service_of_pod(node)
    if s in services:
        return s
    for svc in sorted(services, key=len, reverse=True):
        if node.startswith(svc):
            return svc
    return None


def _birch_ranking(cfg, sample_dir, metrics_dir, start_ts, end_ts, lag_map,
                   gate=False, sparsity=False):
    """Run Birch (get_anomaly_by_df, unchanged), aggregate #anomalous points to
    service level. gate/sparsity are the agent-aware add-ons (both off == the
    original Birch baseline, so detector='birch' vs 'abirch' is a pure ablation):
      gate     : (2) REQUEST-GATING -- keep only anomaly timestamps where the
                 service actually served a request (qps>0);
      sparsity : (3) scale the score by 1-exp(-#request-bins/n0) so a
                 barely-used service is not over-trusted.
    (1) adaptive window is inherent (start_ts/end_ts is Module B.0's window)."""
    import pandas as pd
    services = [c for c in pd.read_csv(
        os.path.join(metrics_dir, 'success_rate.csv'), nrows=1).columns
        if c != 'timestamp']
    anomalies, ats = get_anomaly_by_df(
        cfg, sample_dir, metrics_dir, 'baseline', start_ts, end_ts,
        lag_map=lag_map)
    # per-service request-present timestamps (unix key) inside the window
    req, n_req = {}, {}
    if gate or sparsity:
        qdf = pd.read_csv(os.path.join(metrics_dir, 'svc_qps.csv'))
        ts_key = [int(time_string_2_timestamp(str(t))) for t in qdf['timestamp']]
        for c in qdf.columns:
            if c == 'timestamp':
                continue
            v = pd.to_numeric(qdf[c], errors='coerce')
            present = {ts_key[i] for i in range(len(v))
                       if pd.notna(v.iloc[i]) and v.iloc[i] > 0
                       and start_ts <= ts_key[i] <= end_ts}
            req[c], n_req[c] = present, len(present)
    n0 = float(getattr(cfg, 'abirch_n0', 2.0))
    score = {}
    for node in anomalies:
        svc = _to_service(node, services)
        if svc is None:
            continue
        pts_ts = ats.get(node, [])
        if gate and svc in req:                       # (2) request-gating
            pts_ts = [t for t in pts_ts
                      if int(time_string_2_timestamp(str(t))) in req[svc]]
        pts = len(pts_ts)
        score[svc] = score.get(svc, 0.0) + (pts if pts > 0 else (0 if gate else 1))
    if sparsity:                                      # (3) sparsity confidence
        for svc in list(score):
            ev = n_req.get(svc, 0)
            score[svc] *= (1.0 - math.exp(-ev / n0)) if ev > 0 else 0.0
    ranking = sorted(
        ([svc, sc, {'birch_points': round(sc, 3), 'anomaly_types': ['unknown']}]
         for svc, sc in score.items() if sc > 0),
        key=lambda r: r[1], reverse=True)
    return ranking


def _prep(cfg, sample_dir, align, detector='birch'):
    """Common preprocessing: KPIs, optional lag alignment, adaptive window, and
    the anomaly ranking from the chosen detector ('birch' or 'severity').
    Returns a context dict."""
    ns_dir = os.path.join(sample_dir, cfg.agent_namespace)
    metrics_dir = os.path.join(ns_dir, 'metrics')
    sr, lat, qps = R.load_kpi(metrics_dir)
    if not align:
        lag_map = {}
    elif getattr(cfg, 'lag_mode', 'self') == 'chain':
        lag_map = compute_chain_lags(metrics_dir, ns_dir, cfg)
    else:
        lag_map = compute_service_lags(metrics_dir, cfg)
    aligned_lat = align_metrics_df(lat, lag_map) if lag_map else lat
    start, end, _ = R.adaptive_window(
        sr, cfg, metrics_dir=metrics_dir, latency=aligned_lat, lag_map=lag_map)
    if getattr(cfg, 'full_window', False):            # ablation: no adaptive window
        start, end = sr['timestamp'].iloc[0], sr['timestamp'].iloc[-1]
    lw = getattr(cfg, 'label_window', None)           # explicit label-defined window
    if lw is not None:                                # (RE2: around inject_time)
        import pandas as pd
        start, end = pd.Timestamp(lw[0]), pd.Timestamp(lw[1])
    start_ts = int(time_string_2_timestamp(str(start)))
    end_ts = int(time_string_2_timestamp(str(end)))
    if detector == 'severity':
        ranking = [list(r) for r in R.rank_anomalous_services(
            sr, aligned_lat, qps, start, end, metrics_dir=metrics_dir,
            cfg=cfg, lag_map=lag_map)]
    elif detector == 'overlap':
        from overlap_severity import overlap_rank
        ranking = [list(r) for r in overlap_rank(
            metrics_dir, start, end, cfg, lag_map=lag_map)]
    elif detector == 'abirch':                        # pure ablation base + gate + sparsity
        ranking = _birch_ranking(cfg, sample_dir, metrics_dir, start_ts,
                                 end_ts, lag_map, gate=True, sparsity=True)
    else:
        ranking = _birch_ranking(cfg, sample_dir, metrics_dir, start_ts,
                                 end_ts, lag_map)
    return {'ns_dir': ns_dir, 'metrics_dir': metrics_dir,
            'start': start, 'end': end, 'start_ts': start_ts, 'end_ts': end_ts,
            'lag_map': lag_map, 'ranking': ranking,
            'score_map': {s: v for s, v, _ in ranking}}


def _run_ppr(cfg, sample_dir, align, detector='birch', **_):
    t = {}
    d0 = time.perf_counter()
    ctx = _prep(cfg, sample_dir, align, detector)
    t['detect'] = time.perf_counter() - d0
    d1 = time.perf_counter()
    node_ranking = ppr_rank(
        ctx['ns_dir'], ctx['start_ts'], ctx['end_ts'], ctx['score_map'],
        alpha=getattr(cfg, 'ppr_alpha', 0.85),
        reverse=getattr(cfg, 'ppr_reverse', True),
        max_iter=getattr(cfg, 'ppr_max_iter', 200),
        tol=getattr(cfg, 'ppr_tol', 1e-8))
    t['localize'] = time.perf_counter() - d1
    order = []
    for n in node_ranking:
        svc = R._node_to_service(n)
        if svc.startswith('node-') or svc not in ctx['score_map']:
            continue
        if svc not in order:
            order.append(svc)
    print(f'[{detector}] {len(ctx["ranking"])} anomalous svc; '
          f'[PPR] order top5: {order[:5]}')
    return {'order': order, 'timing': t}


def _run_direct(cfg, sample_dir, align, detector='severity', **_):
    """No localizer: rank services directly by the detector's anomaly score."""
    t = {}
    d0 = time.perf_counter()
    ctx = _prep(cfg, sample_dir, align, detector)
    t['detect'] = time.perf_counter() - d0
    t['localize'] = 0.0
    order = [s for s, _, _ in ctx['ranking']]
    print(f'[{detector}] direct ranking top5: {order[:5]}')
    return {'order': order, 'timing': t}


def _residual_voter_z(cfg, ctx, services):
    """GNN normal-state residual as an extra reliability voter. The per-service
    residual (abnormal-window reconstruction gap) is robustly standardized
    across services into a z >= 0, so it enters reliability_rrf on the same
    'how much does this service stand out' scale as the telemetry voters."""
    import numpy as np
    resid = RES.mrgnn_residual(ctx['ns_dir'], {s: 0.0 for s in services},
                               ctx['start'], ctx['end'], ctx['lag_map'], cfg)
    vals = np.array([max(0.0, resid.get(s, 0.0)) for s in services])
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    scale = max(mad, 0.05 * abs(med), 1e-6)
    return {s: max(0.0, (max(0.0, resid.get(s, 0.0)) - med) / scale)
            for s in services}


def _run_relrrf(cfg, sample_dir, align, detector='severity',
                variant='both', k_rrf=60, include_residual=True,
                signals=None, k=0, mu=0.0, fuse='rrf', **_):
    """Reliability-aware RRF fusion over the RAW per-signal voters (failure /
    latency / cpu / mem / net / qps + GNN residual), so no weighted-sum metric
    term (severity's w_res/w_qps or the localizer's lam) survives. k=0 ranks ALL
    services; k>0 reorders the severity top-k. mu>0 additionally applies the
    separate temporal chain-lag penalty on top of the RRF metric evidence:
        score(s) = RRF_metric_evidence(s) - mu * chain_lag(s)
    leaving mu the single tuned parameter."""
    import reliability_fusion as RF
    t = {}
    d0 = time.perf_counter()
    ctx = _prep(cfg, sample_dir, align, detector)
    t['detect'] = time.perf_counter() - d0
    services = RF._read_services(ctx['metrics_dir'])
    sev_order = [s for s, _, _ in ctx['ranking']]
    candidates = sev_order[:k] if k and k > 0 else None
    d1 = time.perf_counter()
    extra = None
    if include_residual:
        z = _residual_voter_z(cfg, ctx, services)
        extra = {'residual': ({s: z[s] for s in candidates if s in z}
                              if candidates is not None else z)}
    want_scores = mu > 0
    if fuse == 'gate':
        # ablation: NO voter fusion -- metric evidence = the candidate gate's
        # own score (e.g. abirch anomaly evidence); then - mu*lag.
        gate_scores = {s: v for s, v, _ in ctx['ranking']}
        keys = candidates if candidates is not None else list(gate_scores)
        fused = {s: gate_scores.get(s, 0.0) for s in keys}
        if not want_scores:
            fused = sorted(keys, key=lambda s: fused[s], reverse=True)
    else:
        fused = RF.reliability_rrf(
            ctx['ns_dir'], ctx['start'], ctx['end'], ctx['lag_map'],
            variant=variant, k_rrf=k_rrf, extra_z=extra, signals=signals,
            candidates=candidates, return_scores=want_scores, fuse=fuse,
            lat_percentile=getattr(cfg, 'lat_percentile', 'p50'))
    if want_scores:                                   # apply the mu*lag penalty
        me = RES._minmax(fused)                       # metric evidence [0,1]
        lag_pen = RES.chain_lag_penalty(
            ctx['ns_dir'], {s: 0.0 for s in me}, cfg)
        keys = candidates if candidates is not None else list(me)
        fused = sorted(keys, key=lambda s: me[s] - mu * lag_pen.get(s, 0.0),
                       reverse=True)
    order = fused + sev_order[k:] if candidates is not None else fused
    t['localize'] = time.perf_counter() - d1
    return {'order': order, 'timing': t}


# def _run_llm(cfg, sample_dir, align, detector='birch', **_):
#     t = {}
#     d0 = time.perf_counter()
#     ctx = _prep(cfg, sample_dir, align, detector)
#     t['detect'] = time.perf_counter() - d0
#     ranking = ctx['ranking']
#     topk = [s for s, _, _ in ranking[:cfg.llm_topk]]
#     print(f'[{detector}] {len(ranking)} anomalous svc; feeding LLM top-k: {topk}')
#     if not topk:
#         return {'order': [], 'final_root_cause': '', 'timing': t}
#     d1 = time.perf_counter()
#     graphs = load_execution_graphs(ctx['ns_dir'], max_traces=None,
#                                    candidate_services=topk)
#     cap = getattr(cfg, 'exec_max_traces', 40)
#     use_graphs = graphs[:cap] if cap and cap > 0 else graphs
#     log_evidence = build_log_evidence(
#         ctx['ns_dir'], topk, ctx['start'], ctx['end'], cfg,
#         anomaly_type_union=['unknown'])
#     candidate_metadata = {
#         svc: {'selected_by': 'birch',
#               'severity_rank': i, 'severity_score': round(float(sc), 4),
#               'gnn_rank': None, 'gnn_score': None}
#         for i, (svc, sc, _) in enumerate(ranking[:cfg.llm_topk], start=1)}
#     agg = aggregate_topk_slices(
#         topk, use_graphs, cfg, log_evidence=log_evidence,
#         shared_over_services=topk, coarse_ranking=ranking,
#         candidate_metadata=candidate_metadata, gnn_service_ranking=[],
#         trace_analysis_graphs=graphs)
#     result = llm_localize(agg, cfg)
#     t['localize'] = time.perf_counter() - d1
#     order = result.get('reranked_topk') or []
#     return {'order': order, 'final_root_cause': str(result.get('root_cause') or ''),
#             'timing': t}


def _run_residual(cfg, sample_dir, align, detector='severity',
                  kind='fix', k=3, lam=2.0, mu=0.0, merge=None, **_):
    """Severity ranking enhanced by a normal-state topology residual (+ optional
    chain-lag upstream penalty when mu>0), reranked within the severity top-k.
    kind in {fix, gnnA, gnnW, hgnn}. merge in {rrf,avg} collapses severity+
    residual into one parameter-free metric term, leaving mu the only weight."""
    t = {}
    d0 = time.perf_counter()
    ctx = _prep(cfg, sample_dir, align, detector)
    t['detect'] = time.perf_counter() - d0
    sev = {r[0]: float(r[1]) for r in ctx['ranking']}
    d1 = time.perf_counter()
    order = RES.residual_order(
        ctx['ns_dir'], sev, ctx['start'], ctx['end'], ctx['lag_map'],
        kind=kind, k=k, lam=lam, mu=mu, cfg=cfg, merge=merge)
    t['localize'] = time.perf_counter() - d1
    tag = f'{detector}+resid:{kind}:top{k}:mu{mu}' + (f':{merge}' if merge else '')
    print(f'[{tag}] order top5: {order[:5]}')
    return {'order': order, 'timing': t}


def _summarize(tag, rows):
    """Print the SUMMARY block for `rows`: per-sample table, then overall, and
    (auto, only when >1 present) per-group, per-load and per-category metrics."""
    print('\n\n' + '=' * 96 + f'\nSUMMARY [{tag}]\n' + '=' * 96)
    print(f'{"group":<8}{"load":<14}{"fault":<12}{"rank":<6}{"gt":<24}'
          f'{"top1/root_cause":<26}')
    n_hit = 0
    for r in rows:
        n_hit += 1 if r.get('rank') == 1 else 0
        shown = r.get('root_cause') or r.get('top1') or ''
        print(f'{str(r.get("group",""))[:7]:<8}{str(r.get("load",""))[:13]:<14}'
              f'{r["fault"]:<12}{str(r.get("rank", 0)):<6}'
              f'{str(r.get("gt", ""))[:22]:<24}{str(shown)[:24]:<26}')
    tot = len(rows)
    print('-' * 96)
    print(f'Localization accuracy  top1={n_hit}/{tot}')
    print('\nRanking metrics (rank 0/missing = miss)')
    header = (f'{"stage":<26}'
              + ''.join(f'{"ACC@"+str(k):>9}' for k in TOP_K_CUTOFFS)
              + ''.join(f'{"AVG@"+str(n):>9}' for n in TOP_K_CUTOFFS)
              + f'{"MRR":>9}' + f'{"n":>5}')
    print(header)

    def _pm(label, subset):
        if not subset:
            return
        acc, avg = _topk_metrics(subset, 'rank')
        mrr = sum(1.0 / r['rank'] for r in subset if r.get('rank')) / len(subset)
        vals = [acc[k] for k in TOP_K_CUTOFFS] + [avg[n] for n in TOP_K_CUTOFFS]
        print(f'{label:<26}' + ''.join(f'{v:>9.3f}' for v in vals)
              + f'{mrr:>9.3f}' + f'{len(subset):>5}')

    _pm(tag, rows)                                   # overall
    groups = sorted({r.get('group', '') for r in rows})
    if len(groups) > 1:                              # per-group (single/multi)
        for g in groups:
            _pm(f'  «{g}»', [r for r in rows if r.get('group') == g])
    gl = sorted({(r.get('group', ''), r.get('load', '')) for r in rows})
    if len(gl) > 1:                                  # per-load
        for g, ld in gl:
            _pm(f'    <{g}/{ld}>', [r for r in rows
                                    if r.get('group') == g and r.get('load') == ld])
    for cat in CATEGORIES:                            # per root-cause category
        _pm(f'  [{cat}]', [r for r in rows if r.get('category') == cat])
    succ = sum(r.get('status') == 'success' for r in rows)
    print(f'\nBatch  completed={tot} successful={succ} failed={tot - succ}')
    for stage in ('detect', 'localize', 'total'):
        v = [r['timing_seconds'][stage] for r in rows
             if stage in r.get('timing_seconds', {})]
        if v:
            print(f'  mean {stage}: {sum(v)/len(v):.3f}s ({len(v)})')


def _run_batch(method, align, tag, detector='birch', **opts):
    cfg = Config()
    if 'lag_mode' in opts:                         # allow per-run lag mode
        cfg.lag_mode = opts.pop('lag_mode')
    groups = opts.pop('groups', None)              # None -> all groups present
    limit = opts.pop('limit', None)                # None -> all samples (smoke cap)
    samples = []
    for group, load, adir in _discover(groups):
        labs = parse_labels(adir)
        for n in sorted(labs):
            sdir = os.path.join(adir, n)
            if os.path.isdir(sdir):
                samples.append((n, sdir, group, load, labs[n]))
    if limit:
        samples = samples[:limit]
    print(f'Baseline [{tag}]  method={method} align={align} detector={detector}')
    print(f'Found {len(samples)} labeled samples across '
          f'{sorted({(g, l) for _, _, g, l, _ in samples})}.')
    rows = []
    for name, sdir, group, load, lab in samples:
        gt = lab.get('service', '')
        fault = lab.get('fault_type', name)
        print('\n' + '#' * 70
              + f'\n##### [{group}/{load}] {name} (gt={gt}, fault={fault})\n'
              + '#' * 70)
        started = time.perf_counter()
        res, err = None, None
        try:
            fn = {'ppr': _run_ppr, 'direct': _run_direct,
                  'residual': _run_residual,
                  'relrrf': _run_relrrf}.get(method)
            res = fn(cfg, sdir, align, detector, **opts)
        except Exception as exc:
            err = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        elapsed = time.perf_counter() - started
        timing = dict((res or {}).get('timing') or {})
        timing['total'] = elapsed
        base = {'fault': fault, 'gt': gt, 'category': _category(gt),
                'group': group, 'load': load, 'timing_seconds': timing}
        if not res:
            rows.append({**base, 'status': 'failed', 'rank': 0, 'error': err})
            continue
        rank = R._service_rank(res['order'], gt)
        rc = res.get('final_root_cause', '')
        rows.append({**base, 'status': 'success', 'rank': rank,
                     'top1': res['order'][0] if res['order'] else None,
                     'root_cause': rc, 'hit': rank == 1 or _rc_names_gt(rc, gt)})
        print(f'[eval] gt rank = {rank}')

    _summarize(tag, rows)                            # overall -> abnormal/ log
    return rows


def _write_summary_log(path, header, tag, rows):
    with open(path, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            print(header)
            _summarize(tag, rows)


# friendly dataset label for the output/ tree
_DATASET_LABEL = {'MDOC': 'MDOC', 'MARBLEBench': 'MAR'}


def main(method, align, tag, detector='birch', out_method=None, **opts):
    started_at = datetime.now().astimezone()
    t0 = time.perf_counter()
    ts = started_at.strftime('%Y-%m-%d-%H:%M:%S')
    # output base: output/<dataset>/<method>/ when out_method is given (the
    # experiments/ drivers set it); otherwise the legacy data/<ds>/abnormal/.
    if out_method:
        ds = _DATASET_LABEL.get(os.environ.get('AGENT_DATASET', 'MDOC'), 'MDOC')
        out_base = os.path.join('output', ds, out_method)
        os.makedirs(out_base, exist_ok=True)
    else:
        out_base = ABN_BASE
    overall_log = os.path.join(out_base, f'batch-{tag}_{ts}.log')   # everything
    rows = None
    with open(overall_log, 'w', encoding='utf-8') as lf:
        with redirect_stdout(_Tee(sys.stdout, lf)), \
                redirect_stderr(_Tee(sys.stderr, lf)):
            print(f'Batch start: {started_at.isoformat(timespec="seconds")}')
            print(f'Overall log: {overall_log}')
            try:
                rows = _run_batch(method, align, tag, detector, **opts)
            finally:
                print(f'\nBatch wall-clock: {time.perf_counter() - t0:.1f}s')
    if not rows:
        return
    # per-group logs (<base>/<group>/) and per-load logs (<base>/<group>/<load>/)
    for g in sorted({r['group'] for r in rows}):
        gsub = [r for r in rows if r['group'] == g]
        os.makedirs(os.path.join(out_base, g), exist_ok=True)
        _write_summary_log(os.path.join(out_base, g, f'batch-{tag}_{ts}.log'),
                           f'Per-group [{g}] log for [{tag}]  ({ts})', tag, gsub)
        print(f'  per-group log: {os.path.join(out_base, g)}')
        for ld in sorted({r['load'] for r in gsub}):
            lsub = [r for r in gsub if r['load'] == ld]
            os.makedirs(os.path.join(out_base, g, ld), exist_ok=True)
            _write_summary_log(
                os.path.join(out_base, g, ld, f'batch-{tag}_{ts}.log'),
                f'Per-load [{g}/{ld}] log for [{tag}]  ({ts})', tag, lsub)
