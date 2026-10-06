"""GNN coarse ranker for the single-namespace agent-network dataset."""
import os
import json
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import pandas as pd

from graph import (combine_ns_graphs, graph_weight_ns, graph_weight, graph_index,
                   get_hg, graph_prune, combine_timestamps_graph, GraphIndex)
from anomaly_detection import get_anomaly_by_df, get_timestamp_index
from model import train
from Config import TrainType
from util.utils import time_string_2_timestamp, timestamp_2_time_string, df_time_limit
from graph import NodeType
# reuse the explicit group->service mapping from Module C
from execution_graph import group_to_service
from lag_align import _service_of_pod


def _read_services(metrics_dir: str) -> List[str]:
    """k8s service names for this namespace (from success_rate.csv columns)."""
    df = pd.read_csv(os.path.join(metrics_dir, 'success_rate.csv'), nrows=1)
    return [c for c in df.columns if c != 'timestamp']


def build_svc_call_edges(ns_dir: str, services: List[str]) -> Set[Tuple[str, str]]:
    """Extract svc->svc call edges from the execution-graph routes
    (call_chains.json). The agent network is fully asynchronous, so the
    istio-derived graph.csv misses these calls; the true routing lives in the
    execution graphs.

    The flat `edges` field is incomplete (it records group->own-agent hops but
    drops cross-group transitions along a deep pipeline). We instead walk each
    trace's level-ordered `path`, map every vertex to its service, collapse
    consecutive same-service hops (group -> its internal agent), and keep the
    transitions between distinct services (cross-group calls). E.g. path
    [planner, ExcelGroup, ExcelGroup/read, ImageGenAgentGroup, .../gen,
    PdfGenAgentGroup, .../gen] -> planner->excel, excel->image, image->pdf."""
    path = os.path.join(ns_dir, 'graph', 'call_chains.json')
    if not os.path.exists(path):
        return set()
    with open(path, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    cache: Dict[str, Optional[str]] = {}

    def to_svc(v):
        if v not in cache:
            cache[v] = group_to_service(v, services)
        return cache[v]

    edges: Set[Tuple[str, str]] = set()
    for tr in chains:
        seq = []                                    # collapsed service chain
        for v in (tr.get('path', []) or []):
            s = to_svc(v)
            if s is None:
                continue
            if not seq or seq[-1] != s:             # skip intra-group (same svc) hops
                seq.append(s)
        for a, b in zip(seq, seq[1:]):              # cross-group svc->svc edges
            if a != b:
                edges.add((a, b))
    summary = 'agent-network-summarizer'
    if summary not in services:     # not the agent dataset (e.g. RE2) -> no sink
        return edges
    summary_set = set()
    for s, d in list(edges):   # snapshot: we mutate `edges` inside the loop
        summary_s = s + '-' + summary
        summary_d = d + '-' + summary
        # if 'agent-network-planner' != s and summary_s not in summary_set:
        #     edges.add((s, summary))
            # summary_set.add(summary_s)
        if 'agent-network-planner' != d and summary_d not in summary_set:
            edges.add((d, summary))
            summary_set.add(summary_d)
    return edges


def build_graphs(ns_dir: str, reverse: bool = False) -> Dict[str, nx.DiGraph]:
    """Build per-timestamp topology graphs.

    svc->svc call edges come from the execution-graph routes (call_chains.json,
    async-aware) and are static over the window; pod->physical-node placement
    comes from graph.csv per timestamp (istio/prometheus). Physical node name
    is the node's center (multi-center = multiple machines); svc center is
    'svc-none'. Falls back to graph.csv svc edges if no exec routes are found.

    reverse=True transposes the svc->svc call edges (caller->callee becomes
    callee->caller) so the GNN propagates anomaly signal from downstream victims
    back toward the upstream root cause, instead of accumulating it on the
    call-graph sink (summarizer). pod<->node and svc<->pod links stay
    bidirectional and are unaffected.
    """
    metrics_dir = os.path.join(ns_dir, 'metrics')
    graph_csv = os.path.join(metrics_dir, 'graph.csv')
    services = _read_services(metrics_dir)
    svc_edges = build_svc_call_edges(ns_dir, services)

    df = pd.read_csv(graph_csv)
    pod_ts: Dict[str, List] = {}
    svc_ts_fallback: Dict[str, List] = {}
    all_ts: Set[str] = set()
    for _, row in df.iterrows():
        ts = str(time_string_2_timestamp(str(row['timestamp'])))
        all_ts.add(ts)
        src, dst = str(row['source']), str(row['destination'])
        if dst.startswith('node-'):
            pod_ts.setdefault(ts, []).append((src, dst))
        else:
            svc_ts_fallback.setdefault(ts, []).append((src, dst))

    use_fallback = len(svc_edges) == 0
    if use_fallback:
        print('[agent_gnn] no exec-graph svc routes found; falling back to graph.csv svc edges')
    else:
        print(f'[agent_gnn] svc->svc edges from execution-graph routes: {len(svc_edges)}')

    graphs: Dict[str, nx.DiGraph] = {}
    for ts in sorted(all_ts, key=lambda x: int(x)):
        g = nx.DiGraph()
        svc_exist = set()
        # ---- svc->svc edges ----
        if use_fallback:
            for src, dst in svc_ts_fallback.get(ts, []):
                a, b = (dst, src) if reverse else (src, dst)
                g.add_edge(a, b)
                for n in (src, dst):
                    g.nodes[n]['type'] = NodeType.SVC.value
                    g.nodes[n]['center'] = 'svc-none'
                    svc_exist.add(n)
        else:
            for src, dst in svc_edges:      # static async routing
                a, b = (dst, src) if reverse else (src, dst)
                g.add_edge(a, b)
                for n in (src, dst):
                    g.nodes[n]['type'] = NodeType.SVC.value
                    g.nodes[n]['center'] = 'svc-none'
                    svc_exist.add(n)
        # ---- pod->node placement (from graph.csv) + pod<->svc linking ----
        svc_pods_map: Dict[str, List[str]] = {}
        for src, dst in pod_ts.get(ts, []):
            for svc in svc_exist:
                if svc in src:
                    svc_pods_map.setdefault(svc, []).append(src)
            center = dst  # physical node name as center
            g.add_node(src)
            g.nodes[src]['type'] = NodeType.POD.value
            g.nodes[src]['center'] = center
            g.add_edge(src, dst)
            g.add_edge(dst, src)
            g.nodes[dst]['type'] = NodeType.NODE.value
            g.nodes[dst]['center'] = center
        for svc, pods in svc_pods_map.items():
            for p in set(pods):
                g.add_edge(svc, p)
                g.add_edge(p, svc)
        graphs[ts] = g
    return graphs


def _topology_change_times(graphs: Dict[str, nx.DiGraph], start: int, end: int) -> List[str]:
    change = []
    count = 0
    for ts in sorted(graphs.keys(), key=lambda x: int(x)):
        if start <= int(ts) <= end:
            c = graphs[ts].number_of_nodes()
            if c != count:
                change.append(ts)
                count = c
    return change


def _build_node_severity(graphs_combine: Dict[str, nx.DiGraph],
                         severity_map: Dict[str, float]) -> Dict[str, float]:
    """Map each graph node -> severity prior. svc/pod inherit their service's
    severity; physical node = mean severity of adjacent pods' services (B.3)."""
    node_sev: Dict[str, float] = {}
    for g in graphs_combine.values():
        for n, d in g.nodes(data=True):
            t = d.get('type')
            if t == NodeType.SVC.value:
                node_sev[n] = float(severity_map.get(n, 0.0))
            elif t == NodeType.POD.value:
                node_sev[n] = float(severity_map.get(_service_of_pod(n), 0.0))
    for g in graphs_combine.values():
        for n, d in g.nodes(data=True):
            if d.get('type') == NodeType.NODE.value:
                neigh = set(g.predecessors(n)) | set(g.successors(n))
                sevs = [severity_map.get(_service_of_pod(p), 0.0) for p in neigh
                        if g.nodes[p].get('type') == NodeType.POD.value]
                node_sev[n] = float(sum(sevs) / len(sevs)) if sevs else 0.0
    return node_sev


def gnn_rank(config, base_dir: str, start_ts: int, end_ts: int,
             namespace: str = 'agent-network',
             label: str = 'agent', is_train: TrainType = TrainType.TRAIN,
             severity_map: Dict[str, float] = None,
             evaluation_root_cause: Optional[str] = None,
             lag_map: Optional[Dict[Tuple[str, str], int]] = None,
             return_components: bool = False
             ) -> Dict[str, float]:
    """Run the heterogeneous GNN root-cause ranker over [start_ts, end_ts].
    Returns {node_name: score} sorted desc, or {} if the topology is empty."""
    ns = namespace
    ns_dir = os.path.join(base_dir, ns)
    metrics_dir = os.path.join(ns_dir, 'metrics')
    if evaluation_root_cause is None:
        # Benchmark-only evaluation metadata. This value is passed solely to
        # top_k_node after inference and never affects model fitting/ranking.
        sample_name = os.path.basename(os.path.normpath(base_dir))
        services = sorted(_read_services(metrics_dir), key=len, reverse=True)
        evaluation_root_cause = next(
            (service for service in services
             if sample_name == service or sample_name.startswith(service + '_')),
            '')
    print('[agent_gnn] evaluation root cause: '
          f'{evaluation_root_cause or "unavailable"}')

    # B.1: reuse the pipeline-wide lag map. Direct callers that do not provide
    # one retain the old estimate-on-entry fallback.
    if lag_map is None and getattr(config, 'lag_enable', False):
        from lag_align import compute_service_lags
        lag_map = compute_service_lags(metrics_dir, config)
    lag_map = lag_map or {}
    if lag_map:
        nz = {k: v for k, v in lag_map.items() if v > 0}
        print(f'[agent_gnn] using shared lags (non-zero): {nz}')

    graphs_ts = build_graphs(
        ns_dir, reverse=getattr(config, 'gnn_graph_reverse', True))

    # ---- single time_pair covering the whole adaptive window ----
    # (main.py chunks by config.duration; here the window is already the
    # adaptive anomaly window, so one chunk keeps topology intervals nested
    # inside it and avoids skipping node-weighting.)
    time_pair_list = [(int(start_ts), int(end_ts))]
    time_pair_index = {}
    latency_df = pd.read_csv(os.path.join(metrics_dir, 'latency.csv'))
    df = df_time_limit(latency_df, int(start_ts), int(end_ts))
    df_time_index, _ = get_timestamp_index(df)
    time_pair_index[(int(start_ts), int(end_ts))] = df_time_index

    # ---- topology change windows ----
    change_times = _topology_change_times(graphs_ts, start_ts, end_ts)
    change_times = sorted(set(int(t) for t in change_times) | {int(end_ts)})
    interval_keys = []
    graphs_change_time: Dict[str, Dict[str, nx.DiGraph]] = {}
    for i in range(len(change_times) - 1):
        b = change_times[i]
        en = change_times[i + 1] - 1
        key = f'{b}-{en}'
        interval_keys.append(key)
        graphs_change_time[key] = combine_timestamps_graph(graphs_ts, ns, b, en)

    if not interval_keys:
        return {}

    # ---- per time_pair: anomaly detection + build hetero graphs + train ----
    ranking: Dict[str, float] = {}
    # raw components (accumulated across intervals) for offline fusion tuning
    comp_gnn: Dict[str, float] = {}
    comp_sev: Dict[str, float] = {}
    comp_deg: Dict[str, int] = {}
    for time_pair in time_pair_list:
        tw = f'{time_pair[0]}-{time_pair[1]}'
        anomalies_ns, anomaly_time_series = get_anomaly_by_df(
            config, base_dir, metrics_dir, label, time_pair[0], time_pair[1],
            lag_map=lag_map)
        anomalies = {tw: list(set(anomalies_ns))}
        anomaly_index = {a: i for i, a in enumerate(anomalies[tw])}
        t_index_time_window = time_pair_index[(time_pair[0], time_pair[1])]

        # weight ns subgraphs
        for key in graphs_change_time:
            b, en = key.split('-')
            if int(b) < time_pair[0] or int(en) > time_pair[1]:
                continue
            for gkey in graphs_change_time[key]:
                graph = graphs_change_time[key][gkey]
                graph_weight_ns(b, en, graph, base_dir, ns, lag_map=lag_map)

        # combine ns graphs (single ns here) per interval
        graphs_combine: Dict[str, nx.DiGraph] = {}
        for key in interval_keys:
            graphs_combine.update(combine_ns_graphs(graphs_change_time[key]))

        graphs_anomaly_ts_index = {}
        graphs_anomaly_ts_index_map = {}
        graphs_index_time_map = {}
        for tcombine in graphs_combine:
            graph = graphs_combine[tcombine]
            b, en = tcombine.split('-')

            def get_t(begin_t):
                idx = len(t_index_time_window.keys()) - 1
                for i, t in enumerate(sorted(t_index_time_window.keys())):
                    if int(begin_t) <= time_string_2_timestamp(t):
                        return i
                return idx

            graph_weight(b, en, graph, base_dir)
            graph_prune(graph, base_dir, f'{b}-{en}')
            gi_time_map = {}
            for t in t_index_time_window:
                if int(b) <= time_string_2_timestamp(t) <= int(en):
                    gi_time_map[t_index_time_window[t] - get_t(b)] = t
            graphs_index_time_map[tcombine] = gi_time_map

            a_t_index = []
            anomaly_ts_index = {}
            for anomaly in anomaly_time_series:
                a_series = [time_string_2_timestamp(a) for a in anomaly_time_series[anomaly]]
                if not a_series or max(a_series) < int(b) or min(a_series) > int(en):
                    continue
                a_idx = []
                for t in a_series:
                    if int(b) <= t <= int(en):
                        ii = t_index_time_window[timestamp_2_time_string(t)] - get_t(b)
                        a_t_index.append(ii)
                        a_idx.append(ii)
                anomaly_ts_index[anomaly] = a_idx
            graphs_anomaly_ts_index[tcombine] = list(set(a_t_index))
            graphs_anomaly_ts_index_map[tcombine] = anomaly_ts_index

        # drop empty intervals (no edges after weighting/pruning); dgl cannot
        # build an empty heterograph
        graphs_combine = {k: g for k, g in graphs_combine.items()
                          if g.number_of_edges() > 0}
        if not graphs_combine:
            print(f'[agent_gnn] no non-empty topology interval for {tw}, skipping')
            continue

        graphs_combine_index: Dict[str, GraphIndex] = {
            k: graph_index(graphs_combine[k]) for k in graphs_combine}

        hetero_graphs, center_map = get_hg(
            graphs_combine, graphs_combine_index, anomalies,
            graphs_anomaly_ts_index, graphs_anomaly_ts_index_map,
            {k: graphs_index_time_map[k] for k in graphs_combine})

        # B.3: severity prior per node (svc/pod inherit service severity;
        # physical node = mean of adjacent pods' service severity)
        node_severity = _build_node_severity(graphs_combine, severity_map or {})
        # total (in+out) degree per node for optional hub/sink suppression
        deg: Dict[str, int] = {}
        for g in graphs_combine.values():
            for n in g.nodes():
                deg[n] = deg.get(n, 0) + g.in_degree(n) + g.out_degree(n)
        try:
            _, sorted_dict_node = train(
                config, tw, evaluation_root_cause, center_map, anomaly_index,
                hetero_graphs,
                base_dir, is_train, rnn=config.rnn_type, return_ranking=True,
                node_severity=node_severity,
                severity_param=getattr(config, 'severity_prior_param', 0.3))
        except Exception as e:
            print(f'[agent_gnn] train failed for {tw}: {e}')
            continue
        # Anchor the GNN logits to the severity prior and (optionally) penalize
        # high-degree hubs/sinks. The bp objective only uses severity as a
        # training target, so the converged logits still drift toward
        # structurally central nodes (summarizer/word-gen); this readout fusion
        # pulls the final ranking back toward the metric-level severity prior.
        #   final = (1-w)*gnn + w*severity   (w = gnn_severity_anchor)
        #   final /= (1 + lam*degree)        (lam = gnn_hub_penalty, 0=off)
        w = float(getattr(config, 'gnn_severity_anchor', 0.5))
        lam = float(getattr(config, 'gnn_hub_penalty', 0.0))
        for node, score in sorted_dict_node.items():
            sev = float(node_severity.get(node, 0.0))
            fused = (1.0 - w) * float(score) + w * sev
            if lam > 0.0:
                fused /= (1.0 + lam * deg.get(node, 0))
            ranking[node] = ranking.get(node, 0.0) + fused
            comp_gnn[node] = comp_gnn.get(node, 0.0) + float(score)
            comp_sev[node] = max(comp_sev.get(node, 0.0), sev)
            comp_deg[node] = max(comp_deg.get(node, 0), deg.get(node, 0))

    result = dict(sorted(ranking.items(), key=lambda kv: kv[1], reverse=True))
    if return_components:
        return result, comp_gnn, comp_sev, comp_deg
    return result
