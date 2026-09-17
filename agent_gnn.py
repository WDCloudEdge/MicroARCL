"""
GNN coarse ranker for the single-namespace agent-network dataset.

Reuses the original MicroCERCL machinery (graph.py heterogeneous topology
stack + model.py unsupervised GNN + bp objective) but builds the topology
stack OFFLINE from graph.csv (no Prometheus / Kubernetes client), so it can
run directly on a collected sample directory.

Exposes gnn_rank(config, base_dir, start_ts, end_ts) -> {node_name: score},
the GNN node-level root-cause ranking, to be compared side-by-side with the
severity-based coarse ranking in run_agent_rca.py.
"""
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


def build_graphs(ns_dir: str) -> Dict[str, nx.DiGraph]:
    """Build per-timestamp topology graphs.

    svc->svc call edges come from the execution-graph routes (call_chains.json,
    async-aware) and are static over the window; pod->physical-node placement
    comes from graph.csv per timestamp (istio/prometheus). Physical node name
    is the node's center (multi-center = multiple machines); svc center is
    'svc-none'. Falls back to graph.csv svc edges if no exec routes are found.
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
                g.add_edge(src, dst)
                for n in (src, dst):
                    g.nodes[n]['type'] = NodeType.SVC.value
                    g.nodes[n]['center'] = 'svc-none'
                    svc_exist.add(n)
        else:
            for src, dst in svc_edges:      # static async routing
                g.add_edge(src, dst)
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
             severity_map: Dict[str, float] = None) -> Dict[str, float]:
    """Run the heterogeneous GNN root-cause ranker over [start_ts, end_ts].
    Returns {node_name: score} sorted desc, or {} if the topology is empty."""
    ns = namespace
    ns_dir = os.path.join(base_dir, ns)
    metrics_dir = os.path.join(ns_dir, 'metrics')

    # B.1: estimate per-(service,kind) QPS->resource lags once; feed aligned
    # metrics to both Birch and the GNN feature construction.
    lag_map = {}
    if getattr(config, 'lag_enable', False):
        from lag_align import compute_service_lags
        lag_map = compute_service_lags(metrics_dir, config)
        if lag_map:
            nz = {k: v for k, v in lag_map.items() if v > 0}
            print(f'[agent_gnn] estimated lags (non-zero): {nz}')

    graphs_ts = build_graphs(ns_dir)

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
        try:
            _, sorted_dict_node = train(
                config, tw, 'agent', center_map, anomaly_index, hetero_graphs,
                base_dir, is_train, rnn=config.rnn_type, return_ranking=True,
                node_severity=node_severity,
                severity_param=getattr(config, 'severity_prior_param', 0.3))
        except Exception as e:
            print(f'[agent_gnn] train failed for {tw}: {e}')
            continue
        for node, score in sorted_dict_node.items():
            ranking[node] = ranking.get(node, 0.0) + float(score)

    return dict(sorted(ranking.items(), key=lambda kv: kv[1], reverse=True))
