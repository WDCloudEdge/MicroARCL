"""Topology-graph and service-name helpers for the agent-network dataset.
Shared by the PPR ranker (ppr_localize) and the lag/residual modules."""
import os
import json
from enum import Enum
from typing import Dict, List, Set, Tuple, Optional
import networkx as nx
import pandas as pd

from util.utils import time_string_2_timestamp
# reuse the explicit group->service mapping from the execution-graph module
from execution_graph import group_to_service
from lag_align import _service_of_pod


class NodeType(Enum):
    SVC = 'svc'
    POD = 'instance'
    NODE = 'node'
    NIC = 'nic'


def _read_services(metrics_dir: str) -> List[str]:
    """k8s service names for this namespace (from success_rate.csv columns)."""
    df = pd.read_csv(os.path.join(metrics_dir, 'success_rate.csv'), nrows=1)
    return [c for c in df.columns if c != 'timestamp']


def build_svc_call_edges(ns_dir: str, services: List[str]) -> Set[Tuple[str, str]]:
    """svc->svc call edges from the execution-graph routes (call_chains.json).
    The agent network is async, so istio's graph.csv misses these calls. Walk
    each trace's level-ordered path, collapse consecutive same-service hops, and
    keep the transitions between distinct services."""
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
    for s, d in list(edges):   # snapshot: edges is mutated in the loop
        summary_d = d + '-' + summary
        if 'agent-network-planner' != d and summary_d not in summary_set:
            edges.add((d, summary))
            summary_set.add(summary_d)
    return edges


def build_graphs(ns_dir: str, reverse: bool = False) -> Dict[str, nx.DiGraph]:
    """Per-timestamp topology graphs: static svc->svc call edges (from the async
    execution-graph routes, or graph.csv as fallback) plus per-timestamp
    pod->node placement from graph.csv. reverse=True flips the call edges so
    anomaly signal flows from victims back to the upstream cause instead of
    piling up on the sink (summarizer)."""
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
        print('[graph_utils] no exec-graph svc routes found; falling back to graph.csv svc edges')
    else:
        print(f'[graph_utils] svc->svc edges from execution-graph routes: {len(svc_edges)}')

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


def _build_node_severity(graphs_combine: Dict[str, nx.DiGraph],
                         severity_map: Dict[str, float]) -> Dict[str, float]:
    """Map each graph node -> severity prior. svc/pod inherit their service's
    severity; physical node = mean severity of adjacent pods' services."""
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
