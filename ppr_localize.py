"""Personalized PageRank baseline for agent-service root-cause localization."""
from typing import Dict, Optional, Tuple

import networkx as nx

# reuse identical graph construction + severity->node prior from the GNN path
from agent_gnn import build_graphs, _build_node_severity


def window_graph(ns_dir: str, start_ts: int, end_ts: int) -> nx.DiGraph:
    """Union every per-timestamp topology graph inside [start_ts, end_ts] into a
    single static DiGraph. Nodes keep the same 'type'/'center' attributes that
    build_graphs assigns (svc / pod / physical node)."""
    graphs_ts = build_graphs(ns_dir)
    g = nx.DiGraph()
    for ts, gt in graphs_ts.items():
        if start_ts <= int(ts) <= end_ts:
            for n, d in gt.nodes(data=True):
                if n not in g:
                    g.add_node(n, **d)
            g.add_edges_from(gt.edges())
    return g


def ppr_rank(ns_dir: str, start_ts: int, end_ts: int,
             severity_map: Dict[str, float],
             alpha: float = 0.85, reverse: bool = False,
             max_iter: int = 200, tol: float = 1e-8
             ) -> Dict[str, float]:
    """Return {node_name: score} sorted desc via Personalized PageRank.

    - graph construction identical to the GNN ranker (build_graphs);
    - personalization vector = per-node severity prior (_build_node_severity):
      svc/pod inherit their service's severity, physical node = mean of adjacent
      pods' service severity;
    - random walk with restart iterated to convergence (tol); the top-scored
      nodes are the coarse root-cause ranking.
    reverse=True transposes the call graph so probability flows from the failing
    (symptom) services back toward upstream causes.
    """
    g = window_graph(ns_dir, start_ts, end_ts)
    if g.number_of_nodes() == 0:
        return {}

    # severity -> per-node prior (same mapping the GNN B.3 target uses)
    node_sev = _build_node_severity({'window': g}, severity_map or {})
    # personalization must cover every node and sum to > 0
    pers = {n: max(0.0, float(node_sev.get(n, 0.0))) for n in g.nodes()}
    if sum(pers.values()) <= 0.0:
        pers = {n: 1.0 for n in g.nodes()}   # no usable prior -> uniform restart

    walk_graph = g.reverse(copy=True) if reverse else g
    try:
        scores = nx.pagerank(walk_graph, alpha=alpha, personalization=pers,
                             max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        # relax convergence rather than fail the whole sample
        scores = nx.pagerank(walk_graph, alpha=alpha, personalization=pers,
                             max_iter=max_iter * 10, tol=tol * 100)
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))
