#!/usr/bin/env python3
"""Run PC+PageRank or CausalRCA (DAG-GNN+PageRank) on RCAEval RE2."""

import argparse
import os
import sys
import time

import networkx as nx
import numpy as np
from sknetwork.ranking import PageRank

from rcaeval_common import iter_cases, load_case, service_of_column

# repo root (4 levels up) on sys.path so the shared results_log is importable;
# appended (not inserted) so the vendored config.py/graph.py still win.
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))))
import results_log  # noqa: E402

_SUITE2DS = {'ob': 'OB', 'ss': 'SS', 'tt': 'TT'}


def pc_adjacency(data):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    cg = pc(data.to_numpy(), 0.05, fisherz, False, 0, -1,
            show_progress=False)
    raw = cg.G.graph
    graph = nx.DiGraph()
    for i in range(len(raw)):
        for j in range(len(raw)):
            if raw[i, j] == -1:
                graph.add_edge(i, j)
            if raw[i, j] == 1:
                graph.add_edge(j, i)
    nodes = sorted(graph.nodes())
    adjacency = nx.to_numpy_array(graph, nodelist=nodes) if nodes else np.zeros((0, 0))
    return adjacency, nodes


def daggnn_adjacency(data):
    # The legacy DAG-GNN module imports another script with a top-level
    # argparse parser. Hide this runner's flags while loading it.
    argv = sys.argv
    try:
        sys.argv = [argv[0]]
        from train_all_services_MicroARCL_agent_daggnn import run_daggnn
    finally:
        sys.argv = argv

    adjacency = run_daggnn(data)
    return adjacency, list(range(data.shape[1]))


def rank_case(adjacency, nodes, columns, root_cause):
    """Match the agent scripts: PageRank on |A.T| and first matching metric."""
    if not np.any(adjacency):
        return 0, []
    scores = PageRank().fit_predict(np.abs(adjacency.T))
    ranking = sorted(((columns[nodes[i]], float(s)) for i, s in enumerate(scores)),
                     key=lambda pair: pair[1], reverse=True)
    for pos, (metric, _) in enumerate(ranking, 1):
        if service_of_column(metric) == root_cause:
            return pos, ranking[:pos]
    return 0, ranking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('pc', 'daggnn'), required=True)
    parser.add_argument('--system', choices=('ob', 'ss', 'tt'), action='append',
                        help='Repeat to select systems; default is all three')
    parser.add_argument('--limit', type=int, help='Process at most this many cases')
    parser.add_argument('--shard', help='Run shard i/n (0-indexed), e.g. 0/2; splits '
                                        'cases by position so parallel runs stay disjoint')
    args = parser.parse_args()

    discover = pc_adjacency if args.method == 'pc' else daggnn_adjacency
    cases = [c for c in iter_cases() if args.system is None or c[1] in args.system]
    if args.limit is not None:
        cases = cases[:args.limit]
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split('/'))
        cases = [c for idx, c in enumerate(cases) if idx % shard_n == shard_i]
    results = []
    for number, (case, system, root_cause, fault) in enumerate(cases, 1):
        tag = system + '/' + fault + '/' + case
        case_start = time.time()
        data, columns = load_case(case)
        if data is None:
            raise ValueError('No usable metrics for ' + tag)
        adjacency, nodes = discover(data)
        rank, ranking = rank_case(adjacency, nodes, columns, root_cause)
        seconds = time.time() - case_start
        results.append({'group': system, 'load': fault, 'case': case,
                        'root_cause': root_cause, 'rank': rank, 'seconds': seconds})
        print('[{}/{}] {} topK={} {:.3f}s'.format(number, len(cases), tag, rank, seconds), flush=True)

    # uniform output/<dataset>/<method>/ logs (one per RE2 system)
    method_name = os.environ.get(
        'CAUSALRCA_METHOD',
        'CausalRCA' if args.method == 'daggnn' else 'CloudRanger')
    for system in sorted({r['group'] for r in results}):
        ds = _SUITE2DS.get(system, system)
        rows = [{'case': r['case'], 'gt': r['root_cause'], 'fault': r['load'],
                 'category': r['load'], 'rank': r['rank'], 'top1': None,
                 'status': 'success', 'timing': {'total': r['seconds']}}
                for r in results if r['group'] == system]
        results_log.emit_from_rows(ds, method_name, method_name, rows,
                                   kind='re2',
                                   header_lines=[f'{method_name} on {ds} (RE2)'])


if __name__ == '__main__':
    main()
