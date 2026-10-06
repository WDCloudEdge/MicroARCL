#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CausalRCA (PC + PageRank) on the RCAEval RE2 datasets (re2ob / re2ss / re2tt).

Causal discovery: constraint-based PC with the Fisher-Z test (the correct and
tractable choice for continuous metrics, as in RCAEval's own pc_default), then
PageRank on |adj.T| to rank root-cause metrics. A case counts as a hit when the
ground-truth service (from the folder name) owns the first matching column.

Data loading / windowing / variable selection live in rcaeval_common.py.
Results are written per case and summarised for TOTAL / per system / per fault.
"""

import os
import time
import numpy as np
import networkx as nx

from rcaeval_common import (iter_cases, load_case, service_of_column,
                            write_summary, DATA_DIR, LENGTH, KEEP_SUFFIXES)

from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.cit import fisherz
from sknetwork.ranking import PageRank
import contextlib
import io


def run_pc(data):
    """Return (adj, node_index_list) from PC; adj is the dense directed graph
    over the nodes that have edges (same construction as the MicroARCL PC script)."""
    with contextlib.redirect_stdout(io.StringIO()):
        cg = pc(data.to_numpy(), 0.05, fisherz, False, 0, -1)
    raw = cg.G.graph

    G = nx.DiGraph()
    for i in range(len(raw)):
        for j in range(len(raw)):
            if raw[i, j] == -1:
                G.add_edge(i, j)
            if raw[i, j] == 1:
                G.add_edge(j, i)
    nodes = sorted(G.nodes())
    adj = np.asarray(nx.to_numpy_array(G, nodelist=nodes)) if G.number_of_nodes() else np.zeros((0, 0))
    return adj, nodes


if __name__ == '__main__':
    out_root = 'RCAEval_out/pc'
    os.makedirs(out_root, exist_ok=True)

    results = []
    systems = []
    begin_tt = time.time()

    for case, system, gt_service, fault in iter_cases():
        tag = system + '/' + fault + '/' + case
        print(tag, flush=True)
        if system not in systems:
            systems.append(system)

        t0 = time.time()
        data, names = load_case(case)
        if data is None:
            print(tag + ' has no usable data, skip')
            continue

        adj, nodes = run_pc(data)

        out_dir = os.path.join(out_root, system, fault)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, case + '.log')

        rank = 0  # 0 = miss
        if not np.any(adj):
            with open(log_path, 'w') as f:
                print('root cause: ' + gt_service, file=f)
                print('topK: 0', file=f)
        else:
            pagerank = PageRank()
            scores = pagerank.fit_predict(np.abs(adj.T))
            score_dict = {names[nodes[i]]: s for i, s in enumerate(scores)}
            sorted_scores = sorted(score_dict.items(), key=lambda kv: kv[1], reverse=True)
            count = 0
            with open(log_path, 'w') as f:
                print('root cause: ' + gt_service, file=f)
                for col, _s in sorted_scores:
                    count += 1
                    print((col, _s), file=f)
                    if service_of_column(col) == gt_service:
                        rank = count
                        break
                print('topK: ' + str(rank), file=f)

        results.append({'group': system, 'load': fault, 'case': case,
                        'root_cause': gt_service, 'rank': rank,
                        'seconds': time.time() - t0})

    time_cost = time.time() - begin_tt
    print('time cost:' + str(time_cost))

    dataset_desc = 'RCAEval RE2 [%s]  (metrics=%s, window=%dmin)' % (
        ','.join(systems), '+'.join(s.lstrip('_') for s in KEEP_SUFFIXES), LENGTH)
    write_summary(results, systems, out_root, 'PC + PageRank (fisherz)', dataset_desc, time_cost)
