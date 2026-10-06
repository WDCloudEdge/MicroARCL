#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run the CausalRCA (PC + PageRank) pipeline on the agent-service dataset."""

import time
from utils_microarcl import *
from typing import List
from train_latency_MicroARCL_bookinfo import Simple
import torch.optim as optim
from torch.optim import lr_scheduler
from utils import *
from modules import *
from config import CONFIG
import warnings
import argparse

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--indx', type=int, default=0, help='index')
parser.add_argument('--atype', type=str, default='cpu-hog1_', help='anomaly type')
parser.add_argument('--gamma', type=float, default=0.25, help='gamma')
parser.add_argument('--eta', type=int, default=10, help='eta')
args = parser.parse_args()

CONFIG.cuda = torch.cuda.is_available()
CONFIG.factor = not CONFIG.no_factor


# compute constraint h(A) value
def _h_A(A, m):
    expm_A = matrix_poly(A * A, m)
    h_A = torch.trace(expm_A) - m
    return h_A


prox_plus = torch.nn.Threshold(0., 0.)


def stau(w, tau):
    w1 = prox_plus(torch.abs(w) - tau)
    return torch.sign(w) * w1


def update_optimizer(optimizer, original_lr, c_A):
    '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
    MAX_LR = 1e-2
    MIN_LR = 1e-4

    estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
    if estimated_lr > MAX_LR:
        lr = MAX_LR
    elif estimated_lr < MIN_LR:
        lr = MIN_LR
    else:
        lr = estimated_lr

    # set LR
    for parame_group in optimizer.param_groups:
        parame_group['lr'] = lr

    return optimizer, lr


def service_of_column(col):
    """Extract the service name from an instance column, e.g.
    'agent-network-image-gen-5bffdff844-xl6nh_cpu' -> 'agent-network-image-gen'.
    Columns follow `<service>-<deploy-hash>-<pod-hash>_<metric>`; service names
    and metric names contain no dashes/underscores that collide with this."""
    without_metric = col.rsplit('_', 1)[0]
    return without_metric.rsplit('-', 2)[0]


# ===================================
# statistics (ACC@K / AVG@K / MRR, rank 0 = miss) -- same defs as baseline_common.py
# reported for TOTAL, per group (single/multi), and per load
# ===================================
def write_summary(results, groups, out_root, method, dataset_desc, time_cost, maxk=10):
    def topk_metrics(ranks):
        n = len(ranks)
        acc = [(sum(1 for r in ranks if 1 <= r <= k) / n if n else 0.0)
               for k in range(1, maxk + 1)]
        avg = [sum(acc[:k]) / k for k in range(1, maxk + 1)]
        return acc, avg

    def fmt_row(label, ranks):
        acc, avg = topk_metrics(ranks)
        mrr = (sum(1.0 / r for r in ranks if r >= 1) / len(ranks)) if ranks else 0.0
        return '{:<26s}'.format(label) \
            + ''.join('{:>8.3f}'.format(v) for v in acc) \
            + ''.join('{:>8.3f}'.format(v) for v in avg) \
            + '{:>8.3f}'.format(mrr) + '{:>5d}'.format(len(ranks))

    # ordered-unique group and group/load keys (discovery order)
    seen_g, seen_gl = [], []
    for r in results:
        if r['group'] not in seen_g:
            seen_g.append(r['group'])
        gl = (r['group'], r['load'])
        if gl not in seen_gl:
            seen_gl.append(gl)

    all_ranks = [r['rank'] for r in results]
    hits = sum(1 for r in all_ranks if r >= 1)

    ts = datetime.now().strftime('%Y-%m-%d-%H.%M.%S')
    summary_path = os.path.join(out_root, 'summary_' + ts + '.log')
    lines = []
    lines.append('=' * 120)
    lines.append('SUMMARY (CausalRCA: ' + method + ')  dataset=' + dataset_desc)
    lines.append('=' * 120)

    # per-sample topK listing, grouped
    lines.append('{:<60s}{:>8s}'.format('case', 'topK'))
    for r in results:
        lines.append('{:<60s}{:>8d}'.format(r['group'] + '/' + r['load'] + '/' + r['case'], r['rank']))
    lines.append('-' * 120)

    lines.append('Top-K ranking metrics (rank 0/missing = miss)')
    header = '{:<26s}'.format('subset') \
        + ''.join('{:>8s}'.format('ACC@%d' % k) for k in range(1, maxk + 1)) \
        + ''.join('{:>8s}'.format('AVG@%d' % k) for k in range(1, maxk + 1)) \
        + '{:>8s}'.format('MRR') + '{:>5s}'.format('n')
    lines.append(header)
    lines.append(fmt_row('TOTAL', all_ranks))
    for g in seen_g:
        gr = [r['rank'] for r in results if r['group'] == g]
        lines.append(fmt_row('[' + g + ']', gr))
    for (g, l) in seen_gl:
        lr = [r['rank'] for r in results if r['group'] == g and r['load'] == l]
        lines.append(fmt_row('  ' + g + '/' + l, lr))

    lines.append('')
    lines.append('Samples: {}  hits: {}  misses: {}'.format(len(all_ranks), hits, len(all_ranks) - hits))
    lines.append('Time cost: {:.3f}s'.format(time_cost))
    summary_text = '\n'.join(lines)
    print(summary_text)
    with open(summary_path, 'w') as f:
        f.write(summary_text + '\n')
    print('summary written: ' + summary_path)


# ===================================
# main
# ===================================
if __name__ == '__main__':
    def read_label_logs(namespace_path, label_service, simple_list: [Simple]):
        """Parse a `<service>_label.txt` file made of `=== <dir> ===` blocks and
        append one Simple per fault case."""
        if simple_list is None:
            simple_list = []
        file_path = os.path.join(namespace_path, label_service + '_label.txt')
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return simple_list

        cur = None

        def flush(block):
            if not block:
                return
            required = ('dir', 'service', 'begin', 'end')
            if not all(k in block for k in required):
                return
            simple_list.append(Simple(block['begin'], block['end'],
                                      block['dir'], block['service'], block['dir']))

        for line in lines:
            line = line.rstrip('\n')
            stripped = line.strip()
            if stripped.startswith('===') and stripped.endswith('==='):
                flush(cur)
                cur = {'dir': stripped.strip('= ').strip()}
                continue
            if cur is None:
                continue
            if ':' not in stripped:
                continue
            key, value = stripped.split(':', 1)
            key = key.strip()
            value = value.strip()
            if key == 'service':
                cur['service'] = value
            elif key == 'root_cause':
                # root_cause looks like 'agent-network-planner/mem_load'
                cur['root_cause'] = value
            elif key == 'window_range':
                # window_range: <begin> <end>  (unix timestamps, UTC)
                parts = value.split()
                if len(parts) >= 2:
                    cur['begin'] = int(parts[0])
                    cur['end'] = int(parts[1])
        flush(cur)
        return simple_list

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # MicroARCL repo root
    # walk <abnormal>/<group>/<load>/  e.g. single/load-3, single/load-5, multi/load-3-multi
    abnormal_rel = 'data/MDOC/abnormal'
    abnormal_dir = os.path.join(base_dir, abnormal_rel)

    out_root = 'MicroARCL/agent/instance'
    os.makedirs(out_root, exist_ok=True)

    groups = [g for g in sorted(os.listdir(abnormal_dir))
              if os.path.isdir(os.path.join(abnormal_dir, g))]

    # collected per-sample outcomes: dict(group, load, case, root_cause, rank)  rank=0 = miss
    results = []
    _LIMIT = int(os.environ.get('CAUSALRCA_LIMIT', '0')) or None

    begin_tt = time.time()
    for group in groups:
        if _LIMIT and len(results) >= _LIMIT:
            break
        group_dir = os.path.join(abnormal_dir, group)
        loads = [l for l in sorted(os.listdir(group_dir))
                 if os.path.isdir(os.path.join(group_dir, l))]
        for load in loads:
            if _LIMIT and len(results) >= _LIMIT:
                break
            dataset_dir = os.path.join(group_dir, load)
            out_dir = os.path.join(out_root, group, load)
            os.makedirs(out_dir, exist_ok=True)

            # one label file per root-cause service in the dataset directory
            root_cause_services = [f[:-len('_label.txt')]
                                   for f in sorted(os.listdir(dataset_dir))
                                   if f.endswith('_label.txt')]

            for root_cause_service in root_cause_services:
                if _LIMIT and len(results) >= _LIMIT:
                    break
                simples: List[Simple] = []
                read_label_logs(dataset_dir, root_cause_service, simples)
                for simple in simples:
                    if _LIMIT and len(results) >= _LIMIT:
                        break
                    case = simple.label
                    tag = group + '/' + load + '/' + case
                    print(tag)
                    instance_csv = os.path.join(dataset_dir, simple.dir,
                                                'agent-network', 'metrics', 'instance.csv')
                    if not os.path.exists(instance_csv):
                        print(tag + ' has no instance.csv, skip')
                        continue

                    all_data = pd.read_csv(instance_csv)
                    all_data = df_time_limit_normalization(all_data,
                                                           simple.global_now_time,
                                                           simple.global_end_time)
                    name = [i for i in all_data.columns if i != 'timestamp']
                    data = all_data[name]

                    from causallearn.search.ConstraintBased.PC import pc
                    from causallearn.utils.cit import chisq

                    cg = pc(data.to_numpy(), 0.05, chisq, False, 0, -1)
                    adj = cg.G.graph

                    rank = 0  # 0 = root cause not located (miss)
                    # Change the adj to graph
                    G = nx.DiGraph()
                    for i in range(len(adj)):
                        for j in range(len(adj)):
                            if adj[i, j] == -1:
                                G.add_edge(i, j)
                            if adj[i, j] == 1:
                                G.add_edge(j, i)
                    nodes = sorted(G.nodes())
                    adj = np.asarray(nx.to_numpy_array(G, nodelist=nodes)) if G.number_of_nodes() else np.zeros((0, 0))

                    def _record(rk):
                        results.append({'group': group, 'load': load, 'case': case,
                                        'root_cause': simple.root_cause, 'rank': rk})

                    if not np.any(adj):
                        print(tag + ' is absent')
                        with open(os.path.join(out_dir, case + '.log'), "w") as output_file:
                            print('root cause: ' + simple.root_cause, file=output_file)
                            print("topK: 0", file=output_file)
                        _record(rank)
                        continue

                    # PageRank
                    from sknetwork.ranking import PageRank

                    pagerank = PageRank()
                    scores = pagerank.fit_predict(np.abs(adj.T))

                    score_dict = {}
                    for i, s in enumerate(scores):
                        score_dict[name[nodes[i]]] = s
                    sorted_scores = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
                    count = 0
                    with open(os.path.join(out_dir, case + '.log'), "w") as output_file:
                        print('root cause: ' + simple.root_cause, file=output_file)
                        for sorted_score in sorted_scores:
                            count += 1
                            print(sorted_score, file=output_file)
                            if service_of_column(sorted_score[0]) == simple.root_cause:
                                rank = count
                                break
                        # topK: 0 when the root cause never appears in the ranking
                        print("topK: " + str(rank), file=output_file)
                    _record(rank)
    time_cost = time.time() - begin_tt
    print('time cost:' + str(time_cost))

    write_summary(results, groups, out_root, 'PC + PageRank', abnormal_rel, time_cost)
    # uniform output/<dataset>/<method>/ log
    import sys as _s
    _s.path.append(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))))
    import results_log
    _mn = os.environ.get('CAUSALRCA_METHOD', 'CloudRanger')
    _ds = 'MAR' if 'MARBLEBench' in abnormal_rel else 'MDOC'
    _per = time_cost / max(len(results), 1)
    _rows = []
    for _r in results:
        _c = _r['case']; _gt = _r['root_cause']
        _ft = _c[len(_gt) + 1:].rsplit('_', 1)[0] if _c.startswith(_gt) else ''
        _rows.append({'group': _r['group'], 'load': _r['load'], 'case': _c,
                      'fault': _ft, 'gt': _gt, 'category': '', 'rank': _r['rank'],
                      'top1': None, 'status': 'success', 'timing': {'total': _per}})
    results_log.emit_from_rows(_ds, _mn, _mn, _rows, kind='agent')
