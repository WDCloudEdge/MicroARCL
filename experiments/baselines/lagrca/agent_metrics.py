"""Ranking metrics aligned with MicroARCL baseline_common.py."""
import os
import re
import sys
import pickle

TOP_K_CUTOFFS = tuple(range(1, 11))

# root-cause service -> category (same mapping as baseline_common.CATEGORY)
CATEGORY = {
    'agent-network-pdf-parsing': 'doc',
    'agent-network-word-gen': 'doc',
    'agent-network-image': 'image',
    'agent-network-ocr': 'image',
    'agent-network-planner': 'sched',
    'agent-network-summarizer': 'sched',
}
CATEGORIES = ('doc', 'image', 'sched')
MARBLE_CATEGORY = {
    'agent-network-planner': 'sched',
    'agent-network-summarizer': 'sched',
    'agent-network-marble-coding': 'coding',
    'agent-network-marble-database': 'db',
    'agent-network-marble-minecraft': 'game',
    'agent-network-marble-research': 'web',
}
MARBLE_CATEGORIES = ('web', 'coding', 'db', 'game', 'sched')


def instance_to_service(name):
    """Map a LagRCA instance name to a service; physical nodes kept distinct."""
    if name.startswith('node-'):
        return name
    return re.sub(r'-pod\d+$', '', name)


def service_rank(order, gt):
    """1-based rank of gt over the dedup'd service ordering; 0 = miss."""
    if not gt:
        return 0
    seen = []
    for item in order or []:
        s = instance_to_service(str(item))
        if s not in seen:
            seen.append(s)
    return seen.index(gt) + 1 if gt in seen else 0


def topk_metrics(ranks, cutoffs=TOP_K_CUTOFFS):
    total = len(ranks)
    if total == 0:
        return {k: 0.0 for k in cutoffs}, {n: 0.0 for n in cutoffs}
    max_cut = max(cutoffs)
    acc_by_k = {k: sum(0 < r <= k for r in ranks) / total for k in range(1, max_cut + 1)}
    acc = {k: acc_by_k[k] for k in cutoffs}
    avg = {n: sum(acc_by_k[k] for k in range(1, n + 1)) / n for n in cutoffs}
    return acc, avg


def _mrr(ranks):
    if not ranks:
        return 0.0
    return sum(1.0 / r for r in ranks if r) / len(ranks)


def _parse_ts(ts):
    """case id 'group/load/name' -> (group, load). Falls back gracefully."""
    parts = str(ts).split('/')
    if len(parts) >= 3:
        return parts[0], parts[1]
    if len(parts) == 2:
        return '', parts[0]
    return '', ''


def build_rows(all_ans, all_labels, all_ts=None, category_map=CATEGORY):
    if all_ts is None:
        all_ts = [''] * len(all_ans)
    rows = []
    for order, label, ts in zip(all_ans, all_labels, all_ts):
        gt = (label[0] if isinstance(label, (list, tuple)) and label else label) or ''
        group, load = _parse_ts(ts)
        rows.append({'gt': gt, 'category': category_map.get(gt, 'other'),
                     'rank': service_rank(order, gt), 'group': group, 'load': load})
    return rows


def _header():
    return (f'{"stage":<24}'
            + ''.join(f'{"ACC@" + str(k):>9}' for k in TOP_K_CUTOFFS)
            + ''.join(f'{"AVG@" + str(n):>9}' for n in TOP_K_CUTOFFS)
            + f'{"MRR":>9}' + f'{"n":>5}')


def _line(label, subset):
    if not subset:
        return
    ranks = [r['rank'] for r in subset]
    acc, avg = topk_metrics(ranks)
    vals = [acc[k] for k in TOP_K_CUTOFFS] + [avg[n] for n in TOP_K_CUTOFFS]
    print(f'{label:<24}' + ''.join(f'{v:>9.3f}' for v in vals)
          + f'{_mrr(ranks):>9.3f}' + f'{len(subset):>5}')


def print_report(all_ans, all_labels, all_ts=None, tag='LagRCA'):
    marble = tag == 'marble'
    rows = build_rows(all_ans, all_labels, all_ts,
                      MARBLE_CATEGORY if marble else CATEGORY)
    print('\nRanking metrics (rank 0/missing = miss)  [aligned with MicroARCL baseline]')
    print(_header())

    _line(tag, rows)                                        # overall
    for cat in (MARBLE_CATEGORIES if marble else CATEGORIES):
        _line(f'  [{cat}]', [r for r in rows if r['category'] == cat])

    groups = sorted({r['group'] for r in rows if r['group']})
    if groups:
        print('-' * 20 + ' by group ' + '-' * 20)
        for g in groups:
            _line(f'  <{g}>', [r for r in rows if r['group'] == g])

    loads = sorted({r['load'] for r in rows if r['load']})
    if loads:
        print('-' * 20 + ' by load ' + '-' * 20)
        for ld in loads:
            _line(f'  {{{ld}}}', [r for r in rows if r['load'] == ld])

    n_hit1 = sum(r['rank'] == 1 for r in rows)
    print(f'\nTop-1 hits = {n_hit1}/{len(rows)}')
    return rows


# backward-compatible alias
def print_table(all_ans, all_labels, tag='LagRCA'):
    return print_report(all_ans, all_labels, None, tag)


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'data', 'agent', 'eval_results.pkl')
    d = pickle.load(open(path, 'rb'))
    tag = sys.argv[2] if len(sys.argv) > 2 else 'LagRCA'
    print_report(d['all_ans'], d['all_labels'], d.get('all_ts'), tag)
