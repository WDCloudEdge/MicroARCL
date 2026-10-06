#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared result logging for every method x dataset, so all runs emit the SAME batch-log format into a uniform tree:"""
import os
import sys
import time
from datetime import datetime
from contextlib import contextmanager, redirect_stdout, redirect_stderr

TOP_K_CUTOFFS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)


class _Tee:
    """Write batch output to both the terminal and the experiment log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


_REPO = os.path.dirname(os.path.abspath(__file__))


def out_root(dataset, method):
    """<repo>/output/<dataset>/<method>/ (created; absolute, cwd-independent)."""
    d = os.path.join(_REPO, 'output', dataset, method)
    os.makedirs(d, exist_ok=True)
    return d


def _topk_metrics(rows, rank_key='rank', cutoffs=TOP_K_CUTOFFS):
    """ACC@K and AVG@N; missing/zero ranks count as misses."""
    total = len(rows)
    if total == 0:
        return {k: 0.0 for k in cutoffs}, {n: 0.0 for n in cutoffs}
    ranks = []
    for row in rows:
        try:
            ranks.append(int(row.get(rank_key) or 0))
        except (TypeError, ValueError):
            ranks.append(0)
    mx = max(cutoffs)
    acc_by_k = {k: sum(0 < r <= k for r in ranks) / total
                for k in range(1, mx + 1)}
    acc = {k: acc_by_k[k] for k in cutoffs}
    avg = {n: sum(acc_by_k[k] for k in range(1, n + 1)) / n for n in cutoffs}
    return acc, avg


def sample_banner(level, name, gt, fault):
    """Uniform per-sample block header (`level` = 'group/load' or '<fault>')."""
    print('\n' + '#' * 70
          + f'\n##### [{level}] {name} (gt={gt}, fault={fault})\n'
          + '#' * 70)


def sample_eval(rank):
    print(f'[eval] gt rank = {rank}')


def _pm(label, subset):
    if not subset:
        return
    acc, avg = _topk_metrics(subset, 'rank')
    mrr = sum(1.0 / r['rank'] for r in subset if r.get('rank')) / len(subset)
    vals = [acc[k] for k in TOP_K_CUTOFFS] + [avg[n] for n in TOP_K_CUTOFFS]
    print(f'{label:<26}' + ''.join(f'{v:>9.3f}' for v in vals)
          + f'{mrr:>9.3f}' + f'{len(subset):>5}')


def _levels(rows, kind):
    """(label, subset) breakdown rows beneath the overall line."""
    out = []
    if kind == 're2':                                 # SS/OB/TT: per fault only
        for f in sorted({r.get('fault', '') for r in rows}):
            out.append((f'  «{f}»', [r for r in rows if r.get('fault') == f]))
        return out
    groups = sorted({r.get('group', '') for r in rows})   # agent: single/multi
    if len(groups) > 1:
        for g in groups:
            out.append((f'  «{g}»', [r for r in rows if r.get('group') == g]))
    gl = sorted({(r.get('group', ''), r.get('load', '')) for r in rows})
    if len(gl) > 1:
        for g, ld in gl:
            out.append((f'    <{g}/{ld}>',
                        [r for r in rows if r.get('group') == g
                         and r.get('load') == ld]))
    return out


def summarize(tag, rows, kind='agent', categories=()):
    """Print the SUMMARY block (table + ranking metrics + timing) for `rows`."""
    print('\n\n' + '=' * 96 + f'\nSUMMARY [{tag}]\n' + '=' * 96)
    print(f'{"group":<8}{"load":<14}{"fault":<12}{"rank":<6}{"gt":<24}'
          f'{"top1/root_cause":<26}')
    n_hit = 0
    for r in rows:
        n_hit += 1 if r.get('rank') == 1 else 0
        shown = r.get('root_cause') or r.get('top1') or ''
        print(f'{str(r.get("group",""))[:7]:<8}{str(r.get("load",""))[:13]:<14}'
              f'{str(r.get("fault","")):<12}{str(r.get("rank", 0)):<6}'
              f'{str(r.get("gt", ""))[:22]:<24}{str(shown)[:24]:<26}')
    tot = len(rows)
    print('-' * 96)
    print(f'Localization accuracy  top1={n_hit}/{tot}')
    print('\nRanking metrics (rank 0/missing = miss)')
    print(f'{"stage":<26}'
          + ''.join(f'{"ACC@"+str(k):>9}' for k in TOP_K_CUTOFFS)
          + ''.join(f'{"AVG@"+str(n):>9}' for n in TOP_K_CUTOFFS)
          + f'{"MRR":>9}' + f'{"n":>5}')
    _pm(tag, rows)                                    # overall
    for label, subset in _levels(rows, kind):
        _pm(label, subset)
    for cat in categories:                            # per root-cause category
        _pm(f'  [{cat}]', [r for r in rows if r.get('category') == cat])
    succ = sum(r.get('status', 'success') == 'success' for r in rows)
    print(f'\nBatch  completed={tot} successful={succ} failed={tot - succ}')
    for stage in ('materialize', 'detect', 'localize', 'total'):
        v = []
        for r in rows:
            tv = r.get('timing') or r.get('timing_seconds') or {}
            if isinstance(tv, dict) and stage in tv:
                v.append(tv[stage])
        if v:
            print(f'  mean {stage}: {sum(v)/len(v):.3f}s ({len(v)})')


def emit_from_rows(dataset, method, tag, rows, kind='agent', categories=(),
                   header_lines=()):
    """One-shot: write the full uniform batch log for already-collected `rows`
    (used by runners that compute results first, then log). Reconstructs the
    per-sample blocks (banner + gt rank) from the rows, then the summary."""
    with open_batch(dataset, method, tag, header_lines) as (root, ts):
        for r in rows:
            level = (r.get('fault', '') if kind == 're2'
                     else f"{r.get('group','')}/{r.get('load','')}")
            sample_banner(level, r.get('case', r.get('gt', '')),
                          r.get('gt', ''), r.get('fault', ''))
            if r.get('top1') is not None:
                print(f"top1={r.get('top1')}")
            sample_eval(r.get('rank', 0))
        summarize(tag, rows, kind=kind, categories=categories)
    write_sublogs(root, ts, tag, rows, kind=kind, categories=categories)


@contextmanager
def open_batch(dataset, method, tag, header_lines=()):
    """Tee stdout/stderr into output/<dataset>/<method>/batch-<tag>_<ts>.log and
    print the standard header. Yields the log directory."""
    root = out_root(dataset, method)
    ts = datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    log_path = os.path.join(root, f'batch-{tag}_{ts}.log')
    t0 = time.perf_counter()
    with open(log_path, 'w', encoding='utf-8') as lf:
        with redirect_stdout(_Tee(sys.stdout, lf)), \
                redirect_stderr(_Tee(sys.stderr, lf)):
            print(f'Batch start: {datetime.now().astimezone().isoformat(timespec="seconds")}')
            print(f'Overall log: {log_path}')
            for line in header_lines:
                print(line)
            try:
                yield root, ts
            finally:
                print(f'\nBatch wall-clock: {time.perf_counter() - t0:.1f}s')


def write_sublogs(root, ts, tag, rows, kind='agent', categories=()):
    """Write a per-sub-level summary log mirroring the overall one:
    agent -> <group>/ and <group>/<load>/ ; re2 -> <fault>/."""
    if kind == 're2':
        keys = sorted({r.get('fault', '') for r in rows})
        subsets = {(f,): [r for r in rows if r.get('fault') == f] for f in keys}
    else:
        subsets = {}
        for r in rows:
            subsets.setdefault((r.get('group', ''),), []).append(r)
        for r in rows:
            subsets.setdefault((r.get('group', ''), r.get('load', '')), []).append(r)
    for key, sub in subsets.items():
        d = os.path.join(root, *[str(k) for k in key])
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f'batch-{tag}_{ts}.log')
        with open(path, 'w', encoding='utf-8') as f:
            with redirect_stdout(f):
                print(f'Sub-level log [{"/".join(str(k) for k in key)}] for '
                      f'[{tag}]  ({ts})')
                summarize(tag, sub, kind=kind, categories=categories)
