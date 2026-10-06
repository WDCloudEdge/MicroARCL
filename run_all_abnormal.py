"""Batch-validate the agent RCA pipeline over all abnormal samples.

Ground truth is parsed from the label file(s) (data/MDOC/abnormal/*_label.txt);
each sample is run label-free, then we judge whether the localization hit the
ground-truth root-cause service.
"""
import os
import re
import glob
import json
import sys
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import run_agent_rca as R

ABN = 'data/MDOC/abnormal/load-5'
TOP_K_CUTOFFS = tuple(range(1, 11))
TIMING_STAGES = (
    'severity',
    'gnn',
    'llm',
    'total',
)


class _Tee:
    """Write batch output to both the terminal and the experiment log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)()
                   for stream in self.streams)


def parse_labels(abn_dir):
    """Parse '=== <sample> ===' blocks from *_label.txt into
    {sample_name: {service, fault_type, root_cause, ...}}."""
    labels = {}
    for lf in glob.glob(os.path.join(abn_dir, '*_label.txt')):
    # for lf in glob.glob(os.path.join(abn_dir, 'agent-network-planner_label.txt')):
        text = open(lf, encoding='utf-8').read()
        blocks = re.split(r'^===\s*(.+?)\s*===\s*$', text, flags=re.M)
        # blocks = ['', name1, body1, name2, body2, ...]
        for i in range(1, len(blocks), 2):
            name = blocks[i].strip()
            body = blocks[i + 1]
            kv = {}
            for line in body.splitlines():
                m = re.match(r'\s*([a-z_]+):\s*(.+?)\s*$', line)
                if m:
                    kv[m.group(1)] = m.group(2).strip()
            labels[name] = kv
    return labels


def _latest_result(sample_dir):
    fs = glob.glob(os.path.join(sample_dir, 'result-agent_rca_*.json'))
    return max(fs, key=os.path.getmtime) if fs else None


def _judge(gt_service, d):
    """Return (final_hit, sev_hit, gnn_hit, detail dict)."""
    r = d.get('fine_grained_result', {}) or {}
    root_cause = str(r.get('root_cause') or '')
    reranked = r.get('reranked_topk') or []
    top1 = reranked[0] if reranked else None
    final_hit = (gt_service in root_cause) or (top1 == gt_service)
    sev = d.get('coarse_ranking_severity') or []
    sev_top1 = sev[0]['service'] if sev else None
    sev_hit = sev_top1 == gt_service
    gnn = d.get('coarse_ranking_gnn') or []
    gnn_top1 = R._node_to_service(gnn[0]['node']) if gnn else None
    gnn_hit = gnn_top1 == gt_service
    ranks = d.get('evaluation_top_k') or {}
    gnn_k = ranks.get('gnn_top_k')
    sev_k = ranks.get('severity_top_k')
    llm_k = ranks.get('llm_top_k')
    if gnn_k is None:
        gnn_k = R._service_rank(
            [R._node_to_service(row['node']) for row in gnn], gt_service)
    if sev_k is None:
        sev_k = R._service_rank(
            [row['service'] for row in sev], gt_service)
    if llm_k is None:
        llm_k = R._service_rank(reranked, gt_service)
    return final_hit, sev_hit, gnn_hit, {
        'root_cause': root_cause, 'type': r.get('root_cause_type'),
        'conf': r.get('confidence'), 'reranked_top1': top1,
        'sev_top1': sev_top1, 'gnn_top1': gnn_top1,
        'gnn_top_k': gnn_k, 'severity_top_k': sev_k,
        'llm_top_k': llm_k}


def _topk_metrics(rows, rank_key, cutoffs=TOP_K_CUTOFFS):
    """Calculate ACC@K and AVG@N; missing/zero ranks count as misses."""
    total = len(rows)
    if total == 0:
        return ({k: 0.0 for k in cutoffs},
                {n: 0.0 for n in cutoffs})

    ranks = []
    for row in rows:
        try:
            ranks.append(int(row.get(rank_key) or 0))
        except (TypeError, ValueError):
            ranks.append(0)

    max_cutoff = max(cutoffs)
    acc_by_k = {
        k: sum(0 < rank <= k for rank in ranks) / total
        for k in range(1, max_cutoff + 1)
    }
    acc = {k: acc_by_k[k] for k in cutoffs}
    avg = {
        n: sum(acc_by_k[k] for k in range(1, n + 1)) / n
        for n in cutoffs
    }
    return acc, avg


def _print_topk_metrics(rows):
    """Print ranking metrics for each pipeline stage."""
    print('\nTop-K ranking metrics (rank 0/missing = miss)')
    header = f'{"stage":<10}'
    header += ''.join(f'{"ACC@" + str(k):>10}' for k in TOP_K_CUTOFFS)
    header += ''.join(f'{"AVG@" + str(n):>10}' for n in TOP_K_CUTOFFS)
    print(header)
    for stage, rank_key in (
            ('GNN', 'gnn_top_k'),
            ('Severity', 'severity_top_k'),
            ('LLM', 'llm_top_k')):
        acc, avg = _topk_metrics(rows, rank_key)
        values = [acc[k] for k in TOP_K_CUTOFFS]
        values.extend(avg[n] for n in TOP_K_CUTOFFS)
        print(f'{stage:<10}' + ''.join(f'{value:>10.3f}' for value in values))


def _print_timing_summary(rows):
    """Print per-stage means over samples that produced each timing value."""
    print('\nTiming summary (seconds)')
    print(f'{"stage":<32}{"mean":>12}{"samples":>12}')
    for stage in TIMING_STAGES:
        values = [row['timing_seconds'][stage] for row in rows
                  if stage in row.get('timing_seconds', {})]
        if values:
            print(f'{stage:<32}{sum(values) / len(values):>12.3f}'
                  f'{len(values):>12}')
        else:
            print(f'{stage:<32}{"N/A":>12}{0:>12}')


def _run_batch():
    labels = parse_labels(ABN)
    samples = [(n, os.path.join(ABN, n)) for n in sorted(labels)
               if os.path.isdir(os.path.join(ABN, n))]
    print(f'Found {len(samples)} labeled samples with data.')
    rows = []
    for name, sdir in samples:
        gt = labels[name].get('service', '')
        fault = labels[name].get('fault_type', name)
        print('\n' + '#' * 70 + f'\n##### {name}  (gt={gt}, fault={fault})\n' + '#' * 70)
        sample_started = time.perf_counter()
        run_result = None
        error = None
        try:
            run_result = R.main(sample_dir=sdir)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            traceback.print_exc()
        sample_elapsed = time.perf_counter() - sample_started
        timing_seconds = dict((run_result or {}).get('timing_seconds') or {})
        timing_seconds['total'] = sample_elapsed
        print(f'\n[Sample timing] {name}')
        for stage in TIMING_STAGES:
            if stage in timing_seconds:
                print(f'  {stage}: {timing_seconds[stage]:.3f}s')
        if not run_result:
            rows.append({
                'fault': fault,
                'gt': gt,
                'final_hit': None,
                'status': 'failed',
                'error': error or 'No result returned',
                'timing_seconds': timing_seconds,
            })
            continue
        fh, sh, gh, det = _judge(gt, run_result)
        rows.append({'fault': fault, 'gt': gt, 'final_hit': fh,
                     'sev_hit': sh, 'gnn_hit': gh, 'status': 'success',
                     'timing_seconds': timing_seconds, **det})

    print('\n\n' + '=' * 100 + '\nSUMMARY\n' + '=' * 100)
    print(f'{"fault":<13}{"final":<7}{"GNN-k":<7}{"SEV-k":<7}{"LLM-k":<7}'
          f'{"type":<15}{"conf":<6}{"localized root_cause":<40}')
    n_final = n_sev = n_gnn = 0
    for r in rows:
        fh = r.get('final_hit')
        n_final += 1 if fh else 0
        n_sev += 1 if r.get('sev_hit') else 0
        n_gnn += 1 if r.get('gnn_hit') else 0
        mark = lambda b: '✓' if b else ('✗' if b is not None else '?')
        print(f'{r["fault"]:<13}{mark(fh):<7}{str(r.get("gnn_top_k",0)):<7}'
              f'{str(r.get("severity_top_k",0)):<7}{str(r.get("llm_top_k",0)):<7}'
              f'{str(r.get("type",""))[:14]:<15}{str(r.get("conf","")):<6}{str(r.get("root_cause",""))[:38]:<40}')
    tot = len(rows)
    print('-' * 100)
    print(f'Localization accuracy  final(LLM)={n_final}/{tot}  severity#1={n_sev}/{tot}  gnn#1={n_gnn}/{tot}')
    _print_topk_metrics(rows)
    successful = sum(row.get('status') == 'success' for row in rows)
    failed = tot - successful
    print(f'\nBatch samples  planned={len(samples)}  completed={tot}  '
          f'successful={successful}  failed={failed}')
    _print_timing_summary(rows)


def main():
    batch_started_at = datetime.now().astimezone()
    batch_started = time.perf_counter()
    timestamp = batch_started_at.strftime('%Y-%m-%d-%H:%M:%S')
    log_path = os.path.join(ABN, f'batch-agent_rca_{timestamp}.log')
    with open(log_path, 'w', encoding='utf-8') as log_file:
        tee_out = _Tee(sys.stdout, log_file)
        tee_err = _Tee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f'Batch start time: {batch_started_at.isoformat(timespec="seconds")}')
            print(f'Batch log: {log_path}')
            try:
                _run_batch()
            finally:
                batch_ended_at = datetime.now().astimezone()
                batch_elapsed = time.perf_counter() - batch_started
                print(f'\nBatch end time: {batch_ended_at.isoformat(timespec="seconds")}')
                print(f'Batch wall-clock time: {batch_elapsed:.3f}s')
                print(f'Batch log written: {log_path}')


if __name__ == '__main__':
    main()
