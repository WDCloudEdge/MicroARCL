"""Batch-validate the agent RCA pipeline over all abnormal samples.

Ground truth is parsed from the label file(s) (data/thingo/abnormal/*_label.txt);
each sample is run label-free, then we judge whether the localization hit the
ground-truth root-cause service.
"""
import os
import re
import glob
import json
import traceback
import run_agent_rca as R

ABN = 'data/thingo/abnormal/load-5'


def parse_labels(abn_dir):
    """Parse '=== <sample> ===' blocks from *_label.txt into
    {sample_name: {service, fault_type, root_cause, ...}}."""
    labels = {}
    # for lf in glob.glob(os.path.join(abn_dir, '*_label.txt')):
    for lf in glob.glob(os.path.join(abn_dir, 'agent-network-planner_label.txt')):
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
    return final_hit, sev_hit, gnn_hit, {
        'root_cause': root_cause, 'type': r.get('root_cause_type'),
        'conf': r.get('confidence'), 'reranked_top1': top1,
        'sev_top1': sev_top1, 'gnn_top1': gnn_top1}


def main():
    labels = parse_labels(ABN)
    samples = [(n, os.path.join(ABN, n)) for n in sorted(labels)
               if os.path.isdir(os.path.join(ABN, n))]
    print(f'Found {len(samples)} labeled samples with data.')
    rows = []
    for name, sdir in samples:
        gt = labels[name].get('service', '')
        fault = labels[name].get('fault_type', name)
        print('\n' + '#' * 70 + f'\n##### {name}  (gt={gt}, fault={fault})\n' + '#' * 70)
        try:
            R.main(sample_dir=sdir)
        except Exception:
            traceback.print_exc()
        f = _latest_result(sdir)
        if not f:
            rows.append({'fault': fault, 'gt': gt, 'final_hit': None}); continue
        d = json.load(open(f, encoding='utf-8'))
        fh, sh, gh, det = _judge(gt, d)
        rows.append({'fault': fault, 'gt': gt, 'final_hit': fh,
                     'sev_hit': sh, 'gnn_hit': gh, **det})

    print('\n\n' + '=' * 100 + '\nSUMMARY\n' + '=' * 100)
    print(f'{"fault":<13}{"final":<7}{"sev":<5}{"gnn":<5}{"type":<15}{"conf":<6}{"localized root_cause":<40}')
    n_final = n_sev = n_gnn = 0
    for r in rows:
        fh = r.get('final_hit')
        n_final += 1 if fh else 0
        n_sev += 1 if r.get('sev_hit') else 0
        n_gnn += 1 if r.get('gnn_hit') else 0
        mark = lambda b: '✓' if b else ('✗' if b is not None else '?')
        print(f'{r["fault"]:<13}{mark(fh):<7}{mark(r.get("sev_hit")):<5}{mark(r.get("gnn_hit")):<5}'
              f'{str(r.get("type",""))[:14]:<15}{str(r.get("conf","")):<6}{str(r.get("root_cause",""))[:38]:<40}')
    tot = len(rows)
    print('-' * 100)
    print(f'Localization accuracy  final(LLM)={n_final}/{tot}  severity#1={n_sev}/{tot}  gnn#1={n_gnn}/{tot}')


if __name__ == '__main__':
    main()
