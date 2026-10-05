#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ3/RQ4/RQ6 (comparison methods) -- run the baseline RCL approaches on the
five datasets, alongside MicroARCL (experiments/run_microarcl.py).

Wired baselines:
  * MicroRCA -- Birch anomaly detection + Personalized PageRank over the
    trace-derived call topology. NOT run on SockShop (SS): the RCAEval RE2
    SockShop cases ship without traces, so there is no call topology.
  * TORAI    -- multi-source (metric/log/trace) anomaly scoring + GMM clustering
    + RCD refinement (RCAEval). Runs on all five datasets (metrics-only on RE2).

A baseline maps each dataset to a command spec:
  ('agent', opts)              -> experiments/_agent_runner.py (MDOC/MAR)
  ('script', name, args, lim)  -> `python <name> <args>` (RE2 runners / TORAI);
                                  lim = whether the script accepts --limit
A dataset omitted from a baseline's map is skipped (printed as SKIPPED).

Everything runs under the current interpreter (the merged .venv that includes
causal-learn + RCAEval for TORAI); no environment switching is needed.

Usage:
  python experiments/run_baselines.py --baseline TORAI --dataset all
  python experiments/run_baselines.py --baseline MicroRCA --dataset OB --limit 3
"""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
RC = os.path.join('data', 'RCAEval')

AGENT_SELECTOR = {'MDOC': 'MDOC', 'MAR': 'MARBLEBench'}
ORDER = ['MDOC', 'MAR', 'SS', 'OB', 'TT']

BASELINES = {
    'MicroRCA': {
        'MDOC': ('agent', dict(method='ppr', detector='birch', align='false')),
        'MAR':  ('agent', dict(method='ppr', detector='birch', align='false')),
        # SS omitted: SockShop RE2 cases have no traces -> no PPR topology.
        'OB': ('script', 'run_all_abnormal_RE2_birch_ppr.py',
               ['--root', RC, '--suite', 're2ob'], True),
        'TT': ('script', 'run_all_abnormal_RE2_birch_ppr.py',
               ['--root', RC, '--suite', 're2tt'], True),
    },
    'TORAI': {
        'MDOC': ('script', 'run_torai_agent.py',
                 ['data/MDOC/abnormal', '--onset', 'label'], True),
        'MAR':  ('script', 'run_torai_marble.py',
                 ['data/MARBLEBench/abnormal', '--onset', 'label'], True),
        'SS': ('script', 'run_torai_re2.py',
               [RC, '--suites', 're2ss', '--onset', 'label'], True),
        'OB': ('script', 'run_torai_re2.py',
               [RC, '--suites', 're2ob', '--onset', 'label'], True),
        'TT': ('script', 'run_torai_re2.py',
               [RC, '--suites', 're2tt', '--onset', 'label'], True),
    },
}


def _run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd[1:])}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def run_dataset(baseline, name, limit=None):
    spec = BASELINES[baseline].get(name)
    if spec is None:
        print("=" * 72, f"\n[RQ3/4/6 baseline] {baseline} on {name}: SKIPPED "
              f"(not evaluated for this baseline)")
        return
    print("=" * 72, f"\n[RQ3/4/6 baseline] {baseline} on {name}")
    if spec[0] == 'agent':
        opts = spec[1]
        cmd = [PY, os.path.join(HERE, '_agent_runner.py'),
               '--dataset', AGENT_SELECTOR[name], '--method', opts['method'],
               '--detector', opts['detector'], '--align', opts['align'],
               '--tag', f'baseline_{baseline}_{name}']
        if limit:
            cmd += ['--limit', limit]
    else:                                             # ('script', name, args, lim)
        _, script, args, supports_limit = spec
        cmd = [PY, os.path.join(REPO, script), *args]
        if limit and supports_limit:
            cmd += ['--limit', limit]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--baseline', default='TORAI', choices=tuple(BASELINES))
    ap.add_argument('--dataset', default='all', choices=('all', *ORDER))
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    names = ORDER if args.dataset == 'all' else [args.dataset]
    for name in names:
        run_dataset(args.baseline, name, limit=args.limit)
    print("=" * 72, "\nDONE. Baseline metrics/timing are in each run's batch log.")


if __name__ == '__main__':
    main()
