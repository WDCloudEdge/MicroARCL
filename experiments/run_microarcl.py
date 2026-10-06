#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ3/RQ4/RQ6 -- run MicroARCL (the final method) on all five datasets."""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
RCAEVAL_ROOT = os.path.join('data', 'RCAEval')

# name -> (path, selector, k).  path: 'agent' uses _agent_runner with the given
# AGENT_DATASET selector; 're2' uses the RE2 runner with the given --suite.
DATASETS = {
    'MDOC': ('agent', 'MDOC', 15),
    'MAR':  ('agent', 'MARBLEBench', 5),
    'SS':   ('re2', 're2ss', 15),
    'OB':   ('re2', 're2ob', 15),
    'TT':   ('re2', 're2tt', 15),
}
ORDER = ['MDOC', 'MAR', 'SS', 'OB', 'TT']
MU = 1.0


def _run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd[1:])}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def run_dataset(name, limit=None):
    path, selector, k = DATASETS[name]
    print("=" * 72, f"\n[RQ3/4/6] MicroARCL on {name}  (k={k}, mu={MU})")
    if path == 'agent':
        cmd = [PY, os.path.join(HERE, '_agent_runner.py'),
               '--dataset', selector, '--k', k, '--mu', MU,
               '--detector', 'abirch', '--fuse', 'wsum', '--align', 'true',
               '--out-method', 'MicroARCL', '--tag', f'microarcl_rq3_{name}']
        if limit:
            cmd += ['--limit', limit]
    else:
        cmd = [PY, os.path.join(HERE, 'microarcl',
               'run_all_abnormal_RE2_abirch_labelwin_MicroARCL.py'),
               '--root', RCAEVAL_ROOT, '--suite', selector]
        if limit:
            cmd += ['--limit', limit]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all',
                    choices=('all', *ORDER))
    ap.add_argument('--limit', type=int, default=None,
                    help='cap samples/cases per dataset (smoke test)')
    args = ap.parse_args()
    names = ORDER if args.dataset == 'all' else [args.dataset]
    for name in names:
        run_dataset(name, limit=args.limit)
    print("=" * 72, "\nDONE. Per-dataset metrics/timing/breakdowns are in each "
          "run's batch log.")


if __name__ == '__main__':
    main()
