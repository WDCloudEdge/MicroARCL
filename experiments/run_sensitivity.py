#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ7 -- MicroARCL parameter sensitivity on the agent datasets (MDOC / MAR)."""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable

AGENT = {'MDOC': ('MDOC', 15), 'MAR': ('MARBLEBench', 5)}
ORDER = ['MDOC', 'MAR']
K_GRID = (5, 10, 15, 20, 25)
MU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


def _run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd[1:])}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def _point(selector, k, mu, tag, limit):
    cmd = [PY, os.path.join(HERE, '_agent_runner.py'),
           '--dataset', selector, '--k', k, '--mu', mu,
           '--detector', 'abirch', '--fuse', 'wsum', '--align', 'true',
           '--out-method', 'MicroARCL', '--tag', tag]
    if limit:
        cmd += ['--limit', limit]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all', *ORDER))
    ap.add_argument('--sweep', default='all', choices=('all', 'k', 'mu'))
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    names = ORDER if args.dataset == 'all' else [args.dataset]
    for name in names:
        selector, k_final = AGENT[name]
        if args.sweep in ('all', 'k'):
            print("=" * 72, f"\n[RQ7] {name} k-sweep (mu=1.0): {K_GRID}")
            for k in K_GRID:
                _point(selector, k, 1.0,
                       f'sensitivity_{name}_k{k}_mu1.0', args.limit)
        if args.sweep in ('all', 'mu'):
            print("=" * 72,
                  f"\n[RQ7] {name} mu-sweep (k={k_final}): {MU_GRID}")
            for mu in MU_GRID:
                _point(selector, k_final, mu,
                       f'sensitivity_{name}_k{k_final}_mu{mu}', args.limit)
    print("=" * 72, "\nDONE. Read ACC@1/ACC@3/MRR off each tagged batch log.")


if __name__ == '__main__':
    main()
