#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ5 -- MicroARCL component ablation on the agent datasets (MDOC / MAR).

Holds the candidate budget k fixed at the dataset's final value (MAR k=5,
MDOC k=15), mu=1, and removes one module at a time:

  full          agent-aware Birch + reliability wsum fusion - mu*chain_lag
  MicroARCL-A   remove agent-aware detection  (detector=birch, alignment off)
  MicroARCL-W   remove reliability fusion      (fuse=gate: gate score only)
  MicroARCL-L   remove request-level lag        (mu=0)

Each variant reuses experiments/_agent_runner.py so the method path is identical
to run_microarcl.py; only the ablated knob changes.

Usage:
  python experiments/run_ablation.py --dataset all
  python experiments/run_ablation.py --dataset MDOC --limit 3   # smoke test
"""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable

AGENT = {'MDOC': ('MDOC', 15), 'MAR': ('MARBLEBench', 5)}
ORDER = ['MDOC', 'MAR']

# variant -> (detector, fuse, mu, align)
VARIANTS = {
    'full':        ('abirch', 'wsum', 1.0, 'true'),
    'MicroARCL-A': ('birch',  'wsum', 1.0, 'false'),
    'MicroARCL-W': ('abirch', 'gate', 1.0, 'true'),
    'MicroARCL-L': ('abirch', 'wsum', 0.0, 'true'),
}


def _run(cmd):
    print(f"\n$ {' '.join(str(c) for c in cmd[1:])}", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=REPO, check=True)


def run_variant(name, variant, limit=None):
    selector, k = AGENT[name]
    detector, fuse, mu, align = VARIANTS[variant]
    print("=" * 72, f"\n[RQ5] {name}  {variant}  "
          f"(detector={detector}, fuse={fuse}, mu={mu}, align={align}, k={k})")
    cmd = [PY, os.path.join(HERE, '_agent_runner.py'),
           '--dataset', selector, '--k', k, '--mu', mu,
           '--detector', detector, '--fuse', fuse, '--align', align,
           '--tag', f'ablation_{name}_{variant}']
    if limit:
        cmd += ['--limit', limit]
    _run(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='all', choices=('all', *ORDER))
    ap.add_argument('--variant', default='all',
                    choices=('all', *VARIANTS))
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()
    names = ORDER if args.dataset == 'all' else [args.dataset]
    variants = list(VARIANTS) if args.variant == 'all' else [args.variant]
    for name in names:
        for variant in variants:
            run_variant(name, variant, limit=args.limit)
    print("=" * 72, "\nDONE. Compare the batch logs across variants per dataset.")


if __name__ == '__main__':
    main()
