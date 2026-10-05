#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal: run one localization method (baseline_common.main) on ONE agent
dataset (MDOC or MARBLEBench) with an explicit config. Kept as a separate entry
so AGENT_DATASET is fixed BEFORE baseline_common is imported (it resolves the
abnormal-sample root at import time). The experiments/ drivers invoke this as a
subprocess, one per dataset, so each run gets a clean dataset root.

Handles both MicroARCL (`--method relrrf`) and the comparison baselines
(`--method ppr|direct|llm`); method-specific opts (k/mu/fuse) are ignored by the
baselines. Not meant to be called directly -- use the experiments/ drivers.
"""
import os
import sys
import argparse

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _bool(v):
    return str(v).lower() in ('1', 'true', 'yes', 'on')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=('MDOC', 'MARBLEBench'), required=True)
    ap.add_argument('--method', default='relrrf',
                    choices=('relrrf', 'ppr', 'direct', 'llm'))
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--mu', type=float, default=1.0)
    ap.add_argument('--detector', default='abirch',
                    choices=('abirch', 'birch', 'severity'))
    ap.add_argument('--fuse', default='wsum',
                    choices=('wsum', 'gate', 'max', 'mean', 'rrf'))
    ap.add_argument('--align', default='true')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--limit', type=int, default=None,
                    help='cap number of samples (smoke test)')
    args = ap.parse_args()

    os.environ['AGENT_DATASET'] = args.dataset      # before importing the engine
    from baseline_common import main as run_main

    opts = dict(method=args.method, align=_bool(args.align), tag=args.tag,
                detector=args.detector)
    if args.method == 'relrrf':                     # MicroARCL fusion knobs
        opts.update(variant='both', k_rrf=60, include_residual=False,
                    k=args.k, mu=args.mu, fuse=args.fuse)
    if args.limit:
        opts['limit'] = args.limit
    run_main(**opts)


if __name__ == '__main__':
    main()
