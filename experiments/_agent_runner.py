#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Internal: run one localization method (baseline_common.main) on ONE agent dataset (MDOC or MARBLEBench) with an explicit config."""
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
    ap.add_argument('--out-method', default=None,
                    help='method label for the output/<dataset>/<method>/ tree')
    ap.add_argument('--limit', type=int, default=None,
                    help='cap number of samples (smoke test)')
    args = ap.parse_args()

    os.environ['AGENT_DATASET'] = args.dataset      # before importing the engine
    from baseline_common import main as run_main

    opts = dict(method=args.method, align=_bool(args.align), tag=args.tag,
                detector=args.detector, out_method=args.out_method)
    if args.method == 'relrrf':                     # MicroARCL fusion knobs
        opts.update(variant='both', k_rrf=60, include_residual=False,
                    k=args.k, mu=args.mu, fuse=args.fuse)
    if args.limit:
        opts['limit'] = args.limit
    run_main(**opts)


if __name__ == '__main__':
    main()
