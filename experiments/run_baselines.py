#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ3/RQ4/RQ6 (comparison methods) -- run the baseline RCL approaches on the
five datasets, alongside MicroARCL (experiments/run_microarcl.py).

Wired baselines (all run under the single merged .venv):
  * MicroRCA    -- Birch + Personalized PageRank over the trace-derived call
    topology. NOT run on SockShop (SS): RE2 SockShop has no traces -> no graph.
  * TORAI       -- multi-source scoring + GMM + RCD (RCAEval). All 5 datasets.
  * CausalRCA   -- DAG-GNN structure learning + PageRank. All 5 datasets.
  * CloudRanger -- PC (Fisher-Z) causal discovery + PageRank. All 5 datasets.

Runner scripts live under experiments/: baselines/{torai,microrca,causalrca}/.
CausalRCA/CloudRanger (vendored under experiments/baselines/causalrca/) are run
as `python experiments/baselines/causalrca/<script>` so that dir is sys.path[0]
and its config.py/graph.py shadow the root ones; RCAEVAL_DATA points at the
in-repo data/RCAEval; the agent/marble runners take a cap via $CAUSALRCA_LIMIT,
the RE2 runner via --limit.

All results land in output/<dataset>/<method>/ in the uniform log format.

Usage:
  python experiments/run_baselines.py --baseline CausalRCA --dataset all
  python experiments/run_baselines.py --baseline CloudRanger --dataset SS --limit 3
"""
import os
import sys
import subprocess
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable
RC = os.path.join('data', 'RCAEval')
RC_ABS = os.path.join(REPO, 'data', 'RCAEval')
CAUSAL_DIR = os.path.join(HERE, 'baselines', 'causalrca')
TORAI_DIR = os.path.join('experiments', 'baselines', 'torai')
MICRORCA_DIR = os.path.join('experiments', 'baselines', 'microrca')
LAGRCA_DIR = os.path.join(HERE, 'baselines', 'lagrca')

AGENT_SELECTOR = {'MDOC': 'MDOC', 'MAR': 'MARBLEBench'}
ORDER = ['MDOC', 'MAR', 'SS', 'OB', 'TT']

# LagRCA dataset key + its dataset-builder script (built once into lagrca/data/).
LAGRCA_DS = {'MDOC': 'agent', 'MAR': 'marble',
             'SS': 're2ss', 'OB': 're2ob', 'TT': 're2tt'}
LAGRCA_BUILD = {'agent': ('build_agent_dataset.py',), 'marble': ('build_mar_dataset.py',),
                're2ss': ('build_rcaeval_dataset.py', 're2ss'),
                're2ob': ('build_rcaeval_dataset.py', 're2ob'),
                're2tt': ('build_rcaeval_dataset.py', 're2tt')}


def _ext(script, args=(), daggnn=False, limit='env'):
    """Spec for a script vendored under experiments/baselines/causalrca/."""
    return ('ext', dict(script=script, args=list(args), daggnn=daggnn,
                        limit=limit))


BASELINES = {
    'MicroRCA': {
        'MDOC': ('agent', dict(method='ppr', detector='birch', align='false')),
        'MAR':  ('agent', dict(method='ppr', detector='birch', align='false')),
        # SS omitted: SockShop RE2 cases have no traces -> no PPR topology.
        'OB': ('script', os.path.join(MICRORCA_DIR, 'run_all_abnormal_RE2_birch_ppr.py'),
               ['--root', RC, '--suite', 're2ob'], True),
        'TT': ('script', os.path.join(MICRORCA_DIR, 'run_all_abnormal_RE2_birch_ppr.py'),
               ['--root', RC, '--suite', 're2tt'], True),
    },
    'TORAI': {
        'MDOC': ('script', os.path.join(TORAI_DIR, 'run_torai_agent.py'),
                 ['data/MDOC/abnormal', '--onset', 'label'], True),
        'MAR':  ('script', os.path.join(TORAI_DIR, 'run_torai_marble.py'),
                 ['data/MARBLEBench/abnormal', '--onset', 'label'], True),
        'SS': ('script', os.path.join(TORAI_DIR, 'run_torai_re2.py'),
               [RC, '--suites', 're2ss', '--onset', 'label'], True),
        'OB': ('script', os.path.join(TORAI_DIR, 'run_torai_re2.py'),
               [RC, '--suites', 're2ob', '--onset', 'label'], True),
        'TT': ('script', os.path.join(TORAI_DIR, 'run_torai_re2.py'),
               [RC, '--suites', 're2tt', '--onset', 'label'], True),
    },
    'CausalRCA': {   # DAG-GNN + PageRank
        'MDOC': _ext('train_all_services_MicroARCL_agent_daggnn.py', daggnn=True),
        'MAR':  _ext('train_all_services_MicroARCL_marble_daggnn.py', daggnn=True),
        'SS': _ext('train_rcaeval_re2.py',
                   ['--method', 'daggnn', '--system', 'ss'], daggnn=True, limit='flag'),
        'OB': _ext('train_rcaeval_re2.py',
                   ['--method', 'daggnn', '--system', 'ob'], daggnn=True, limit='flag'),
        'TT': _ext('train_rcaeval_re2.py',
                   ['--method', 'daggnn', '--system', 'tt'], daggnn=True, limit='flag'),
    },
    'CloudRanger': {  # PC (Fisher-Z) + PageRank
        'MDOC': _ext('train_all_services_MicroARCL_agent.py'),
        'MAR':  _ext('train_all_services_MicroARCL_marble.py'),
        'SS': _ext('train_rcaeval_re2.py',
                   ['--method', 'pc', '--system', 'ss'], limit='flag'),
        'OB': _ext('train_rcaeval_re2.py',
                   ['--method', 'pc', '--system', 'ob'], limit='flag'),
        'TT': _ext('train_rcaeval_re2.py',
                   ['--method', 'pc', '--system', 'tt'], limit='flag'),
    },
    'LagRCA': {name: ('lagrca', LAGRCA_DS[name]) for name in ORDER},
}


def _run(cmd, cwd=REPO, env=None):
    print(f"\n$ {' '.join(str(c) for c in cmd[1:])}  (cwd={cwd})", flush=True)
    subprocess.run([str(c) for c in cmd], cwd=cwd, env=env, check=True)


def run_dataset(baseline, name, limit=None):
    spec = BASELINES[baseline].get(name)
    if spec is None:
        print("=" * 72, f"\n[RQ3/4/6 baseline] {baseline} on {name}: SKIPPED "
              f"(not evaluated for this baseline)")
        return
    print("=" * 72, f"\n[RQ3/4/6 baseline] {baseline} on {name}")
    kind = spec[0]
    if kind == 'agent':
        opts = spec[1]
        cmd = [PY, os.path.join(HERE, '_agent_runner.py'),
               '--dataset', AGENT_SELECTOR[name], '--method', opts['method'],
               '--detector', opts['detector'], '--align', opts['align'],
               '--out-method', baseline, '--tag', f'baseline_{baseline}_{name}']
        if limit:
            cmd += ['--limit', limit]
        _run(cmd)
    elif kind == 'script':
        _, script, args, supports_limit = spec
        cmd = [PY, os.path.join(REPO, script), *args]
        if limit and supports_limit:
            cmd += ['--limit', limit]
        _run(cmd)
    elif kind == 'ext':                               # experiments/baselines/causalrca
        cfg = spec[1]
        env = os.environ.copy()
        env['RCAEVAL_DATA'] = RC_ABS
        env['CAUSALRCA_METHOD'] = baseline             # CausalRCA / CloudRanger
        if cfg['daggnn']:
            env.setdefault('DAGGNN_REL_THRESHOLD', '0.3')
            for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                      'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                env.setdefault(k, '4')
        args = list(cfg['args'])
        if limit and cfg['limit'] == 'flag':
            args += ['--limit', str(limit)]
        elif limit and cfg['limit'] == 'env':
            env['CAUSALRCA_LIMIT'] = str(limit)
        cmd = [PY, os.path.join(CAUSAL_DIR, cfg['script']), *args]
        _run(cmd, cwd=CAUSAL_DIR, env=env)
    else:                                             # 'lagrca' -> PyG GNN; build once, then train+eval
        ds = spec[1]
        env = os.environ.copy()
        env['RCAEVAL_DATA'] = RC_ABS                  # for build_rcaeval_dataset
        data_dir = os.path.join(LAGRCA_DIR, 'data', ds)
        if not os.path.exists(os.path.join(data_dir, 'case_data.pkl')):
            build = LAGRCA_BUILD[ds]
            _run([PY, os.path.join(LAGRCA_DIR, build[0]), *build[1:]],
                 cwd=LAGRCA_DIR, env=env)
        margs = ['-ds', ds, '--stride', '3', '--epochs',
                 str(limit) if limit else '20']       # --limit N -> quick N-epoch smoke
        _run([PY, os.path.join(LAGRCA_DIR, 'main.py'), *margs],
             cwd=LAGRCA_DIR, env=env)


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
