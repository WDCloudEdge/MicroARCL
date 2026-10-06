#!/usr/bin/env python3
"""Summarize spatial lag tables and plot their key comparisons."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = _REPO_ROOT / 'analysis' / 'tables'
FIG = _REPO_ROOT / 'analysis' / 'figures'


def make_figure(dataset: str) -> str:
    spatial = pd.read_csv(DATA / f'{dataset}_spatial.csv')
    timing = pd.read_csv(DATA / f'{dataset}_timing.csv')
    audit = pd.read_csv(DATA / f'{dataset}_audit.csv')
    if spatial.empty or timing.empty:
        raise ValueError(f'{dataset}: missing spatial or timing data')

    # One observation per sample/category: avoid pseudo-replication from the
    # many source-target-metric combinations within one injection experiment.
    pair = (spatial.groupby(['load', 'sample', 'adjacent'], as_index=False)
            .delta_r2.median())
    wide = pair.pivot(index=['load', 'sample'], columns='adjacent', values='delta_r2').dropna()
    if wide.empty:
        raise ValueError(f'{dataset}: no paired adjacent/nonadjacent samples')
    edge = wide[True].to_numpy(float)
    control = wide[False].to_numpy(float)

    # Fix a single cohort of requests that reach the deepest commonly observed
    # service depth. Forward-fill missing intermediate service depths using
    # that request's last completed level, then aggregate within sample and
    # across samples. This preserves the within-request paired comparison.
    sample_depth = (timing.groupby(['load', 'sample']).depth.max()
                    .reset_index(name='max_depth'))
    eligible = [(d, int((sample_depth.max_depth >= d).sum()))
                for d in sorted(timing.depth.unique())]
    deepest = max(d for d, n in eligible if n >= 20)
    trace_depth = (timing.groupby(['load', 'sample', 'trace_id']).depth.max()
                   .reset_index(name='max_depth'))
    keep = trace_depth.loc[trace_depth.max_depth >= deepest,
                           ['load', 'sample', 'trace_id']]
    cohort = timing.merge(keep, on=['load', 'sample', 'trace_id'])
    pivot = cohort.pivot_table(index=['load', 'sample', 'trace_id'],
                               columns='depth', values='cumulative_s', aggfunc='max')
    pivot = pivot.reindex(columns=range(int(deepest) + 1)).ffill(axis=1)
    by_sample = pivot.groupby(level=['load', 'sample']).median()
    summary = pd.DataFrame({
        'depth': by_sample.columns.astype(int),
        'median': by_sample.median(axis=0).to_numpy(float),
        'q25': by_sample.quantile(.25, axis=0).to_numpy(float),
        'q75': by_sample.quantile(.75, axis=0).to_numpy(float),
    })
    n_cohort = len(by_sample)

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), layout='constrained')
    ax = axes[0]
    bp = ax.boxplot([control, edge], positions=[0, 1], widths=.44,
                    patch_artist=True, showfliers=False,
                    medianprops={'color': '#202836', 'linewidth': 2})
    for patch, color in zip(bp['boxes'], ['#96b2c6', '#eb9b69']):
        patch.set_facecolor(color); patch.set_alpha(.85)
    # Light, deterministic per-sample paired connectors.
    ax.axhline(0, color='#333', lw=.8, ls='--')
    ax.set_xticks([0, 1], ['Nonadjacent', 'Trace adjacent'])
    ax.set_ylabel('Added test $R^2$ from source metric')
    ax.set_title('(a) Weak local temporal predictability')
    ax.grid(axis='y', alpha=.2)
    ax.text(.02, .04,
            f'Paired samples: {len(wide)}\n'
            f'Medians: {np.median(control):+.4f} / {np.median(edge):+.4f}\n'
            f'Edge − control: {np.median(edge-control):+.4f}',
            transform=ax.transAxes, va='bottom', ha='left', fontsize=9,
            bbox=dict(boxstyle='round,pad=.3', fc='white', ec='#dedede', alpha=.95))

    ax = axes[1]
    x = summary.depth.to_numpy(int)
    ax.fill_between(x, summary.q25.to_numpy(float), summary.q75.to_numpy(float),
                    color='#6073aa', alpha=.23, label='Across-sample IQR')
    ax.plot(x, summary['median'].to_numpy(float), marker='o', color='#445d9c',
            lw=2.1, label='Across-sample median')
    ax.set_xticks(x)
    ax.set_xlabel('Call-chain depth (first service occurrence)')
    ax.set_ylabel('Cumulative request execution time (s)')
    ax.set_title(f'(b) Execution delay along deep paths (n={n_cohort})')
    ax.grid(axis='y', alpha=.2)
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    n_ok = int((audit.status == 'ok').sum())
    fig.suptitle(f'{dataset}: trace-grounded spatial and temporal evidence '
                 f'({n_ok} fault samples)', fontsize=12)
    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / f'{dataset}_spatial_lag'
    fig.savefig(stem.with_suffix('.png'), dpi=220)
    fig.savefig(stem.with_suffix('.pdf'))
    plt.close(fig)
    return (f'{dataset}: samples={len(wide)}, median ΔR² '
            f'nonedge={np.median(control):+.4f}, edge={np.median(edge):+.4f}, '
            f'paired difference={np.median(edge-control):+.4f}; '
            f'deep-path samples={n_cohort}, depth={deepest}, '
            f'first/deep median={summary.iloc[0]["median"]:.1f}/'
            f'{summary.iloc[-1]["median"]:.1f}s')


def main() -> None:
    lines = [make_figure(dataset) for dataset in ('MARBLEBench', 'MDOC')]
    (DATA / 'spatial_lag_summary.txt').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
