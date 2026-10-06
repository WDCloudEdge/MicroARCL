#!/usr/bin/env python3
"""RQ2 spatial + execution-lag evidence (paper S5.2.2) over the MicroASBench failure runs."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
ROOT = _REPO_ROOT / 'data'
OUT = _REPO_ROOT / 'analysis' / 'tables'
METRICS = ('cpu_usage', 'mem_usage', 'net_receive', 'net_trainsmit')


def service(vertex: str, dataset: str) -> str | None:
    group = vertex.split('/')[0]
    if group == 'AgentNetworkPlannerGroup':
        return 'agent-network-planner'
    if dataset == 'MARBLEBench':
        m = re.fullmatch(r'Marble(.+?)AgentGroup', group)
        return 'agent-network-marble-' + re.sub(r'(?<!^)(?=[A-Z])', '-', m[1]).lower() if m else None
    names = {
        'CSVGeneratorAgentGroup': 'csv-gen', 'DirectionAgentGroup': 'direction',
        'ExcelGenGroup': 'excel-gen', 'ExcelGroup': 'excel-parsing',
        'ImageAgentGroup': 'image', 'ImageGenAgentGroup': 'image-gen',
        'OCRParserGroup': 'ocr', 'PdfAgentGroup': 'pdf-parsing',
        'PdfGenAgentGroup': 'pdf-gen', 'WordAgentGroup': 'word-parsing',
        'WordGenerationAgentGroup': 'word-gen',
    }
    return 'agent-network-' + names[group] if group in names else None


def label_fields(sample: Path) -> dict[str, str]:
    label = sample.parent / (sample.name.split('_')[0] + '_label.txt')
    if not label.exists():
        return {}
    text = label.read_text()
    match = re.search(r'^===\s*' + re.escape(sample.name) + r'\s*===\s*$', text, re.M)
    if not match:
        return {}
    end = re.search(r'^===', text[match.end():], re.M)
    segment = text[match.end():match.end() + end.start()] if end else text[match.end():]
    return dict(re.findall(r'^([a-z_]+):\s*(.*)$', segment, re.M))


def epoch_field(fields: dict[str, str], key: str) -> int | None:
    m = re.search(r'\((\d{9,})\)', fields.get(key, ''))
    return int(m[1]) if m else None


def graph_services(path: Path, metric_services: set[str]) -> set[str]:
    names = set()
    with path.open(newline='') as handle:
        for row in csv.DictReader(handle):
            for node in (row['source'], row['destination']):
                if node.startswith('unknown_'):
                    node = node[8:]
                for svc in metric_services:
                    if node == svc or node.startswith(svc + '-'):
                        names.add(svc)
    return names


def call_edges(chains: list[dict], dataset: str, valid: set[str]) -> set[tuple[str, str]]:
    edges = set()
    for chain in chains:
        for edge in chain.get('edges', []):
            a, b = service(edge['source'], dataset), service(edge['destination'], dataset)
            if a in valid and b in valid and a != b:
                edges.add((a, b))
    return edges


def z_difference(frame: pd.DataFrame, columns: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for col in columns:
        x = pd.to_numeric(frame[col], errors='coerce').to_numpy(float)
        if np.isfinite(x).sum() < 25:
            continue
        x = pd.Series(x).interpolate(limit_direction='both').to_numpy()
        dx = np.diff(x)
        scale = np.median(np.abs(dx - np.median(dx))) * 1.4826
        if scale < 1e-8:
            scale = np.std(dx)
        if scale < 1e-8:
            continue
        out[col] = np.clip((dx - np.median(dx)) / scale, -10, 10)
    return out


def delta_r2(y: np.ndarray, own: np.ndarray, source: np.ndarray, common: np.ndarray) -> float:
    # One-step lag, chronological 70/30 split. Intercept and ridge regularizer.
    n = min(map(len, (y, own, source, common)))
    if n < 24:
        return float('nan')
    target = y[1:n]
    base = np.column_stack((np.ones(n - 1), own[:n-1], common[:n-1]))
    full = np.column_stack((base, source[:n-1]))
    split = int(len(target) * .7)
    if len(target) - split < 7:
        return float('nan')

    def mse(x: np.ndarray) -> float:
        train, test = x[:split], x[split:]
        penalty = np.eye(x.shape[1]) * 2.0
        penalty[0, 0] = 0
        beta = np.linalg.solve(train.T @ train + penalty, train.T @ target[:split])
        return float(np.mean((target[split:] - test @ beta) ** 2))

    var = float(np.var(target[split:]))
    if var < 1e-8:
        return float('nan')
    return (mse(base) - mse(full)) / var


def spatial_rows(sample: Path, dataset: str, edges: set[tuple[str, str]],
                 valid: set[str], fields: dict[str, str]) -> list[dict]:
    fault = epoch_field(fields, 'fault_start')
    end = epoch_field(fields, 'fault_end')
    if fault is None or end is None:
        return []
    mdir = sample / 'agent-network' / 'metrics'
    df = pd.read_csv(mdir / 'svc_metric.csv')
    times = pd.to_datetime(df.timestamp, utc=True).astype('int64') // 10**9
    df = df.loc[(times >= fault - 90) & (times <= end + 60)].sort_values('timestamp')
    if len(df) < 30:
        return []
    services = sorted(valid)
    values = z_difference(df, [f'{svc}&{metric}' for svc in services
                               for metric in METRICS if f'{svc}&{metric}' in df])
    qps = pd.read_csv(mdir / 'svc_qps.csv')
    qps['timestamp'] = pd.to_datetime(qps.timestamp, utc=True)
    qps = qps.set_index('timestamp').sort_index()
    qps = qps.reindex(pd.to_datetime(df.timestamp, utc=True), method='nearest', tolerance=pd.Timedelta('3s'))
    qcols = [c for c in qps if c in valid]
    if qcols:
        qarray = qps[qcols].to_numpy(float)
        count = np.isfinite(qarray).sum(axis=1)
        common = np.nansum(qarray, axis=1) / np.maximum(count, 1)
        common[count == 0] = np.nan
        common = pd.Series(common).interpolate(limit_direction='both').fillna(0).to_numpy()
        common = np.diff(common)
        common /= max(np.std(common), 1e-8)
    else:
        common = np.zeros(len(df) - 1)
    rows = []
    for a in services:
        for b in services:
            if a == b:
                continue
            adjacent = (a, b) in edges
            for metric in METRICS:
                x, y = values.get(f'{a}&{metric}'), values.get(f'{b}&{metric}')
                if x is None or y is None:
                    continue
                gain = delta_r2(y, y, x, common)
                if np.isfinite(gain):
                    rows.append(dict(dataset=dataset, load=sample.parent.name,
                                     sample=sample.name, source=a, target=b,
                                     metric=metric, adjacent=adjacent, delta_r2=gain))
    return rows


def timing_rows(sample: Path, dataset: str, valid: set[str]) -> list[dict]:
    chains_path = sample / 'agent-network' / 'graph' / 'call_chains.json'
    raw_dir = chains_path.parent / 'graph_json'
    chains = json.loads(chains_path.read_text())
    rows = []
    for trace in chains:
        raw = raw_dir / (trace['trace_id'] + '.json')
        if not raw.exists():
            continue
        try:
            detail = json.loads(raw.read_text())['level_details']
        except (KeyError, ValueError):
            continue
        elapsed = 0.0
        seen = set()
        for depth, level in enumerate(detail):
            spans = level.get('level_spans', {})
            times = [float(v.get('time', 0)) for v in spans.values()
                     if isinstance(v, dict) and np.isfinite(float(v.get('time', 0)))]
            if not times:
                continue
            # Parallel spans within a level finish when the slowest completes.
            elapsed += max(times)
            for vertex in spans:
                svc = service(vertex, dataset)
                if svc in valid and svc not in seen:
                    seen.add(svc)
                    rows.append(dict(dataset=dataset, load=sample.parent.name,
                                     sample=sample.name, trace_id=trace['trace_id'],
                                     service=svc, depth=depth, cumulative_s=elapsed))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['MARBLEBench', 'MDOC'], nargs='*',
                        default=['MARBLEBench', 'MDOC'])
    parser.add_argument('--max-samples', type=int, default=None,
                        help='development only; omit for all samples')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for dataset in args.dataset:
        paths = sorted((ROOT / dataset / 'abnormal').rglob('call_chains.json'))
        if args.max_samples is not None:
            paths = paths[:args.max_samples]
        spatial, timing, audit = [], [], []
        for i, path in enumerate(paths, 1):
            sample = path.parents[2]
            mdir = sample / 'agent-network' / 'metrics'
            try:
                fields = label_fields(sample)
                metric_names = set(c.split('&')[0] for c in
                                   pd.read_csv(mdir / 'svc_metric.csv', nrows=0).columns if '&' in c)
                valid = graph_services(mdir / 'graph.csv', metric_names)
                chains = json.loads(path.read_text())
                edges = call_edges(chains, dataset, valid)
                s = spatial_rows(sample, dataset, edges, valid, fields)
                t = timing_rows(sample, dataset, valid)
                spatial.extend(s); timing.extend(t)
                audit.append(dict(dataset=dataset, sample=sample.name, load=sample.parent.name,
                                  graph_services=len(valid), call_edges=len(edges),
                                  spatial_rows=len(s), timing_rows=len(t), status='ok'))
            except Exception as exc:
                audit.append(dict(dataset=dataset, sample=sample.name, load=sample.parent.name,
                                  graph_services=0, call_edges=0, spatial_rows=0,
                                  timing_rows=0, status=f'{type(exc).__name__}: {exc}'))
            if i % 20 == 0:
                print(f'{dataset}: {i}/{len(paths)} samples', flush=True)
        for name, data in [('spatial', spatial), ('timing', timing), ('audit', audit)]:
            pd.DataFrame(data).to_csv(OUT / f'{dataset}_{name}.csv', index=False)
        print(dataset, 'samples', len(paths), 'spatial rows', len(spatial),
              'timing rows', len(timing), 'errors', sum(x['status'] != 'ok' for x in audit))


if __name__ == '__main__':
    main()
