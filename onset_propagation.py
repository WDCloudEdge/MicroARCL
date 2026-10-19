"""Onset-propagation helpers (RQ2 motivation evidence): detect per-service
anomaly onsets over a labelled/adaptive window and classify root-vs-victim
onset order. Consumed by analysis/diag_temporal.py."""
import os
import re

import pandas as pd

import agent_rca as R
from baseline_common import _to_service
from anomaly_detection import get_anomaly_by_df
from lag_align import (align_metrics_df, compute_service_lags,
                       compute_chain_lags)
from util.utils import time_string_2_timestamp


def _label_window(kv):
    """Epoch (begin, end) from a parsed label block; None if unavailable."""
    wr = kv.get('window_range', '')
    m = re.findall(r'\d{9,}', wr)
    if len(m) >= 2:
        return int(m[0]), int(m[1])
    begin = re.search(r'\((\d{9,})\)', kv.get('window_start', ''))
    end = re.search(r'\((\d{9,})\)', kv.get('window_end', ''))
    if begin and end:
        return int(begin.group(1)), int(end.group(1))
    return None


def _label_epoch(kv, field):
    """Parenthesised epoch of a label field (e.g. fault_start); None if absent."""
    m = re.search(r'\((\d{9,})\)', kv.get(field, ''))
    return int(m.group(1)) if m else None


def _onset_floor(cfg, kv, mode):
    """Earliest timestamp an onset may be credited to, per --from mode."""
    if mode == 'load':
        ref = _label_epoch(kv, 'load_start')
        return ref
    if mode == 'fault':
        ref = _label_epoch(kv, 'fault_start')
        return None if ref is None else ref - cfg.step
    return None


def _service_onsets(cfg, sample_dir, align, since_ts=None):
    """Detect anomalies and return {service: onset_epoch} over the labelled (or,
    as fallback, adaptive) analysis window."""
    ns_dir = os.path.join(sample_dir, cfg.agent_namespace)
    metrics_dir = os.path.join(ns_dir, 'metrics')
    services = [c for c in pd.read_csv(
        os.path.join(metrics_dir, 'success_rate.csv'), nrows=1).columns
        if c != 'timestamp']

    lag_map = {}
    if align:
        lag_map = (compute_chain_lags(metrics_dir, ns_dir, cfg)
                   if getattr(cfg, 'lag_mode', 'self') == 'chain'
                   else compute_service_lags(metrics_dir, cfg))

    win = _label_window(_service_onsets._kv)
    if win is not None:
        start_ts, end_ts = win
    else:
        sr, lat, qps = R.load_kpi(metrics_dir)
        aligned_lat = align_metrics_df(lat, lag_map) if lag_map else lat
        start, end, _ = R.adaptive_window(
            sr, cfg, metrics_dir=metrics_dir, latency=aligned_lat,
            lag_map=lag_map)
        start_ts = int(time_string_2_timestamp(str(start)))
        end_ts = int(time_string_2_timestamp(str(end)))

    _anomalies, ats = get_anomaly_by_df(
        cfg, sample_dir, metrics_dir, 'onset', start_ts, end_ts,
        lag_map=lag_map)

    onsets = {}
    for node, points in ats.items():
        svc = _to_service(node, services)
        if svc is None or not points:
            continue
        ts = [time_string_2_timestamp(str(t)) for t in points]
        if since_ts is not None:
            ts = [t for t in ts if t >= since_ts]
        if not ts:
            continue
        node_onset = min(ts)
        if svc not in onsets or node_onset < onsets[svc]:
            onsets[svc] = node_onset
    return onsets


def _classify(onsets, gt, tol):
    """Per-sample onset-order classification."""
    res = {'n_anomalous': len(onsets), 'root_detected': gt in onsets,
           'root_onset': onsets.get(gt), 'category': None,
           'earliest_other_svc': None, 'earliest_other_onset': None,
           'lead_seconds': None, 'root_rank_by_onset': None,
           'deltas': []}
    if gt not in onsets:
        res['category'] = 'root_undetected'
        return res
    t_r = onsets[gt]
    others = {s: t for s, t in onsets.items() if s != gt}

    res['deltas'] = sorted((t - t_r) for t in others.values())

    res['root_rank_by_onset'] = 1 + sum(1 for t in others.values() if t < t_r)
    if not others:
        res['category'] = 'isolated'
        return res
    v, t_v = min(others.items(), key=lambda kv: kv[1])
    lead = t_v - t_r
    res['earliest_other_svc'] = v
    res['earliest_other_onset'] = t_v
    res['lead_seconds'] = lead
    if lead > tol:
        res['category'] = 'root_first'
    elif lead < -tol:
        res['category'] = 'victim_first'
    else:
        res['category'] = 'near_sync'
    return res
