"""Convert the raw MicroARCL "agent-network" dataset (marble service dataset) into the preprocessed .pkl format expected by LagRCA:"""
import os
import glob
import pickle
import re
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

RAW_ROOT = os.environ.get(
    "MARBLE_RAW_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "data", "MARBLEBench"),
)
ABNORMAL_ROOT = os.path.join(RAW_ROOT, "abnormal")
NORMAL_DIR = os.environ.get("MARBLE_NORMAL_DIR", os.path.join(RAW_ROOT, "normal"))


def abnormal_sample_dirs():
    """Fault samples in grouped or flat load directories.

    Ignore backup directories such as "load-5 copy" to avoid duplicate cases.
    Returns (group, load, name, path).
    """
    out = []
    for group in ("single", "multi"):
        for load_dir in sorted(glob.glob(os.path.join(ABNORMAL_ROOT, group, "load-*"))):
            if not os.path.isdir(load_dir) or not re.fullmatch(r"load-\d+(?:-\d+)?", os.path.basename(load_dir)):
                continue
            load = os.path.basename(load_dir)
            for d in sorted(glob.glob(os.path.join(load_dir, "*"))):
                if os.path.isdir(d):
                    out.append((group, load, os.path.basename(d), d))
    for load_dir in sorted(glob.glob(os.path.join(ABNORMAL_ROOT, "load-*"))):
        load = os.path.basename(load_dir)
        if not os.path.isdir(load_dir) or not re.fullmatch(r"load-\d+(?:-\d+|-single)?", load):
            continue
        group = "single" if load.endswith("-single") else "multi"
        for d in sorted(glob.glob(os.path.join(load_dir, "*"))):
            if os.path.isdir(d):
                out.append((group, load, os.path.basename(d), d))
    return out


def normal_sample_dirs():
    """Normal collections directly under normal/ or under normal/load-*/."""
    dirs = []
    for d in sorted(glob.glob(os.path.join(NORMAL_DIR, "*"))):
        if not os.path.isdir(d):
            continue
        if os.path.isdir(os.path.join(d, "agent-network", "metrics")):
            dirs.append(d)
        elif re.fullmatch(r"load-\d+(?:-\d+)?", os.path.basename(d)):
            dirs.extend(sorted(p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)))
    return dirs
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "marble")

SERVICES = [
    "agent-network-marble-web",
    "agent-network-planner", "agent-network-summarizer",    
    'agent-network-marble-coding',
    'agent-network-marble-database',
    'agent-network-marble-minecraft',
    'agent-network-marble-research',
    'agent-network-marble-werewolf',
    'agent-network-marble-world',
]
SERVICES_BY_LEN = sorted(SERVICES, key=len, reverse=True)

NODES = ["node-5", "node-6", "node-7", "node-15", "node-171",
         "node-218", "node-219", "node-221", "node-227"]

SVC_METRICS = ["cpu_usage", "cpu_limit", "mem_usage", "mem_usage_rate", "mem_limit",
               "fs_usage", "fs_write", "fs_read", "net_receive", "net_trainsmit"]
POD_METRICS = ["cpu", "memory", "network"]
NODE_METRICS = ["cpu", "memory", "network_x", "network_y"]

ABSENT = np.nan            # marker while building; becomes -1 after scaling
ABSENT_VALUE = -1.0        # final sentinel for "pod absent"
ABSENT_MIN_RUN = 10        # >= this many leading/trailing zero rows => born-late / died-early


def _match_service(name):
    for s in SERVICES_BY_LEN:
        if name == s or name.startswith(s + "-") or name.startswith(s + "_"):
            return s
    return None


def _read_csv(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    return None if df.empty else df


# ---------------------------------------------------------------------------
# pod-tier: load per-service pod metric arrays, classify, produce slots
# ---------------------------------------------------------------------------
def _load_pod_arrays(inst, index):
    """Return {service: {podname: {metric: np.array aligned to `index`}}}."""
    out = defaultdict(lambda: defaultdict(dict))
    for col in inst.columns:
        m = re.match(r"^(.*)_(cpu|memory|network)$", col)
        if not m:
            continue
        pod, metric = m.group(1), m.group(2)
        service = _match_service(pod)
        if service is None:
            continue
        arr = pd.to_numeric(inst[col], errors="coerce").reindex(index).values.astype(float)
        out[service][pod][metric] = arr
    return out


def _pod_active_info(pod_metrics, T):
    """active mask + leading/trailing absent run lengths for one pod."""
    act = np.zeros(T, dtype=bool)
    for metric in POD_METRICS:
        arr = pod_metrics.get(metric)
        if arr is not None:
            act |= (np.nan_to_num(arr, nan=0.0) != 0)
    if act.any():
        first = int(np.argmax(act))
        last = int(T - 1 - np.argmax(act[::-1]))
    else:
        first, last = T, -1
    lead = first
    trail = (T - 1 - last) if last >= 0 else T
    return act, first, last, lead, trail


def _build_service_slots(pod_map, T):
    """
    pod_map: {podname: {metric: np.array}} for ONE service.
    Returns list of slots; each slot is {metric: np.array} (absent regions = np.nan).
    """
    pods = list(pod_map.keys())
    if len(pods) == 0:
        return []
    if len(pods) == 1:
        p = pods[0]
        return [{m: np.nan_to_num(pod_map[p].get(m, np.zeros(T)), nan=0.0) for m in POD_METRICS}]

    info = {p: _pod_active_info(pod_map[p], T) for p in pods}
    kill_replace = any(max(info[p][3], info[p][4]) >= ABSENT_MIN_RUN for p in pods)

    if not kill_replace:
        # concurrent replicas -> average into a single slot
        slot = {}
        for m in POD_METRICS:
            arrs = [np.nan_to_num(pod_map[p].get(m, np.zeros(T)), nan=0.0) for p in pods]
            slot[m] = np.mean(arrs, axis=0)
        return [slot]

    # kill-replace -> keep separate slots, order by first-active time
    order = sorted(pods, key=lambda p: (info[p][1], p))
    slots = []
    for p in order:
        _, first, last, lead, trail = info[p]
        slot = {}
        for m in POD_METRICS:
            arr = np.array(pod_map[p].get(m, np.zeros(T)), dtype=float)
            arr = np.nan_to_num(arr, nan=0.0)
            if lead > 0:
                arr[:first] = ABSENT           # born late -> absent before birth
            if last >= 0 and trail > 0:
                arr[last + 1:] = ABSENT         # died early -> absent after death
            slot[m] = arr
        slots.append(slot)
    return slots


def compute_max_pods():
    """Global pass: max #slots any single collection needs per service."""
    maxp = {s: 1 for s in SERVICES}
    dirs = [p for _, _, _, p in abnormal_sample_dirs()]
    dirs += normal_sample_dirs()
    for d in dirs:
        inst = _read_csv(os.path.join(d, "agent-network", "metrics", "instance.csv"))
        if inst is None or "timestamp" not in inst.columns:
            continue
        inst = inst.set_index("timestamp").sort_index()
        T = len(inst)
        pods = _load_pod_arrays(inst, inst.index)
        for s, pm in pods.items():
            slots = _build_service_slots(pm, T)
            maxp[s] = max(maxp[s], len(slots))
    return maxp


MAX_PODS = compute_max_pods()


def build_all_enum():
    enum, i = {}, 0
    for s in SERVICES:
        enum[s] = i; i += 1
    for s in SERVICES:
        for k in range(MAX_PODS[s]):
            enum[f"{s}-pod{k}"] = i; i += 1
    for n in NODES:
        enum[n] = i; i += 1
    return enum


ALL_ENUM = build_all_enum()


# ---------------------------------------------------------------------------
# merged dataframe per collection
# ---------------------------------------------------------------------------
def load_merged_df(case_root):
    metrics_dir = os.path.join(case_root, "agent-network", "metrics")
    svc = _read_csv(os.path.join(metrics_dir, "svc_metric.csv"))
    inst = _read_csv(os.path.join(metrics_dir, "instance.csv"))
    node = _read_csv(os.path.join(case_root, "node", "node.csv"))
    if svc is None or inst is None or node is None:
        return None
    if not ({"timestamp"} <= set(svc.columns) and {"timestamp"} <= set(inst.columns)
            and {"timestamp"} <= set(node.columns)):
        return None

    svc = svc.set_index("timestamp").sort_index()
    inst = inst.set_index("timestamp").sort_index()
    node = node.set_index("timestamp").sort_index()
    index = svc.index
    T = len(index)

    cols = {}

    # service tier
    for col in svc.columns:
        if "&" not in col:
            continue
        service, metric = col.split("&", 1)
        if service in SERVICES and metric in SVC_METRICS:
            cols[f"{metric}&{service}"] = pd.to_numeric(svc[col], errors="coerce").values

    # pod tier
    pod_map = _load_pod_arrays(inst, index)
    for s in SERVICES:
        slots = _build_service_slots(pod_map.get(s, {}), T)
        for k in range(MAX_PODS[s]):
            if k < len(slots):
                for m in POD_METRICS:
                    cols[f"{m}&{s}-pod{k}"] = slots[k][m]
            else:
                for m in POD_METRICS:              # fully-absent slot
                    cols[f"{m}&{s}-pod{k}"] = np.full(T, ABSENT)

    # node tier
    for col in node.columns:
        m = re.match(r"^\(node\)(node-\d+)_(cpu|memory|network_x|network_y)$", col)
        if not m:
            continue
        nd, metric = m.group(1), m.group(2)
        if nd in NODES:
            cols[f"{metric}&{nd}"] = pd.to_numeric(node[col], errors="coerce").reindex(index).values

    merged = pd.DataFrame(cols, index=index)

    # ensure every all_enum instance / metric column exists
    for s in SERVICES:
        for met in SVC_METRICS:
            c = f"{met}&{s}"
            if c not in merged.columns:
                merged[c] = 0.0
    for s in SERVICES:
        for k in range(MAX_PODS[s]):
            for met in POD_METRICS:
                c = f"{met}&{s}-pod{k}"
                if c not in merged.columns:
                    merged[c] = np.full(T, ABSENT)
    for n in NODES:
        for met in NODE_METRICS:
            c = f"{met}&{n}"
            if c not in merged.columns:
                merged[c] = 0.0

    # fill service / node NaNs (not pod-absent) with ffill/0; keep pod ABSENT as NaN
    pod_cols = [c for c in merged.columns if re.search(r"-pod\d+$", c.split("&", 1)[1])]
    other_cols = [c for c in merged.columns if c not in pod_cols]
    merged[other_cols] = merged[other_cols].apply(pd.to_numeric, errors="coerce").ffill().bfill().fillna(0.0)

    merged = merged.reset_index().rename(columns={merged.index.name or "index": "timestamp"})
    if merged.columns[0] != "timestamp":
        merged = merged.rename(columns={merged.columns[0]: "timestamp"})
    return merged


# ---------------------------------------------------------------------------
# adjacency: service <-> its pod slots, pod <-> node (union placement), self loops
# ---------------------------------------------------------------------------
def service_node_edges():
    """Union of nodes each service's pods were ever placed on (from graph.csv)."""
    svc2nodes = defaultdict(set)
    graph_files = [os.path.join(p, "agent-network", "metrics", "graph.csv")
                   for _, _, _, p in abnormal_sample_dirs()]
    graph_files += [os.path.join(p, "agent-network", "metrics", "graph.csv")
                    for p in normal_sample_dirs()]
    for gf in graph_files:
        try:
            g = pd.read_csv(gf)
        except Exception:
            continue
        if not {"source", "destination"}.issubset(g.columns):
            continue
        for src, dst in zip(g["source"].astype(str), g["destination"].astype(str)):
            if dst.startswith("node-") and dst in NODES:
                s = _match_service(src)
                if s is not None:
                    svc2nodes[s].add(dst)
    return svc2nodes


def build_adj():
    n = len(ALL_ENUM)
    A = np.zeros((n, n), dtype=np.float32)
    idx = ALL_ENUM
    for k in idx:
        A[idx[k], idx[k]] = 1.0
    # service <-> its pod slots
    for s in SERVICES:
        for k in range(MAX_PODS[s]):
            i, j = idx[s], idx[f"{s}-pod{k}"]
            A[i, j] = A[j, i] = 1.0
    # pod <-> node (union placement across all collections) -- faithful reproduction
    svc2nodes = service_node_edges()
    for s in SERVICES:
        for nd in svc2nodes.get(s, ()):
            for k in range(MAX_PODS[s]):
                i, j = idx[f"{s}-pod{k}"], idx[nd]
                A[i, j] = A[j, i] = 1.0
    return torch.tensor(A, dtype=torch.float32)


# ---------------------------------------------------------------------------
# normalization: per-column for svc/node; shared per (service,metric) for pods
# ---------------------------------------------------------------------------
def _pod_group(col):
    metric, inst = col.split("&", 1)
    m = re.match(r"^(.*)-pod\d+$", inst)
    return (m.group(1), metric) if m else None


def fit_scaler(normal_df):
    feat = list(normal_df.columns[1:])
    mn, rng = {}, {}
    groups = defaultdict(list)
    for c in feat:
        g = _pod_group(c)
        if g is not None:
            groups[g].append(c)
    # pod columns: shared stats per (service, metric)
    for g, gcols in groups.items():
        vals = normal_df[gcols].values.astype(float)
        vals = vals[~np.isnan(vals)]
        lo = float(np.min(vals)) if vals.size else 0.0
        hi = float(np.max(vals)) if vals.size else 1.0
        r = (hi - lo) or 1.0
        for c in gcols:
            mn[c], rng[c] = lo, r
    # non-pod columns: per-column
    for c in feat:
        if c in mn:
            continue
        col = normal_df[c].values.astype(float)
        col = col[~np.isnan(col)]
        lo = float(np.min(col)) if col.size else 0.0
        hi = float(np.max(col)) if col.size else 1.0
        mn[c], rng[c] = lo, (hi - lo) or 1.0
    return feat, mn, rng


def apply_scaler(df, feat, mn, rng):
    df = df.copy()
    for c in feat:
        df[c] = (df[c].astype(float) - mn[c]) / rng[c]
    pod_cols = [c for c in feat if _pod_group(c) is not None]
    other = [c for c in feat if c not in pod_cols]
    if pod_cols:
        df[pod_cols] = df[pod_cols].fillna(ABSENT_VALUE)   # absent -> -1
    if other:
        df[other] = df[other].fillna(0.0)
    return df


# ---------------------------------------------------------------------------
def build_normal():
    frames = []
    for rd in normal_sample_dirs():
        df = load_merged_df(rd)
        if df is not None and len(df):
            frames.append(df)
            print(f"  [normal] {os.path.basename(rd)}: {df.shape}")
    if not frames:
        raise RuntimeError("No normal runs loaded.")
    return pd.concat(frames, axis=0, ignore_index=True)


def build_cases():
    cases = []
    for group, load, name, cd in abnormal_sample_dirs():
        service = _match_service(name)
        if service is None:
            continue
        df = load_merged_df(cd)
        if df is None or len(df) == 0:
            print(f"  [skip] {group}/{load}/{name}: no metrics")
            continue
        case_id = f"{group}/{load}/{name}"   # group/load/name to enable breakdowns
        cases.append([df, [service], case_id])
        print(f"  [case] {case_id}: {df.shape} label={service}")
    return cases


def main():
    if not os.path.isdir(NORMAL_DIR):
        raise FileNotFoundError(
            f"Normal-run directory not found: {NORMAL_DIR}. "
            "Set MARBLE_NORMAL_DIR to the MARBLEBench normal data directory."
        )
    os.makedirs(OUT_DIR, exist_ok=True)
    n_pod = sum(MAX_PODS[s] for s in SERVICES)
    print(f"MAX_PODS per service (>1 means kill-replace observed): "
          f"{ {s: MAX_PODS[s] for s in SERVICES if MAX_PODS[s] > 1} }")
    print(f"all_enum: {len(ALL_ENUM)} instances "
          f"({len(SERVICES)} services + {n_pod} pods + {len(NODES)} nodes)")

    print("Building adjacency ...")
    adj = build_adj()
    print(f"  adj shape {tuple(adj.shape)}, edges={int(adj.sum().item())}")

    print("Building normal_data ...")
    normal = build_normal()
    print(f"  normal_data shape {normal.shape}")

    print("Building case_data ...")
    cases = build_cases()
    print(f"  {len(cases)} cases")
    if not cases:
        raise RuntimeError(f"No abnormal cases loaded from {ABNORMAL_ROOT}")

    print("Normalizing ...")
    feat, mn, rng = fit_scaler(normal)
    normal = apply_scaler(normal, feat, mn, rng)
    cases = [[apply_scaler(df, feat, mn, rng), label, ts] for df, label, ts in cases]
    v = normal[feat].values
    print(f"  normal range after scaling: [{np.nanmin(v):.3f}, {np.nanmax(v):.3f}] "
          f"(absent sentinel = {ABSENT_VALUE})")

    with open(os.path.join(OUT_DIR, "adj.pkl"), "wb") as f:
        pickle.dump(adj, f)
    with open(os.path.join(OUT_DIR, "normal_data.pkl"), "wb") as f:
        pickle.dump(normal, f)
    with open(os.path.join(OUT_DIR, "case_data.pkl"), "wb") as f:
        pickle.dump(cases, f)
    with open(os.path.join(OUT_DIR, "all_enum.json"), "w") as f:
        json.dump(ALL_ENUM, f, indent=2)

    print("\nALL_ENUM =", ALL_ENUM)
    print(f"\nWrote outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
