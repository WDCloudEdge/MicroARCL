#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ1 QPS->resource metric-lag re-check on the NORMAL MDOC runs: single vs multi-replica for 1/3/5 users (Fig."""
import os
import numpy as np
import pandas as pd
from common import (plt, FIG, TAB, COLORS, PALETTE, mpath, cv_series, cv_pos,
                    load_traces, trace_node_times, save_fig, scale_figure_text)

SAMPLE_SEC = 5

# (load, single-replica root, multi-replica root)
PAIRS = [
    ("1 user",
     "data/MDOC/normal/20260904-14.10-14.15-normal-thingo-1user-5min_14.20",
     "data/MDOC/normal/20260907-21.00-21.05-normal-thingo-1user-5min-multi-21.10"),
    ("3 users",
     "data/MDOC/normal/20260913-14.40-14.45-normal-thingo-3user-5min-14.50",
     "data/MDOC/normal/20260913-14.10-14.15-normal-thingo-3user-5min-multi-14.20"),
    ("5 users",
     "data/MDOC/normal/20260904-14.30-14.35-normal-thingo-5user-5min_14.50",
     "data/MDOC/normal/20260907-22.45-22.50-normal-thingo-5user-5min-multi-23.00"),
]
# flatten to labelled runs
RUNS = []
for load, s, m in PAIRS:
    RUNS.append((f"{load} single", s))
    RUNS.append((f"{load} multi", m))

REPLICATED = ["planner", "csv-gen", "excel-parsing", "word-gen", "word-parsing"]


# ---------- Q1: lag cross-correlation ----------
def ccf_curve(x, y, maxlag=12):
    x = (x - x.mean()) / (x.std() + 1e-9)
    y = (y - y.mean()) / (y.std() + 1e-9)
    n = len(x)
    lags, vals = [], []
    for L in range(-maxlag, maxlag + 1):
        a, b = (x[:n - L], y[L:]) if L >= 0 else (x[-L:], y[:n + L])
        if len(a) < 5:
            continue
        lags.append(L * SAMPLE_SEC)
        vals.append(np.corrcoef(a, b)[0, 1])
    return np.array(lags), np.array(vals)


def _total_qps_and_res(root, resp):
    q = (pd.read_csv(mpath(root, "svc_qps.csv")).drop(columns=["timestamp"])
         .apply(pd.to_numeric, errors="coerce").sum(axis=1).values)
    res = pd.read_csv(mpath(root, "resource.csv"))
    n = min(len(q), len(res))
    return q[:n], res[resp].values[:n]


# per-load MULTI runs for the lag figure (loads 1/3/5, all multi)
FIG_MULTI = {
    "1 user":  "data/MDOC/normal/20260907-21.00-21.05-normal-thingo-1user-5min-multi-21.10",
    "3 users": "data/MDOC/normal/20260913-14.10-14.15-normal-thingo-3user-5min-multi-14.20",
    "5 users": "data/MDOC/normal/20260907-22.45-22.50-normal-thingo-5user-5min-multi-23.00",
}
def lag_recheck():
    # peak-lag table: single vs multi (1 & 5 users) — feeds the recheck tables
    rows = []
    for load, s_root, m_root in PAIRS:
        for resp in ["vCPU", "memory"]:
            for tag, root in [("single", s_root), ("multi", m_root)]:
                q, r = _total_qps_and_res(root, resp)
                lags, vals = ccf_curve(q, r)
                peak = int(lags[int(np.nanargmax(vals))])
                pr = float(np.nanmax(vals))
                rows.append({"load": load, "replicas": tag, "response": resp,
                             "peak_lag_s": peak, "peak_corr": round(pr, 2)})

    # One axes: workload uses color; resource uses line style and marker.
    fig, ax = plt.subplots(figsize=(13, 8.6))
    for resp, resource, linestyle, marker in [
        ("vCPU", "CPU", "-", "o"),
        ("memory", "Memory", "--", "s"),
    ]:
        for name in ["1 user", "3 users", "5 users"]:
            q, r = _total_qps_and_res(FIG_MULTI[name], resp)
            lags, vals = ccf_curve(q, r)
            peak = int(lags[int(np.nanargmax(vals))])
            pr = float(np.nanmax(vals))
            ax.plot(lags, vals, marker=marker, ms=4, ls=linestyle,
                    lw=1.8, color=PALETTE[name],
                    label=f"{resource} {name.split()[0]}u: {peak:+d}s/{pr:.2f}")
            ax.axvline(peak, ls="--", alpha=.4, color=PALETTE[name])
            ax.plot(peak, pr, marker=marker, ms=8, color=PALETTE[name],
                    markeredgecolor="black", markeredgewidth=.8, zorder=4)
    ax.axvline(0, color=COLORS["reference"], lw=.8)
    ax.set_xlabel("Metric lag (s)")
    ax.set_ylabel("Cross-correlation")
    ax.grid(ls=":", alpha=.4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    handles, labels = ax.get_legend_handles_labels()
    ax.set_ylim(0, 1)
    ax.legend(handles, labels, loc="lower center", ncol=2,
              fontsize=8, frameon=True, facecolor="white", framealpha=.95,
              handlelength=1.2, handletextpad=.4, columnspacing=.8,
              labelspacing=.2, borderpad=.25)
    # Equal-height paper placement: match c's 20 pt text on a 4.4-inch canvas.
    scale_figure_text(fig, 2 * fig.get_figheight() / 4.4)
    fig.tight_layout()
    # Match vertical plot bounds for equal-height placement in the paper.
    fig.subplots_adjust(bottom=0.20, top=0.96)
    save_fig(fig, "recheck_lag_ccf")
    plt.close(fig)
    return pd.DataFrame(rows)


# ---------- Q2: re-run all conclusions ----------
def informative_pct(root, fn):
    df = pd.read_csv(mpath(root, fn))
    v = df[[c for c in df.columns if c != "timestamp"]].apply(pd.to_numeric, errors="coerce")
    tot = v.size
    return round(100 * (tot - int(v.isna().sum().sum()) - int((v == 0).sum().sum())) / tot, 1)


def col_cv_stats(root, fn, suffix):
    df = pd.read_csv(mpath(root, fn))
    cvs = [cv_series(df[c]) for c in df.columns if c.endswith(suffix)]
    cvs = [x for x in cvs if not np.isnan(x)]
    return (round(float(np.median(cvs)), 2), round(float(np.max(cvs)), 2)) if cvs else (np.nan, np.nan)


def compare():
    rows = []
    for label, root in RUNS:
        traces = load_traces(root)
        tt = trace_node_times(traces)
        lens = [t["total_level"] for t in traces]
        paths = [tuple(t["path"]) for t in traces if t["path"]]

        def vcv(grp):
            return round(cv_pos(tt.get(grp, [])), 2) if tt.get(grp) else np.nan

        latcv_med, latcv_max = col_cv_stats(root, "latency.csv", "&p99")
        netcv_med, netcv_max = col_cv_stats(root, "svc_metric.csv", "&net_receive")
        rows.append({
            "run": label,
            "lat_info%": informative_pct(root, "latency.csv"),
            "succ_info%": informative_pct(root, "success_rate.csv"),
            "qps_nonzero%": informative_pct(root, "svc_qps.csv"),
            "latP99_CV_med": latcv_med, "latP99_CV_max": latcv_max,
            "netRecv_CV_max": netcv_max,
            "traces": len(traces),
            "chain_len_mean": round(float(np.mean(lens)), 2),
            "chain_len_max": int(np.max(lens)),
            "distinct_paths": len(set(paths)),
            "planner_execCV": vcv("AgentNetworkPlannerGroup"),
            "ocr_execCV": vcv("OCRParserGroup"),
        })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TAB, "recheck_compare.csv"), index=False)
    return tab


# ---------- Q2b: multi-replica effect ----------
def replica_effect():
    svc_rows, cl_rows = [], []
    for label, root in RUNS:
        inum = pd.read_csv(mpath(root, "instances_num.csv"))
        res = pd.read_csv(mpath(root, "resource.csv"))
        inst = pd.read_csv(mpath(root, "instance.csv"))
        lat = pd.read_csv(mpath(root, "latency.csv"))
        for svc in REPLICATED + ["ocr", "summarizer"]:
            col = f"agent-network-{svc}&count"
            reps = int(pd.to_numeric(inum[col]).max()) if col in inum.columns else np.nan
            latcol = f"agent-network-{svc}&p99"
            info = (round(100 * (~pd.to_numeric(lat[latcol], errors="coerce").isna()).mean(), 1)
                    if latcol in lat.columns else np.nan)
            svc_rows.append({"run": label, "service": svc, "replicas": reps,
                             "lat_p99_nonnull%": info})
        cl_rows.append({"run": label,
                        "pods": len([c for c in inst.columns if c.endswith("_cpu")]),
                        "vCPU_mean": round(float(pd.to_numeric(res["vCPU"]).mean()), 2),
                        "mem_mean": round(float(pd.to_numeric(res["memory"]).mean()), 1)})
    svc = pd.DataFrame(svc_rows)
    cl = pd.DataFrame(cl_rows)
    svc.to_csv(os.path.join(TAB, "recheck_replica_effect.csv"), index=False)
    cl.to_csv(os.path.join(TAB, "recheck_cluster_resource.csv"), index=False)
    return svc, cl


def main():
    lag = lag_recheck()
    cmp = compare()
    rep, cl = replica_effect()
    print("=== Q1 lag re-check (single vs multi) ===")
    print(lag.to_string(index=False))
    print("\n=== Q2 all conclusions (single vs multi) ===")
    print(cmp.to_string(index=False))
    print("\n=== Q2b per-service replicas & latency availability ===")
    print(rep.to_string(index=False))
    print("\n=== Q2b cluster resource ===")
    print(cl.to_string(index=False))
    print("\nDONE -> recheck figures & tables")


if __name__ == "__main__":
    main()
