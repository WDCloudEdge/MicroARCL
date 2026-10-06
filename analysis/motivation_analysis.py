#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RQ1 metric sparsity, variance, and inter-service heterogeneity on the NORMAL MDOC runs (1/3/5 users)."""
import os
import numpy as np
import pandas as pd
from common import (plt, FIG, TAB, DATASETS, ORDER, PALETTE, COLORS, SAMPLE_SEC,
                    mpath, cv_series as cv, load_traces, trace_node_times, save_fig,
                    scale_figure_text)


# ===================================================================
# C2-A  SPARSITY
# ===================================================================
def sparsity():
    rows = []
    METRICS = {
        "latency (p50/p90/p99)": "latency.csv",
        "success_rate":          "success_rate.csv",
        "qps":                   "svc_qps.csv",
        "call (p50/p90/p99)":    "call.csv",
    }
    for name, root in DATASETS.items():
        for label, fn in METRICS.items():
            df = pd.read_csv(mpath(root, fn))
            cols = [c for c in df.columns if c != "timestamp"]
            v = df[cols].apply(pd.to_numeric, errors="coerce")
            total = v.size
            nnull = int(v.isna().sum().sum())
            nzero = int((v == 0).sum().sum())
            info = total - nnull - nzero
            rows.append({
                "load": name, "metric": label,
                "null_%": round(100 * nnull / total, 1),
                "zero_%": round(100 * nzero / total, 1),
                "informative_%": round(100 * info / total, 1),
            })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TAB, "sparsity.csv"), index=False)

    # figure: informative-rate grouped bars
    metrics = list(METRICS.keys())
    x = np.arange(len(metrics))
    w = 0.25
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, name in enumerate(ORDER):
        sub = tab[tab.load == name].set_index("metric").loc[metrics]
        ax.bar(x + (i - 1) * w, sub["informative_%"], w, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=12)
    ax.set_ylabel("Proportion of valid information (%)")
    ax.legend(title="User Workload")
    ax.grid(axis="y", ls=":", alpha=.5)
    scale_figure_text(fig, 1.4)
    fig.tight_layout()
    save_fig(fig, "c2_sparsity_bars")
    plt.close(fig)
    return tab


# ===================================================================
# C2-B  VARIANCE (coefficient of variation on normal data)
# ===================================================================
def variance():
    """Per-service coefficient of variation on NORMAL data, per metric kind.

    Feeds the paper claim that more than half of the services have a
    latency-metric CV > 0.5 (table variance_cv.csv)."""
    KINDS = {
        "CPU":         ("svc_metric.csv", "&cpu_usage"),
        "Memory":      ("svc_metric.csv", "&mem_usage"),
        "Net recv":    ("svc_metric.csv", "&net_receive"),
        "Latency p99": ("latency.csv",    "&p99"),
        "Latency p50": ("latency.csv",    "&p50"),
    }
    rows = []
    for name, root in DATASETS.items():
        for kind, (fn, suffix) in KINDS.items():
            df = pd.read_csv(mpath(root, fn))
            cols = [c for c in df.columns if c.endswith(suffix)]
            cvs = [cv(df[c]) for c in cols]
            cvs = [x for x in cvs if not np.isnan(x)]
            if cvs:
                a = np.array(cvs)
                rows.append({
                    "load": name, "metric": kind,
                    "median_CV": round(float(np.median(a)), 2),
                    "p90_CV": round(float(np.percentile(a, 90)), 2),
                    "max_CV": round(float(a.max()), 2),
                    "frac_CV>0.5": round(float((a > 0.5).mean()), 2),
                    "frac_CV>1": round(float((a > 1).mean()), 2),
                })
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TAB, "variance_cv.csv"), index=False)
    return tab


# ===================================================================
# C2-C  INTER-SERVICE HETEROGENEITY
#   Normal-state per-service CPU / memory / exec-time span 1-2 orders of
#   magnitude, and the per-metric ranking differs across services.
# ===================================================================
G2S = {"AgentNetworkPlannerGroup": "planner", "OCRParserGroup": "ocr",
       "WordGenerationAgentGroup": "word-gen", "PdfGenAgentGroup": "pdf-gen",
       "DirectionAgentGroup": "direction", "CSVGeneratorAgentGroup": "csv-gen",
       "ImageGenAgentGroup": "image-gen", "ExcelGenGroup": "excel-gen"}


def heterogeneity():
    root = DATASETS["5 users"]
    sm = pd.read_csv(mpath(root, "svc_metric.csv"))
    tt = trace_node_times(load_traces(root))
    rows = []
    for grp, svc in G2S.items():
        cpu = pd.to_numeric(sm.get(f"agent-network-{svc}&cpu_usage"), errors="coerce").mean()
        mem = pd.to_numeric(sm.get(f"agent-network-{svc}&mem_usage"), errors="coerce").mean()
        et = np.array(tt.get(grp, []), float)
        rows.append({"service": svc, "cpu_mean": round(float(cpu), 3),
                     "mem_mean_MB": round(float(mem), 0),
                     "exec_med_s": round(float(np.median(et)), 2) if len(et) else np.nan})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(TAB, "service_heterogeneity.csv"), index=False)

    # Compare different units using each metric's minimum service value as 1x.
    metrics = [("cpu_mean", "CPU", "#0072B2"),
               ("mem_mean_MB", "Memory", "#E69F00"),
               ("exec_med_s", "Execution", "#009E73")]
    from matplotlib.ticker import NullLocator
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    positions = np.arange(len(tab))
    bar_width = 0.23
    for i, (col, label, color) in enumerate(metrics):
        values = tab[col].where(tab[col] > 0)
        relative = values / values.min()
        ratio = float(relative.max())
        ax.bar(positions + (i - 1) * bar_width, relative,
                width=bar_width, color=color, alpha=.9,
                label=f"{label} (≈{ratio:.0f}×)")
    ax.set_xticks(positions, labels=[s.replace("-", "\n") for s in tab["service"]])
    ax.set_yscale("log")
    ax.set_ylim(0.8, 70)
    ax.margins(x=0.025)
    ticks = [1, 2, 5, 10, 20, 50]
    ax.set_yticks(ticks, labels=[f"{tick}×" for tick in ticks])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.axhline(1, color=COLORS["reference"], ls="--", lw=.8)
    ax.set_ylabel("Relative to metric minimum\n(log scale)")
    ax.set_axisbelow(True)
    ax.grid(axis="y", ls=":", alpha=.35)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
    ax.legend(loc="upper right", ncol=1, fontsize=10,
              frameon=True, facecolor="white", framealpha=.95,
              handlelength=1.2, labelspacing=.2, borderpad=.25)
    # Match c's font size after equal-height placement in the paper.
    scale_figure_text(fig, 2 * fig.get_figheight() / 4.4)
    fig.tight_layout()
    # Match vertical plot bounds for equal-height placement in the paper.
    fig.subplots_adjust(bottom=0.20, top=0.96)
    save_fig(fig, "c2b_service_heterogeneity")
    plt.close(fig)
    return tab


def main():
    np.random.seed(0)
    sp = sparsity()
    va = variance()
    het = heterogeneity()
    print("DONE. figures ->", FIG)
    print("tables  ->", TAB)
    print("\n=== sparsity ==="); print(sp.to_string(index=False))
    print("\n=== variance ==="); print(va.to_string(index=False))
    print("\n=== inter-service heterogeneity (5 users) ===")
    print(het.to_string(index=False))


if __name__ == "__main__":
    main()
