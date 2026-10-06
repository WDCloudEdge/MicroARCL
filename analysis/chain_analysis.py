#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Analyze call-chain completion time, depth, and invocation sparsity."""
import os
import collections
import numpy as np
import pandas as pd
from common import (plt, FIG, TAB, DATASETS, ORDER, PALETTE, COLORS,
                    load_traces, cv_pos as cv, save_fig, scale_figure_text)


# ===================================================================
# C1  completion-time accumulation along chain depth  +  dynamism
# ===================================================================
# Multi-replica 1/3/5-user runs for the completion-time-by-depth figure.
C1_MULTI = {
    "1 user (multi)":  "data/MDOC/normal/20260907-21.00-21.05-normal-thingo-1user-5min-multi-21.10",
    "3 users (multi)": "data/MDOC/normal/20260913-14.10-14.15-normal-thingo-3user-5min-multi-14.20",
    "5 users (multi)": "data/MDOC/normal/20260907-22.45-22.50-normal-thingo-5user-5min-multi-23.00",
}
C1_COLOR = {"1 user (multi)": PALETTE["1 user"], "3 users (multi)": PALETTE["3 users"],
            "5 users (multi)": PALETTE["5 users"]}

def chain_c1(data):
    c1data = {name: load_traces(root) for name, root in C1_MULTI.items()}
    c1_loads = list(C1_MULTI.keys())

    # ---- cumulative completion time vs chain depth (Fig. 3c) ----
    fig, axes = plt.subplots(1, len(c1_loads), figsize=(5.2 * len(c1_loads), 4.4), sharey=True)
    depth_tab = []
    for ax, name in zip(np.atleast_1d(axes), c1_loads):
        by_depth = collections.defaultdict(list)
        for tr in c1data[name]:
            cum = 0.0
            for d, lv in enumerate(tr["levels"]):
                cum += lv["time"]
                by_depth[d].append(cum)
        depths = sorted(by_depth.keys())
        bp = ax.boxplot([by_depth[d] for d in depths], positions=depths, widths=0.6,
                        patch_artist=True, showfliers=False)
        for patch in bp["boxes"]:
            patch.set_facecolor(C1_COLOR[name]); patch.set_alpha(.7)
        ax.text(0.5, 0.96, name.replace(" (multi)", ""),
                transform=ax.transAxes, ha="center", va="top", fontsize=12)
        ax.set_xlabel("Call chain depth")
        ax.grid(axis="y", ls=":", alpha=.5)
        for d in depths:
            v = np.array(by_depth[d])
            depth_tab.append({"load": name, "depth": d, "n": len(v),
                              "median_s": round(float(np.median(v)), 1),
                              "p90_s": round(float(np.percentile(v, 90)), 1),
                              "iqr_s": round(float(np.percentile(v, 75) - np.percentile(v, 25)), 1)})
    np.atleast_1d(axes)[0].set_ylabel("Cumulative completion time (s)")
    scale_figure_text(fig, 2)
    fig.tight_layout()
    # Match vertical plot bounds for equal-height placement in the paper.
    fig.subplots_adjust(bottom=0.20, top=0.96)
    save_fig(fig, "c1r_completion_by_depth")
    plt.close(fig)
    pd.DataFrame(depth_tab).to_csv(os.path.join(TAB, "chain_completion_by_depth.csv"), index=False)

    # ---- chain-length / path dynamism table ----
    dyn_rows = []
    for name in ORDER:
        lens = [t["total_level"] for t in data[name]]
        paths = [tuple(t["path"]) for t in data[name] if t["path"]]
        cnt = collections.Counter(paths)
        p = np.array(list(cnt.values()), float); p = p / p.sum() if p.sum() else p
        ent = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
        dyn_rows.append({"load": name, "traces": len(lens),
                         "len_mean": round(np.mean(lens), 2), "len_max": int(np.max(lens)),
                         "distinct_paths": len(set(paths)),
                         "distinct_ratio": round(len(set(paths)) / max(len(paths), 1), 2),
                         "path_entropy_bits": round(ent, 2)})
    dyn = pd.DataFrame(dyn_rows)
    dyn.to_csv(os.path.join(TAB, "chain_dynamism.csv"), index=False)
    return dyn


# ===================================================================
# C2  per-vertex sparsity  +  execution-time variance
# ===================================================================
def chain_c2(data):
    # collapse to group name (before '/') so tool = one logical service
    def grp(v):
        return v.split("/")[0]

    # ---- invocation frequency (sparsity), 5 users ----
    # Use compact display names while retaining full identifiers in the data.
    service_labels = {
        "AgentNetworkPlannerGroup": "Planner",
        "WordGenerationAgentGroup": "WordGen",
        "ExcelGroup": "Excel",
        "PdfAgentGroup": "PDF",
        "PdfGenAgentGroup": "PDFGen",
        "ExcelGenGroup": "ExcelGen",
        "DirectionAgentGroup": "Direction",
        "ImageGenAgentGroup": "ImageGen",
        "WordAgentGroup": "Word",
        "OCRParserGroup": "OCR",
        "CSVGeneratorAgentGroup": "CSVGen",
    }
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
    vtab = []
    for name in ORDER:
        cnt = collections.Counter()
        times = collections.defaultdict(list)
        n_tr = len(data[name])
        for tr in data[name]:
            seen = set()
            for lv in tr["levels"]:
                for node, t in lv["node_times"].items():
                    g = grp(node)
                    times[g].append(t)
                    if g not in seen:
                        cnt[g] += 1
                        seen.add(g)
        for g in cnt:
            vtab.append({"load": name, "vertex": g, "traces_with": cnt[g],
                         "coverage": round(cnt[g] / max(n_tr, 1), 3),
                         "invocations": len(times[g]),
                         "time_mean_s": round(float(np.mean(times[g])), 2),
                         "time_cv": round(cv(times[g]), 2)})
    vt = pd.DataFrame(vtab)
    vt.to_csv(os.path.join(TAB, "chain_vertex_stats.csv"), index=False)

    # left: coverage bars (5 users), sorted
    sub = vt[vt.load == "5 users"].sort_values("coverage", ascending=True)
    axes[0].barh(sub["vertex"], sub["coverage"] * 100, color=PALETTE["5 users"], alpha=.8)
    axes[0].set_yticks(range(len(sub)),
                      labels=[service_labels.get(v, v) for v in sub["vertex"]])
    axes[0].set_xlabel("Percentage of traces\ncovering the service (%)")
    axes[0].tick_params(axis="y", labelsize=7)
    axes[0].grid(axis="x", ls=":", alpha=.5)

    # right: per-vertex execution-time CV (box across loads)
    kinds = sub["vertex"].tolist()
    data_cv = [vt[vt.vertex == k]["time_cv"].dropna().values for k in kinds]
    data_cv = [d for d in data_cv if len(d)]
    allcv = vt.groupby("vertex")["time_cv"].mean().sort_values(ascending=True)
    axes[1].barh(allcv.index, allcv.values, color=COLORS["variance"], alpha=.8)
    axes[1].set_yticks(range(len(allcv)),
                      labels=[service_labels.get(v, v) for v in allcv.index])
    axes[1].axvline(1.0, ls="--", color=COLORS["reference"])
    axes[1].set_xlabel("Coefficient of variation\nof execution time (CV) = σ/μ\n(standard deviation/mean)")
    axes[1].tick_params(axis="y", labelsize=7)
    axes[1].grid(axis="x", ls=":", alpha=.5)
    scale_figure_text(fig, 2.8)
    fig.tight_layout(w_pad=2.0)
    save_fig(fig, "c2r_vertex_sparsity_variance")
    plt.close(fig)
    return vt


def main():
    data = {name: load_traces(root) for name, root in DATASETS.items()}
    for name in ORDER:
        print(f"{name}: {len(data[name])} traces loaded")
    dyn = chain_c1(data)
    vt = chain_c2(data)

    print("\n=== chain dynamism ==="); print(dyn.to_string(index=False))
    print("\n=== top vertices (5 users) by coverage ===")
    top = vt[vt.load == "5 users"].sort_values("coverage", ascending=False).head(15)
    print(top.to_string(index=False))
    print("\nDONE -> figures & tables updated")


if __name__ == "__main__":
    main()
