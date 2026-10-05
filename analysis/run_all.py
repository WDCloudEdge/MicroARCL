#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproduce every RQ1 and RQ2 result and figure of the paper.

    .venv/bin/python analysis/run_all.py

RQ1 (empirical study, paper Fig. 2 / Fig. 3) is produced in-process from the
NORMAL executions; figures land in analysis/figures/ and tables in
analysis/tables/. RQ2 (common-mode diffusion and flattened temporal ordering)
is produced by the two diagnostics over the FAILURE executions, each run as a
subprocess per dataset so the AGENT_DATASET root is fixed cleanly at import.
"""
import os
import sys
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PY = sys.executable


def _rq2(script, *args):
    """Run an RQ2 diagnostic from the repo root and stream its output."""
    cmd = [PY, os.path.join(HERE, script), *args]
    print(f"\n$ {' '.join(cmd[1:])}")
    subprocess.run(cmd, cwd=REPO, check=True)


def main():
    import motivation_analysis
    import chain_analysis
    import multi_replica_recheck

    print("=" * 70, "\n[RQ1 1/3] motivation_analysis "
          "(metric sparsity + service heterogeneity -> Fig. 2a, Fig. 3b)")
    motivation_analysis.main()
    print("=" * 70, "\n[RQ1 2/3] chain_analysis "
          "(completion time by depth + vertex sparsity/variance -> "
          "Fig. 2b, Fig. 3c)")
    chain_analysis.main()
    print("=" * 70, "\n[RQ1 3/3] multi_replica_recheck "
          "(QPS->resource metric lag, single vs multi -> Fig. 3a)")
    multi_replica_recheck.main()

    print("=" * 70, "\n[RQ2 1/3] common-mode diffusion "
          "(eta, rho_bar, severity CV, >=root count) -- S5.2.1")
    _rq2("diag_common_mode.py", "--dataset", "MDOC")
    _rq2("diag_common_mode.py", "--dataset", "MARBLEBench")

    print("=" * 70, "\n[RQ2 2/3] flattened temporal ordering "
          "(onset order, lag=0) -- S5.2.2")
    _rq2("diag_temporal.py", "--dataset", "MARBLEBench", "--from", "fault")
    _rq2("diag_temporal.py", "--dataset", "MDOC", "--loads", "load-5")

    print("=" * 70, "\n[RQ2 3/3] spatial predictability + execution-lag by depth "
          "(ridge lift, cumulative time) -- S5.2.2")
    _rq2("prepare_spatial_lag.py")
    _rq2("plot_spatial_lag.py")

    print("=" * 70, "\nALL DONE -> analysis/figures, analysis/tables, "
          "data/<dataset>/diag_common_mode_*.csv")


if __name__ == "__main__":
    main()
