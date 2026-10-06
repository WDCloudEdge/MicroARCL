# MicroARCL

**Root Cause Localization for Microservice-based Agent Systems**

[中文](README_zh.md) · English

MicroARCL is an unsupervised root cause localization method for microservice-based agent systems. This repository accompanies the paper *Root Cause Localization for Microservice-based Agent Systems*. The paper also introduces **MicroASBench** for studying failures in agent services.

## Paper overview

Agent services invoke different tools and other services for different requests. Their execution paths are dynamic, observations are sparse and heterogeneous, and stages finish asynchronously. During a failure, shared load changes and delays along a request path can make affected services look more suspicious than the actual root cause. The paper characterizes these behaviors with MicroASBench and develops MicroARCL to rank root-cause services from monitoring data and real request paths, without historical failure labels, model training, or manually supplied failure windows.

## MicroASBench benchmark

MicroASBench combines deployable agent services, configurable workloads and replica settings, controlled failure injection, and aligned observations with root-cause labels for evaluation.

- **Systems:** 13 services in MDOC, a multimodal document assistant, and 9 services in MAR, adapted from MARBLE. They run alongside 81 conventional microservices in separate namespaces.
- **Workloads:** simple and composed agent tasks with request-dependent service activation; failure experiments use 3 or 5 concurrent users and single- or multi-replica deployments.
- **Failures:** CPU stress, memory stress, network delay, pod failure, and pod deletion. The paper reports 240 failure runs across MDOC and MAR.
- **Observations:** service, instance, and node metrics sampled every 5 seconds, container logs, and request-level execution graphs. The graphs preserve service order, stage duration, tool invocations, and other agent-execution details. Failure labels are used for evaluation, not localization.

### Kubernetes base environment setup

`benchmark/k8s-base-deploy` contains 81 traditional microservices deployed together, including Bookinfo, SockShop, Online Boutique, and TrainTicket, along with infrastructure for service management and monitoring, such as Istio, Prometheus, and Jaeger.

### Build and run the multi-agent service scheduling and execution framework image

The base framework for multi-agent services must first be started on a host machine. It supports scheduling, coordination, file storage, and monitoring for multi-agent services across the Kubernetes cluster.

See [benchmark/scheduler/README.md](benchmark/scheduler/README.md) for details.

### Workload injection, failure injection, and data collection

MDOC: `benchmark/failure_injection/failure_injection_MDOC/run_all_services.sh`

MAR: `benchmark/failure_injection/failure_injection_MARBLEbench/run_all_services.sh`

Single-service failures: `benchmark/failure_injection/failure_injection_{DATASET}/sum_chaos_cloud_{service}.sh`

The scripts automatically start the MDOC or MAR multi-agent service cluster, inject workload with Locust and failures with Chaos Mesh YAML files, and collect metrics, logs, and execution graphs through `benchmark/data-collector`. The collected data is stored in `benchmark/data` and has the same format as the previously collected, publicly available datasets linked below.

Replace the placeholders in:

- `benchmark/failure_injection/failure_injection_MDOC/deployments.yaml`
- `benchmark/failure_injection/failure_injection_MARBLEbench/deployments.yaml`

1. `$HOST_IP$`: IP address of the host running the base multi-agent service scheduling and execution framework.
2. `$OPENAI_API_KEY$`: API key for the large language model.

Optional:

3. The model's base URL and specific model. The current configuration uses Qwen models and an Alibaba Cloud base URL.

### Dataset

MDOC and MAR download: https://huggingface.co/datasets/Zhuyuhan2333/MicroARCL
SS, TT, and OB download: https://github.com/phamquiluan/RCAEval

## MicroARCL framework

MicroARCL takes telemetry and request-level execution paths as input and returns a ranked list of root-cause services:

1. **Agent-aware candidate detection.** Align metric response lag with service QPS, infer an adaptive anomaly window, and use BIRCH with request gating and sparsity-aware confidence to retrieve likely root-cause candidates.
2. **Reliability-weighted metric voters.** Compare each candidate's failure rate, latency, CPU, memory, network, and QPS with its own normal prefix. Weight each voter by the strength and concentration of its evidence, reducing the influence of weak or shared changes across services.
3. **Request-level lag correction.** Use observed request paths to estimate delay accumulated at downstream services. Discount evidence explained by this lag and rerank the candidates.

## Main experimental results

The paper evaluates MicroARCL against five unsupervised root cause localization baselines. On the two MicroASBench agent-service datasets, it reports:

| Dataset | MicroARCL ACC@1 | MicroARCL MRR | Mean localization time |
| --- | ---: | ---: | ---: |
| MDOC | 0.500 | 0.634 | 5.314 s |
| MAR | 0.492 | 0.680 | 0.648 s |
| Average across MDOC and MAR | **0.496** | **0.657** | — |

The average ACC@1 gain over the strongest baseline is **18.8 percentage points**. Ablation results show that candidate detection, reliability weighting, and request-level lag correction each contribute to accuracy. The paper also evaluates conventional microservice datasets (SockShop, Online Boutique, and TrainTicket), where MicroARCL achieves ACC@1 scores of 0.978, 0.833, and 0.456, respectively.

## Reproducing the paper (RQ1–RQ7)

Run from the repository root. Use Python 3.10.

```bash
python3.10 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

Data: [MDOC/MAR](https://huggingface.co/datasets/Zhuyuhan2333/MicroARCL) → `data/MDOC/`, `data/MARBLEBench/`; [RCAEval](https://github.com/phamquiluan/RCAEval) → `data/RCAEval/`. Dataset names: `MAR` = `MARBLEBench`; `SS`/`OB`/`TT` = `re2ss`/`re2ob`/`re2tt`.

### RQ1 Scripts

```text
analysis/motivation_analysis.py
analysis/chain_analysis.py
analysis/multi_replica_recheck.py
analysis/marblebench_stats.py
```

```bash
./.venv/bin/python analysis/marblebench_stats.py
```

### RQ2 Scripts

```text
analysis/diag_common_mode.py
analysis/diag_temporal.py
analysis/prepare_spatial_lag.py
analysis/plot_spatial_lag.py
```

RQ1 and RQ2 entry point:

```bash
./.venv/bin/python analysis/run_all.py
```

Outputs: `analysis/figures/`, `analysis/tables/`, `data/<dataset>/diag_common_mode_*.csv`.

### RQ3, RQ4, and RQ6 Scripts

```text
experiments/run_microarcl.py                 # driver (MicroARCL)
experiments/run_baselines.py                 # driver (comparison methods)
experiments/microarcl/                       # MicroARCL runner entry scripts
experiments/baselines/{torai,microrca,causalrca}/   # per-baseline runners
```

```bash
./.venv/bin/python experiments/run_microarcl.py --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline TORAI --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline MicroRCA --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline CausalRCA --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline CloudRanger --dataset all
```

`--dataset`: `all`, `MDOC`, `MAR`, `SS`, `OB`, `TT`. `--baseline`: `MicroRCA` (skips `SS`, no traces), `TORAI`, `CausalRCA` (DAG-GNN), `CloudRanger` (PC). All run on the same `.venv`; the shared engine (`baseline_common.py`, `Config.py`, …) stays at the repo root and the runners add it to `sys.path`.

Logs: `data/<dataset>/abnormal/` (MDOC/MAR), `output/re2_materialized/` (SS/OB/TT), `experiments/baselines/causalrca/{RCAEval/re2,MicroCERC}/` (CausalRCA/CloudRanger).

### RQ5 Scripts

```text
experiments/run_ablation.py
```

```bash
./.venv/bin/python experiments/run_ablation.py --dataset all
```

`--dataset`: `all`, `MDOC`, `MAR`; `--variant`: `all`, `full`, `MicroARCL-A`, `MicroARCL-W`, `MicroARCL-L`.

### RQ7 Scripts

```text
experiments/run_sensitivity.py
```

```bash
./.venv/bin/python experiments/run_sensitivity.py --dataset all
```

`--dataset`: `all`, `MDOC`, `MAR`; `--sweep`: `all`, `k`, `mu`. All experiment runners accept `--limit N` and `--help`.

## Future research

Future work will extend MicroASBench with more agent applications and agent-specific failure types, and further evaluate root cause localization performance.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
