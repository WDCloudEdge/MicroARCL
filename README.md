# MicroARCL

**Root Cause Localization for Microservice-based Agent Systems**

[中文](README_zh.md) · English

MicroARCL is an unsupervised root cause localization method for microservice-based agent systems. This repository accompanies the paper *Root Cause Localization for Microservice-based Agent Systems*, which also introduces **MicroASBench**, a benchmark for studying failures in deployable agent services.

## Paper overview

Agent services invoke different tools and other services for different requests. Their execution paths are dynamic, observations are sparse and heterogeneous, and stages finish asynchronously. During a failure, shared load changes and delays along a request path can make affected services look more suspicious than the actual root cause. The paper characterizes these behaviors with MicroASBench and develops MicroARCL to rank root-cause services from monitoring data and real request paths, without historical failure labels, model training, or manually supplied failure windows.

## MicroASBench benchmark

MicroASBench combines deployable agent services, configurable workloads and replica settings, controlled failure injection, and aligned observations with root-cause labels for evaluation.

- **Systems:** 13 services in MDOC, a multimodal document assistant, and 9 services in MAR, adapted from MARBLE. They run alongside 81 conventional microservices in separate namespaces.
- **Workloads:** simple and composed agent tasks with request-dependent service activation; failure experiments use 3 or 5 concurrent users and single- or multi-replica deployments.
- **Faults:** CPU stress, memory stress, network delay, pod failure, and pod kill. The paper reports 240 failure runs across MDOC and MAR.
- **Observations:** service, instance, and node metrics sampled every 5 seconds, container logs, and request-level execution graphs. The graphs preserve service order, stage duration, tool invocations, and other agent-execution details. Failure labels are used for evaluation, not localization.

### Dataset

The dataset release information will be added here.

| Item | MDOC | MAR |
| --- | --- | --- |
| Download | To be added | To be added |
| Data format and schema | To be added | To be added |
| Preparation instructions | To be added | To be added |

## MicroARCL framework

MicroARCL takes telemetry and request-level execution paths as input and returns a ranked list of root-cause services:

1. **Agent-aware candidate detection.** Align metric response lag with service QPS, infer an adaptive anomaly window, and use BIRCH with request gating and sparsity-aware confidence to retrieve likely root-cause candidates.
2. **Reliability-weighted metric voters.** Compare each candidate's failure rate, latency, CPU, memory, network, and QPS with its own normal prefix. Weight each voter by the strength and concentration of its evidence, reducing the influence of weak or shared changes across services.
3. **Request-level lag correction.** Use observed request paths to estimate delay accumulated at downstream services. Discount evidence explained by this lag and rerank the candidates.

## Main experimental results

The paper evaluates MicroARCL against five unsupervised root cause localization baselines. On the two MicroASBench agent-service datasets, it reports:

| Dataset | MicroARCL ACC@1 | Strongest baseline ACC@1 | MicroARCL MRR | Mean localization time |
| --- | ---: | ---: | ---: | ---: |
| MDOC | 0.500 | 0.333 (MicroRCA) | 0.634 | 5.314 s |
| MAR | 0.492 | 0.283 (MicroRCA) | 0.680 | 0.648 s |
| Average across MDOC and MAR | **0.496** | **0.308** | **0.657** | — |

The average ACC@1 gain over the strongest baseline is **18.8 percentage points**. Ablation results show that candidate detection, reliability weighting, and request-level lag correction each contribute to accuracy. The paper also evaluates conventional microservice datasets (SockShop, Online Boutique, and TrainTicket), where MicroARCL achieves ACC@1 scores of 0.978, 0.833, and 0.456, respectively.

## Future research

The paper identifies broader evaluation as the next step: extend MicroASBench with more agent applications and agent-specific fault types, and test localization on production failures.

## License

This repository is licensed under the [Apache License 2.0](LICENSE).
