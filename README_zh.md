# MicroARCL

**面向微服务化 Agent 系统的故障根因定位**

中文 · [English](README.md)

MicroARCL 是一种面向微服务化 Agent 系统的无监督故障根因定位方法。本仓库对应论文 *Root Cause Localization for Microservice-based Agent Systems*。论文同时提出 **MicroASBench**，用于研究Agent 服务中的故障。

## 论文简介

Agent 服务会根据不同请求调用不同工具和服务，因此执行路径动态变化，观测数据稀疏且异构，各阶段也会异步完成。发生故障时，共享负载变化以及沿请求路径累积的延迟，可能让受影响的下游服务看起来比真正的根因更可疑。论文通过 MicroASBench 分析这些特征，并提出 MicroARCL，利用监控数据和真实请求路径对根因服务排序；定位过程无需历史故障标签、模型训练，也无需人工提供故障时间窗口。

## MicroASBench 基准

MicroASBench 将可部署的 Agent 服务、可配置的工作负载与副本设置、受控故障注入，以及带有评估标签的对齐观测数据结合起来。

- **系统：**MDOC 多模态文档助手包含 13 个服务；由 MARBLE 改造的 MAR 包含 9 个服务。它们与 81 个传统微服务在独立命名空间中共同运行。
- **工作负载：**包含简单任务和组合任务，服务激活取决于具体请求；故障实验采用 3 或 5 个并发用户，以及单副本或多副本部署。
- **故障：**CPU 压力、内存压力、网络延迟、Pod 故障和 Pod 删除。论文报告了 MDOC 与 MAR 共 240 次故障运行。
- **观测数据：**每 5 秒采样的服务、实例和节点指标，容器日志，以及请求级执行图。执行图保留服务顺序、阶段耗时、工具调用等 Agent 执行细节。故障标签仅用于评估，不参与定位。

### k8s基础环境搭建

`benchmark/k8s-base-deploy`中包含了81个混合部署的传统微服务系统（包含bookinfo、sockshop、onlineBoutique、trainticket），以及istio、prometheus、jaeger等基础服务管理和监控设施

### 多智能体服务基础调度及执行框架镜像构建及运行

多智能体服务化的基础框架，必须在一台宿主机上先行启动以支撑整个k8s集群多智能体服务调度、协同、文件存储和监控

详见 [benchmark/scheduler/README.md](benchmark/scheduler/README.md)

### 负载注入、故障注入和数据收集

需要设置的环境变量:

1. `$HOST_IP$`：多智能体服务基础调度及执行框架所在宿主机IP

MDOC:
`benchmark/failure_injection/failure_injection_MDOC/run_all_services.sh`

MAR:
`benchmark/failure_injection/failure_injection_MARBLEbench/run_all_services.sh`

单服务故障：
`benchmark/failure_injection/failure_injection_{DATASET}/sum_chaos_cloud_{service}.sh`

脚本会自动完成包括MDOC或MAR多智能体服务集群启动、locust负载注入、chaos_mesh故障yaml注入和`benchmark/data-collector` metrics、logs和执行图数据收集全过程，收集到的数据位于`benchmark/data`中，与下文已收集并公开的数据集格式一致。

需要替换或设置的环境变量：

- `benchmark/failure_injection/failure_injection_MDOC/deployments.yaml`
- `benchmark/failure_injection/failure_injection_MARBLEbench/deployments.yaml`

1. `$HOST_IP$`：多智能体服务基础调度及执行框架所在宿主机IP
2. `$OPENAI_API_KEY$`：大模型API KEY

可选

3. 大模型BaseUrl和具体模型，当前使用为qwen系列模型和阿里云平台base url


### 数据集

MDOC 和 MAR 下载地址: https://huggingface.co/datasets/Zhuyuhan2333/MicroARCL
SS TT OB下载地址: https://github.com/phamquiluan/RCAEval


## MicroARCL 框架

MicroARCL 以监控数据和请求级执行路径为输入，输出根因服务排序：

1. **Agent 感知的候选检测。**将指标响应相对服务 QPS 的延迟对齐，自适应确定异常时间窗口，再结合 BIRCH、请求活跃门控和稀疏性感知置信度筛选候选根因服务。
2. **可靠性加权的指标投票。**将候选服务的故障率、延迟、CPU、内存、网络和 QPS 与该服务自身的正常前缀比较，并根据证据强度及其在候选服务间的集中程度确定权重，降低弱证据和跨服务共同变化的影响。
3. **请求级延迟校正。**根据真实请求路径估计下游服务累积的延迟，扣减可由这种延迟解释的证据，再对候选服务重新排序。

## 核心实验结果

论文将 MicroARCL 与五种无监督故障根因定位基线比较。在 MicroASBench 的两个 Agent 服务数据集上，结果如下：

| 数据集 | MicroARCL ACC@1 | MicroARCL MRR | 平均定位耗时 |
| --- | ---: | ---: | ---: |
| MDOC | 0.500 | 0.634 | 5.314 秒 |
| MAR | 0.492 | 0.680 | 0.648 秒 |
| MDOC 与 MAR 平均 | **0.496** | **0.657** | — |

相较最强基线，平均 ACC@1 **提高 18.8 个百分点**。消融实验表明，候选检测、可靠性加权和请求级延迟校正都对定位准确率有贡献。论文还在传统微服务数据集 SockShop、Online Boutique 和 TrainTicket 上评估了 MicroARCL，其 ACC@1 分别为 0.978、0.833 和 0.456。

## 论文复现（RQ1–RQ7）

在仓库根目录运行，使用 Python 3.10。

```bash
python3.10 -m venv .venv
./.venv/bin/python -m pip install -r requirements.txt
```

数据：[MDOC/MAR](https://huggingface.co/datasets/Zhuyuhan2333/MicroARCL) → `data/MDOC/`、`data/MARBLEBench/`；[RCAEval](https://github.com/phamquiluan/RCAEval) → `data/RCAEval/`。数据集名称：`MAR` = `MARBLEBench`；`SS`/`OB`/`TT` = `re2ss`/`re2ob`/`re2tt`。

### RQ1 脚本

```text
analysis/motivation_analysis.py
analysis/chain_analysis.py
analysis/multi_replica_recheck.py
analysis/marblebench_stats.py
```

```bash
./.venv/bin/python analysis/marblebench_stats.py
```

### RQ2 脚本

```text
analysis/diag_common_mode.py
analysis/diag_temporal.py
analysis/prepare_spatial_lag.py
analysis/plot_spatial_lag.py
```

RQ1 和 RQ2 运行入口：

```bash
./.venv/bin/python analysis/run_all.py
```

输出：`analysis/figures/`、`analysis/tables/`、`data/<dataset>/diag_common_mode_*.csv`。

### RQ3、RQ4、RQ6 脚本

```text
experiments/run_microarcl.py                 # 驱动（MicroARCL）
experiments/run_baselines.py                 # 驱动（对比方法）
experiments/microarcl/                       # MicroARCL runner 入口脚本
experiments/baselines/{torai,microrca,causalrca}/   # 各对比方法 runner
```

```bash
./.venv/bin/python experiments/run_microarcl.py --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline TORAI --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline MicroRCA --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline CausalRCA --dataset all
./.venv/bin/python experiments/run_baselines.py --baseline CloudRanger --dataset all
```

`--dataset`：`all`、`MDOC`、`MAR`、`SS`、`OB`、`TT`。`--baseline`：`MicroRCA`（无 trace，跳过 `SS`）、`TORAI`、`CausalRCA`（DAG-GNN）、`CloudRanger`（PC）。都在同一 `.venv`；共享引擎（`baseline_common.py`、`Config.py` 等）留在仓库根，runner 自行把它加入 `sys.path`。

日志：`data/<dataset>/abnormal/`（MDOC/MAR）、`output/re2_materialized/`（SS/OB/TT）、`experiments/baselines/causalrca/{RCAEval/re2,MicroCERC}/`（CausalRCA/CloudRanger）。

### RQ5 脚本

```text
experiments/run_ablation.py
```

```bash
./.venv/bin/python experiments/run_ablation.py --dataset all
```

`--dataset`：`all`、`MDOC`、`MAR`；`--variant`：`all`、`full`、`MicroARCL-A`、`MicroARCL-W`、`MicroARCL-L`。

### RQ7 脚本

```text
experiments/run_sensitivity.py
```

```bash
./.venv/bin/python experiments/run_sensitivity.py --dataset all
```

`--dataset`：`all`、`MDOC`、`MAR`；`--sweep`：`all`、`k`、`mu`。所有实验入口均支持 `--limit N` 和 `--help`。

## 未来研究方向

计划扩展 MicroASBench，加入更多 Agent 应用和 Agent 特有的故障类型，进一步评估根因定位效果。

## 许可证

本仓库采用 [Apache License 2.0](LICENSE)。
