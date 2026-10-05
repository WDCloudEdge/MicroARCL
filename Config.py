import time
import os
from enum import Enum


class RnnType(Enum):
    LSTM = 'lstm'
    GRU = 'gru'


class TrainType(Enum):
    TRAIN = 0
    EVAL = 1
    TRAIN_CHECKPOINT = 2


class Config:
    def __init__(self):
        # base
        self.train = TrainType.TRAIN
        self.collect = False
        self.rnn_type = RnnType.LSTM
        self.namespace = ''
        self.dataset = 'sock_shop_chaos'
        self.nodes = None
        self.svcs = set()
        self.pods = set()

        self.time_window = 20
        self.interval = self.time_window * 60
        # duration related to interval
        self.duration = self.interval
        self.window_size = 60
        self.start = int(round((time.time() - self.duration)))
        self.end = int(round(time.time()))

        # prometheus
        self.prom_range_url = "http://47.99.240.112:31444/api/v1/query_range"
        self.prom_range_url_node = "http://47.99.240.112:31222/api/v1/query_range"
        self.prom_no_range_url = "http://47.99.240.112:31444/api/v1/query"
        self.step = 5

        # jarger
        self.jaeger_url = 'http://47.99.200.176:16686/api/traces?'
        self.lookBack = str(int(self.duration / 60)) + 'm'
        self.limit = 100000

        # kubernetes
        self.k8s_config = 'local-config'

        # graph
        self.graph_min_gap = 12 * self.step

        # anomaly threshold
        self.anomaly_threshold = 0.07
        # loss threshold
        self.delta = 1e-8
        self.min_epoch = 100
        self.patience = 5

        # ===== Agent-service RCA (initial version) =====
        # single-namespace agent dataset root (a failure/normal sample directory)
        self.agent_namespace = 'agent-network'
        self.agent_sample_dir = 'data/MDOC/abnormal/agent-network-pdf-parsing_cpu_load_1'

        # --- Module B.0 label-free adaptive window (aligned KPI timeline) ---
        self.win_seg_min_len = 2        # min significant anomaly segment length (steps)
        self.win_sig_thresh = 0.15      # in-segment mean deviation significance threshold
        self.win_guard = 2 * self.step  # outer window guard band (seconds)
        # KPI baselines exclude the anomaly window and prefer its history.
        self.baseline_min_points = 3
        self.metric_spike_k = 5.0       # robust median + k*MAD threshold
        self.metric_min_rel = 0.3       # minimum relative CPU/mem/latency/QPS change
        self.availability_min_rel = 0.01  # minimum success-rate decrease
        # --- Module B.1 lag alignment (QPS <-> CPU/mem/latency/network) ---
        # Shared by adaptive window, severity, Birch, and GNN features.
        self.lag_enable = True
        # 'self': QPS->own-resource lag (compute_service_lags);
        # 'chain': call-chain propagation lag (compute_chain_lags, needs exec graph)
        self.lag_mode = 'self'
        self.lag_tau_max = 12           # cross-correlation lag search bound (steps, ~60s)
        self.lag_min_corr = 0.3         # minimum accepted QPS/metric correlation
        self.lag_min_improvement = 0.05 # required gain over zero-lag correlation
        self.lag_min_points = 12        # minimum valid paired observations
        # --- Module B.3 severity-as-GNN-prior ---
        self.severity_prior_param = 0.3  # fixed normal/anomaly balance coefficient
        self.severity_resource_weight = 0.5  # weight of cpu/mem resource-deviation term in severity
        self.severity_qps_weight = 0.1  # QPS is supporting rather than primary evidence
        # --- Module A (M1) service-aware dynamic metric modeling ---
        self.metric_gate_enable = True   # sparse-mask + per-type channel gating before conv

        # --- Module C: execution-graph driven LLM fine-grained localization ---
        self.llm_enable = False
        self.llm_mock = False            # True: rule-based mock (no network); False: call OpenAI-compatible API
        self.llm_base_url = os.getenv("OPENAI_BASE_URL")
        self.llm_api_key = os.getenv("OPENAI_API_KEY")
        self.llm_model = os.getenv("OPENAI_MODEL")
        self.llm_max_tokens = 4096
        self.llm_temperature = 0.0
        self.llm_topk = 5               # number of coarse candidates fed to the LLM
        # Three auditable LLM passes: GNN-anchored screening, multi-evidence
        # verification, then a final decision over the surviving candidates.
        self.llm_staged = False         # False: one LLM call that ranks candidates; True: legacy 3-stage
        self.llm_stage1_topk = 3
        self.llm_stage2_topk = 2
        self.exec_slice_max_chars = 1500  # truncation budget per messages/task text
        self.exec_max_traces = 40       # cap on execution-graph traces scanned
        # A log signature must recur within the anomaly window before it is
        # promoted to root-cause evidence. One-off matches remain discarded.
        self.log_signature_min_count = 2

        # --- coarse ranker selection ---
        # severity ranking is always computed; the GNN ranker runs alongside it
        # for comparison when enabled (needs torch/dgl/networkx; auto-skips if
        # the deps are missing).
        self.gnn_enable = True          # try the heterogeneous GNN ranker
        self.gnn_feed_module_c = False  # if True and GNN ran, feed GNN top-k to Module C
        self.gnn_train = 'train'        # 'train' or 'eval'
        # Transpose the svc->svc call edges so the GNN propagates anomaly signal
        # from downstream victims back toward the upstream root cause (avoids
        # accumulating mass on the call-graph sink; see PPR reverse analysis).
        self.gnn_graph_reverse = True
        # Readout fusion: anchor converged GNN logits to the severity prior and
        # optionally suppress high-degree hubs/sinks (see gnn_rank).
        #   final = (1-w)*gnn + w*severity ; final /= (1 + lam*degree)
        self.gnn_severity_anchor = 0.85  # w in [0,1]; 0 = pure GNN, 1 = pure severity
        self.gnn_hub_penalty = 0.0       # lam >= 0; 0 = off (deg gives no separation here)

        # --- Baseline: Personalized PageRank ranker (comparison method) ---
        # Same graph construction (agent_gnn.build_graphs) and severity prior
        # (agent_gnn._build_node_severity) as the GNN; only the ranker differs:
        # severity becomes the per-node personalization vector and a random walk
        # with restart is iterated to convergence over the topology.
        self.ppr_alpha = 0.85           # restart/damping factor
        self.ppr_reverse = True         # transpose call graph: walk symptom -> upstream cause
        self.ppr_max_iter = 200
        self.ppr_tol = 1e-8


class Node:
    def __init__(self, name, ip, node_name, cni_ip, status, center):
        self.name = name
        self.ip = ip
        self.node_name = node_name
        self.cni_ip = cni_ip
        self.status = status
        self.center = center


class Pod:
    def __init__(self, node, namespace, host_ip, ip, name, center):
        self.node = node
        self.namespace = namespace
        self.host_ip = host_ip
        self.ip = ip
        self.name = name
        self.center = center
