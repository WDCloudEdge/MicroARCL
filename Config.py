import time


class Config:
    def __init__(self):
        # base
        self.collect = False
        self.namespace = ''
        self.dataset = 'sock_shop_chaos'
        self.nodes = None
        self.svcs = set()
        self.pods = set()

        self.time_window = 20
        self.interval = self.time_window * 60
        # duration related to interval
        self.duration = self.interval
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

        # anomaly threshold
        self.anomaly_threshold = 0.07

        # ===== Agent-service RCA (initial version) =====
        # single-namespace agent dataset root (a failure/normal sample directory)
        self.agent_namespace = 'agent-network'

        # --- label-free adaptive window (aligned KPI timeline) ---
        self.win_seg_min_len = 2        # min significant anomaly segment length (steps)
        self.win_guard = 2 * self.step  # outer window guard band (seconds)
        # KPI baselines exclude the anomaly window and prefer its history.
        self.baseline_min_points = 3
        self.metric_spike_k = 5.0       # robust median + k*MAD threshold
        self.metric_min_rel = 0.3       # minimum relative CPU/mem/latency/QPS change
        self.availability_min_rel = 0.01  # minimum success-rate decrease
        # --- lag alignment (QPS <-> CPU/mem/latency/network) ---
        # Shared by adaptive window, severity, and Birch.
        # 'self': QPS->own-resource lag (compute_service_lags);
        # 'chain': call-chain propagation lag (compute_chain_lags, needs exec graph)
        self.lag_mode = 'self'
        self.lag_tau_max = 12           # cross-correlation lag search bound (steps, ~60s)
        self.lag_min_corr = 0.3         # minimum accepted QPS/metric correlation
        self.lag_min_improvement = 0.05 # required gain over zero-lag correlation
        self.lag_min_points = 12        # minimum valid paired observations
        # --- severity resource/QPS weighting ---
        self.severity_resource_weight = 0.5  # weight of cpu/mem resource-deviation term in severity
        self.severity_qps_weight = 0.1  # QPS is supporting rather than primary evidence

        # --- execution-graph log slicing ---
        self.exec_slice_max_chars = 1500  # truncation budget per messages/task text
        # A log signature must recur within the anomaly window before it is
        # promoted to root-cause evidence. One-off matches remain discarded.
        self.log_signature_min_count = 2

        # --- Personalized PageRank ranker (MicroRCA baseline) ---
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
