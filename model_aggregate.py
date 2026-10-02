import torch
import torch as th
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from dgl import (
    broadcast_nodes,
    max_nodes,
    mean_nodes,
    softmax_nodes,
    sum_nodes,
    topk_nodes,
    DGLHeteroGraph,
    to_homogeneous
)
from graph import EdgeType, HeteroWithGraphIndex, NodeType
from typing import Dict
from model_attention import AttentionLayer
from Config import RnnType


# Module A (M1): toggle for the service-aware channel gating. train() sets it
# from config.metric_gate_enable so it can be ablated.
METRIC_GATE_ENABLE = True


class MetricGate(nn.Module):
    """A.3 service-aware channel gating. From the mask-aware temporal mean of a
    node type's metrics, learn a per-channel gate (sigmoid) that reweights the
    discriminative channels, applied with the sparse mask so inactive entries
    stay zero. Shapes: x,m [N,T,C] -> [N,T,C]. Each node type owns its own gate
    (heterogeneous / service-aware)."""

    def __init__(self, feat_num):
        super(MetricGate, self).__init__()
        self.gate = nn.Linear(feat_num, feat_num)

    def forward(self, x, m):
        denom = m.sum(dim=1) + 1e-6            # [N, C]  active steps per channel
        s = (x * m).sum(dim=1) / denom          # [N, C]  masked temporal mean
        g = th.sigmoid(self.gate(s))            # [N, C]  channel gate in (0,1)
        return x * g.unsqueeze(1) * m           # broadcast over T, keep sparse


class AggrHGraphConvLayer(nn.Module):
    def __init__(self, out_channel, svc_feat_num, instance_feat_num, node_feat_num):
        super(AggrHGraphConvLayer, self).__init__()
        self.svc_feat_num = svc_feat_num
        self.instance_feat_num = instance_feat_num
        self.node_feat_num = node_feat_num
        self.gate = nn.ModuleDict({
            NodeType.SVC.value: MetricGate(svc_feat_num),
            NodeType.POD.value: MetricGate(instance_feat_num),
            NodeType.NODE.value: MetricGate(node_feat_num),
        })
        self.conv = dglnn.HeteroGraphConv({
            EdgeType.SVC_CALL_EDGE.value: dglnn.GraphConv(self.svc_feat_num, out_channel),
            EdgeType.INSTANCE_NODE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel),
            EdgeType.NODE_INSTANCE_EDGE.value: dglnn.GraphConv(self.node_feat_num, out_channel),
            EdgeType.INSTANCE_INSTANCE_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel),
            EdgeType.SVC_INSTANCE_EDGE.value: dglnn.GraphConv(self.svc_feat_num, out_channel),
            EdgeType.INSTANCE_SVC_EDGE.value: dglnn.GraphConv(self.instance_feat_num, out_channel)
        },
            aggregate='mean')
        if th.cuda.is_available():
            self.conv = self.conv.to('cpu')
        self.activation = nn.ReLU()

    def forward(self, graph: HeteroWithGraphIndex, feat_dict, mask_dict=None):
        # A.3: service-aware channel gating before graph convolution
        if METRIC_GATE_ENABLE and mask_dict is not None:
            feat_dict = {t: self.gate[t](feat_dict[t], mask_dict[t]) for t in feat_dict}
        dict = self.conv(graph.hetero_graph, feat_dict)
        node_feat = dict[NodeType.NODE.value]
        instance_feat = dict[NodeType.POD.value]
        svc_feat = dict[NodeType.SVC.value]
        return self.activation(th.cat([node_feat, instance_feat, svc_feat], dim=0))


class AggrHGraphConvWindow(nn.Module):
    def __init__(self, hidden_size, output_size, svc_feat_num, instance_feat_num, node_feat_num, rnn: RnnType = RnnType.LSTM):
        super(AggrHGraphConvWindow, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size, hidden_size=self.output_size, num_layers=2,
                                    batch_first=True)
        self.hGraph_conv_layer_list = []
        self.svc_feat_num = svc_feat_num
        self.instance_feat_num = instance_feat_num
        self.node_feat_num = node_feat_num
        self.activation = nn.ReLU()
        self.hGraph_conv_layer = AggrHGraphConvLayer(self.hidden_size, self.svc_feat_num, self.instance_feat_num,
                                                     self.node_feat_num)

    def forward(self, graph: HeteroWithGraphIndex):
        time_series = graph.hetero_graph.nodes[NodeType.SVC.value].data[
            'feat'].shape[1]
        feat_dict = {
            NodeType.NODE.value: graph.hetero_graph.nodes[NodeType.NODE.value].data[
                                     'feat'][:, :time_series, :],
            NodeType.SVC.value: graph.hetero_graph.nodes[NodeType.SVC.value].data[
                                    'feat'][:, :time_series, :],
            NodeType.POD.value: graph.hetero_graph.nodes[NodeType.POD.value].data[
                                    'feat'][:, :time_series, :],
        }
        # A.2 sparse mask aligned with feat_dict. Legacy graphs without an
        # explicit mask treat every stored value as observed because both 0 and
        # the -1 missing-metric sentinel carry meaning.
        def _mask(nt):
            data = graph.hetero_graph.nodes[nt].data
            m = data['mask'] if 'mask' in data else th.ones_like(data['feat'])
            return m[:, :time_series, :]
        mask_dict = {NodeType.NODE.value: _mask(NodeType.NODE.value),
                     NodeType.SVC.value: _mask(NodeType.SVC.value),
                     NodeType.POD.value: _mask(NodeType.POD.value)}
        graph_time_series_feat = self.hGraph_conv_layer(graph, feat_dict, mask_dict)
        node_num = len(graph.hetero_graph_index.index[NodeType.NODE.value])
        instance_num = len(graph.hetero_graph_index.index[NodeType.POD.value])
        hetero_graph_feat_dict = {NodeType.NODE.value: graph_time_series_feat[:node_num],
                                  NodeType.POD.value: graph_time_series_feat[node_num:node_num + instance_num],
                                  NodeType.SVC.value: graph_time_series_feat[node_num + instance_num:]}
        return hetero_graph_feat_dict, self.activation(self.rnn_layer(graph_time_series_feat)[0]), graph.center_type_name, graph.anomaly_name


class HeteroGlobalAttentionPooling(nn.Module):
    def __init__(self, gate_nn, feat_nn=None):
        super(HeteroGlobalAttentionPooling, self).__init__()
        self.gate_nn = gate_nn
        self.feat_nn = feat_nn
        self.activation = nn.Softmax(dim=0)

    def forward(self, h_graph_index: HeteroWithGraphIndex, feat_dict, get_attention=False):
        h_graph = h_graph_index.hetero_graph
        assert isinstance(h_graph, DGLHeteroGraph), "graph is not an instance of DGLHeteroGraph"
        ntypes = h_graph.ntypes
        index = {}
        count = 0
        for ntype in h_graph.ntypes:
            for key in h_graph_index.hetero_graph_index.index[ntype]:
                index[key] = h_graph_index.hetero_graph_index.index[ntype][key] + count
            count += len(h_graph_index.hetero_graph_index.index[ntype].keys())
        feat_all = th.cat([feat_dict[ntype] for ntype in ntypes], dim=0)
        graph = to_homogeneous(h_graph)
        with graph.local_scope():
            gate = self.gate_nn(feat_all)
            # assert (
            #         gate.shape[-1] == 1
            # ), "The output of gate_nn should have size 1 at the last axis."
            feat = self.feat_nn(feat_all) if self.feat_nn else feat_all

            graph.ndata["gate"] = gate
            gate = softmax_nodes(graph, "gate")
            graph.ndata.pop("gate")

            graph.ndata["r"] = feat * gate
            readout = th.sum(sum_nodes(graph, "r"), dim=1)
            graph.ndata.pop("r")
            time_series_size = gate.shape[1]
            if get_attention:
                attention_scores = torch.max(input=gate, dim=1)[0]
                return readout, index, time_series_size, attention_scores
            else:
                return readout, index, time_series_size


class AggrHGraphConvWindows(nn.Module):
    def __init__(self, out_channel, hidden_channel, svc_feat_num, instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM):
        super(AggrHGraphConvWindows, self).__init__()
        self.hidden_size = hidden_channel
        self.out_size = out_channel
        if rnn == RnnType.LSTM:
            self.rnn_layer = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                     batch_first=True)
        elif rnn == RnnType.GRU:
            self.rnn_layer = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=2,
                                    batch_first=True)
        self.graph_window_conv = AggrHGraphConvWindow(64, self.hidden_size, svc_feat_num, instance_feat_num,
                                                      node_feat_num, rnn)
        self.linear = nn.Linear(self.hidden_size, self.out_size)
        self.output_layer = nn.Softmax(dim=0)
        self.activation = nn.ReLU()
        self.center_gate = nn.Sigmoid()
        self.pooling = HeteroGlobalAttentionPooling(gate_nn=nn.Linear(self.hidden_size, hidden_channel))
        self.center_attention = AttentionLayer(hidden_channel, hidden_channel, num_heads=1)

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        output_data_list = []
        window_graphs_center_node_index = []
        window_graphs_anomaly_node_index = []
        window_graphs_index = []
        window_time_series_sizes = []
        window_anomaly_time_series = []
        times = graphs.keys()
        times_sorted = sorted(times)
        atten_sorted = []
        for time in times_sorted:
            graph = graphs[time]
            hetero_graph_feat_dict, single_graph_window_feat, graphs_center_node_name, graphs_anomaly_node_name = self.graph_window_conv(
                graph)
            output_feat, index, time_series_size, attention_scores = self.pooling(graph, hetero_graph_feat_dict, True)
            window_time_series_sizes.append(time_series_size)
            window_anomaly_time_series.append(graph.anomaly_time_series)
            output_data_list.append(output_feat)
            window_graphs_index.append(index)
            # convert to the index of the current graph
            graph_center_node_index = {}
            for center in graphs_center_node_name:
                if center not in graph_center_node_index:
                    graph_center_node_index[center] = {}
                graph_center_node_name = graphs_center_node_name[center]
                for node_type in graph_center_node_name:
                    if node_type not in graph_center_node_index:
                        graph_center_node_index[center][node_type] = []
                    graph_center_nodes = graph_center_node_name[node_type]
                    for graph_center_node in graph_center_nodes:
                        graph_center_node_index[center][node_type].append(index[graph_center_node])
            graphs_anomaly_node_index = {}
            for anomaly in graphs_anomaly_node_name:
                anomaly_n = anomaly[anomaly.find('$') + 1:]
                if anomaly_n not in graph.node_exist:
                    continue
                if anomaly not in graphs_anomaly_node_index:
                    graphs_anomaly_node_index[anomaly] = {}
                graph_anomaly_node_name = graphs_anomaly_node_name[anomaly]
                for node_type in graph_anomaly_node_name:
                    graph_anomaly_nodes = graph_anomaly_node_name[node_type]
                    for graph_anomaly_node in graph_anomaly_nodes:
                        is_neighbor = 'neighbor' in graph_anomaly_node
                        if is_neighbor:
                            center = graph_anomaly_node[9:][:graph_anomaly_node[9:].find('$')]
                        else:
                            center = graph_anomaly_node[:graph_anomaly_node.find('$')]
                        graph_anomaly_node = graph_anomaly_node[graph_anomaly_node.rfind('$') + 1:]
                        if 'neighbor' not in graphs_anomaly_node_index[anomaly]:
                            graphs_anomaly_node_index[anomaly]['neighbor'] = {}
                        if is_neighbor:
                            neighbors_type = graphs_anomaly_node_index[anomaly]['neighbor'].get(center, [])
                            neighbors_type.append(index[graph_anomaly_node])
                            graphs_anomaly_node_index[anomaly]['neighbor'][center] = neighbors_type
                        else:
                            graphs_anomaly_node_index[anomaly]['source'] = [index[graph_anomaly_node]]
            # Apply center attention
            attention_scores_after_center = th.zeros([attention_scores.shape[0], self.out_size]).to('cpu')
            center_embeddings = []
            for center in graph_center_node_index:
                center_nodes_index = []
                for _, nodes_index in graph_center_node_index[center].items():
                    center_nodes_index.extend(nodes_index)
                aggr_center = th.mean(attention_scores[sorted(center_nodes_index)], dim=0, keepdim=True)
                center_embeddings.append(aggr_center)
            center_embeddings = th.cat(center_embeddings, dim=0)
            aggr_feat_weighted, attention_weights_center = self.center_attention(center_embeddings, center_embeddings,
                                                                                 center_embeddings)
            for i, center in enumerate(graph_center_node_index):
                center_nodes_index = []
                for _, nodes_index in graph_center_node_index[center].items():
                    center_nodes_index.extend(nodes_index)
                # Both factors are probabilities/gates in [0,1]. ReLU here
                # used to leave the center weight unbounded, allowing node
                # anomaly scores to grow above one during fitting.
                center_weight = self.center_gate(aggr_feat_weighted[i])
                attention_scores_after_center[sorted(center_nodes_index)] = th.max(
                    attention_scores[sorted(center_nodes_index)] * center_weight,
                    dim=1)[0].unsqueeze(-1)
            atten_sorted.append(attention_scores_after_center)
            window_graphs_center_node_index.append(graph_center_node_index)
            window_graphs_anomaly_node_index.append(graphs_anomaly_node_index)
        output = self.activation(self.linear(self.rnn_layer(th.stack(output_data_list, dim=0))[0]))
        output = output.reshape(output.shape[0], -1)
        graphs_probability = self.output_layer(torch.sum(output, dim=1, keepdim=True))
        bounded_scores = [
            (graphs_probability[g_index] * atten_sorted[g_index]).clamp(0.0, 1.0)
            for g_index in range(len(atten_sorted))
        ]
        return bounded_scores, window_graphs_center_node_index, window_graphs_anomaly_node_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series


class AggrUnsupervisedGNN(nn.Module):
    def __init__(self, sorted_graphs, center_map, anomaly_index, out_channels, hidden_size, svc_feat_num,
                 instance_feat_num, node_feat_num,
                 rnn: RnnType = RnnType.LSTM,
                 node_severity=None, severity_param=0.3):
        super(AggrUnsupervisedGNN, self).__init__()
        # B.3: severity-as-prior. When node_severity is provided, the bp target
        # for every node becomes severity*coeff (coeff=1 if Birch-anomaly else
        # the fixed severity_param); when empty, falls back to a fixed bounded
        # propagation target. Target coefficients are never learned.
        self.node_severity = node_severity or {}
        self.severity_param = severity_param
        self.conv = AggrHGraphConvWindows(out_channel=out_channels, hidden_channel=hidden_size,
                                          svc_feat_num=svc_feat_num, instance_feat_num=instance_feat_num,
                                          node_feat_num=node_feat_num, rnn=rnn)
        anomaly_index_reverse = {idx: an for an, idx in anomaly_index.items()}
        anomaly_nodes_maps = [sorted_graph.anomaly_name for sorted_graph in sorted_graphs]
        self.graphs_anomaly_center_nodes = []
        for graph_idx in range(len(anomaly_nodes_maps)):
            graph_anomaly_center_nodess = {}
            anomaly_nodes_map = anomaly_nodes_maps[graph_idx]
            for i in range(len(anomaly_index)):
                ano = anomaly_index_reverse[i]
                graph_anomaly_center_nodes = {}
                if ano in anomaly_nodes_map:
                    for node_type in anomaly_nodes_map[ano]:
                        graph_anomaly_nodes = anomaly_nodes_map[ano][node_type]
                        for graph_anomaly_node in graph_anomaly_nodes:
                            is_neighbor = 'neighbor' in graph_anomaly_node
                            if is_neighbor:
                                center = graph_anomaly_node[9:][:graph_anomaly_node[9:].find('$')]
                            else:
                                continue
                            graph_anomaly_node = graph_anomaly_node[graph_anomaly_node.rfind('$') + 1:]
                            if center not in graph_anomaly_center_nodes:
                                graph_anomaly_center_nodes[center] = []
                            if is_neighbor:
                                neighbors_type = graph_anomaly_center_nodes.get(center, [])
                                neighbors_type.append(graph_anomaly_node)
                                graph_anomaly_center_nodes[center] = neighbors_type
                    graph_anomaly_center_nodess[ano] = graph_anomaly_center_nodes
            self.graphs_anomaly_center_nodes.append(graph_anomaly_center_nodess)

        self.anomaly_index = anomaly_index
        self.center_map = center_map
        self.criterion = nn.MSELoss()

    def forward(self, graphs: Dict[str, HeteroWithGraphIndex]):
        aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series = self.conv(
            graphs)
        return aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes, window_anomaly_time_series

    def loss(self, aggr_feat, aggr_center_index, aggr_anomaly_index, window_graphs_index, window_time_series_sizes,
             window_anomaly_time_series):
        sum_criterion = 0
        # An all-zero severity map carries no prior information and must use the
        # fixed fallback rather than silently fitting an all-zero target.
        use_prior = any(float(value) > 0 for value in self.node_severity.values())

        for idx, anomaly_index_combine in enumerate(aggr_anomaly_index):
            aggr_feat_idx = aggr_feat[idx]

            if use_prior:
                # B.3: one target per graph = severity * coeff for every node.
                # The coefficient is fixed: severity_param for normal nodes and
                # 1 for Birch anomaly sources. No part of the target is learned.
                sev = torch.zeros_like(aggr_feat_idx)
                for name, pos in window_graphs_index[idx].items():
                    sev[pos] = min(1.0, max(
                        0.0, float(self.node_severity.get(name, 0.0))))
                target = self.severity_param * sev.clone()
                for anomaly in anomaly_index_combine:
                    if len(anomaly_index_combine[anomaly]) > 0:
                        aai = anomaly_index_combine[anomaly]
                        source_index_matrix = torch.tensor(aai['source'])
                        target[source_index_matrix] = sev[source_index_matrix]
                target = target.clamp(0.0, 1.0)
                sum_criterion += self.criterion(aggr_feat_idx, target)
            else:
                # No usable severity prior: fixed back-propagation target.
                # Sources are 1, immediate neighbors use the same bounded,
                # non-learnable coefficient, and all remaining nodes are 0.
                for anomaly in anomaly_index_combine:
                    if len(anomaly_index_combine[anomaly]) > 0:
                        aggr_anomaly_nodes_index = anomaly_index_combine[anomaly]
                        rate = 1
                        aggr_feat_label_weight = torch.zeros_like(aggr_feat_idx)
                        source_index_matrix = torch.tensor(aggr_anomaly_nodes_index['source'])
                        aggr_feat_label_weight[source_index_matrix] = rate
                        if 'neighbor' in aggr_anomaly_nodes_index:
                            for center in aggr_anomaly_nodes_index['neighbor']:
                                for ano_idx_idx, ano_idx in enumerate(aggr_anomaly_nodes_index['neighbor'][center]):
                                    precessor_rate = self.severity_param
                                    neighbor_index_matrix = torch.tensor(
                                        aggr_anomaly_nodes_index['neighbor'][center][ano_idx_idx])
                                    aggr_feat_label_weight[neighbor_index_matrix] = precessor_rate
                        aggr_feat_label_weight = aggr_feat_label_weight.clamp(
                            0.0, 1.0)
                        sum_criterion += self.criterion(aggr_feat_idx, aggr_feat_label_weight)
        return sum_criterion
