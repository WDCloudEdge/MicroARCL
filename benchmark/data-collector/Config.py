import os
import time

class Config:
    def __init__(self):
        self.namespace = 'agent-network'
        self.nodes = None
        self.svcs = set()
        self.pods = set()

        self.interval = 20 * 60  # interval between data collections (5 min)
        # duration related to interval
        self.duration = self.interval
        # self.start = int(round((time.time() - self.duration)))
        self.end = 1789282200

        self.start = self.end - self.duration

        # prometheus
        # self.prom_range_url = "http://192.168.31.227:30210/api/v1/query_range"  # istio support
        self.prom_range_url = "http://192.168.31.227:30202/api/v1/query_range"  # istio support
        self.prom_range_url_node = "http://192.168.31.227:30200/api/v1/query_range"  # native Prometheus
        self.prom_no_range_url_node = "http://192.168.31.227:30200/api/v1/query"
        # self.prom_no_range_url = "http://192.168.31.227:30210/api/v1/query"
        self.prom_no_range_url = "http://192.168.31.227:30200/api/v1/query"
        self.step = 5

        # jaeger
        self.jaeger_url = 'http://192.168.31.171:16686/api/traces?'
        self.lookBack = str(int(self.duration / 60)) + 'm'
        self.limit = 100000

        # kiali
        self.kiali_url = 'http://47.99.200.176:32001/kiali/api'

        # kubernetes
        # kubernetes config file path; anchored to this file's directory so it
        # resolves no matter which working directory the collector is run from.
        self.k8s_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')

        self.user = ''


class Node:
    def __init__(self, name, ip, node_name, cni_ip, status):
        self.name = name
        self.ip = ip
        self.node_name = node_name
        self.cni_ip = cni_ip
        self.status = status


class Pod:
    def __init__(self, node, namespace, host_ip, ip, name):
        self.node = node
        self.namespace = namespace
        self.host_ip = host_ip
        self.ip = ip
        self.name = name
