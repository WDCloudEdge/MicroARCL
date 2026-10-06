
import torch
import os

class Config:
    def __init__(self,data_set):
        self.data_set = data_set
        self.max_length = 512
        self.batch_size = 16
        self.topk = 10
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = 42
        if data_set == "D1":
            self.all_enum = {'emailservice-0': 0, 'emailservice2-0': 1, 'emailservice-2': 2, 'emailservice-1': 3, 'adservice-0': 4, 'adservice2-0': 5, 'adservice-2': 6, 'adservice-1': 7, 'checkoutservice-0': 8, 'checkoutservice2-0': 9, 'checkoutservice-2': 10, 'checkoutservice-1': 11, 'paymentservice-0': 12, 'paymentservice2-0': 13, 'paymentservice-2': 14, 'paymentservice-1': 15, 'productcatalogservice-0': 16, 'productcatalogservice2-0': 17, 'productcatalogservice-2': 18, 'productcatalogservice-1': 19, 'shippingservice-0': 20, 'shippingservice2-0': 21, 'shippingservice-2': 22, 'shippingservice-1': 23, 'frontend-0': 24, 'frontend2-0': 25, 'frontend-2': 26, 'frontend-1': 27, 'recommendationservice-0': 28, 'recommendationservice2-0': 29, 'recommendationservice-2': 30, 'recommendationservice-1': 31, 'cartservice-0': 32, 'cartservice2-0': 33, 'cartservice-2': 34, 'cartservice-1': 35, 'currencyservice-0': 36, 'currencyservice2-0': 37, 'currencyservice-2': 38, 'currencyservice-1': 39, 'node-1': 40, 'node-2': 41, 'node-3': 42, 'node-4': 43, 'node-5': 44, 'node-6': 45}
        elif data_set == "D2":
            self.all_enum = {'adservice-0': 0, 'adservice-1': 1, 'adservice-2': 2,'aiops-k8s-01': 3, 'aiops-k8s-02': 4, 'aiops-k8s-03': 5, 'aiops-k8s-04': 6, 'aiops-k8s-05': 7, 'aiops-k8s-06': 8, 'aiops-k8s-07': 9, 'aiops-k8s-08': 10,'cartservice-0': 11, 'cartservice-1': 12, 'cartservice-2': 13,'checkoutservice-2': 14,'currencyservice-0': 15, 'currencyservice-1': 16, 'currencyservice-2': 17,'emailservice-0': 18, 'emailservice-1': 19, 'emailservice-2': 20,'frontend-0': 21, 'frontend-1': 22, 'frontend-2': 23,'k8s-master1': 24, 'k8s-master2': 25, 'k8s-master3': 26,'paymentservice-0': 27, 'paymentservice-1': 28, 'paymentservice-2': 29,'productcatalogservice-0': 30, 'productcatalogservice-1': 31, 'productcatalogservice-2': 32,'recommendationservice-0': 33,'redis-cart-0':34,'shippingservice-0': 35, 'shippingservice-1': 36, 'shippingservice-2': 37,'tidb_pd': 38, 'tidb_tidb': 39, 'tidb_tikv': 40}
        elif data_set == "agent":
            # MicroARCL agent-network dataset: 13 services + 15 pod slots + 9 physical nodes.
            # Pod slots come from union handling of scaling/pod_kill (kill-replace services
            # pdf-parsing & planner get 2 slots; concurrent replicas are averaged into 1).
            self.all_enum = {'agent-network-csv-gen': 0, 'agent-network-direction': 1, 'agent-network-excel-gen': 2, 'agent-network-excel-parsing': 3, 'agent-network-image': 4, 'agent-network-image-gen': 5, 'agent-network-ocr': 6, 'agent-network-pdf-gen': 7, 'agent-network-pdf-parsing': 8, 'agent-network-planner': 9, 'agent-network-summarizer': 10, 'agent-network-word-gen': 11, 'agent-network-word-parsing': 12, 'agent-network-csv-gen-pod0': 13, 'agent-network-direction-pod0': 14, 'agent-network-excel-gen-pod0': 15, 'agent-network-excel-parsing-pod0': 16, 'agent-network-image-pod0': 17, 'agent-network-image-gen-pod0': 18, 'agent-network-ocr-pod0': 19, 'agent-network-pdf-gen-pod0': 20, 'agent-network-pdf-parsing-pod0': 21, 'agent-network-pdf-parsing-pod1': 22, 'agent-network-planner-pod0': 23, 'agent-network-planner-pod1': 24, 'agent-network-summarizer-pod0': 25, 'agent-network-word-gen-pod0': 26, 'agent-network-word-parsing-pod0': 27, 'node-5': 28, 'node-6': 29, 'node-7': 30, 'node-15': 31, 'node-171': 32, 'node-218': 33, 'node-219': 34, 'node-221': 35, 'node-227': 36}
        else:
            # generic datasets (e.g. RCAEval re2ob/re2ss/re2tt): all_enum stored as a
            # json file next to the data, written by the dataset converter.
            import json
            base = os.path.dirname(os.path.abspath(__file__))
            enum_path = os.path.join(base, 'data', data_set, 'all_enum.json')
            if not os.path.exists(enum_path):
                raise ValueError(f"Unknown dataset '{data_set}' and no all_enum.json at {enum_path}")
            with open(enum_path) as f:
                self.all_enum = json.load(f)

