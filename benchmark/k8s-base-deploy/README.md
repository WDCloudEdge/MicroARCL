# Kubernetes Base Environment

Deployment manifests for the microservice benchmark used by MicroARCL. The cluster hosts
four conventional microservice systems that the agent services run alongside, plus the
monitoring stack that the data collector reads from.

## Components

| Component | Version | Deployment |
|-------|-------|-------|
| Kubernetes | v1.19.16 | kubeadm, `kube-init.yaml` |
| Flannel | — | VXLAN over the `wg1` tunnel interface |
| Istio | 1.15.1 | `istioctl install` |
| Prometheus | kube-prometheus 0.7.0 | `prometheus/` (NodePort 30200) |
| Prometheus (Istio metrics) | — | NodePort 30202 |
| Grafana | — | `prometheus/` (NodePort 30100) |
| Chaos Mesh | 2.3.3 | Helm, dashboard on NodePort 31439 |
| Elasticsearch | 8.11.4 | Host process, stores Jaeger spans |
| Kibana | 8.11.4 | Container |
| Jaeger | 1.52 | Container, all-in-one |
| Harbor | 2.4.2 | Image registry, listens on port 81 |
| tcpdump | 4.9.2 | Host package |

Kuboard, Nacos, and MinIO run as standalone processes on a single host rather than inside
the cluster, and are reached through the hostnames `minio.agent.network.com` and
`center.agent.network.com`.

## Layout

```
cluster/           Cluster bootstrap manifests
  kube-init.yaml              kubeadm InitConfiguration for the control plane
  kube-flannel.yml            Flannel DaemonSet bound to the tunnel interface
bookinfo/          Bookinfo: 6 Deployments + 4 Services
  bookinfo.yaml               default topology
  bookinfo-experiment.yaml    reviews-v1 and reviews-v3 scaled to 0
hipster/           Hipster Shop: 11 Deployments + 11 Services
sock-shop/         Sock Shop: 14 Deployments + 14 Services
train-ticket/      TrainTicket: 46 Deployments
prometheus/        kube-prometheus: operator, Prometheus, Alertmanager,
                   Grafana, kube-state-metrics, node-exporter,
                   blackbox-exporter, prometheus-adapter
istio/             Istio `istioctl` completion scripts and certificate tooling
```

`prometheus/setup/` holds the CustomResourceDefinitions and must be applied before the
rest of the monitoring stack.

## Prerequisites

1. **Image registry reachable from every node.** Workload images are served from a local
   Harbor instance:

   ```bash
   # /etc/docker/daemon.json on each node
   {
     "exec-opts": ["native.cgroupdriver=systemd"],
     "insecure-registries": ["<REGISTRY_HOST>:81"]
   }
   ```

   ```bash
   kubectl create secret docker-registry private-registry-secret \
     --docker-server=<REGISTRY_HOST>:81 \
     --docker-username=<REGISTRY_USER> \
     --docker-password=<REGISTRY_PASSWORD> \
     -n <namespace>
   ```

   The Sock Shop manifest references this secret through `imagePullSecrets`.

2. **Helm**, for Chaos Mesh: <https://helm.sh/docs/helm/helm_install/>

3. **`kubectl` access to the cluster**, via `~/.kube/config`.

## 1. Build the cluster

The nodes reach each other over a WireGuard tunnel (`wg1`), each holding a virtual IP in
`30.0.0.0/24`; physical neighbours are kept on the local link with per-node routes. This
keeps the cluster independent of the cloud server that terminates the tunnel.

Every node needs swap disabled (after each reboot), the time zone set, and the tunnel
addresses in `/etc/hosts`:

```bash
sudo swapoff -a
sudo timedatectl set-timezone Asia/Shanghai
```

Initialise the control plane from `cluster/kube-init.yaml`, which pins the Kubernetes
version, the pod and service subnets, and IPVS proxy mode:

```bash
kubeadm init --config cluster/kube-init.yaml --ignore-preflight-errors=all
```

Join workers with the command printed by:

```bash
kubeadm token create --print-join-command
```

Install Flannel from `cluster/kube-flannel.yml`. Its DaemonSet runs `flanneld` with
`--iface=wg1` so that VXLAN traffic follows the tunnel; without it pods on different
subnets cannot reach each other:

```bash
kubectl apply -f cluster/kube-flannel.yml
```

Verify the tunnel and pod-to-pod connectivity:

```bash
bridge fdb show | grep flannel          # entries pointing at the peer virtual IPs
kubectl get pods -n kube-flannel -o wide
```

When a node on another subnet cannot reach the API server through the tunnel, add a
MASQUERADE rule. The packet is DNATed to the control plane address but leaves through
`wg1` without source translation, so the reply bypasses conntrack on the sending node and
the connection never establishes; Flannel then crash-loops on the same failure:

```bash
sudo iptables -t nat -A POSTROUTING -d 30.0.0.69/32 -p tcp --dport 6443 -o wg1 -j MASQUERADE
```

Persist the rule across reboots:

```bash
sudo apt install -y iptables-persistent
sudo netfilter-persistent save
```

## 2. Deploy the monitoring stack

```bash
cd benchmark
make monitor-deploy
```

This applies `prometheus/setup` followed by `prometheus/`, installs Chaos Mesh 2.3.3, and
installs tcpdump on the host. Prometheus and Grafana are exposed as NodePorts so that the
data collector can query them from outside the cluster:

| Service | NodePort | Used by |
| --- | ---: | --- |
| `monitoring/prometheus-k8s` | 30200 | node, pod, and service metrics |
| `istio-system/prometheus` | 30202 | Istio mesh metrics |
| `monitoring/grafana` | 30100 | dashboards |

NodePort 30200 and 30100 are set by `prometheus/prometheus-service.yaml` and
`prometheus/grafana-service.yaml`. Port 30202 is served by the Istio addon Prometheus in
`istio-system`, which is installed separately from the Istio release and is not part of
this directory.

Prometheus and Grafana addresses are configured in `benchmark/data-collector/Config.py`.

## 3. Deploy the microservice systems

```bash
cd benchmark
make service-deploy
```

Each system is applied into its own namespace (`bookinfo`, `hipster`, `sock-shop`,
`train-ticket`). Hipster does not declare a namespace in its manifest, so the `-n hipster`
flag is required. Enable sidecar injection before deploying if the system should join the
mesh — see [Install Istio](#4-install-istio) below.

## 4. Install Istio

Download the release and put `istioctl` on the path:

```bash
curl -L https://istio.io/downloadIstio | ISTIO_VERSION=1.15.1 sh -
cd istio-1.15.1
export PATH=$PWD/bin:$PATH
```

Install the control plane with tracing enabled; `istio/tracing.yaml` carries the same
settings:

```bash
istioctl install -f istio/tracing.yaml
kubectl get pods -n istio-system
```

Enable sidecar injection for each namespace that should join the mesh, then restart its
workloads:

```bash
kubectl label namespace bookinfo istio-injection=enabled
kubectl rollout restart deployment -n bookinfo
```

`istio/tools/` holds the `istioctl` completion scripts and the cfssl Makefiles used to
generate mesh certificates.

## Data operation

Load generation, fault injection, and metric collection are driven from
`benchmark/scripts` and `benchmark/failure_injection`; see the repository root README for
the run order. Failures are injected with Chaos Mesh (CPU, memory, network delay, pod
failure, pod deletion) and metrics, logs, and execution graphs are collected through
`benchmark/data-collector`.

## License

This directory is covered by the [Apache License 2.0](../LICENSE). The microservice
systems are third-party projects and credit goes to their original authors:

- [Sock Shop](https://github.com/microservices-demo/microservices-demo)
- [Hipster Shop / Online Boutique](https://github.com/GoogleCloudPlatform/microservices-demo)
- [TrainTicket](https://github.com/FudanSELab/train-ticket)
- [Bookinfo](https://github.com/istio/istio/tree/master/samples/bookinfo)
