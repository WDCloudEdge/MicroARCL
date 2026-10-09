##Prometheus
kubectl apply -f ./k8s-base-deploy/prometheus/setup
kubectl apply -f ./k8s-base-deploy/prometheus

##Chaos-mesh
helm repo add chaos-mesh https://charts.chaos-mesh.org

helm install chaos-mesh chaos-mesh/chaos-mesh \
  --namespace chaos-mesh \
  --create-namespace \
  --version 2.3.3 \
  --set dashboard.create=true \
  --set dashboard.service.type=NodePort \
  --set chaosDaemon.privileged=true \
  --set controllerManager.replicaCount=1

##Tcpdump
sudo apt-get install -y tcpdump
