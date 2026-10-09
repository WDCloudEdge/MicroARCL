##bookinfo
kubectl apply -n bookinfo -f ./k8s-base-deploy/bookinfo/bookinfo.yaml

##hipster
kubectl apply -n hipster -f ./k8s-base-deploy/hipster/hipster.yaml

##sock-shop
kubectl apply -n sock-shop -f ./k8s-base-deploy/sock-shop/sock-shop-multiarch-deploy.yaml

##train-ticket
kubectl apply -n train-ticket -f ./k8s-base-deploy/train-ticket/deploy.yaml
