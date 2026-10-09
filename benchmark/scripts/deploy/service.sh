##bookinfo
kubectl apply -f ./k8s-base-deploy/bookinfo/bookinfo.yaml

##hipster
kubectl apply -f ./k8s-base-deploy/hipster/hipster.yaml
kubectl apply -f ./k8s-base-deploy/hipster/hipster2.yaml

##sock-shop
kubectl apply -f ./k8s-base-deploy/sock-shop/

##horsecoder
kubectl apply -f ./k8s-base-deploy/test-horsecoder

##train-ticket
kubectl apply -f ./k8s-base-deploy/train-ticket/deploy.yaml
