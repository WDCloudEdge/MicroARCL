dir="$(dirname "$0")"
total_count=$1
is_scale=$2
let count=1

for ((i=1;i<=$total_count;i++))
do
	echo ${3}_net_latency_${count}
    start_timestamp=$(date +%s) 
    start_timestamp=$((start_timestamp - 6 * 60))  
	kubectl apply -f $dir/net_latency.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") start create."
    sleep 60
    if [ "$is_scale" = "true" ]; then
        echo "$(date +"%Y-%m-%d %T") start scale up 1."
        kubectl scale deployment agent-network-marble-research --replicas=$(( $(kubectl get deployment agent-network-marble-research -n agent-network -o=jsonpath='{.spec.replicas}') + 1)) -n agent-network
    fi
	sleep 120
	echo "$(date +"%Y-%m-%d %T") start delete."
	kubectl delete -f $dir/net_latency.yaml -n chaos-mesh
	echo "$(date +"%Y-%m-%d %T") finish delete."
    sleep 60
    before_timestamp=$(date +%s) 
    if [ "$is_scale" = "true" ]; then
        echo "$(date +"%Y-%m-%d %T") start scale down 1."
        python $dir/../log-collect/Log.py ${3}_net_latency_${count} before $start_timestamp $before_timestamp
        kubectl scale deployment agent-network-marble-research --replicas=$(( $(kubectl get deployment agent-network-marble-research -n agent-network -o=jsonpath='{.spec.replicas}') - 1)) -n agent-network
    fi
    after_timestamp=$(date +%s)
	echo -e "\n"
	sleep $((660 - $after_timestamp + $before_timestamp))
    if [ "$is_scale" = "true" ]; then
        end_timestamp=$((before_timestamp + 5 * 60)) 
        python $dir/../log-collect/Log.py ${3}_net_latency_${count} after $before_timestamp $end_timestamp &
    fi
    ((count++))
done