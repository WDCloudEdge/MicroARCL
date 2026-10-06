#!/usr/bin/env bash
#
# Full automation: run every service, every fault type, N samples each.
# Windows run strictly sequentially (only one fault active at a time), each
# service writing its own <service>_label.txt.
#
# WARNING: this is long. With 11 services x 5 faults x 3 samples x 20 min
# it takes ~55 hours. Trim `services` / `faults` / `total_count` as needed.
#
# Run:  bash run_all_services.sh

total_count=1
faults=(cpu_load mem_load net_latency pod_failure pod_kill)
services=(
    agent-network-planner
    agent-network-csv-gen
    agent-network-direction
    agent-network-excel-gen
    agent-network-excel-parsing
    agent-network-image
    agent-network-image-gen
    agent-network-ocr
    agent-network-pdf-gen
    agent-network-pdf-parsing
    agent-network-summarizer
    agent-network-word-gen
    agent-network-word-parsing
)

export LOCUST_USERS="${LOCUST_USERS:-5}"           # concurrent users
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
# true: inject all configured faults; false: collect normal data only.
export INJECT_FAULT="${INJECT_FAULT:-true}"

dir="$(cd "$(dirname "$0")" && pwd)"
for service in "${services[@]}"; do
    echo "########## $service ##########"
    bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
done
