#!/usr/bin/env bash
#
# Full automation for MARBLEbench: run every service, every fault type,
# and N samples each. Windows run strictly sequentially so only one fault is
# active at a time.
#
# Run: bash run_all_services.sh

total_count=1
faults=(cpu_load mem_load net_latency pod_failure pod_kill)
services=(
    agent-network-marble-research
    agent-network-marble-web
    agent-network-marble-coding
    agent-network-marble-database
    agent-network-marble-world
    agent-network-marble-werewolf
    agent-network-marble-minecraft
    agent-network-planner
    agent-network-summarizer
)

export LOCUST_USERS="${LOCUST_USERS:-5}"
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
# true: inject all configured faults; false: collect normal data only.
export INJECT_FAULT="${INJECT_FAULT:-false}"

dir="$(cd "$(dirname "$0")" && pwd)"
for service in "${services[@]}"; do
    echo "########## $service ##########"
    bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
done
