#!/usr/bin/env bash
#
# Entry point: collect chaos samples for one service.
# Runs the full 20-minute data-collection window (baseline -> load -> fault ->
# recovery) for each fault type and appends parseable labels to
#   agent-network-marble-database_label.txt
#
# Just run:  bash sum_chaos_cloud_agent-network-marble-database.sh

service="agent-network-marble-database"
total_count=1                                      # samples per fault type
faults=(cpu_load mem_load net_latency pod_failure pod_kill)

# ---- load (locust) settings; override here or via environment --------------
export LOCUST_USERS="${LOCUST_USERS:-3}"           # concurrent users
export LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-10}"
# true: inject faults; false: collect normal data without applying fault YAML.
export INJECT_FAULT="${INJECT_FAULT:-true}"
# export LOCUST_HOST="http://192.168.31.15:35696"

dir="$(cd "$(dirname "$0")" && pwd)"
bash "$dir/chaos_service.sh" "$service" "$total_count" "${faults[@]}"
