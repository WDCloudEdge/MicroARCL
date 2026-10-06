#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Ratio between the two workloads (intent : original). Override on the CLI, e.g.
#   MARBLE_WEIGHT_INTENT=3 MARBLE_WEIGHT_ORIG=1 ./locust_mix.sh
export MARBLE_WEIGHT_INTENT="${MARBLE_WEIGHT_INTENT:-1}"
export MARBLE_WEIGHT_ORIG="${MARBLE_WEIGHT_ORIG:-1}"

# Total users must be large enough for the ratio to be realizable.
USERS="${USERS:-3}"
SPAWN_RATE="${SPAWN_RATE:-10}"

locust \
  -f "$SCRIPT_DIR/locustfile_MARBLEBench_mix.py" \
  --host="http://192.168.31.15:35696" \
  --only-summary \
  -u "$USERS" \
  -r "$SPAWN_RATE" \
  >> "$SCRIPT_DIR/task.log" 2>&1
