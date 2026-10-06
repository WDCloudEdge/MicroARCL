#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

locust \
  -f "$SCRIPT_DIR/locustfile_MARBLEBench.py" \
  --host="http://192.168.31.15:35696" \
  --only-summary \
  -u 1 \
  -r 10 \
  >> "$SCRIPT_DIR/task.log" 2>&1
