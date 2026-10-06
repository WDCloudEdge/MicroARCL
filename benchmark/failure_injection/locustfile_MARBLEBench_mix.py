#!/usr/bin/env python3
"""Combined MARBLEBench workload: run both user types at a configurable ratio.

This locustfile imports both workloads so a single ``locust`` run spawns them
together:

* ``MARBLEBenchUser``      — hand-written natural-language *intent* prompts
                             (``locustfile_MARBLEBench.py``)
* ``MARBLEBenchOrigUser``  — verbatim *original* MultiAgentBench task prompts
                             (``locustfile_MARBLEBench_orig.py``)

Locust spawns users of each class in proportion to their ``weight``. Since both
classes share the same ``wait_time`` and issue one request per task, the user
ratio is also (approximately) the request ratio. Configure it with:

    MARBLE_WEIGHT_INTENT   weight for the intent workload      (default 1)
    MARBLE_WEIGHT_ORIG     weight for the original workload     (default 1)

Examples:
    # 1:1 (default) — half intent, half original
    locust -f locustfile_MARBLEBench_mix.py --host=... -u 10 -r 10

    # 3:1 — 75% intent, 25% original
    MARBLE_WEIGHT_INTENT=3 MARBLE_WEIGHT_ORIG=1 \
        locust -f locustfile_MARBLEBench_mix.py --host=... -u 8 -r 10

    # original-only
    MARBLE_WEIGHT_INTENT=0 MARBLE_WEIGHT_ORIG=1 locust -f ... -u 10

Note: pick ``-u`` (total users) large enough for the ratio to be realizable —
with ``-u 1`` only a single user spawns and the ratio cannot show.
"""

# Importing the classes makes locust discover both in this module's namespace.
from benchmark.failure_injection.locustfile_MARBLEBench import MARBLEBenchUser  # noqa: F401
from benchmark.failure_injection.locustfile_MARBLEBench_orig import MARBLEBenchOrigUser  # noqa: F401
