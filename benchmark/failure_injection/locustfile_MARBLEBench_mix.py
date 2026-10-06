#!/usr/bin/env python3
"""Combined MARBLEBench workload: run both user types at a configurable ratio."""

# Importing the classes makes locust discover both in this module's namespace.
from benchmark.failure_injection.locustfile_MARBLEBench import MARBLEBenchUser  # noqa: F401
from benchmark.failure_injection.locustfile_MARBLEBench_orig import MARBLEBenchOrigUser  # noqa: F401
