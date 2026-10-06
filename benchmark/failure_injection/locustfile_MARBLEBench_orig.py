#!/usr/bin/env python3
"""MARBLEBench workload driven by the *original* MultiAgentBench tasks."""

import json
import os
import random
import uuid

import locust.stats
from locust import HttpUser, between, task


locust.stats.CSV_STATS_INTERVAL_SEC = 5
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 5


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# The chaos runner sets this variable for each fault-injection window.
FAULT_SERVICE = os.getenv("FAULT_SERVICE", "agent-network-marble-research")

# Max tasks randomly sampled per scenario (0 = keep every task in the file).
MAX_TASKS_PER_SCENARIO = int(os.getenv("MARBLE_MAX_TASKS_PER_SCENARIO", "20"))

# Optional seed for the per-scenario random sampling (empty = non-deterministic).
_SAMPLE_SEED = os.getenv("MARBLE_SAMPLE_SEED", "").strip()
_SAMPLER = random.Random(int(_SAMPLE_SEED)) if _SAMPLE_SEED else random

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_BENCH_DIR = os.path.normpath(
    os.path.join(_HERE, "../../../MARBLE/multiagentbench")
)
MARBLE_BENCH_DIR = os.getenv("MARBLE_BENCH_DIR", _DEFAULT_BENCH_DIR)

W_FAULT_COMPLEX = 6
W_FAULT_SIMPLE = 3
W_OTHER_COMPLEX = 2
W_OTHER_SIMPLE = 1


# scenario -> (service name, relative jsonl path, task "type" for weighting)
SCENARIO_SPECS = {
    "research": (
        "agent-network-marble-research",
        "research/research_main.jsonl",
        "COMPLEX",
    ),
    "coding": (
        "agent-network-marble-coding",
        "coding/coding_main.jsonl",
        "COMPLEX",
    ),
    # "database": (
    #     "agent-network-marble-database",
    #     "database/database_main.jsonl",
    #     "COMPLEX",
    # ),
    "bargaining": (
        "agent-network-marble-world",
        "bargaining/bargaining_main.jsonl",
        "SIMPLE",
    ),
    "minecraft": (
        "agent-network-marble-minecraft",
        "minecraft/minecraft_main.jsonl",
        "SIMPLE",
    ),
}


# Minimal, format-faithful fallbacks used only when the JSONL files are absent.
# They mirror the shape of the upstream ``task.content`` for each scenario.
FALLBACK_TASKS = {
    "research": (
        "Dear Research Team,\n\n"
        "You are collaborating to generate a new research idea based on the "
        "following Introduction:\n\n**Introduction**\n\n"
        "Federated learning (FL) can leverage distributed user data while "
        "preserving privacy, but heterogeneous client data and limited "
        "communication remain open challenges. Propose a concrete, novel and "
        "well-motivated research idea that advances this direction, and outline "
        "the method, expected experiments and evaluation."
    ),
    "coding": (
        "Software Development Task:\n\n"
        "Please write a software system called TaskBoard that lets teams create "
        "projects, add tasks, assign owners and track status.\n"
        "1. Implementation requirements:\n"
        "   - User and project management must be built first.\n"
        "   - A task CRUD module with status transitions built next.\n"
        "   - A reporting module summarizing progress built last.\n\n"
        "2. Project structure:\n   - solution.py (main implementation)\n\n"
        "3. Development process:\n   - Developer: Create the code.\n"
        "   - Developer: Revise the code.\n   - Developer: Optimize the code.\n\n"
        "Please work together following software engineering best practices."
    ),
    # "database": (
    #     "This database powers an e-commerce platform (users, products, orders, "
    #     "payments). Recently the database has seen performance issues. Use SQL "
    #     "queries to find out what is wrong and the root cause. The root cause "
    #     "can be only two of: 'INSERT_LARGE_DATA', 'MISSING_INDEXES', "
    #     "'LOCK_CONTENTION', 'VACUUM', 'REDUNDANT_INDEX', 'FETCH_LARGE_DATA'. "
    #     "Assign a different agent to analyze each candidate cause, share "
    #     "findings, and only decide after exploring all of them. You have "
    #     "read-only access to pg_stat_statements, pg_locks, "
    #     "pg_stat_user_indexes, pg_indexes, pg_stat_all_tables, "
    #     "pg_stat_progress_vacuum and pg_stat_user_tables."
    # ),
    "bargaining": (
        "Welcome to the negotiation table for a Replacement Remote Control for "
        "a HISENSE AC. As a buyer, your focus is on value and premium features; "
        "the seller is looking to secure a higher profit margin. Both parties "
        "have their own priorities to navigate to reach a mutually beneficial "
        "agreement. The agents are encouraged to actively use the tools "
        "provided, such as offering a price, rejecting and countering offers, "
        "providing information, or ending the negotiation, to achieve their "
        "respective goals."
    ),
    "minecraft": (
        "This is in the game of Minecraft. Build a building according to a "
        "blueprint. The blueprint contains necessary information about the "
        "material, facing direction and position of each block.\n"
        "*** The blueprint ***\n"
        "[\n"
        '    "material: cut_sandstone facing: A position: [-8, -60, 0]",\n'
        '    "material: terracotta facing: A position: [-8, -59, 0]",\n'
        '    "material: torch facing: A position: [-8, -58, 0]"\n'
        "]"
    ),
}


def _load_scenario_prompts(rel_path):
    """Return a random sample of verbatim ``task.content`` prompts for a scenario.

    All tasks in the JSONL file are read, then a random subset of up to
    ``MAX_TASKS_PER_SCENARIO`` is drawn (0 = keep every task).
    """
    path = os.path.join(MARBLE_BENCH_DIR, rel_path)
    prompts = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                content = (record.get("task") or {}).get("content")
                if content and content.strip():
                    prompts.append(content.strip())
    except (OSError, json.JSONDecodeError):
        return []
    if MAX_TASKS_PER_SCENARIO and len(prompts) > MAX_TASKS_PER_SCENARIO:
        prompts = _SAMPLER.sample(prompts, MAX_TASKS_PER_SCENARIO)
    return prompts


def build_task_definitions():
    """Build the weighted task table from the upstream benchmark tasks."""
    definitions = []
    for scenario, (service, rel_path, task_type) in SCENARIO_SPECS.items():
        prompts = _load_scenario_prompts(rel_path)
        if not prompts:
            fallback = FALLBACK_TASKS.get(scenario)
            prompts = [fallback] if fallback else []
        for prompt in prompts:
            definitions.append(
                {
                    "name": service,
                    "scenario": scenario,
                    "type": task_type,
                    "services": [service],
                    "prompt": prompt,
                }
            )
    return definitions


def task_services(task_def):
    """Return the service chain expected to handle a task."""
    return task_def.get("services", [task_def["name"]])


def task_weight(task_def):
    """Bias traffic toward the service under fault injection."""
    hits_fault = FAULT_SERVICE in task_services(task_def)
    is_complex = task_def.get("type") == "COMPLEX"
    if hits_fault and is_complex:
        return W_FAULT_COMPLEX
    if hits_fault:
        return W_FAULT_SIMPLE
    if is_complex:
        return W_OTHER_COMPLEX
    return W_OTHER_SIMPLE


class MARBLEBenchOrigUser(HttpUser):
    # Share of spawned users that run the original-benchmark workload. The ratio
    # against MARBLEBenchUser (intent workload) is set via MARBLE_WEIGHT_ORIG /
    # MARBLE_WEIGHT_INTENT when both classes run together.
    weight = int(os.getenv("MARBLE_WEIGHT_ORIG", "1"))
    wait_time = between(5, 10)

    TASK_DEFINITIONS = build_task_definitions()
    TASK_WEIGHTS = [task_weight(item) for item in TASK_DEFINITIONS]

    @task
    def run_mixed_tasks(self):
        task_def = random.choices(
            self.TASK_DEFINITIONS,
            weights=self.TASK_WEIGHTS,
            k=1,
        )[0]

        prompt = task_def["prompt"]
        payload = {
            "flowId": "@cn.com.thingo.intelligentAgentPlatform.taskScheduling/FLOW_OPEN_TASK",
            "params": {
                "openTask": {
                    "userId": "CDA7B6E29E2B4CBBB53E424A68A1B85C",
                    "organizeId": "C122FBF575E94BF8867580A535E20BD3",
                },
                "task": prompt,
            },
        }

        request_id = uuid.uuid4()
        task_name = task_def["name"]
        task_type = task_def["type"]
        # Prompts (e.g. research intros) can be long; log only a short preview.
        preview = " ".join(prompt.split())[:120]
        print(f"id:{request_id} {task_type}/{task_name} | {preview}")

        with self.client.post(
            "/api/engine/flow",
            json=payload,
            name=f"{task_type}/{task_name}",
            catch_response=True,
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"Status {response.status_code} for {task_name}"
                )
