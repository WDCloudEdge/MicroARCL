#!/usr/bin/env python3
"""MARBLEBench workload for the task-scheduling flow."""

import os
import random
import uuid

import locust.stats
from locust import HttpUser, between, task


locust.stats.CSV_STATS_INTERVAL_SEC = 5
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 5


# The chaos runner sets this variable for each fault-injection window.
FAULT_SERVICE = os.getenv(
    "FAULT_SERVICE", "agent-network-marble-research"
)

W_FAULT_COMPLEX = 6
W_FAULT_SIMPLE = 3
W_OTHER_COMPLEX = 2
W_OTHER_SIMPLE = 1


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


class MARBLEBenchUser(HttpUser):
    # Share of spawned users that run the intent workload. The ratio against
    # MARBLEBenchOrigUser (original-benchmark workload) is set via
    # MARBLE_WEIGHT_INTENT / MARBLE_WEIGHT_ORIG when both classes run together.
    weight = int(os.getenv("MARBLE_WEIGHT_INTENT", "1"))
    wait_time = between(5, 10)

    TASK_DEFINITIONS = [
        # Research tasks.
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "I've recently been studying large language model multi-agent "
                "collaboration. Help me find a few representative papers in this "
                "direction, preferably with titles, abstracts, and links."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "I want to learn about diffusion models for image generation. "
                "Help me find a few relevant works to look at."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "What recent progress has there been in reinforcement learning? "
                "Help me check out a few of the latest papers published in this field."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "Help me keep an eye on the latest papers coming out on vector "
                "databases and approximate nearest neighbor search."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "I want to find a few representative papers on retrieval augmented "
                "generation. Help me search by this topic."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "Help me find a few papers by the keyword 'mixture of experts'."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "I want to learn about Yoshua Bengio's research. Help me put "
                "together his main papers and his frequent collaborators."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "Help me check which influential papers Geoffrey Hinton has "
                "published, and who he often collaborates with."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "There's an arXiv ID 1706.03762. Help me look up what this paper is about."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "Help me look up the title and abstract of arXiv 1810.04805."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "There's a paper titled 'Attention Is All You Need'. Help me find it and give detailed information."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "Help me find the paper titled 'Deep Residual Learning for Image Recognition'."
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "Help me read the page at https://arxiv.org/abs/1706.03762, "
                "and organize the body text and basic information for me."
            ),
        },

        # Web tasks.
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "Help me fetch the body text of the page https://www.python.org/about/, "
                "strip out the clutter, and keep only the clean content."
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "Help me read the RFC 9110 spec, the URL is "
                "https://www.rfc-editor.org/rfc/rfc9110, and extract the main content."
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "What is the wiki page https://en.wikipedia.org/wiki/Multi-agent_system "
                "roughly about? Help me fetch the body text and take a look."
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "Help me fetch the content of the test page https://httpbin.org/html."
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "I want to see what the homepage https://fastapi.tiangolo.com/ introduces. "
                "Help me fetch the body text."
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "Help me fetch the page https://docs.python.org/3/whatsnew/3.12.html "
                "and summarize the update points in it."
            ),
        },

        # Coding tasks avoid execution tools controlled by ALLOW_CODE_EXECUTION.
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Help me write a stable merge sort in Python that correctly handles "
                "lists with duplicate elements. Save it as merge_sort.py, overwriting "
                "any existing file with the same name."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Help me implement a thread-safe LRU cache with TTL support, "
                "write it to ttl_cache.py, and feel free to overwrite the old file."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Help me write a Python function that checks whether a string is "
                "valid parentheses matching, and save it as valid_parentheses.py."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "I want to add exponential-backoff retries and a circuit breaker to an "
                "async HTTP client. Help me analyze the complexity, main directions, "
                "and potential risks of this task."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Help me assess 'converting a synchronous crawler into an asyncio-based "
                "concurrent version' — what's hard about it and what should I watch out for."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "I want to develop a REST API with authentication, caching, and audit "
                "logging. Help me break it into executable subtasks and assign them to "
                "the three roles coder, reviewer, and tester respectively."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "There's a Python data-cleaning tool to build, and the available staff "
                "are coder, reviewer, debugger, tester, and analyst. Help me arrange "
                "who is responsible for which part."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Help me review and improve the merge sort in merge_sort.py, focusing "
                "on empty lists, duplicate elements, and sort stability."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Create a new notes.txt in the workspace with the content: MARBLEBench load test artifact."
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "Read out the code of that merge_sort.py in the workspace and show it to me."
            ),
        },

        # Database tasks.
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "Help me see what tables are in the database and roughly what each table is used for."
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "First figure out the structure of the database, then count how many records are in each business table."
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "Which tables in the database have time fields? Help me check, per table, "
                "the time range of the latest few records."
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "Help me sort out the primary keys, indexes, and possible relation fields "
                "of each table, and give me a short data-model overview."
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "Help me find the few tables with the most records and tell me roughly what data they store."
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": "Help me confirm that the database is currently reachable and healthy.",
        },

        # Negotiation tasks.
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "I'm the buyer and I want to buy this used car. Having referenced recent "
                "transaction prices of similarly configured cars, I plan to offer 12000. "
                "Help me put this offer out."
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "I'm the seller and I think the buyer's current offer is too low to accept. "
                "The car is in good condition and low mileage, so I counter at 13800."
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": "I'm the buyer, I can accept the other party's offer, let's close the deal.",
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "I'm the seller. Explain the vehicle history to the buyer: no major "
                "accidents, serviced at an authorized dealer every year."
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "I'm the buyer and I want to ask the seller: if we can settle today, how much more can you come down on the price?"
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "I'm the seller. Provide the buyer with market conditions: recent deals on the same model are around 13000."
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": "I'm the seller, we can't reach agreement this round, let's end this negotiation for now.",
        },

        # Minecraft tasks.
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": (
                "Help me check the current status of player player1 — time, weather, "
                "location, health, hunger, and what's in the inventory."
            ),
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": (
                "Have player1 look for Oak Log within a 10-block range nearby, up to 3 of them."
            ),
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 walk to the position at coordinates x=1, y=64, z=1.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 mine the block at coordinates x=2, y=63, z=2.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 place a dirt block at coordinates x=2, y=64, z=2.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 take the wooden pickaxe in hand and equip it.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 hand over 1 dirt to player bob.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 withdraw 1 iron ingot from the chest.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 build a dirt ladder up to the top coordinates x=4, y=68, z=4.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Have player1 dismantle that dirt ladder leading up to the top coordinates x=4, y=68, z=4.",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "Help me check what's currently in player1's own inventory.",
        },

        # Werewolf tasks.
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "I'm playing Werewolf and my role is the Seer. It's now the first night "
                "and I need to inspect a player. The players still alive are player_1, "
                "player_2, player_3, player_4, and I haven't inspected anyone yet. I want "
                "to prioritize whoever has contradictions in their statements — help me decide who to inspect tonight."
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "I'm a Villager. It's now the day-two discussion phase and player_5 was "
                "eliminated last night. Help me compose a cautious statement based on the "
                "currently public information, focusing on player_1, player_2, and player_3."
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "I'm a Villager and it's the daytime vote. I'm fairly suspicious of "
                "player_2, then player_3. Help me decide which of these two to vote out."
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "I'm a Werewolf and my teammate is player_4. We're acting tonight, and "
                "still alive are player_1, player_2, player_3. I want to prioritize killing "
                "the one most likely to be the Seer — help me decide who to kill."
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "I'm the Witch. It's the first night and I still have both the antidote "
                "and the poison in hand. player_3 was killed tonight — help me decide "
                "whether and how to use my potions."
            ),
        },

        # Cross-agent call chains.
        {
            "name": "marble-combo-web-research",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-web",
                "agent-network-marble-research",
            ],
            "prompt": (
                "First help me fetch the webpage body text of the paper at "
                "https://arxiv.org/abs/1706.03762, then find 5 related papers based on the topic it discusses."
            ),
        },
        {
            "name": "marble-combo-research-coding",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-research",
                "agent-network-marble-coding",
            ],
            "prompt": (
                "First help me find a few resources about LRU caches, then following "
                "those ideas implement a TTL-enabled LRU cache in Python, saving it as "
                "researched_cache.py, and feel free to overwrite the old file."
            ),
        },
        {
            "name": "marble-combo-database-coding",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-database",
                "agent-network-marble-coding",
            ],
            "prompt": (
                "First take a read-only look at roughly what the database's tables and "
                "indexes look like, then help me analyze what query optimizations these "
                "tables might need, without modifying any data."
            ),
        },
        {
            "name": "marble-combo-world-research",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-research",
                "agent-network-marble-world",
            ],
            "prompt": (
                "First help me find a few papers on automated negotiation strategy, then "
                "I'll offer 12000 as the buyer, with the reasoning for the offer drawing on those strategies."
            ),
        },
        {
            "name": "marble-combo-werewolf-research",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-research",
                "agent-network-marble-werewolf",
            ],
            "prompt": (
                "First help me find some papers on decision-making in social deduction "
                "games, then, as a Villager, set a speaking strategy for the daytime discussion."
            ),
        },
        {
            "name": "marble-combo-web-coding",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-web",
                "agent-network-marble-coding",
            ],
            "prompt": (
                "First help me fetch the page https://httpbin.org/html, then analyze how "
                "to design a robust HTML body-text extractor, taking parsing, encoding, "
                "and exception handling into account."
            ),
        },
        {
            "name": "marble-combo-research-web-coding",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-research",
                "agent-network-marble-web",
                "agent-network-marble-coding",
            ],
            "prompt": (
                "Step 1: help me find the paper 'Attention Is All You Need'; "
                "Step 2: fetch its paper page; "
                "Step 3: break the goal of 'reproducing the core attention module in the "
                "paper' into executable subtasks, assigned to coder, reviewer, and tester respectively."
            ),
        },
        {
            "name": "marble-combo-database-coding-world",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-database",
                "agent-network-marble-coding",
                "agent-network-marble-world",
            ],
            "prompt": (
                "Step 1: read-only, check roughly how large each data table is; "
                "Step 2: assess how complex it would be to implement a 'generate an offer "
                "summary' feature; "
                "Step 3: as the buyer, I offer 10000, with the reasoning based on the earlier data analysis."
            ),
        },
        {
            "name": "marble-combo-research-minecraft",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-research",
                "agent-network-marble-minecraft",
            ],
            "prompt": (
                "First help me find a few resources on early-game resource gathering in "
                "Minecraft, then have player player1 look for Oak Log within a 12-block "
                "range nearby, up to 3 of them."
            ),
        },
        {
            "name": "marble-combo-minecraft-coding",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-minecraft",
                "agent-network-marble-coding",
            ],
            "prompt": (
                "First check player player1's current environment and inventory, then "
                "analyze how to build a safe resource-gathering decision maker based on these states."
            ),
        },
    ]

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
        print(f"id:{request_id} {task_type}/{task_name} | {prompt}")

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
