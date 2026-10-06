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
        # ================================================================
        # MARBLE Research: 论文检索、作者背景、arXiv/标题查询、网页抓取
        # 能力面: related / recent / keyword / author / arxiv_id / title /
        #        fetch_webpage
        # ================================================================
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "我最近在研究大语言模型的多智能体协作(large language model "
                "multi-agent collaboration)，帮我找几篇这个方向比较有代表性的"
                "论文，最好带上标题、摘要和链接。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "我想了解扩散模型在图像生成方面的研究(diffusion models for "
                "image generation)，帮我找几篇相关的工作看看。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "强化学习(reinforcement learning)这块最近有什么新进展？"
                "帮我看看这个领域最新发表的几篇论文。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "帮我关注一下向量数据库与近似最近邻检索(vector database and "
                "approximate nearest neighbor search)方向最新出来的论文。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "我想找几篇讲“检索增强生成 (retrieval augmented generation)”的"
                "代表性论文，帮我按这个主题搜一下。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "帮我按“混合专家模型 (mixture of experts)”这个关键词找几篇论文。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "我想了解 Yoshua Bengio 的研究，帮我梳理一下他的主要论文"
                "和常见的合作者。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "帮我看看 Geoffrey Hinton 都发过哪些有影响力的论文，"
                "以及他经常和谁合作。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "有个 arXiv 编号是 1706.03762，帮我查一下这篇论文讲的是什么。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "帮我查一下 arXiv 1810.04805 这篇论文的题目和摘要。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "有一篇论文叫《Attention Is All You Need》，帮我找到它并给出详细信息。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "帮我找一下标题是《Deep Residual Learning for Image Recognition》"
                "的那篇论文。"
            ),
        },
        {
            "name": "agent-network-marble-research",
            "type": "SIMPLE",
            "prompt": (
                "帮我把 https://arxiv.org/abs/1706.03762 这个页面读一下，"
                "正文和基本信息整理出来给我。"
            ),
        },

        # ================================================================
        # MARBLE Web: 网页抓取、正文清洗、请求级缓存
        # 能力面: fetch_webpage
        # ================================================================
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "帮我把 https://www.python.org/about/ 这个页面的正文抓下来，"
                "去掉杂七杂八的东西，只留干净的内容。"
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "帮我读一下 RFC 9110 这份规范，网址是 "
                "https://www.rfc-editor.org/rfc/rfc9110，把主要内容提取出来。"
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "https://en.wikipedia.org/wiki/Multi-agent_system 这个维基页面"
                "大概讲了什么？帮我抓下正文看看。"
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "帮我把 https://httpbin.org/html 这个测试页面的内容取下来。"
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "我想看看 https://fastapi.tiangolo.com/ 首页介绍了些什么，"
                "帮我抓一下正文。"
            ),
        },
        {
            "name": "agent-network-marble-web",
            "type": "SIMPLE",
            "prompt": (
                "帮我抓取 https://docs.python.org/3/whatsnew/3.12.html 这个页面，"
                "总结下里面的更新点。"
            ),
        },

        # ================================================================
        # MARBLE Coding: coder / reviewer / debugger / tester / analyst
        # 能力面: create_solution / give_advice_and_revise / analyze_task /
        #        assign_roles / decompose_task / create_file / read_solution
        # (执行类 run_and_debug/run_tests/create_sandbox 依赖
        #  ALLOW_CODE_EXECUTION，生产可能关闭，这里不主动触发)
        # ================================================================
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "用 Python 帮我写一个稳定的归并排序，要能正确处理有重复元素的列表，"
                "存成 merge_sort.py，已有同名文件就覆盖。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "帮我实现一个线程安全、支持过期时间(TTL)的 LRU 缓存，"
                "写到 ttl_cache.py 里，可以覆盖旧文件。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "帮我用 Python 写个函数，判断一个字符串是不是有效的括号匹配，"
                "保存为 valid_parentheses.py。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "我想给一个异步 HTTP 客户端加上指数退避重试和熔断机制，"
                "帮我分析一下这个任务的复杂度、主要方向和潜在风险。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "帮我评估下“把一个同步爬虫改造成基于 asyncio 的并发版本”"
                "这件事难在哪、要注意什么。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "我要开发一个带鉴权、缓存和审计日志的 REST API，"
                "帮我把它拆成可执行的子任务，分别交给 coder、reviewer、tester 三种角色。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "现在有一个 Python 数据清洗工具要做，可用的人手有 coder、reviewer、"
                "debugger、tester、analyst，帮我安排一下谁负责哪部分。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "帮我评审并改进一下 merge_sort.py 里的归并排序，"
                "重点看空列表、重复元素和排序稳定性这几点。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "在工作区里新建一个 notes.txt，内容写上：MARBLEBench load test artifact。"
            ),
        },
        {
            "name": "agent-network-marble-coding",
            "type": "SIMPLE",
            "prompt": (
                "把工作区里那份 merge_sort.py 的代码读出来给我看看。"
            ),
        },

        # ================================================================
        # MARBLE Database: 基于 ReAct 的只读 MySQL 自然语言查询 + 健康检查
        # 能力面: react_query / health
        # ================================================================
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "帮我看看数据库里都有哪些表，每张表大概是干什么用的。"
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "先弄清楚数据库的结构，再统计一下每张业务表里有多少条记录。"
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "数据库里哪些表带时间字段？帮我按表看看各自最新几条记录的时间范围。"
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "帮我梳理一下各张表的主键、索引和可能的关联字段，"
                "给我一个简短的数据模型概览。"
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": (
                "帮我找出记录数最多的那几张表，说说它们大致存的是什么数据。"
            ),
        },
        {
            "name": "agent-network-marble-database",
            "type": "SIMPLE",
            "prompt": "帮我确认一下数据库现在连得上、是不是健康的。",
        },

        # ================================================================
        # MARBLE World: 二手车谈判(无状态动作)
        # 能力面: offer_price / reject_and_counter / accept_offer /
        #        provide_information / inquire_intentions / end_negotiation
        # ================================================================
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "我是买家，想买这台二手车，参考了同配置车近期的成交价，"
                "打算出价 12000，帮我把这个报价给出去。"
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "我是卖家，觉得买家现在的报价太低了，不能接受，"
                "车况好、里程也低，我还价 13800。"
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": "我是买家，对方这个报价我可以接受，就这么成交吧。",
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "我是卖家，跟买家说明一下车辆历史：没出过大事故，"
                "每年都在授权门店保养。"
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "我是买家，想问问卖家：如果今天就能定下来，价格还能再让多少？"
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": (
                "我是卖家，跟买家提供一下市场行情：同款车最近几笔成交价在 13000 上下。"
            ),
        },
        {
            "name": "agent-network-marble-world",
            "type": "SIMPLE",
            "prompt": "我是卖家，这轮聊不拢，先结束这次谈判吧。",
        },

        # ================================================================
        # MARBLE Minecraft: 环境查询 + 控制器动作
        # 能力面: get_environment_info / scanNearbyEntities / navigateTo /
        #        MineBlock / placeBlock / equipItem / handoverBlock /
        #        withdrawItem / erectDirtLadder / dismantleDirtLadder /
        #        fetchContainerContents
        # ================================================================
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": (
                "帮我看看玩家 player1 现在的情况——时间、天气、所在位置、"
                "生命值、饥饿值还有背包里都有什么。"
            ),
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": (
                "让 player1 在附近 10 格范围内找找橡木原木(Oak Log)，最多找 3 个。"
            ),
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 走到坐标 x=1、y=64、z=1 那个位置。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 把坐标 x=2、y=63、z=2 处的方块挖掉。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 在坐标 x=2、y=64、z=2 处放一块泥土(dirt)。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 把木镐(wooden pickaxe)拿到手上装备好。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 把 1 个泥土(dirt)交给玩家 bob。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 从箱子(chest)里取出 1 个铁锭(iron ingot)。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 搭一条通往顶部坐标 x=4、y=68、z=4 的泥土梯。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "让 player1 把通往顶部坐标 x=4、y=68、z=4 的那条泥土梯拆掉。",
        },
        {
            "name": "agent-network-marble-minecraft",
            "type": "SIMPLE",
            "prompt": "帮我看看 player1 自己背包(inventory)里现在都有些什么东西。",
        },

        # ================================================================
        # MARBLE Werewolf: 狼人杀角色事件决策
        # 能力面: decide_action / speak / vote / kill
        # ================================================================
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "我在玩狼人杀，我的身份是预言家。现在是第一晚要查验一名玩家，"
                "场上活着的有 player_1、player_2、player_3、player_4，我还没查过任何人，"
                "我想优先查发言有矛盾的人，帮我决定今晚查谁。"
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "我是村民，现在是第二天白天讨论环节，昨晚 player_5 出局了，"
                "帮我基于目前公开的信息组织一段谨慎点的发言，重点关注 player_1、"
                "player_2、player_3。"
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "我是村民，到白天投票了，我比较怀疑 player_2，其次是 player_3，"
                "帮我在这两个人里决定投谁出局。"
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "我是狼人，队友是 player_4。今晚要动手，场上还活着 player_1、"
                "player_2、player_3，我想优先干掉最可能是预言家的那个，帮我决定刀谁。"
            ),
        },
        {
            "name": "agent-network-marble-werewolf",
            "type": "SIMPLE",
            "prompt": (
                "我是女巫，现在第一晚，解药和毒药都还在手上，今晚 player_3 被刀了，"
                "帮我决定要不要用药、怎么用。"
            ),
        },

        # ================================================================
        # 跨智能体任务: 更长的调用链，用于观察故障传播
        # ================================================================
        {
            "name": "marble-combo-web-research",
            "type": "COMPLEX",
            "services": [
                "agent-network-marble-web",
                "agent-network-marble-research",
            ],
            "prompt": (
                "先帮我把 https://arxiv.org/abs/1706.03762 这篇论文的网页正文取下来，"
                "再根据里面讲的主题去找 5 篇相关的论文。"
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
                "先帮我找几篇讲 LRU 缓存的资料，然后照着这些思路用 Python 实现一个"
                "带 TTL 的 LRU 缓存，存成 researched_cache.py，可以覆盖旧文件。"
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
                "先只读地看看数据库的表和索引大概什么样，"
                "再帮我分析下这些表可能需要哪些查询优化，别改动任何数据。"
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
                "先帮我找几篇讲自动谈判策略(automated negotiation strategy)的"
                "论文，然后我以买家身份出价 12000，报价理由就参考这些策略。"
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
                "先帮我找些社交推理游戏(social deduction game)决策方面的论文，"
                "然后我作为村民，在白天讨论里定一个发言策略。"
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
                "先帮我抓取 https://httpbin.org/html 这个页面，"
                "再分析一下要做一个健壮的 HTML 正文抽取器该怎么设计，"
                "解析、编码和异常处理这些点都要考虑到。"
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
                "第一步帮我找到《Attention Is All You Need》这篇论文，"
                "第二步把它的论文页面抓下来，"
                "第三步把“复现论文里的核心注意力模块”这个目标拆成可执行的子任务，"
                "分别交给 coder、reviewer、tester。"
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
                "第一步只读地查一下各张数据表大概多大规模，"
                "第二步评估下“生成一份报价摘要”这个功能实现起来有多复杂，"
                "第三步我以买家身份出价 10000，理由就基于前面的数据分析。"
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
                "先帮我找几篇 Minecraft 早期资源采集策略(early-game resource "
                "gathering in Minecraft)相关的资料，然后让玩家 player1 在附近 12 格"
                "范围内找找橡木原木(Oak Log)，最多找 3 个。"
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
                "先看看玩家 player1 当前的环境和背包情况，"
                "再分析一下怎么根据这些状态做一个安全的资源采集决策器。"
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
