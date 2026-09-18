"""
Module C (part 2): execution-graph driven fine-grained localization.

Takes the aggregated top-k evidence (from execution_graph.aggregate_topk_slices)
and asks an LLM (OpenAI-compatible Chat Completions API) to localize the
fine-grained root cause and failure stage, returning structured JSON.

Design goals:
- OpenAI-compatible: the user only needs to supply `sk` + `base_url`.
- Dependency-light: uses the `openai` SDK if installed, else falls back to a
  plain urllib HTTP POST, so no hard dependency is required.
- Mockable/offline: `cfg.llm_mock=True` (or missing key) -> deterministic
  rule-based localization derived from the shared error signature, so the
  whole pipeline runs end-to-end without network access.
"""
import os
import json
import urllib.request
from typing import Dict

_SYSTEM_PROMPT = (
    "你是资深的 Agent 系统故障诊断专家。给定一次故障时间窗内的粗粒度候选根因服务"
    "（候选顺序由 GNN 或 severity 粗排产生，并同时提供两类分数）及其在执行图中的失败证据，"
    "请判定更细粒度的根因。判定规则（务必遵守）：\n"
    "0) 证据优先级：第一级是 `gnn_service_ranking` 与 `coarse_ranking`/`coarse_evidence`，"
    "前者表示拓扑传播和上下游因果可疑度，后者表示指标直接严重程度；"
    "第二级是执行图失败位置和空结果；第三级才是日志模板。"
    "日志只是辅助证据，根因不一定会在日志中显式出现；缺少日志 signature 不能降低或否定 GNN/severity 候选。\n"
    "1) GNN 和 severity 同时指向同一服务时，应视为最强的单服务根因候选。"
    "两者冲突时，结合服务是否位于失败传播上游、severity 各分量（特别是 res_dev）和执行图失败起点裁决，"
    "并在 reason 中解释为什么选择其中一个。\n"
    "2) `signature` 是从当前系统日志动态抽取的模板，不是预定义错误类别。"
    "高频且与 GNN/severity/执行图一致的模板可提升置信度；与一级证据冲突的单服务日志不得单独推翻粗排结果。"
    "只有同一模板在多个独立服务上出现，且能解释候选服务的共同失效时，才可判为 `systemic`。\n"
    "3) 若 `shared_signatures` 为空：**不要**仅凭空结果/级联失败判为 systemic。"
    "应回到 GNN/severity 的高排名候选，把空结果视为可能的下游症状；资源型故障可以完全没有错误日志。\n"
    "4) root_cause 要给出**具体服务名**（single_service 时）或具体共享依赖（systemic 时），"
    "并与 reranked_topk[0] 保持一致。\n"
    "仅输出一个 JSON 对象，不要输出多余文字。字段："
    '{"root_cause":"","root_cause_type":"systemic|single_service",'
    '"failure_stage":"planning|reasoning|tool-invocation|response-summary",'
    '"reason":"","evidence":[],"confidence":0.0,"reranked_topk":[],"suggested_fix":""}'
)

_EVIDENCE_FIELD_GUIDE = (
    "\n\n输入 JSON 字段含义（必须按这些定义解读，不能自行猜测）：\n"
    "A. `global.num_traces` 是用于详细 vertex 切片的 trace 数（可受上下文上限截断）；"
    "`num_trace_analysis_graphs` 是调用链频率统计实际遍历的全部候选相关 trace 数；"
    "`num_failed_traces` 是详细切片 trace 中候选服务 vertex 失败的 trace 数；"
    "`num_problem_traces` 是至少有一个 top-k 候选服务 vertex 失败的完整 trace 数；"
    "`failed_trace_examples` 只是少量文本样例，"
    "完整问题 trace 以 `problem_trace_analysis.traces` 为准。\n"
    "B. `coarse_ranking` 是指标严重程度排名。所有指标均优先使用异常窗前数据建立基线，"
    "再检测异常窗内偏差。`severity` = `fail_rate` + "
    "0.1*log1p(max(`latency_spike`-1,0)) + resource_weight*`res_dev` + "
    "qps_weight*`qps_dev`，数值越大越可疑；"
    "它是样本内相对分数，不是概率。`fail_rate` 是相对窗外成功率基线的平均下降；"
    "`n_fail_points` 是持续低于该基线阈值的采样点数；`latency_spike` 是异常窗峰值/窗外基线中位数；"
    "`qps_dev` 是QPS相对窗外基线的双向最大偏离；`res_dev` 是通过持续性尖峰检测后，"
    "CPU 或内存峰值相对窗外基线中位数的最大偏离；"
    "`cpu_dev`/`mem_dev` 分别表示各资源通道的偏离。`anomaly_types` 是观测到的表象类型全集；"
    "`anomaly_type_scores` 是每种表象对 severity 的实际加权贡献，其中 availability=fail_rate，"
    "latency=0.1*log1p(latency_spike-1)，cpu/memory共同分配resource_weight*res_dev，"
    "qps=qps_weight*qps_dev。\n"
    "C. `gnn_service_ranking` 是 GNN 综合指标异常、服务拓扑和故障传播后得到的服务排名；"
    "`gnn_score` 越大越可疑，只能在本次样本内部比较，不是概率。候选中的 `coarse_evidence` 同时给出"
    "该服务的 `gnn_rank/gnn_score` 与 `severity_rank/severity_score`；null 表示该证据源没有该服务。\n"
    "D. `matched_failed_vertexes` 是状态失败 trace 中候选服务的直接失败切片。`failed=true` 表示该服务自身"
    "messages、level_spans输入输出或level_routes传给下游的params中出现动态错误模板，或它是task_status=8"
    "对应的summarizer；`empty_results` 只作描述，visible和空输出都不能作为失败判据；"
    "`signatures/error_texts` 是执行图内错误证据。"
    "`status` 是原始执行状态码，不能脱离 empty_results/signatures 单独解释为成功或失败。\n"
    "E. `problem_trace_analysis` 中问题 trace 取两类判据的并集：task_status=3 是任务失败，7 是状态成功，"
    "8 是总结失败（状态错误归给summarizer）；status=3表示链路中途终止，错误归给实际链尾且不补summarizer；"
    "或者服务vertex中检测到错误签名。"
    "trace 入选后保留完整上下游链及链上所有失败 vertex。"
    "`vertex_chain` 按执行 level 从左到右排列，只保留 Planner 首节点及各 xxxGroup，不计 xxxGroup/xxAgent；"
    "`service_chain` 同样按上游到下游排列。仅task_status为7/8时，原始level_routes未记录的summarizer会补在链尾。"
    "`call_chain_frequencies.trace_count/trace_ratio` 是完全相同服务链出现的次数/占问题 trace 比例。"
    "`service_frequencies.traces_containing_service/trace_coverage` 是全部已加载trace中经过该服务的数量/比例；"
    "`problem_trace_count/problem_trace_ratio` 是问题trace中经过该服务的数量/比例；"
    "`chain_occurrences` 保留服务在链中非连续重复出现的次数；`status_error_count/status_total_count` 是状态通道，"
    "`vertex_error_count/vertex_total_count` 是错误签名通道；`combined_error_count` 和 `combined_total_count` 分别"
    "按trace_id对两路错误集合、总集合取并集去重，`combined_error_ratio` 是相除结果。`failed_trace_count` 是两路判据按trace取并集"
    "后的数量。高覆盖率只表示常经服务（如 planner），不能单独证明它是根因；"
    "应重点比较链上最早失败位置、失败频率及 GNN/severity。\n"
    "F. 日志字段仅为辅助证据。`log_signature_counts` 是异常窗内动态模板次数；"
    "`log_signature_frequency_per_minute` 是每分钟频率；`frequent_signatures` 是达到最低频次的模板；"
    "`shared_signatures` 只包含至少两个候选服务共同出现的模板；`log_errors` 是匹配模板的原始日志样例。\n"
    "G. 输出中 `failure_stage`：planning=规划/任务拆分，reasoning=模型推理，"
    "tool-invocation=服务或工具调用，response-summary=最终汇总/响应生成。\n"
)


def build_messages(agg: Dict) -> list:
    user = (
        "故障证据聚合如下（JSON）：\n"
        + json.dumps(agg, ensure_ascii=False, indent=2)
        + "\n请据此定位细粒度根因并输出规定的 JSON。"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT + _EVIDENCE_FIELD_GUIDE},
        {"role": "user", "content": user},
    ]


# ----------------------------- rule-based fallback -----------------------------
def rule_based_localize(agg: Dict) -> Dict:
    shared_list = agg.get('shared_signatures', [])
    candidates = agg.get('candidates', [])
    reranked = [c['service'] for c in candidates]
    # Use dynamic templates' service support and measured frequency directly;
    # no predefined root-cause/signature taxonomy is required.
    if shared_list:
        primary = shared_list[0]
        signature = primary.get('signature', '')
        services = primary.get('services', [])
        n = primary.get('num_services', len(services))
        occurrences = primary.get('log_occurrences', 0)
        # Logs may strengthen the topology/metric result, but must not replace
        # it on their own. The shared template must include the coarse top-1.
        corroborates_coarse = bool(reranked and reranked[0] in services)
        systemic = n >= 2 and corroborates_coarse
        if not systemic:
            shared_list = []
        else:
            root = f'共享日志故障：{signature[:180]}'
            promoted = services + [s for s in reranked if s not in services]
            return {
                'root_cause': root,
                'root_cause_type': 'systemic',
                'failure_stage': 'tool-invocation',
                'reason': (f"动态日志模板在 {n} 个服务中出现，"
                           f"异常窗内累计 {occurrences} 次，且覆盖粗排 top-1。"),
                'evidence': [f'signature={signature}',
                             f'num_services={n}',
                             f'log_occurrences={occurrences}',
                             f'per_service_counts={primary.get("per_service_log_counts", {})}',
                             f'corroborates_coarse_top1={corroborates_coarse}'],
                'confidence': 0.85,
                'reranked_topk': promoted,
                'suggested_fix': '根据原始日志样例和执行图上下文处理该高频故障模板。',
                '_mode': 'rule_based_dynamic_signature',
            }
    # No cross-service dynamic template -> defer to coarse top-1.
    top = candidates[0]['service'] if candidates else None
    return {
        'root_cause': top,
        'root_cause_type': 'single_service',
        'failure_stage': 'tool-invocation',
        'reason': '未发现与粗排 top-1 一致的跨服务日志模板，回到 GNN/severity top-1。',
        'evidence': [],
        'confidence': 0.3,
        'reranked_topk': reranked,
        'suggested_fix': '结合日志进一步确认。',
        '_mode': 'rule_based_fallback',
    }


# ----------------------------- OpenAI-compatible call -----------------------------
def _call_openai_compatible(messages, cfg) -> str:
    base_url = (cfg.llm_base_url or '').rstrip('/')
    api_key = cfg.llm_api_key or os.environ.get('OPENAI_API_KEY', '')
    if not base_url or not api_key:
        raise RuntimeError('missing llm_base_url or api_key')
    payload = {
        'model': cfg.llm_model,
        'messages': messages,
        'temperature': cfg.llm_temperature,
        'max_tokens': cfg.llm_max_tokens,
    }
    # prefer the openai SDK if present, else raw HTTP
    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(**payload)
        return resp.choices[0].message.content
    except ImportError:
        url = base_url + '/chat/completions'
        data = json.dumps(payload).encode('utf-8')
        req = urllib.request.Request(
            url, data=data,
            headers={'Content-Type': 'application/json',
                     'Authorization': 'Bearer ' + api_key})
        with urllib.request.urlopen(req, timeout=60) as r:
            body = json.loads(r.read().decode('utf-8'))
        return body['choices'][0]['message']['content']


def _extract_json(text: str) -> Dict:
    text = text.strip()
    # strip ```json fences if present
    if text.startswith('```'):
        text = text.strip('`')
        text = text[text.find('{'):]
    start, end = text.find('{'), text.rfind('}')
    if start != -1 and end != -1:
        return json.loads(text[start:end + 1])
    raise ValueError('no JSON object in LLM response')


def llm_localize(agg: Dict, cfg) -> Dict:
    """Localize fine-grained root cause. Falls back to rule-based on
    mock mode / missing credentials / any API or parse error."""
    if getattr(cfg, 'llm_mock', True):
        return rule_based_localize(agg)
    try:
        content = _call_openai_compatible(build_messages(agg), cfg)
        result = _extract_json(content)
        result['_mode'] = 'llm'
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[llm_localize] API/parse failed ({e}); falling back to rule-based.')
        res = rule_based_localize(agg)
        res['_mode'] = 'rule_based_after_error'
        res['_error'] = str(e)
        return res
