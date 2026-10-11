"""execution-graph driven fine-grained localization."""
import os
import json
import urllib.request
from typing import Dict, List

_SYSTEM_PROMPT = (
    "You are a senior fault-diagnosis expert for Agent systems. Given the "
    "coarse-grained candidate root-cause services within a fault time window "
    "(candidate order is produced by GNN or severity coarse ranking, and both "
    "kinds of scores are provided) and their failure evidence in the execution "
    "graph, determine the finer-grained root cause. Decision rules (must follow):\n"
    "0) Evidence priority: the first tier is `gnn_service_ranking` and "
    "`coarse_ranking`/`coarse_evidence` -- the former reflects topological "
    "propagation and upstream/downstream causal suspicion, the latter reflects "
    "direct metric severity; the second tier is execution-graph failure "
    "positions and empty results; the third tier is log templates. "
    "Logs are only auxiliary evidence, and the root cause does not necessarily "
    "appear explicitly in logs; a missing log signature must not lower or negate "
    "a GNN/severity candidate.\n"
    "1) When GNN and severity point to the same service, treat it as the "
    "strongest single-service root-cause candidate. "
    "When they conflict, adjudicate using whether the service sits upstream in "
    "the failure propagation, the severity components (especially res_dev), and "
    "the execution-graph failure origin, "
    "and explain in `reason` why you chose one of them.\n"
    "2) A `signature` is a template dynamically extracted from the current "
    "system logs, not a predefined error category. "
    "A high-frequency template consistent with GNN/severity/execution-graph can "
    "raise confidence; single-service logs conflicting with first-tier evidence "
    "must not by themselves overturn the coarse-ranking result. "
    "Only when the same template appears across multiple independent services "
    "and can explain the common failure of the candidate services may it be "
    "judged `systemic`.\n"
    "3) If `shared_signatures` is empty: do **not** judge systemic based on "
    "empty results / cascading failures alone. "
    "Fall back to the high-ranked GNN/severity candidates and treat empty "
    "results as possible downstream symptoms; a resource-type fault may have no "
    "error logs at all.\n"
    "4) root_cause must give a **specific service name** (for single_service) or "
    "a specific shared dependency (for systemic), "
    "and be consistent with reranked_topk[0].\n"
    "Output exactly one JSON object, no extra text. Fields:"
    '{"root_cause":"","root_cause_type":"systemic|single_service",'
    '"failure_stage":"planning|reasoning|tool-invocation|response-summary",'
    '"reason":"","evidence":[],"confidence":0.0,"reranked_topk":[],"suggested_fix":""}'
)

_EVIDENCE_FIELD_GUIDE = (
    "\n\nMeaning of the input JSON fields (interpret strictly by these "
    "definitions, do not guess):\n"
    "A. `global.num_traces` is the number of traces used for detailed vertex "
    "slicing (may be truncated by the context limit); "
    "`num_trace_analysis_graphs` is the full number of candidate-related traces "
    "actually traversed for call-chain frequency statistics; "
    "`num_failed_traces` is the number of detailed-slice traces in which a "
    "candidate-service vertex failed; "
    "`num_problem_traces` is the number of complete traces in which at least one "
    "top-k candidate-service vertex failed; "
    "`failed_trace_examples` is only a small set of text samples, "
    "the complete problem traces are given by `problem_trace_analysis.traces`.\n"
    "B. `coarse_ranking` is the metric-severity ranking. All metrics first build "
    "a baseline from pre-anomaly-window data, "
    "then detect deviations within the anomaly window. `severity` = `fail_rate` "
    "+ 0.1*log1p(max(`latency_spike`-1,0)) + resource_weight*`res_dev` + "
    "qps_weight*`qps_dev`, larger is more suspicious; "
    "it is an in-sample relative score, not a probability. `fail_rate` is the "
    "average drop relative to the out-of-window success-rate baseline; "
    "`n_fail_points` is the number of sampling points continuously below that "
    "baseline threshold; `latency_spike` is the anomaly-window peak / "
    "out-of-window baseline median; "
    "`qps_dev` is the maximum bidirectional QPS deviation from the out-of-window "
    "baseline; `res_dev` is, after passing sustained-spike detection, "
    "the maximum deviation of the CPU or memory peak from the out-of-window "
    "baseline median; "
    "`cpu_dev`/`mem_dev` denote the deviation of each resource channel. "
    "`anomaly_types` is the full set of observed symptom types; "
    "`anomaly_type_scores` is the actual weighted contribution of each symptom "
    "to severity, where availability=fail_rate, "
    "latency=0.1*log1p(latency_spike-1), cpu/memory jointly share "
    "resource_weight*res_dev, qps=qps_weight*qps_dev.\n"
    "C. `gnn_service_ranking` is the service ranking the GNN produces after "
    "fusing metric anomalies, service topology and fault propagation; "
    "a larger `gnn_score` is more suspicious, comparable only within this "
    "sample, not a probability. The `coarse_evidence` in each candidate also "
    "gives that service's `gnn_rank/gnn_score` and "
    "`severity_rank/severity_score`; null means the evidence source lacks that "
    "service.\n"
    "D. `matched_failed_vertexes` are the direct failure slices of candidate "
    "services in status-failed traces. `failed=true` means the service itself "
    "shows a dynamic error template in its messages, level_spans input/output, "
    "or the params passed downstream in level_routes, or it is the summarizer "
    "corresponding to task_status=8; `empty_results` is descriptive only, and "
    "neither visible nor empty output can serve as a failure criterion; "
    "`signatures/error_texts` are error evidence within the execution graph. "
    "`status` is the raw execution status code and must not be interpreted as "
    "success or failure independently of empty_results/signatures.\n"
    "E. In `problem_trace_analysis`, a problem trace takes the union of two "
    "criteria: task_status=3 is task failure, 7 is status success, "
    "8 is summarization failure (the status error is attributed to the "
    "summarizer); status=3 means the chain terminated midway, with the error "
    "attributed to the actual chain tail and no summarizer appended; "
    "or an error signature is detected in a service vertex. "
    "Once a trace is selected, the complete upstream/downstream chain and all "
    "failed vertexes on it are retained. "
    "`vertex_chain` is arranged left to right by execution level, keeping only "
    "the Planner head node and each xxxGroup, excluding xxxGroup/xxAgent; "
    "`service_chain` is likewise arranged upstream to downstream. Only when "
    "task_status is 7/8 is a summarizer not recorded in the raw level_routes "
    "appended to the chain tail. "
    "`call_chain_frequencies.trace_count/trace_ratio` is the count / "
    "proportion-of-problem-traces of identical service chains. "
    "`service_frequencies.traces_containing_service/trace_coverage` is the count "
    "/ proportion of all loaded traces passing through that service; "
    "`problem_trace_count/problem_trace_ratio` is the count / proportion of "
    "problem traces passing through that service; "
    "`chain_occurrences` retains the count of non-consecutive repeated "
    "appearances of a service in the chain; `status_error_count/"
    "status_total_count` is the status channel, "
    "`vertex_error_count/vertex_total_count` is the error-signature channel; "
    "`combined_error_count` and `combined_total_count` are respectively "
    "the deduplicated union by trace_id of the two error sets and total sets, "
    "and `combined_error_ratio` is their quotient. `failed_trace_count` is the "
    "count after taking the union of the two criteria by trace. "
    "High coverage only indicates a frequently traversed service (e.g. planner) "
    "and cannot by itself prove it is the root cause; "
    "focus on comparing the earliest failure position on the chain, failure "
    "frequency, and GNN/severity.\n"
    "F. Log fields are auxiliary evidence only. `log_signature_counts` is the "
    "count of dynamic templates within the anomaly window; "
    "`log_signature_frequency_per_minute` is the per-minute frequency; "
    "`frequent_signatures` are templates reaching the minimum frequency; "
    "`shared_signatures` contains only templates appearing in at least two "
    "candidate services together; `log_errors` are raw log samples matching the "
    "templates.\n"
    "G. In the output, `failure_stage`: planning=planning/task decomposition, "
    "reasoning=model reasoning, "
    "tool-invocation=service or tool call, response-summary=final "
    "aggregation/response generation.\n"
)


_STAGE1_PROMPT = (
    "You handle the first stage of root-cause localization: GNN-anchored "
    "candidate screening. "
    "GNN has already fused topology, metric anomalies and propagation relations "
    "and is the primary basis of this stage; severity checks direct metric "
    "severity. "
    "Compare candidates one by one, but output only concise, auditable evidence "
    "conclusions, not the implicit thought process. "
    "You must retain the service with GNN rank=1. Besides it, select the "
    "candidates most supported by severity or multiple metrics; "
    "the shortlist contains at most the max_shortlist services specified in the "
    "input. Output JSON only:"
    '{"candidate_assessments":[{"service":"","gnn_assessment":"",'
    '"severity_assessment":"","provisional_role":"root|propagated|uncertain"}],'
    '"shortlist":[],"summary":""}'
)


_STAGE2_PROMPT = (
    "You handle the second stage of root-cause localization: verify only the "
    "candidates in the shortlist and narrow the scope further. "
    "Evidence priority is GNN > metric severity and anomaly dimensions > "
    "execution-graph direct failure frequency/upstream position > logs. "
    "Logs can only corroborate, not independently overturn the GNN top-1; a "
    "service passing through many traces also cannot by itself prove it is the "
    "root cause. "
    "To eliminate the GNN top-1, both must hold: (a) another candidate has more "
    "direct metric or failure-frequency evidence; "
    "(b) the execution graph shows another candidate is an earlier direct "
    "failure point; and override_gnn_top1 must be filled in explicitly. "
    "Give conclusions on metric severity, problem-trace frequency, propagation "
    "position and log consistency one by one, "
    "but do not output the implicit thought process. survivors is at most "
    "max_survivors. Output JSON only:"
    '{"candidate_checks":[{"service":"","metric_evidence":"",'
    '"frequency_evidence":"","graph_position":"","log_evidence":"",'
    '"decision":"keep|eliminate"}],"survivors":[],"ranking":[],'
    '"override_gnn_top1":{"applied":false,"replacement":null,"justification":""},'
    '"summary":""}'
)


_FINAL_STAGE_PROMPT = (
    "You handle the final stage of root-cause localization. You may only choose "
    "a single-service root cause from the second-stage survivors; "
    "systemic is allowed only when the same high-frequency template spans "
    "multiple independent services and truly explains the common failure. "
    "By default choose the highest-ranked GNN service still surviving. Only when "
    "the second stage has explicitly approved "
    "override_gnn_top1 under the dual conditions may another service surpass it. "
    "Combining severity components, direct failure frequency, "
    "upstream failure position and log corroboration, output concise auditable "
    "conclusions, not the implicit thought process. "
    "When root_cause is single_service it must be a service in survivors and "
    "consistent with reranked_topk[0]. "
    "Output JSON only:"
    '{"root_cause":"","root_cause_type":"systemic|single_service",'
    '"failure_stage":"planning|reasoning|tool-invocation|response-summary",'
    '"reason":"","evidence":[],"confidence":0.0,"reranked_topk":[],'
    '"suggested_fix":""}'
)


_RANK_SYSTEM_PROMPT = (
    "You are a root-cause localization expert for Agent systems. Given the "
    "candidate root-cause services within a fault time window and their "
    "evidence, "
    "reorder the candidate services from most to least likely to be the root "
    "cause.\n"
    "Evidence priority:\n"
    "1) metric severity coarse_ranking: larger severity is more suspicious; "
    "focus on the res_dev (resource), "
    "latency, and availability (success rate) components;\n"
    "2) execution-graph direct failure position and frequency: the more "
    "upstream and the earlier the failure, the more suspicious; empty results "
    "are usually downstream symptoms and do not alone determine the root "
    "cause;\n"
    "3) log signature: high-frequency and consistent across multiple services "
    "can add points, but cannot alone overturn metric evidence.\n"
    "Output exactly one JSON object, no extra text:\n"
    '{"reranked_topk":["service names, most to least suspicious"],'
    '"root_cause":"reranked_topk[0]",'
    '"failure_stage":"planning|reasoning|tool-invocation|response-summary",'
    '"reason":"","confidence":0.0}'
)

_RANK_FIELD_GUIDE = (
    "\n\nField notes: `coarse_ranking` is the metric-severity ranking, "
    "`severity`=fail_rate+0.1*log1p(latency_spike-1)+w*res_dev+w*qps_dev, larger "
    "is more suspicious; "
    "`candidates[].coarse_evidence` gives that service's "
    "severity_rank/severity_score; "
    "`candidates[].matched_failed_vertexes` are the candidate service's failure "
    "slices in the execution graph; "
    "`candidates[].log_signatures` are dynamic log error templates; "
    "`problem_trace_analysis.service_frequencies` are each service's "
    "occurrence/failure frequency in problem traces; "
    "`shared_signatures` are log templates co-occurring across multiple "
    "services."
)


def single_pass_rank_localize(agg: Dict, cfg) -> Dict:
    """One LLM call: rank the coarse candidates by root-cause likelihood.

    Simpler replacement for the 3-stage staged pipeline: feed the aggregated
    evidence once and let the LLM produce a single ranked list. The top-1 is the
    localized root cause; the ranking is sanitized against the candidate set and
    backfilled from the coarse ranking so no candidate is ever dropped."""
    allowed = [row.get('service') for row in (agg.get('candidates') or [])
               if row.get('service')]
    if not allowed:
        return rule_based_localize(agg)
    messages = [
        {'role': 'system', 'content': _RANK_SYSTEM_PROMPT + _RANK_FIELD_GUIDE},
        {'role': 'user', 'content': (
            "The aggregated fault evidence is as follows (JSON):\n"
            + json.dumps(agg, ensure_ascii=False, indent=2)
            + "\nReorder the candidate services and output the specified JSON.")},
    ]
    result = _extract_json(_call_openai_compatible(messages, cfg))
    order = _valid_service_order(result.get('reranked_topk'), allowed)
    # backfill from the coarse ranking, then any remaining candidate
    coarse = _valid_service_order(
        [row.get('service') for row in (agg.get('coarse_ranking') or [])],
        allowed)
    for svc in coarse + allowed:
        if svc not in order:
            order.append(svc)
    root = result.get('root_cause')
    if root not in allowed:
        root = order[0] if order else None
    else:
        order = [root] + [svc for svc in order if svc != root]
    result['root_cause'] = root
    result['reranked_topk'] = order
    result.setdefault('root_cause_type', 'single_service')
    result['_mode'] = 'llm_single_pass_rank'
    return result


def build_messages(agg: Dict) -> list:
    user = (
        "The aggregated fault evidence is as follows (JSON):\n"
        + json.dumps(agg, ensure_ascii=False, indent=2)
        + "\nLocate the fine-grained root cause accordingly and output the specified JSON."
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT + _EVIDENCE_FIELD_GUIDE},
        {"role": "user", "content": user},
    ]


# ----------------------------- rule-based fallback -----------------------------
def rule_based_localize(agg: Dict) -> Dict:
    shared_list = agg.get('shared_signatures', [])
    candidates = agg.get('candidates', [])
    candidate_services = [c['service'] for c in candidates]
    # The fallback obeys the same GNN-first policy as the staged LLM path.
    reranked = _gnn_order(agg, candidate_services) if candidate_services else []
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
            root = f'Shared log fault: {signature[:180]}'
            promoted = services + [s for s in reranked if s not in services]
            return {
                'root_cause': root,
                'root_cause_type': 'systemic',
                'failure_stage': 'tool-invocation',
                'reason': (f"The dynamic log template appears in {n} services, "
                           f"{occurrences} times total within the anomaly window, "
                           f"and covers the coarse-ranking top-1."),
                'evidence': [f'signature={signature}',
                             f'num_services={n}',
                             f'log_occurrences={occurrences}',
                             f'per_service_counts={primary.get("per_service_log_counts", {})}',
                             f'corroborates_coarse_top1={corroborates_coarse}'],
                'confidence': 0.85,
                'reranked_topk': promoted,
                'suggested_fix': 'Handle this high-frequency fault template based on the raw log samples and execution-graph context.',
                '_mode': 'rule_based_dynamic_signature',
            }
    # No cross-service dynamic template -> defer to coarse top-1.
    top = reranked[0] if reranked else None
    return {
        'root_cause': top,
        'root_cause_type': 'single_service',
        'failure_stage': 'tool-invocation',
        'reason': 'No multi-source evidence sufficient to override the GNN top-1 was found; fall back to the GNN top-1.',
        'evidence': [],
        'confidence': 0.3,
        'reranked_topk': reranked,
        'suggested_fix': 'Confirm further with the logs.',
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


def _valid_service_order(values, allowed: List[str]) -> List[str]:
    """Return unique, known services while preserving model order."""
    out = []
    for value in values or []:
        if isinstance(value, dict):
            value = value.get('service')
        if value in allowed and value not in out:
            out.append(value)
    return out


def _gnn_order(agg: Dict, allowed: List[str]) -> List[str]:
    rows = agg.get('gnn_service_ranking') or []
    order = _valid_service_order(
        [row.get('service') for row in rows], allowed)
    return order + [svc for svc in allowed if svc not in order]


def _rows_for_services(rows, services):
    wanted = set(services)
    return [row for row in (rows or []) if row.get('service') in wanted]


def _signatures_for_services(rows, services):
    wanted = set(services)
    return [row for row in (rows or [])
            if wanted.intersection(row.get('services') or [])]


def _representative_problem_traces(traces, limit=12):
    """Deduplicate traces by chain/failure shape for a compact LLM payload."""
    out = []
    seen = set()
    for trace in traces or []:
        failed = trace.get('failed_vertexes') or []
        failure_shape = tuple(
            (row.get('service'), row.get('level'),
             tuple(row.get('signatures') or []))
            for row in failed
        )
        key = (trace.get('task_status'),
               tuple(trace.get('service_chain') or []), failure_shape)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            'task_status': trace.get('task_status'),
            'vertex_chain': trace.get('vertex_chain') or [],
            'service_chain': trace.get('service_chain') or [],
            'failed_vertexes': failed,
        })
        if len(out) >= limit:
            break
    return out


def _compact_chain_frequencies(rows):
    return [
        {key: row.get(key) for key in (
            'service_chain', 'trace_count', 'trace_ratio')}
        for row in (rows or [])
    ]


def _stage1_evidence(agg: Dict, cfg, allowed: List[str]) -> Dict:
    order = _gnn_order(agg, allowed)
    candidates = {
        row.get('service'): row for row in (agg.get('candidates') or [])
    }
    return {
        'instruction': 'Anchor on GNN first, then use severity to check direct metric support.',
        'max_shortlist': min(
            max(1, int(getattr(cfg, 'llm_stage1_topk', 3))), len(allowed)),
        'gnn_top1': order[0] if order else None,
        'gnn_service_ranking': _rows_for_services(
            agg.get('gnn_service_ranking'), allowed),
        'coarse_ranking': _rows_for_services(
            agg.get('coarse_ranking'), allowed),
        'candidate_coarse_evidence': [
            {'service': svc,
             'coarse_evidence': candidates.get(svc, {}).get(
                 'coarse_evidence', {})}
            for svc in order
        ],
    }


def _stage2_evidence(agg: Dict, cfg, shortlist: List[str],
                     gnn_top1: str) -> Dict:
    problem = agg.get('problem_trace_analysis') or {}
    return {
        'instruction': ('Verify metrics, direct failure frequency, propagation '
                        'position and logs item by item; logs cannot override '
                        'GNN alone.'),
        'shortlist': shortlist,
        'gnn_top1': gnn_top1,
        'max_survivors': min(
            max(1, int(getattr(cfg, 'llm_stage2_topk', 2))),
            len(shortlist)),
        'global': agg.get('global') or {},
        'gnn_service_ranking': _rows_for_services(
            agg.get('gnn_service_ranking'), shortlist),
        'coarse_ranking': _rows_for_services(
            agg.get('coarse_ranking'), shortlist),
        'candidates': _rows_for_services(
            agg.get('candidates'), shortlist),
        'problem_trace_analysis': {
            'problem_trace_definition': problem.get(
                'problem_trace_definition'),
            'num_problem_traces': problem.get('num_problem_traces'),
            'representative_trace_note': (
                'Representative samples deduplicated by task_status, service '
                'chain and failed-node shape; overall frequency is governed by '
                'call_chain_frequencies and service_frequencies.'),
            'representative_traces': _representative_problem_traces(
                problem.get('traces')),
            'call_chain_frequencies': _compact_chain_frequencies(
                problem.get('call_chain_frequencies')),
            'service_frequencies': _rows_for_services(
                problem.get('service_frequencies'), shortlist),
        },
        'frequent_signatures': _signatures_for_services(
            agg.get('frequent_signatures'), shortlist),
        'shared_signatures': _signatures_for_services(
            agg.get('shared_signatures'), shortlist),
    }


def _final_evidence(agg: Dict, survivors: List[str], stage1: Dict,
                    stage2: Dict) -> Dict:
    problem = agg.get('problem_trace_analysis') or {}
    return {
        'survivors': survivors,
        'stage1_screening': stage1,
        'stage2_verification': stage2,
        'gnn_service_ranking': _rows_for_services(
            agg.get('gnn_service_ranking'), survivors),
        'coarse_ranking': _rows_for_services(
            agg.get('coarse_ranking'), survivors),
        'candidate_evidence': _rows_for_services(
            agg.get('candidates'), survivors),
        'service_frequencies': _rows_for_services(
            problem.get('service_frequencies'), survivors),
        'shared_signatures': _signatures_for_services(
            agg.get('shared_signatures'), survivors),
    }


def _call_stage(prompt: str, evidence: Dict, cfg) -> Dict:
    messages = [
        {'role': 'system', 'content': prompt + _EVIDENCE_FIELD_GUIDE},
        {'role': 'user', 'content': json.dumps(
            evidence, ensure_ascii=False, indent=2)},
    ]
    return _extract_json(_call_openai_compatible(messages, cfg))


def staged_llm_localize(agg: Dict, cfg) -> Dict:
    """Run three auditable narrowing passes with GNN as the primary anchor."""
    allowed = [row.get('service') for row in (agg.get('candidates') or [])
               if row.get('service')]
    if not allowed:
        return rule_based_localize(agg)
    gnn_order = _gnn_order(agg, allowed)
    gnn_top1 = gnn_order[0]

    print('[LLM 1/3] GNN-anchored candidate screening')
    stage1_input = _stage1_evidence(agg, cfg, allowed)
    stage1 = _call_stage(_STAGE1_PROMPT, stage1_input, cfg)
    stage1_limit = stage1_input['max_shortlist']
    shortlist = _valid_service_order(stage1.get('shortlist'), allowed)
    if gnn_top1 not in shortlist:
        shortlist.insert(0, gnn_top1)
    for svc in gnn_order:
        if len(shortlist) >= stage1_limit:
            break
        if svc not in shortlist:
            shortlist.append(svc)
    shortlist = shortlist[:stage1_limit]
    stage1['shortlist'] = shortlist
    stage1['gnn_top1_forced_retained'] = True

    print(f'[LLM 2/3] multi-evidence verification: {shortlist}')
    stage2_input = _stage2_evidence(agg, cfg, shortlist, gnn_top1)
    stage2 = _call_stage(_STAGE2_PROMPT, stage2_input, cfg)
    survivor_limit = stage2_input['max_survivors']
    survivors = _valid_service_order(stage2.get('survivors'), shortlist)
    override = stage2.get('override_gnn_top1') or {}
    override_applied = override.get('applied') is True
    replacement = override.get('replacement')
    valid_override = (override_applied and replacement in shortlist
                      and replacement != gnn_top1
                      and bool(str(override.get('justification') or '').strip()))
    if not valid_override:
        override['applied'] = False
        override['replacement'] = None
        if gnn_top1 not in survivors:
            survivors.insert(0, gnn_top1)
    ranking = _valid_service_order(stage2.get('ranking'), shortlist)
    for svc in ranking + shortlist:
        if len(survivors) >= survivor_limit:
            break
        if svc not in survivors:
            survivors.append(svc)
    survivors = survivors[:survivor_limit]
    stage2['survivors'] = survivors
    stage2['ranking'] = (
        _valid_service_order(stage2.get('ranking'), survivors)
        + [svc for svc in survivors
           if svc not in _valid_service_order(stage2.get('ranking'), survivors)]
    )
    stage2['override_gnn_top1'] = override

    print(f'[LLM 3/3] final decision over survivors: {survivors}')
    final_input = _final_evidence(agg, survivors, stage1, stage2)
    result = _call_stage(_FINAL_STAGE_PROMPT, final_input, cfg)
    result_type = result.get('root_cause_type', 'single_service')
    final_order = _valid_service_order(result.get('reranked_topk'), survivors)
    for svc in stage2['ranking'] + survivors:
        if svc not in final_order:
            final_order.append(svc)
    root = result.get('root_cause')
    if result_type != 'systemic':
        if not valid_override:
            root = gnn_top1
        elif root not in survivors:
            root = final_order[0]
        final_order = [root] + [svc for svc in final_order if svc != root]
        result['root_cause'] = root
        result['root_cause_type'] = 'single_service'
    for svc in gnn_order:
        if svc not in final_order:
            final_order.append(svc)
    result['reranked_topk'] = final_order
    result['_mode'] = 'llm_staged'
    result['_analysis_stages'] = {
        'stage1_gnn_screening': stage1,
        'stage2_evidence_verification': stage2,
    }
    return result


def llm_localize(agg: Dict, cfg) -> Dict:
    """Localize fine-grained root cause. Falls back to rule-based on
    mock mode / missing credentials / any API or parse error."""
    if getattr(cfg, 'llm_mock', True):
        return rule_based_localize(agg)
    try:
        if getattr(cfg, 'llm_staged', False):
            return staged_llm_localize(agg, cfg)
        return single_pass_rank_localize(agg, cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f'[llm_localize] API/parse failed ({e}); falling back to rule-based.')
        res = rule_based_localize(agg)
        res['_mode'] = 'rule_based_after_error'
        res['_error'] = str(e)
        return res
