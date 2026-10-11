from Config import Config
from util.KubernetesClient import KubernetesClient


global_namespace_svcs_dict = {}
current_namespace = None


def handle_traces_no_istio(trace_jsons, config: Config):
    """Collect traces from microservices outside the service mesh."""
    global global_namespace_svcs_dict
    global_namespace_svcs_dict = KubernetesClient(config).get_all_svc()
    global current_namespace
    current_namespace = config.namespace

    normal_dicts = {}
    inbound_dicts = {}
    outbound_dicts = {}
    abnormal_dicts = {}
    inbound_half_dicts = {}
    outbound_half_dicts = {}
    abnormal_half_dicts = {}

    for trace_json in trace_jsons:
        trace_dict = get_empty_trace_dict(trace_json['traceID'])
        half_trace_dict = get_empty_trace_dict(trace_json['traceID'])
        process_dicts = get_trace_process(trace_json)
        spans_dict = {}
        for span_json in trace_json['spans']:
            spans_dict[span_json['spanID']] = span_json

        is_inbound = False
        is_outbound = False

        for span_json in trace_json['spans']:
            is_span_useful, span_json = get_span_useful(span_json)
            outbound_span = get_outbound_span(spans_dict, span_json, trace_dict)
            if is_span_useful is False or outbound_span is None:
                continue

            inbound_pod_ip, inbound_pod_name, callee_svc, inbound_namespace = get_span_information(span_json, process_dicts)
            outbound_pod_ip, outbound_pod_name, caller_svc, outbound_namespace = get_span_information(outbound_span, process_dicts)
            trace_dict = add_trace_information(span_json, outbound_span, trace_dict, spans_dict)

            trace_dict['svc'].append(callee_svc)
            trace_dict['call'].append((caller_svc, callee_svc))
            trace_dict['call_instance'].append((handle_node_id_no_istio(outbound_pod_name, outbound_pod_ip, outbound_namespace), handle_node_id_no_istio(inbound_pod_name, inbound_pod_ip, inbound_namespace)))

            # Keep only the local side of a cross-namespace call in the half trace.
            if current_namespace != inbound_namespace and current_namespace == outbound_namespace:
                is_outbound = True
                half_trace_dict = add_outbound_trace_information(outbound_span, half_trace_dict, span_json, spans_dict)
                half_trace_dict['svc'].append('OTHER_SVC')
                half_trace_dict['call'].append((caller_svc, 'OTHER_SYSTEM'))
                half_trace_dict['call_instance'].append((handle_node_id_no_istio(outbound_pod_name, outbound_pod_ip, outbound_namespace),handle_node_id_no_istio('OTHER_NODE', '', '')))
            elif current_namespace == inbound_namespace and current_namespace != outbound_namespace:
                is_inbound = True
                half_trace_dict = add_inbound_trace_information(span_json, half_trace_dict)
                half_trace_dict['svc'].append(callee_svc)
                half_trace_dict['call'].append(('OTHER_SYSTEM', callee_svc))
                half_trace_dict['call_instance'].append((handle_node_id_no_istio('OTHER_NODE', '', ''), handle_node_id_no_istio(inbound_pod_name, inbound_pod_ip, inbound_namespace)))
            elif current_namespace == inbound_namespace and current_namespace == outbound_namespace:
                half_trace_dict = add_trace_information(span_json, outbound_span, half_trace_dict, spans_dict)
                half_trace_dict['svc'].append(callee_svc)
                half_trace_dict['call'].append((caller_svc, callee_svc))
                half_trace_dict['call_instance'].append((handle_node_id_no_istio(outbound_pod_name, outbound_pod_ip, outbound_namespace), handle_node_id_no_istio(inbound_pod_name, inbound_pod_ip, inbound_namespace)))

        if trace_dict['call'] == []:
            continue

        if is_abnormal(trace_dict):
            abnormal_dicts[trace_json['traceID']] = trace_dict
        else:
            normal_dicts[trace_json['traceID']] = trace_dict
        if is_abnormal(half_trace_dict) and (is_inbound is True or is_outbound is True):
            abnormal_half_dicts[trace_json['traceID']] = half_trace_dict
        if is_inbound is True:
            inbound_dicts[trace_json['traceID']] = trace_dict
            inbound_half_dicts[trace_json['traceID']] = half_trace_dict
        if is_outbound is True:
            outbound_dicts[trace_json['traceID']] = trace_dict
            outbound_half_dicts[trace_json['traceID']] = half_trace_dict

    return normal_dicts, inbound_dicts, outbound_dicts, abnormal_dicts, inbound_half_dicts, outbound_half_dicts, abnormal_half_dicts

def get_empty_trace_dict(traceId):
    trace_dict = {'traceId': traceId}
    trace_dict['call'] = []
    trace_dict['timestamp'] = []
    trace_dict['latency'] = []
    trace_dict['http_status'] = []
    trace_dict['svc'] = []
    trace_dict['call_instance'] = []

    return trace_dict


def get_trace_process(trace_json):
    """Extract process ID, service name, pod name, and pod IP from a trace."""
    process_json = trace_json['processes']
    process_dicts = {}
    for key in process_json.keys():
        process_dict = {'processId': key}
        process = process_json[key]
        process_dict['service_name'] = process['serviceName']
        for tag in process['tags']:
            if tag['key'] == 'host.name':
                process_dict['pod_name'] = tag['value']
            if tag['key'] == 'ip':
                process_dict['pod_ip'] = tag['value']
        process_dicts[process_dict['processId']] = process_dict

    return process_dicts


def get_span_useful(span_json):
    is_span_useful = False

    for tag in span_json['tags']:
        if tag['key'] == 'net.host.name':
            is_span_useful = True
            break
        if tag['key'] == 'thread.name' and 'DubboServerHandler' in tag['value']:
            for tg in span_json['tags']:
                if tg['key'] == 'rpc.method':
                    is_span_useful = True
                    break
            break

    return is_span_useful, span_json


def get_outbound_span(spans_dict, span_json, trace_dict):
    outbound_span = {}
    if span_json['references'] == []:
        return None
    else:
        for ref in span_json['references']:
            if ref['refType'] == 'CHILD_OF':
                if ref['spanID'] not in spans_dict.keys():
                    return None
                else:
                    tmp_span = spans_dict[ref['spanID']]
                    tmp_span_useful, test_span = get_span_useful(tmp_span)
                    # Follow unusable parents until a useful ancestor is found.
                    if tmp_span_useful is True:
                        outbound_span = tmp_span
                        break
                    else:
                        outbound_span = get_outbound_span(spans_dict, tmp_span, trace_dict)
    return outbound_span


def add_trace_information(span_json, outbound_span, trace_dict, spans_dict):
    is_horsecoder_outbound_span = True
    outbound_timestamp, outbound_latency = get_outbound_time(span_json, spans_dict)
    trace_dict['timestamp'].append(outbound_timestamp + clock_skew_adjust(span_json))
    trace_dict['latency'].append(outbound_latency)
    for tag in outbound_span['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_outbound_span = False
            trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_outbound_span == True:
        trace_dict['http_status'].append(200)
    is_horsecoder_span = True
    trace_dict['timestamp'].append(span_json['startTime'] + clock_skew_adjust(span_json))
    trace_dict['latency'].append(span_json['duration'])
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_span = False
            trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_span == True:
        trace_dict['http_status'].append(200)

    return trace_dict


def add_outbound_trace_information(outbound_span, half_trace_dict, span_json, spans_dict):
    is_horsecoder_span = True
    outbound_timestamp, outbound_latency = get_outbound_time(span_json, spans_dict)
    half_trace_dict['timestamp'].append(outbound_timestamp + clock_skew_adjust(outbound_span))
    half_trace_dict['latency'].append(outbound_latency)
    for tag in outbound_span['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_span = False
            half_trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_span == True:
        half_trace_dict['http_status'].append(200)
    # Unknown remote endpoint values use -1.
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)

    return half_trace_dict


def add_inbound_trace_information(span_json, half_trace_dict):
    is_horsecoder_span = True
    # Unknown remote endpoint values use -1.
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)
    half_trace_dict['timestamp'].append(span_json['startTime'] + + clock_skew_adjust(span_json))
    half_trace_dict['latency'].append(span_json['duration'])
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_span = False
            half_trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_span == True:
        half_trace_dict['http_status'].append(200)

    return half_trace_dict


def get_span_information(span_json, process_dicts):
    # Map horsecoder process names to Kubernetes service names.
    horsecoder_test_svcs = {'edge-gateway': 'edge-gateway-svc', 'edge-llm-1.0.0': 'edge-llm-svc', 'edge-paraformer-serverless-1.0.0': 'edge-paraformer-serverless-svc'}

    processId = span_json['processID']
    process_dict = process_dicts[processId]

    pod_ip = process_dict['pod_ip']
    pod_name = process_dict['pod_name']
    svc_name = process_dict['service_name']
    if (svc_name in horsecoder_test_svcs.keys()):
        svc_name = horsecoder_test_svcs[svc_name]
    namespace = get_svc_namespace(svc_name)

    return pod_ip, pod_name, svc_name, namespace


def get_svc_namespace(svc):
    """Find a service namespace; hipster and hipster2 names may collide."""
    global global_namespace_svcs_dict
    for key in global_namespace_svcs_dict.keys():
        svc_list = global_namespace_svcs_dict[key]
        if svc in svc_list:
            return key
    print(svc + "not found.")
    return None


def handle_node_id_no_istio(node_name, node_id, namespace):
    if node_name == 'OTHER_NODE':
        return 'OTHER_NODE'
    return str(node_name) + '.' + str(namespace) + '-' + node_id


def is_abnormal(trace_dict):
    if trace_dict['call'] == []:
        return True
    for code in trace_dict['http_status']:
        if (code < 200 and code != -1) or code >= 300:
            return True
    return False


def get_outbound_time(span_json, spans_dict):
    outbound_timestamp = None
    outbound_latency = None
    if span_json['references'] == []:
        return None
    else:
        for ref in span_json['references']:
            if ref['refType'] == 'CHILD_OF':
                if ref['spanID'] not in spans_dict.keys():
                    return None
                else:
                    tmp_span = spans_dict[ref['spanID']]
                    outbound_timestamp = tmp_span['startTime'] + clock_skew_adjust(tmp_span)
                    outbound_latency = tmp_span['duration']

    return outbound_timestamp, outbound_latency


def clock_skew_adjust(span_json):
    """Apply Jaeger clock skew adjustments from span warnings."""
    if span_json['warnings'] != None:
        for warning in span_json['warnings']:
            if 'clock skew adjustment' in warning:
                try:
                    adjust_flag = warning.split(' ')[-1][-2:]
                    adjust_time = float(warning.split(' ')[-1].strip('µms'))
                    if adjust_flag == 'ms':
                        return int(adjust_time * 1000)
                    elif adjust_flag == 'µs':
                        return int(adjust_time)
                    else:
                        return 0
                except Exception:
                    return 0
    else:
        return 0






