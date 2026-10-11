from Config import Config
from util.KubernetesClient import KubernetesClient

global_namespace_svcs_dict = {}
current_namespace = None

'''
    Collect traces for a microservice system inside a service mesh
'''
def handle_traces_istio(trace_jsons, config: Config):
    # Get service names for each namespace
    global global_namespace_svcs_dict
    global_namespace_svcs_dict = KubernetesClient(config).get_all_svc()
    global current_namespace
    current_namespace = config.namespace

    # Define the return data
    normal_dicts = {}
    inbound_dicts = {}
    outbound_dicts = {}
    abnormal_dicts = {}
    inbound_half_dicts = {}
    outbound_half_dicts = {}
    abnormal_half_dicts = {}
    # Iterate each trace
    for trace_json in trace_jsons:
        # Define trace_dict
        trace_dict = get_empty_trace_dict(trace_json['traceID'])
        half_trace_dict = get_empty_trace_dict(trace_json['traceID'])

        # Build a flag array holding spans not yet visited
        span_flag = []
        # Get all spans in the trace
        spans_dict = {}
        for span_json in trace_json['spans']:
            spans_dict[span_json['spanID']] = span_json
            span_flag.append(span_json['spanID'])

        # Define flags
        is_inbound = False
        is_outbound = False
        is_abnormal = False

        for span_json in trace_json['spans']:
            # if inbound, get the outbound span info
            outbound_span = None
            if is_inbound_span(span_json):
                outbound_span = get_outbound_span(spans_dict, span_json, trace_dict)
            # with an outbound edge, get its info and remove this span's flag
            if outbound_span != None:
                if span_json['spanID'] in span_flag:
                    span_flag.remove(span_json['spanID'])
                if outbound_span['spanID'] in span_flag:
                    span_flag.remove(outbound_span['spanID'])
            else:
                continue
            # add info
            node_id, svc, namespace = get_span_information(span_json)
            outbound_node_id, caller_svc, outbound_namespace = get_span_information(outbound_span)
            trace_dict = add_trace_information(outbound_span, trace_dict)
            trace_dict = add_trace_information(span_json, trace_dict)

            # add the call relation to the trace
            trace_dict['svc'].append(svc)
            trace_dict['call'].append((caller_svc, svc))
            trace_dict['call_instance'].append((handle_node_id_istio(outbound_node_id), handle_node_id_istio(node_id)))

            # determine whether it crosses systems and whether to add it to half_trace_dict
            # inbound crosses systems, outbound is in-system
            if current_namespace != namespace and current_namespace == outbound_namespace:
                is_outbound = True
                half_trace_dict = add_outbound_trace_information(outbound_span, half_trace_dict)
                half_trace_dict['svc'].append('OTHER_SVC')
                half_trace_dict['call'].append((caller_svc, 'OTHER_SYSTEM'))
                half_trace_dict['call_instance'].append((handle_node_id_istio(outbound_node_id), handle_node_id_istio('OTHER_NODE')))
            # outbound crosses systems, inbound is in-system
            elif current_namespace == namespace and current_namespace != outbound_namespace:
                is_inbound = True
                half_trace_dict = add_inbound_trace_information(span_json, half_trace_dict)
                half_trace_dict['svc'].append(svc)
                half_trace_dict['call'].append(('OTHER_SYSTEM', svc))
                half_trace_dict['call_instance'].append((handle_node_id_istio('OTHER_NODE'), handle_node_id_istio(node_id)))
            # both in-system, store it fully into half_trace_dict
            elif current_namespace == namespace and current_namespace == outbound_namespace:
                half_trace_dict = add_trace_information(outbound_span, half_trace_dict)
                half_trace_dict = add_trace_information(span_json, half_trace_dict)
                half_trace_dict['svc'].append(svc)
                half_trace_dict['call'].append((caller_svc, svc))
                half_trace_dict['call_instance'].append((handle_node_id_istio(outbound_node_id), handle_node_id_istio(node_id)))
            # if both cross systems, do not add it to half_trace data

        # Iterate again to find outbound edges without a matching call relation
        for span_id in span_flag:
            span = spans_dict[span_id]
            node_id, svc, namespace = get_span_information(span)
            # outbound edge, fill in the info
            if is_outbound_span(span) is True:
                span_flag.remove(span['spanID'])
                # fill in trace info
                trace_dict = add_outbound_trace_information(span, trace_dict)
                trace_dict['svc'].append('OTHER_SVC')
                trace_dict['call'].append((svc, 'OTHER_SYSTEM'))
                trace_dict['call_instance'].append(
                    (handle_node_id_istio(node_id), handle_node_id_istio('OTHER_NODE')))
                # fill in half_trace info
                half_trace_dict = add_outbound_trace_information(span, half_trace_dict)
                half_trace_dict['svc'].append('OTHER_SVC')
                half_trace_dict['call'].append((svc, 'OTHER_SYSTEM'))
                half_trace_dict['call_instance'].append(
                    (handle_node_id_istio(node_id), handle_node_id_istio('OTHER_NODE')))
            # inbound edge, fill in the info
            elif is_inbound_span(span) is True:
                span_flag.remove(span['spanID'])
                # fill in trace info
                trace_dict = add_inbound_trace_information(span, trace_dict)
                trace_dict['svc'].append(svc)
                trace_dict['call'].append(('OTHER_SYSTEM', svc))
                trace_dict['call_instance'].append(
                    (handle_node_id_istio('OTHER_NODE'), handle_node_id_istio(node_id)))
                # fill in half_trace info
                half_trace_dict = add_inbound_trace_information(span, half_trace_dict)
                half_trace_dict['svc'].append(svc)
                half_trace_dict['call'].append(('OTHER_SYSTEM', svc))
                half_trace_dict['call_instance'].append(
                    (handle_node_id_istio('OTHER_NODE'), handle_node_id_istio(node_id)))

        # Decide, based on flags, whether to add to each trace's info
        # Judge whether trace_dict is normal
        if is_span_abnormal(trace_dict) or is_abnormal:
            abnormal_dicts[trace_json['traceID']] = trace_dict
        else:
            normal_dicts[trace_json['traceID']] = trace_dict
        # Judge whether half_trace_dict is normal
        if is_span_abnormal(half_trace_dict) and (is_inbound is True or is_outbound is True):
            abnormal_half_dicts[trace_json['traceID']] = half_trace_dict
        # Determine outbound and inbound edges
        if is_inbound is True:
            inbound_dicts[trace_json['traceID']] = trace_dict
            inbound_half_dicts[trace_json['traceID']] = half_trace_dict
        if is_outbound is True:
            outbound_dicts[trace_json['traceID']] = trace_dict
            outbound_half_dicts[trace_json['traceID']] = half_trace_dict

    return normal_dicts, inbound_dicts, outbound_dicts, abnormal_dicts, inbound_half_dicts, outbound_half_dicts, abnormal_half_dicts


'''
    Get an empty trace_dict
'''
def get_empty_trace_dict(traceId):
    trace_dict = {'traceId': traceId}
    # Define trace fields
    trace_dict['call'] = []
    trace_dict['timestamp'] = []
    trace_dict['latency'] = []
    trace_dict['http_status'] = []
    trace_dict['svc'] = []
    trace_dict['call_instance'] = []

    return trace_dict


'''
    Get the parent span of this span
'''
def get_outbound_span(spans_dict, span_json, trace_dict):
    outbound_span = {}
    # reference is empty: top-level node, return None
    if span_json['references'] == []:
        return None
    else:
        for ref in span_json['references']:
            if ref['refType'] == 'CHILD_OF':
                # references an invalid span, return None
                if ref['spanID'] not in spans_dict.keys():
                    return None
                else:  # the referenced span exists, check if it is useful
                    tmp_span = spans_dict[ref['spanID']]
                    outbound_span = tmp_span
    return outbound_span


'''
    Determine whether this span is an outbound edge
'''
def is_outbound_span(span_json):
    for tag in span_json['tags']:
        if 'upstream_cluster' == tag['key']:
            if 'outbound' in tag['value']:
                return True

    return False


'''
    Determine whether this span is an inbound edge
'''
def is_inbound_span(span_json):
    for tag in span_json['tags']:
        if 'upstream_cluster' == tag['key']:
            if 'inbound' in tag['value']:
                return True

    return False


'''
    Get the basic info of a span
'''
def get_span_information(span_json):

    node_id = None
    svc = None
    namespace = None

    for tag in span_json['tags']:
        if 'node_id' == tag['key']:
            node_id = tag['value']
        if 'istio.canonical_service' == tag['key']:
            svc = tag['value']
            if svc == 'unknown':
                svc = 'istio-ingressgateway'
        if 'istio.namespace' == tag['key']:
            namespace = tag['value']
    if namespace == 'istio-system':
        namespace = 'bookinfo'

    return node_id, svc, namespace


'''
    Add info to trace_dict
'''
def add_trace_information(span_json, trace_dict):
    trace_dict['timestamp'].append(span_json['startTime'] + clock_skew_adjust(span_json))
    trace_dict['latency'].append(span_json['duration'])
    # Judge normality from the status code
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            trace_dict['http_status'].append(int(tag['value']))

    return trace_dict


'''
    Process node_id
'''
def handle_node_id_istio(node_id: str) -> str:
    if node_id == 'OTHER_NODE':
        return 'OTHER_NODE'
    s = node_id.split('~')
    if len(s) != 4:
        raise Exception('istio trace node id illegal')
    return s[2] + '-' + s[1]


'''
    Process pod_id
'''
def handle_istio_pod_id(node_id: str) -> str:
    if node_id is None:
        return 'None'
    s = node_id.split('~')
    if len(s) != 4:
        raise Exception('istio trace node id illegal')
    return s[2].split('.')[0]


'''
    Add info to a half_trace whose outbound edge is in-system
'''
def add_outbound_trace_information(outbound_span, half_trace_dict):
    # add outbound trace info
    half_trace_dict['timestamp'].append(outbound_span['startTime'] + clock_skew_adjust(outbound_span))
    half_trace_dict['latency'].append(outbound_span['duration'])
    for tag in outbound_span['tags']:
        if tag['key'] == 'http.status_code':
            half_trace_dict['http_status'].append(int(tag['value']))
    # add inbound trace info
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)

    return half_trace_dict


'''
    Add info to a half_trace whose inbound edge is in-system
'''
def add_inbound_trace_information(span_json, half_trace_dict):
    # add outbound trace info
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)
    # add outbound trace info
    half_trace_dict['timestamp'].append(span_json['startTime'] + clock_skew_adjust(span_json))
    half_trace_dict['latency'].append(span_json['duration'])
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            half_trace_dict['http_status'].append(int(tag['value']))

    return half_trace_dict


'''
    Determine whether a span is abnormal
'''
def is_span_abnormal(trace_dict):
    # empty call relation, mark as abnormal
    if trace_dict['call'] == []:
        return True
    for code in trace_dict['http_status']:
        if (code < 200 and code != -1) or code >= 300:
            return True
    return False


'''Clock adjustment using jaeger'''
def clock_skew_adjust(span_json):
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
