import requests
from Config import Config
import pickle
from util.KubernetesClient import KubernetesClient
from .TraceNoIstio import handle_traces_no_istio
from .TraceIstio import handle_traces_istio
import os

global_svcs = []
global_namespace_svcs_dict = {}

def collect(config: Config, _dir: str):
    if not os.path.exists(_dir):
        os.makedirs(_dir)
    global global_namespace_svcs_dict
    global_namespace_svcs_dict = KubernetesClient(config).get_all_svc()
    # config.svcs = KubernetesClient(config).get_svc_list_name()
    # Determine the services to access in the namespace
    svcs = [svc for svc in config.svcs if
            'unknown' not in svc and 'redis' not in svc and 'istio-ingressgateway' not in svc]
    if config.namespace == 'horsecoder-test':
        svcs = ['edge-llm-svc', 'edge-paraformer-serverless-svc', 'edge-gateway-svc']
    elif config.namespace == 'trainticket':
        svcs = KubernetesClient(config).get_svc_list_name()
    global global_svcs
    global_svcs = svcs
    # urls = build_trace_urls(config)
    normal_dicts = {}
    inbound_dicts = {}
    outbound_dicts = {}
    abnormal_dicts = {}
    inbound_half_dicts = {}
    outbound_half_dicts = {}
    abnormal_half_dicts = {}
    pod_latency = {}
    net_latency = {}

    # Iterate services, collecting the various trace data
    for svc in svcs:
        if config.namespace == 'cloud-sock-shop' or config.namespace == 'trainticket':
            url = build_trace_url(config, svc)
            normal_dict, inbound_dict, outbound_dict, abnormal_dict, inbound_half_dict, outbound_half_dict, abnormal_half_dict = handle_traces_no_istio(pull(url), config)
        elif config.namespace == 'horsecoder-test':
            url = build_trace_url(config, svc[:-4] + '-1.0.0')
            normal_dict, inbound_dict, outbound_dict, abnormal_dict, inbound_half_dict, outbound_half_dict, abnormal_half_dict = handle_traces_no_istio(pull(url), config)
        else:
            url = build_trace_url(config, svc + '.' + config.namespace)
            normal_dict, inbound_dict, outbound_dict, abnormal_dict, inbound_half_dict, outbound_half_dict, abnormal_half_dict = handle_traces_istio(pull(url), config)
        normal_dicts = {**normal_dicts, **normal_dict}
        inbound_dicts = {**inbound_dicts, **inbound_dict}
        outbound_dicts = {**outbound_dicts, **outbound_dict}
        abnormal_dicts = {**abnormal_dicts, **abnormal_dict}
        inbound_half_dicts = {**inbound_half_dicts, **inbound_half_dict}
        outbound_half_dicts = {**outbound_half_dicts, **outbound_half_dict}
        abnormal_half_dicts = {**abnormal_half_dicts, **abnormal_half_dict}

    # print(normal_dicts)
    # Process pod_latency and net_latency data
    pod_latency, net_latency = get_latency([normal_dicts, abnormal_dicts])
    # print(net_latency)

    # Persist the data
    normal_path = _dir + '/' + 'normal.pkl'
    inbound_path = _dir + '/' + 'inbound.pkl'
    outbound_path = _dir + '/' + 'outbound.pkl'
    abnormal_path = _dir + '/' + 'abnormal.pkl'
    inbound_half_path = _dir + '/' + 'inbound_half.pkl'
    outbound_half_path = _dir + '/' + 'outbound_half.pkl'
    abnormal_half_path = _dir + '/' + 'abnormal_half.pkl'
    trace_pod_latency_path = _dir + '/' + 'trace_pod_latency.pkl'
    trace_net_latency_path = _dir + '/' + 'trace_net_latency.pkl'

    with open(normal_path, 'ab') as fw:
        pickle.dump(list(normal_dicts.values()), fw)
    with open(inbound_path, 'ab') as fw:
        pickle.dump(list(inbound_dicts.values()), fw)
    with open(outbound_path, 'ab') as fw:
        pickle.dump(list(outbound_dicts.values()), fw)
    with open(abnormal_path, 'ab') as fw:
        pickle.dump(list(abnormal_dicts.values()), fw)
    with open(inbound_half_path, 'ab') as fw:
        pickle.dump(list(inbound_half_dicts.values()), fw)
    with open(outbound_half_path, 'ab') as fw:
        pickle.dump(list(outbound_half_dicts.values()), fw)
    with open(abnormal_half_path, 'ab') as fw:
        pickle.dump(list(abnormal_half_dicts.values()), fw)
    with open(trace_pod_latency_path, 'ab') as fw:
        pickle.dump(pod_latency, fw)
    with open(trace_net_latency_path, 'ab') as fw:
        pickle.dump(net_latency, fw)


def build_trace_url(config: Config, svc):
    url = '{}end={}&start={}&limit={}&lookback={}&maxDuration&minDuration&service={}' \
        .format(config.jaeger_url, config.end * 1000000, config.start * 1000000, config.limit, config.lookBack,
                svc)
    return url


def build_trace_urls(config: Config):
    '''
        Build the per-service per-minute trace fetch path (avoids too much data)
    '''
    svcs = [svc + '.' + config.namespace for svc in config.svcs if
            'unknown' not in svc and 'redis' not in svc and 'istio-ingressgateway' not in svc]
    urls = ['{}end={}&start={}&limit={}&lookback={}&maxDuration&minDuration&service={}' \
                .format(config.jaeger_url, config.end * 1000000, config.start * 1000000, config.limit, config.lookBack,
                        svc) for
            svc in
            svcs]
    return urls


def pull(url):
    '''
        Fetch traces
    '''
    response = requests.get(url)
    return response.json()['data']


def handle_istio_node_id(node_id: str) -> str:
    if node_id == 'OTHER_NODE':
        return 'OTHER_NODE'
    if node_id is None:
        return 'None'
    s = node_id.split('~')
    if len(s) != 4:
        raise Exception('istio trace node id illegal')
    return s[2] + '-' + s[1]


def handle_istio_pod_id(node_id: str) -> str:
    if node_id is None:
        return 'None'
    s = node_id.split('~')
    if len(s) != 4:
        raise Exception('istio trace node id illegal')
    return s[2].split('.')[0]


def handle_traces_istio_old(trace_jsons, current_svc):
    # List of traces to build
    normal_dicts = {}
    inbound_dicts = {}
    outbound_dicts = {}
    abnormal_dicts = {}
    inbound_half_dicts = {}
    outbound_half_dicts = {}
    abnormal_half_dicts = {}
    pod_latency = {}
    # Iterate all traces
    for trace_json in trace_jsons:

        # Define flags
        is_abnormal = False
        is_inbound = False
        is_outbound = False
        is_half = False  # whether it needs to be split

        # Define a trace
        half_trace_dict = {}
        trace_dict = {'traceId': trace_json['traceID']}
        # Define trace fields
        trace_dict['call'] = []
        trace_dict['timestamp'] = []
        trace_dict['latency'] = []
        trace_dict['http_status'] = []
        trace_dict['svc'] = []
        trace_dict['call_instance'] = []

        # Build a flag array holding spans not yet visited
        span_flag = []

        # Get all spans in the trace
        spans_dict = {}
        for span_json in trace_json['spans']:
            spans_dict[span_json['spanID']] = span_json
            span_flag.append(span_json['spanID'])

        # Iterate all spans in the trace, processing data (every span looks up other spans within this trace)
        for span_json in trace_json['spans']:
            # Get basic info
            trace_dict['timestamp'].append(span_json['startTime'])
            trace_dict['latency'].append(span_json['duration'])
            # Judge normality from the status code
            for tag in span_json['tags']:
                if tag['key'] == 'http.status_code':
                    http_status = int(tag['value'])
                    trace_dict['http_status'].append(tag['value'])
                    if http_status < 200 or http_status >= 300:
                        is_abnormal = True

            # Set variables
            node_id = None  # node id
            caller_svc = None  # caller service name
            callee_svc = None  # callee service name
            outbound_node_id = None  # outbound node id
            namespace = None  # namespace
            outbound_namespace = None  # outbound namespace

            # Iterate all tags in the span
            for tag in span_json['tags']:
                # find the node id
                if 'node_id' == tag['key']:
                    node_id = tag['value']
                # find the callee service name (if present)
                if 'istio.canonical_service' == tag['key']:
                    callee_svc = tag['value']
                if 'istio.namespace' == tag['key']:
                    namespace = tag['value']

            # Check whether it is an outbound or inbound edge
            for tag in span_json['tags']:
                # if inbound, check for references
                if 'upstream_cluster' == tag['key']:
                    if 'inbound' in tag['value']:
                        # remove this span's flag
                        span_flag.remove(span_json['spanID'])
                        # with a reference, find the matching outbound edge and bind them as a pair
                        if span_json['references'] != []:
                            for ref in span_json['references']:
                                if ref['refType'] == 'CHILD_OF':
                                    # find the outbound span
                                    if ref['spanID'] not in spans_dict.keys():
                                        break
                                    else:
                                        outbound_span = spans_dict[ref['spanID']]
                                        # remove this span's flag
                                        if ref['spanID'] in span_flag:
                                            span_flag.remove(ref['spanID'])
                                        # get the outbound edge info
                                        for outbound_tag in outbound_span['tags']:
                                            # find the outbound node id
                                            if 'node_id' == outbound_tag['key']:
                                                outbound_node_id = outbound_tag['value']
                                            # find the caller service name
                                            if 'istio.canonical_service' == outbound_tag['key']:
                                                caller_svc = outbound_tag['value']
                                                # if the service name is unknown, replace it
                                                if caller_svc == 'unknown':
                                                    caller_svc = 'istio-ingressgateway'
                                            if 'istio.namespace' == outbound_tag['key']:
                                                outbound_namespace = outbound_tag['value']
                                        # determine whether it crosses systems (by namespace)
                                        # determine outbound vs inbound (by service name)
                                        # change the flag
                                        if caller_svc == 'istio-ingressgateway':
                                            outbound_namespace = namespace
                                        if namespace != outbound_namespace:
                                            # cross-system trace: determine inbound vs outbound and build a truncated trace_dict
                                            half_trace_dict = get_half_trace(trace_json)
                                            is_half = True
                                            if current_svc != callee_svc:
                                                # this side is the caller, i.e. outbound
                                                is_outbound = True
                                                half_trace_dict['timestamp'][len(half_trace_dict['timestamp']) - 1] = -1
                                                half_trace_dict['latency'][len(half_trace_dict['latency']) - 1] = -1
                                                half_trace_dict['http_status'][len(half_trace_dict['http_status']) - 1] = -1
                                                half_trace_dict['svc'].append(caller_svc)
                                                half_trace_dict['call'].append((caller_svc, 'OTHER_SYSTEM'))
                                                half_trace_dict['call_instance'].append(
                                                    (handle_istio_node_id(node_id), handle_istio_node_id('OTHER_NODE')))
                                            else:
                                                is_inbound = True
                                                half_trace_dict['timestamp'][len(half_trace_dict['timestamp']) - 2] = -1
                                                half_trace_dict['latency'][len(half_trace_dict['latency']) - 2] = -1
                                                half_trace_dict['http_status'][len(half_trace_dict['http_status']) - 2] = -1
                                                half_trace_dict['svc'].append(callee_svc)
                                                half_trace_dict['call'].append(('OTHER_SYSTEM', callee_svc))
                                                half_trace_dict['call_instance'].append(
                                                    (handle_istio_node_id('OTHER_NODE'), handle_istio_node_id(node_id)))

                        # if no reference, substitute a specific placeholder
                        else:
                            caller_svc = 'OTHER_SYSTEM'
                            outbound_node_id = 'OTHER_NODE'
                            # inbound edge, change the flag
                            is_inbound = True
                            # other info
                            trace_dict['timestamp'].insert(len(trace_dict['timestamp']) - 1, -1)
                            trace_dict['latency'].insert(len(trace_dict['latency']) - 1, -1)
                            trace_dict['http_status'].insert(len(trace_dict['http_status']) - 1, -1)
                        # add the call relation to the trace
                        trace_dict['svc'].append(callee_svc)
                        # fill in the info
                        if node_id is not None:
                            trace_dict['call'].append((caller_svc, callee_svc))
                            trace_dict['call_instance'].append(
                                (handle_istio_node_id(outbound_node_id), handle_istio_node_id(node_id)))
                            if outbound_node_id != 'OTHER_NODE' and node_id != 'OTHER_NODE':
                                pod_latency_key = handle_istio_pod_id(outbound_node_id) + '&' + handle_istio_pod_id(
                                    node_id)
                                pod_latency_list = pod_latency.get(pod_latency_key, [])
                                pod_latency_list.append(span_json['duration'])
                                pod_latency[pod_latency_key] = pod_latency_list

        # Iterate again to find outbound edges without a matching call relation
        for span_id in span_flag:
            span = spans_dict[span_id]
            node_id = None
            caller_svc = None
            # get info
            for span_tag in span['tags']:
                # find the node id
                if 'node_id' == span_tag['key']:
                    node_id = span_tag['value']
                # find the service name
                if 'istio.canonical_service' == span_tag['key']:
                    caller_svc = span_tag['value']

            # check if it is an outbound edge; if so, add it to the trace
            for span_tag in span['tags']:
                if 'upstream_cluster' == span_tag['key']:
                    if 'outbound' in span_tag['value']:
                        # remove its flag
                        span_flag.remove(span['spanID'])
                        trace_dict['svc'].append('OTHER_SERVICE')
                        # fill in the info
                        if node_id is not None:
                            trace_dict['call'].append((caller_svc, 'OTHER_SYSTEM'))
                            trace_dict['call_instance'].append(
                                (handle_istio_node_id(node_id), handle_istio_node_id('OTHER_NODE')))
                            # other info
                            trace_dict['timestamp'].insert(len(trace_dict['timestamp']), -1)
                            trace_dict['latency'].insert(len(trace_dict['latency']), -1)
                            trace_dict['http_status'].insert(len(trace_dict['latency']), -1)
                        # outbound edge, change the flag
                        is_outbound = True

        # Iterate again to judge abnormal conditions
        if span_flag != []:
            is_abnormal = True

        # Decide, based on flags, whether to add to each trace's info
        if is_abnormal == True:
            abnormal_dicts[trace_json['traceID']] = trace_dict
            if is_half == True:
                abnormal_half_dicts[trace_json['traceID']] = half_trace_dict
            else:
                abnormal_half_dicts[trace_json['traceID']] = trace_dict
        else:
            normal_dicts[trace_json['traceID']] = trace_dict
        if is_inbound == True:
            inbound_dicts[trace_json['traceID']] = trace_dict
            if is_half == True:
                inbound_half_dicts[trace_json['traceID']] = half_trace_dict
            else:
                inbound_half_dicts[trace_json['traceID']] = trace_dict
        if is_outbound == True:
            outbound_dicts[trace_json['traceID']] = trace_dict
            if is_half == True:
                outbound_half_dicts[trace_json['traceID']] = half_trace_dict
            else:
                outbound_half_dicts[trace_json['traceID']] = trace_dict
    return normal_dicts, inbound_dicts, outbound_dicts, abnormal_dicts, inbound_half_dicts, outbound_half_dicts, abnormal_half_dicts, pod_latency



def get_half_trace(trace_json):  # get the needed half trace_dict
    # Define a trace
    trace_dict = {'traceId': trace_json['traceID']}
    # Define trace fields
    trace_dict['call'] = []
    trace_dict['timestamp'] = []
    trace_dict['latency'] = []
    trace_dict['http_status'] = []
    trace_dict['svc'] = []
    trace_dict['call_instance'] = []

    for span_json in trace_json['spans']:
        # Get basic info
        trace_dict['timestamp'].append(span_json['startTime'])
        trace_dict['latency'].append(span_json['duration'])
        # Judge normality from the status code
        for tag in span_json['tags']:
            if tag['key'] == 'http.status_code':
                trace_dict['http_status'].append(tag['value'])

    return trace_dict


'''
    Get pod and net latency (discard entries that do not meet requirements)
'''
def get_latency(trace_list):
    pod_latency = {}
    net_latency = {}
    # Iterate each trace to find relevant data
    for trace_dict in trace_list:
        for key in trace_dict.keys():
            trace = trace_dict[key]
            # iterate each call relation in the trace
            for i in range(len(trace['call'])):
                call = trace['call'][0]
                # determine whether it is communication between two pods
                if call[0] != 'OTHER_SYSTEM' and call[1] != 'OTHER_SYSTEM':
                    latency_key = call_instance_to_pod_name(trace['call_instance'][i])
                    # process net_latency
                    net_latency_list = net_latency.get(latency_key, [])
                    # net_latency is the time between sending and receiving packets
                    send_latency = trace['timestamp'][2 * i + 1] - trace['timestamp'][2 * i]
                    res_latency = (trace['timestamp'][2 * i] + trace['latency'][2 * i]) - (trace['timestamp'][2 * i + 1] + trace['latency'][2 * i + 1])
                    # if the send/receive time is negative, discard it
                    # if send_latency < 0 or res_latency < 0:
                    #     continue
                    net_latency_list.append((send_latency, res_latency))
                    net_latency[latency_key] = net_latency_list
                    # process pod_latency
                    pod_latency_list = pod_latency.get(latency_key, [])
                    # pod_latency is the duration of the outbound service
                    pod_latency_list.append(trace['latency'][2 * i])
                    pod_latency[latency_key] = pod_latency_list

    return pod_latency, net_latency


'''
    Extract the pod name from call_instance
'''
def call_instance_to_pod_name(call_instance):
    # return call_instance[0].split('.')[0] + '&' + call_instance[1].split('.')[1]
    return call_instance[0] + '&' + call_instance[1]

