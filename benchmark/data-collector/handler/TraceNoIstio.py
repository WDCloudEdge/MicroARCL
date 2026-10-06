from Config import Config
from util.KubernetesClient import KubernetesClient


global_namespace_svcs_dict = {}
current_namespace = None


'''
    对不在服务网格中的微服务系统收集trace
'''
def handle_traces_no_istio(trace_jsons, config: Config):
    # 获取各命名空间的服务名称
    global global_namespace_svcs_dict
    global_namespace_svcs_dict = KubernetesClient(config).get_all_svc()
    global current_namespace
    current_namespace = config.namespace

    # 定义返回数据
    normal_dicts = {}
    inbound_dicts = {}
    outbound_dicts = {}
    abnormal_dicts = {}
    inbound_half_dicts = {}
    outbound_half_dicts = {}
    abnormal_half_dicts = {}

    # 遍历所有trace
    for trace_json in trace_jsons:
        # 定义trace_dict
        trace_dict = get_empty_trace_dict(trace_json['traceID'])
        half_trace_dict = get_empty_trace_dict(trace_json['traceID'])
        # 获取trace的process
        process_dicts = get_trace_process(trace_json)
        # 获取trace中的所有span
        spans_dict = {}
        for span_json in trace_json['spans']:
            spans_dict[span_json['spanID']] = span_json

        # 定义标志位
        is_inbound = False
        is_outbound = False

        # 处理所有的span
        for span_json in trace_json['spans']:
            # 判断该span是否有用,如果无用或为顶级span，则跳过这个span
            is_span_useful, span_json = get_span_useful(span_json)
            # 获取出边span信息
            outbound_span = get_outbound_span(spans_dict, span_json, trace_dict)
            if is_span_useful is False or outbound_span is None:
                continue

            # 添加信息
            inbound_pod_ip, inbound_pod_name, callee_svc, inbound_namespace = get_span_information(span_json, process_dicts)
            outbound_pod_ip, outbound_pod_name, caller_svc, outbound_namespace = get_span_information(outbound_span, process_dicts)
            trace_dict = add_trace_information(span_json, outbound_span, trace_dict, spans_dict)

            # 将调用关系添加到trace中
            trace_dict['svc'].append(callee_svc)
            trace_dict['call'].append((caller_svc, callee_svc))
            trace_dict['call_instance'].append((handle_node_id_no_istio(outbound_pod_name, outbound_pod_ip, outbound_namespace), handle_node_id_no_istio(inbound_pod_name, inbound_pod_ip, inbound_namespace)))

            # 判断是否跨系统，并选择将其加入到half_trace_dict中
            # 入边跨系统，出边为系统内
            if current_namespace != inbound_namespace and current_namespace == outbound_namespace:
                is_outbound = True
                half_trace_dict = add_outbound_trace_information(outbound_span, half_trace_dict, span_json, spans_dict)
                half_trace_dict['svc'].append('OTHER_SVC')
                half_trace_dict['call'].append((caller_svc, 'OTHER_SYSTEM'))
                half_trace_dict['call_instance'].append((handle_node_id_no_istio(outbound_pod_name, outbound_pod_ip, outbound_namespace),handle_node_id_no_istio('OTHER_NODE', '', '')))
            # 出边跨系统，入边为系统内
            elif current_namespace == inbound_namespace and current_namespace != outbound_namespace:
                is_inbound = True
                half_trace_dict = add_inbound_trace_information(span_json, half_trace_dict)
                half_trace_dict['svc'].append(callee_svc)
                half_trace_dict['call'].append(('OTHER_SYSTEM', callee_svc))
                half_trace_dict['call_instance'].append((handle_node_id_no_istio('OTHER_NODE', '', ''), handle_node_id_no_istio(inbound_pod_name, inbound_pod_ip, inbound_namespace)))
            # 都在服务内，将其完整装入half_trace_dict中
            elif current_namespace == inbound_namespace and current_namespace == outbound_namespace:
                half_trace_dict = add_trace_information(span_json, outbound_span, half_trace_dict, spans_dict)
                half_trace_dict['svc'].append(callee_svc)
                half_trace_dict['call'].append((caller_svc, callee_svc))
                half_trace_dict['call_instance'].append((handle_node_id_no_istio(outbound_pod_name, outbound_pod_ip, outbound_namespace), handle_node_id_no_istio(inbound_pod_name, inbound_pod_ip, inbound_namespace)))
            # 两方都跨系统则不将其加入half_trace数据中

        # 如果trace的调用关系为空，则部将其加入数据中
        if trace_dict['call'] == []:
            continue

        # 根据标志位判断是否加入个trace信息中
        # 判断trace_dict是否正常
        if is_abnormal(trace_dict):
            abnormal_dicts[trace_json['traceID']] = trace_dict
        else:
            normal_dicts[trace_json['traceID']] = trace_dict
        # 判断half_trace_dict是否正常
        if is_abnormal(half_trace_dict) and (is_inbound is True or is_outbound is True):
            abnormal_half_dicts[trace_json['traceID']] = half_trace_dict
        # 判断出边和入边
        if is_inbound is True:
            inbound_dicts[trace_json['traceID']] = trace_dict
            inbound_half_dicts[trace_json['traceID']] = half_trace_dict
        if is_outbound is True:
            outbound_dicts[trace_json['traceID']] = trace_dict
            outbound_half_dicts[trace_json['traceID']] = half_trace_dict

    return normal_dicts, inbound_dicts, outbound_dicts, abnormal_dicts, inbound_half_dicts, outbound_half_dicts, abnormal_half_dicts

'''
    得到一个空的trace_dict
'''
def get_empty_trace_dict(traceId):
    trace_dict = {'traceId': traceId}
    # 定义trace字段
    trace_dict['call'] = []
    trace_dict['timestamp'] = []
    trace_dict['latency'] = []
    trace_dict['http_status'] = []
    trace_dict['svc'] = []
    trace_dict['call_instance'] = []

    return trace_dict


'''
    获取trace数据中的process数据
    process 的 key: processId, service_name, pod_name, pod_ip
'''
def get_trace_process(trace_json):
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


'''
    判断span是否有用
'''
def get_span_useful(span_json):
    is_span_useful = False

    for tag in span_json['tags']:
        # 查找是否含有net.host.name的标签
        if tag['key'] == 'net.host.name':
            is_span_useful = True
            break
        # 查找是否含有edge请求相关的标签
        if tag['key'] == 'thread.name' and 'DubboServerHandler' in tag['value']:
            for tg in span_json['tags']:
                if tg['key'] == 'rpc.method':
                    is_span_useful = True
                    break
            break

    return is_span_useful, span_json


'''
    寻找该span的父span
'''
def get_outbound_span(spans_dict, span_json, trace_dict):
    outbound_span = {}
    # 引用为空，为顶级节点，返回None
    if span_json['references'] == []:
        return None
    else:
        for ref in span_json['references']:
            if ref['refType'] == 'CHILD_OF':
                # 引用了无效的span，返回None
                if ref['spanID'] not in spans_dict.keys():
                    return None
                else:  # 引用的span存在，判断是否有用
                    tmp_span = spans_dict[ref['spanID']]
                    tmp_span_useful, test_span = get_span_useful(tmp_span)
                    # 如果有用，则返回；否则继续向上递归
                    if tmp_span_useful is True:
                        outbound_span = tmp_span
                        break
                    else:
                        outbound_span = get_outbound_span(spans_dict, tmp_span, trace_dict)
    return outbound_span


'''
    向trace_dict添加timestamp,latency,http_status数据
'''
def add_trace_information(span_json, outbound_span, trace_dict, spans_dict):
    # 添加出边信息
    is_horsecoder_outbound_span = True
    # 获取出边的时间信息
    outbound_timestamp, outbound_latency = get_outbound_time(span_json, spans_dict)
    trace_dict['timestamp'].append(outbound_timestamp + clock_skew_adjust(span_json))
    trace_dict['latency'].append(outbound_latency)
    for tag in outbound_span['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_outbound_span = False
            trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_outbound_span == True:
        trace_dict['http_status'].append(200)
    # 添加入边信息
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


'''
    向出边为系统内部的half_trace添加信息
'''
def add_outbound_trace_information(outbound_span, half_trace_dict, span_json, spans_dict):
    is_horsecoder_span = True
    # 添加出边的trace信息
    outbound_timestamp, outbound_latency = get_outbound_time(span_json, spans_dict)
    half_trace_dict['timestamp'].append(outbound_timestamp + clock_skew_adjust(outbound_span))
    half_trace_dict['latency'].append(outbound_latency)
    for tag in outbound_span['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_span = False
            half_trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_span == True:
        half_trace_dict['http_status'].append(200)
    # 添加入边的trace信息
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)

    return half_trace_dict


'''
    向入边为系统内部的half_trace添加信息
'''
def add_inbound_trace_information(span_json, half_trace_dict):
    is_horsecoder_span = True
    # 添加出边的trace信息
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)
    # 添加出边的trace信息
    half_trace_dict['timestamp'].append(span_json['startTime'] + + clock_skew_adjust(span_json))
    half_trace_dict['latency'].append(span_json['duration'])
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            is_horsecoder_span = False
            half_trace_dict['http_status'].append(tag['value'])
    if is_horsecoder_span == True:
        half_trace_dict['http_status'].append(200)

    return half_trace_dict


'''
    获得span信息
'''
def get_span_information(span_json, process_dicts):
    # horsecoder 服务名转换
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


'''
    获得服务对应的命名空间（对hipster和hipster2可能会产生冲突，但应该不存在与二者相关的链路）
'''
def get_svc_namespace(svc):
    global global_namespace_svcs_dict
    for key in global_namespace_svcs_dict.keys():
        svc_list = global_namespace_svcs_dict[key]
        if svc in svc_list:
            return key
    print(svc + "not found.")
    return None


'''
    合成调用实例名
'''
def handle_node_id_no_istio(node_name, node_id, namespace):
    if node_name == 'OTHER_NODE':
        return 'OTHER_NODE'
    return str(node_name) + '.' + str(namespace) + '-' + node_id


'''
    判断trace是否正常
'''
def is_abnormal(trace_dict):
    # 调用关系为空，则设定为不正常
    if trace_dict['call'] == []:
        return True
    for code in trace_dict['http_status']:
        if (code < 200 and code != -1) or code >= 300:
            return True
    return False


'''
    获取父span的时间参数
'''
def get_outbound_time(span_json, spans_dict):
    outbound_timestamp = None
    outbound_latency = None
    # 引用为空，为顶级节点，返回None
    if span_json['references'] == []:
        return None
    else:
        for ref in span_json['references']:
            if ref['refType'] == 'CHILD_OF':
                # 引用了无效的span，返回None
                if ref['spanID'] not in spans_dict.keys():
                    return None
                else:  # 引用的span存在，判断是否有用
                    tmp_span = spans_dict[ref['spanID']]
                    outbound_timestamp = tmp_span['startTime'] + clock_skew_adjust(tmp_span)
                    outbound_latency = tmp_span['duration']

    return outbound_timestamp, outbound_latency


'''使用jaeger的时钟调整'''
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







