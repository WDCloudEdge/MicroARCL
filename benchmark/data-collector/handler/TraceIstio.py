from Config import Config
from util.KubernetesClient import KubernetesClient

global_namespace_svcs_dict = {}
current_namespace = None

'''
    对处于服务网格中的微服务系统收集trace
'''
def handle_traces_istio(trace_jsons, config: Config):
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
    # 遍历每一个trace
    for trace_json in trace_jsons:
        # 定义trace_dict
        trace_dict = get_empty_trace_dict(trace_json['traceID'])
        half_trace_dict = get_empty_trace_dict(trace_json['traceID'])

        # 建立标志位数组，存放没有被访问过的span
        span_flag = []
        # 获取trace中的所有span
        spans_dict = {}
        for span_json in trace_json['spans']:
            spans_dict[span_json['spanID']] = span_json
            span_flag.append(span_json['spanID'])

        # 定义标志位
        is_inbound = False
        is_outbound = False
        is_abnormal = False

        for span_json in trace_json['spans']:
            # 如果是入边，获取出边span信息
            outbound_span = None
            if is_inbound_span(span_json):
                outbound_span = get_outbound_span(spans_dict, span_json, trace_dict)
            # 有出边，获取出边的信息，移除该span的标志位
            if outbound_span != None:
                if span_json['spanID'] in span_flag:
                    span_flag.remove(span_json['spanID'])
                if outbound_span['spanID'] in span_flag:
                    span_flag.remove(outbound_span['spanID'])
            else:
                continue
            # 添加信息
            node_id, svc, namespace = get_span_information(span_json)
            outbound_node_id, caller_svc, outbound_namespace = get_span_information(outbound_span)
            trace_dict = add_trace_information(outbound_span, trace_dict)
            trace_dict = add_trace_information(span_json, trace_dict)

            # 将调用关系添加到trace中
            trace_dict['svc'].append(svc)
            trace_dict['call'].append((caller_svc, svc))
            trace_dict['call_instance'].append((handle_node_id_istio(outbound_node_id), handle_node_id_istio(node_id)))

            # 判断是否跨系统，并选择将其加入到half_trace_dict中
            # 入边跨系统，出边为系统内
            if current_namespace != namespace and current_namespace == outbound_namespace:
                is_outbound = True
                half_trace_dict = add_outbound_trace_information(outbound_span, half_trace_dict)
                half_trace_dict['svc'].append('OTHER_SVC')
                half_trace_dict['call'].append((caller_svc, 'OTHER_SYSTEM'))
                half_trace_dict['call_instance'].append((handle_node_id_istio(outbound_node_id), handle_node_id_istio('OTHER_NODE')))
            # 出边跨系统，入边为系统内
            elif current_namespace == namespace and current_namespace != outbound_namespace:
                is_inbound = True
                half_trace_dict = add_inbound_trace_information(span_json, half_trace_dict)
                half_trace_dict['svc'].append(svc)
                half_trace_dict['call'].append(('OTHER_SYSTEM', svc))
                half_trace_dict['call_instance'].append((handle_node_id_istio('OTHER_NODE'), handle_node_id_istio(node_id)))
            # 都在服务内，将其完整装入half_trace_dict中
            elif current_namespace == namespace and current_namespace == outbound_namespace:
                half_trace_dict = add_trace_information(outbound_span, half_trace_dict)
                half_trace_dict = add_trace_information(span_json, half_trace_dict)
                half_trace_dict['svc'].append(svc)
                half_trace_dict['call'].append((caller_svc, svc))
                half_trace_dict['call_instance'].append((handle_node_id_istio(outbound_node_id), handle_node_id_istio(node_id)))
            # 两方都跨系统则不将其加入half_trace数据中

        # 再次进行遍历，找到没有对应调用关系的出边
        for span_id in span_flag:
            span = spans_dict[span_id]
            node_id, svc, namespace = get_span_information(span)
            # 是出边，则将信息补全
            if is_outbound_span(span) is True:
                span_flag.remove(span['spanID'])
                # 补充trace信息
                trace_dict = add_outbound_trace_information(span, trace_dict)
                trace_dict['svc'].append('OTHER_SVC')
                trace_dict['call'].append((svc, 'OTHER_SYSTEM'))
                trace_dict['call_instance'].append(
                    (handle_node_id_istio(node_id), handle_node_id_istio('OTHER_NODE')))
                # 补充half_trace信息
                half_trace_dict = add_outbound_trace_information(span, half_trace_dict)
                half_trace_dict['svc'].append('OTHER_SVC')
                half_trace_dict['call'].append((svc, 'OTHER_SYSTEM'))
                half_trace_dict['call_instance'].append(
                    (handle_node_id_istio(node_id), handle_node_id_istio('OTHER_NODE')))
            # 是入边，则将信息补全
            elif is_inbound_span(span) is True:
                span_flag.remove(span['spanID'])
                # 补充trace信息
                trace_dict = add_inbound_trace_information(span, trace_dict)
                trace_dict['svc'].append(svc)
                trace_dict['call'].append(('OTHER_SYSTEM', svc))
                trace_dict['call_instance'].append(
                    (handle_node_id_istio('OTHER_NODE'), handle_node_id_istio(node_id)))
                # 补充half_trace信息
                half_trace_dict = add_inbound_trace_information(span, half_trace_dict)
                half_trace_dict['svc'].append(svc)
                half_trace_dict['call'].append(('OTHER_SYSTEM', svc))
                half_trace_dict['call_instance'].append(
                    (handle_node_id_istio('OTHER_NODE'), handle_node_id_istio(node_id)))

        # 根据标志位判断是否加入个trace信息中
        # 判断trace_dict是否正常
        if is_span_abnormal(trace_dict) or is_abnormal:
            abnormal_dicts[trace_json['traceID']] = trace_dict
        else:
            normal_dicts[trace_json['traceID']] = trace_dict
        # 判断half_trace_dict是否正常
        if is_span_abnormal(half_trace_dict) and (is_inbound is True or is_outbound is True):
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
    获得空的trace_dict
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
    获得该span的父span
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
                    outbound_span = tmp_span
    return outbound_span


'''
    判断该span是否是出边
'''
def is_outbound_span(span_json):
    for tag in span_json['tags']:
        if 'upstream_cluster' == tag['key']:
            if 'outbound' in tag['value']:
                return True

    return False


'''
    判断该span是否是入边
'''
def is_inbound_span(span_json):
    for tag in span_json['tags']:
        if 'upstream_cluster' == tag['key']:
            if 'inbound' in tag['value']:
                return True

    return False


'''
    获得span的基本信息
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
    向trace_dict添加信息
'''
def add_trace_information(span_json, trace_dict):
    trace_dict['timestamp'].append(span_json['startTime'] + clock_skew_adjust(span_json))
    trace_dict['latency'].append(span_json['duration'])
    # 根据状态码判断是否正常
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            trace_dict['http_status'].append(int(tag['value']))

    return trace_dict


'''
    处理node_id
'''
def handle_node_id_istio(node_id: str) -> str:
    if node_id == 'OTHER_NODE':
        return 'OTHER_NODE'
    s = node_id.split('~')
    if len(s) != 4:
        raise Exception('istio trace node id illegal')
    return s[2] + '-' + s[1]


'''
    处理pod_id
'''
def handle_istio_pod_id(node_id: str) -> str:
    if node_id is None:
        return 'None'
    s = node_id.split('~')
    if len(s) != 4:
        raise Exception('istio trace node id illegal')
    return s[2].split('.')[0]


'''
    向出边为系统内部的half_trace添加信息
'''
def add_outbound_trace_information(outbound_span, half_trace_dict):
    # 添加出边的trace信息
    half_trace_dict['timestamp'].append(outbound_span['startTime'] + clock_skew_adjust(outbound_span))
    half_trace_dict['latency'].append(outbound_span['duration'])
    for tag in outbound_span['tags']:
        if tag['key'] == 'http.status_code':
            half_trace_dict['http_status'].append(int(tag['value']))
    # 添加入边的trace信息
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)

    return half_trace_dict


'''
    向入边为系统内部的half_trace添加信息
'''
def add_inbound_trace_information(span_json, half_trace_dict):
    # 添加出边的trace信息
    half_trace_dict['timestamp'].append(-1)
    half_trace_dict['latency'].append(-1)
    half_trace_dict['http_status'].append(-1)
    # 添加出边的trace信息
    half_trace_dict['timestamp'].append(span_json['startTime'] + clock_skew_adjust(span_json))
    half_trace_dict['latency'].append(span_json['duration'])
    for tag in span_json['tags']:
        if tag['key'] == 'http.status_code':
            half_trace_dict['http_status'].append(int(tag['value']))

    return half_trace_dict


'''
    判断span是否异常
'''
def is_span_abnormal(trace_dict):
    # 调用关系为空，则设定为不正常
    if trace_dict['call'] == []:
        return True
    for code in trace_dict['http_status']:
        if (code < 200 and code != -1) or code >= 300:
            return True
    return False


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
