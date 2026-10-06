
import json
import time
import os
import pymysql
import requests
from collections import defaultdict

from collections import Counter

TASK_STATUS_MAP = {
    0: "新建",
    1: "运行中",
    2: "成功",
    3: "失败",
    4: "取消",
    5: "暂停",
    6: "人为协助",
    7: "完成",
    8: "完成失败"
}

# ========== 参数配置 ==========
DB_CONFIG = {
    'host': '192.168.31.15',
    'port': 3306,
    'user': 'root',
    'password': '',
    'database': 'taskScheduling',
    'charset': 'utf8mb4'
}
FILE_SERVER_BASE_URL = "http://192.168.31.15:12104/api/sys/storage/file/download"  # 根据你的文件服务器实际地址替换
# SAVE_DIR = "./task_files_20260524_prod"  # 下载文件保存的根目录
# CHECKPOINT_FILE = os.path.join(SAVE_DIR, "checkpoint.json")
REPORT_INTERVAL = 1000

DOWNLOAD_COUNT_MAP = {}

# 需要记录的数据
task_total_time = [0]  # 任务消耗总览
task_total_token = [0]
agent_subs_nums = [0]  # 智能体子任务和rpa子任务的总数和各状态占比
tbot_subs_nums = [0]
agent_subs_map = Counter()
tbot_subs_map = Counter()
task_trace_map = Counter()  # 每种任务的链路长度
task_time_map = Counter()  # 每种任务的耗时
task_token_map = Counter()  # 每种任务的token消耗
agent_vertex_time = [0]  # 智能体和rpa vertex的消耗总览
rpa_vertex_time = [0]
agent_vertex_token = [0]
rpa_vertex_token = [0]
agent_vertex_nums = [0]
rpa_vertex_nums = [0]
success_agent_vertex_map = Counter()  # 每种vertex的成功与失败计数
fail_agent_vertex_map = Counter()
success_rpa_vertex_map = Counter()  # 每种vertex的成功与失败计数
fail_rpa_vertex_map = Counter()
contribution_agent_counter = Counter()  # 智能体和rpa vertex的贡献统计
contribution_rpa_counter = Counter()
contribution_agent_map = Counter()
contribution_rpa_map = Counter()
success_task_map = [2, 7]  # 成功任务的状态码
result_location = ['description', 'title']  # 结果字段的位置

COUNTER_STATE_NAMES = [
    "agent_subs_map",
    "tbot_subs_map",
    "task_trace_map",
    "task_time_map",
    "task_token_map",
    "success_agent_vertex_map",
    "fail_agent_vertex_map",
    "success_rpa_vertex_map",
    "fail_rpa_vertex_map",
    "contribution_agent_counter",
    "contribution_rpa_counter",
    "contribution_agent_map",
    "contribution_rpa_map",
]

LIST_STATE_NAMES = [
    "task_total_time",
    "task_total_token",
    "agent_subs_nums",
    "tbot_subs_nums",
    "agent_vertex_time",
    "rpa_vertex_time",
    "agent_vertex_token",
    "rpa_vertex_token",
    "agent_vertex_nums",
    "rpa_vertex_nums",
]


def record_vertex(vertex, success):
    #  记录vertex是否执行成功
    if success:
        if 'Group' in vertex:
            success_agent_vertex_map[vertex] += 1
        else:
            success_rpa_vertex_map[vertex] += 1
    else:
        if 'Group' in vertex:
            fail_agent_vertex_map[vertex] += 1
        else:
            fail_rpa_vertex_map[vertex] += 1


def add_vertex(vertex, cost_time, token):
    if cost_time > 10000:
        cost_time /= 1000
    #  记录vertex的耗时和token消耗
    if 'Group' in vertex:
        agent_vertex_nums[0] += 1
        agent_vertex_time[0] += min(3600, cost_time)
        agent_vertex_token[0] += token
    else:
        rpa_vertex_nums[0] += 1
        rpa_vertex_time[0] += min(3600, cost_time)
        rpa_vertex_token[0] += token


def similarity_with_length(current, target):
    if current is None or target is None:
        return 0.0
    current = str(current).replace(" ", "")
    target = str(target).replace(" ", "")
    cnt_current = Counter(current)
    cnt_target = Counter(target)
    intersection = cnt_current & cnt_target
    union = cnt_current | cnt_target
    if not union:
        return 0.0
    return sum(intersection.values()) / sum(union.values())


def similarity_ignore_length(current, target):
    if current is None or target is None:
        return 0.0
    current = str(current).replace(" ", "")
    target = str(target).replace(" ", "")
    cnt_target = Counter(target)
    all_count = sum(cnt_target.values())
    if all_count == 0:
        return 0.0
    for c in current:
        if c in cnt_target:
            cnt_target[c] -= 1
            if cnt_target[c] == 0:
                del cnt_target[c]
    return 1 - sum(cnt_target.values()) / all_count


def add_contribution(front_contribution, final_output, group, span):
    cur_value = get_value(span)
    cur_contribution = 0
    cur_contribution += similarity_with_length(cur_value, final_output) * 0.8
    cur_contribution += similarity_ignore_length(cur_value, final_output) * 0.2
    if 'Group' in group:
        contribution_agent_counter[group] += 1
        contribution_agent_map[group] += max(cur_contribution - front_contribution, 0)
    else:
        contribution_rpa_counter[group] += 1
        contribution_rpa_map[group] += max(cur_contribution - front_contribution, 0)
    return max(front_contribution, cur_contribution)


def get_value(span):
    res = None
    if 'results' in span:
        for result in span['results']:
            # if 'description' in result and '结果' in result['description']:
            #     if 'value' in result:
            #         return result['value']
            # if 'name' in result and ('result' in result['name'] or 'flag' in result['name']):
            #     if 'value' in result:
            #         return result['value']
            # if 'title' in result and ('result' in result['title'] or 'flag' in result['title']):
            #     if 'value' in result:
            #         return result['value']
            if 'value' in result:
                res = result['value']
    return res


def my_merge(my_map):
    if 7 in my_map:
        my_map[2] += my_map[7]
        my_map.pop(7)
    if 8 in my_map:
        my_map[3] += my_map[8]
        my_map.pop(8)
    return my_map


def overview(tasks):
    status_counter = Counter(task['status'] for task in tasks)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    overview_path = os.path.join(SAVE_DIR, "overview.txt")

    with open(overview_path, "w", encoding="utf-8") as f:
        f.write(f"任务总数: {len(tasks)}\n\n")
        f.write("各状态任务数量:\n")
        for status_code in sorted(TASK_STATUS_MAP):
            count = status_counter.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count}\n")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return set()

    with open(CHECKPOINT_FILE, encoding="utf-8") as f:
        checkpoint = json.load(f)

    for name in LIST_STATE_NAMES:
        if name in checkpoint:
            globals()[name][0] = checkpoint[name]

    int_key_counters = {
        "agent_subs_map",
        "tbot_subs_map",
        "task_trace_map",
        "task_time_map",
        "task_token_map",
    }
    for name in COUNTER_STATE_NAMES:
        if name not in checkpoint:
            continue
        key_parser = int if name in int_key_counters else str
        globals()[name].clear()
        globals()[name].update({key_parser(k): v for k, v in checkpoint[name].items()})

    processed_task_ids = set(str(task_id) for task_id in checkpoint.get("processed_task_ids", []))
    print(f"检测到断点文件，已完成任务数: {len(processed_task_ids)}")
    return processed_task_ids


def save_checkpoint(processed_task_ids):
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    checkpoint = {
        "processed_task_ids": sorted(processed_task_ids),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    for name in LIST_STATE_NAMES:
        checkpoint[name] = globals()[name][0]
    for name in COUNTER_STATE_NAMES:
        checkpoint[name] = dict(globals()[name])

    tmp_path = CHECKPOINT_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, CHECKPOINT_FILE)


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0
    return numerator / denominator


def write_analyze_reports(tasks_num, task_map, processed_count):
    analyze_dir = os.path.join(SAVE_DIR, "analyze")
    os.makedirs(analyze_dir, exist_ok=True)

    with open(os.path.join(analyze_dir, "任务消耗总览.txt"), "w", encoding="utf-8") as f:
        f.write(f"已处理任务数: {processed_count} / {tasks_num}\n")
        f.write(f"任务平均消耗时间: {task_total_time[0]} / {tasks_num} = {safe_ratio(task_total_time[0], tasks_num)}\n")
        f.write(f"任务平均消耗token: {task_total_token[0]} / {tasks_num} = {safe_ratio(task_total_token[0], tasks_num)}\n")
    with open(os.path.join(analyze_dir, "子任务分析.txt"), "w", encoding="utf-8") as f:
        f.write(f"agent子任务总数: {agent_subs_nums[0]}\n")
        f.write("agent子任务各状态占比:\n")
        for status_code in sorted(agent_subs_map):
            count = agent_subs_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {agent_subs_nums[0]}"
                    f" = {safe_ratio(count, agent_subs_nums[0])}\n")
        f.write(f"\ntbot子任务总数: {tbot_subs_nums[0]}\n")
        f.write("tbot子任务各状态占比:\n")
        for status_code in sorted(tbot_subs_map):
            count = tbot_subs_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {tbot_subs_nums[0]}"
                    f" = {safe_ratio(count, tbot_subs_nums[0])}\n")
    with open(os.path.join(analyze_dir, "任务细节分析.txt"), "w", encoding="utf-8") as f:
        f.write("各任务平均链路长度:\n")
        for status_code in sorted(task_trace_map):
            count = task_trace_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {len(task_map[status_code])}"
                    f" = {safe_ratio(count, len(task_map[status_code]))}\n")
        f.write("各任务平均耗时:\n")
        for status_code in sorted(task_time_map):
            count = task_time_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {len(task_map[status_code])}"
                    f" = {safe_ratio(count, len(task_map[status_code]))}\n")
        f.write("各任务平均token消耗:\n")
        for status_code in sorted(task_token_map):
            count = task_token_map.get(status_code, 0)
            f.write(f"  [{status_code}] {TASK_STATUS_MAP[status_code]}: {count} / {len(task_map[status_code])} ="
                    f" {safe_ratio(count, len(task_map[status_code]))}\n")
    with open(os.path.join(analyze_dir, "vertex总览.txt"), "w", encoding="utf-8") as f:
        f.write(f"总vertex数量: {agent_vertex_nums[0] + rpa_vertex_nums[0]}\n")
        f.write(f"agent vertex数量: {agent_vertex_nums[0]}\n")
        f.write(f"agent vertex消耗时间: {agent_vertex_time[0]}\n")
        f.write(f"agent vertex消耗token: {agent_vertex_token[0]}\n")
        f.write(f"tbot vertex数量: {rpa_vertex_nums[0]}\n")
        f.write(f"tbot vertex消耗时间: {rpa_vertex_time[0]}\n")
        f.write(f"tbot vertex消耗token: {rpa_vertex_token[0]}\n")
    with open(os.path.join(analyze_dir, "vertex数量及成功率.txt"), "w", encoding="utf-8") as f:
        f.write("agentVertexes: \n")
        for vertex in success_agent_vertex_map:
            count = success_agent_vertex_map.get(vertex, 0)
            total = count + fail_agent_vertex_map.get(vertex, 0)
            f.write(f"  [{vertex}]: {total}\n")
            f.write(f"  成功率: {safe_ratio(count, total)}\n")
        f.write(f"tbotVertexes: \n")
        for vertex in success_rpa_vertex_map:
            count = success_rpa_vertex_map.get(vertex, 0)
            total = count + fail_rpa_vertex_map.get(vertex, 0)
            f.write(f"  [{vertex}]: {total}\n")
            f.write(f"  成功率: {safe_ratio(count, total)}\n")
        f.write("\n成功agent vertex占比: ")
        f.write(f"{safe_ratio(sum(success_agent_vertex_map.values()), sum(success_agent_vertex_map.values()) + sum(fail_agent_vertex_map.values()))}\n")
        f.write("成功tbot vertex占比: ")
        f.write(f"{safe_ratio(sum(success_rpa_vertex_map.values()), sum(success_rpa_vertex_map.values()) + sum(fail_rpa_vertex_map.values()))}\n")
        f.write("成功vertex占比: ")
        success_vertex_count = sum(success_agent_vertex_map.values()) + sum(success_rpa_vertex_map.values())
        vertex_count = success_vertex_count + sum(fail_agent_vertex_map.values()) + sum(fail_rpa_vertex_map.values())
        f.write(f"{safe_ratio(success_vertex_count, vertex_count)}\n")
    with open(os.path.join(analyze_dir, "vertex贡献.txt"), "w", encoding="utf-8") as f:
        f.write("agent vertex: \n")
        for vertex in contribution_agent_counter:
            count = contribution_agent_counter.get(vertex, 0)
            f.write(f"  [{vertex}]: {count}\n")
            f.write(f"  平均贡献: {safe_ratio(contribution_agent_map.get(vertex, 0), count)}\n")
        f.write(f"\ntbot vertex: \n")
        for vertex in contribution_rpa_counter:
            count = contribution_rpa_counter.get(vertex, 0)
            f.write(f"  [{vertex}]: {count}\n")
            f.write(f"  平均贡献: {safe_ratio(contribution_rpa_map.get(vertex, 0), count)}\n")


def print_progress(processed_count, tasks_num):
    print(
        f"已处理 {processed_count} / {tasks_num} 个任务，"
        f"agent子任务: {agent_subs_nums[0]}，tbot子任务: {tbot_subs_nums[0]}，"
        f"总vertex: {agent_vertex_nums[0] + rpa_vertex_nums[0]}，"
        f"任务总耗时: {task_total_time[0]}，任务总token: {task_total_token[0]}"
    )


# ========== 数据库查询 ==========
def fetch_data():
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    # 查询主任务
    cursor.execute("SELECT * FROM sys_task")
    tasks = cursor.fetchall()

    task_map = defaultdict(list)
    graph_ids = set()

    for task in tasks:
        status = task['status']
        task_id = task['id']
        task_map[status].append(task)
        graph_ids.add(task['graph_id'])

    # 查询子任务（agent 和 tbot）
    cursor.execute("SELECT * FROM sys_agent_subtask")
    agent_subtasks = cursor.fetchall()

    cursor.execute("SELECT * FROM sys_tbot_subtask")
    tbot_subtasks = cursor.fetchall()

    # 查询 graph file_path
    cursor.execute(f"SELECT * FROM sys_graph WHERE id IN ({','.join(['%s'] * len(graph_ids))})", list(graph_ids))
    graph_rows = cursor.fetchall()
    graph_path_map = {g['id']: g['file_path'] for g in graph_rows}

    cursor.close()
    conn.close()

    return tasks, task_map, agent_subtasks, tbot_subtasks, graph_path_map


# ========== 文件下载 ==========
def download_file(file_path, save_path, status):
    download_count = DOWNLOAD_COUNT_MAP.get(file_path, 0)
    if download_count >= 3:
        return
    DOWNLOAD_COUNT_MAP[file_path] = download_count + 1
    url = f"{FILE_SERVER_BASE_URL}?user=agent&fileName={file_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        if not os.path.exists(save_path + ".json"):
            response = requests.get(url)
            if response.status_code == 200:
                with open(save_path + '.json', 'wb') as f:
                    f.write(response.content)
                print(f"下载成功: {save_path}")
            else:
                print(f"下载失败: {url}, 状态码: {response.status_code}")
        with open(save_path + '.json', encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            if 'time' in load_dict:
                task_total_time[0] += load_dict['time']
                task_time_map[status] += load_dict['time']
            if 'token' in load_dict:
                task_total_token[0] += int(load_dict['token'])
                task_token_map[status] += int(load_dict['token'])
            if 'total_level' in load_dict:
                task_trace_map[status] += load_dict['total_level']
            if 'level_details' in load_dict:
                final_output = None
                front_contribution = 0
                if status in task_trace_map:
                    final_output = " "
                    for level in load_dict['level_details']:
                        if level['level'] >= len(load_dict['level_details']) and 'level_spans' in level:
                            for span in level['level_spans']:
                                tmp_value = get_value(level['level_spans'][span])
                                if tmp_value is not None:
                                    final_output = tmp_value
                                    break
                for level in load_dict['level_details']:
                    route_num = 0
                    if 'level_routes' in level:
                        for route in level['level_routes']:
                            route_num += 1
                            if route_num >= 2:
                                if 'Group' in route and '/' not in route and level['level'] >= len(load_dict['level_details']) and level['level_routes'][route]:
                                    for next_group in level['level_routes'][route]:
                                        record_vertex(next_group.split('/')[0], False)
                                        add_vertex(next_group.split('/')[0], 0, 0)
                                continue
                            group = route.split('/')[0]
                            if final_output is not None and 'level_spans' in level:
                                for span in level['level_spans']:
                                    front_contribution = add_contribution(front_contribution, final_output,
                                                                          group, level['level_spans'][span])
                                    break
                            # 没有调度Group下具体服务的AgentGroup需要判断是否在最后一层，若在最后一层则记为fail，否则记为success
                            if 'Group' not in route or '/' in route:
                                # 对于AgentNetworkPlannerGroup，若在最后一层则记为fail，否则记为success
                                if group == 'AgentNetworkPlannerGroup':
                                    if level['level'] >= len(load_dict['level_details']):
                                        record_vertex(group, False)
                                    else:
                                        record_vertex(group, True)
                                    if 'level_spans' in level:
                                        for span in level['level_spans']:
                                            cost_time = level['level_spans'][span].get('time', 0)
                                            token = level['level_spans'][span].get('token', 0)
                                            add_vertex(group, cost_time, token)
                                    else:
                                        add_vertex(group, 0, 0)
                                elif 'level_spans' in level:
                                    for span in level['level_spans']:
                                        cost_time = level['level_spans'][span].get('time', 0)
                                        token = level['level_spans'][span].get('token', 0)
                                        add_vertex(group, cost_time, token)
                                        if 'status' in level['level_spans'][span]:
                                            record_vertex(group, level['level_spans'][span]['status'] in success_task_map)
                                        elif 'results' in level['level_spans'][span]:
                                            success = False
                                            has_success_message = False
                                            for result in level['level_spans'][span]['results']:
                                                location = result_location[0]
                                                if 'description' in result and result['description'] == '':
                                                    location = result_location[1]
                                                # TODO：去gitlab上看看，这个逻辑是否可行
                                                if location in result and '成功' in result[location] and '信息' not in result[location]:
                                                    has_success_message = True
                                                    special = False  # 部分服务的判断标准不太一样
                                                    if "".join(filter(str.isdigit, result[location])) != '':
                                                        special = True
                                                    if 'value' in result:
                                                        if special:
                                                            if "".join(filter(str.isdigit, result[location]))[0] in str(result['value']).lower():
                                                                success = True
                                                        elif 'true' in str(result['value']).lower() or '1' in str(result['value']) or '成功' in str(result['value']):
                                                            success = True
                                                    break
                                            if has_success_message:
                                                record_vertex(group, success)
                                            else:
                                                record_vertex(group, False)
                                        else:
                                            # 对于版本较老的任务，有'level_spans'就视为执行成功
                                            record_vertex(group, True)
                                else:
                                    record_vertex(group, False)
                                    add_vertex(group, 0, 0)
                            else:
                                if 'level_routes' in level:
                                    cost_time = 0
                                    token = 0
                                    if 'level_spans' in level:
                                        for span in level['level_spans']:
                                            cost_time = level['level_spans'][span].get('time', 0)
                                            token = level['level_spans'][span].get('token', 0)
                                    record_vertex(group, True)
                                    add_vertex(group, cost_time, token)
                                    if level['level'] >= len(load_dict['level_details']) and level['level_routes'][route]:
                                        # 出现这种情况说明存在未能调度的下一层服务，视为被调度的服务失败
                                        for next_group in level['level_routes'][route]:
                                            record_vertex(next_group.split('/')[0], False)
                                            add_vertex(next_group.split('/')[0], 0, 0)
                                else:
                                    record_vertex(group, False)
                                    add_vertex(group, 0, 0)
            else:
                print(load_dict)
    except Exception as e:
        print(f"下载或读取异常: {url}, 错误: {e}。正在尝试重新下载或读取...")
        time.sleep(0.5)
        download_file(file_path, save_path, status)

# ========== 调用链路收集 ==========
# 任务时间列的候选名（不同版本 schema 可能不同），用于按采集时间窗口过滤
TASK_TIME_KEYS = ['create_time', 'start_time', 'gmt_create', 'begin_time', 'update_time', 'end_time']


def _task_epoch(task):
    """尽力从任务记录中解析出一个 epoch 秒时间戳，无法识别返回 None。"""
    import datetime
    for k in TASK_TIME_KEYS:
        v = task.get(k)
        if v is None:
            continue
        if isinstance(v, datetime.datetime):
            return v.timestamp()
        if isinstance(v, (int, float)):
            # 毫秒级时间戳归一化到秒
            return float(v) / (1000.0 if v > 1e12 else 1.0)
        try:
            return time.mktime(time.strptime(str(v)[:19], "%Y-%m-%d %H:%M:%S"))
        except Exception:
            continue
    return None


def fetch_tasks_in_window(config):
    """从 MySQL 拉取任务及其 graph 文件路径，按 [config.start, config.end] 时间窗口过滤。

    返回 [(task_row, graph_file_path), ...]。若无法识别任务时间列则返回全部任务。
    """
    conn = pymysql.connect(**DB_CONFIG)
    cursor = conn.cursor(pymysql.cursors.DictCursor)

    cursor.execute("SELECT * FROM sys_task")
    tasks = cursor.fetchall()

    graph_ids = {t['graph_id'] for t in tasks if t.get('graph_id') is not None}
    graph_path_map = {}
    if graph_ids:
        cursor.execute(
            f"SELECT * FROM sys_graph WHERE id IN ({','.join(['%s'] * len(graph_ids))})",
            list(graph_ids),
        )
        graph_path_map = {g['id']: g['file_path'] for g in cursor.fetchall()}

    cursor.close()
    conn.close()

    start = getattr(config, 'start', None)
    end = getattr(config, 'end', None)
    has_time = any(_task_epoch(t) is not None for t in tasks)

    if has_time and start is not None and end is not None:
        selected = [t for t in tasks
                    if (_task_epoch(t) is not None and start <= _task_epoch(t) <= end)]
        print(f"按时间窗口 [{start}, {end}] 过滤到 {len(selected)} / {len(tasks)} 个任务")
    else:
        selected = tasks
        print(f"未识别到任务时间列或未设置窗口，处理全部 {len(selected)} 个任务")

    return [(t, graph_path_map.get(t.get('graph_id'))) for t in selected]


def fetch_graph_json(file_path, save_path, retries=3):
    """从文件服务器下载 graph.json（若本地已存在则复用），返回解析后的 dict，失败返回 None。"""
    json_path = save_path + '.json'
    url = f"{FILE_SERVER_BASE_URL}?user=agent&fileName={file_path}"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    for attempt in range(retries):
        try:
            if not os.path.exists(json_path):
                resp = requests.get(url, timeout=30)
                if resp.status_code != 200:
                    print(f"下载失败: {url}, 状态码: {resp.status_code}")
                    return None
                with open(json_path, 'wb') as f:
                    f.write(resp.content)
            with open(json_path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"下载或读取异常: {url}, 错误: {e} (重试 {attempt + 1}/{retries})")
            time.sleep(0.5)
    return None


def parse_call_chain(graph):
    """从 graph.json 的 level_details.level_routes 解析出一条 trace 的有序调用链路。

    level_routes 结构为 {调用方: {被调方: {...}, ...}, ...}，据此可还原每层的调用边；
    按 level 递增顺序即得到 入口(planner) -> 中间工具 -> 出口(summarizer) 的有序链路。
    """
    details = sorted(graph.get('level_details', []) or [], key=lambda d: d.get('level', 0))

    levels = []
    edges = []
    for lvl in details:
        level_no = lvl.get('level')
        routes = lvl.get('level_routes', {}) or {}
        vertexes = lvl.get('level_vertexes', []) or list(routes.keys())
        level_edges = []
        for caller, callees in routes.items():
            if not callees:
                continue
            for callee in callees.keys():
                level_edges.append({'source': caller, 'destination': callee})
                edges.append({'level': level_no, 'source': caller, 'destination': callee})
        levels.append({'level': level_no, 'vertexes': vertexes, 'edges': level_edges})

    # 线性有序路径：按层级顺序记录顶点首次出现（入口 -> ... -> 出口）
    seen = set()
    path = []
    for lvl in levels:
        for v in lvl['vertexes']:
            if v not in seen:
                seen.add(v)
                path.append(v)
    # 兜底：仅作为被调方出现、未列入 level_vertexes 的顶点补到链尾
    for e in edges:
        if e['destination'] not in seen:
            seen.add(e['destination'])
            path.append(e['destination'])

    return {
        'trace_id': graph.get('trace_id'),
        'total_level': graph.get('total_level'),
        'vertexes_count': graph.get('vertexes_count'),
        'participated_vertexes': graph.get('participated_vertexes'),
        'path': path,              # 有序调用链路（顶点序列）
        'chain_length': len(path),
        'levels': levels,          # 每一层的顶点与调用边
        'edges': edges,            # 展平的调用边（含 level）
    }


def collect_graph(config, out_dir):
    """收集时间窗口内所有任务的调用链路，逐条解析 graph.json 并写出有序链路。

    输出: <out_dir>/call_chains.json —— 每条 trace 的有序链路列表。
    原始 graph.json 缓存到 <out_dir>/graph_json/ 便于复用与排查。
    """
    os.makedirs(out_dir, exist_ok=True)
    raw_dir = os.path.join(out_dir, 'graph_json')
    os.makedirs(raw_dir, exist_ok=True)

    tasks = fetch_tasks_in_window(config)
    chains = []
    for task, graph_path in tasks:
        if not graph_path:
            continue
        save_path = os.path.join(raw_dir, str(task.get('id')))
        graph = fetch_graph_json(graph_path, save_path)
        if not graph or 'level_details' not in graph:
            continue
        chain = parse_call_chain(graph)
        chain['task_id'] = task.get('id')
        chain['task_status'] = task.get('status')
        chains.append(chain)

    out_file = os.path.join(out_dir, 'call_chains.json')
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(chains, f, ensure_ascii=False, indent=2)
    print(f"调用链路收集完成: {len(chains)} 条 -> {out_file}")
    return chains


# ========== 主逻辑 ==========
def main():
    tasks, task_map, agent_subtasks, tbot_subtasks, graph_path_map = fetch_data()

    overview(tasks)
    tasks_num = len(tasks)
    processed_task_ids = load_checkpoint()
    processed_count = len(processed_task_ids)
    print_progress(processed_count, tasks_num)
    over = False

    for status, tasks in task_map.items():
        if over:
            break
        status_dir = os.path.join(SAVE_DIR, f"status_{status}")
        for task in tasks:
            if over:
                break
            task_id = task['id']
            task_id_key = str(task_id)
            if task_id_key in processed_task_ids:
                continue

            task_dir = os.path.join(status_dir, task_id_key)
            os.makedirs(task_dir, exist_ok=True)

            # 保存 graph 文件
            graph_id = task['graph_id']
            if graph_id in graph_path_map:
                # file_path = graph_path_map[graph_id]
                file_path = task_id_key
                local_path = os.path.join(task_dir, os.path.basename(file_path))
                download_file(graph_path_map[graph_id], local_path, status)

            # 收集子任务
            # 收集并排序子任务
            agent_subs_sorted = sorted(
                filter(lambda s: s['task_id'] == task_id, agent_subtasks),
                key=lambda x: x['start_time']
            )
            tbot_subs_sorted = sorted(
                filter(lambda s: s['task_id'] == task_id, tbot_subtasks),
                key=lambda x: x['start_time']
            )

            subtask_file = os.path.join(task_dir, "subtasks.txt")
            with open(subtask_file, "w", encoding="utf-8") as f:
                f.write("Agent Subtasks:\n")
                for sub in agent_subs_sorted:
                    f.write(f"  [{sub['status']}] {sub['start_time']} {sub['name']} ({sub['type']})\n")
                    agent_subs_nums[0] += 1
                    agent_subs_map[sub['status']] += 1

                f.write("\nTBot Subtasks:\n")
                for sub in tbot_subs_sorted:
                    f.write(f"  [{sub['status']}] {sub['start_time']} {sub['name']} - {sub['flow_id']}\n")
                    tbot_subs_nums[0] += 1
                    tbot_subs_map[sub['status']] += 1

            processed_task_ids.add(task_id_key)
            processed_count += 1
            save_checkpoint(processed_task_ids)
            if processed_count % REPORT_INTERVAL == 0 or processed_count == tasks_num:
                write_analyze_reports(tasks_num, task_map, processed_count)
                print_progress(processed_count, tasks_num)

    # 保存记录数据
    write_analyze_reports(tasks_num, task_map, processed_count)
    save_checkpoint(processed_task_ids)

    print("数据处理完成。")


if __name__ == "__main__":
    main()
