import sys
import re
import argparse
import shutil

import pkl_concat
from Config import Config
import MetricCollector
import LogCollector
from handler import Trace, Log
import time
from util.utils import *
import os

if __name__ == "__main__":
    # namespaces = ['bookinfo', 'hipster', 'hipster2', 'cloud-sock-shop', 'horsecoder-test'， 'openclaw']
    namespaces = ['agent-network']
    config = Config()


    class Simple:
        def __init__(self, label, begin, end):
            self.label = label
            self.begin = begin
            self.end = end


    def read_label_logs(label_file, simple_list: [Simple]):
        if simple_list is None:
            simple_list = []
        file_path = label_file
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
                # 如果文件为空则跳过
                if not lines:
                    return simple_list
                for line in lines:
                    if 'cpu_' in line or 'mem_' in line or 'net_' in line:
                        root_cause = line.strip()
                        simple = Simple(root_cause, None, None)
                    elif 'start create' in line:
                        begin = line[:18]
                    elif 'finish delete' in line:
                        end = line[:18]
                        simple.begin = timestamp_2_time_string(time_string_2_timestamp(begin) - 360)
                        simple.end = timestamp_2_time_string(time_string_2_timestamp(end) + 360)
                        simple_list.append(simple)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    def parse_label_file(label_file):
        """解析故障 label 文件，返回每组故障的信息列表。

        文件中每个 `=== xxx ===` 块对应一组服务故障根因，一个文件可包含多组。
        每组包含完整时间窗口 window_start ~ window_end 以及 root_cause 等字段；
        带括号的时间字段会额外解析出 `<字段>_ts` 的 epoch 时间戳。
        """
        groups = []
        current = None
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"读取 label 文件失败 {label_file}: {e}")
            return groups
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('===') and line.endswith('==='):
                if current is not None:
                    groups.append(current)
                current = {'name': line.strip('= ').strip()}
            elif current is not None and ':' in line:
                key, _, val = line.partition(':')
                key, val = key.strip(), val.strip()
                current[key] = val
                m = re.search(r'\((\d+)\)', val)
                if m:
                    current[key + '_ts'] = int(m.group(1))
        if current is not None:
            groups.append(current)
        # window_range: <start> <end> 优先作为完整时间窗口
        for g in groups:
            parts = g.get('window_range', '').split()
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                g['window_start_ts'] = int(parts[0])
                g['window_end_ts'] = int(parts[1])
        return groups

    def user_dir_for(label_file, group_name):
        """根据 label 文件所在目录 + 故障组名，生成与 label 对应的输出文件夹(config.user)。"""
        data_root = os.path.abspath('./data')
        label_dir = os.path.dirname(os.path.abspath(label_file))
        rel = os.path.relpath(label_dir, data_root)
        if rel == '.':
            return group_name
        return os.path.join(rel, group_name)

    def collect_for_config(config, namespaces):
        """按 config 当前的 user / start / end 收集完整时间窗口内的所有指标。"""
        for n in namespaces:
            config.namespace = n
            config.svcs.clear()
            config.pods.clear()
            data_folder = './data/' + str(config.user) + '/' + config.namespace
            print('获取 [' + config.namespace + '] 数据')
            MetricCollector.collect(config, os.path.join(data_folder, 'metrics'), True)
            Trace.collect(config, os.path.join(data_folder, 'trace'))
            LogCollector.collect_logs_(config.start, config.end, n, os.path.join(data_folder, 'log'))
            # Log.collect(config, os.path.join(data_folder, 'log'))
            config.pods.clear()
            # 将trace数据合并，写入到原文件中
            # pkl_concat.data_concat(data_folder, data_folder)
            # 收集调用链路（graph.json -> 每条 trace 的有序链路），存到与 metrics/trace/log 同级的 graph 子目录
            try:
                import GraphCollector
                GraphCollector.collect_graph(config, os.path.join(data_folder, 'graph'))
            except Exception as e:
                print(f"调用链路收集失败: {e}")
        # node 指标
        node_folder = './data/' + str(config.user) + '/node'
        print('获取 [node] 数据')
        MetricCollector.collect_node(config, node_folder, True)

    def merge_prefault_logs(config, namespaces, tmp_subdir):
        """把"注入前日志快照"里、最终收集时已消失的 Pod 日志并入 log/。

        pod_kill/pod_failure 会重启 Pod。注入前的快照(tmp_subdir)与最终 log/
        按文件名(<pod>_<container>.log)比较：
          - 快照文件在 log/ 里已有同名 → 该 Pod 仍在(如 pod_failure 原地重启，
            最终日志更完整)，跳过；
          - 快照文件在 log/ 里没有同名 → 该 Pod 已消失(如 pod_kill 换了名字，
            重启前日志无法再取)，保留其快照并加 _log_prefault 后缀存入 log/。
        合并后删除临时快照目录，最终只保留一个 log/ 目录。
        """
        for n in namespaces:
            data_folder = './data/' + str(config.user) + '/' + n
            log_dir = os.path.join(data_folder, 'log')
            tmp_dir = os.path.join(data_folder, tmp_subdir)
            if not os.path.isdir(tmp_dir):
                continue
            os.makedirs(log_dir, exist_ok=True)
            existing = set(os.listdir(log_dir))
            kept = 0
            for fn in os.listdir(tmp_dir):
                if not fn.endswith('.log'):
                    continue
                if fn in existing:
                    continue  # Pod 仍在，最终日志已覆盖，丢弃快照
                dst = os.path.join(log_dir, f"{fn[:-4]}_log_prefault.log")
                shutil.move(os.path.join(tmp_dir, fn), dst)
                kept += 1
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if kept:
                print(f"[{n}] {kept} 个已消失 Pod 的重启前日志已并入 log/ (后缀 _log_prefault)")

    # ------------------------------------------------------------------
    # 命令行单窗口模式：供 chaos_service.sh 在销毁 deployment 之前调用，
    # 直接指定 user 输出目录以及完整时间窗口 [start, end]，收集这一个窗口。
    #   python Main.py --user <dir> --start <epoch> --end <epoch>
    # 不带参数时回退到下面写死的 label_files 批量收集逻辑。
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='收集指定时间窗口内的所有指标数据')
    parser.add_argument('--user', help='输出目录(config.user)，相对 ./data')
    parser.add_argument('--start', type=int, help='窗口开始时间(epoch 秒)')
    parser.add_argument('--end', type=int, help='窗口结束时间(epoch 秒)')
    parser.add_argument('--logs-only', action='store_true',
                        help='只收集日志(不收集 metrics/trace/graph/node)；'
                             '用于 pod_failure/pod_kill 在 Pod 重启前抢救日志')
    parser.add_argument('--log-subdir', default='log',
                        help='日志输出子目录名，默认 log；抢救快照建议用临时目录名')
    parser.add_argument('--merge-prefault', metavar='TMP_SUBDIR',
                        help='最终收集完成后，把该临时快照目录里已消失 Pod 的日志'
                             '并入 log/(加 _log_prefault 后缀)，并删除临时目录')
    cli_args, _ = parser.parse_known_args()

    if cli_args.user and cli_args.start and cli_args.end:
        config.user = cli_args.user
        config.start = cli_args.start
        config.end = cli_args.end
        config.duration = config.end - config.start

        if cli_args.logs_only:
            print(f"==> 单窗口仅收集日志 [{config.start} ~ {config.end}] "
                  f"-> data/{config.user}/<ns>/{cli_args.log_subdir}")
            for n in namespaces:
                config.namespace = n
                data_folder = './data/' + str(config.user) + '/' + n
                LogCollector.collect_logs_(
                    config.start, config.end, n,
                    os.path.join(data_folder, cli_args.log_subdir))
            sys.exit(0)

        print(f"==> 单窗口收集 [{config.start} ~ {config.end}] "
              f"({config.duration}s) -> data/{config.user}")
        collect_for_config(config, namespaces)
        if cli_args.merge_prefault:
            merge_prefault_logs(config, namespaces, cli_args.merge_prefault)
        sys.exit(0)

    # 需要收集的 label 文件列表；每个文件可包含多组故障(=== xxx ===)，
    # 每组对应一个服务故障根因，收集其完整时间窗口(window_start~window_end)的所有指标。
    label_files = [
        'data/MDOC/abnormal/load-5/agent-network-pdf-parsing_label.txt',
    ]

    for label_file in label_files:
        groups = parse_label_file(label_file)
        print(f'{label_file} 解析到 {len(groups)} 组故障')
        for g in groups:
            if 'window_start_ts' not in g or 'window_end_ts' not in g:
                print(f"跳过 {g.get('name')}: 缺少完整时间窗口(window_start/window_end)")
                continue
            config.user = user_dir_for(label_file, g['name'])
            config.start = g['window_start_ts']
            config.end = g['window_end_ts']
            config.duration = config.end - config.start
            print(f"==> 收集 {g['name']}  root_cause={g.get('root_cause')}  "
                  f"[{config.start} ~ {config.end}] ({config.duration}s) -> data/{config.user}")
            collect_for_config(config, namespaces)
