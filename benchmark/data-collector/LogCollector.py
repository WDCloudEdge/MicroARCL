#!/usr/bin/env python3

import argparse
import os
import re
import time

from kubernetes import client, config

from datetime import datetime, timezone


def timestamp_to_rfc3339(timestamp):
    """
    Unix timestamp -> Kubernetes RFC3339 UTC时间
    """
    dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def collect_logs(
        namespace,
        start_timestamp,
        end_timestamp,
        output_dir,
):
    config.load_kube_config()

    v1 = client.CoreV1Api()

    start_rfc3339 = timestamp_to_rfc3339(start_timestamp)

    pods = v1.list_namespaced_pod(
        namespace=namespace
    ).items

    print(f"Namespace: {namespace}")
    print(f"Start: {start_timestamp}")
    print(f"End:   {end_timestamp}")
    print(f"Pods:  {len(pods)}")

    os.makedirs(output_dir, exist_ok=True)

    for pod in pods:

        pod_name = pod.metadata.name

        containers = [
            c.name
            for c in pod.spec.containers
        ]

        for container_name in containers:

            if container_name == "istio-proxy":
                continue

            output_file = os.path.join(
                output_dir,
                f"{pod_name}_{container_name}.log"
            )

            print(
                f"Collecting: "
                f"{pod_name}/{container_name}"
            )

            try:
                since_seconds = max(
                    1,
                    int(time.time() - start_timestamp)
                )

                logs = v1.read_namespaced_pod_log(
                    name=pod_name,
                    namespace=namespace,
                    container=container_name,
                    timestamps=True,
                    since_seconds=since_seconds,
                    _preload_content=False,
                )

                with open(output_file, "w", encoding="utf-8") as f:

                    for line in logs:
                        if isinstance(line, bytes):
                            line = line.decode("utf-8", errors="replace")

                        line = line.rstrip("\n")

                        if not line:
                            continue

                        try:
                            # Kubernetes timestamps=True 后：
                            #
                            # 2026-09-04T06:20:31.123456789Z log content
                            #
                            timestamp_str = line.split(" ", 1)[0]

                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1] + "+00:00"

                            # 截断纳秒到微秒
                            if "." in timestamp_str:
                                main_part, fraction = timestamp_str.split(".", 1)

                                # fraction 可能是：
                                # 012475323+00:00
                                fraction = fraction[:6]

                                timestamp_str = f"{main_part}.{fraction}+00:00"

                            log_dt = datetime.fromisoformat(timestamp_str)
                            log_timestamp = log_dt.timestamp()

                            # 超过结束时间
                            if log_timestamp > end_timestamp:
                                break

                            # 理论上不会出现，因为 since_seconds
                            # 已经限制了开始时间，但再检查一次更安全
                            if log_timestamp < start_timestamp:
                                continue

                        except (ValueError, IndexError):
                            # 无法解析 timestamp
                            continue

                        f.write(line + "\n")

            except client.exceptions.ApiException as e:
                print(
                    f"Failed: "
                    f"{pod_name}/{container_name}: "
                    f"{e}"
                )


def safe_filename(name):
    """
    防止 Pod/Container 名称中出现特殊字符
    """
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def main():
    parser = argparse.ArgumentParser(
        description="收集 Kubernetes namespace 下所有 Pod 指定时间范围内的日志"
    )

    parser.add_argument(
        "--namespace",
        "-n",
        required=True,
        help="Kubernetes namespace"
    )

    parser.add_argument(
        "--start",
        required=True,
        help="开始时间，例如: 2026-09-04 10:00:00"
    )

    parser.add_argument(
        "--end",
        required=True,
        help="结束时间，例如: 2026-09-04 11:00:00"
    )

    parser.add_argument(
        "--output",
        "-o",
        default="./k8s_logs",
        help="日志输出目录，默认 ./k8s_logs"
    )

    parser.add_argument(
        "--previous",
        action="store_true",
        help="同时收集 previous container 日志"
    )

    args = parser.parse_args()

    start_time = parse_time(args.start)
    end_time = parse_time(args.end)

    if end_time <= start_time:
        raise ValueError(
            "结束时间必须大于开始时间"
        )

    collect_logs(
        namespace=args.namespace,
        start_time=start_time,
        end_time=end_time,
        output_dir=args.output,
        include_previous=args.previous,
    )


def collect_logs_(start_time, end_time, namespace, output_dir):
    collect_logs(
        namespace=namespace,
        start_timestamp=start_time,
        end_timestamp=end_time,
        output_dir=output_dir,
        # include_previous=args.previous,
    )


if __name__ == "__main__":
    main()
