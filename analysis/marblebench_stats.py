#!/usr/bin/env python3
"""Count observable records in a local MARBLEBench export.

Usage: python analysis/marblebench_stats.py [DATASET_DIR] [--output JSON_FILE]

The six record categories use these concrete files:
  request_execution_graphs: graph/graph_json/*.json
  service_executions: entries in each JSON's level_details[].level_spans
  llm_messages: messages in those spans
  llm_tokens: each graph JSON's top-level token (aggregate, not span sum)
  metric_records: data rows in numeric agent-network/metrics/*.csv and node/*.csv
  log_records: lines in agent-network/log/*.log

Execution spans include both agent and group nodes; span_types reports the split.
Topology graph.csv, historical metrics.pre_recollect files, and task.log
orchestration output are excluded.
"""

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def count_lines(path):
    count = 0
    last = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            count += chunk.count(b"\n")
            last = chunk[-1:]
    return count + (last not in (b"", b"\n"))


def count_csv_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="", errors="replace") as stream:
        rows = csv.reader(stream)
        next(rows, None)
        return sum(1 for _ in rows)


def parse_labels(root):
    windows = []
    for path in root.rglob("*_label.txt"):
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in text.split("=== ")[1:]:
            fields = {}
            for line in block.splitlines()[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    fields[key.strip()] = value.strip()
            if "window_range" in fields:
                start, end = map(int, fields["window_range"].split()[:2])
                windows.append((start, end, fields.get("fault_injection")))
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", type=Path, default=Path("data/MARBLEBench"))
    parser.add_argument("--output", type=Path, help="Save the result as JSON")
    args = parser.parse_args()
    root = args.dataset.resolve()
    if not root.is_dir():
        parser.error(f"Dataset directory does not exist: {root}")

    samples = [p.parent for p in root.rglob("task.log") if p.parent.parent.parent == root]
    # The sample layout is root/{normal,abnormal}/{load}/{sample}/task.log.
    if not samples:
        samples = [p.parent for p in root.rglob("task.log")]
    sample_counts = Counter(p.relative_to(root).parts[0] for p in samples)
    service_names = set()
    metric_records = 0
    metric_by_file = Counter()
    log_records = 0
    graph_count = 0
    span_count = 0
    span_types = Counter()
    message_count = 0
    token_count = 0
    graph_by_split = Counter()

    for sample in samples:
        split = sample.relative_to(root).parts[0]
        metrics_dir = sample / "agent-network" / "metrics"
        for path in metrics_dir.glob("*.csv"):
            if path.name == "graph.csv":
                continue
            rows = count_csv_rows(path)
            metric_records += rows
            metric_by_file[path.name] += rows
            if path.name == "svc_metric.csv":
                with path.open("r", encoding="utf-8-sig", newline="") as stream:
                    service_names.update(
                        col.split("&", 1)[0] for col in next(csv.reader(stream), [])
                        if col != "timestamp" and "&" in col
                    )
        for path in (sample / "node").glob("*.csv"):
            rows = count_csv_rows(path)
            metric_records += rows
            metric_by_file[path.name] += rows
        for path in (sample / "agent-network" / "log").glob("*.log"):
            log_records += count_lines(path)
        for path in (sample / "agent-network" / "graph" / "graph_json").glob("*.json"):
            with path.open("r", encoding="utf-8") as stream:
                graph = json.load(stream)
            graph_count += 1
            graph_by_split[split] += 1
            token_count += int(graph.get("token") or 0)
            for level in graph.get("level_details", []):
                for span in level.get("level_spans", {}).values():
                    span_count += 1
                    span_types[span.get("type", "unknown")] += 1
                    message_count += len(span.get("messages") or [])

    windows = parse_labels(root)
    start = min((a for a, _, _ in windows), default=None)
    end = max((b for _, b, _ in windows), default=None)
    result = {
        "dataset": str(root),
        "samples": dict(sample_counts),
        "observed_microservices": len(service_names),
        "service_names": sorted(service_names),
        "agent_services": len(service_names),
        "failures": sample_counts.get("abnormal", 0),
        "time_span": {
            "first_window_start": datetime.fromtimestamp(start).isoformat(sep=" ") if start else None,
            "last_window_end": datetime.fromtimestamp(end).isoformat(sep=" ") if end else None,
            "wall_clock_hours": round((end - start) / 3600, 3) if start else None,
            "sum_sample_window_hours": round(sum(b - a for a, b, _ in windows) / 3600, 3),
            "abnormal_window_hours": round(sum(b - a for a, b, flag in windows if flag == "true") / 3600, 3),
            "labeled_windows": len(windows),
        },
        "records": {
            "request_execution_graphs": graph_count,
            "service_executions": span_count,
            "llm_messages": message_count,
            "llm_tokens": token_count,
            "metric_records": metric_records,
            "log_records": log_records,
        },
        "span_types": dict(span_types),
        "graphs_by_split": dict(graph_by_split),
        "metric_rows_by_file": dict(sorted(metric_by_file.items())),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
