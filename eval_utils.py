"""Evaluation utilities for the agent RCA pipeline: label parsing, ACC@K / AVG@N
ranking metrics, and a stdout/log tee shared by the batch runners."""
import os
import re
import glob

TOP_K_CUTOFFS = tuple(range(1, 11))


class _Tee:
    """Write batch output to both the terminal and the experiment log."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(getattr(stream, 'isatty', lambda: False)()
                   for stream in self.streams)


def parse_labels(abn_dir):
    """Parse '=== <sample> ===' blocks from *_label.txt into
    {sample_name: {service, fault_type, root_cause, ...}}."""
    labels = {}
    for lf in glob.glob(os.path.join(abn_dir, '*_label.txt')):
        text = open(lf, encoding='utf-8').read()
        blocks = re.split(r'^===\s*(.+?)\s*===\s*$', text, flags=re.M)
        # blocks = ['', name1, body1, name2, body2, ...]
        for i in range(1, len(blocks), 2):
            name = blocks[i].strip()
            body = blocks[i + 1]
            kv = {}
            for line in body.splitlines():
                m = re.match(r'\s*([a-z_]+):\s*(.+?)\s*$', line)
                if m:
                    kv[m.group(1)] = m.group(2).strip()
            labels[name] = kv
    return labels


def _topk_metrics(rows, rank_key, cutoffs=TOP_K_CUTOFFS):
    """Calculate ACC@K and AVG@N; missing/zero ranks count as misses."""
    total = len(rows)
    if total == 0:
        return ({k: 0.0 for k in cutoffs},
                {n: 0.0 for n in cutoffs})

    ranks = []
    for row in rows:
        try:
            ranks.append(int(row.get(rank_key) or 0))
        except (TypeError, ValueError):
            ranks.append(0)

    max_cutoff = max(cutoffs)
    acc_by_k = {
        k: sum(0 < rank <= k for rank in ranks) / total
        for k in range(1, max_cutoff + 1)
    }
    acc = {k: acc_by_k[k] for k in cutoffs}
    avg = {
        n: sum(acc_by_k[k] for k in range(1, n + 1)) / n
        for n in cutoffs
    }
    return acc, avg
