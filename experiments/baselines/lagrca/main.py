from ast import arg
import sys
import os
import time


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import pandas as pd
import argparse

from config import Config

from module.DataProcessor import DataProcessor,TimeWindowDataset

from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch.utils.data import DataLoader
from module.NodeDecoder import NodeDecoder

from model.LagRCA import LagRCA
from tqdm import tqdm
import torch
import torch.nn.functional as F
import pickle
from module.RootCauseScorer import RootCauseScorer



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('-ds','--dataset', type=str, default="D1")
    parser.add_argument('--eval_only', action='store_true',
                        help='skip training, load saved checkpoint and only evaluate')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='checkpoint path (default: <data_path>/model.pt)')
    return parser.parse_args()

def _service_base(instance_name):
    """Map an instance name to the service it belongs to (exact, not prefix).

    Strips a trailing replica/pod suffix so a service's own pods count as that
    service, while a *different* service (e.g. image-gen vs image) does not.
      agent-network-image-pod0   -> agent-network-image
      agent-network-image-gen    -> agent-network-image-gen   (NOT image)
      emailservice-0             -> emailservice
      node-15                    -> node   (never equals a service label)
    """
    return re.sub(r'(-pod\d+|-\d+)$', '', instance_name)


def evaluate_topk_accuracy(all_ans, all_labels, topk_list=[1, 3, 5]):
    results = {k: 0 for k in topk_list}
    total = len(all_ans)

    for pred_list, label in zip(all_ans, all_labels):
        # 确保 label 是列表格式
        label = [label] if isinstance(label, str) else label
        label_set = set(label)
        for k in topk_list:
            topk_pred = pred_list[:k]
            # exact service-level match (no prefix collision)
            hit = any(_service_base(p) in label_set for p in topk_pred)
            if hit:
                results[k] += 1

    return {k: results[k] / total for k in topk_list}



if __name__ == "__main__":
 
    args = parse_args()
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    

    base_dir = os.path.dirname(os.path.abspath(__file__))
    
   
    args.data_path = os.path.join(base_dir, 'data', args.dataset)
    
    
    config = Config(args.dataset)
    config.batch_size = args.batch_size
    
    data_path = os.path.join(args.data_path, 'normal_data.pkl')
    adj_path = os.path.join(args.data_path, 'adj.pkl')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(adj_path, 'rb') as f:
        adj = pickle.load(f)

    # Data Processing
    data_processor = DataProcessor(data, config, window_size=args.window_size, stride=args.stride)
    config.instance_metric_count_dict = data_processor.instance_metric_count_dict
    
    train_dataset = TimeWindowDataset(data_processor, config)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)
    
    first_batch = next(iter(train_loader))
    x_window = first_batch[0]
    y_window = first_batch[2]
    metric_num = y_window.shape[1]
    input_dim = x_window.shape[-1]
 
    model = LagRCA(config, device=device, adj=adj, metric_num=metric_num, 
                input_dim=input_dim, hidden_dim=64, sca_hidden_dim=64).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    root_cause_scorer = RootCauseScorer(alpha=1.0, beta=0.01, config=config, device=device)

    ckpt_path = args.ckpt or os.path.join(args.data_path, 'model.pt')

    if args.eval_only:
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded checkpoint: {ckpt_path} (eval-only, skipping training)")
        train_time = 0.0
    else:
        print("\nStart Training...")
        train_t0 = time.perf_counter()
        model.train()
        for epoch in tqdm(range(args.epochs), desc="Epochs"):
            epoch_loss = 0.0
            batch_count = 0

            for x_window, instance_names, y_window in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                x_window = x_window.to(device)
                y_window = y_window.to(device)

                optimizer.zero_grad()
                loss = model(x_window, y_window)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

        train_time = time.perf_counter() - train_t0
        print(f"[TRAIN TIME] total {train_time:.1f}s | mean {train_time / max(args.epochs, 1):.2f}s/epoch "
              f"({args.epochs} epochs)")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

    model.eval()
    
    case_path = os.path.join(args.data_path, 'case_data.pkl')
    if not os.path.exists(case_path):
        print(f"Test case file not found: {case_path}")
        sys.exit(1)
    else:
        with open(case_path, 'rb') as f:
            test_case = pickle.load(f)

        all_ans = []
        all_labels = []
        all_ts = []
        all_times = []
        results_data = []

        print("\nStart Testing...")

        for case_id, (case_data, label, ts) in enumerate(tqdm(test_case)):
            all_labels.append(label)
            all_ts.append(ts)
            case_t0 = time.perf_counter()

            if case_data.shape[0] < args.window_size:

                all_ans.append([])
                all_times.append(time.perf_counter() - case_t0)
                continue

            case_processor = DataProcessor(case_data, config, window_size=args.window_size, stride=args.stride)
            test_dataset = TimeWindowDataset(case_processor, config)
            test_loader_case = DataLoader(test_dataset, batch_size=1, shuffle=False)
            
            all_score = None
            for x_window, _, y_window in test_loader_case:
                x_window = x_window.to(device)
                y_window = y_window[0].to(device)
                
                with torch.no_grad():
                    anomaly_score = model.calculate_metric_loss(x_window, y_window)
                
                if all_score is None:
                    all_score = anomaly_score
                else:
                    all_score += anomaly_score
            
            if len(test_loader_case) > 0:
                all_score = all_score / len(test_loader_case)
                all_score = all_score.squeeze(-1).cpu()

                A_list = model.causal_learner.get_last_graph()
                
                A_list = A_list.squeeze(0)
                ans, _, _ = root_cause_scorer.get_root_cause_by_upstream_adjustment(
                    all_score, A_list, label, topk=len(config.all_enum), beta=0.5)
            else:
                ans = []
            all_ans.append(ans)
            all_times.append(time.perf_counter() - case_t0)

        # per-case localization (inference) time
        n_timed = len(all_times)
        total_infer = sum(all_times)
        mean_infer = total_infer / n_timed if n_timed else 0.0
        print(f"\n[INFER TIME] total {total_infer:.1f}s | mean {mean_infer:.3f}s/case "
              f"({n_timed} cases)")
        # per-group inference time
        grp_t = {}
        for ts, tt in zip(all_ts, all_times):
            grp_t.setdefault(str(ts).split('/')[0], []).append(tt)
        for g in ('single', 'multi'):
            if grp_t.get(g):
                v = grp_t[g]
                print(f"    <{g}> total {sum(v):.1f}s | mean {sum(v)/len(v):.3f}s/case (n={len(v)})")

        # save full ranked predictions for offline candidate-pool ablation (no retrain)
        with open(os.path.join(args.data_path, 'eval_results.pkl'), 'wb') as f:
            pickle.dump({'all_ans': all_ans, 'all_labels': all_labels,
                         'all_ts': all_ts, 'all_times': all_times,
                         'train_time': train_time}, f)

        # Evaluate
        accuracy = evaluate_topk_accuracy(all_ans, all_labels, topk_list=[1, 5])
        print("\nEvaluation Results:")
        print(f"Top-1 Accuracy: {accuracy[1]:.2%}")
        print(f"Top-5 Accuracy: {accuracy[5]:.2%}")

        # metrics table aligned with MicroARCL baseline (ACC@K / AVG@N / MRR / n)
        import agent_metrics
        agent_metrics.print_report(all_ans, all_labels, all_ts, tag=args.dataset)

        # uniform output tree: output/<dataset>/LagRCA/ (shared results_log)
        try:
            _REPO = os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__)))))
            if _REPO not in sys.path:
                sys.path.insert(0, _REPO)
            import results_log
            _DSMAP = {'agent': ('MDOC', 'agent'), 'marble': ('MAR', 'agent'),
                      're2ob': ('OB', 're2'), 're2ss': ('SS', 're2'),
                      're2tt': ('TT', 're2')}
            ds_name, kind = _DSMAP.get(args.dataset, (args.dataset, 'agent'))
            rows = agent_metrics.build_rows(all_ans, all_labels, all_ts)
            for r, order in zip(rows, all_ans):
                r['top1'] = order[0] if order else None
                if kind == 're2':
                    r['fault'] = r.get('load') or ''
            if kind == 'agent':
                cats = tuple(sorted({r['category'] for r in rows
                                     if r.get('category') and r['category'] != 'other'}))
            else:
                cats = tuple(sorted({r['fault'] for r in rows if r.get('fault')}))
            results_log.emit_from_rows(ds_name, 'LagRCA', f'LagRCA_{args.dataset}',
                                       rows, kind=kind, categories=cats)
        except Exception as _e:
            print(f'[results_log] uniform-output hook skipped: {_e}')

        # timing summary: training and inference kept separate (total + mean)
        print("\n" + "=" * 50 + "\nTiming summary")
        if not args.eval_only:
            print(f"  Training : total {train_time:.1f}s | mean {train_time / max(args.epochs, 1):.2f}s/epoch "
                  f"({args.epochs} epochs)")
        print(f"  Inference: total {total_infer:.1f}s | mean {mean_infer:.3f}s/case "
              f"({n_timed} cases)")
