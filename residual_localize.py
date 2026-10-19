"""Normal-state topology residual localizers + severity top-k rerank."""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from graph_utils import build_svc_call_edges, _read_services
from lag_align import align_metrics_df

# metric channels with a pre-window normal segment (svc_metric); severity-like
# component weights. Network (instance) was tested and REJECTED: busy/central
# services have high network residual regardless of fault, which pollutes the
# rerank and demotes the true root cause. Latency/call have no normal segment.
CHAN = ['&cpu_usage', '&mem_usage']
COMP_W = [0.5, 0.5]


def _neighbors(ns_dir):
    services = _read_services(os.path.join(ns_dir, 'metrics'))
    edges = build_svc_call_edges(ns_dir, services)
    nb = {}
    for a, b in edges:
        nb.setdefault(a, set()).add(b)
        nb.setdefault(b, set()).add(a)
    return nb


def rerank_within_topk(sev, resid_norm, k, lam, lag_pen=None, mu=0.0):
    """Rank all services by severity, then reorder ONLY the severity top-k by
    (severity + lam*residual - mu*chain_lag_penalty). Candidate set unchanged ->
    ACC@>=k cannot drop. lag_pen[s] in [0,1] is the normalized chain-propagation
    depth (0=entry/upstream, 1=deepest downstream); penalizing it demotes
    downstream victims and lifts upstream root causes."""
    lag_pen = lag_pen or {}
    order = [s for s, _ in sorted(sev.items(), key=lambda kv: kv[1],
                                  reverse=True)]
    head = sorted(order[:k],
                  key=lambda s: (sev[s] + lam * resid_norm.get(s, 0.0)
                                 - mu * lag_pen.get(s, 0.0)),
                  reverse=True)
    return head + order[k:]


def _minmax(d):
    """Min-max normalize a dict's values to [0,1] (constant -> all 0)."""
    if not d:
        return {}
    lo, hi = min(d.values()), max(d.values())
    if hi - lo < 1e-12:
        return {k: 0.0 for k in d}
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def _metric_evidence(sev, resid_norm, cands, mode='rrf', k0=60):
    """Parameter-free fusion of the two METRIC-based signals -- severity and
    normal-state residual -- into a single [0,1] metric-evidence score over the
    candidates. Both measure metric-level abnormality, so they are merged
    without a weight (killing lam); only the temporal chain-lag stays separate.
      mode='rrf' : reciprocal-rank fusion of the two rankings (scale-free),
      mode='avg' : mean of the two min-max normalized scores."""
    if mode == 'avg':
        sn = _minmax({s: float(sev.get(s, 0.0)) for s in cands})
        rn = _minmax({s: float(resid_norm.get(s, 0.0)) for s in cands})
        me = {s: 0.5 * sn[s] + 0.5 * rn[s] for s in cands}
    else:                                            # rrf
        rs = {s: i + 1 for i, s in enumerate(
            sorted(cands, key=lambda s: sev.get(s, 0.0), reverse=True))}
        rr = {s: i + 1 for i, s in enumerate(
            sorted(cands, key=lambda s: resid_norm.get(s, 0.0), reverse=True))}
        me = {s: 1.0 / (k0 + rs[s]) + 1.0 / (k0 + rr[s]) for s in cands}
    return _minmax(me)                               # comparable with lag_pen


def rerank_merged(sev, resid_norm, k, mu, lag_pen=None, mode='rrf'):
    """One-parameter rerank: merge severity+residual into a parameter-free
    metric-evidence term, then apply only the chain-lag penalty:
        score(s) = metric_evidence(sev, residual) - mu * chain_lag_penalty(s)
    Candidate set = severity top-k (unchanged, ACC@>=k preserved)."""
    lag_pen = lag_pen or {}
    order = [s for s, _ in sorted(sev.items(), key=lambda kv: kv[1],
                                  reverse=True)]
    cands = order[:k]
    me = _metric_evidence(sev, resid_norm, cands, mode=mode)
    head = sorted(cands, key=lambda s: me[s] - mu * lag_pen.get(s, 0.0),
                  reverse=True)
    return head + order[k:]


def _load_svc_metric(ns_dir, lag):
    p = os.path.join(ns_dir, 'metrics', 'svc_metric.csv')
    if not os.path.exists(p):
        return None
    df = align_metrics_df(pd.read_csv(p), lag) if lag else pd.read_csv(p)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


# ---------------------- fixed neighbor-mean residual ----------------------
def _channel_residual_fixed(df, suffix, services, nb, start, end,
                            include_self=False):
    cols = {c.split('&')[0]: c for c in df.columns
            if c.endswith(suffix) and c.split('&')[0] in services}
    normal = df['timestamp'] < start
    abn = (df['timestamp'] >= start) & (df['timestamp'] <= end)
    if normal.sum() < 3 or abn.sum() < 1:
        return {}
    dev = {}
    for svc, col in cols.items():
        x = pd.to_numeric(df[col], errors='coerce')
        base = x[normal].median()
        if not np.isfinite(base) or abs(base) < 1e-9:
            base = max(abs(x[normal].mean()), 1e-6)
        dev[svc] = (x - base) / (abs(base) + 1e-9)
    out = {}
    for svc in cols:
        neigh = [n for n in nb.get(svc, ()) if n in dev]
        if include_self:                       # include the node itself in pred
            neigh = neigh + [svc]
        pred = (pd.concat([dev[n] for n in neigh], axis=1).mean(axis=1)
                if neigh else pd.Series(0.0, index=df.index))
        gap = dev[svc] - pred
        out[svc] = float(gap[abn].mean() - gap[normal].mean())
    return out


def fixed_residual(ns_dir, sev, start, end, lag, include_self=False):
    df = _load_svc_metric(ns_dir, lag)
    nb = _neighbors(ns_dir)
    resid = {s: 0.0 for s in sev}
    if df is None:
        return resid
    for suf in CHAN:
        for s, r in _channel_residual_fixed(
                df, suf, set(sev), nb, start, end, include_self).items():
            resid[s] = resid.get(s, 0.0) + max(0.0, r)
    return resid


class NormalGNN(nn.Module):
    def __init__(self, adj_mask, C, use_W=False, prior_bias=None):
        super().__init__()
        self.mask = adj_mask
        self.raw = nn.Parameter(torch.zeros_like(adj_mask))
        self.bias = prior_bias if prior_bias is not None \
            else torch.zeros_like(adj_mask)
        self.W = nn.Linear(C, C, bias=False) if use_W else None

    def alpha(self):
        neg = torch.full_like(self.raw, -1e9)
        scores = torch.where(self.mask > 0, self.raw + self.bias, neg)
        a = torch.softmax(scores, dim=1)
        return a * (self.mask.sum(1, keepdim=True) > 0)

    def forward(self, X):
        Wx = self.W(X) if self.W is not None else X
        return torch.einsum('sj,tjc->tsc', self.alpha(), Wx)


def _build_tensor(ns_dir, svcs, start, end, lag):
    df = _load_svc_metric(ns_dir, lag)
    if df is None:
        return None
    ts = df['timestamp'].reset_index(drop=True)
    dfx = df.set_index('timestamp')
    idx = {s: i for i, s in enumerate(svcs)}
    N, T, C = len(svcs), len(ts), len(CHAN)
    normal = (ts < start).values
    abn = ((ts >= start) & (ts <= end)).values
    if normal.sum() < 3 or abn.sum() < 1:
        return None
    X = np.zeros((T, N, C), dtype=np.float32)
    for ci, suf in enumerate(CHAN):
        for s in svcs:
            col = f'{s}{suf}'
            if col not in dfx.columns:
                continue
            ser = pd.to_numeric(dfx[col], errors='coerce').reindex(ts).values
            base = np.nanmedian(ser[normal])
            if not np.isfinite(base) or abs(base) < 1e-9:
                base = max(abs(np.nanmean(ser[normal])), 1e-6)
            X[:, idx[s], ci] = np.nan_to_num((ser - base) / (abs(base) + 1e-9))
    return (torch.tensor(X), torch.tensor(normal), torch.tensor(abn), idx)


def gnn_residual(ns_dir, sev, start, end, lag, use_W=False, epochs=200,
                 seed=0, include_self=False):
    svcs = list(sev.keys())
    packed = _build_tensor(ns_dir, svcs, start, end, lag)
    if packed is None:
        return {s: 0.0 for s in svcs}
    torch.manual_seed(seed)
    X, normal, abn, idx = packed
    N, C = X.shape[1], X.shape[2]
    edges = build_svc_call_edges(ns_dir, _read_services(
        os.path.join(ns_dir, 'metrics')))
    A = torch.zeros(N, N)
    for a, b in edges:
        if a in idx and b in idx:
            A[idx[a], idx[b]] = 1
            A[idx[b], idx[a]] = 1
    if include_self:
        A += torch.eye(N)
    model = NormalGNN(A, C, use_W=use_W)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xn = X[normal]
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(Xn) - Xn) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gap = X - model(X)
        resid = gap[abn].mean(0) - gap[normal].mean(0)
        resid = torch.clamp(resid, min=0.0)
        comp = torch.tensor(COMP_W[:C])
        Rvec = (resid * comp).sum(1)
    return {s: float(Rvec[idx[s]]) for s in svcs}


def _chain_depth(ns_dir, cfg, sev):
    from lag_align import compute_chain_lags
    cl = compute_chain_lags(os.path.join(ns_dir, 'metrics'), ns_dir, cfg)
    return {s: float(cl.get((s, 'cpu'), 0)) for s in sev}


def _lead_lag(xi, xj, tau_max, min_corr):
    n = min(len(xi), len(xj))
    xi, xj = xi[:n], xj[:n]
    best_c, best_s = 0.0, 0
    for s in range(-tau_max, tau_max + 1):
        a, b = (xi[:n - s], xj[s:]) if s >= 0 else (xi[-s:], xj[:n + s])
        if len(a) < 5 or a.std() < 1e-9 or b.std() < 1e-9:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if np.isfinite(c) and abs(c) > abs(best_c):
            best_c, best_s = c, s
    if abs(best_c) < min_corr or best_s == 0:
        return 0, 0.0
    return (1 if best_s > 0 else -1), abs(best_c)


def _lead_lag_sign(xi, xj, tau_max, min_corr):
    return _lead_lag(xi, xj, tau_max, min_corr)[0]


def _call_freq(ns_dir):
    import json
    from graph_utils import _read_services
    from execution_graph import group_to_service
    p = os.path.join(ns_dir, 'graph', 'call_chains.json')
    if not os.path.exists(p):
        return {}
    with open(p, 'r', encoding='utf-8') as f:
        chains = json.load(f)
    services = _read_services(os.path.join(ns_dir, 'metrics'))
    cache = {}

    def to_svc(v):
        if v not in cache:
            cache[v] = group_to_service(v, services)
        return cache[v]

    freq = {}
    for tr in chains:
        seq = []
        for v in (tr.get('path', []) or []):
            s = to_svc(v)
            if s is None:
                continue
            if not seq or seq[-1] != s:
                seq.append(s)
        for a, b in zip(seq, seq[1:]):
            if a != b:
                key = frozenset((a, b))
                freq[key] = freq.get(key, 0) + 1
    return freq


def mrgnn_residual(ns_dir, sev, start, end, lag, cfg, epochs=200, seed=0,
                   standardize=False):
    svcs = list(sev.keys())
    packed = _build_tensor(ns_dir, svcs, start, end, lag)
    if packed is None:
        return {s: 0.0 for s in svcs}
    torch.manual_seed(seed)
    X, normal, abn, idx = packed
    N, C = X.shape[1], X.shape[2]
    depth = _chain_depth(ns_dir, cfg, sev)
    edges = build_svc_call_edges(ns_dir, _read_services(
        os.path.join(ns_dir, 'metrics')))
    freq = _call_freq(ns_dir)
    fmax = float(max(freq.values())) if freq else 1.0
    Xnp = X.numpy()
    series = {s: Xnp[:, idx[s], :].sum(axis=1) for s in svcs}
    tau_max = int(getattr(cfg, 'lag_tau_max', 12))
    min_corr = float(getattr(cfg, 'lag_min_corr', 0.3))
    A = torch.zeros(N, N)
    bias = torch.zeros(N, N)
    for a, b in edges:
        if a not in idx or b not in idx:
            continue
        sign, conf = _lead_lag(series[a], series[b], tau_max, min_corr)
        if sign > 0:
            u, d = a, b
        elif sign < 0:
            u, d = b, a
        else:
            da, db = depth.get(a, 0.0), depth.get(b, 0.0)
            u, d = (a, b) if da <= db else (b, a)
        A[idx[d], idx[u]] = 1
        fnorm = freq.get(frozenset((a, b)), 1) / fmax
        bias[idx[d], idx[u]] = float(np.log(fnorm + 1e-3)) + conf
    model = NormalGNN(A, C, use_W=False, prior_bias=bias)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xn = X[normal]
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(Xn) - Xn) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gap = X - model(X)
        if standardize:
            sd = gap[normal].std(0) + 1e-6
            resid = (gap[abn].mean(0) - gap[normal].mean(0)) / sd
        else:
            resid = gap[abn].mean(0) - gap[normal].mean(0)
        resid = torch.clamp(resid, min=0.0)
        Rvec = (resid * torch.tensor(COMP_W[:C])).sum(1)
    return {s: float(Rvec[idx[s]]) for s in svcs}


def lag_gnn_residual(ns_dir, sev, start, end, lag, cfg, epochs=200, seed=0):
    svcs = list(sev.keys())
    packed = _build_tensor(ns_dir, svcs, start, end, lag)
    if packed is None:
        return {s: 0.0 for s in svcs}
    torch.manual_seed(seed)
    X, normal, abn, idx = packed
    N, C = X.shape[1], X.shape[2]
    depth = _chain_depth(ns_dir, cfg, sev)
    edges = build_svc_call_edges(ns_dir, _read_services(
        os.path.join(ns_dir, 'metrics')))
    Xnp = X.numpy()
    series = {s: Xnp[:, idx[s], :].sum(axis=1) for s in svcs}
    tau_max = int(getattr(cfg, 'lag_tau_max', 12))
    min_corr = float(getattr(cfg, 'lag_min_corr', 0.3))
    A = torch.zeros(N, N)
    for a, b in edges:
        if a not in idx or b not in idx:
            continue
        sign = _lead_lag_sign(series[a], series[b], tau_max, min_corr)
        if sign > 0:
            u, d = a, b
        elif sign < 0:
            u, d = b, a
        else:
            da, db = depth.get(a, 0.0), depth.get(b, 0.0)
            u, d = (a, b) if da <= db else (b, a)
        A[idx[d], idx[u]] = 1
    model = NormalGNN(A, C, use_W=False)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    Xn = X[normal]
    for _ in range(epochs):
        opt.zero_grad()
        loss = ((model(Xn) - Xn) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gap = X - model(X)
        resid = gap[abn].mean(0) - gap[normal].mean(0)
        resid = torch.clamp(resid, min=0.0)
        Rvec = (resid * torch.tensor(COMP_W[:C])).sum(1)
    return {s: float(Rvec[idx[s]]) for s in svcs}


def _dev_from(df, ts, normal, cols_map, idx, C):
    dfx = df.set_index('timestamp')
    X = np.zeros((len(ts), len(idx), C), dtype=np.float32)
    for ent, cols in cols_map.items():
        for ci, col in enumerate(cols[:C]):
            if col is None or col not in dfx.columns:
                continue
            ser = pd.to_numeric(dfx[col], errors='coerce').reindex(ts).values
            base = np.nanmedian(ser[normal])
            if not np.isfinite(base) or abs(base) < 1e-9:
                base = max(abs(np.nanmean(ser[normal])), 1e-6)
            X[:, idx[ent], ci] = np.nan_to_num((ser - base) / (abs(base) + 1e-9))
    return torch.tensor(X)


class HeteroGNN(nn.Module):
    def __init__(self, adj_svc, C):
        super().__init__()
        self.mask = adj_svc
        self.raw = nn.Parameter(torch.zeros_like(adj_svc))
        self.Wsvc = nn.Linear(C, C, bias=False)
        self.Wnode = nn.Linear(C, C, bias=False)
        self.beta = nn.Parameter(torch.zeros(adj_svc.shape[0]))

    def forward(self, Xsvc, Xnode_pad, node_idx):
        neg = torch.full_like(self.raw, -1e9)
        a = torch.softmax(torch.where(self.mask > 0, self.raw, neg), dim=1)
        a = a * (self.mask.sum(1, keepdim=True) > 0)
        svc_msg = torch.einsum('sj,tjc->tsc', a, self.Wsvc(Xsvc))
        node_feat = self.Wnode(Xnode_pad)[:, node_idx, :]
        beta = torch.sigmoid(self.beta).view(1, -1, 1)
        return svc_msg + beta * node_feat


def hgnn_residual(ns_dir, sev, start, end, lag, epochs=200, seed=0):
    import pandas as _pd
    from lag_align import _service_of_pod
    sample_dir = os.path.dirname(ns_dir)
    md = os.path.join(ns_dir, 'metrics')
    smdf = _load_svc_metric(ns_dir, lag)
    if smdf is None:
        return {s: 0.0 for s in sev}
    ts = smdf['timestamp'].reset_index(drop=True)
    normal = (ts < start).values
    abn = ((ts >= start) & (ts <= end)).values
    if normal.sum() < 3 or abn.sum() < 1:
        return {s: 0.0 for s in sev}
    svcs = list(sev.keys())
    sidx = {s: i for i, s in enumerate(svcs)}
    C = len(CHAN)
    Xsvc = _dev_from(smdf, ts, normal,
                     {s: [f'{s}{suf}' for suf in CHAN] for s in svcs}, sidx, C)
    npath = os.path.join(sample_dir, 'node', 'node.csv')
    if not os.path.exists(npath):
        return {s: 0.0 for s in sev}
    ndf = _pd.read_csv(npath)
    ndf['timestamp'] = _pd.to_datetime(ndf['timestamp'])
    nodecols = {}
    for c in ndf.columns:
        if c.startswith('(node)'):
            nm, ch = c[len('(node)'):].rsplit('_', 1)
            if ch in ('cpu', 'memory'):
                nodecols.setdefault(nm, {})[ch] = c
    nodes = sorted(nodecols)
    nidx = {n: i for i, n in enumerate(nodes)}
    Xnode = _dev_from(ndf, ts, normal,
                      {n: [nodecols[n].get('cpu'), nodecols[n].get('memory')]
                       for n in nodes}, nidx, C)
    svc2node = {}
    gdf = _pd.read_csv(os.path.join(md, 'graph.csv'))
    for _, row in gdf.iterrows():
        src, dst = str(row['source']), str(row['destination'])
        if dst.startswith('node-'):
            sv = _service_of_pod(src)
            if sv in sidx and sv not in svc2node and dst in nidx:
                svc2node[sv] = nidx[dst]
    Nn = len(nodes)
    node_idx = torch.full((len(svcs),), Nn, dtype=torch.long)
    for s, i in sidx.items():
        if s in svc2node:
            node_idx[i] = svc2node[s]
    Xnode_pad = torch.cat(
        [Xnode, torch.zeros(Xnode.shape[0], 1, Xnode.shape[2])], dim=1)
    edges = build_svc_call_edges(ns_dir, _read_services(md))
    A = torch.zeros(len(svcs), len(svcs))
    for a, b in edges:
        if a in sidx and b in sidx:
            A[sidx[a], sidx[b]] = 1
            A[sidx[b], sidx[a]] = 1
    torch.manual_seed(seed)
    model = HeteroGNN(A, C)
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    nmask = torch.tensor(normal)
    for _ in range(epochs):
        opt.zero_grad()
        pred = model(Xsvc[nmask], Xnode_pad[nmask], node_idx)
        loss = ((pred - Xsvc[nmask]) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        gap = Xsvc - model(Xsvc, Xnode_pad, node_idx)
        amask = torch.tensor(abn)
        resid = gap[amask].mean(0) - gap[nmask].mean(0)
        resid = torch.clamp(resid, min=0.0)
        Rvec = (resid * torch.tensor(COMP_W[:C])).sum(1)
    return {s: float(Rvec[sidx[s]]) for s in svcs}


# ---------------------------- unified entry ----------------------------
def chain_lag_penalty(ns_dir, sev, cfg, tau_max=12):
    """Normalized chain-propagation depth per service (0=upstream/entry,
    1=deepest downstream) from compute_chain_lags. Used to demote downstream
    victims in the rerank."""
    from lag_align import compute_chain_lags
    cl = compute_chain_lags(os.path.join(ns_dir, 'metrics'), ns_dir, cfg)
    return {s: cl.get((s, 'cpu'), 0) / float(tau_max) for s in sev}


def residual_order(ns_dir, sev, start, end, lag, kind='fix', k=3, lam=2.0,
                   include_self=False, mu=0.0, cfg=None, merge=None):
    """Service order after residual + optional chain-lag upstream penalty,
    reranked within severity top-k. mu>0 enables the chain-lag penalty.

    merge in {'rrf','avg'} collapses the two metric signals (severity+residual)
    into a parameter-free metric-evidence term and keeps ONLY the chain-lag
    penalty: score = metric_evidence - mu*chain_lag. This drops lam, leaving mu
    as the single tuned parameter."""
    if kind in ('laggnn', 'mrgnn', 'mrgnnz'):
        if kind == 'laggnn':
            resid = lag_gnn_residual(ns_dir, sev, start, end, lag, cfg)
        else:
            resid = mrgnn_residual(ns_dir, sev, start, end, lag, cfg,
                                   standardize=(kind == 'mrgnnz'))
        mx = max((abs(v) for v in resid.values()), default=1.0) or 1.0
        rn = {s: resid.get(s, 0.0) / mx for s in sev}
        order = [s for s, _ in sorted(sev.items(), key=lambda kv: kv[1],
                                      reverse=True)]
        head = sorted(order[:k], key=lambda s: rn.get(s, 0.0), reverse=True)
        return head + order[k:]
    if kind == 'fix':
        resid = fixed_residual(ns_dir, sev, start, end, lag, include_self)
    elif kind == 'hgnn':
        resid = hgnn_residual(ns_dir, sev, start, end, lag)
    else:
        resid = gnn_residual(ns_dir, sev, start, end, lag,
                             use_W=(kind == 'gnnW'), include_self=include_self)
    mx = max((abs(v) for v in resid.values()), default=1.0) or 1.0
    rn = {s: resid.get(s, 0.0) / mx for s in sev}
    lag_pen = chain_lag_penalty(ns_dir, sev, cfg) if mu > 0 and cfg else None
    if merge:
        return rerank_merged(sev, rn, k, mu, lag_pen=lag_pen, mode=merge)
    return rerank_within_topk(sev, rn, k, lam, lag_pen=lag_pen, mu=mu)
