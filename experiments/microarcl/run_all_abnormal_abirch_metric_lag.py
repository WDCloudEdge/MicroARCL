"""agent-aware Birch candidate gate + reliability-weighted SUM (wsum) of the TELEMETRY metric anomaly voters (failure/latency/cpu/mem/net/ qps) − mu*chain_lag."""
# --- path bootstrap: shared engine stays at repo root; add it + sibling runner dirs ---
import os as _os, sys as _sys
_r = _os.path.dirname(_os.path.abspath(__file__))
while _r != _os.path.dirname(_r) and not _os.path.exists(_os.path.join(_r, 'baseline_common.py')):
    _r = _os.path.dirname(_r)
for _p in (_r, _os.path.join(_r, 'experiments', 'microarcl'),
           _os.path.join(_r, 'experiments', 'baselines', 'microrca'),
           _os.path.join(_r, 'experiments', 'baselines', 'torai')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
# --- end path bootstrap ---
from baseline_common import main
if __name__ == '__main__':
    # for mu in (0.0, 0.5,):
    # for mu in (1.0,):
        # main(method='relrrf', align=True, tag=f'abirch_metric_lag_k_5_mu{mu}_',
            #  detector='abirch', variant='both', k_rrf=60,
            #  include_residual=False, k=5, mu=mu, fuse='wsum')
    for k in (5, 10, 15, 20, 25,):
    # for k in (5,):
        main(method='relrrf', align=True, tag=f'abirch_metric_lag_k{k}_mu0.0_',
             detector='abirch', variant='both', k_rrf=60,
             include_residual=False, k=k, mu=0.0, fuse='wsum')
