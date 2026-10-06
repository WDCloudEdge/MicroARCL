"""ABLATION MicroARCL-L: remove the request-level invocation-chain LAG correction.
Main line = abirch gate + reliability-weighted wsum of telemetry voters
- mu*chain_lag (k=15, mu=1, no GNN). This variant sets mu=0.0: no chain-lag
temporal-accumulation penalty. Everything else identical.
"""
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
    main(method='relrrf', align=True, tag='abirch_ablation_L_no_lag_',
         detector='abirch', variant='both', k_rrf=60,
         include_residual=False, k=5, mu=0.0, fuse='wsum')
