"""ABLATION MicroARCL-A: remove the AGENT-AWARE anomaly detection module."""
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
    main(method='relrrf', align=False, tag='abirch_ablation_A_no_agent_aware_',
         detector='birch', variant='both', k_rrf=60,
         include_residual=False, k=5, mu=1.0, fuse='wsum')
