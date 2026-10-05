"""Vendored, patched causal-learn (0.1.2.3) used by the TORAI comparison
baseline (RCAEval).

Upstream causal-learn 0.1.2.3 lacks
`causallearn.utils.PCUtils.SkeletonDiscovery.local_skeleton_discovery`, which
RCAEval's RCD / TORAI require. This copy carries that patch, so it is installed
in place of the PyPI package. Pure Python; runtime deps (numpy/scipy/sklearn/
networkx/...) are pinned in the repository-level requirements.txt.

    pip install -e third_party/causallearn
"""
from setuptools import setup, find_packages

setup(
    name="causal-learn",
    version="0.1.2.3",
    packages=find_packages(include=["causallearn", "causallearn.*"]),
    include_package_data=True,
    description="causal-learn (vendored + patched for RCAEval/TORAI).",
    install_requires=[],
)
