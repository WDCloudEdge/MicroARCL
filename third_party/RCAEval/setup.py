"""Minimal, self-contained setup for the vendored RCAEval package.

Only the RCAEval source is vendored here (for the TORAI comparison baseline);
runtime dependencies are pinned in the repository-level requirements.txt, so
this setup declares none of its own. Install editable with:

    pip install -e third_party/RCAEval
"""
from setuptools import setup, find_packages

setup(
    name="RCAEval",
    version="1.7.0",
    packages=find_packages(include=["RCAEval", "RCAEval.*"]),
    include_package_data=True,
    description="RCAEval (vendored) -- Root Cause Analysis benchmark; used here "
                "for the TORAI comparison baseline.",
    install_requires=[],
)
