#!/usr/bin/env python
"""Run RCAEval's TORAI method on the MicroARCL agent-network (MDOC) dataset."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import torai_common

DEFAULT_LOAD_DIR = os.path.join(HERE, "data", "MDOC", "abnormal", "load-5")


if __name__ == "__main__":
    torai_common.main(DEFAULT_LOAD_DIR)
