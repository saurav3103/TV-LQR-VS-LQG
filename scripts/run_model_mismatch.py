#!/usr/bin/env python3
"""
run_model_mismatch.py

Sweeps true-vs-nominal tire stiffness ratio and plots final closed-loop cost
for LQR and LQG at each mismatch level, to see how estimator-based feedback
degrades (or holds up) under plant/model mismatch.

Usage:
    python scripts/run_model_mismatch.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from lane_keep.simulate import model_mismatch_experiment
from lane_keep.plotting import plot_model_mismatch

X0 = np.array([0.6, 0.12, 0.0, 0.0])
HORIZON = 200
MISMATCH_LEVELS = (0.5, 0.8, 1.0, 1.2, 1.5)


def main():
    print("Running model mismatch experiment...")
    results = model_mismatch_experiment(X0, horizon=HORIZON, mismatch_levels=MISMATCH_LEVELS)
    for r in results:
        print(f"Scale={r['scale']:.2f} | LQR cost={r['cost_lqr']:.2f}, LQG cost={r['cost_lqg']:.2f}")
    plot_model_mismatch(results)


if __name__ == "__main__":
    main()
