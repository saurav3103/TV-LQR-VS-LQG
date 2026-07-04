#!/usr/bin/env python3
"""
run_comparison.py

Single paired LQR vs LQG run: trajectory, estimation error, control input,
and cost comparison plots, plus a numeric run summary.

Usage:
    python scripts/run_comparison.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from lane_keep.simulate import simulate_pair
from lane_keep.plotting import (
    plot_control_inputs,
    plot_costs,
    plot_estimation_error,
    plot_state_trajectory_pair,
    print_run_summary,
)

HORIZON = 240
X0 = np.array([0.6, 0.12, 0.0, 0.0])
SEED = 42


def main():
    print("Running a paired LQR vs LQG experiment (same noise for both)...")
    out = simulate_pair(X0, horizon=HORIZON, seed=SEED, verbose=False)

    plot_state_trajectory_pair(out["res_lqr"], out["res_lqg"], title_suffix=f"(horizon={HORIZON})")
    plot_estimation_error(out["res_lqg"])
    plot_control_inputs(out["res_lqr"], out["res_lqg"])
    plot_costs(out["metrics_lqr"], out["metrics_lqg"])
    print_run_summary(out["metrics_lqr"], out["metrics_lqg"], out["est_rmse"])


if __name__ == "__main__":
    main()
