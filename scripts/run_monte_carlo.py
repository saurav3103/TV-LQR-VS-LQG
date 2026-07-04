#!/usr/bin/env python3
"""
run_monte_carlo.py

Paired Monte Carlo comparison of LQR vs LQG across many noise realizations:
prints control-energy / total-cost summary statistics and plots histograms.

Usage:
    python scripts/run_monte_carlo.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

from lane_keep.simulate import monte_carlo_compare
from lane_keep.plotting import plot_monte_carlo_histograms, print_theory

X0 = np.array([0.6, 0.12, 0.0, 0.0])
TRIALS = 20
HORIZON = 160


def main():
    print_theory(
        "Paired Monte Carlo Comparison",
        "Each trial uses the exact same sequence of process and measurement "
        "noise for both LQR and LQG, so differences are caused only by the "
        "controller/estimator interaction. We collect total cost and control "
        "energy and show histograms & summary statistics.",
    )
    mc_results = monte_carlo_compare(X0, trials=TRIALS, horizon=HORIZON)
    plot_monte_carlo_histograms(mc_results)


if __name__ == "__main__":
    main()
