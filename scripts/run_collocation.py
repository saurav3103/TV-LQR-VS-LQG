#!/usr/bin/env python3
"""
run_collocation.py

Solves the open-loop optimal control problem via direct collocation
(CasADi/IPOPT), applies the resulting steering sequence open-loop to the
nonlinear plant, and plots the optimal vs. applied trajectory.

Requires the optional `casadi` dependency (see requirements.txt).

Usage:
    python scripts/run_collocation.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import matplotlib.pyplot as plt

from lane_keep.vehicle_model import nonlinear_bicycle
from lane_keep.collocation import direct_collocation_solver

X0 = np.array([0.6, 0.12, 0.0, 0.0])
N = 80
TF = 6.0
STEERING_LIMIT = 0.4
EULER_SUBSTEPS = 5  # sub-steps per interval when forward-simulating Uopt


def main():
    print("Solving open-loop optimal control via direct collocation (CasADi)...")
    sol = direct_collocation_solver(X0, N=N, Tf=TF, steering_limit=STEERING_LIMIT)
    X_opt, U_opt, dt = sol["X_opt"], sol["U_opt"], sol["dt"]
    t_opt = np.linspace(0, TF, X_opt.shape[1])

    # Plot the open-loop optimal state/input trajectory.
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_opt, X_opt[0, :], "-o", label="opt e_y")
    plt.plot(t_opt, X_opt[1, :], "-x", label="opt psi")
    plt.title("Open-loop optimal state trajectory (Direct Collocation)")
    plt.ylabel("states")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.step(t_opt[:-1], U_opt[0, :], where="post", label="opt steering")
    plt.ylabel("steering (rad)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Forward-simulate the same open-loop steering sequence on the nonlinear
    # plant (small Euler sub-steps for numerical robustness) to see how much
    # the applied response drifts from the planned optimal trajectory.
    x = X0.copy()
    X_sim = np.zeros_like(X_opt)
    X_sim[:, 0] = X0
    for k in range(U_opt.shape[1]):
        uk = U_opt[:, k]
        for _ in range(EULER_SUBSTEPS):
            x = x + (dt / EULER_SUBSTEPS) * nonlinear_bicycle(x, uk)
        X_sim[:, k + 1] = x

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t_opt, X_opt[0, :], label="opt e_y")
    plt.plot(t_opt, X_sim[0, :], "--", label="open-loop applied (nonlinear sim)")
    plt.ylabel("e_y (m)")
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.step(t_opt[:-1], U_opt[0, :], where="post", label="opt steering")
    plt.ylabel("steering (rad)")
    plt.xlabel("time (s)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("Direct collocation run complete. Compare against run_comparison.py "
          "for the closed-loop TV-LQR/LQG response.")


if __name__ == "__main__":
    main()
