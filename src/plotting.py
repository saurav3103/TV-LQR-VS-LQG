"""
plotting.py

Plotting utilities for the LQR vs LQG lane-keeping comparison: state
trajectories, estimation error, control inputs, cost curves, Monte Carlo
histograms, and a Lyapunov function heatmap.
"""

import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg
import matplotlib.pyplot as plt

from .vehicle_model import Ts_default


def print_theory(header, body):
    """Print a short theory/context block ahead of a figure."""
    bar = "=" * 80
    print(f"\n{bar}\n{header}\n{bar}\n{body}\n")


def plot_cov_ellipse(ax, mean, cov, nstd=2.0, facecolor="none", **kwargs):
    """Draw an nstd-sigma covariance ellipse for a 2D (sub)covariance."""
    cov = (cov + cov.T) * 0.5
    vals, vecs = linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * nstd * np.sqrt(vals[:2])
    ell = Ellipse(xy=(mean[0], mean[1]), width=width, height=height, angle=theta,
                  facecolor=facecolor, **kwargs)
    ax.add_patch(ell)
    return ell


def plot_state_trajectory_pair(res_lqr, res_lqg, title_suffix=""):
    print_theory(
        "State Trajectory Comparison (True states, e_y vs psi)",
        "LQR uses the full true state in the control law (perfect state knowledge). "
        "LQG uses a Kalman filter estimate. Differences in paths reveal how "
        "estimation error and measurement noise influence closed-loop behavior.",
    )
    x_lqr = res_lqr["x_hist"]
    x_lqg = res_lqg["x_hist"]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(x_lqr[:, 0], x_lqr[:, 1], "-b", label="LQR (true-state feedback)")
    ax.plot(x_lqg[:, 0], x_lqg[:, 1], "-r", label="LQG (KF-based)")
    ax.set_xlabel("e_y (m)")
    ax.set_ylabel("psi (rad)")
    ax.set_title("Trajectory comparison (e_y vs psi) " + title_suffix)
    ax.legend()
    ax.grid(True)
    plt.show()


def plot_estimation_error(res_lqg):
    print_theory(
        "Estimation Error (Kalman Filter Performance)",
        "True state minus estimate, over time. Small, zero-mean, bounded errors "
        "indicate a well-tuned observer; large bias or growth indicates poor "
        "model/noise tuning.",
    )
    err = res_lqg["x_hist"] - res_lqg["x_est_hist"]
    t = np.arange(err.shape[0])
    labels = ["e_y (m)", "psi (rad)", "v_y (m/s)", "r (rad/s)"]
    fig, axs = plt.subplots(4, 1, figsize=(9, 7), sharex=True)
    for i in range(4):
        axs[i].plot(t, err[:, i], label=f"error {labels[i]}")
        axs[i].axhline(0, color="k", lw=0.5)
        axs[i].legend()
        axs[i].grid(True)
    axs[-1].set_xlabel("time step")
    plt.suptitle("Kalman Filter Estimation Error (true - estimate)")
    plt.show()


def plot_control_inputs(res_lqr, res_lqg, Ts=Ts_default):
    print_theory(
        "Control Input Comparison",
        "Steering sequences show how much control effort each method uses. "
        "LQG may steer differently because it acts on estimated states; "
        "differences also reflect estimator delay/noise.",
    )
    t = np.arange(res_lqr["u_hist"].shape[0]) * Ts
    u1 = res_lqr["u_hist"][:, 0]
    u2 = res_lqg["u_hist"][:, 0]
    fig, ax = plt.subplots(1, 1, figsize=(9, 4))
    ax.plot(t, u1, label="LQR (full-state)", lw=1.5)
    ax.plot(t, u2, label="LQG (with KF)", lw=1.5, alpha=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("steering (rad)")
    ax.set_title("Control input comparison")
    ax.grid(True)
    ax.legend()
    plt.show()


def plot_costs(metrics_lqr, metrics_lqg, Ts=Ts_default):
    print_theory(
        "Instantaneous & Cumulative Cost",
        "Instantaneous cost = x^T Q x + u^T R u per step. Cumulative cost shows "
        "which controller attains lower total objective J across the horizon.",
    )
    t = np.arange(metrics_lqr["cost_inst"].shape[0]) * Ts
    fig, axs = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    axs[0].plot(t, metrics_lqr["cost_inst"], label="LQR")
    axs[0].plot(t, metrics_lqg["cost_inst"], label="LQG")
    axs[0].set_ylabel("instantaneous cost")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(t, metrics_lqr["cost_cum"], label="LQR")
    axs[1].plot(t, metrics_lqg["cost_cum"], label="LQG")
    axs[1].set_xlabel("time (s)")
    axs[1].set_ylabel("cumulative cost J")
    axs[1].legend()
    axs[1].grid(True)
    plt.suptitle("Cost Comparison")
    plt.show()


def print_run_summary(metrics_lqr, metrics_lqg, est_rmse):
    print_theory(
        "Run Summary (numeric)",
        f"Final cumulative cost: LQR = {metrics_lqr['cost_cum'][-1]:.4f}, "
        f"LQG = {metrics_lqg['cost_cum'][-1]:.4f}\n"
        f"Control energy (sum u^2): LQR = {metrics_lqr['energy']:.4f}, "
        f"LQG = {metrics_lqg['energy']:.4f}\n"
        f"Estimator RMSE (e_y, psi, v_y, r): {est_rmse.round(4)}",
    )


def plot_monte_carlo_histograms(mc_results):
    """mc_results: dict from simulate.monte_carlo_compare()."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].hist(mc_results["energies_lqr"], bins=10)
    axs[0, 0].set_title("Control energy LQR")
    axs[0, 1].hist(mc_results["energies_lqg"], bins=10)
    axs[0, 1].set_title("Control energy LQG")
    axs[1, 0].hist(mc_results["costs_lqr"], bins=10)
    axs[1, 0].set_title("Total cost LQR")
    axs[1, 1].hist(mc_results["costs_lqg"], bins=10)
    axs[1, 1].set_title("Total cost LQG")
    plt.suptitle("Monte Carlo paired histograms")
    plt.show()


def plot_model_mismatch(results):
    """results: list of dicts from simulate.model_mismatch_experiment()."""
    scales = [r["scale"] for r in results]
    cost_lqr = [r["cost_lqr"] for r in results]
    cost_lqg = [r["cost_lqg"] for r in results]
    plt.figure(figsize=(7, 5))
    plt.plot(scales, cost_lqr, "o-b", label="LQR")
    plt.plot(scales, cost_lqg, "s-r", label="LQG")
    plt.xlabel("Tire stiffness scale (true/nominal)")
    plt.ylabel("Final cumulative cost J")
    plt.title("Model mismatch effect on LQR vs LQG")
    plt.grid(True)
    plt.legend()
    plt.show()


def lyapunov_heatmap(A, Q=None, x_range=(-2, 2), y_range=(-2, 2), N=50):
    """
    Solve the discrete Lyapunov equation A^T P A - P + Q = 0 and plot level
    sets of V(x) = x^T P x over the (e_y, psi) plane (v_y, r held at 0).
    """
    if Q is None:
        Q = np.eye(4)
    P = linalg.solve_discrete_lyapunov(A.T, Q)
    xs = np.linspace(*x_range, N)
    ys = np.linspace(*y_range, N)
    V = np.zeros((N, N))
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(ys):
            x_vec = np.array([x1, x2, 0, 0])
            V[j, i] = x_vec.T @ P @ x_vec
    plt.figure(figsize=(6, 5))
    plt.contourf(xs, ys, V, levels=20, cmap="viridis")
    plt.colorbar(label="Lyapunov function V(x)")
    plt.xlabel("e_y")
    plt.ylabel("psi")
    plt.title("Lyapunov heatmap")
    plt.show()
