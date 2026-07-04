"""
simulate.py

Paired LQR vs LQG simulation of the nonlinear bicycle lane-keeping model,
plus a Monte Carlo comparison and a model-mismatch sensitivity sweep.

"Paired" means both controllers see the exact same process/measurement noise
realization on a given trial, so any performance gap is attributable to the
state-feedback source (true state vs. Kalman filter estimate), not to random
variation between runs.
"""

import time

import numpy as np
from scipy import linalg

from .vehicle_model import (
    C_f0_nom,
    C_r0_nom,
    Ts_default,
    discretize_linear,
    linearize,
    nonlinear_bicycle,
)
from .controllers import discrete_kalman_predict, discrete_kalman_update, tv_lqr_gains

# Default measurement matrix: observe lateral offset and heading error only.
# Pass C_mat=np.eye(4) (with a correspondingly sized Rv) to simulate_pair for
# a full-state-measurement configuration instead.
C_MAT_PARTIAL = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])


def compute_metrics(res, Q, R):
    """Instantaneous/cumulative quadratic cost and total control energy."""
    xh = res["x_hist"]
    uh = res["u_hist"]
    cost_inst = np.einsum("ti,ij,ti->t", xh, Q, xh) + np.einsum("ti,ij,ti->t", uh, R, uh)
    cost_cum = np.cumsum(cost_inst)
    return {"cost_inst": cost_inst, "cost_cum": cost_cum, "energy": float(np.sum(uh ** 2))}


def simulate_pair(
    x0,
    horizon=200,
    Ts=Ts_default,
    Q=None,
    R=None,
    Qf=None,
    seed=None,
    verbose=False,
    C_mat=None,
    Qw_scale=1.0,
    Qw_diag=(0.01, 0.01, 0.05, 0.05),
    Rv_diag=(0.5, 0.2),
    C_f0_lin=C_f0_nom,
    C_r0_lin=C_r0_nom,
    C_f0_true=C_f0_nom,
    C_r0_true=C_r0_nom,
    steering_limit=0.5,
):
    """
    Run a paired LQR (true-state feedback) vs LQG (Kalman-filter feedback)
    simulation using an identical noise realization for both.

    The controller is designed by linearizing/discretizing about a nominal
    (zero-input) rollout using (C_f0_lin, C_r0_lin), then applied to a "true"
    plant using (C_f0_true, C_r0_true). Setting these equal (the default)
    reproduces the nominal (no mismatch) case; see model_mismatch_experiment
    for a sweep over mismatch levels.

    Parameters
    ----------
    C_mat : measurement matrix. Defaults to C_MAT_PARTIAL (observe e_y, psi).
            Pass np.eye(4) for full-state measurement (with a 4x4 Rv_diag).
    Qw_diag, Rv_diag : diagonal process/measurement noise covariances.
    Qw_scale : multiplier applied to Qw_diag (kept separate for convenient
               sweeps over process noise intensity).

    Returns
    -------
    dict with keys: res_lqr, res_lqg, metrics_lqr, metrics_lqg, est_rmse,
    Q, R, Qf, Qw, Rv
    """
    rng = np.random.default_rng(seed)

    N = horizon
    x0 = np.asarray(x0, dtype=float)
    n = x0.size
    m_u = 1

    if Q is None:
        Q = np.diag([50.0, 200.0, 1.0, 1.0])
    if R is None:
        R = np.array([[10.0]])
    if Qf is None:
        Qf = Q * 10.0
    if C_mat is None:
        C_mat = C_MAT_PARTIAL

    Qw = np.diag(Qw_diag) * float(Qw_scale)
    Rv = np.diag(Rv_diag)

    # Design TV-LQR gains from a nominal (zero-input) rollout linearized with
    # the "design" tire parameters (C_f0_lin, C_r0_lin).
    A_seq, B_seq = [], []
    x_nom = x0.copy()
    u_nom = np.zeros((N, m_u))
    for k in range(N):
        Ac, Bc = linearize(x_nom, u_nom[k], C_f0_lin, C_r0_lin)
        Ad, Bd = discretize_linear(Ac, Bc, Ts)
        A_seq.append(Ad)
        B_seq.append(Bd)
        x_nom = x_nom + Ts * nonlinear_bicycle(x_nom, u_nom[k], C_f0_lin, C_r0_lin)

    P_seq, K_seq = tv_lqr_gains(A_seq, B_seq, Q, R, Qf)

    # Pre-sample noise once so LQR and LQG see identical realizations.
    w_seq = rng.multivariate_normal(np.zeros(n), Qw, size=N)
    v_seq = rng.multivariate_normal(np.zeros(C_mat.shape[0]), Rv, size=N)

    def run(use_kf):
        x = x0.copy()
        x_est = x0.copy()
        P_est = np.eye(n) * 0.1
        x_hist = np.zeros((N, n))
        x_est_hist = np.zeros((N, n))
        u_hist = np.zeros((N, m_u))

        for k in range(N):
            Ad, Bd, Kk = A_seq[k], B_seq[k], K_seq[k]
            y = C_mat @ x + v_seq[k]

            if use_kf:
                u_for_predict = np.zeros((m_u,))
                x_pred, P_pred = discrete_kalman_predict(x_est, P_est, Ad, Bd, u_for_predict, Qw)
                x_est, P_est, _ = discrete_kalman_update(x_pred, P_pred, C_mat, y, Rv)
            else:
                # LQR: perfect state knowledge.
                x_est = x.copy()

            u = (-Kk @ x_est).reshape(m_u)
            u = np.clip(u, -steering_limit, steering_limit)

            x = x + Ts * nonlinear_bicycle(x, u, C_f0_true, C_r0_true) + w_seq[k]

            x_hist[k, :] = x
            x_est_hist[k, :] = x_est
            u_hist[k, :] = u

            if verbose and (k % (N // 5 + 1) == 0):
                tag = "LQG" if use_kf else "LQR"
                print(f"[{tag}] step {k} | e_y={x[0]:.3f} psi={x[1]:.3f} u={u[0]:.3f}")

        return {"x_hist": x_hist, "x_est_hist": x_est_hist, "u_hist": u_hist}

    res_lqr = run(use_kf=False)
    res_lqg = run(use_kf=True)

    metrics_lqr = compute_metrics(res_lqr, Q, R)
    metrics_lqg = compute_metrics(res_lqg, Q, R)

    est_err = res_lqg["x_hist"] - res_lqg["x_est_hist"]
    est_rmse = np.sqrt(np.mean(est_err ** 2, axis=0))

    return {
        "res_lqr": res_lqr,
        "res_lqg": res_lqg,
        "metrics_lqr": metrics_lqr,
        "metrics_lqg": metrics_lqg,
        "est_rmse": est_rmse,
        "Q": Q,
        "R": R,
        "Qf": Qf,
        "Qw": Qw,
        "Rv": Rv,
    }


def monte_carlo_compare(x0, trials=20, horizon=160, **simulate_kwargs):
    """
    Run `trials` paired LQR/LQG simulations (each with its own noise seed) and
    summarize control-energy and total-cost distributions.

    Returns a dict of lists: energies_lqr, energies_lqg, costs_lqr, costs_lqg.
    """
    energies_lqr, energies_lqg = [], []
    costs_lqr, costs_lqg = [], []

    t0 = time.time()
    for i in range(trials):
        out = simulate_pair(x0, horizon=horizon, seed=int(1000 + i), **simulate_kwargs)
        energies_lqr.append(out["metrics_lqr"]["energy"])
        energies_lqg.append(out["metrics_lqg"]["energy"])
        costs_lqr.append(out["metrics_lqr"]["cost_cum"][-1])
        costs_lqg.append(out["metrics_lqg"]["cost_cum"][-1])
    elapsed = time.time() - t0

    print(f"Monte Carlo: {trials} trials done in {elapsed:.1f}s")
    print(f"Control energy LQR: {np.mean(energies_lqr):.4f} +/- {np.std(energies_lqr):.4f}")
    print(f"Control energy LQG: {np.mean(energies_lqg):.4f} +/- {np.std(energies_lqg):.4f}")
    print(f"Total cost LQR:     {np.mean(costs_lqr):.4f} +/- {np.std(costs_lqr):.4f}")
    print(f"Total cost LQG:     {np.mean(costs_lqg):.4f} +/- {np.std(costs_lqg):.4f}")

    return {
        "energies_lqr": energies_lqr,
        "energies_lqg": energies_lqg,
        "costs_lqr": costs_lqr,
        "costs_lqg": costs_lqg,
    }


def model_mismatch_experiment(x0, horizon=200, mismatch_levels=(0.5, 0.8, 1.0, 1.2, 1.5), seed=42, **simulate_kwargs):
    """
    Sweep the ratio of true-to-nominal tire stiffness and record final
    cumulative cost for LQR and LQG at each mismatch level.

    Returns a list of dicts: {"scale", "cost_lqr", "cost_lqg"}.
    """
    results = []
    for scale in mismatch_levels:
        out = simulate_pair(
            x0,
            horizon=horizon,
            seed=seed,
            C_f0_lin=C_f0_nom,
            C_r0_lin=C_r0_nom,
            C_f0_true=C_f0_nom * scale,
            C_r0_true=C_r0_nom * scale,
            **simulate_kwargs,
        )
        results.append({
            "scale": scale,
            "cost_lqr": out["metrics_lqr"]["cost_cum"][-1],
            "cost_lqg": out["metrics_lqg"]["cost_cum"][-1],
        })
    return results
