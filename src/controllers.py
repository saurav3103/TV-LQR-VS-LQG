"""
controllers.py

Finite-horizon time-varying LQR (backward Riccati recursion) and a discrete-
time Kalman filter (predict/update), used respectively as the feedback law
and the state estimator in the LQR vs LQG comparison.
"""

import numpy as np
from scipy import linalg


def tv_lqr_gains(A_seq, B_seq, Q, R, Qf):
    """
    Backward Riccati recursion for a finite-horizon time-varying LQR.

    A_seq, B_seq : lists of discrete-time (A_k, B_k) matrices, length N
    Q, R         : stage cost weights (state, input)
    Qf           : terminal cost weight

    Returns
    -------
    P_seq : list of N+1 cost-to-go matrices (P_seq[N] = Qf)
    K_seq : list of N feedback gains, u_k = -K_seq[k] @ x_k
    """
    N = len(A_seq)
    P_seq = [None] * (N + 1)
    K_seq = [None] * N
    P_seq[N] = Qf.copy()

    for k in range(N - 1, -1, -1):
        A_k = A_seq[k]
        B_k = B_seq[k]
        S = R + B_k.T @ P_seq[k + 1] @ B_k
        S_inv = linalg.inv(S)
        K_k = S_inv @ (B_k.T @ P_seq[k + 1] @ A_k)
        P_k = Q + A_k.T @ P_seq[k + 1] @ (A_k - B_k @ K_k)
        P_seq[k] = P_k
        K_seq[k] = K_k

    return P_seq, K_seq


def discrete_kalman_predict(x, P, A, B, u, Qw):
    """Time update: propagate state estimate and covariance through (A, B)."""
    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Qw
    return x_pred, P_pred


def discrete_kalman_update(x_pred, P_pred, C, y, Rv):
    """Measurement update given observation y = C @ x + v, v ~ N(0, Rv)."""
    S = C @ P_pred @ C.T + Rv
    K_gain = P_pred @ C.T @ linalg.inv(S)
    x_upd = x_pred + K_gain @ (y - C @ x_pred)
    P_upd = (np.eye(P_pred.shape[0]) - K_gain @ C) @ P_pred
    return x_upd, P_upd, K_gain
