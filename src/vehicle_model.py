"""
vehicle_model.py

Nonlinear single-track (bicycle) lane-keeping model with a saturating tire
model, plus finite-difference linearization and zero-order-hold discretization
about a nominal trajectory.

State:   x = [e_y, psi, v_y, r]
           e_y  lateral offset from lane center (m)
           psi  heading error (rad)
           v_y  lateral velocity (m/s)
           r    yaw rate (rad/s)
Input:   u = [delta]  steering angle (rad)
"""

import numpy as np
from scipy.linalg import expm

# -------------------------
# Nominal vehicle parameters
# -------------------------
m = 1500.0          # vehicle mass (kg)
I_z = 3000.0        # yaw moment of inertia (kg m^2)
l_f = 1.2           # distance from CG to front axle (m)
l_r = 1.6           # distance from CG to rear axle (m)
C_f0_nom = 80000.0  # nominal front cornering stiffness (N/rad)
C_r0_nom = 80000.0  # nominal rear cornering stiffness (N/rad)
v_x = 15.0          # longitudinal speed, held constant (m/s)
Ts_default = 0.05   # default sample time (s)


def wrap_angle(x):
    """Wrap an angle (rad) to (-pi, pi]."""
    return (x + np.pi) % (2 * np.pi) - np.pi


def tire_stiffness(alpha, C0=C_f0_nom, k=5.0):
    """Cornering stiffness that saturates (decays) with slip angle magnitude."""
    return C0 * np.exp(-k * np.abs(alpha))


def nonlinear_bicycle(x, u, C_f0=C_f0_nom, C_r0=C_r0_nom):
    """
    Continuous-time nonlinear bicycle dynamics.

    C_f0 / C_r0 are exposed as arguments (rather than hardcoded globals) so the
    same function can be used both to design controllers against a *nominal*
    model and to simulate a *true* plant with mismatched tire parameters
    (see simulate.model_mismatch_experiment).
    """
    e_y, psi, v_y, r = x
    delta = float(u[0]) if hasattr(u, "__len__") else float(u)

    alpha_f = (v_y + l_f * r) / v_x - delta
    alpha_r = (v_y - l_r * r) / v_x

    C_f = tire_stiffness(alpha_f, C_f0)
    C_r = tire_stiffness(alpha_r, C_r0)

    F_yf = -2.0 * C_f * alpha_f
    F_yr = -2.0 * C_r * alpha_r

    e_y_dot = v_y + v_x * psi
    psi_dot = r
    v_y_dot = (F_yf + F_yr) / m - v_x * r
    r_dot = (l_f * F_yf - l_r * F_yr) / I_z

    return np.array([e_y_dot, psi_dot, v_y_dot, r_dot])


def linearize(x, u, C_f0=C_f0_nom, C_r0=C_r0_nom, eps=1e-6):
    """
    Finite-difference linearization of nonlinear_bicycle about (x, u).

    Returns continuous-time (A, B) such that
        dx/dt ~= A @ (x - x0) + B @ (u - u0) + f(x0, u0)
    """
    x = np.asarray(x, dtype=float)
    u = np.atleast_1d(u).astype(float)
    n = x.size
    mu = u.size

    f0 = nonlinear_bicycle(x, u, C_f0, C_r0)
    A = np.zeros((n, n))
    B = np.zeros((n, mu))

    for i in range(n):
        dx = np.zeros_like(x)
        dx[i] = eps
        A[:, i] = (nonlinear_bicycle(x + dx, u, C_f0, C_r0) - f0) / eps

    for j in range(mu):
        du = np.zeros_like(u)
        du[j] = eps
        B[:, j] = (nonlinear_bicycle(x, u + du, C_f0, C_r0) - f0) / eps

    return A, B


def discretize_linear(Ac, Bc, Ts):
    """
    Exact zero-order-hold discretization via matrix exponential of the
    augmented [[Ac, Bc], [0, 0]] block (van Loan's method).
    """
    n = Ac.shape[0]
    mu = Bc.shape[1]
    M = np.zeros((n + mu, n + mu))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Mexp = expm(M * Ts)
    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
    return Ad, Bd
