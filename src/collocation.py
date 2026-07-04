"""
collocation.py

Open-loop optimal control baseline for the nonlinear bicycle lane-keeping
model, computed via direct (trapezoidal) collocation with CasADi/IPOPT. This
is a Pontryagin-style numerical solution used to compare against the
closed-loop TV-LQR / LQG controllers in simulate.py.

Requires the optional `casadi` dependency (see requirements.txt).
"""

import numpy as np

from .vehicle_model import C_f0_nom, C_r0_nom, I_z, l_f, l_r, m, v_x

try:
    import casadi as ca
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "collocation.py requires casadi. Install it with `pip install casadi` "
        "(only needed for the direct-collocation baseline)."
    ) from exc


def tire_stiffness_ca(alpha, C0=C_f0_nom, k=5.0):
    """CasADi-symbolic version of vehicle_model.tire_stiffness."""
    return C0 * ca.exp(-k * ca.fabs(alpha))


def casadi_bicycle_dynamics(x, u, C_f0=C_f0_nom, C_r0=C_r0_nom):
    """CasADi-symbolic bicycle dynamics. x, u are CasADi SX/MX types."""
    e_y, psi, v_y, r = x[0], x[1], x[2], x[3]
    delta = u[0]

    alpha_f = (v_y + l_f * r) / v_x - delta
    alpha_r = (v_y - l_r * r) / v_x

    C_f = tire_stiffness_ca(alpha_f, C_f0)
    C_r = tire_stiffness_ca(alpha_r, C_r0)

    F_yf = -2.0 * C_f * alpha_f
    F_yr = -2.0 * C_r * alpha_r

    e_y_dot = v_y + v_x * psi
    psi_dot = r
    v_y_dot = (F_yf + F_yr) / m - v_x * r
    r_dot = (l_f * F_yf - l_r * F_yr) / I_z

    return ca.vertcat(e_y_dot, psi_dot, v_y_dot, r_dot)


def direct_collocation_solver(
    x0_np,
    N=60,
    Tf=6.0,
    steering_limit=0.5,
    Q=None,
    R=None,
    Qf=None,
    solver_opts=None,
    warm_start=None,
):
    """
    Solve the open-loop optimal control problem

        min  integral_0^Tf ( x^T Q x + u^T R u ) dt  +  x(Tf)^T Qf x(Tf)
        s.t. dx/dt = f(x, u),  x(0) = x0,  |u| <= steering_limit

    via trapezoidal direct collocation, using CasADi + IPOPT.

    Parameters
    ----------
    x0_np : initial state (numpy array, shape (4,))
    N : number of control intervals
    Tf : horizon length (s)
    warm_start : optional dict with 'X_nom' (4 x N+1) and 'U_nom' (1 x N)
                 initial guesses, e.g. from a nominal TV-LQR rollout.

    Returns
    -------
    dict with keys: X_opt (4 x N+1), U_opt (1 x N), dt, w_opt
    """
    nx, nu = 4, 1
    dt = Tf / N

    X = ca.SX.sym("X", nx, N + 1)
    U = ca.SX.sym("U", nu, N)

    if Q is None:
        Q = np.diag([50.0, 200.0, 1.0, 1.0])
    if R is None:
        R = np.array([[10.0]])
    if Qf is None:
        Qf = Q * 10.0

    obj = 0
    g, lbg, ubg = [], [], []

    for k in range(N):
        xk, xk1, uk = X[:, k], X[:, k + 1], U[:, k]
        fk = casadi_bicycle_dynamics(xk, uk)
        fk1 = casadi_bicycle_dynamics(xk1, uk)
        # Trapezoidal collocation defect constraint.
        gk = xk1 - xk - (dt / 2.0) * (fk + fk1)
        g.append(gk)
        lbg.extend([0.0] * nx)
        ubg.extend([0.0] * nx)
        obj = obj + (ca.mtimes([xk.T, Q, xk]) + ca.mtimes([uk.T, R, uk])) * dt

    xN = X[:, -1]
    obj = obj + ca.mtimes([xN.T, Qf, xN])

    g.append(X[:, 0] - ca.DM(x0_np))
    lbg.extend([0.0] * nx)
    ubg.extend([0.0] * nx)

    g = ca.vertcat(*g)
    w = ca.vertcat(ca.reshape(X, nx * (N + 1), 1), ca.reshape(U, nu * N, 1))

    nlp = {"x": w, "f": obj, "g": g}
    if solver_opts is None:
        solver_opts = {"ipopt.print_level": 0, "print_time": 0, "ipopt.max_iter": 1000}
    solver = ca.nlpsol("solver", "ipopt", nlp, solver_opts)

    w0 = np.zeros((nx * (N + 1) + nu * N, 1))
    lbw = -1e20 * np.ones_like(w0)
    ubw = 1e20 * np.ones_like(w0)

    def idx_X(k):
        return slice(k * nx, (k + 1) * nx)

    def idx_U(k):
        return slice(nx * (N + 1) + k * nu, nx * (N + 1) + (k + 1) * nu)

    for k in range(N + 1):
        alpha = k / float(N)
        x_init = (1.0 - alpha) * x0_np + alpha * np.zeros_like(x0_np)
        w0[idx_X(k), 0] = x_init
        lbw[idx_X(k)] = -1e6
        ubw[idx_X(k)] = 1e6

    for k in range(N):
        w0[idx_U(k), 0] = 0.0
        lbw[idx_U(k)] = -steering_limit
        ubw[idx_U(k)] = steering_limit

    if warm_start is not None:
        Xg = warm_start.get("X_nom")
        Ug = warm_start.get("U_nom")
        if Xg is not None and Ug is not None:
            for k in range(min(N + 1, Xg.shape[1])):
                w0[idx_X(k), 0] = Xg[:, k]
            for k in range(min(N, Ug.shape[1])):
                w0[idx_U(k), 0] = Ug[:, k]

    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=np.array(lbg), ubg=np.array(ubg))
    w_opt = sol["x"].full().flatten()

    X_opt = np.zeros((nx, N + 1))
    U_opt = np.zeros((nu, N))
    for k in range(N + 1):
        X_opt[:, k] = w_opt[idx_X(k)]
    for k in range(N):
        U_opt[:, k] = w_opt[idx_U(k)]

    return {"X_opt": X_opt, "U_opt": U_opt, "dt": dt, "w_opt": w_opt}
