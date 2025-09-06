"""
Merged TV-LQR vs LQG comparison + model mismatch + Lyapunov heatmap.
"""

import numpy as np
from scipy import linalg
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

np.random.seed(0)

# -------------------------
# Utility
# -------------------------
def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def plot_cov_ellipse(ax, mean, cov, nstd=2.0, facecolor='none', **kwargs):
    cov = (cov + cov.T) * 0.5
    vals, vecs = linalg.eigh(cov)
    vals = np.clip(vals, 1e-12, None)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
    width, height = 2 * nstd * np.sqrt(vals[:2])
    ell = Ellipse(xy=(mean[0], mean[1]), width=width, height=height, angle=theta,
                  facecolor=facecolor, **kwargs)
    ax.add_patch(ell)
    return ell

# -------------------------
# Nonlinear bicycle model
# -------------------------
m = 1500.0
I_z = 3000.0
l_f = 1.2
l_r = 1.6
C_f0_nom = 80000.0
C_r0_nom = 80000.0
v_x = 15.0
Ts_default = 0.05

def tire_stiffness(alpha, C0=80000.0):
    k = 5.0
    return C0 * np.exp(-k * np.abs(alpha))

def nonlinear_bicycle(x, u, C_f0=C_f0_nom, C_r0=C_r0_nom):
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

# -------------------------
# Linearize & discretize
# -------------------------
def linearize(x, u, C_f0=C_f0_nom, C_r0=C_r0_nom, eps=1e-6):
    x = x.astype(float)
    u = np.atleast_1d(u).astype(float)
    n = x.size
    mu = u.size
    f0 = nonlinear_bicycle(x, u, C_f0, C_r0)
    A = np.zeros((n, n))
    B = np.zeros((n, mu))
    for i in range(n):
        dx = np.zeros_like(x); dx[i] = eps
        A[:, i] = (nonlinear_bicycle(x + dx, u, C_f0, C_r0) - f0) / eps
    for j in range(mu):
        du = np.zeros_like(u); du[j] = eps
        B[:, j] = (nonlinear_bicycle(x, u + du, C_f0, C_r0) - f0) / eps
    return A, B

def discretize_linear(Ac, Bc, Ts):
    n = Ac.shape[0]; m = Bc.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = Ac
    M[:n, n:] = Bc
    Mexp = expm(M * Ts)
    Ad = Mexp[:n, :n]
    Bd = Mexp[:n, n:]
    return Ad, Bd

# -------------------------
# TV-LQR (backward Riccati)
# -------------------------
def tv_lqr_gains(A_seq, B_seq, Q, R, Qf):
    N = len(A_seq)
    P_seq = [None]*(N+1)
    K_seq = [None]*N
    P_seq[N] = Qf.copy()
    for k in range(N-1, -1, -1):
        A_k = A_seq[k]; B_k = B_seq[k]
        S = R + B_k.T @ P_seq[k+1] @ B_k
        S_inv = linalg.inv(S)
        K_k = S_inv @ (B_k.T @ P_seq[k+1] @ A_k)
        P_k = Q + A_k.T @ P_seq[k+1] @ (A_k - B_k @ K_k)
        P_seq[k] = P_k; K_seq[k] = K_k
    return P_seq, K_seq

# -------------------------
# Discrete Kalman filter
# -------------------------
def discrete_kalman_predict(x, P, A, B, u, Qw):
    x_pred = A @ x + B @ u
    P_pred = A @ P @ A.T + Qw
    return x_pred, P_pred

def discrete_kalman_update(x_pred, P_pred, C, y, Rv):
    S = C @ P_pred @ C.T + Rv
    K_gain = P_pred @ C.T @ linalg.inv(S)
    x_upd = x_pred + K_gain @ (y - C @ x_pred)
    P_upd = (np.eye(P_pred.shape[0]) - K_gain @ C) @ P_pred
    return x_upd, P_upd, K_gain

C_mat = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
])

# -------------------------
# Simulation (paired-noise)
# -------------------------
def simulate_pair(x0, horizon=200, Ts=Ts_default, noise_levels=(1.0,5.0),
                  Q=None, R=None, Qf=None, seed=None,
                  C_f0_lin=C_f0_nom, C_r0_lin=C_r0_nom,
                  C_f0_true=C_f0_nom*1.1, C_r0_true=C_r0_nom*1.1):  # slight mismatch

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    N = horizon
    x0 = x0.astype(float)
    n = x0.size
    m_u = 1

    if Q is None:
        Q = np.diag([50.0, 200.0, 1.0, 1.0])
    if R is None:
        R = np.array([[10.0]])
    if Qf is None:
        Qf = Q*10.0

    # Design TV-LQR gains
    A_seq=[]; B_seq=[]
    x_nom = x0.copy()
    u_nom = np.zeros((N, m_u))
    for k in range(N):
        Ac, Bc = linearize(x_nom, u_nom[k], C_f0_lin, C_r0_lin)
        Ad, Bd = discretize_linear(Ac, Bc, Ts)
        A_seq.append(Ad); B_seq.append(Bd)
        x_nom = x_nom + Ts * nonlinear_bicycle(x_nom, u_nom[k], C_f0_lin, C_r0_lin)

    P_seq, K_seq = tv_lqr_gains(A_seq, B_seq, Q, R, Qf)

    # LQG tuning: higher process noise, lower measurement noise
    Qw = np.diag([0.05, 0.05, 0.2, 0.2]) * float(noise_levels[0])
    Rv = np.diag([0.05, 0.02, 0.05, 0.02])  # observe all states
    global C_mat
    C_mat = np.eye(4)

    w_seq = rng.multivariate_normal(np.zeros(n), Qw, size=N)
    v_seq = rng.multivariate_normal(np.zeros(C_mat.shape[0]), Rv, size=N)

    def run(use_kf):
        x = x0.copy()
        x_est = x0.copy()
        P_est = np.eye(n)*0.1
        x_hist = np.zeros((N, n))
        x_est_hist = np.zeros((N, n))
        u_hist = np.zeros((N, m_u))
        for k in range(N):
            Ad = A_seq[k]; Bd = B_seq[k]; Kk = K_seq[k]
            y = C_mat @ x + v_seq[k]
            if use_kf:
                u_for_predict = np.zeros((m_u,))
                x_pred, P_pred = discrete_kalman_predict(x_est, P_est, Ad, Bd, u_for_predict, Qw)
                x_est, P_est, _ = discrete_kalman_update(x_pred, P_pred, C_mat, y, Rv)
            else:
                x_est = x.copy()
            x_est_hist[k,:] = x_est
            u = (-Kk @ x_est).reshape(m_u)
            u = np.clip(u, -0.5, 0.5)
            x = x + Ts * nonlinear_bicycle(x, u, C_f0_true, C_r0_true) + w_seq[k]
            x_hist[k,:] = x
            u_hist[k,:] = u
        return {'x_hist': x_hist, 'x_est_hist': x_est_hist, 'u_hist': u_hist}

    res_lqr = run(use_kf=False)
    res_lqg = run(use_kf=True)

    def compute_metrics(res):
        xh = res['x_hist']; uh = res['u_hist']
        cost_inst = np.einsum('ti,ij,ti->t', xh, Q, xh) + np.einsum('ti,ij,ti->t', uh, R, uh)
        cost_cum = np.cumsum(cost_inst)
        return {'cost_inst': cost_inst, 'cost_cum': cost_cum, 'energy': float(np.sum(uh**2))}

    return {'metrics': compute_metrics(res_lqr), **res_lqr}, {'metrics': compute_metrics(res_lqg), **res_lqg}


# -------------------------
# State trajectory plot
# -------------------------
def plot_state_trajectory_pair(x_hist_lqr, x_hist_lqg, title_suffix=''):
    plt.figure(figsize=(8,6))
    plt.plot(x_hist_lqr[:,0], x_hist_lqr[:,1], '-b', label='LQR (true-state)')
    plt.plot(x_hist_lqg[:,0], x_hist_lqg[:,1], '-r', label='LQG (KF-estimated)')
    plt.xlabel('e_y (m)')
    plt.ylabel('psi (rad)')
    plt.title('State trajectory comparison ' + title_suffix)
    plt.grid(True)
    plt.legend()
    plt.show()

# -------------------------
# KF estimation error plot
# -------------------------
def plot_estimation_error(res_lqg):
    err = res_lqg['x_hist'] - res_lqg['x_est_hist']
    t = np.arange(err.shape[0])
    labels = ['e_y', 'psi', 'v_y', 'r']
    plt.figure(figsize=(9,6))
    for i in range(4):
        plt.plot(t, err[:,i], label=f'error {labels[i]}')
    plt.axhline(0, color='k', lw=0.5)
    plt.xlabel('Time step')
    plt.ylabel('Estimation error')
    plt.title('Kalman Filter Estimation Error')
    plt.grid(True)
    plt.legend()
    plt.show()

# -------------------------
# Model mismatch experiment
# -------------------------
def model_mismatch_experiment(x0, horizon=200, mismatch_levels=[0.5,0.8,1.0,1.2,1.5]):
    results = []
    for scale in mismatch_levels:
        res_lqr, res_lqg = simulate_pair(
            x0, horizon=horizon, seed=42,
            C_f0_lin=C_f0_nom, C_r0_lin=C_r0_nom,
            C_f0_true=C_f0_nom*scale, C_r0_true=C_r0_nom*scale
        )
        results.append({
            'scale': scale,
            'cost_lqr': res_lqr['metrics']['cost_cum'][-1],
            'cost_lqg': res_lqg['metrics']['cost_cum'][-1]
        })
    scales = [r['scale'] for r in results]
    cost_lqr = [r['cost_lqr'] for r in results]
    cost_lqg = [r['cost_lqg'] for r in results]
    plt.figure(figsize=(7,5))
    plt.plot(scales, cost_lqr, 'o-b', label='LQR')
    plt.plot(scales, cost_lqg, 's-r', label='LQG')
    plt.xlabel('Tire stiffness scale (true/nominal)')
    plt.ylabel('Final cumulative cost J')
    plt.title('Model mismatch effect on LQR vs LQG')
    plt.grid(True); plt.legend()
    plt.show()
    return results

# -------------------------
# Lyapunov heatmap
# -------------------------
def lyapunov_heatmap(A, Q=np.eye(4), x_range=(-2,2), y_range=(-2,2), N=50):
    P = linalg.solve_discrete_lyapunov(A.T, Q)
    xs = np.linspace(*x_range, N)
    ys = np.linspace(*y_range, N)
    V = np.zeros((N,N))
    for i, x1 in enumerate(xs):
        for j, x2 in enumerate(ys):
            x_vec = np.array([x1, x2, 0, 0])
            V[j,i] = x_vec.T @ P @ x_vec
    plt.figure(figsize=(6,5))
    plt.contourf(xs, ys, V, levels=20, cmap='viridis')
    plt.colorbar(label='Lyapunov function V(x)')
    plt.xlabel('e_y'); plt.ylabel('psi')
    plt.title('Lyapunov heatmap')
    plt.show()

# -------------------------
# Main
# -------------------------
if __name__ == '__main__':
    X0 = np.array([0.6, 0.12, 0.0, 0.0])

    print("Running single simulation for LQR vs LQG plots...")
    metrics_lqr, metrics_lqg = simulate_pair(X0, horizon=200, seed=42)

    # State trajectory plot
    plot_state_trajectory_pair(metrics_lqr['x_hist'], metrics_lqg['x_hist'], title_suffix='(single run)')

    # KF estimation error plot
    plot_estimation_error(metrics_lqg)

    # Cumulative cost
    plt.figure(figsize=(7,5))
    plt.plot(metrics_lqr['metrics']['cost_cum'], label='LQR cumulative cost', color='blue')
    plt.plot(metrics_lqg['metrics']['cost_cum'], label='LQG cumulative cost', color='red')
    plt.xlabel('Time step')
    plt.ylabel('Cumulative cost')
    plt.title('TV-LQR vs LQG cumulative cost over time')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Model mismatch
    print("Running model mismatch experiment...")
    results = model_mismatch_experiment(X0, horizon=200)
    for r in results:
        print(f"Scale={r['scale']:.2f} | LQR cost={r['cost_lqr']:.2f}, LQG cost={r['cost_lqg']:.2f}")

    # Lyapunov heatmap
    print("Generating Lyapunov heatmap...")
    A_lin, _ = linearize(np.zeros(4), np.zeros(1))
    lyapunov_heatmap(A_lin)
