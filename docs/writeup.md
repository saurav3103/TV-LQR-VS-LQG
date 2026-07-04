# TV-LQR vs LQG for Lane-Keeping Control: Write-up

## 1. Problem setup

The goal is to regulate lateral lane-keeping error for a vehicle modeled as a
nonlinear single-track (bicycle) system, and to compare two feedback
strategies built on the same finite-horizon LQR gain sequence:

- **LQR** — the gain sequence applied to the *true* state (perfect state
  knowledge; not physically realizable, but a useful upper bound on
  achievable performance).
- **LQG** — the same gain sequence applied to a *Kalman filter estimate*,
  built from noisy, partial (or full) state measurements.

Because LQR and LQG share the same gain sequence, any performance gap comes
entirely from the quality of the state estimate — this isolates the cost of
imperfect measurement/estimation, which is the point of the comparison.

A third baseline, an **open-loop optimal control** solved via direct
(trapezoidal) collocation with CasADi/IPOPT, gives a Pontryagin-style
reference trajectory to compare closed-loop performance against.

### State and dynamics

State `x = [e_y, psi, v_y, r]`: lateral offset, heading error, lateral
velocity, yaw rate. Input `u = delta`: steering angle. Longitudinal speed
`v_x` is held constant. Tire cornering stiffness saturates with slip angle,
`C(alpha) = C0 * exp(-k|alpha|)`, which is the model's main nonlinearity —
without it the model reduces to a standard linear bicycle model.

The controller is designed by linearizing (finite-difference Jacobians) and
zero-order-hold discretizing about a nominal zero-input rollout, then solving
a finite-horizon backward Riccati recursion for a time-varying gain sequence
`K_k`. The LQG estimator is a standard discrete-time Kalman filter run with
the same `(A_k, B_k)` sequence used for control design.

## 2. Method: paired simulation

LQR and LQG are compared using a **paired-noise** protocol: for a given
trial, the exact same process-noise and measurement-noise sequences are drawn
once and used for both controllers. This removes random variation as an
explanation for any cost gap — differences are attributable only to whether
the controller acts on the true state or the KF estimate.

Four experiments were run:

1. **Single paired run** — trajectory, estimation error, control input, and
   cost curves for one noise realization.
2. **Monte Carlo comparison** — 20 paired trials, summarizing control energy
   and total cost distributions (mean ± std).
3. **Model mismatch sweep** — the *true* plant's tire stiffness is scaled
   relative to the stiffness assumed during controller design/linearization
   (`0.5x` to `1.5x`), to see how each strategy degrades under model error.
4. **Open-loop optimal control** — direct collocation trajectory compared
   against the closed-loop responses.

## 3. Results

### Single run (horizon = 240 steps, seed = 42)

| Metric | LQR | LQG |
|---|---|---|
| Final cumulative cost | 188,450 | 4,831,106 |
| Control energy (sum u²) | 54.6 | 59.0 |
| Estimator RMSE (e_y, psi, v_y, r) | — | [0.66, 0.19, 1.05, 0.70] |

### Monte Carlo (20 trials, horizon = 160)

| Metric | LQR (mean ± std) | LQG (mean ± std) |
|---|---|---|
| Control energy | 37.9 ± 1.9 | 38.7 ± 1.1 |
| Total cost | 1.77M ± 3.14M | 2.58M ± 3.14M |

### Model mismatch sweep (horizon = 200, seed = 42)

| True/nominal tire stiffness | LQR cost | LQG cost |
|---|---|---|
| 0.5 | 2,505,321 | 506,862 |
| 0.8 | 773,409 | 1,146,035 |
| 1.0 (no mismatch) | 173,969 | 1,553,200 |
| 1.2 | 80,559 | 207,135 |
| 1.5 | 26,551 | 86,148 |

## 4. Open finding: LQG underperforms LQR

Across every experiment, **LQG's cumulative cost is substantially higher than
LQR's**, including at zero model mismatch — the opposite of the expected
result. A well-tuned LQG loop should approach LQR's performance as
measurement quality improves, and should never do dramatically worse when the
model is correct.

This is flagged here explicitly rather than smoothed over, since it changes
the conclusion the results support. Likely causes, in rough order of
suspicion:

1. **Noise covariance mismatch.** The default `Rv_diag = (0.5, 0.2)` measurement
   noise (partial 2-state observation of `e_y`, `psi`) may be too large relative
   to `Qw_diag` and to the state magnitudes the LQR gains were tuned for,
   causing the KF to lag or produce noisy estimates that the aggressive LQR
   gain then amplifies.
2. **Estimator/controller interaction under saturation.** The steering
   command is clipped to `±0.5` rad. If the KF estimate overshoots due to
   noise, the resulting control can saturate more often than under true-state
   feedback, and saturation is not accounted for in the LQR design (which
   assumes unconstrained linear control).
3. **Filter convergence not yet verified.** The estimation-error trace should
   be checked against the KF's own predicted covariance (`P_hist`) to confirm
   the filter is actually converging rather than diverging over the horizon —
   this hasn't been checked yet.

**Next step before treating any of the above numbers as a final result:**
re-run `run_comparison.py`, inspect the covariance trace and estimation-error
plot, and re-tune `Qw_diag` / `Rv_diag` (or switch to the full-state
measurement configuration via `C_mat=np.eye(4)`) to confirm the KF is well
tuned before drawing conclusions about LQR vs LQG performance.

## 5. Reproducing these results

```bash
pip install -r requirements.txt
python scripts/run_comparison.py       # Table in Section 3.1
python scripts/run_monte_carlo.py      # Table in Section 3.2
python scripts/run_model_mismatch.py   # Table in Section 3.3
python scripts/run_collocation.py      # open-loop baseline (requires casadi)
```

All experiments use `X0 = [0.6, 0.12, 0.0, 0.0]` unless otherwise noted; see
`src/lane_keep/simulate.py` for the full set of tunable parameters
(`Q`, `R`, `Qf`, `Qw_diag`, `Rv_diag`, `C_mat`, `steering_limit`).
