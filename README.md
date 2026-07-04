# TV-LQR vs LQG: Lane-Keeping Control

A comparison of finite-horizon time-varying LQR (full-state feedback) against LQG
(Kalman-filter-based output feedback) for lane-keeping control of a nonlinear
bicycle vehicle model, with an additional open-loop optimal control baseline
computed via direct collocation.

## Overview

The plant is a nonlinear single-track (bicycle) model with a saturating tire
model (cornering stiffness decays with slip angle), linearized about a nominal
trajectory at each time step and controlled with a backward-Riccati-recursion
TV-LQR gain sequence.

Two feedback configurations are compared under **identical noise realizations**
(paired simulation, for a fair comparison):

- **LQR** — control law applied to the true state (perfect state knowledge)
- **LQG** — the same gain sequence applied to a discrete Kalman filter estimate,
  driven by noisy partial (or full) state measurements

An additional open-loop **direct collocation** solve (via CasADi/IPOPT) provides
a Pontryagin-style optimal control baseline to compare against the closed-loop
TV-LQR/LQG trajectories.

## What's compared

- **State trajectories** — true system response, `e_y` (lateral offset) vs `psi` (yaw)
- **Estimation error** — Kalman filter estimate vs. true state, per state
- **Control effort** — steering input over time
- **Cost** — instantaneous and cumulative quadratic cost `xᵀQx + uᵀRu`
- **Model mismatch** — closed-loop cost as true tire stiffness deviates from the
  stiffness assumed during linearization/design
- **Monte Carlo** — paired trials (same noise draws across LQR/LQG) summarizing
  control energy and total cost distributions
- **Lyapunov heatmap** — quadratic Lyapunov function level sets for the
  linearized closed-loop system

## Repository structure

```
TV-LQR-VS-LQG/
├── src/lane_keep/
│   ├── vehicle_model.py   # nonlinear bicycle model, linearization, discretization
│   ├── controllers.py     # TV-LQR (backward Riccati), discrete Kalman filter
│   ├── simulate.py        # paired simulation, Monte Carlo, model-mismatch sweep
│   ├── collocation.py     # CasADi direct-collocation open-loop solver
│   └── plotting.py        # trajectory, error, cost, and Lyapunov plots
├── scripts/               # runnable entry points for each experiment
├── notebooks/             # original exploratory notebook
├── results                # saved output plots
└── docs/                  # writeup.md file
```

## Installation

```bash
git clone https://github.com/saurav3103/TV-LQR-VS-LQG.git
cd TV-LQR-VS-LQG
pip install -r requirements.txt
```

`casadi` is only required for the direct collocation baseline
(`src/lane_keep/collocation.py` and `scripts/run_collocation.py`); the core
LQR/LQG comparison only needs `numpy`, `scipy`, and `matplotlib`.

## Usage

```bash
# Single paired LQR vs LQG run with trajectory/error/cost plots
python scripts/run_comparison.py

# Monte Carlo comparison across noise realizations
python scripts/run_monte_carlo.py

# Closed-loop cost sensitivity to tire-stiffness model mismatch
python scripts/run_model_mismatch.py

# Open-loop optimal control via direct collocation vs closed-loop response
python scripts/run_collocation.py
```

## Model

State vector `x = [e_y, psi, v_y, r]`:

| Symbol | Meaning |
|---|---|
| `e_y` | lateral offset from lane center (m) |
| `psi` | heading error (rad) |
| `v_y` | lateral velocity (m/s) |
| `r`   | yaw rate (rad/s) |

Control input `u = delta` (steering angle, rad), applied at fixed longitudinal
speed `v_x`. Cornering stiffness saturates with slip angle:
`C(alpha) = C0 * exp(-k|alpha|)`.

## Notes

- LQR and LQG are driven by the **same noise sequence** per trial so that any
  performance gap is attributable to the estimator, not to random variation.
- The model-mismatch experiment currently shows LQG's mean cost advantage over
  LQR narrowing at higher mismatch scales — worth a closer look before treating
  it as a clean result.

## License

MIT
