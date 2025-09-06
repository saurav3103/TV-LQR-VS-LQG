# TV-LQR vs LQG Comparison with Model Mismatch & Lyapunov Analysis

This repository contains a Python implementation for comparing **Time-Varying LQR (TV-LQR)** and **LQG (Kalman Filter + TV-LQR)** control strategies on a nonlinear bicycle vehicle model. It includes simulations under **model mismatch**, visualizations of **state trajectories**, **Kalman filter estimation errors**, **cumulative costs**, and a **Lyapunov heatmap** for stability analysis.

---

## Features

- **Nonlinear bicycle model** with lateral error, heading error, lateral velocity, and yaw rate.
- **Linearization and discretization** utilities for TV-LQR design.
- **Time-Varying LQR (TV-LQR)** implemented using backward Riccati recursion.
- **Discrete Kalman Filter** for state estimation in LQG control.
- **Paired simulations** to compare LQR (true states) vs LQG (KF-estimated states).
- **Model mismatch experiments** by scaling tire stiffness.
- **Lyapunov heatmap** for visualizing stability of linearized system.
- Visualization:
  - State trajectory comparison
  - Kalman Filter estimation error
  - Cumulative cost over time
  - Model mismatch effect
  - Lyapunov function heatmap

---

## Requirements

- Python 3.8+
- `numpy`
- `scipy`
- `matplotlib`

Install dependencies via pip:

```bash
pip install numpy scipy matplotlib
---

## Usage

Clone this repository and run the main script:

```bash
python tv_lqr_vs_lqg.py
```

The script performs the following:

1. Runs a single simulation comparing TV-LQR vs LQG.
2. Plots state trajectories (`e_y` vs `psi`).
3. Plots Kalman Filter estimation errors.
4. Plots cumulative cost over time.
5. Runs a **model mismatch experiment** by scaling tire stiffness.
6. Generates a **Lyapunov heatmap** for the linearized system.

---

## Code Overview

* `nonlinear_bicycle(x, u)`: Computes nonlinear bicycle dynamics.
* `linearize(x, u)`: Linearizes the nonlinear model around a state-input pair.
* `discretize_linear(Ac, Bc, Ts)`: Discretizes the linear system using matrix exponential.
* `tv_lqr_gains(A_seq, B_seq, Q, R, Qf)`: Computes time-varying LQR gains.
* `discrete_kalman_predict` / `discrete_kalman_update`: Implements discrete Kalman filter for LQG.
* `simulate_pair(x0, horizon, ...)`: Runs paired LQR vs LQG simulations.
* `plot_state_trajectory_pair`, `plot_estimation_error`: Visualization utilities.
* `model_mismatch_experiment`: Evaluates LQR vs LQG performance under model uncertainty.
* `lyapunov_heatmap`: Plots the Lyapunov function of the linearized system.

---

## Example Results

* LQG control can **better handle model mismatch** than standard LQR due to state estimation.
  <img width="600" height="470" alt="image" src="https://github.com/user-attachments/assets/5375fc31-ca3f-4555-9e4a-af0563a33a7f" />
* State trajectory plots compare LQR (true-state) vs LQG (KF-estimated) paths.
  <img width="689" height="547" alt="image" src="https://github.com/user-attachments/assets/b387b021-a75f-4268-a613-fe62dacddacd" />
* Estimation error plots show the Kalman filter performance.
  <img width="600" height="470" alt="image" src="https://github.com/user-attachments/assets/908d3fe1-e214-4c81-bb44-6566a2567a7d" />
* Lyapunov heatmap visualizes stability regions for the linearized system.
  <img width="553" height="470" alt="image" src="https://github.com/user-attachments/assets/164c3852-e5fe-42bf-9af7-c96ea0ee6e6b" />

---

## References

* Kwakernaak, H., & Sivan, R. (1972). *Linear Optimal Control Systems*. Wiley.
* Ogata, K. (2010). *Modern Control Engineering*. Prentice Hall.
* Rajamani, R. (2011). *Vehicle Dynamics and Control*. Springer.

---

## License

MIT License

---

**Author:** Saurav Avachat
**Email:** saurav310304@gmail.com

