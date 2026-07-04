"""
lane_keep

TV-LQR vs LQG lane-keeping control comparison for a nonlinear bicycle model,
plus an optional CasADi direct-collocation open-loop baseline.
"""

from .vehicle_model import (
    C_f0_nom,
    C_r0_nom,
    Ts_default,
    discretize_linear,
    linearize,
    nonlinear_bicycle,
    tire_stiffness,
    wrap_angle,
)
from .controllers import discrete_kalman_predict, discrete_kalman_update, tv_lqr_gains
from .simulate import model_mismatch_experiment, monte_carlo_compare, simulate_pair

__all__ = [
    "C_f0_nom",
    "C_r0_nom",
    "Ts_default",
    "discretize_linear",
    "linearize",
    "nonlinear_bicycle",
    "tire_stiffness",
    "wrap_angle",
    "discrete_kalman_predict",
    "discrete_kalman_update",
    "tv_lqr_gains",
    "model_mismatch_experiment",
    "monte_carlo_compare",
    "simulate_pair",
]

__version__ = "0.1.0"

# collocation.py is intentionally NOT imported here since it requires the
# optional `casadi` dependency — import it directly if needed:
#   from lane_keep import collocation
