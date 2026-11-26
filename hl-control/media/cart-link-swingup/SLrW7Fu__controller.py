"""Hybrid energy-shaping and discrete LQR controller for cart-link swing-up on a bounded track.

Control Strategy:
1. Energy-shaping swing-up with partial feedback linearization:
   - Computes total mechanical energy of the pole relative to the upright equilibrium.
   - Applies an energy-pumping acceleration proportional to the energy error (E - E0)
     multiplied by sign(theta_dot * cos(theta)) to pump energy into the pendulum.
   - Combines energy pumping with PD cart position regulation (-k_px * x - k_dx * x_dot)
     to keep the cart strictly within track limits (|x| <= 2.4).
   - Uses exact nonlinear partial feedback linearization to translate the desired cart
     acceleration into the required actuator force F.

2. Discrete Linear Quadratic Regulator (DLQR) near upright:
   - When the pole enters the region of attraction (|theta| < 0.45 rad, |theta_dot| < 2.5 rad/s),
     the controller switches to a discrete-time LQR state-feedback gain K:
     u = -K * [x, x_dot, theta, theta_dot]^T.
   - DLQR stabilizes the cart at x = 0 and the pole at theta = 0 with high precision
     and zero steady-state error.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm, solve_discrete_are

# Physical plant parameters (matching CartLinkConfig)
M = 1.0  # Cart mass (kg)
m = 0.1  # Pole mass (kg)
l = 0.5  # Pole half-length (m)
g = 9.81  # Acceleration of gravity (m/s^2)
bc = 0.02  # Cart damping coefficient
bp = 0.01  # Pole damping coefficient
Mt = M + m  # Total mass
ml = m * l  # Mass-length product
J = (4.0 / 3.0) * m * l**2  # Pole moment of inertia about pivot (kg*m^2)
E0 = 0.0  # Upright equilibrium energy reference
sample_dt = 0.02  # Sample time step (s)
force_limit = 10.0  # Maximum actuator force (N)

# Gains for energy swing-up and track bounding
K_E = 18.0  # Energy pumping gain
K_PX = 8.0  # Cart position proportional centering gain
K_DX = 3.5  # Cart velocity derivative damping gain
A_MAX = 8.5  # Maximum desired cart acceleration during swing-up (m/s^2)

# Catch region thresholds for switching to LQR
CATCH_ANGLE = 0.45  # Angle threshold from upright (|wrap_angle(theta)| < CATCH_ANGLE)
CATCH_VELOCITY = 2.5  # Angular velocity threshold (|theta_dot| < CATCH_VELOCITY)


def _compute_dlqr_gain() -> np.ndarray:
    """Compute the discrete-time LQR gain matrix around the upright equilibrium."""
    D0 = l * (4.0 / 3.0 - m / Mt)
    A = np.zeros((4, 4))
    A[0, 1] = 1.0
    A[1, 1] = -bc / Mt * (1.0 + ml / (Mt * D0))
    A[1, 2] = -(ml / Mt) * g / D0
    A[1, 3] = (ml / Mt) * bp / D0
    A[2, 3] = 1.0
    A[3, 1] = bc / (Mt * D0)
    A[3, 2] = g / D0
    A[3, 3] = -bp / D0

    B = np.zeros((4, 1))
    B[1, 0] = (1.0 + ml / (Mt * D0)) / Mt
    B[3, 0] = -1.0 / (Mt * D0)

    # Discretize continuous system matrices using matrix exponential
    M_mat = np.zeros((5, 5))
    M_mat[:4, :4] = A
    M_mat[:4, 4:] = B
    M_exp = expm(M_mat * sample_dt)
    Ad = M_exp[:4, :4]
    Bd = M_exp[:4, 4:]

    # State penalty Q: weights on [x, x_dot, theta, theta_dot]
    Q = np.diag([40.0, 8.0, 150.0, 20.0])
    # Control effort penalty R
    R = np.array([[0.005]])

    P = solve_discrete_are(Ad, Bd, Q, R)
    k_vec = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
    return k_vec.reshape(-1)


# Precompute DLQR gain vector K = [k_x, k_xdot, k_theta, k_thetadot]
K_LQR = _compute_dlqr_gain()


def _force_for_desired_xacc(x_dot: float, theta: float, theta_dot: float, a_des: float) -> float:
    """Exact partial feedback linearization: compute actuator force F to achieve cart acceleration a_des."""
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    D = l * (4.0 / 3.0 - (m * cos_th**2) / Mt)
    alpha = ml * cos_th / (Mt * D)
    temp = (a_des + alpha * (g * sin_th - bp * theta_dot)) / (1.0 + alpha * cos_th)
    F = Mt * temp + bc * x_dot - ml * theta_dot**2 * sin_th
    return F


def controller(obs: dict) -> np.ndarray:
    """Compute the horizontal force action for the cart-link system.

    Args:
        obs: Observation dictionary containing state measurements:
             x, x_dot, theta, theta_dot, sin_theta, cos_theta, force_limit, etc.

    Returns:
        np.ndarray of shape (1,) with horizontal force in [-force_limit, force_limit].
    """
    x = float(obs["x"])
    x_dot = float(obs["x_dot"])
    theta_dot = float(obs["theta_dot"])
    # Map theta into [-pi, pi] where 0 is upright and pi / -pi is hanging downward
    theta = float(np.arctan2(obs["sin_theta"], obs["cos_theta"]))
    f_lim = float(obs.get("force_limit", force_limit))

    # Mode 1: DLQR Catch and Upright Regulation
    if abs(theta) < CATCH_ANGLE and abs(theta_dot) < CATCH_VELOCITY:
        state = np.array([x, x_dot, theta, theta_dot], dtype=float)
        force = -float(K_LQR @ state)
        return np.array([np.clip(force, -f_lim, f_lim)], dtype=float)

    # Mode 2: Energy-Shaping Swing-Up with Track Centering
    # Pole energy: kinetic + potential (E = 0 at upright theta=0, theta_dot=0)
    E = 0.5 * J * theta_dot**2 + m * g * l * (np.cos(theta) - 1.0)
    delta_E = E - E0

    # Direction of energy injection: sign(theta_dot * cos(theta))
    # When near bottom with zero velocity, use sin(theta) sign to break initial symmetry
    direction = theta_dot * np.cos(theta)
    if abs(direction) < 1e-3:
        direction = 1.0 if np.sin(theta) >= 0 else -1.0
    else:
        direction = float(np.sign(direction))

    # Pumping acceleration and track centering PD terms
    a_pump = K_E * delta_E * direction
    a_des = a_pump - K_PX * x - K_DX * x_dot
    a_des = float(np.clip(a_des, -A_MAX, A_MAX))

    # Exact partial feedback linearization to compute actuator force
    force = _force_for_desired_xacc(x_dot, theta, theta_dot, a_des)
    return np.array([np.clip(force, -f_lim, f_lim)], dtype=float)
