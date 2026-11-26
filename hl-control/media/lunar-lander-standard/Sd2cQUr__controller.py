"""Feedback controller for Gymnasium's continuous LunarLander."""

import numpy as np


def controller(obs: dict) -> np.ndarray:
    """Track a centered glide slope with damped attitude and descent control.

    Position/velocity gains set the desired lean toward the pad; attitude/rate
    gains drive the side jets.  The main-engine gains track a height-dependent
    descent rate, slowing the vehicle near touchdown.
    """
    x = obs["x"]
    y = obs["y"]
    vx = obs["x_dot"]
    vy = obs["y_dot"]
    angle = obs["angle"]
    omega = obs["angular_velocity"]

    target_angle = np.clip(0.5 * x + vx, -0.4, 0.4)
    target_height = 0.55 * abs(x)
    angle_error = 0.5 * (target_angle - angle) - omega
    vertical_error = 0.5 * (target_height - y) - 0.5 * vy

    if obs["left_leg_contact"] or obs["right_leg_contact"]:
        # Keep attitude damping active during touchdown.  Disabling it on the
        # first leg contact can let a shallow landing slide and tip over.
        vertical_error = -0.5 * vy

    main = 20.0 * vertical_error - 1.0
    side = -20.0 * angle_error
    return np.clip(np.array([main, side], dtype=float), -1.0, 1.0)
