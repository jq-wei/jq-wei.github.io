"""LunarLander heuristic controller with tunable gains.

This mirrors Gymnasium's built-in heuristic structure:
- target angle from horizontal position/velocity;
- angle PD and hover PD terms;
- leg-contact override that only damps vertical speed.
The gain dictionary is exposed for `hl_control.harbor.tune`.
"""

import numpy as np


PARAMS = {
    # Target tilt from lateral state.
    "x_to_angle": 0.7050157457971014,
    "vx_to_angle": 1.1289899892415982,
    "angle_limit": 0.5350832800664541,
    # Target altitude profile tied to horizontal offset.
    "hover_x_weight": 1.1800601799308787,
    # Attitude damping / tracking gains.
    "angle_kp": 0.6199712474708293,
    "angle_kd": 1.1799758526458755,
    # Vertical guidance gains before touchdown.
    "hover_kp": 0.44292670835687675,
    "vy_kd": 1.1342402509311351,
    # Vertical damping once either leg has contact.
    "touch_vy_kd": 0.4810698364372912,
    # Mapping from PD demands to continuous action channels.
    "main_scale": 14.398245046838458,
    "main_bias": 1.9064205293587988,
    "side_scale": 31.305755024051326
}

PARAM_BOUNDS = {
    "x_to_angle": [0.2, 1.2],
    "vx_to_angle": [0.4, 1.8],
    "angle_limit": [0.2, 0.6],
    "hover_x_weight": [0.2, 1.2],
    "angle_kp": [0.2, 1.2],
    "angle_kd": [0.2, 1.8],
    "hover_kp": [0.2, 1.2],
    "vy_kd": [0.2, 1.2],
    "touch_vy_kd": [0.2, 1.5],
    "main_scale": [8.0, 32.0],
    "main_bias": [0.2, 2.0],
    "side_scale": [8.0, 32.0],
}


def controller(obs: dict) -> np.ndarray:
    x = float(obs["x"])
    y = float(obs["y"])
    x_dot = float(obs["x_dot"])
    y_dot = float(obs["y_dot"])
    angle = float(obs["angle"])
    angular_velocity = float(obs["angular_velocity"])
    left_leg = bool(obs["left_leg_contact"])
    right_leg = bool(obs["right_leg_contact"])

    p = PARAMS
    legs_touching = left_leg or right_leg

    angle_target = np.clip(
        p["x_to_angle"] * x + p["vx_to_angle"] * x_dot,
        -p["angle_limit"],
        p["angle_limit"],
    )
    hover_target = p["hover_x_weight"] * abs(x)

    angle_todo = p["angle_kp"] * (angle_target - angle) - p["angle_kd"] * angular_velocity
    hover_todo = p["hover_kp"] * (hover_target - y) - p["vy_kd"] * y_dot

    if legs_touching:
        angle_todo = 0.0
        hover_todo = -p["touch_vy_kd"] * y_dot

    main = float(np.clip(hover_todo * p["main_scale"] - p["main_bias"], -1.0, 1.0))
    side = float(np.clip(-angle_todo * p["side_scale"], -1.0, 1.0))

    action = np.array([main, side], dtype=np.float32)
    return np.clip(action, -1.0, 1.0)
