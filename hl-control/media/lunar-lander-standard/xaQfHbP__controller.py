"""Hand-written LunarLander continuous controller.

This is a continuous adaptation of the classic heuristic policy:
- Build a target tilt from lateral error and horizontal speed.
- Build a target hover height from lateral offset.
- Convert both to correction terms (`angle_todo`, `hover_todo`).
- Map those terms to continuous engines with deadbands that match actuator
  behavior (main engine only for positive command; side engines only for
  |command| > 0.5).
"""

from __future__ import annotations

import numpy as np


PARAMS = {
    "angle_from_x": 0.8858298011884678,
    "angle_from_xdot": 0.7250191895865324,
    "angle_limit": 0.33469330324511554,
    "hover_from_abs_x": 0.7393385114283076,
    "angle_kp": 0.8824951760524704,
    "angle_kd": 0.7393678273780868,
    "hover_kp": 0.29595840484808855,
    "hover_kd": 0.8024960675847508,
    "main_bias": 0.02973557828417043,
    "main_scale": 1.0509755110909336,
    "main_touch_kd": 0.5457488512066642,
    "side_scale": 4.345548445579052,
}

# Optional bounds for harbor.tune; not used directly by controller logic.
PARAM_BOUNDS = {
    "angle_from_x": (0.30, 0.90),
    "angle_from_xdot": (0.60, 1.40),
    "angle_limit": (0.25, 0.55),
    "hover_from_abs_x": (0.35, 0.75),
    "angle_kp": (0.30, 0.90),
    "angle_kd": (0.55, 1.40),
    "hover_kp": (0.25, 0.90),
    "hover_kd": (0.30, 0.90),
    "main_bias": (-0.04, 0.20),
    "main_scale": (0.8, 2.6),
    "main_touch_kd": (0.2, 0.9),
    "side_scale": (1.8, 5.0),
}


def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def controller(obs: dict) -> np.ndarray:
    x = float(obs["x"])
    y = float(obs["y"])
    x_dot = float(obs["x_dot"])
    y_dot = float(obs["y_dot"])
    angle = float(obs["angle"])
    angular_velocity = float(obs["angular_velocity"])
    left_contact = bool(obs["left_leg_contact"])
    right_contact = bool(obs["right_leg_contact"])

    p = PARAMS

    # Heuristic targets.
    angle_targ = p["angle_from_x"] * x + p["angle_from_xdot"] * x_dot
    angle_targ = _clip(angle_targ, -p["angle_limit"], p["angle_limit"])
    hover_targ = p["hover_from_abs_x"] * abs(x)

    angle_todo = (angle_targ - angle) * p["angle_kp"] - angular_velocity * p["angle_kd"]
    hover_todo = (hover_targ - y) * p["hover_kp"] - y_dot * p["hover_kd"]

    any_leg = left_contact or right_contact
    both_legs = left_contact and right_contact
    if any_leg:
        # Once touching down, keep upright and softly null vertical speed.
        angle_todo = -angular_velocity * 0.25 - angle * 0.35
        hover_todo = -y_dot * p["main_touch_kd"]
    if both_legs and abs(y_dot) < 0.08 and abs(x_dot) < 0.08 and abs(angle) < 0.08:
        return np.array([-1.0, 0.0], dtype=np.float32)

    # Map to continuous main throttle.
    main_cmd = hover_todo * p["main_scale"] + p["main_bias"]
    main_cmd = _clip(main_cmd, -1.0, 1.0)

    # Map to side thrusters with deadband to avoid chattering.
    side_raw = -angle_todo * p["side_scale"]
    if abs(side_raw) < 0.10:
        side_cmd = 0.0
    else:
        side_cmd = _clip(side_raw, -1.0, 1.0)

    return np.array([main_cmd, side_cmd], dtype=np.float32)
