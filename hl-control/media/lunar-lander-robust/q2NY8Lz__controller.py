"""Descent-rate-tracking PD controller for continuous LunarLander.

Robustness idea: with gravity +20% and main power -20% (scenario
``heavy_weak_main_v0``) full main thrust roughly equals weight, so fall speed
can be held but barely reduced. The controller therefore tracks an explicit,
bounded descent-rate reference and never lets fall speed exceed what the
weakest graded engine can still arrest near the ground.

Control law:

1. Attitude: tilt target ``angle_kx * x + angle_kvx * x_dot`` (clipped to
   ``angle_limit`` rad) steers toward the pad; side engine fires on angle
   error (``att_kp``) plus rate damping (``att_kd``), scaled by ``side_gain``.
   Below altitude ``flare_y`` the tilt target fades linearly to zero so the
   lander touches down level - a tilted touchdown rocks onto one leg.
2. Hover target: stay ``hover_k * |x| + hover_kv * |x_dot|`` high until
   centered and no longer drifting, so the lander translates over the pad
   before committing to the final descent (touching down while drifting can
   slide it off the pad edge and tip it over).
3. Vertical reference: desired vertical velocity
   ``vy_ref = -clip(descent_k * (y - hover_targ) + v_touch, -v_climb, v_max)``
   i.e. descend at most ``v_max``, slow to ``v_touch`` at the pad, and climb
   (up to ``v_climb``) if below the hover target while still off-center.
4. Main engine: high-gain tracking ``(vy_ref - vy) * main_gain + main_bias``;
   the negative bias keeps the engine off (Gym ignores main <= 0) unless real
   deceleration is needed, saving fuel on the nominal scenario.
5. Touchdown: once a leg touches, zero the attitude command and only damp
   remaining descent (``contact_kd``) so the lander settles without bouncing.
6. Anti-sleep nudge: if the lander rests nearly motionless on exactly one leg
   for ``nudge_wait`` steps (Box2D would put the body to sleep after ~25 still
   steps and end the episode without both leg flags set), fire the main engine
   for ``nudge_len`` steps; the tiny hop re-plants both legs. It only fires
   when level, slow and on the pad (a tilted pulse off-pad pushes the lander
   sideways), and at most ``nudge_budget`` times per episode.
"""

import numpy as np

PARAMS = {
    "angle_kx": 0.7,
    "angle_kvx": 1.0,
    "angle_limit": 0.4,
    "flare_y": 0.2,
    "hover_k": 0.55,
    "hover_kv": 0.0,
    "descent_k": 0.45,
    "v_touch": 0.10,
    "v_max": 0.30,
    "v_climb": 0.30,
    "att_kp": 0.5,
    "att_kd": 1.0,
    "contact_kd": 0.5,
    "main_gain": 20.0,
    "main_bias": -1.0,
    "side_gain": 20.0,
    "nudge_wait": 12.0,
    "nudge_len": 2.0,
    "nudge_budget": 4.0,
}

PARAM_BOUNDS = {
    "angle_kx": (0.2, 1.0),
    "angle_kvx": (0.4, 2.0),
    "angle_limit": (0.2, 0.6),
    "flare_y": (0.05, 0.5),
    "hover_k": (0.2, 1.0),
    "hover_kv": (0.0, 1.2),
    "descent_k": (0.2, 0.9),
    "v_touch": (0.02, 0.25),
    "v_max": (0.2, 0.6),
    "v_climb": (0.1, 0.6),
    "att_kp": (0.2, 1.2),
    "att_kd": (0.4, 2.0),
    "contact_kd": (0.2, 1.0),
    "main_gain": (5.0, 40.0),
    "main_bias": (-1.5, 0.0),
    "side_gain": (5.0, 40.0),
    "nudge_wait": (6.0, 20.0),
    "nudge_len": (1.0, 4.0),
    "nudge_budget": (0.0, 8.0),
}

# Episode-local state for the anti-sleep nudge (reset when obs["step"] == 0).
_state = {"still": 0, "pulse": 0, "used": 0}


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    if obs["step"] == 0:
        _state["still"] = 0
        _state["pulse"] = 0
        _state["used"] = 0

    x, y = obs["x"], obs["y"]
    vx, vy = obs["x_dot"], obs["y_dot"]
    angle, vang = obs["angle"], obs["angular_velocity"]
    left, right = obs["left_leg_contact"], obs["right_leg_contact"]
    contact = left or right

    # Anti-sleep nudge: motionless on exactly one leg -> brief main-engine hop.
    if _state["pulse"] > 0:
        _state["pulse"] -= 1
        return np.array([1.0, 0.0])
    nudge_safe = (
        abs(angle) < 0.08 and abs(x) < 0.3 and abs(vx) < 0.1
        and _state["used"] < p["nudge_budget"]
    )
    if (left != right) and abs(vy) < 0.03 and abs(vang) < 0.05 and nudge_safe:
        _state["still"] += 1
        if _state["still"] >= p["nudge_wait"]:
            _state["still"] = 0
            _state["used"] += 1
            _state["pulse"] = int(p["nudge_len"]) - 1
            return np.array([1.0, 0.0])
    else:
        _state["still"] = 0

    # Attitude: tilt toward the pad, damp rotation, flare level near ground.
    angle_targ = np.clip(
        x * p["angle_kx"] + vx * p["angle_kvx"], -p["angle_limit"], p["angle_limit"]
    )
    angle_targ *= np.clip(y / p["flare_y"], 0.0, 1.0)
    angle_todo = (angle_targ - angle) * p["att_kp"] - vang * p["att_kd"]

    # Vertical: track a bounded descent-rate reference.
    hover_targ = p["hover_k"] * abs(x) + p["hover_kv"] * abs(vx)
    vy_ref = -np.clip(
        p["descent_k"] * (y - hover_targ) + p["v_touch"],
        -p["v_climb"],
        p["v_max"],
    )
    hover_todo = vy_ref - vy

    if contact:
        angle_todo = 0.0
        hover_todo = -vy * p["contact_kd"]

    action = np.array(
        [
            hover_todo * p["main_gain"] + p["main_bias"],
            -angle_todo * p["side_gain"],
        ]
    )
    return np.clip(action, -1.0, 1.0)
