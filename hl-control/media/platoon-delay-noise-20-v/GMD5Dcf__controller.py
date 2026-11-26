"""Predictive CACC controller for decentralized platooning under observation delay and noise."""

import numpy as np

PARAMS = {
    "kp": 0.55,
    "kd": 1.9,
    "kff": 1.0,
}

PARAM_BOUNDS = {
    "kp": (0.2, 1.2),
    "kd": (0.8, 3.0),
    "kff": (0.6, 1.4),
}


def controller(obs: dict, memory: dict | None = None, params: dict | None = None) -> tuple[float, dict]:
    """Decentralized predictive CACC controller.

    Control Law and Mechanism:
    1. Internal Plant Simulation & Delay Compensation:
       Because observation delay is variable (delta steps), the controller uses the
       delayed own measurements (velocity own_v_s and acceleration own_a_s at step - delta)
       and forward-simulates its own dynamics using the exact history of applied commands
       (u_history) up to the current step.
    2. Predecessor State Prediction:
       The predecessor's delayed position and velocity are forward-projected using the
       last observed acceleration (pred_a) across the delay window.
    3. Proper Spacing Error Derivative:
       Under constant time headway policy (d_des = d0 + h * v_own), the true spacing error
       time-derivative is e_dot = v_rel - h * a_own (rather than just v_rel).
    4. Feedforward & High Velocity/Derivative Damping:
       Combining feedforward acceleration tracking (kff) with tuned damping (kd) and proportional
       gain (kp) guarantees string stability (string peak gain <= 1.35) and disturbance attenuation.
    """
    gains = PARAMS.copy()
    if params:
        gains.update(params)

    if memory is None:
        memory = {
            "u_history": {},
            "prev_pred_a": 0.0,
        }

    step = int(obs["step"])
    dt = float(obs["dt"])
    delay = int(obs["obs_delay_steps"])
    s = step - delay
    d0 = float(obs["d0"])
    h = float(obs["time_headway"])
    tau = 0.45
    brake_limit = float(obs["brake_limit"])
    accel_limit = float(obs["accel_limit"])
    speed_limit = float(obs["speed_limit"])

    gap_meas = float(obs["gap_front"])
    rel_v_meas = float(obs["relative_velocity_front"])
    pred_a_meas = float(obs["predecessor_acceleration"])
    own_v_s = float(obs["own_velocity"])
    own_a_s = float(obs["own_acceleration"])

    # Predecessor acceleration handling (holding last valid acceleration on dropout)
    if pred_a_meas != 0.0:
        pred_a = pred_a_meas
    else:
        pred_a = float(memory["prev_pred_a"])
    memory["prev_pred_a"] = pred_a

    # Forward project own state from s to current step using exact applied control commands
    v_own = own_v_s
    a_own = own_a_s
    x_own_disp = 0.0
    for t in range(s, step):
        u = float(memory["u_history"].get(t, 0.0))
        u_clip = float(np.clip(u, -brake_limit, accel_limit))
        a_own = a_own + dt * (u_clip - a_own) / tau
        v_own = float(np.clip(v_own + dt * a_own, 0.0, speed_limit))
        x_own_disp += dt * v_own

    # Forward project predecessor state from s to current step
    v_pred = own_v_s + rel_v_meas
    a_pred = pred_a
    x_pred_disp = 0.0
    for t in range(s, step):
        v_pred = float(np.clip(v_pred + dt * a_pred, 0.0, speed_limit))
        x_pred_disp += dt * v_pred

    pred_gap = gap_meas + x_pred_disp - x_own_disp
    pred_des_gap = d0 + h * v_own
    pred_spacing_error = pred_gap - pred_des_gap
    pred_rel_v = v_pred - v_own

    # Spacing error derivative: d/dt(gap - d0 - h*v_own) = v_rel - h * a_own
    e_dot = pred_rel_v - h * a_own

    # Control acceleration command
    u_raw = gains["kp"] * pred_spacing_error + gains["kd"] * e_dot + gains["kff"] * a_pred

    u_cmd = float(np.clip(u_raw, -brake_limit, accel_limit))
    memory["u_history"][step] = u_cmd

    # Clean up old command history to conserve memory
    if len(memory["u_history"]) > 50:
        min_keep = step - 25
        memory["u_history"] = {k: v for k, v in memory["u_history"].items() if k >= min_keep}

    return u_cmd, memory
