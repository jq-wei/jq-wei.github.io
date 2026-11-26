"""Decentralized platooning controller with delay-compensating Kalman filter and CACC."""

from __future__ import annotations
import math
import numpy as np

# Module-level tunable parameters for grader tuning (<= 12 entries)
PARAMS = {
    "kp": 0.6340101587354854,
    "kv": 0.8361381307310738,
    "ka_ff": 0.6911383346177336,
    "ka_lag": 0.07475316888307652,
    "ki": 0.0162490826993675,
    "q_accel": 3.8256558137804513,
    "r_gap": 0.09538735262801054,
    "r_rel_v": 0.09939627295278652,
    "r_accel": 0.013381701683221125,
    "dropout_thresh": 0.700872587104266,
    "command_smoothing": 0.08020435562334681,
}

PARAM_BOUNDS = {
    "kp": (0.15, 0.70),
    "kv": (0.40, 1.60),
    "ka_ff": (0.50, 1.10),
    "ka_lag": (0.0, 0.80),
    "ki": (0.0, 0.04),
    "q_accel": (0.5, 6.0),
    "r_gap": (0.05, 0.40),
    "r_rel_v": (0.02, 0.30),
    "r_accel": (0.01, 0.15),
    "dropout_thresh": (0.15, 1.0),
    "command_smoothing": (0.0, 0.25),
}


def controller(
    obs: dict,
    memory: dict | None = None,
    params: dict | None = None,
) -> tuple[float, dict]:
    """Decentralized vehicle platooning controller under observation delay and sensor noise.

    Control Law Architecture:
      1. Kalman Filter (at observation timestamp m = step - delay):
         Tracks 4D state: [bumper gap, relative velocity, predecessor acceleration, relative velocity bias].
         Filters Gaussian measurement noise, estimates sensor bias, and handles predecessor
         accelerometer dropout (p = 0.35) via velocity innovation updates.
      2. Forward Delay Predictor across [m, step]:
         Forward-simulates the plant over the variable delay interval using exact recorded own
         commands and forward-projected predecessor state, eliminating observation latency.
      3. CACC Feedback & Feedforward:
         Applies feedforward predecessor acceleration tracking, proportional-derivative
         spacing regulation, and integral bias correction:
           a_target = ka_ff * a_pred_hat + kp * e_spacing + kv * v_rel_hat + ki * int_error
      4. Actuator Lag Inversion & Command Rate Limiting:
         Inverts first-order actuator dynamics (tau = 0.45s) and limits jerk/chatter:
           u = a_target + ka_lag * (a_target - a_own)
    """
    p = dict(PARAMS)
    if params:
        p.update(params)

    dt = float(obs["dt"])
    tau = 0.45
    step = int(obs["step"])
    delay = int(obs["obs_delay_steps"])
    m_obs = step - delay

    brake_limit = float(obs["brake_limit"])
    accel_limit = float(obs["accel_limit"])
    speed_limit = float(obs["speed_limit"])
    d0 = float(obs["d0"])
    h = float(obs["time_headway"])

    if memory is None:
        init_gap = float(obs["gap_front"])
        init_rel_v = float(obs["relative_velocity_front"])
        memory = {
            "kf_step": 0,
            # State: [gap, rel_v, pred_a, rel_v_bias]
            "x_hat": [init_gap, init_rel_v, 0.0, 0.0],
            "P": [0.5, 0.2, 1.0, 0.05],
            "meas_dict": {},
            "cmd_hist": {},
            "a_pred_hist": {},
            "prev_u": 0.0,
            "int_error": 0.0,
        }

    # Store incoming measurement indexed by its true observation step m_obs
    memory["meas_dict"][m_obs] = (
        float(obs["gap_front"]),
        float(obs["relative_velocity_front"]),
        float(obs["predecessor_acceleration"]),
        float(obs["own_acceleration"]),
        float(obs["own_velocity"]),
    )

    # Advance Kalman Filter sequentially up to m_obs
    kf_step = memory["kf_step"]
    x = list(memory["x_hat"])
    P = list(memory["P"])

    q_a = float(p["q_accel"])
    r_g = float(p["r_gap"])
    r_v = float(p["r_rel_v"])
    r_a = float(p["r_accel"])
    drop_thresh = float(p["dropout_thresh"])

    while kf_step < m_obs:
        next_step = kf_step + 1
        meas = memory["meas_dict"].get(next_step)
        own_a = meas[3] if meas else 0.0

        g, v_rel, a_pred, b_v = x
        g_pred = g + dt * v_rel + 0.5 * dt * dt * (a_pred - own_a)
        v_rel_pred = v_rel + dt * (a_pred - own_a)
        a_pred_pred = a_pred
        b_v_pred = b_v

        P_g = P[0] + dt * dt * P[1] + 1e-4 * dt
        P_v = P[1] + dt * dt * P[2] + 1e-4 * dt
        P_a = P[2] + q_a * q_a * dt
        P_bv = P[3] + 1e-6 * dt

        if meas is not None:
            z_g, z_v, z_a, _, _ = meas
            is_dropout = (z_a == 0.0) and (abs(a_pred_pred) > drop_thresh)

            # Update gap estimate
            K_g = P_g / (P_g + r_g * r_g)
            g_up = g_pred + K_g * (z_g - g_pred)
            P_g = (1.0 - K_g) * P_g

            # Update relative velocity & bias estimate
            v_meas_est = v_rel_pred + b_v_pred
            res_v = z_v - v_meas_est
            S_v = P_v + P_bv + r_v * r_v
            K_v = P_v / S_v
            K_bv = P_bv / S_v
            v_rel_up = v_rel_pred + K_v * res_v
            b_v_up = b_v_pred + K_bv * res_v
            P_v = (1.0 - K_v) * P_v
            P_bv = (1.0 - K_bv) * P_bv

            # Update predecessor acceleration estimate
            if is_dropout:
                # Accelerometer dropout: update a_pred from velocity innovation
                K_av = 0.25 * (P_a * dt) / S_v
                a_pred_up = a_pred_pred + K_av * res_v
                P_a = (1.0 - min(K_av * dt, 0.5)) * P_a
            else:
                K_a = P_a / (P_a + r_a * r_a)
                a_pred_up = a_pred_pred + K_a * (z_a - a_pred_pred)
                P_a = (1.0 - K_a) * P_a

            x = [g_up, v_rel_up, a_pred_up, b_v_up]
        else:
            x = [g_pred, v_rel_pred, a_pred_pred, b_v_pred]

        P = [P_g, P_v, P_a, P_bv]
        kf_step = next_step
        memory["a_pred_hist"][kf_step] = x[2]

    memory["kf_step"] = kf_step
    memory["x_hat"] = x
    memory["P"] = P

    # Estimated state at m_obs
    meas_latest = memory["meas_dict"][m_obs]
    sim_own_a = meas_latest[3]
    sim_own_v = meas_latest[4]
    sim_pred_a = x[2]
    sim_pred_v = sim_own_v + x[1]
    sim_gap = x[0]

    # Reconstruct predecessor command u_pred
    u_pred = float(np.clip(sim_pred_a, -brake_limit, accel_limit))

    # Forward projection across delay interval [m_obs, step]
    for s in range(m_obs, step):
        u_self = memory["cmd_hist"].get(s, memory["prev_u"])
        u_self_clipped = max(-brake_limit, min(accel_limit, u_self))

        # Integrate own vehicle
        sim_own_a += dt * (u_self_clipped - sim_own_a) / tau
        sim_own_v = max(0.0, min(speed_limit, sim_own_v + dt * sim_own_a))

        # Integrate predecessor vehicle
        sim_pred_a += dt * (u_pred - sim_pred_a) / tau
        sim_pred_v = max(0.0, min(speed_limit, sim_pred_v + dt * sim_pred_a))

        # Integrate gap
        sim_gap += dt * (sim_pred_v - sim_own_v)

    sim_rel_v = sim_pred_v - sim_own_v
    sim_des_gap = d0 + h * sim_own_v
    sim_spacing_error = sim_gap - sim_des_gap

    # Integral error with anti-windup
    int_error = memory["int_error"] + dt * sim_spacing_error
    int_error = max(-10.0, min(10.0, int_error))
    memory["int_error"] = int_error

    # CACC target acceleration
    a_target = (
        float(p["ka_ff"]) * sim_pred_a
        + float(p["kp"]) * sim_spacing_error
        + float(p["kv"]) * sim_rel_v
        + float(p["ki"]) * int_error
    )

    # Actuator lag compensation
    ka_lag = float(p["ka_lag"])
    raw_u = a_target + ka_lag * (a_target - sim_own_a)

    # Command smoothing & saturation
    smooth = float(p["command_smoothing"])
    u_cmd = (1.0 - smooth) * raw_u + smooth * memory["prev_u"]
    u_clipped = float(max(-brake_limit, min(accel_limit, u_cmd)))

    memory["prev_u"] = u_clipped
    memory["cmd_hist"][step] = u_clipped

    # Clean old buffer entries
    if step % 20 == 0:
        cutoff = step - 30
        for k in list(memory["meas_dict"].keys()):
            if k < cutoff:
                del memory["meas_dict"][k]
        for k in list(memory["cmd_hist"].keys()):
            if k < cutoff:
                del memory["cmd_hist"][k]
        for k in list(memory["a_pred_hist"].keys()):
            if k < cutoff:
                del memory["a_pred_hist"][k]

    return u_clipped, memory
