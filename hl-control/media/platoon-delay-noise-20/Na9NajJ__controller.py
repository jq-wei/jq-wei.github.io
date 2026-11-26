"""Delay-robust decentralized longitudinal controller for one platoon follower.

Control law (all local):
- Main term: filtered CACC/PD blend on spacing and relative-speed errors.
- Delay compensation: project spacing/relative-speed errors forward by the
  reported observation delay.
- Feed-forward: use predecessor acceleration with dropout-aware filtering.
- Safety override: trigger stronger braking for small gap / low TTC.
- Command shaping: adaptive smoothing + jerk limiting to avoid chatter.
"""

from __future__ import annotations


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    mem = dict(memory or {})

    dt = float(obs["dt"])
    h = float(obs["time_headway"])
    delay_steps = int(obs.get("obs_delay_steps", 0))
    vehicle_index = int(obs.get("vehicle_index", 1))
    delay_t = max(0.0, delay_steps * dt)

    accel_limit = float(obs["accel_limit"])
    brake_limit = float(obs["brake_limit"])

    # Raw delayed/noisy measurements.
    e_raw = float(obs["spacing_error"])
    rv_raw = float(obs["relative_velocity_front"])
    own_a = float(obs["own_acceleration"])
    pred_a_meas = float(obs["predecessor_acceleration"])
    gap = float(obs["gap_front"])
    own_v = float(obs["own_velocity"])

    # Slow bias cancellation under calm conditions (sensor fixed bias).
    e_bias = float(mem.get("e_bias", 0.0))
    rv_bias = float(mem.get("rv_bias", 0.0))
    calm = abs(rv_raw) < 0.4 and abs(own_a) < 0.35
    beta = 0.002 if calm else 0.0004
    e_bias = (1.0 - beta) * e_bias + beta * e_raw
    rv_bias = (1.0 - beta) * rv_bias + beta * rv_raw
    e = e_raw - e_bias
    rv = rv_raw - rv_bias

    # Delay-adaptive low-pass filtering.
    alpha_state = _clip(0.55 + 0.06 * delay_steps, 0.55, 0.9)
    e_f = alpha_state * float(mem.get("e_f", e)) + (1.0 - alpha_state) * e
    rv_f = alpha_state * float(mem.get("rv_f", rv)) + (1.0 - alpha_state) * rv

    # Predecessor acceleration dropout handling:
    # if a near-zero sample appears while history indicates active accel, hold
    # previous estimate with decay instead of trusting likely dropout.
    pred_a_prev = float(mem.get("pred_a_f", 0.0))
    likely_dropout = abs(pred_a_meas) < 0.03 and abs(pred_a_prev) > 0.35
    if likely_dropout:
        pred_a_est = 0.97 * pred_a_prev
    else:
        alpha_pred = 0.5
        pred_a_est = alpha_pred * pred_a_prev + (1.0 - alpha_pred) * pred_a_meas

    # Predict delayed state forward.
    e_pred = e_f + delay_t * (rv_f - h * own_a)
    rv_pred = rv_f + delay_t * (pred_a_est - own_a)

    # Conservative delay-aware margin: emulate a slightly larger effective headway
    # when delay/closing speed grows.
    closing = max(-rv_f, 0.0)
    delay_margin = 0.32 * own_v * delay_t + 0.4 * delay_t * closing
    e_ctrl = e_pred - delay_margin

    # Delay-robust gains (weaken proportional/derivative terms as delay grows).
    gain_scale = 1.0 / (1.0 + 0.18 * delay_steps)
    # Slightly stronger damping deeper in the platoon to suppress propagation.
    gain_scale /= 1.0 + 0.012 * max(vehicle_index - 1, 0)
    ff_scale = _clip(1.0 - 0.12 * delay_steps, 0.2, 1.0)
    kp = 0.25 * gain_scale
    kv = 1.00 * gain_scale
    ka_ff = 0.90 * ff_scale
    ka_self = 0.30

    raw = kp * e_ctrl + kv * rv_pred + ka_ff * pred_a_est - ka_self * own_a

    # Avoid aggressive throttle while re-opening very large delayed gaps.
    if e_pred > 12.0:
        raw = min(raw, 0.6 * accel_limit)

    # Safety override: braking for low TTC / low gap, with delay margin.
    ttc = gap / max(closing, 1e-3)
    gap_guard = 3.2 + 0.45 * delay_steps
    ttc_guard = 2.3 + 0.15 * delay_steps
    emergency = False
    if gap < gap_guard or ttc < ttc_guard:
        safe_brake = -0.9 * brake_limit - 0.75 * closing
        raw = min(raw, safe_brake)
        emergency = True
    if gap < 1.7:
        raw = -1.0 * brake_limit
        emergency = True
    if own_v < 0.6 and e_pred < -1.0:
        raw = min(raw, 0.0)

    # Adaptive smoothing + jerk limiter.
    prev_cmd = float(mem.get("previous_command", obs.get("previous_command", 0.0)))
    smooth = _clip(0.2 + 0.07 * delay_steps + 0.008 * max(vehicle_index - 1, 0), 0.1, 0.82)
    cmd = (1.0 - smooth) * raw + smooth * prev_cmd

    jerk_limit = 7.0 if emergency else (2.5 + 0.25 * delay_steps)
    max_delta = jerk_limit * dt
    cmd = _clip(cmd, prev_cmd - max_delta, prev_cmd + max_delta)
    cmd = _clip(cmd, -brake_limit, accel_limit)

    mem["e_bias"] = e_bias
    mem["rv_bias"] = rv_bias
    mem["e_f"] = e_f
    mem["rv_f"] = rv_f
    mem["pred_a_f"] = pred_a_est
    mem["previous_command"] = cmd
    return cmd, mem
