"""Delay-robust decentralized CTH controller for noisy platoon observations.

The command combines:
- predicted spacing-error and relative-velocity feedback (kp, kv);
- predecessor-acceleration feedforward with dropout hold/recovery (ka_ff);
- own-acceleration damping and mild cruise recovery (ka_self, k_cruise);
- delay-aware safety shaping and TTC/gap brake overrides.

Braking is allowed to react quickly, while acceleration is smoothed to limit
string amplification under delayed/noisy sensing.
"""


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    kp = 0.304
    kv = 0.622
    ka_ff = 0.214
    ka_self = 0.422
    k_cruise = 0.165

    meas_alpha = 0.45
    pred_acc_alpha = 0.104
    cmd_alpha_accel = 0.695
    cmd_alpha_brake = 0.624

    dt = float(obs["dt"])
    h = float(obs["time_headway"])
    d0 = float(obs["d0"])
    accel_limit = float(obs["accel_limit"])
    brake_limit = float(obs["brake_limit"])

    if memory is None:
        memory = {
            "gap_f": float(obs["gap_front"]),
            "rel_v_f": float(obs["relative_velocity_front"]),
            "pred_a_f": float(obs["predecessor_acceleration"]),
            "cmd_prev": float(obs["previous_command"]),
        }

    # Filter noisy delayed measurements.
    gap_f = (1.0 - meas_alpha) * memory["gap_f"] + meas_alpha * float(obs["gap_front"])
    rel_v_f = (1.0 - meas_alpha) * memory["rel_v_f"] + meas_alpha * float(obs["relative_velocity_front"])

    # Acceleration dropout handling: if the sensor reads near zero, retain a decaying
    # estimate from history instead of trusting the instantaneous sample.
    pred_a_meas = float(obs["predecessor_acceleration"])
    if abs(pred_a_meas) < 0.057 and abs(memory["pred_a_f"]) > 0.40:
        pred_a_used = 0.91 * memory["pred_a_f"]
    else:
        pred_a_used = pred_a_meas
    pred_a_f = (1.0 - pred_acc_alpha) * memory["pred_a_f"] + pred_acc_alpha * pred_a_used
    pred_a_f = _clip(pred_a_f, -brake_limit, accel_limit)

    own_v_delayed = float(obs["own_velocity"])
    own_a = float(obs["own_acceleration"])
    delay = max(0.0, float(obs.get("obs_delay_steps", 0.0))) * dt

    # Predict current relative state from delayed measurements.
    rel_v_now = rel_v_f + delay * (pred_a_f - own_a)
    gap_now = gap_f + delay * rel_v_f + 0.5 * delay * delay * (pred_a_f - own_a)
    own_v_now = max(0.0, own_v_delayed + delay * own_a)
    closing_speed = max(-rel_v_now, 0.0)
    delay_margin = 0.103 * own_v_now * delay + 0.313 * closing_speed * delay
    desired_gap = d0 + h * own_v_now + delay_margin
    spacing_error = gap_now - desired_gap

    raw = (
        kp * spacing_error
        + kv * rel_v_now
        + ka_ff * pred_a_f
        - ka_self * own_a
        + k_cruise * (20.0 - own_v_now)
    )

    # Extra damping when closing aggressively.
    if closing_speed > 0.5:
        raw -= 0.358 * closing_speed

    ttc = gap_now / max(closing_speed, 1e-3)
    emergency = False
    if gap_now < 2.19 or ttc < 1.56:
        raw = min(raw, -1.00 * brake_limit - 0.58 * closing_speed)
        emergency = True
    elif gap_now < 3.05 or ttc < 1.95:
        raw = min(raw, -0.74 * brake_limit - 0.45 * closing_speed)
    elif gap_now < 3.2 or ttc < 2.96:
        raw = min(raw, -0.67 * brake_limit - 0.13 * closing_speed)

    raw = _clip(raw, -brake_limit, accel_limit)

    prev_cmd = float(memory["cmd_prev"])
    if emergency:
        cmd = raw
    else:
        alpha = cmd_alpha_brake if raw < prev_cmd else cmd_alpha_accel
        cmd = prev_cmd + alpha * (raw - prev_cmd)

    cmd = _clip(cmd, -brake_limit, accel_limit)

    memory["gap_f"] = gap_f
    memory["rel_v_f"] = rel_v_f
    memory["pred_a_f"] = pred_a_f
    memory["cmd_prev"] = cmd
    return cmd, memory
