"""Decentralized ACC/CACC hybrid for homogeneous platoons.

Control law (per follower):
    u = kp * e + kv * (v_front - v_own) + ka_ff * a_front - ka_self * a_own

where e is spacing error (gap - desired_gap). A light low-pass filter and
delay-based extrapolation reduce sensitivity to observation delay in later
stages. An emergency layer caps the command when gap or time-to-collision
is small. Output is smoothed and rate-limited before saturation.

Gains:
    kp       — spacing-error feedback (restores bumper gap to d0 + h*v)
    kv       — relative-velocity damping (reduces closing speed)
    ka_ff    — predecessor acceleration feedforward (weak; robust to dropout)
    ka_self  — cancels own actuator lag via measured acceleration
    smoothing — exponential filter on commanded acceleration
    jerk_limit — limits |du/dt| for ride comfort and string stability
"""

from __future__ import annotations

import math
from typing import Any


# --- tunable gains (ACC-like baseline, mild CACC feedforward) ---
KP = 0.22
KV = 0.82
KA_FF = 0.15
KA_SELF = 0.25
OBS_LPF = 0.25
DELAY_COMP = 0.85
BIAS_ALPHA = 0.008
COMMAND_SMOOTHING = 0.22
JERK_LIMIT = 2.4

# Emergency braking thresholds
GAP_MARGIN = 4.0
TTC_MARGIN = 2.8
EMERGENCY_BRAKE_FRAC = 0.92
CLOSING_BRAKE_GAIN = 0.6


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _smooth_and_clip(
    raw: float,
    obs: dict,
    memory: dict,
    smoothing: float,
    jerk_limit: float | None,
) -> tuple[float, dict]:
    previous = float(memory.get("previous_command", obs.get("previous_command", 0.0)))
    alpha = _clip(smoothing, 0.0, 0.95)
    command = (1.0 - alpha) * raw + alpha * previous
    if jerk_limit is not None and math.isfinite(jerk_limit):
        max_delta = jerk_limit * float(obs["dt"])
        command = _clip(command, previous - max_delta, previous + max_delta)
    command = _clip(command, -float(obs["brake_limit"]), float(obs["accel_limit"]))
    memory["previous_command"] = command
    return command, memory


def controller(
    obs: dict,
    memory: dict | None = None,
) -> tuple[float, dict]:
    mem: dict[str, Any] = dict(memory or {})
    dt = float(obs["dt"])
    delay_steps = int(obs.get("obs_delay_steps", 0))

    spacing_error = float(obs["spacing_error"])
    rel_v = float(obs["relative_velocity_front"])
    own_accel = float(obs["own_acceleration"])
    pred_accel = float(obs["predecessor_acceleration"])
    gap = float(obs["gap_front"])
    brake_limit = float(obs["brake_limit"])

    # Slow bias estimator (helps fixed per-vehicle sensor offset in hard scenarios).
    bias = float(mem.get("gap_bias_est", 0.0))
    bias = (1.0 - BIAS_ALPHA) * bias + BIAS_ALPHA * spacing_error
    mem["gap_bias_est"] = bias
    e = spacing_error - bias

    e_f = (1.0 - OBS_LPF) * e + OBS_LPF * float(mem.get("e_f", e))
    rv_f = (1.0 - OBS_LPF) * rel_v + OBS_LPF * float(mem.get("rv_f", rel_v))
    mem["e_f"], mem["rv_f"] = e_f, rv_f

    # Extrapolate spacing error over known observation delay.
    delay_t = delay_steps * dt
    e_comp = e_f + DELAY_COMP * rv_f * delay_t
    rv_comp = rv_f

    raw = (
        KP * e_comp
        + KV * rv_comp
        + KA_FF * pred_accel
        - KA_SELF * own_accel
    )

    closing_speed = max(-rv_comp, 0.0)
    ttc = gap / max(closing_speed, 1e-6)
    emergency = gap < GAP_MARGIN or ttc < TTC_MARGIN
    if emergency:
        emergency_cmd = -EMERGENCY_BRAKE_FRAC * brake_limit - CLOSING_BRAKE_GAIN * closing_speed
        raw = min(raw, emergency_cmd)

    smoothing = 0.0 if emergency else COMMAND_SMOOTHING
    jerk = None if emergency else JERK_LIMIT
    command, mem = _smooth_and_clip(raw, obs, mem, smoothing, jerk)
    return command, mem
