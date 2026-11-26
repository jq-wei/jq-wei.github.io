"""Predictor-based CACC for a follower with delayed, noisy local observations.

Control law, evaluated on states *predicted to the step at which the command acts*:

    u = KP * e_hat + KV * dv_hat + KA * a_pred_hat - KS * a_own_hat

  e_hat       predicted spacing error  gap - (d0 + h * v_own)        [m]
  dv_hat      predicted relative velocity v_pred - v_own             [m/s]
  a_pred_hat  predicted predecessor acceleration (feed-forward)      [m/s^2]
  a_own_hat   reconstructed own acceleration (extra actuator damping)

Gains.  KP pulls the spacing error back (it is the term that limits string stability,
so it is small).  KV damps the relative velocity and is the main string-stability
lever.  KA is the CACC feed-forward of the predecessor acceleration; with KA = tau/h
and KS = KV*h + KA - 1 the delay-free closed loop is the textbook string-stable
velocity filter 1/(1 + h s).  KS adds damping to the first-order actuator.

Delay handling.  Every observation is `obs_delay_steps` old (plus one step until the
command acts).  The follower keeps its own command history in `memory`, so its own
velocity and acceleration at the acting step are reconstructed exactly with the known
actuator model (TAU).  The predecessor is dead-reckoned over the information age: its
acceleration is extrapolated linearly with a jerk estimated from the two most recent
valid samples (clamped to +-J_MAX), velocity and gap are integrated consistently.
Samples that arrive out of order (delay jitter) are ignored; a dropped predecessor
acceleration (reads exactly 0.0) is treated as missing, and the estimate keeps running
on the previous valid samples.  The extrapolation is a lead, so the predicted
acceleration and relative velocity pass through first-order lags (TF_A, TF_V) that cap
the high-frequency gain; otherwise irregular sampling excites 1-2 s oscillations that
grow down the string.  The constant relative-velocity sensor bias is estimated from the
mismatch between the integrated relative velocity and the change of the measured gap
(phased in over BIAS_SETTLE seconds) and subtracted.  A hard brake override protects
against very small gaps.
"""

from __future__ import annotations

TAU = 0.45          # actuator lag used for own-state reconstruction [s]
KP = 0.35           # spacing-error gain [1/s^2]
KV = 0.80           # relative-velocity gain [1/s]
KA = 0.30           # predecessor-acceleration feed-forward [-]
KS = 0.25           # own-acceleration damping [-]
J_MAX = 1.0         # clamp on the extrapolated predecessor jerk [m/s^3]
TF_A = 0.3          # low-pass time constant on the predicted predecessor acceleration [s]
TF_V = 0.2          # low-pass time constant on the predicted relative velocity [s]
BIAS_SETTLE = 6.0   # time over which the relative-velocity bias estimate is phased in [s]
EXTRAP_STEPS = 12   # cap on the extrapolation horizon of the predecessor [steps]
LEAD_STEPS = 1      # the command acts one step after the observation is taken
GAP_MARGIN = 2.0    # emergency braking below this bumper-to-bumper gap [m]


def _init_memory(obs: dict) -> dict:
    v = float(obs["own_velocity"])
    return {
        "cmds": {},                    # step -> command issued at that step
        "last_sample_step": -1,
        "a_samples": [],               # [(sample_step, a_pred)] of valid readings
        "gap": float(obs["gap_front"]),
        "v_pred": v + float(obs["relative_velocity_front"]),
        "a_pred": 0.0,
        "jerk": 0.0,
        "a_f": 0.0,
        "v_f": v + float(obs["relative_velocity_front"]),
        # relative-velocity bias estimator: integral of rel_v minus change of gap
        "bias_t0": None, "bias_gap0": 0.0, "bias_integral": 0.0,
        "bias_last": (0.0, 0.0), "rel_v_bias": 0.0,
    }


def _reconstruct_own(obs: dict, mem: dict) -> tuple[float, float, list[float]]:
    """Own (v, a) at the acting step and own velocities along the way."""
    dt = float(obs["dt"])
    n = int(obs["step"])
    d = int(obs["obs_delay_steps"])
    v = float(obs["own_velocity"])
    a = float(obs["own_acceleration"])
    v_path = []
    prev_u = float(obs["previous_command"])
    for k in range(n - d, n + LEAD_STEPS):
        u = mem["cmds"].get(k, prev_u)
        prev_u = u
        a = a + dt * (u - a) / TAU
        v = min(max(v + dt * a, 0.0), float(obs["speed_limit"]))
        v_path.append(v)
    return v, a, v_path


def _dead_reckon(gap: float, v_p: float, a_p: float, jerk: float, v_path: list[float],
                 dt: float, obs: dict) -> tuple[float, float, float]:
    """Integrate the predecessor (jerk-extrapolated) and the gap along own velocities."""
    lo, hi = -float(obs["brake_limit"]), float(obs["accel_limit"])
    for i, v_o in enumerate(v_path):
        if i < EXTRAP_STEPS:
            a_p = min(max(a_p + dt * jerk, lo), hi)
            v_p = min(max(v_p + dt * a_p, 0.0), float(obs["speed_limit"]))
        gap = gap + dt * (v_p - v_o)
    return gap, v_p, a_p


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    mem = memory if memory is not None else _init_memory(obs)
    dt = float(obs["dt"])
    n = int(obs["step"])
    d = int(obs["obs_delay_steps"])

    v_own, a_own, v_path = _reconstruct_own(obs, mem)

    sample_step = n - d
    if sample_step > mem["last_sample_step"] or memory is None:
        mem["last_sample_step"] = sample_step
        a_meas = float(obs["predecessor_acceleration"])
        if a_meas != 0.0:                           # exactly 0.0 means dropout
            mem["a_samples"] = (mem["a_samples"] + [(sample_step, a_meas)])[-2:]
        samples = mem["a_samples"]
        if len(samples) == 2 and samples[1][0] > samples[0][0]:
            jerk = (samples[1][1] - samples[0][1]) / (dt * (samples[1][0] - samples[0][0]))
            jerk = min(max(jerk, -J_MAX), J_MAX)
        else:
            jerk = 0.0
        if samples:
            # bring the latest valid acceleration sample up to this sample's time
            a_s = samples[-1][1] + jerk * dt * (sample_step - samples[-1][0])
        else:
            a_s = 0.0
        rel_v_raw = float(obs["relative_velocity_front"])
        gap_raw = float(obs["gap_front"])
        # The bias is the mismatch between the integrated relative velocity and the change
        # of the measured gap (the constant gap bias cancels in the difference).
        t_s = sample_step * dt
        if mem["bias_t0"] is None:
            mem["bias_t0"], mem["bias_gap0"] = t_s, gap_raw
        else:
            t_prev, rv_prev = mem["bias_last"]
            mem["bias_integral"] += 0.5 * (rel_v_raw + rv_prev) * (t_s - t_prev)
            elapsed = t_s - mem["bias_t0"]
            if elapsed > 1.0:
                raw_bias = (mem["bias_integral"] - (gap_raw - mem["bias_gap0"])) / elapsed
                mem["rel_v_bias"] = min(1.0, elapsed / BIAS_SETTLE) * raw_bias
        mem["bias_last"] = (t_s, rel_v_raw)
        v_s = float(obs["own_velocity"]) + rel_v_raw - mem["rel_v_bias"]
        gap, v_p, a_p = _dead_reckon(gap_raw, v_s, a_s, jerk, v_path, dt, obs)
        mem["jerk"] = jerk
    else:
        # no newer information: propagate the previous estimate by one step
        gap, v_p, a_p = _dead_reckon(mem["gap"], mem["v_pred"], mem["a_pred"], mem["jerk"],
                                     v_path[-1:], dt, obs)
    mem["gap"], mem["v_pred"], mem["a_pred"] = gap, v_p, a_p

    # lead-lag: the extrapolation above is a lead, these first-order lags cap its
    # high-frequency gain so that sampling irregularities do not excite the string
    if memory is None:
        mem["a_f"], mem["v_f"] = a_p, v_p
    mem["a_f"] += (dt / max(TF_A, dt)) * (a_p - mem["a_f"])
    mem["v_f"] += (dt / max(TF_V, dt)) * (v_p - mem["v_f"])

    e_hat = gap - (float(obs["d0"]) + float(obs["time_headway"]) * v_own)
    dv_hat = mem["v_f"] - v_own
    u = KP * e_hat + KV * dv_hat + KA * mem["a_f"] - KS * a_own

    if gap < GAP_MARGIN and dv_hat < 0.0:
        u = min(u, -float(obs["brake_limit"]))

    u = min(max(u, -float(obs["brake_limit"])), float(obs["accel_limit"]))
    mem["cmds"][n] = u
    for k in [k for k in mem["cmds"] if k < n - 12]:
        del mem["cmds"][k]
    return u, mem
