"""Soft continuum arm: adaptive nominal-map feedforward + slow task-space PI + camera-based active damping
through a fast inner pressure loop.

Model: the seed's arm is treated as the nominal arm driven by an effective differential command g * dp
(g = per-segment actuation gain, absorbing the stiffness scale and chamber gains). g is identified online
from steady-state consistency between the measured tip and the measured pressure differences (plant
identification, independent of the tracking error, so it does not fight the integrator).

Structure (dp = the two differential commands, one per segment; u = four chamber set points):
  dp_ff   : inverse of the nominal static map at the target (small-|dp| branch) divided by g, ramped
            linearly over `ramp` seconds when the target changes so the step does not ring the 1.25 Hz mode.
  e_f     : tip error low-passed with `err_tau` (keeps the PI from pumping the resonance).
  dp_int += ki * J^+ e_f * dt_cam       integral: creep, hysteresis, gravity mismatch; clipped to +-int_max,
                                        only allowed to unwind while a command is saturated, reset at each
                                        target change (its correction is handed to the ramp start).
  dp_fb   = kp * J^+ e_f - kd * J^+ v   P on the filtered error, D on the camera-differenced tip
                                        velocity (filter `vel_tau`) for active damping.
  p_des   = chamber_commands(dp_ff + dp_fb + dp_int, bias)   co-contraction bias keeps headroom.
  u       = p_des + kpi * (p_des - p_meas)  inner pressure loop: shortens the 0.15 s lag so the
                                        damping moment arrives in phase with the velocity.
J = J_nominal(g * dp) diag(g); J^+ is its damped pseudo-inverse: damping `lam` for the integral term, the
heavier `lam_pd` for the P and D terms so camera noise is not amplified along the weak (tip-raising) axis.
Gain identification: when the tip is quiet (|v| < `quiet_v`) and the target has been on for > `id_wait` s,
the residual r = tip - map(g * dp_meas) (dp_meas from the measured pressures) drives a damped Gauss-Newton
step on g with sensitivity A = J_nominal(g * dp_meas) diag(dp_meas):
g += `g_rate` dt (A'A + `id_mu` I)^-1 (A' r - `id_prior` (g - 1)); the prior keeps g near 1 where the data is weak.
`id_mu` keeps g still along directions the tip does not inform (the near-singular "raise the tip" axis).
Identification is skipped near saturation (|dp_meas| > `id_max_dp`, joint limits break the scaled model), and
each g update is removed from the integrator so the total command is continuous (no double correction).
"""

import numpy as np

from hl_control.envs.soft_arm import SoftArmConfig, chamber_commands, interpolate_tip, map_jacobian, static_tip_map

PARAMS = {
    # task-space PI(D) through the map Jacobian (error in m -> differential command)
    "kp": 0.1, "ki": 0.8, "kd": 0.12, "lam": 0.002, "lam_pd": 0.01,
    # filters (s), feedforward ramp (s), co-contraction bias, inner pressure-loop gain
    "vel_tau": 0.05, "err_tau": 0.15, "ramp": 0.8, "bias": 0.15, "kpi": 2.5,
    # integrator clip and gain identification
    "int_max": 0.35, "g_rate": 2.5, "quiet_v": 0.03, "id_wait": 1.0, "id_mu": 0.002, "id_max_dp": 0.8, "id_prior": 0.0005,
}
PARAM_BOUNDS = {
    "kp": (0.0, 0.5), "ki": (0.1, 2.0), "kd": (0.0, 0.4), "lam": (1e-4, 0.01), "lam_pd": (1e-3, 0.05),
    "vel_tau": (0.02, 0.2), "err_tau": (0.02, 0.5), "ramp": (0.0, 2.0), "bias": (0.05, 0.35), "kpi": (0.0, 6.0),
    "int_max": (0.1, 0.7), "g_rate": (0.1, 4.0), "quiet_v": (0.01, 0.1), "id_wait": (0.5, 4.0), "id_mu": (1e-4, 0.02),
    "id_max_dp": (0.5, 1.0), "id_prior": (0.0, 0.005),
}
G_MIN, G_MAX = 0.6, 1.7

_GRID = static_tip_map(SoftArmConfig())
_FINE = np.linspace(-1.0, 1.0, 81)
_FINE_TIPS = np.array([[interpolate_tip(_GRID, np.array([a, b])) for b in _FINE] for a in _FINE])  # (81, 81, 2)
_FINE_NORM = np.sqrt(_FINE[:, None] ** 2 + _FINE[None, :] ** 2)

_S = {"step": -1}


def _newton_refine(dp: np.ndarray, target: np.ndarray, iters: int = 3) -> np.ndarray:
    for _ in range(iters):
        J = map_jacobian(_GRID, dp, h=0.02)
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(2))
        dp = np.clip(dp + Jp @ (target - interpolate_tip(_GRID, dp)), -1.0, 1.0)
    return dp


def inverse_map(target: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """Nominal differential command whose static tip is closest to target; prefers the unsaturated branch
    (ref None) or the branch near ref."""
    d = np.linalg.norm(_FINE_TIPS - target, axis=2)
    if ref is None:
        d = d + 0.01 * _FINE_NORM
    else:
        d = d + 0.05 * np.sqrt((_FINE[:, None] - ref[0]) ** 2 + (_FINE[None, :] - ref[1]) ** 2)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    return _newton_refine(np.array([_FINE[i], _FINE[j]]), target)


def _reset():
    _S.clear()
    _S.update({"step": -1, "tidx": -1, "ff_from": np.zeros(2), "ff_to": np.zeros(2), "ff_t0": 0.0, "ff_nom": np.zeros(2),
               "dp_int": np.zeros(2), "tip_prev": None, "vel": np.zeros(2), "e_f": None, "dp": np.zeros(2),
               "dp_fb": np.zeros(2), "sat": False, "g": np.ones(2)})


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or obs["step"] <= _S["step"]:
        _reset()
    _S["step"] = obs["step"]
    t = float(obs["time"])
    target = np.asarray(obs["target_xz"], dtype=float)
    circle = obs.get("target_mode") == "circle"
    g = _S["g"]

    # --- feedforward: inverse nominal map scaled by the identified gains, ramped on target changes ---
    if obs["target_index"] != _S["tidx"] or circle:
        _S["tidx"] = obs["target_index"]
        _S["ff_nom"] = inverse_map(target)
        if circle:
            _S["ff_from"] = _S["ff_to"] = _S["ff_nom"] / g
        else:
            _S["ff_from"] = _ff_now(t) + _S["dp_int"]   # hand the integrator's correction to the ramp start
            _S["ff_to"] = _S["ff_nom"] / g
            _S["ff_t0"] = t
            _S["dp_int"] = np.zeros(2)                  # g carries the material knowledge across targets
    else:
        _S["ff_to"] = _S["ff_nom"] / g
    dp_ff = _ff_now(t)

    p_meas = np.asarray(obs["pressures"], dtype=float) / float(obs["p_max"])
    dp_meas = (p_meas[[0, 2]] - p_meas[[1, 3]]) / (1.0 - P["bias"])

    # --- feedback on fresh camera frames ---
    if obs["tip_fresh"]:
        tip = np.asarray(obs["tip_xz"], dtype=float)
        dt = float(obs["camera_dt"])
        if _S["tip_prev"] is not None:
            raw_v = (tip - _S["tip_prev"]) / dt
            a = dt / max(P["vel_tau"], dt)
            _S["vel"] = (1 - a) * _S["vel"] + a * raw_v
        _S["tip_prev"] = tip
        e = target - tip
        if _S["e_f"] is None:
            _S["e_f"] = e
        else:
            b = dt / max(P["err_tau"], dt)
            _S["e_f"] = (1 - b) * _S["e_f"] + b * e
        J = map_jacobian(_GRID, np.clip(g * _S["dp"], -1, 1)) * g[None, :]
        Jp = J.T @ np.linalg.inv(J @ J.T + P["lam"] * np.eye(2))
        Jpd = J.T @ np.linalg.inv(J @ J.T + P["lam_pd"] * np.eye(2))   # heavier damping: no noise blow-up on weak axes
        d_int = P["ki"] * dt * (Jp @ _S["e_f"])
        if _S["sat"]:  # anti-windup: only let the integrator move back out of saturation
            d_int = np.where(np.sign(d_int) == np.sign(_S["dp"]), 0.0, d_int)
        _S["dp_int"] = np.clip(_S["dp_int"] + d_int, -P["int_max"], P["int_max"])
        _S["dp_fb"] = P["kp"] * (Jpd @ _S["e_f"]) - P["kd"] * (Jpd @ _S["vel"])

        # --- gain identification (damped Gauss-Newton on g from the steady-state tip residual) ---
        if (np.linalg.norm(_S["vel"]) < P["quiet_v"] and float(obs["target_time"]) > P["id_wait"]
                and np.all(np.abs(dp_meas) < P["id_max_dp"]) and np.all(np.abs(_S["dp"]) < 0.98)):
            dp_eff = np.clip(g * dp_meas, -1, 1)
            r = tip - interpolate_tip(_GRID, dp_eff)
            A = map_jacobian(_GRID, dp_eff, h=0.02) * dp_meas[None, :]      # d tip / d g
            # damped Gauss-Newton with a weak prior pulling g toward 1 where the data is uninformative
            step = np.linalg.solve(A.T @ A + P["id_mu"] * np.eye(2), A.T @ r - P["id_prior"] * (g - 1.0))
            g_new = np.clip(g + min(1.0, P["g_rate"] * dt) * step, G_MIN, G_MAX)
            # keep the total command continuous: what g now explains leaves the integrator
            _S["dp_int"] -= _S["ff_nom"] / g_new - _S["ff_nom"] / g
            _S["g"] = g_new

    dp = dp_ff + _S["dp_fb"] + _S["dp_int"]
    _S["dp"] = np.clip(dp, -1.0, 1.0)
    _S["sat"] = bool(np.any(np.abs(dp) > 1.0))

    # --- inner pressure loop around the first-order lag ---
    p_des = chamber_commands(_S["dp"], P["bias"])
    u = p_des + P["kpi"] * (p_des - p_meas)
    return np.clip(u, 0.0, 1.0)


def _ff_now(t: float) -> np.ndarray:
    r = PARAMS["ramp"]
    s = 1.0 if r <= 0 else min(1.0, max(0.0, (t - _S["ff_t0"]) / r))
    return (1 - s) * _S["ff_from"] + s * _S["ff_to"]
