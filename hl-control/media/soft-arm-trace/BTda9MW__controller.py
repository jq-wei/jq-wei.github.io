"""Soft continuum arm tip controller: nominal inverse map + notch-filtered PI tip servo in differential-command space.

Control law (every control step, 50 Hz; the camera gives a fresh tip frame every second step):

  1. Feed-forward.  The target, led ahead along its own velocity by ``lead`` seconds (cancels the
     pressure lag and the arm's mechanical lag while the target moves on the circle), is pushed through
     the inverse of the nominal static tip map ``static_tip_map`` (Newton iterations on the bilinear
     interpolant, warm-started) giving the nominal differential commands ``dp_ff`` of the two segments.
  2. Feedback.  On every fresh camera frame the tip error ``e = target - tip`` is notch-filtered at the
     arm's first bending mode (about 1.25 Hz; the notch is wide so it covers the per-seed stiffness range
     and lets a lightly damped arm be controlled without exciting the mode), saturated at ``e_sat`` so the
     feedback only trims what the feed-forward left, mapped into differential-command space through a
     damped least-squares inverse of the map Jacobian at the applied command (the damping keeps the
     correction from walking along the map's near-singular fold direction), and drives a PI correction
     ``dp_fb = kp * J^+ e_f + dp_i``,
     ``dp_i += ki * camera_dt * J^+ e_f`` (scaled down while |e| exceeds ``i_gate`` so a large transient
     cannot wind the integrator up; also frozen against a saturated chamber and clamped to ``i_max``).
     The integrator absorbs the per-seed material (stiffness, chamber gains), the hysteresis offset, the
     creep drift, payloads and the inverse-map mismatch.
  3. Output.  ``dp = dp_ff + dp_fb`` is clipped to [-1, 1] and turned into the four chamber commands by
     ``chamber_commands(dp, bias)``; ``bias`` is the co-contraction floor that keeps both chambers of a
     pair pressurised so small corrections of either sign are possible.

PARAMS: kp proportional loop gain (dimensionless: dp per m times the map's m per dp), ki integral gain
[1/s], lead target lead [s], bias co-contraction floor [fraction of p_max], i_max integrator clamp [dp],
i_gate error radius [m] above which integration is scaled down (quadratically), e_sat saturation of the error
the feedback sees [m], vel_filter smoothing of the target-velocity
estimate, damp damping of the Jacobian pseudo-inverse [m per dp], notch_f notch centre [Hz], notch_zn /
notch_zd numerator / denominator damping of the notch (depth = zn / zd).
"""

from __future__ import annotations

import math

import numpy as np

from hl_control.envs.soft_arm import SoftArmConfig, chamber_commands, interpolate_tip, map_jacobian, static_tip_map

PARAMS = {
    "kp": 0.5,
    "ki": 2.5,
    "lead": 0.35,
    "bias": 0.08,
    "i_max": 0.6,
    "i_gate": 0.04,
    "e_sat": 0.03,
    "vel_filter": 0.3,
    "damp": 0.06,
    "notch_f": 1.25,
    "notch_zn": 0.1,
    "notch_zd": 0.7,
}
PARAM_BOUNDS = {
    "kp": (0.05, 0.8),
    "ki": (0.3, 4.0),
    "lead": (0.0, 0.8),
    "bias": (0.05, 0.3),
    "i_max": (0.2, 0.9),
    "i_gate": (0.01, 0.2),
    "e_sat": (0.01, 0.1),
    "vel_filter": (0.05, 0.8),
    "damp": (0.01, 0.1),
    "notch_f": (0.8, 1.8),
    "notch_zn": (0.02, 0.5),
    "notch_zd": (0.3, 1.5),
}

_GRID = None
_S: dict = {}


def _grid() -> dict:
    global _GRID
    if _GRID is None:
        _GRID = static_tip_map(SoftArmConfig(scenario="fixed_material_v0"))
    return _GRID


def _inverse_map(grid: dict, target: np.ndarray, dp0: np.ndarray, iters: int = 6) -> np.ndarray:
    """Newton iterations on the bilinear static map: differential commands whose nominal tip is ``target``."""
    dp = np.clip(np.asarray(dp0, dtype=float).copy(), -0.9, 0.9)
    for _ in range(iters):
        r = target - interpolate_tip(grid, dp)
        J = map_jacobian(grid, dp, h=0.05)
        try:
            step = np.linalg.solve(J + 1e-3 * np.eye(2), r)
        except np.linalg.LinAlgError:
            break
        n = np.linalg.norm(step)
        if n > 0.3:
            step *= 0.3 / n
        dp = np.clip(dp + step, -0.95, 0.95)
        if n < 1e-5:
            break
    return dp


def _notch_coeffs(f0: float, zn: float, zd: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear-transform (pre-warped) biquad of (s^2 + 2 zn w s + w^2) / (s^2 + 2 zd w s + w^2)."""
    w = 2.0 / dt * math.tan(math.pi * f0 * dt)  # pre-warped so the notch sits at f0 after discretisation
    K = 2.0 / dt
    b = np.array([K * K + 2 * zn * w * K + w * w, 2 * (w * w - K * K), K * K - 2 * zn * w * K + w * w])
    a = np.array([K * K + 2 * zd * w * K + w * w, 2 * (w * w - K * K), K * K - 2 * zd * w * K + w * w])
    return b / a[0], a / a[0]


def _reset(obs: dict) -> None:
    _S.clear()
    _S["dp_i"] = np.zeros(2)
    _S["dp_ff"] = np.zeros(2)
    _S["tgt_prev"] = np.asarray(obs["target_xz"], dtype=float).copy()
    _S["tgt_vel"] = np.zeros(2)
    _S["fb"] = np.zeros(2)
    _S["t_prev"] = float(obs["time"])
    _S["nx"] = np.zeros((2, 2))  # notch input history (x[k-1], x[k-2]) per axis
    _S["ny"] = np.zeros((2, 2))  # notch output history
    _S["notch"] = None


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if not _S or int(obs["step"]) == 0 or float(obs["time"]) < _S.get("t_prev", 0.0):
        _reset(obs)
    grid = _grid()
    dt = float(obs["dt"])
    cam_dt = float(obs.get("camera_dt", 2 * dt))
    target = np.asarray(obs["target_xz"], dtype=float)
    tip = np.asarray(obs["tip_xz"], dtype=float)

    # target velocity (filtered finite difference; a target jump is not a velocity) and lead
    v_raw = (target - _S["tgt_prev"]) / dt
    if np.linalg.norm(v_raw) > 0.3:
        v_raw = np.zeros(2)
    a = P["vel_filter"]
    _S["tgt_vel"] = (1 - a) * _S["tgt_vel"] + a * v_raw
    _S["tgt_prev"] = target.copy()
    target_led = target + P["lead"] * _S["tgt_vel"]

    # feed-forward through the nominal inverse map
    _S["dp_ff"] = _inverse_map(grid, target_led, _S["dp_ff"] if np.any(_S["dp_ff"]) else np.array([0.3, 0.3]))
    dp_ff = _S["dp_ff"]

    # feedback on fresh camera frames
    if bool(obs.get("tip_fresh", True)):
        e = target - tip
        if _S["notch"] is None:
            _S["notch"] = _notch_coeffs(P["notch_f"], P["notch_zn"], P["notch_zd"], cam_dt)
            _S["nx"][:] = e  # start the filter at rest on the current error
            _S["ny"][:] = e
        b, a_ = _S["notch"]
        nx, ny = _S["nx"], _S["ny"]
        e_f = b[0] * e + b[1] * nx[0] + b[2] * nx[1] - a_[1] * ny[0] - a_[2] * ny[1]
        nx[1], nx[0] = nx[0], e.copy()
        ny[1], ny[0] = ny[0], e_f.copy()

        # the feedback only trims: saturate the error it sees (the feed-forward does the large moves, and the
        # nominal Jacobian is only valid near the commanded point)
        n_e = float(np.linalg.norm(e_f))
        if n_e > P["e_sat"]:
            e_f = e_f * (P["e_sat"] / n_e)
        # damped least-squares inverse of the map Jacobian at the command actually applied; the damping limits
        # the gain along the map's near-singular direction (raising the tip), where a nominal-map correction
        # would otherwise walk the command along the fold toward saturation
        J = map_jacobian(grid, np.clip(dp_ff + _S["fb"], -0.75, 0.75), h=0.05)
        Jinv = J.T @ np.linalg.inv(J @ J.T + P["damp"] ** 2 * np.eye(2))
        e_dp = Jinv @ e_f
        # integrate only near the target: a large transient must not wind the integrator up
        incr = P["ki"] * cam_dt * e_dp * min(1.0, (P["i_gate"] / max(float(np.linalg.norm(e)), 1e-9)) ** 2)
        dp_total = dp_ff + P["kp"] * e_dp + _S["dp_i"]
        for k in range(2):
            if (dp_total[k] >= 0.98 and incr[k] > 0) or (dp_total[k] <= -0.98 and incr[k] < 0):
                incr[k] = 0.0
        _S["dp_i"] = np.clip(_S["dp_i"] + incr, -P["i_max"], P["i_max"])
        _S["fb"] = P["kp"] * e_dp + _S["dp_i"]

    dp = np.clip(dp_ff + _S["fb"], -1.0, 1.0)
    _S["t_prev"] = float(obs["time"])
    return chamber_commands(dp, P["bias"])
