"""Soft-arm tip controller: static-map feedforward + Jacobian-based integral feedback.

Control law
-----------
The two differential commands dp = (dp_prox, dp_dist) are mapped to the four chamber commands with
``chamber_commands(dp, BIAS)`` (the same co-contraction bias the nominal static map was built with);
beyond |dp| = 1 the opposing chamber is vented so the full pressure difference stays available.

* Feedforward: when a new target appears, dp is set to the nominal static map's inverse at the target
  (dense search of the bilinearly interpolated map in [-DP_SEARCH, DP_SEARCH]^2, nearest point; among
  near-ties the solution closest to the current dp is chosen so the arm does not jump branches).
* Feedback: on every fresh camera frame the tip error e = target - tip is pushed through a damped
  least-squares inverse of the map Jacobian at the current dp, dp += KI * camera_dt * J^+ e, i.e. an
  integrator in command space. It absorbs the unknown stiffness/gain scales, hysteresis and creep.
  An optional proportional term KP * J^+ e can be added on top of the integrator state (off: KP = 0).

Gains
-----
KI      integral gain [1/s]: convergence rate of the tip error (~1/KI s time constant); 1.0 keeps
        the loop well clear of the lightly damped 1.25 Hz mode on low-damping seeds (1.5 oscillated).
KP      proportional gain on the error (dimensionless, in units of the inverted map).
DAMP    Tikhonov damping of J^+ [m]: limits the command step where the map is nearly singular.
STEP_MAX maximum dp change per frame from the integrator.
BIAS    co-contraction bias of both chambers.
DP_SEARCH half-width of the dp box searched for the feedforward inverse.
T_FREEZE seconds after a target change during which the integrator is held (0 = off).
"""

import numpy as np

from hl_control.envs.soft_arm import SoftArmConfig, interpolate_tip, map_jacobian, static_tip_map

PARAMS = {"KI": 1.0, "KP": 0.0, "DAMP": 0.03, "STEP_MAX": 0.05, "BIAS": 0.1, "DP_SEARCH": 0.65, "T_FREEZE": 0.0}

_S = {}


def _setup():
    grid = static_tip_map(SoftArmConfig())
    lin = np.linspace(-PARAMS["DP_SEARCH"], PARAMS["DP_SEARCH"], 105)
    dd = np.array([[a, b] for a in lin for b in lin])
    tips = np.array([interpolate_tip(grid, d) for d in dd])
    _S.update(grid=grid, dd=dd, tips=tips)


def _inverse(target, dp_now):
    err = np.linalg.norm(_S["tips"] - target, axis=1)
    cand = err < err.min() + 0.004
    idx = np.flatnonzero(cand)
    j = idx[np.argmin(np.linalg.norm(_S["dd"][idx] - dp_now, axis=1))]
    return _S["dd"][j].copy()


def _pinv(J, lam):
    return J.T @ np.linalg.inv(J @ J.T + lam ** 2 * np.eye(2))


def _chambers(dp, bias):
    """chamber_commands(dp, bias) for |dp| <= 1; beyond it the opposing chamber is vented below the bias,
    so the pressure difference can reach full scale (|dp| up to 1 / (1 - bias))."""
    d = (1.0 - bias) * np.clip(dp, -1.0 / (1.0 - bias), 1.0 / (1.0 - bias))
    pos = bias + np.maximum(d, 0.0)
    neg = bias + np.maximum(-d, 0.0)
    shift = np.maximum(np.maximum(pos, neg) - 1.0, 0.0)
    pos, neg = np.clip(pos - shift, 0.0, 1.0), np.clip(neg - shift, 0.0, 1.0)
    return np.array([pos[0], neg[0], pos[1], neg[1]])


def controller(obs: dict) -> np.ndarray:
    if "grid" not in _S:
        _setup()
    if obs["step"] == 0 or "idx" not in _S:
        _S.update(idx=-1, dp=np.zeros(2), target=None)
    target = np.asarray(obs["target_xz"], float)
    if obs["target_index"] != _S["idx"]:
        _S["idx"] = obs["target_index"]
        _S["dp"] = _inverse(target, _S["dp"])
    dp = _S["dp"]
    e = target - np.asarray(obs["tip_xz"], float)
    Jp = _pinv(map_jacobian(_S["grid"], dp), PARAMS["DAMP"])
    if obs["tip_fresh"] and obs["target_time"] >= PARAMS["T_FREEZE"]:
        step = PARAMS["KI"] * obs["camera_dt"] * (Jp @ e)
        n = np.linalg.norm(step)
        if n > PARAMS["STEP_MAX"]:
            step *= PARAMS["STEP_MAX"] / n
        lim = 1.0 / (1.0 - PARAMS["BIAS"])
        dp = np.clip(dp + step, -lim, lim)
        _S["dp"] = dp
    return _chambers(dp + PARAMS["KP"] * (Jp @ e), PARAMS["BIAS"])
