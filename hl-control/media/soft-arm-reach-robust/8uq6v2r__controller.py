"""Soft-arm tip controller: static-map feedforward + Jacobian integral feedback in differential-command space.

Control law:
  * On a new target, the nominal static map (``static_tip_map``) is inverted by a dense grid search over the
    differential commands dp in [-DP_SEARCH, DP_SEARCH]^2 (the unfolded branch, smallest |dp| among near-equal
    fits): dp_nom. The feedforward is dp_ff = ratio * dp_nom, where ``ratio`` (per segment and per side) is this
    seed's converged command over the nominal one, learnt at the end of each well-held target (|dp_nom| >=
    LEARN_MIN, blend weight >= LEARN_W). It absorbs stiffness, chamber gains, the weak chamber and the payload.
  * An integral state dp_i (reset on each target) is updated on each fresh camera frame with a damped-least-squares
    step  dp_i += KI * camera_dt * J^T (J J^T + LAMBDA^2 I)^-1 (target - tip_f),  J the map Jacobian at the
    "effective nominal" command dp / ratio (clipped inside the grid so a column never vanishes), scaled by 1/ratio.
    It removes the residual mismatch and tracks hysteresis, creep, warm-up drift and a hooked payload. Integration
    pauses while the rate limiter is still ramping (no windup during the transit); dp_ff + dp_i is clipped to [-1, 1].
  * dp is rate-limited (RATE per second, keeps the 1.25 Hz bending mode quiet) and turned into chamber commands
    with ``chamber_commands(dp, BIAS)``.
Gains: KI integral rate [1/s] (higher excites the bending mode); LAMBDA DLS damping [m] (robustness in the nearly
singular directions); TIP_LP camera smoothing per frame [0..1]; RATE command slew [1/s]; BIAS co-contraction
(0.1, as the map was built with).
"""

import numpy as np

from hl_control.envs.soft_arm import SoftArmConfig, chamber_commands, static_tip_map

PARAMS = {"KI": 1.0, "LAMBDA": 0.1, "TIP_LP": 0.6, "RATE": 1.5, "BIAS": 0.1, "LEARN_MIN": 0.15, "LEARN_W": 0.5, "DP_SEARCH": 0.6}

_GRID = static_tip_map(SoftArmConfig())


def _tip_batch(dp):
    """Vectorised bilinear interpolation of the static map, dp (N, 2) -> tip (N, 2)."""
    dps, tip = _GRID["dp"], _GRID["tip"]
    d = np.clip(dp, dps[0], dps[-1])
    x = (d - dps[0]) / (dps[1] - dps[0])
    i = np.clip(np.floor(x).astype(int), 0, len(dps) - 2)
    f = (x - i)[:, :, None]
    a, b = i[:, 0], i[:, 1]
    return ((1 - f[:, 0]) * (1 - f[:, 1]) * tip[a, b] + f[:, 0] * (1 - f[:, 1]) * tip[a + 1, b]
            + (1 - f[:, 0]) * f[:, 1] * tip[a, b + 1] + f[:, 0] * f[:, 1] * tip[a + 1, b + 1])


_S = np.linspace(-1.0, 1.0, 201)
_CAND = np.stack(np.meshgrid(_S, _S, indexing="ij"), -1).reshape(-1, 2)
_CAND_TIP = _tip_batch(_CAND)


def _invert(target):
    lim = PARAMS["DP_SEARCH"]
    m = np.all(np.abs(_CAND) <= lim + 1e-9, axis=1)
    c, t = _CAND[m], _CAND_TIP[m]
    err = np.linalg.norm(t - target, axis=1)
    cost = err + 0.002 * np.linalg.norm(c, axis=1)
    return c[np.argmin(cost)].copy()


def _jac(dp, h=0.05):
    pts = np.array([dp + [h, 0], dp - [h, 0], dp + [0, h], dp - [0, h]])
    t = _tip_batch(pts)
    return np.stack([(t[0] - t[1]) / (2 * h), (t[2] - t[3]) / (2 * h)], axis=1)


_st = {}


def _reset():
    _st.clear()
    _st.update(index=-1, dp_nom=np.zeros(2), dp_ff=np.zeros(2), dp_i=np.zeros(2), dp=np.zeros(2), tip=None,
               ratio=np.ones((2, 2)), nratio=np.zeros((2, 2)), err=1.0)


def _ratio(dp):
    """Learnt command ratio per segment for the side (sign) of dp."""
    return np.where(dp >= 0, _st["ratio"][:, 0], _st["ratio"][:, 1])


def _learn():
    """At the end of a target: the converged command over the nominal one is this seed's gain ratio."""
    if _st["err"] > 0.006:
        return
    tot = _st["dp_ff"] + _st["dp_i"]
    for j in range(2):
        n = _st["dp_nom"][j]
        if abs(n) < PARAMS["LEARN_MIN"] or tot[j] * n <= 0:
            continue
        side = 0 if n >= 0 else 1
        r = float(np.clip(tot[j] / n, 0.6, 1.8))
        w = max(PARAMS["LEARN_W"], 1.0 / (1.0 + _st["nratio"][j, side]))
        _st["ratio"][j, side] += w * (r - _st["ratio"][j, side])
        _st["nratio"][j, side] += 1


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or not _st:
        _reset()
    target = np.asarray(obs["target_xz"], float)
    dt = obs["dt"]
    if obs["target_index"] != _st["index"] or obs["target_mode"] == "circle":
        if obs["target_index"] != _st["index"] and _st["index"] >= 0:
            _learn()
            _st["dp_i"] *= 0.0
        _st["index"] = obs["target_index"]
        _st["dp_nom"] = _invert(target)
        _st["dp_ff"] = np.clip(_st["dp_nom"] * _ratio(_st["dp_nom"]), -1.0, 1.0)
    tip = np.asarray(obs["tip_xz"], float)
    if _st["tip"] is None:
        _st["tip"] = tip.copy()
    if obs["tip_fresh"]:
        _st["tip"] += P["TIP_LP"] * (tip - _st["tip"])
        e = target - _st["tip"]
        _st["err"] = float(np.linalg.norm(e))
        r = _ratio(_st["dp"])
        J = _jac(np.clip(_st["dp"] / r, -0.9, 0.9)) / r[None, :]
        step = J.T @ np.linalg.solve(J @ J.T + P["LAMBDA"] ** 2 * np.eye(2), e)
        ramping = np.max(np.abs(_st["dp_ff"] + _st["dp_i"] - _st["dp"])) > P["RATE"] * dt
        if not ramping:
            _st["dp_i"] += P["KI"] * obs["camera_dt"] * step
        _st["dp_i"] = np.clip(_st["dp_ff"] + _st["dp_i"], -1.0, 1.0) - _st["dp_ff"]  # anti-windup
    want = _st["dp_ff"] + _st["dp_i"]
    _st["dp"] += np.clip(want - _st["dp"], -P["RATE"] * dt, P["RATE"] * dt)
    return chamber_commands(_st["dp"], P["BIAS"])
