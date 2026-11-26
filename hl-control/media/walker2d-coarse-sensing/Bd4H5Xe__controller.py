"""SIMBICON-style walker for coarse encoders with no velocity sensing.

State machine: one leg is the stance leg, the other swings; the roles swap when the swing
foot touches down (after t_min) or the swing lasts longer than t_max.

* Forward speed is rebuilt from the stance foot: its foot_xz_rel is measured exactly and the
  foot is planted, so v = -d(foot_x_rel)/dt (low-passed).  d = -foot_x_rel is how far the torso
  is ahead of the stance foot.
* Swing leg: world angle target phi = phi0 + cd*d + cv*v (foot placement), converted to a thigh
  target phi - pitch (thigh <= 0), knee flexed kn_sw for the first t_flex s then straightened.
* Stance leg: hip torque holds the torso at pitch p0 (positive hip torque reduces pitch),
  knee held straight, ankle torque ank_st plus k_pa*(pitch error).
Gains: kp_* joint stiffness in command per rad, k_pt torso pitch gain on the stance hip.
"""
import numpy as np

PARAMS = {
    "phi0": 0.127,
    "cd": 1.1396,
    "cv": -0.0237,
    "p0": 0.317,
    "k_pt": -0.9781,
    "kp_hip": 3.4455,
    "kp_knee": 2.027,
    "kp_ank": 0.353,
    "kn_sw": 0.1459,
    "t_flex": 0.343,
    "t_min": 0.0975,
    "t_max": 0.6429,
    "ank_st": -0.5052,
    "k_pa": 2.5123,
    "t_settle": 0.4,
    "kp_as": 0.0
}

_S = {}


def _reset():
    _S.update(swing=0, t0=0.0, prev=[None, None], v=0.0, last_step=-1)


def controller(obs):
    p = PARAMS
    if obs["step"] == 0 or "swing" not in _S or obs["step"] < _S["last_step"]:
        _reset()
    _S["last_step"] = obs["step"]
    t, dt = obs["time"], obs["dt"]
    if t < p["t_settle"]:
        _S["t0"] = p["t_settle"]
        return np.clip(-p["kp_hip"] * obs["joint_angles"], -1, 1)
    cont = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]
    rel = [obs["foot_xz_rel"], obs["foot_left_xz_rel"]]
    # swap on touchdown
    sw = _S["swing"]
    if (t - _S["t0"] > p["t_min"] and cont[sw]) or t - _S["t0"] > p["t_max"]:
        _S["swing"] = sw = 1 - sw
        _S["t0"] = t
    st = 1 - sw
    # velocity from the stance foot (planted)
    pv = _S["prev"][st]
    if cont[st] and pv is not None:
        _S["v"] += 0.3 * (-(rel[st][0] - pv[0]) / dt - _S["v"])
    _S["prev"] = [rel[0].copy() if cont[0] else None, rel[1].copy() if cont[1] else None]
    v = _S["v"]
    d = -rel[st][0]
    pitch = obs["pitch"]
    q = obs["joint_angles"]
    ts = t - _S["t0"]
    a = np.zeros(6)
    # swing leg
    phi = p["phi0"] + p["cd"] * d + p["cv"] * v
    th_t = min(0.0, phi - pitch)
    kn_t = -p["kn_sw"] if ts < p["t_flex"] else 0.0
    i, j = 3 * sw, 3 * st
    a[i] = p["kp_hip"] * (th_t - q[i])
    a[i + 1] = p["kp_knee"] * (kn_t - q[i + 1])
    a[i + 2] = p["kp_ank"] * (0.0 - q[i + 2])
    # stance leg
    e = pitch - p["p0"]
    a[j] = p["k_pt"] * e
    a[j + 1] = p["kp_knee"] * (-0.02 - q[j + 1])
    flat = pitch - q[j] - q[j + 1]  # ankle angle that puts the sole flat
    a[j + 2] = p["kp_as"] * (flat - q[j + 2]) + p["ank_st"] - p["k_pa"] * e
    return np.clip(a, -1, 1)
