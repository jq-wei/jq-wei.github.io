"""SIMBICON-style Walker2d gait controller for coarse encoders without velocity sensing.

Control law: a two-state machine, one state per stance leg. A step ends after T_STEP seconds,
or earlier when the swing foot touches down after T_MIN*T_STEP (contacts are measured exactly,
so touchdown-triggered switching keeps the clock in phase with the real gait; this alone took
the alive rate from ~60 % to ~90 %). Within a step:
  * swing hip: PD on the world-frame thigh angle (thigh - pitch) toward PHI0 + CV*(v - V_REF),
    i.e. the swing foot is placed further ahead when the walker is slower than V_REF
    (KP_SWING, KD_SWING);
  * stance hip: virtual torso actuator  -KP_TORSO*(pitch - pitch_ref) - KD_TORSO*pitch_rate,
    minus the swing-hip torque so the torso, not the swing leg, feels the correction; the
    pitch target PITCH_REF (nose-up lean, which sets the walking speed) is faded in from upright
    over the first T_RAMP seconds so the start-up does not saturate the hips;
  * swing knee: flex to KNEE_SWING for the first SWING_FLEX fraction of the step, then extend
    to KNEE_STANCE; stance knee: hold KNEE_STANCE (PD with KP_KNEE, KD_KNEE; a soft stance knee
    is essential, stiffening it makes the gait bounce and fall);
  * stance ankle: PD to ANKLE_STANCE plus a constant push-off torque ANKLE_PUSH;
    swing ankle: keep the foot level with the ground (PD with KP_ANKLE, KD_ANKLE).

Sensing: positions arrive quantised to 0.05 rad and no velocities are measured, so
  * joint and pitch rates are first-order low-pass filtered finite differences of the
    quantised angles (time constant TAU_RATE, ~2 control steps; longer lags destabilise);
  * forward speed is estimated from the exact foot positions relative to the torso: while a
    foot is on the ground its torso-relative x shrinks at the walking speed, so
    v = -d/dt(foot_x_rel) of the contact foot, low-pass filtered (time constant TAU_V).
    The first V_SKIP steps after a touchdown are ignored (the foot still slides/rolls) and
    each raw sample is clipped to within V_CLIP of the running estimate.

Gains: SIMBICON structure of the hl_control reference design, re-tuned on the coarse plant
(less nose-up lean, weaker speed feedback, softer torso and knee damping, shorter cadence).
Measured on coarse_sensing_v1, seeds 0-1535: 1535 alive and forward (~13 m in 8 s, brisk).
"""

import numpy as np

# Fixed structure constants (not searched by the tuner).
GAIT = {
    "KP_SWING": 2.662, "KD_SWING": 0.0593,
    "KNEE_STANCE": -0.13, "SWING_FLEX": 0.73,
    "ANKLE_STANCE": -0.115, "KP_ANKLE": 0.39, "KD_ANKLE": 0.1125,
    "TAU_V": 0.06, "V_SKIP": 4, "V_CLIP": 0.5,
    "T_RAMP": 0.4,  # s over which the pitch target is faded in from upright at the start
    "T_MIN": 0.5,   # fraction of T_STEP after which a swing-foot touchdown ends the step early (1.0 = never)
}

# Tunable parameters (at most 12 for hl_control.harbor.tune).
PARAMS = {
    "PITCH_REF": -0.25, "KP_TORSO": 4.0, "KD_TORSO": 0.1445,
    "PHI0": 0.4, "CV": 0.15, "V_REF": 0.3,
    "KNEE_SWING": -1.748, "KP_KNEE": 0.5, "KD_KNEE": 0.15,
    "ANKLE_PUSH": 0.452, "T_STEP": 0.28, "TAU_RATE": 0.015,
}

PARAM_BOUNDS = {
    "PITCH_REF": (-0.7, -0.1), "KP_TORSO": (2.0, 8.0), "KD_TORSO": (0.0, 0.4),
    "PHI0": (0.2, 0.8), "CV": (0.0, 0.6), "V_REF": (0.0, 1.2),
    "KNEE_SWING": (-2.5, -0.8), "KP_KNEE": (0.2, 1.5), "KD_KNEE": (0.0, 0.4),
    "ANKLE_PUSH": (0.0, 0.9), "T_STEP": (0.2, 0.45), "TAU_RATE": (0.005, 0.05),
}

_state = {}


def _reset():
    _state.update(stance=0, t=0.0, q_prev=None, dq=np.zeros(6), p_prev=None, dp=0.0,
                  v=0.0, fx_prev=[None, None], contact_steps=[0, 0])


def controller(obs: dict) -> np.ndarray:
    P = {**GAIT, **PARAMS}
    if obs["step"] == 0 or not _state:
        _reset()
    dt = float(obs["dt"])
    q = np.asarray(obs["joint_angles"], dtype=float)
    p = float(obs["pitch"])

    # --- rate estimates from quantised positions (filtered finite differences) ---
    a = dt / (P["TAU_RATE"] + dt)
    if _state["q_prev"] is not None:
        _state["dq"] = (1 - a) * _state["dq"] + a * (q - _state["q_prev"]) / dt
        _state["dp"] = (1 - a) * _state["dp"] + a * (p - _state["p_prev"]) / dt
    _state["q_prev"], _state["p_prev"] = q, p
    dq, dp = _state["dq"], _state["dp"]

    # --- speed estimate from the exact position of the contact foot ---
    feet = (obs["foot_xz_rel"], obs["foot_left_xz_rel"])
    contacts = (obs["foot_contact"], obs["foot_left_contact"])
    av = dt / (P["TAU_V"] + dt)
    vs = []
    for k in range(2):
        fx = float(feet[k][0])
        _state["contact_steps"][k] = _state["contact_steps"][k] + 1 if contacts[k] else 0
        if contacts[k] and _state["fx_prev"][k] is not None and _state["contact_steps"][k] > P["V_SKIP"]:
            vs.append(-(fx - _state["fx_prev"][k]) / dt)
        _state["fx_prev"][k] = fx if contacts[k] else None
    if vs:
        raw = float(np.clip(np.mean(vs), _state["v"] - P["V_CLIP"], _state["v"] + P["V_CLIP"]))
        _state["v"] = (1 - av) * _state["v"] + av * raw
    v = _state["v"]

    # --- gait phase ---
    _state["t"] += dt
    sw_td = contacts[1 - _state["stance"]] and _state["contact_steps"][1 - _state["stance"]] == 1
    if _state["t"] > P["T_STEP"] or (sw_td and _state["t"] > P["T_MIN"] * P["T_STEP"]):
        _state["stance"] = 1 - _state["stance"]
        _state["t"] = 0.0
    st = _state["stance"]; sw = 1 - st; t = _state["t"]
    hs, ks, as_ = 3 * st, 3 * st + 1, 3 * st + 2
    hw, kw, aw = 3 * sw, 3 * sw + 1, 3 * sw + 2
    u = np.zeros(6)

    # swing hip: world-frame thigh placement with speed feedback
    phi_d = P["PHI0"] + P["CV"] * (v - P["V_REF"])
    u[hw] = P["KP_SWING"] * (phi_d - (q[hw] - p)) - P["KD_SWING"] * (dq[hw] - dp)
    # stance hip: virtual torso actuator
    pitch_ref = P["PITCH_REF"] * min(1.0, float(obs["time"]) / max(P["T_RAMP"], 1e-6))
    tau_torso = -P["KP_TORSO"] * (p - pitch_ref) - P["KD_TORSO"] * dp
    u[hs] = tau_torso - u[hw]
    # knees
    knee_sw_target = P["KNEE_SWING"] if t < P["SWING_FLEX"] * P["T_STEP"] else P["KNEE_STANCE"]
    u[kw] = P["KP_KNEE"] * (knee_sw_target - q[kw]) - P["KD_KNEE"] * dq[kw]
    u[ks] = P["KP_KNEE"] * (P["KNEE_STANCE"] - q[ks]) - P["KD_KNEE"] * dq[ks]
    # ankles
    u[as_] = P["KP_ANKLE"] * (P["ANKLE_STANCE"] - q[as_]) - P["KD_ANKLE"] * dq[as_] + P["ANKLE_PUSH"]
    foot_level = p - q[hw] - q[kw]
    u[aw] = P["KP_ANKLE"] * (foot_level - q[aw]) - P["KD_ANKLE"] * dq[aw]
    return np.clip(u, -1.0, 1.0)
