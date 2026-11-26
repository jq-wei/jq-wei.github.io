"""SIMBICON-style walking controller for the MuJoCo Walker2d.

Conventions: pitch p > 0 leans the torso forward; a hip angle q <= 0 swings the thigh
backward relative to the torso, so the thigh's forward angle in the world is (q - p).
The torso is therefore held leaned back (PITCH_REF < 0) so the swing leg can reach ahead.

Control law (two-state machine, one stance leg at a time):
  * Timing: stance and swing roles swap every T_STEP seconds (a fixed-cadence clock;
    the gait is stabilised by foot placement and torso control, not by contact events).
  * Torso: the stance hip produces the torque that holds the pitch at PITCH_REF
    (PD gains KP_TORSO, KD_TORSO) minus the swing-hip torque (virtual torso actuator).
  * Swing hip: world-frame PD (KP_SWING, KD_SWING) on the thigh forward angle
        phi_d = PHI0 + CV * (v - V_REF)
    i.e. step further ahead when moving faster than V_REF (balance via foot placement).
  * Knees: the swing knee is flexed towards KNEE_SWING for the first SWING_FLEX fraction
    of the step (foot clearance), then extended to KNEE_STANCE for touchdown; the stance
    knee is held at KNEE_STANCE (PD gains KP_KNEE, KD_KNEE).
  * Ankles: stance ankle PD to ANKLE_STANCE plus a constant ANKLE_PUSH torque (toe
    push-off that drives forward speed); swing ankle keeps the foot level in the world
    (PD gains KP_ANKLE, KD_ANKLE).

Gains were found by random search on simulated rollouts; commands are in units of
100 N m, angles in rad, speeds in m/s.  Typical result: ~12 m in 8 s without falling.
"""

import numpy as np

PARAMS = {
    "PITCH_REF": -0.368,
    "KP_TORSO": 4.766,
    "KD_TORSO": 0.1445,
    "PHI0": 0.450,
    "CV": 0.299,
    "V_REF": 0.3,
    "KP_SWING": 2.662,
    "KD_SWING": 0.0593,
    "KNEE_SWING": -1.748,
    "KNEE_STANCE": -0.130,
    "SWING_FLEX": 0.730,
    "KP_KNEE": 0.5,
    "KD_KNEE": 0.232,
    "ANKLE_STANCE": -0.115,
    "ANKLE_PUSH": 0.452,
    "KP_ANKLE": 0.390,
    "KD_ANKLE": 0.1125,
    "T_STEP": 0.3,
}

PARAM_BOUNDS = {
    "PITCH_REF": (-0.6, 0.3),
    "KP_TORSO": (0.5, 8.0),
    "KD_TORSO": (0.02, 1.0),
    "PHI0": (0.0, 0.6),
    "CV": (0.0, 0.5),
    "V_REF": (0.3, 2.5),
    "KP_SWING": (0.5, 6.0),
    "KD_SWING": (0.02, 0.6),
    "KNEE_SWING": (-1.8, -0.3),
    "KNEE_STANCE": (-0.4, 0.0),
    "SWING_FLEX": (0.2, 1.0),
    "KP_KNEE": (0.5, 6.0),
    "KD_KNEE": (0.02, 0.5),
    "ANKLE_STANCE": (-0.4, 0.4),
    "ANKLE_PUSH": (-0.5, 0.5),
    "KP_ANKLE": (0.2, 4.0),
    "KD_ANKLE": (0.0, 0.3),
    "T_STEP": (0.15, 0.6),
}

_state = {"stance": 0, "t": 0.0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0:
        _state["stance"] = 0
        _state["t"] = 0.0
    q = np.asarray(obs["joint_angles"], dtype=float)
    dq = np.asarray(obs["joint_rates"], dtype=float)
    p, dp, v = obs["pitch"], obs["pitch_rate"], obs["x_velocity"]

    _state["t"] += obs["dt"]
    if _state["t"] > P["T_STEP"]:
        _state["stance"] = 1 - _state["stance"]
        _state["t"] = 0.0
    st = _state["stance"]
    sw = 1 - st
    t = _state["t"]

    hs, ks, as_ = 3 * st, 3 * st + 1, 3 * st + 2
    hw, kw, aw = 3 * sw, 3 * sw + 1, 3 * sw + 2
    u = np.zeros(6)

    # swing hip: world-frame thigh forward angle phi = q_hip - p
    phi_d = P["PHI0"] + P["CV"] * (v - P["V_REF"])
    u[hw] = P["KP_SWING"] * (phi_d - (q[hw] - p)) - P["KD_SWING"] * (dq[hw] - dp)

    # stance hip: virtual torso actuator (positive hip torque pitches the torso forward)
    tau_torso = -P["KP_TORSO"] * (p - P["PITCH_REF"]) - P["KD_TORSO"] * dp
    u[hs] = tau_torso - u[hw]

    # knees
    knee_sw_target = P["KNEE_SWING"] if t < P["SWING_FLEX"] * P["T_STEP"] else P["KNEE_STANCE"]
    u[kw] = P["KP_KNEE"] * (knee_sw_target - q[kw]) - P["KD_KNEE"] * dq[kw]
    u[ks] = P["KP_KNEE"] * (P["KNEE_STANCE"] - q[ks]) - P["KD_KNEE"] * dq[ks]

    # ankles; foot toe-up angle in the world is q_hip + q_knee + q_ankle - p
    u[as_] = P["KP_ANKLE"] * (P["ANKLE_STANCE"] - q[as_]) - P["KD_ANKLE"] * dq[as_] + P["ANKLE_PUSH"]
    foot_level = p - q[hw] - q[kw]
    u[aw] = P["KP_ANKLE"] * (foot_level - q[aw]) - P["KD_ANKLE"] * dq[aw]

    return np.clip(u, -1.0, 1.0)
