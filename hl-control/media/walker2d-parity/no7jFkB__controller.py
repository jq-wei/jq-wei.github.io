"""SIMBICON-style running controller for Walker2d (hand-structured state machine, gains found by
derivative-free CMA-ES search over the parameters below on simulated rollouts; no learning model).

State machine: the legs swap stance/swing every T_STEP s (scaled 1 +- ASYM_T per leg), or earlier
when the swing foot touches down after SW_MIN * T_STEP.  s = t / T_STEP is the phase in a step.

  * swing hip   : thigh angle (hip - pitch) driven to phi_d = PHI0 + CV (v - V_REF) (+- ASYM_PHI);
                  PD gains KP_SWING / KD_SWING.  CV is the SIMBICON speed feedback (faster -> step further).
  * stance hip  : virtual torso actuator  -KP_TORSO (pitch - pitch_ref) - KD_TORSO pitch_rate, minus the
                  swing-hip torque (reaction cancel), plus a constant HIP_DRIVE;
                  pitch_ref = PITCH_REF + KV_PITCH (v - V_REF) (+- ASYM_PITCH): the torso leans back so the
                  hip, which only flexes forward of the torso, can still sweep the stance leg behind.
  * swing knee  : fold to KNEE_SWING for SWING_FLEX * T_STEP, then KNEE_EXT (PD KP_KNEE / KD_KNEE).
  * stance knee : PD to KNEE_STANCE (KP_KNEE_ST / KD_KNEE_ST) plus extension thrust KNEE_THRUST after
                  THRUST_START * T_STEP.
  * stance ankle: PD to ANKLE_STANCE (KP_ANKLE / KD_ANKLE) plus push-off ANKLE_PUSH (+- ASYM_PUSH) after
                  PUSH_START * T_STEP.
  * swing ankle : keep the foot level (target pitch - hip - knee + ANKLE_SW).
  * height guard: when the torso is below H_REF the stance knee and hip get KH_KNEE, KH_HIP times the
                  height deficit (keeps the crouched gait above the 0.8 m termination height).
  * feed-forward: half-sine/cosine phase profiles FF_*_S sin(pi s) + FF_*_C cos(pi s) added to the stance
                  hip (HS), swing hip (HW) and stance knee (KS) commands.
Commands are clipped to [-1, 1] (100 N m).
"""

import numpy as np

PARAMS = {
    "PITCH_REF": -0.693661138911969,
    "KP_TORSO": 3.901365436017863,
    "KD_TORSO": 0.2125473454914104,
    "PHI0": 0.8184618947376964,
    "CV": 0.3232055284746088,
    "V_REF": 0.6981956682820833,
    "KP_SWING": 1.2934351784351237,
    "KD_SWING": 0.10042820213256125,
    "KNEE_SWING": -3.9437477181329275,
    "KNEE_STANCE": -0.08152641169652967,
    "KNEE_EXT": -0.11130767793754173,
    "SWING_FLEX": 1.355827965657081,
    "KP_KNEE": 0.3446031857696411,
    "KD_KNEE": 0.12273846737088936,
    "ANKLE_STANCE": -0.14138562729948345,
    "ANKLE_PUSH": 0.5037789552336973,
    "KP_ANKLE": 0.4182664354329112,
    "KD_ANKLE": 0.02117115810161374,
    "ANKLE_SW": -0.06430364723348458,
    "KP_KNEE_ST": 0.7120958180836433,
    "KD_KNEE_ST": 0.23511474772666302,
    "PUSH_START": 0.06352328925886383,
    "KV_PITCH": -0.023566363743137107,
    "SW_MIN": 1.1473196265362964,
    "HIP_DRIVE": 0.004751397391749977,
    "ASYM_PHI": 0.010853583244877776,
    "ASYM_T": -0.0006738132774360694,
    "ASYM_PUSH": 0.0063156857677927995,
    "ASYM_PITCH": 0.012511342253466548,
    "KNEE_THRUST": 0.3867550987362264,
    "THRUST_START": 0.5163607094680216,
    "KH_KNEE": -5.8798736471322215,
    "H_REF": 0.9339004375546176,
    "KH_HIP": 3.178498196870298,
    "FF_HS_S": 0.2548456568428261,
    "FF_HS_C": 0.13077197096164217,
    "FF_HW_S": 0.21242099032654171,
    "FF_HW_C": 0.3515723484585485,
    "FF_KS_S": 1.1431085028223371,
    "FF_KS_C": 0.5601171640705549,
    "T_STEP": 0.21730387510913532,
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

    sgn = 1.0 if _state["stance"] == 0 else -1.0
    T_STEP = P["T_STEP"] * (1.0 + sgn * P["ASYM_T"])
    _state["t"] += obs["dt"]
    sw_contact = obs["foot_left_contact"] if _state["stance"] == 0 else obs["foot_contact"]
    if _state["t"] > T_STEP or (sw_contact and _state["t"] > P["SW_MIN"] * T_STEP):
        _state["stance"] = 1 - _state["stance"]
        _state["t"] = 0.0
    st = _state["stance"]
    sw = 1 - st
    t = _state["t"]
    sgn = 1.0 if st == 0 else -1.0
    T_STEP = P["T_STEP"] * (1.0 + sgn * P["ASYM_T"])
    hs, ks, as_ = 3 * st, 3 * st + 1, 3 * st + 2
    hw, kw, aw = 3 * sw, 3 * sw + 1, 3 * sw + 2
    u = np.zeros(6)

    phi_d = P["PHI0"] + sgn * P["ASYM_PHI"] + P["CV"] * (v - P["V_REF"])
    u[hw] = P["KP_SWING"] * (phi_d - (q[hw] - p)) - P["KD_SWING"] * (dq[hw] - dp)

    pitch_ref = P["PITCH_REF"] + sgn * P["ASYM_PITCH"] + P["KV_PITCH"] * (v - P["V_REF"])
    tau_torso = -P["KP_TORSO"] * (p - pitch_ref) - P["KD_TORSO"] * dp
    u[hs] = tau_torso - u[hw] + P["HIP_DRIVE"]

    knee_t = P["KNEE_SWING"] if t < P["SWING_FLEX"] * T_STEP else P["KNEE_EXT"]
    u[kw] = P["KP_KNEE"] * (knee_t - q[kw]) - P["KD_KNEE"] * dq[kw]
    u[ks] = P["KP_KNEE_ST"] * (P["KNEE_STANCE"] - q[ks]) - P["KD_KNEE_ST"] * dq[ks]
    if t >= P["THRUST_START"] * T_STEP:
        u[ks] += P["KNEE_THRUST"]

    u[as_] = P["KP_ANKLE"] * (P["ANKLE_STANCE"] - q[as_]) - P["KD_ANKLE"] * dq[as_]
    if t >= P["PUSH_START"] * T_STEP:
        u[as_] += P["ANKLE_PUSH"] + sgn * P["ASYM_PUSH"]
    foot_level = p - q[hw] - q[kw] + P["ANKLE_SW"]
    u[aw] = P["KP_ANKLE"] * (foot_level - q[aw]) - P["KD_ANKLE"] * dq[aw]
    h_err = max(P["H_REF"] - obs["height"], 0.0)
    u[ks] += P["KH_KNEE"] * h_err
    u[hs] += P["KH_HIP"] * h_err
    s_ph = np.pi * min(t / T_STEP, 1.0)
    sn, cs = np.sin(s_ph), np.cos(s_ph)
    u[hs] += P["FF_HS_S"] * sn + P["FF_HS_C"] * cs
    u[hw] += P["FF_HW_S"] * sn + P["FF_HW_C"] * cs
    u[ks] += P["FF_KS_S"] * sn + P["FF_KS_C"] * cs
    return np.clip(u, -1.0, 1.0)
