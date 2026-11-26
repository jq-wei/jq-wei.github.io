"""Hopper controller: a phase oscillator (central pattern generator) with attitude and speed feedback.

Gait found by tuning gains (CMA-ES, black-box) on all five physics scenarios; no learning of a policy.
Each joint command is   u_j = A_j + B_j sin(phase + ph_j) + feedback_j,   phase advancing at `freq` Hz
(re-synchronised to `ph_td` at foot touchdown if `lock` > 0.5).

  hip   (thigh): A0 + B0 sin(phase)        - kp (pitch - p_des) - kd pitch_rate
                 bias keeps the thigh off its 0-rad limit; the PD holds the torso inside the +-0.2 rad band
                 (in stance a hip-extension torque both pitches the torso back and propels the body forward).
  knee  (leg)  : A1 + B1 sin(phase + ph1)  + kpk (pitch - p_des)
                 mostly a constant extension against the straight-knee limit (a stiff pogo leg).
  ankle (foot) : A2 + B2 sin(phase + ph2)  + ka (pitch - p_des) + kv (v_x - vdes)
                 the oscillation produces the toe-down push-off that launches each hop; kv < 0 tips the body
                 forward (more ankle torque) when slower than vdes, regulating speed.

Status (visible seeds 0-4): alive on every seed of the nominal plant and of heavy_torso_20, low_friction_30
and gravity_high_10, about 8.3 m (return ~2030) on the nominal plant -- short of the 10 m / 2500 gate.
"""
import numpy as np
PARAMS = dict(freq=1.53532, A0=-0.0366005, B0=0.51797, A1=0.694862, B1=0.192436, ph1=0.672801, A2=-0.0888619, B2=0.482487, ph2=1.99111, kp=1.13882, kd=0.432303, p_des=0.0636863, ka=-2.82197, kv=-0.53922, vdes=1.32701, kpk=0.799208, ph_td=2.30464, lock=0.175297)
BOUNDS = dict(freq=(1.5, 6.0), A0=(-0.5, 0.5), B0=(0, 0.8), A1=(-0.5, 1.0), B1=(0, 1.0), ph1=(-3.2, 3.2), A2=(-0.6, 0.6), B2=(0, 0.8), ph2=(-3.2, 3.2),
              kp=(-1, 4), kd=(-0.3, 0.5), p_des=(-0.15, 0.2), ka=(-3, 3), kv=(-1, 1), vdes=(0.5, 3), kpk=(-3, 3), ph_td=(-3.2, 3.2), lock=(0, 1))
P = dict(PARAMS)
S = dict(phase=0.0, contact=False)
def controller(obs):
    if obs["step"] == 0:
        S["phase"] = 0.0; S["contact"] = bool(obs["foot_contact"])
    S["phase"] += 2 * np.pi * P["freq"] * obs["dt"]
    c = bool(obs["foot_contact"])
    if c and not S["contact"] and P["lock"] > 0.5:  # touchdown: re-synchronise the oscillator
        S["phase"] = P["ph_td"]
    S["contact"] = c
    w = S["phase"]
    p, pd, v = obs["pitch"], obs["pitch_rate"], obs["x_velocity"]
    e = p - P["p_des"]
    a = np.array([P["A0"] + P["B0"] * np.sin(w) - P["kp"] * e - P["kd"] * pd,
                  P["A1"] + P["B1"] * np.sin(w + P["ph1"]) + P["kpk"] * e,
                  P["A2"] + P["B2"] * np.sin(w + P["ph2"]) + P["ka"] * e + P["kv"] * (v - P["vdes"])])
    return np.clip(a, -1, 1)
