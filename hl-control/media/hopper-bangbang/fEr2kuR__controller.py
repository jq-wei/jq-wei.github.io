"""Hopper controller for bang-bang (3-level) motors.

Each motor command is realised by error-feedback (sigma-delta) dithering so the
time-averaged torque equals a continuous command; at 125 Hz with the heavy joint
armature this behaves like a continuous torque.

Stance (foot on the floor):
  knee   : virtual spring about q0 (ks) plus a thrust 'boost' while it extends
  ankle  : balance leg lean (xrel-xd)/h with gain kf, damped by body speed kfd,
           plus a toe push 'fpush' while the knee extends
  hip    : torso-pitch PD (kp, kpd) about pdes, plus a pull toward thigh angle thd
Flight:
  knee tucks to kl, hip goes to thigh angle thf with torso-pitch PD kpf,
  ankle returns to neutral (kfl).
"""
import numpy as np

PARAMS = dict(
    kl=0, pdes=-0.01944, thd=-0.1194, thf=-0.1181, ks=0.3664, q0=-0.3615, boost=0.514,
    kf=2.985, kfd=0.4656, xd=-0.32, kp=2.542, kpd=1.346, kfl=1.227, kpf=0.9741,
    fpush=0.1191, pdf=-0.1373, kth=0.2398)
S = {}


def dither(u):
    v = u + S["e"]
    q = np.clip(np.round(v), -1, 1)
    S["e"] = np.clip(v - q, -1, 1)
    return q


def controller(obs):
    P = PARAMS
    if obs["step"] == 0:
        S["e"] = np.zeros(3)
    p = obs["pitch"]; pr = obs["pitch_rate"]
    xrel, zrel = obs["foot_xz_rel"]
    if obs["foot_contact"]:
        h = max(-zrel, 0.5)
        ext = obs["leg_rate"] > 0
        u_k = P["ks"] * (P["q0"] - obs["leg"]) - 0.3 * obs["leg_rate"] + (P["boost"] if ext else 0.0)
        u_f = P["kf"] * (xrel - P["xd"]) / h - P["kfd"] * obs["x_velocity"] / h + (P["fpush"] if ext else 0.0)
        u_h = -P["kp"] * (p - P["pdes"]) - P["kpd"] * pr + P["kth"] * (P["thd"] - obs["thigh"])
    else:
        u_k = 3.0 * (P["kl"] - obs["leg"]) - 0.3 * obs["leg_rate"]
        u_h = 0.3 * (P["thf"] - obs["thigh"]) - 0.1 * obs["thigh_rate"] - P["kpf"] * (p - P["pdf"]) - 0.3 * P["kpf"] * pr
        u_f = P["kfl"] * (0 - obs["foot"]) - 0.1 * obs["foot_rate"]
    return dither(np.array([u_h, u_k, u_f]))
