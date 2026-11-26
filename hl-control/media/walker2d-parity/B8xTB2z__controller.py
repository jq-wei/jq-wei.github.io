"""Walker2d central-pattern-generator gait with joint PD tracking and pitch feedback.

A phase oscillator phi = 2*pi*f*t drives joint targets
    q*_j = c_j + A_j * sin(phi + p_j)            (right leg; left leg uses phi + pi + dphi)
for hip (thigh), knee (leg) and ankle (foot).  Each motor command is the PD law
    u = kp_j * (q* - q) - kd * qdot                 (normalised, 1.0 = 100 N m)
Torso pitch feedback shifts the hip targets: q*_thigh += kpitch * pitch + kpr * pitch_rate.
PARAMS:  f gait frequency; c/A/p offset, amplitude, phase of each joint pattern;
kp* proportional gains; kd damping; dphi left/right phase offset; kpitch/kpr pitch feedback.
"""

import numpy as np

PARAMS = {
    "f": 2.19117,
    "dphi": 0.00593,
    "c0": 0.0,
    "A0": 1.0,
    "p0": 3.14,
    "c1": -0.77509,
    "A1": 1.5,
    "p1": 1.2026,
    "c2": 0.6,
    "A2": 0.24769,
    "p2": 3.14,
    "kp0": 12.0,
    "kp1": 0.5,
    "kp2": 5.70745,
    "kd": 0.16855,
    "kpitch": -2.42367,
    "kpr": -0.11537
}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    phi = 2 * np.pi * P["f"] * obs["time"]
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    tgt = np.zeros(6)
    for leg, ph in ((0, phi), (3, phi + np.pi + P["dphi"])):
        for j in range(3):
            tgt[leg + j] = P[f"c{j}"] + P[f"A{j}"] * np.sin(ph + P[f"p{j}"])
        tgt[leg] += P["kpitch"] * obs["pitch"] + P["kpr"] * obs["pitch_rate"]
    kp = np.array([P["kp0"], P["kp1"], P["kp2"]] * 2)
    u = kp * (tgt - q) - P["kd"] * qd
    return np.clip(u, -1.0, 1.0)
