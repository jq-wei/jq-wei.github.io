"""Walker2d running controller: clocked central pattern generator + PD tracking + torso feedback.

Control law (all gains found by derivative-free CMA-ES search over rollouts, no learning of networks):

1. Clock.  phase phi = 2 pi F t (F = 1.5924 Hz stride frequency); the left leg runs half a cycle
   behind the right (phi + pi).  Gait amplitude ramps in linearly over T_RAMP s from standing.
2. Joint targets.  For each leg, q*_j(phi) = C[j] . [1, cos phi, sin phi, cos 2phi, sin 2phi]
   (rows: hip, knee, ankle; C[:,0] is the mean posture, the rest the 2-harmonic swing/stance shape).
   LEFT_OFFSET shifts the left leg's mean posture (the gait is allowed to be asymmetric).
   Hip targets get foot-placement feedback K_V (v_x - V_REF) + K_PT (pitch - P_REF).
3. PD tracking.  u = KP (q* - q) - KD qdot, per joint type (hip, knee, ankle), in units of 100 N m.
4. Torso balance.  Both hips get K_P (pitch - P_REF) + K_PD pitch_rate, weighted by
   (1 + K_ST (contact - 1/2)) so the stance leg does more of the torso work.
5. State feedback.  u += K_FB s, s = [pitch err, 0.3 pitch rate, v_x err, 0.5 v_z, 4 (height - 1.2)]
   (rows: the 6 motors), small corrections to the open-loop pattern.
"""

import numpy as np

F = 1.592379507899953
C = np.array([[-0.34653,  0.35910,  0.30036,  0.07867,  0.13896],
 [-1.07638,  0.27760, -0.37903, -0.24656, -0.49079],
 [ 1.08147,  1.07672,  0.45013, -0.41586, -0.19954]])
KP = np.array([2.59546, 4.70413, 4.18396] * 2)
KD = np.array([0.18458, 0.18469, 0.03152] * 2)
P_REF, K_P, K_PD, K_V, T_RAMP = -0.03164146350315361, -0.15352717628095325, 0.0468293357505688, -0.21159846672178895, 0.2514072585206014
LEFT_OFFSET = np.array([-0.08661, -0.10704,  0.10645])
K_PT, K_ST = -0.29705159505511874, -0.31212802443239623
K_FB = np.array([[-0.48126, -0.10524, -0.41990,  0.38496,  0.14836],
 [ 0.35396, -0.39634,  0.17302, -0.31569,  0.19857],
 [-0.09343, -0.41330, -0.50978,  0.18751, -0.23461],
 [-0.62345,  0.49540, -0.03574,  0.29387,  0.17624],
 [ 0.42514, -0.22977, -0.13346, -0.02885, -0.08758],
 [-0.41256,  0.20769, -0.56645, -0.03414, -0.04936]])
V_REF = 3.02319888590956


def controller(obs: dict) -> np.ndarray:
    t = obs["time"]
    phi = 2 * np.pi * F * t
    ramp = min(1.0, t / max(T_RAMP, 1e-3))
    target = np.empty(6)
    for leg, p in ((0, phi), (1, phi + np.pi)):
        basis = np.array([1.0, ramp * np.cos(p), ramp * np.sin(p), ramp * np.cos(2 * p), ramp * np.sin(2 * p)])
        target[3 * leg:3 * leg + 3] = C @ basis
    target[3:6] += LEFT_OFFSET * ramp
    e_pitch = obs["pitch"] - P_REF
    e_v = obs["x_velocity"] - V_REF
    target[[0, 3]] += (K_V * e_v + K_PT * e_pitch) * ramp

    u = KP * (target - obs["joint_angles"]) - KD * obs["joint_rates"]
    contact = np.array([obs["foot_contact"], obs["foot_left_contact"]], float)
    u[[0, 3]] += (K_P * e_pitch + K_PD * obs["pitch_rate"]) * (1.0 + K_ST * (contact - 0.5))
    s = np.array([e_pitch, 0.3 * obs["pitch_rate"], e_v, 0.5 * obs["z_velocity"], 4 * (obs["height"] - 1.2)])
    u += ramp * (K_FB @ s)
    return np.clip(u, -1.0, 1.0)
