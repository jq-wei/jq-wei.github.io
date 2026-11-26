"""Stepping controller for the MuJoCo Humanoid (point feet, no ankles): SIMBICON-style sagittal
foot placement + wide fixed stance with lateral-sway-triggered step timing.

Every 15 ms (commands in [-0.4, 0.4] = fraction of motor gear):

* Finite-state machine: one leg is stance, the other swings.  A step ends when the swing foot
  touches down after half the nominal step time, when the mass centre is already falling toward
  the swing side faster than V_TRIG (lateral sway timing: the free leg is put down just when the
  body starts falling onto it), or after STEP_T at the latest.
* Swing leg: PD position control.  The swing thigh pitch target is given in the world frame
  (pelvis pitch compensated): phi_d = -(TH0 + C_V (v_x - V_DES) + C_D d_x), with d_x the mass
  centre's lead over the stance foot and v_x its speed -> the faster the body falls forward, the
  further ahead the foot is placed.  The swing thigh roll target is a fixed abduction ABD (wide
  stance, feet ~0.2 m either side) plus C_VY * v_y; the knee bends by KNEE_SW early in swing
  (sine profile) for foot clearance and is straight again by mid-step.
* Stance leg: the hip keeps the pelvis upright (PD on pelvis pitch/roll, KP_T KD_T / KP_R KD_R,
  rates low-pass filtered with RATE_LP) and cancels the reaction of the swing hip torque
  (SW_COMP); the knee is held nearly straight (KNEE_ST) with a small extension bias (KNEE_FF).
* Abdomen and arms: PD to a fixed posture; hip yaw: PD toward K_YAW * heading error so the
  planted feet steer the body back to the +x heading during double support.
* Start: a short settling phase after the drop (soft knees), then stepping.

GAINS are joint PD gains (command per rad, per rad/s) chosen from discrete-time stability of each
joint loop at 15 ms (measured effective inertias); PARAMS are gait parameters (searched with
hl_control.harbor.tune on the visible seeds).
"""

import numpy as np

PARAMS = {   # gait parameters (found with hl_control.harbor.tune, ES sampler, on seeds 0-23)
    "STEP_T": 0.268,    # maximum step duration [s]
    "TH0": 0.393,       # nominal swing thigh pitch (forward, rad)
    "C_V": 0.268,       # foot placement gain on forward velocity error [s]
    "C_D": 0.354,       # foot placement gain on CoM lead over the stance foot
    "V_DES": 0.339,     # target forward speed [m/s]
    "ABD": 0.193,       # fixed thigh abduction (world roll of each thigh, rad): wide stance
    "C_VY": 0.490,      # lateral placement gain on lateral velocity [s]
    "V_TRIG": 0.361,    # lateral CoM speed toward the swing side that triggers touchdown [m/s]
    "KNEE_SW": 0.865,   # swing knee bend amplitude [rad] (foot clearance; the most sensitive parameter)
    "KNEE_ST": 0.176,   # stance knee bend [rad]
    "PITCH_DES": -0.054,# pelvis pitch target [rad], forward positive
    "KP_T": 0.974,      # pelvis pitch P gain via stance hip_y (gear 300)
}
GAINS = {    # joint-level PD gains, command units per rad and per rad/s
    "KD_T": 0.08,                # pelvis pitch D gain (filtered rate)
    "KP_R": 3.0, "KD_R": 0.3,    # pelvis roll PD via stance hip_x (gear 100)
    "RATE_LP": 0.35,             # low-pass factor per step on pelvis rates (kills the 11 Hz torso mode)
    "KP_H": 0.8, "KD_H": 0.01,   # swing hip_y PD (gear 300)
    "KP_HX": 2.0, "KD_HX": 0.08, # swing hip_x PD (gear 100)
    "KP_K": 0.25, "KD_K": 0.01,  # swing knee PD (gear 200, light shin: soft)
    "KP_KS": 1.0, "KD_KS": 0.01, # stance knee PD (loaded by the body: stiff)
    "KNEE_FF": 0.05,             # stance knee extension feed-forward
    "KP_AB": 1.5, "KD_AB": 0.0,  # abdomen P (gear 100; joint has 5 N m s/rad intrinsic damping)
    "K_YAW": 0.5,                # hip yaw target = K_YAW * heading error
    "SW_COMP": 1.0,              # fraction of the swing-hip torque fed back into the stance hip
    "T_MIN": 0.12,               # earliest lateral-triggered touchdown [s]
}
PARAM_BOUNDS = {
    "STEP_T": (0.25, 0.6), "TH0": (0.0, 0.5), "C_V": (0.0, 0.8), "C_D": (0.0, 0.8), "V_DES": (0.2, 0.9),
    "ABD": (0.0, 0.3), "C_VY": (0.0, 0.6), "V_TRIG": (0.05, 0.5), "KNEE_SW": (0.2, 1.0), "KNEE_ST": (0.05, 0.4),
    "PITCH_DES": (-0.2, 0.25), "KP_T": (0.5, 4.0),
}

AB_Y, AB_Z, AB_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_KN = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_KN = 7, 8, 9, 10
ARMS = [11, 12, 13, 14, 15, 16]
ARM_POSE = np.array([0.4, -0.4, -0.6, -0.4, 0.4, -0.6])
START_T = 0.2

_S = {"stance": 1, "t_sw": 0.0, "started": False, "settled": 0, "wy_f": 0.0, "wx_f": 0.0}


def _heading_frame(vx, vy, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return c * vx + s * vy, -s * vx + c * vy


def controller(obs: dict) -> np.ndarray:
    P, G, S = PARAMS, GAINS, _S
    if obs["step"] == 0:
        S.update(stance=1, t_sw=0.0, started=False, settled=0, wy_f=0.0, wx_f=0.0)
    dt = obs["dt"]
    q, qd = obs["joint_angles"], obs["joint_rates"]
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    w = obs["root_angular_velocity"]
    wx, wy = _heading_frame(w[0], w[1], yaw)
    vx, vy = _heading_frame(obs["com_velocity_xy"][0], obs["com_velocity_xy"][1], yaw)
    rc, lc = obs["foot_contact"], obs["foot_left_contact"]
    rf, lf = obs["foot_xyz_rel"], obs["foot_left_xyz_rel"]

    u = np.zeros(17)
    for k in (AB_Y, AB_Z, AB_X):
        u[k] = G["KP_AB"] * (0.0 - q[k]) - G["KD_AB"] * qd[k]
    yaw_err = float(np.clip(yaw, -0.5, 0.5))
    u[R_HZ] = 0.35 * (G["K_YAW"] * yaw_err - q[R_HZ]) - 0.02 * qd[R_HZ]
    u[L_HZ] = 0.35 * (-G["K_YAW"] * yaw_err - q[L_HZ]) - 0.02 * qd[L_HZ]
    for k, qa in zip(ARMS, ARM_POSE):
        u[k] = 0.5 * (qa - q[k]) - 0.03 * qd[k]

    # pelvis attitude (colocated with the hips); filtered rates
    p_pitch, p_roll = pitch + q[AB_Y], roll + q[AB_X]
    a = G["RATE_LP"]
    S["wy_f"] += a * ((wy + qd[AB_Y]) - S["wy_f"])
    S["wx_f"] += a * ((wx + qd[AB_X]) - S["wx_f"])
    u_t = P["KP_T"] * (P["PITCH_DES"] - p_pitch) - G["KD_T"] * S["wy_f"]
    u_r = G["KP_R"] * (0.0 - p_roll) - G["KD_R"] * S["wx_f"]

    # fixed-abduction hip_x targets (world thigh roll -ABD right, +ABD left)
    hxR_d = -P["ABD"] - p_roll
    hxL_d = p_roll - P["ABD"]

    t = obs["time"]
    if not S["started"]:
        settled = t > START_T and rc and lc and abs(obs["root_velocity"][2]) < 0.25
        S["settled"] = S["settled"] + 1 if settled else 0
        if S["settled"] < 1:
            for hy, kn, hx, hx_d, sgn in ((R_HY, R_KN, R_HX, hxR_d, 1.0), (L_HY, L_KN, L_HX, hxL_d, -1.0)):
                u[hy] = G["KP_H"] * (0.0 - q[hy]) - G["KD_H"] * qd[hy] - (0.5 * u_t if (rc or lc) else 0.0)
                u[kn] = G["KP_K"] * (-P["KNEE_ST"] - q[kn]) - G["KD_K"] * qd[kn]
                u[hx] = G["KP_HX"] * (hx_d - q[hx]) - G["KD_HX"] * qd[hx] - (0.5 * sgn * u_r if (rc or lc) else 0.0)
            return np.clip(u, -0.4, 0.4)
        S["started"] = True
        S["t_sw"] = 0.0
        S["stance"] = -1 if vy > 0 else 1   # swing the leg the body is moving away from

    # ---------- state machine ----------
    S["t_sw"] += dt
    stance = S["stance"]
    sw_contact = lc if stance == 1 else rc
    sw_kn_idx = L_KN if stance == 1 else R_KN
    v_toward_swing = vy if stance == 1 else -vy          # CoM speed toward the swing side
    ts = S["t_sw"]
    if (ts > P["STEP_T"]
            or (ts > 0.5 * P["STEP_T"] and sw_contact and q[sw_kn_idx] > -0.45)
            or (ts > G["T_MIN"] and v_toward_swing > P["V_TRIG"] and q[sw_kn_idx] > -0.45)):
        S["stance"] = stance = -stance
        S["t_sw"] = ts = 0.0
    s = min(ts / P["STEP_T"], 1.0)

    if stance == 1:
        st_hy, st_kn, st_hx, sw_hy, sw_kn, sw_hx = R_HY, R_KN, R_HX, L_HY, L_KN, L_HX
        st_foot, sw_sgn, hx_d_base = rf, -1.0, hxL_d
    else:
        st_hy, st_kn, st_hx, sw_hy, sw_kn, sw_hx = L_HY, L_KN, L_HX, R_HY, R_KN, R_HX
        st_foot, sw_sgn, hx_d_base = lf, 1.0, hxR_d
    dx, dy = _heading_frame(-st_foot[0], -st_foot[1], yaw)   # CoM lead over / left of stance foot

    # ---- swing leg ----
    phi_d = -(P["TH0"] + P["C_V"] * (vx - P["V_DES"]) + P["C_D"] * dx)
    phi_d = float(np.clip(phi_d, -1.0, 0.3))
    hy_d = phi_d - p_pitch
    hx_d = hx_d_base + sw_sgn * P["C_VY"] * vy                 # extra lateral placement along the velocity
    kn_d = -P["KNEE_ST"] - P["KNEE_SW"] * np.sin(np.pi * min(s / 0.5, 1.0))
    u_sw_hy = G["KP_H"] * (hy_d - q[sw_hy]) - G["KD_H"] * qd[sw_hy]
    u_sw_hx = G["KP_HX"] * (hx_d - q[sw_hx]) - G["KD_HX"] * qd[sw_hx]
    u[sw_hy], u[sw_hx] = u_sw_hy, u_sw_hx
    u[sw_kn] = G["KP_K"] * (kn_d - q[sw_kn]) - G["KD_K"] * qd[sw_kn]

    # ---- stance leg ----
    u[st_hy] = -u_t - G["SW_COMP"] * float(np.clip(u_sw_hy, -0.4, 0.4))
    u[st_hx] = -(1.0 if stance == 1 else -1.0) * u_r - G["SW_COMP"] * float(np.clip(u_sw_hx, -0.4, 0.4))
    u[st_kn] = G["KP_KS"] * (-P["KNEE_ST"] - q[st_kn]) - G["KD_KS"] * qd[st_kn] + G["KNEE_FF"]
    return np.clip(u, -0.4, 0.4)
