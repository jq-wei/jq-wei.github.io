"""Raibert-style stepping controller for the MuJoCo Humanoid (point feet, no ankles).

Hand-designed, no learning.  The humanoid cannot stand still (spherical feet, no ankle),
so it balances by stepping.  Structure:

* Landing.  The model starts 11 cm in the air; until first contact the legs hold a
  landing pose, then both legs absorb the impact in double support for T_SETTLE.
* Clock.  A step clock of period T_step alternates stance and swing legs.  A swing ends
  when its foot touches down (after S_TOUCH of the period) or the period runs out; an
  overdue swing lowers the foot until contact.
* Stance leg (strut + attitude).  The hip torques regulate the pelvis attitude
  (kp_pitch, kp_roll with fixed damping KD_PITCH_FRAC*kp_pitch, KD_ROLL) and the heading
  (KP_YAW); loaded legs share the attitude torque.  The knee supports the weight with a
  gravity feed-forward (knee_ff*sin(flexion)) plus a soft PD (knee_kp, KNEE_KD) that
  starts flexed at touchdown (knee_land) and extends to knee_stance through the stance.
* Swing leg (lift + place).  The knee flexes by KNEE_SWING and is straight again by
  S_EXT of the swing.  The foot is placed relative to the mass centre in the heading
  frame by a capture-point law: x_f = k_cp*v + k_v*(v - V_DES) forward, and
  y_f = -/+ y_off + k_y*v_y laterally (half step width, shifted along the lateral
  velocity).  The swing-hip position relative to the mass centre is estimated from the
  measured stance-foot position and the stance-leg forward kinematics, and the target is
  converted to hip angles by a virtual-leg inverse kinematics that compensates the
  current knee flexion.  Hip joints track the targets with PD.
* Abdomen is held stiffly at zero (k_abd) so the torso follows the pelvis; arms are held
  at zero.  Torques are divided by the gear ratios and clipped to [-0.4, 0.4].

PARAMS were searched with hl_control.harbor.tune on the visible seeds; PARAM_BOUNDS give
the search ranges.
"""
import numpy as np

PARAMS = {
    "T_step": 0.393,      # s, swing (= stance) duration
    "k_cp": 0.254,        # s, capture-point foot placement gain (x_f = k_cp*v + k_v*(v - v_des))
    "k_v": 0.076,         # s, foot placement gain on speed error
    "kp_pitch": 205.135,   # N m / rad, stance-hip pelvis pitch gain
    "kp_roll": 620.083,    # N m / rad, stance-hip pelvis roll gain
    "knee_land": 0.308,    # rad, stance knee flexion at touchdown
    "k_abd": 130.303,      # N m / rad, abdomen hold stiffness (pitch and roll)
    "k_y": 0.316,         # s, lateral foot placement gain
    "y_off": 0.272,        # m, half step width (lateral foot distance from the mass centre)
    "knee_stance": 0.022, # rad, stance knee bend
    "knee_ff": 55.008,     # N m, stance knee gravity feed-forward (body weight on the bent shin)
    "knee_kp": 77.112,     # N m / rad, stance knee PD stiffness
}
PARAM_BOUNDS = {
    "T_step": (0.2, 0.6), "knee_ff": (20, 130), "knee_kp": (30, 160), "k_v": (0.0, 0.5), "k_cp": (0.1, 0.5), 
    "kp_pitch": (50, 600), "kp_roll": (100, 1200), "knee_land": (0.2, 1.0), "k_abd": (50, 400),
    "k_y": (0.0, 0.4), "y_off": (0.08, 0.3),  "knee_stance": (0.02, 0.5), 
}

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
A_TH, B_SH = 0.40, 0.35   # thigh length, knee to foot centre [m]
S_TOUCH = 0.6                  # swing fraction after which a touchdown ends the swing
T_SETTLE = 0.15                 # s, double-support landing absorption before the gait starts
V_DES = 0.525                   # m/s, desired forward speed
KNEE_SWING = 1.25               # rad, swing knee flexion amplitude
KNEE_KD = 8.0                   # N m s / rad, stance knee damping
S_MAX = 1.5                     # swing is extended (foot lowered) until contact, at most to S_MAX
S_EXT = 0.6                     # swing fraction by which the knee is straight again
PITCH_DES = 0.0                 # rad, desired pelvis/torso pitch (forward positive)
KD_PITCH_FRAC = 0.15            # pitch damping as a fraction of kp_pitch
KD_ROLL = 15.0                  # N m s / rad, stance-hip roll damping
KP_YAW = 100.0                  # N m / rad, stance-hip heading hold
# joint indices
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_KNEE = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_KNEE = 7, 8, 9, 10

_state = {"stance": 0, "t_sw": 0.0, "last_step": -1}


def rot_zyx(yaw, pitch, roll):
    """World-from-body rotation for the (roll, pitch, yaw) Euler angles used by the environment."""
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


def leg_vector(hx_about_x, hy, knee):
    """Hip-to-foot vector in the pelvis frame (hip_x given as a rotation about +x; knee axis is -y)."""
    Rx = rot_zyx(0.0, 0.0, hx_about_x)
    Ry = rot_zyx(0.0, hy, 0.0)
    thigh = np.array([0.0, 0.0, -A_TH])
    shin = rot_zyx(0.0, -knee, 0.0) @ np.array([0.0, 0.0, -B_SH])
    return Rx @ Ry @ (thigh + shin)


def pd(q, qd, q_des, kp, kd):
    return kp * (q_des - q) - kd * qd


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    q = obs["joint_angles"]; qd = obs["joint_rates"]
    dt = obs["dt"]; step = obs["step"]
    pitch, roll = obs["pitch"], obs["roll"]
    w = obs["root_angular_velocity"]        # world frame; w[1] ~ pitch rate, w[0] ~ roll rate
    v = obs["root_velocity"]
    vx, vy = float(v[0]), float(v[1])
    st = _state
    if step == 0 or step <= st["last_step"]:   # new episode: reset the gait state
        st["stance"] = 0; st["t_sw"] = 0.0; st["v"] = np.zeros(2); st["landed"] = False; st["t_land"] = 0.0
    st["last_step"] = step

    contacts = (obs["foot_contact"], obs["foot_left_contact"])
    vc = obs["com_velocity_xy"]
    st["v"] = 0.7 * st["v"] + 0.3 * vc           # low-pass mass-centre velocity
    vx, vy = float(st["v"][0]), float(st["v"][1])
    if not st["landed"]:
        if contacts[0] or contacts[1]:
            st["landed"] = True; st["t_sw"] = 0.0; st["t_land"] = obs["time"]
        else:   # still falling from the initial 1.4 m: hold a landing pose
            tau = np.zeros(17)
            for j, qd_ in ((ABD_Y, 0), (ABD_Z, 0), (ABD_X, 0), (R_HX, 0), (R_HZ, 0), (R_HY, 0), (R_KNEE, -P["knee_stance"]),
                           (L_HX, 0), (L_HZ, 0), (L_HY, 0), (L_KNEE, -P["knee_stance"])):
                tau[j] = pd(q[j], qd[j], qd_, 150, 2)
            return np.clip(tau / GEAR, -0.4, 0.4)
    if obs["time"] - st["t_land"] < T_SETTLE:     # absorb the landing in double support before stepping
        tau = np.zeros(17)
        tau[ABD_Y] = pd(q[ABD_Y], qd[ABD_Y], 0.0, 150, 5); tau[ABD_X] = pd(q[ABD_X], qd[ABD_X], 0.0, 150, 5); tau[ABD_Z] = pd(q[ABD_Z], qd[ABD_Z], 0.0, 100, 3)
        pel_pitch = pitch + q[ABD_Y]; pel_roll = roll + q[ABD_X]
        for l_hx, l_hz, l_hy, l_kn, l_sgn in ((R_HX, R_HZ, R_HY, R_KNEE, 1.0), (L_HX, L_HZ, L_HY, L_KNEE, -1.0)):
            tau[l_hy] = 0.5 * (P["kp_pitch"] * pel_pitch + KD_PITCH_FRAC * P["kp_pitch"] * w[1])
            tau[l_hx] = 0.5 * l_sgn * (P["kp_roll"] * pel_roll + KD_ROLL * w[0])
            tau[l_hz] = pd(q[l_hz], qd[l_hz], 0.0, 50, 2)
            tau[l_kn] = pd(q[l_kn], qd[l_kn], -P["knee_land"] - 0.2, 80, 12)
        st["t_sw"] = 0.0
        return np.clip(tau / GEAR, -0.4, 0.4)
    st["t_sw"] += dt
    s = st["t_sw"] / P["T_step"]
    swing = 1 - st["stance"]
    if (s > S_TOUCH and contacts[swing]) or s >= S_MAX or (s >= 1.0 and not contacts[st["stance"]]):
        st["stance"] = swing; st["t_sw"] = 0.0; s = 0.0; swing = 1 - st["stance"]
    stance = st["stance"]

    tau = np.zeros(17)
    # ----- torso / abdomen / arms: hold nominal -----
    # abdomen: stiff hold at zero so the torso follows the pelvis (the hips control the pelvis attitude)
    tau[ABD_Y] = pd(q[ABD_Y], qd[ABD_Y], 0.0, P["k_abd"], 0.05 * P["k_abd"])
    tau[ABD_Z] = pd(q[ABD_Z], qd[ABD_Z], 0.0, 100, 3)
    tau[ABD_X] = pd(q[ABD_X], qd[ABD_X], 0.0, P["k_abd"], 0.05 * P["k_abd"])
    for j in range(11, 17):
        tau[j] = pd(q[j], qd[j], 0.0, 10, 0.3)

    legs = ((R_HX, R_HZ, R_HY, R_KNEE, +1.0), (L_HX, L_HZ, L_HY, L_KNEE, -1.0))
    # ----- stance leg: strut + attitude control -----
    hx, hz, hy, kn, sgn = legs[stance]
    pel_pitch = pitch + q[ABD_Y]          # pelvis attitude (abdomen y/x fold the lower body relative to the torso)
    pel_roll = roll + q[ABD_X]
    tau_pitch = P["kp_pitch"] * (pel_pitch - PITCH_DES) + KD_PITCH_FRAC * P["kp_pitch"] * w[1]   # + hip_y torque pitches the pelvis back
    tau_roll = P["kp_roll"] * pel_roll + KD_ROLL * w[0]
    n_contact = int(contacts[0]) + int(contacts[1])
    for leg_i, (l_hx, l_hz, l_hy, l_kn, l_sgn) in enumerate(legs):     # every loaded leg shares the attitude torques
        if contacts[leg_i] and (leg_i == stance or s < 0.5):
            share = 1.0 / n_contact
            tau[l_hy] += share * tau_pitch
            tau[l_hx] += share * l_sgn * tau_roll
    tau[hz] = sgn * (KP_YAW * obs["yaw"] + 0.1 * KP_YAW * w[2]) + pd(q[hz], qd[hz], 0.0, 50, 2)   # hold the heading
    knee_st_des = -(P["knee_land"] + (P["knee_stance"] - P["knee_land"]) * min(s, 1.0))   # compliant landing, extending through stance
    tau[kn] = P["knee_ff"] * np.sin(-q[kn]) + pd(q[kn], qd[kn], knee_st_des, P["knee_kp"], KNEE_KD)   # weight support feed-forward + soft PD
    # ----- swing leg: lift and place (foot placement relative to the mass centre, world frame) -----
    hx, hz, hy, kn, sgn = legs[swing]
    R_pel = rot_zyx(obs["yaw"], pitch + q[ABD_Y], roll + q[ABD_X])   # pelvis orientation (abdomen y/x fold the lower body)
    # hip position relative to the mass centre = stance foot (measured) - stance leg vector (forward kinematics)
    s_hx, s_hz, s_hy, s_kn, s_sgn = legs[stance]
    stance_foot = obs["foot_xyz_rel"] if stance == 0 else obs["foot_left_xyz_rel"]
    hip_rel_com = stance_foot - R_pel @ leg_vector(q[s_hx] * s_sgn, q[s_hy], q[s_kn])
    hip_rel_com = hip_rel_com + R_pel @ np.array([0.0, -sgn * 0.2, 0.0])     # the swing hip is across the pelvis
    yaw = obs["yaw"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    vx_h, vy_h = cy * vx + sy * vy, -sy * vx + cy * vy                 # mass-centre velocity in the heading frame
    x_f = P["k_cp"] * vx_h + P["k_v"] * (vx_h - V_DES)
    x_f = float(np.clip(x_f, -0.3, 0.45))
    y_f = -sgn * P["y_off"] + P["k_y"] * vy_h                           # half step width, shifted along the lateral velocity
    foot_des = np.array([cy * x_f - sy * y_f, sy * x_f + cy * y_f, 0.0])   # back to world axes
    target = foot_des - hip_rel_com                                     # desired foot position relative to the swing hip (world)
    target[2] = -0.70 - 0.15 * max(s - 1.0, 0.0)          # lower the foot if the swing runs over time
    knee_des = -P["knee_stance"] - KNEE_SWING * np.sin(np.pi * min(s / S_EXT, 1.0))   # flexed early, straight by s_ext
    kappa = -float(q[kn])                                  # current knee flexion (>= 0)
    knee_offset = np.arctan2(B_SH * np.sin(kappa), A_TH + B_SH * np.cos(kappa))   # virtual leg is behind the thigh by this
    v = R_pel.T @ target                                   # desired virtual-leg direction in the pelvis frame
    v = v / (np.linalg.norm(v) + 1e-9)
    hy_virtual = -np.arcsin(np.clip(v[0], -0.95, 0.95))
    hx_des = sgn * np.arcsin(np.clip(v[1] / max(np.cos(hy_virtual), 0.3), -0.9, 0.9))
    hy_des = hy_virtual - knee_offset
    tau[hy] += pd(q[hy], qd[hy], hy_des, 400, 4)
    tau[hx] += pd(q[hx], qd[hx], hx_des, 200, 3)
    tau[hz] = pd(q[hz], qd[hz], 0.0, 100, 2)
    tau[kn] = pd(q[kn], qd[kn], knee_des, 200, 4)

    u = tau / GEAR
    return np.clip(u, -0.4, 0.4)
