"""Capture-point stepping controller for the MuJoCo Humanoid (5 cm stairs, uneven ground).

No learning and no on-line simulation: a contact-free kinematic copy of the plant (same XML)
is set to the measured state each call and used only for forward kinematics, the mass-centre
velocity, leg Jacobians and gravity/bias torques (about 0.1 ms per call).

Gait (state machine, reset when obs["step"] == 0):
* Single support of nominal length T_SWING, then a double support of T_DS after each
  touchdown (touchdown = swing-foot contact after half the swing).
* Balance by foot placement (the feet are spheres, there is no ankle). The capture point
  xi = c + v / omega (omega = sqrt(g / z), c and v of the body *without* the swing leg, so
  the swing leg's own motion does not move its target) is propagated to the end of the step
  with the inverted-pendulum law; the swing foot is aimed at xi_T - B_X (B_X = V_DES T /
  (e^{omega T} - 1): steady walking at V_DES) and xi_T +- B_Y (B_Y = STEP_W / (1 + e^{omega T})),
  plus velocity feedback K_VX (v_x - V_DES), K_VY v_y; clipped to REACH (fore/aft, from the
  hips) and W_MAX (lateral, from the stance foot).
* Swing foot: rises to CLEAR above the stance foot by phase S_Z, moves horizontally between
  phases S_X0 and S_X (lift first, so a riser is cleared), then descends until contact. Joint
  targets from an analytic leg IK in the pelvis frame; joint PD (scaled by KSW) with
  velocity feed-forward and gravity compensation. If the swing runs late the stance hip is
  lowered by up to SINK so the foot can reach the ground.
* Stance leg: hip pitch/roll hold the torso attitude in the world (KP_T/KD_T pitch to LEAN,
  KP_R/KD_R roll to ROLL_LEAN toward the stance side) with the SIMBICON rule (stance hip torque
  = torso torque - swing hip torque; SIMB_X scales the roll part); the knee is a virtual
  vertical spring KP_H/KD_H holding the hip H_LEG above the stance foot (weight feed-forward
  through the leg Jacobian). The first touchdown from the 10 cm initial drop uses H_STAND.
* Yaw: a spherical foot cannot transmit a yaw torque, so yaw is corrected only in double
  support, by differential hip-pitch torques (K_YAW, D_YAW) reacting through floor friction;
  the trailing knee carries DS_SHARE of the weight meanwhile.
* Abdomen: PD (KP_AB, KD_AB) to a posture rolled AB_LEAN toward the stance side; arms: PD to a
  mirrored posture, shoulders shifted ARM_SH toward the stance side.
All gains are discrete-time safe for the 15 ms control period (low derivative gains on
light links: the abdomen twist and knee chatter otherwise).
"""

import numpy as np
import mujoco
from hl_control.envs.humanoid import XML_PATH

PARAMS = dict(
    V_DES=0.27613,      # m/s desired forward speed
    T_SWING=0.27065,    # s nominal swing duration
    STEP_W=0.3,     # m lateral distance between feet
    CLEAR=0.10069,      # m swing-foot lift above stance foot
    H_LEG=0.7008,      # m hip-to-foot-centre vertical distance in stance
    LEAN=-0.02,       # rad forward torso lean
    KP_T=187.9, KD_T=4.7223,   # torso attitude gains (N m / rad, N m s / rad)
    KP_R=95.937, KD_R=10.435,   # torso roll gains
    KP_H=2749, KD_H=377.74, # stance height spring (N/m, N s/m)
    K_CP=1.0,        # gain on the capture-point offset
    T_START=0.18,    # s, start of stepping
    K_VX=-0.012023, K_VY=0.26707,  # extra velocity feedback on the foot target (s)
    K_YAW=120.11,      # yaw stiffness in double support (N m / rad)
    KP_AB=202.5, KD_AB=4.1991,  # abdomen pitch/roll PD
    REACH=0.1599,      # m, max fore/aft foot target from the hip centre
    H_STAND=0.745,   # m, leg length while landing from the initial drop
    SIMB_X=0.81983,      # 1: stance hip roll cancels the swing hip roll torque (SIMBICON)
    KFF=1.0,         # swing joint velocity feed-forward
    AB_LEAN=0.11906,     # rad, abdomen roll toward the stance side
    S_X=0.67514,         # phase at which the swing foot reaches its horizontal target
    S_X0=0.08,       # phase at which the swing foot starts moving horizontally
    S_Z=0.3,         # phase at which the swing foot reaches full clearance
    SINK=0.007002,       # m, stance height reduction when the swing is late
    ROLL_LEAN=0.040876,   # rad, torso roll toward the stance side
    W_MAX=0.44554,      # m, widest lateral step
    T_DS=0.064619,        # s, double support after each touchdown
    DS_SHARE=0.2361,    # fraction of weight on the trailing knee in double support
    D_YAW=19.058,      # yaw damping in double support (N m s / rad)
    ARM_SH=0.02176,      # rad, shoulder2 offset toward the stance side
    X_SLOW=99.0,     # m of progress after which the target speed drops to V_SLOW
    V_SLOW=0.1,      # m/s
    ARM_SWING=0.0,   # rad, contralateral arm-swing amplitude
    KSW=0.92446,         # scale of swing-leg PD gains
)

G = 9.81
R_FOOT = 0.075
L1, L2 = 0.383, 0.37

_m = mujoco.MjModel.from_xml_path(str(XML_PATH))
_m.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
_d = mujoco.MjData(_m)
_MASS = float(_m.body_mass.sum())
_FOOT_G = [_m.geom("right_foot").id, _m.geom("left_foot").id]
_HIP_B = [_m.body("right_thigh").id, _m.body("left_thigh").id]
_PELVIS = _m.body("pelvis").id
_TORSO = _m.body("torso").id
# motor indices: leg = (hip_x, hip_z, hip_y, knee)
_LEG = [(3, 4, 5, 6), (7, 8, 9, 10)]
_GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
# dof index of each motor joint in qvel
_DOF = np.array([_m.jnt_dofadr[_m.joint(n).id] for n in (
    "abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
    "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")])
_jacp = np.zeros((3, _m.nv))
_jacr = np.zeros((3, _m.nv))
_ARM_REF = np.array([0.2, -0.3, -1.2, -0.2, 0.3, -1.2])   # left shoulder axes are negated mirrors


def _arm_forward_dirs():
    """Unit shoulder (s1, s2) velocity that moves each hand forward (+x) fastest, at the reference arm pose."""
    d = mujoco.MjData(_m)
    d.qpos[3] = 1.0
    for j, v in zip(range(11, 17), _ARM_REF):
        d.qpos[_m.jnt_qposadr[_m.dof_jntid[_DOF[j]]]] = v
    mujoco.mj_forward(_m, d)
    out = []
    for hand, dofs in (("right_hand", (11, 12)), ("left_hand", (14, 15))):
        jp = np.zeros((3, _m.nv)); jr = np.zeros((3, _m.nv))
        mujoco.mj_jacGeom(_m, d, jp, jr, _m.geom(hand).id)
        v = np.array([jp[0, _DOF[dofs[0]]], jp[0, _DOF[dofs[1]]]])
        out.append(v / (np.linalg.norm(v) + 1e-9))
    return out


_ARM_FWD = _arm_forward_dirs()

_S = {}


def _reset():
    _S.clear()
    _S.update(stance=1, t_ph=0.0, started=False, p0=None, swing=0, n_steps=0, ds=0.0, qdes_prev=None)


def _leg_ik(d, side):
    """Joint targets (hip_x, hip_y, knee) placing the foot centre at d (pelvis frame, rel. hip)."""
    dx, dy, dz = d
    if side == 0:
        phi = np.arctan2(dy, -dz)
    else:
        phi = np.arctan2(-dy, -dz)
    sz = np.sqrt(dy * dy + dz * dz)
    L = np.sqrt(dx * dx + sz * sz)
    L = np.clip(L, 0.25, L1 + L2 - 0.005)
    cb = (L * L - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    beta = np.arccos(np.clip(cb, -1.0, 1.0))
    alpha = np.arctan2(-dx, sz)
    delta = np.arctan2(L2 * np.sin(beta), L1 + L2 * np.cos(beta))
    return phi, alpha - delta, -beta


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or not _S:
        _reset()
        _S["x0"] = float(obs["com_xyz"][0])
    dt = obs["dt"]
    g = obs["gym_obs"]
    _d.qpos[0:2] = 0.0
    _d.qpos[2:] = g[0:22]
    _d.qvel[:] = g[22:45]
    mujoco.mj_forward(_m, _d)
    mujoco.mj_subtreeVel(_m, _d)
    com = _d.subtree_com[0].copy()
    vcom = _d.subtree_linvel[0].copy()
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    feet = [_d.geom_xpos[_FOOT_G[0]].copy(), _d.geom_xpos[_FOOT_G[1]].copy()]
    hips = [_d.xpos[_HIP_B[0]].copy(), _d.xpos[_HIP_B[1]].copy()]
    Rp = _d.xmat[_PELVIS].reshape(3, 3)
    contact = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]
    # absolute horizontal frame: shift so the model COM matches the measured one
    off = np.array([obs["com_xyz"][0] - com[0], obs["com_xyz"][1] - com[1], 0.0])
    com = com + off
    feet = [f + off for f in feet]
    hips = [h + off for h in hips]

    tau = np.zeros(17)
    t = obs["time"]

    # ---- gait state machine ----
    S = _S
    if not S["started"]:
        if t > P["T_START"] and (contact[0] or contact[1]):
            S["started"] = True
            S["t0"] = t
            # first swing: the leg on the side the COM is NOT over
            S["stance"] = 1 if (com[1] - 0.5 * (feet[0][1] + feet[1][1])) >= 0 else 0
            S["t_ph"] = 0.0
            S["p0"] = feet[1 - S["stance"]].copy()
    else:
        sw = 1 - S["stance"]
        T = P["T_SWING"]
        if S["ds"] > 0.0:
            S["ds"] -= dt
            if S["ds"] <= 0.0:          # trailing foot lifts off: swing starts
                S["t_ph"] = 0.0
                S["p0"] = feet[sw].copy()
                S["qdes_prev"] = None
        else:
            S["t_ph"] += dt
            if (S["t_ph"] > 0.5 * T and contact[sw]) or S["t_ph"] > 2.5 * T:
                S["stance"] = sw
                S["n_steps"] += 1
                if P["T_DS"] > 0.0:
                    S["ds"] = P["T_DS"]
                    S["t_ph"] = 0.0
                else:
                    S["t_ph"] = 0.0
                    S["p0"] = feet[S["stance"] ^ 1].copy()
                    S["qdes_prev"] = None

    st = S["stance"]
    sw = 1 - st
    T = P["T_SWING"]

    # ---- torso attitude (world) ----
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]
    # lean the upper body over the stance hip: roll > 0 tilts the torso toward -y (right)
    r_des = 0.0
    if S["started"]:
        r_des = P["ROLL_LEAN"] * (1.0 if st == 0 else -1.0)
    tq_w = np.array([-P["KP_R"] * (roll - r_des) - P["KD_R"] * w[0],
                     -P["KP_T"] * (pitch - P["LEAN"]) - P["KD_T"] * w[1],
                     -60.0 * yaw - 6.0 * w[2]])
    tq_p = Rp.T @ tq_w   # desired torque on pelvis/torso in pelvis axes

    # gravity / bias compensation
    bias = _d.qfrc_bias[_DOF]

    # ---- abdomen + arms ----
    kp_ab = np.array([P["KP_AB"], 40.0, P["KP_AB"]])
    kd_ab = np.array([P["KD_AB"], 2.0, P["KD_AB"]])
    # lean the torso over the stance hip (abdomen_x > 0 tilts the torso toward +y = left)
    ss = 0.0
    if S["started"]:
        ss = 1.0 if S["stance"] == 1 else -1.0
        ss *= min(S["t_ph"] / 0.08, 1.0) if S["n_steps"] > 0 else 1.0
    ab_ref = np.array([0.0, 0.0, P["AB_LEAN"] * ss])
    tau[0:3] = kp_ab * (ab_ref - q[0:3]) - kd_ab * qd[0:3] + bias[0:3]
    arm_ref = _ARM_REF.copy()
    if S["started"] and P["ARM_SWING"] != 0.0:
        # contralateral arm swing: while the left leg swings forward (right stance) the right arm
        # goes forward and the left arm back, cancelling the leg's yaw momentum
        prog = _smooth(S["t_ph"] / P["T_SWING"]) if S["ds"] <= 0.0 else 0.0
        a = P["ARM_SWING"] * (2.0 * prog - 1.0) * (1.0 if S["stance"] == 0 else -1.0)
        arm_ref[0:2] += a * _ARM_FWD[0]
        arm_ref[3:5] -= a * _ARM_FWD[1]
    # arms toward the stance side: shoulder2 (right axis (0,-1,1), left (0,1,1))
    arm_ref[1] += P["ARM_SH"] * ss
    arm_ref[4] += P["ARM_SH"] * ss
    tau[11:17] = 15.0 * (arm_ref - q[11:17]) - 1.0 * qd[11:17]

    if not S["started"]:
        # stand: both legs to a nominal stance
        for side in (0, 1):
            ix, iz, iy, ik = _LEG[side]
            pt = np.array([com[0], hips[side][1], hips[side][2] - P["H_STAND"]])
            phi, th, kn = _leg_ik(Rp.T @ (pt - hips[side]), side)
            tau[ix] = 150 * (phi - q[ix]) - 6 * qd[ix]
            tau[iz] = 60 * (0 - q[iz]) - 3 * qd[iz]
            tau[iy] = 250 * (th - q[iy]) - 10 * qd[iy]
            tau[ik] = 200 * (kn - q[ik]) - 6 * qd[ik]
        return np.clip(tau / _GEAR, -0.4, 0.4)

    # ---- foot placement ----
    # mass centre and velocity of the body without the swing leg (the swing leg's own motion
    # must not move its target: that loop is unstable)
    b = _HIP_B[sw]
    ml = _m.body_subtreemass[b]
    c_ns = (_MASS * com - ml * (_d.subtree_com[b] + off)) / (_MASS - ml)
    v_ns = (_MASS * vcom - ml * _d.subtree_linvel[b]) / (_MASS - ml)
    zc = max(c_ns[2] - (feet[st][2] - R_FOOT), 0.5)
    om = np.sqrt(G / zc)
    p_st = feet[st]
    xi = c_ns[:2] + v_ns[:2] / om
    trem = max(T - S["t_ph"], 0.0)
    xi_T = p_st[:2] + (xi - p_st[:2]) * np.exp(om * trem)
    eT = np.exp(om * T)
    v_des = P["V_DES"] if com[0] - S["x0"] < P["X_SLOW"] else P["V_SLOW"]   # ease off once far enough
    bx = v_des * T / (eT - 1.0)
    by = P["STEP_W"] / (1.0 + eT)
    side_sign = -1.0 if sw == 0 else 1.0     # right foot to -y of the capture point
    tgt = np.array([xi_T[0] - bx + P["K_VX"] * (v_ns[0] - v_des),
                    xi_T[1] + side_sign * by + P["K_VY"] * v_ns[1]])
    # kinematic limits relative to the stance foot and the hip's reach
    tgt[0] = np.clip(tgt[0], p_st[0] - 0.35, p_st[0] + 0.45)
    hc = 0.5 * (hips[0][:2] + hips[1][:2])
    tgt[0] = np.clip(tgt[0], hc[0] - P["REACH"], hc[0] + P["REACH"])
    if sw == 0:
        tgt[1] = np.clip(tgt[1], p_st[1] - P["W_MAX"], p_st[1] - 0.10)
    else:
        tgt[1] = np.clip(tgt[1], p_st[1] + 0.10, p_st[1] + P["W_MAX"])

    # ---- swing trajectory ----
    s = S["t_ph"] / T
    p0 = S["p0"]
    ground = p_st[2] - R_FOOT
    z_top = ground + R_FOOT + P["CLEAR"]
    hb = _smooth((s - P["S_X0"]) / (P["S_X"] - P["S_X0"]))
    pxy = p0[:2] + (tgt - p0[:2]) * hb
    s_dn = P["S_X"] - 0.05
    if s < P["S_Z"]:
        z = p0[2] + (z_top - p0[2]) * _smooth(s / P["S_Z"])
    elif s < s_dn:
        z = z_top
    else:
        z = z_top - (P["CLEAR"] + 0.06) * min((s - s_dn) / (0.95 - s_dn), 1.5)
    p_sw = np.array([pxy[0], pxy[1], z])
    d_sw = Rp.T @ (p_sw - hips[sw])
    phi, th, kn = _leg_ik(d_sw, sw)
    qdes = np.array([phi, th, kn])
    prev = S.get("qdes_prev")
    qd_des = (qdes - prev) / dt if (prev is not None and S["t_ph"] > dt * 1.5) else np.zeros(3)
    qd_des = np.clip(qd_des, -8.0, 8.0) * P["KFF"]
    S["qdes_prev"] = qdes
    ix, iz, iy, ik = _LEG[sw]
    tau_sw = np.zeros(4)
    k = P["KSW"]
    tau_sw[0] = k * (150 * (phi - q[ix]) + 6 * (qd_des[0] - qd[ix])) + bias[ix]
    tau_sw[1] = k * (60 * (0.0 - q[iz]) - 3 * qd[iz]) + bias[iz]
    tau_sw[2] = k * (250 * (th - q[iy]) + 12 * (qd_des[1] - qd[iy])) + bias[iy]
    tau_sw[3] = k * (150 * (kn - q[ik]) + 5 * (qd_des[2] - qd[ik])) + bias[ik]
    in_ds = S["ds"] > 0.0
    if in_ds:
        # trailing leg stays planted: damped hip roll, shares weight through its knee
        tau_sw[0] = -6 * qd[ix] + bias[ix]
        tau_sw[1] = 60 * (0.0 - q[iz]) - 3 * qd[iz]
        tau_sw[2] = -8 * qd[iy]
        mujoco.mj_jacGeom(_m, _d, _jacp, _jacr, _FOOT_G[sw])
        tau_sw[3] = -_jacp[2, _DOF[ik]] * P["DS_SHARE"] * _MASS * G - 5.0 * qd[ik]
    tau[ix], tau[iz], tau[iy], tau[ik] = tau_sw

    # ---- stance leg ----
    jx, jz, jy, jk = _LEG[st]
    # hip pitch: torso torque about pelvis y minus swing hip
    tau[jy] = -tq_p[1] - tau[iy]
    # hip roll: right axis +x, left axis -x
    if st == 0:
        tau[jx] = P["SIMB_X"] * tau[ix] - tq_p[0]
    else:
        tau[jx] = tq_p[0] + P["SIMB_X"] * tau[ix]
    # hip yaw: the spherical foot spins freely, so the stance hip_z only keeps the leg aligned
    tau[jz] = 60 * (0.0 - q[jz]) - 3 * qd[jz]
    if in_ds:
        tz = -P["K_YAW"] * yaw - P["D_YAW"] * w[2]   # desired pelvis yaw torque
        dy = 1.0 if st == 0 else -1.0                 # right hip pitch pushes the right hip forward
        tau[jy] += dy * tz
        tau[iy] -= dy * tz
    # knee: virtual vertical spring on the hip height above the stance foot
    h = hips[st][2] - p_st[2]
    hd = vcom[2]
    ramp = min((t - S["t0"]) / 0.6, 1.0)
    h_des = P["H_STAND"] + (P["H_LEG"] - P["H_STAND"]) * ramp
    s_late = S["t_ph"] / T - 0.85
    if s_late > 0:
        h_des -= P["SINK"] * min(s_late / 0.4, 1.0)   # let a late swing foot reach the ground
    Fz = _MASS * G + P["KP_H"] * (h_des - h) - P["KD_H"] * hd
    Fz = max(Fz, 0.0)
    mujoco.mj_jacGeom(_m, _d, _jacp, _jacr, _FOOT_G[st])
    J = _jacp[2, _DOF[jk]]
    tau[jk] = -J * Fz + bias[jk] * 0.0 - 5.0 * qd[jk]
    # keep stance knee away from full extension
    if q[jk] > -0.15:
        tau[jk] += 300 * (-0.15 - q[jk])

    S["dbg"] = dict(tgt=tgt, xi_T=xi_T, p_sw=p_sw, h=h, om=om)
    return np.clip(tau / _GEAR, -0.4, 0.4)


def H0(P):
    return P["H_LEG"]
