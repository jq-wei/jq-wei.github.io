"""Model-based walking controller for the MuJoCo Humanoid on uneven ground (torque control, point feet).

Each call copies the measured posture (gym_obs) into a kinematic copy of the plant (mj_kinematics) to
get joint axes, hip positions and gravity moments; no dynamics are simulated. Then:

  * Gait: a two-state machine (right / left stance) after T_STAND s of double support. The swing leg
    becomes the stance leg when it touches down after T_MIN s of swing, or after T_MAX s regardless.
  * Torso (SIMBICON trick): a world-frame PD on the torso up-vector (lean TORSO_PITCH forward, LEAN
    toward the stance side; gains KP_TORSO/KD_TORSO in pitch, KP_ROLL/KD_ROLL in roll). The stance hip
    supplies  -torso torque - SWC * swing-hip torque + GRAV * gravity moment of the upper body and
    swing leg about the stance hip, solved onto the stance hip_x / hip_y axes.
  * Foot placement (capture point): the swing foot is aimed, relative to the mass centre, at
        dx = TC * v_x + KV * (v_x - V_DES) + X_OFF - side * (K_YAW * yaw + KD_YAW * yaw_rate)
        dy = (TC + KV_LAT) * v_y + side * STEP_WIDTH
    (v low-passed with V_LP; side = +1 for the left foot). A point foot cannot resist yaw, so heading is
    steered by lengthening one side's steps and shortening the other's (orbital angular momentum).
    The foot follows a smoothstep path from lift-off over T_SW s with CLEAR of clearance, ending PRESS
    below the expected ground (stance-foot height plus the estimated slope), and afterwards keeps
    reaching down at DESC m/s until it finds the ground. Two-link leg inverse kinematics through the
    pelvis attitude and the measured hip_z gives joint targets, tracked by PD (KP_SW/KD_SW hip_y,
    KP_SWX/KD_SWX hip_x, KP_SWK/KD_SWK knee, KP_SWK_LATE after T_LATE) plus the leg's gravity torque.
  * Stance knee: PD (KP_KNEE/KD_KNEE) to STANCE_KNEE, ramped from the touchdown angle over T_KN s, plus
    the static gravity torque of everything above the knee. hip_z joints: weak PD (KP_HZ/KD_HZ); while
    both feet are down they add a heading-correcting twist (KZ_P, KZ_D).
  * Ground slope: estimated from the feet's height and x difference at each touchdown (filter SLOPE_LP);
    it shifts the expected ground height of the swing target. Optional slope adaptations of the torso
    lean (PITCH_SLOPE), step offset (K_SLOPE_X), downhill knee bend (KNEE_SLOPE) and downhill speed
    (V_SLOPE) are left at 0: none helped in testing. KNEE_FF scales the stance-knee gravity torque and
    KNEE_MINFLEX is the least swing-knee flexion the inverse kinematics may ask for.
  * Abdomen and arms: PD to zero (KP_AB/KD_AB; arms 20/2 N m per rad, rad/s).
"""

import numpy as np
import mujoco
from hl_control.envs.humanoid import XML_PATH

PARAMS = dict(
    TORSO_PITCH=0.05, KP_TORSO=300.0, KD_TORSO=30.0, KP_ROLL=300.0, KD_ROLL=30.0, LEAN=0.0, GRAV=1.0, TC=0.2,
    KV=0.15, V_DES=0.4, X_OFF=0.0, KV_LAT=0.1, STEP_WIDTH=0.1, T_SW=0.27, CLEAR=0.08, PRESS=0.03581,
    KNEE_MINFLEX=0.15, KP_SW=250.0, KD_SW=8.0, KP_SWX=250.0, KD_SWX=9.16365,
    STANCE_KNEE=-0.4, KP_KNEE=150.0, KD_KNEE=8.0, KP_SWK=100.0, KD_SWK=6.0, T_MIN=0.15, T_MAX=0.6, DESC=0.3,
    SLOPE_LP=0.3, PITCH_SLOPE=0.0, K_SLOPE_X=0.0, KZ_P=0.0, KZ_D=-2.14781, KP_HZ=10.0, KD_HZ=1.0,
    KNEE_SLOPE=0.0, V_SLOPE=0.0, T_LATE=0.15, KP_SWK_LATE=100.0, V_LP=1.0, T_STAND=0.3,
    T_KN=0.15, KNEE_FF=1.0, KP_AB=150.0, KD_AB=15.0, SWC=1.0, K_YAW=0.21385, KD_YAW=0.03,
)
P = PARAMS

_M = mujoco.MjModel.from_xml_path(str(XML_PATH))
_D = mujoco.MjData(_M)
GEAR = _M.actuator_gear[:, 0].copy()
NAMES = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
         "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
         "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
JID = np.array([_M.joint(n).id for n in NAMES])
LEG = ((3, 4, 5, 6), (7, 8, 9, 10))       # (hip_x, hip_z, hip_y, knee) motor indices
HX_SIGN = (1.0, -1.0)                      # hip_x axis is +x on the right leg, -x on the left
B = {_M.body(i).name: i for i in range(_M.nbody)}
UPPER = [B[n] for n in ("torso", "lwaist", "pelvis", "right_upper_arm", "right_lower_arm", "left_upper_arm", "left_lower_arm")]
LEG_BODIES = ([B["right_thigh"], B["right_shin"], B["right_foot"]], [B["left_thigh"], B["left_shin"], B["left_foot"]])
MASS = _M.body_mass
M_LEG = MASS[LEG_BODIES[0]].sum()
GZ = np.array([0.0, 0.0, -9.81])
ALL = list(range(1, _M.nbody))
L1, L2 = 0.383, 0.37
REST_KNEE = [[b for b in ALL if b not in LEG_BODIES[k][1:]] for k in (0, 1)]

_S = {}


def _com(bodies):
    m = MASS[bodies]
    return (m[:, None] * _D.xipos[bodies]).sum(0) / m.sum(), m.sum()


def _axis(mi):
    return _D.xaxis[JID[mi]]


def _hip_solve(leg, T):
    """hip_x, hip_y torques whose world torque best matches T (x, y components)."""
    hx, _, hy, _ = LEG[leg]
    A = np.stack([_axis(hx), _axis(hy)], axis=1)[:2]
    try:
        return np.linalg.solve(A, T[:2])
    except np.linalg.LinAlgError:   # degenerate posture (only once fallen): least squares instead
        return np.linalg.lstsq(A, T[:2], rcond=None)[0]


def _knee_grav(leg, share=1.0):
    kn = LEG[leg][3]
    c, m = _com(REST_KNEE[leg])
    return share * (_axis(kn) @ np.cross(c - _D.xanchor[JID[kn]], m * GZ))


def controller(obs: dict) -> np.ndarray:
    if obs["step"] == 0 or not _S:
        _S.clear()
        _S.update(stance=0, t=0.0, slope=0.0)
    dt = obs["dt"]
    q = np.asarray(obs["joint_angles"], dtype=float)
    qd = np.asarray(obs["joint_rates"], dtype=float)
    _D.qpos[:2] = 0.0
    _D.qpos[2:] = obs["gym_obs"][:22]
    mujoco.mj_kinematics(_M, _D)
    standing = obs["time"] < P["T_STAND"]

    # ---- gait state machine ----
    contact = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))
    _S["t"] = 0.0 if standing else _S["t"] + dt
    st = _S["stance"]
    sw = 1 - st
    if (_S["t"] > P["T_MIN"] and contact[sw]) or _S["t"] > P["T_MAX"]:
        f_new = np.asarray(obs["foot_xyz_rel"] if sw == 0 else obs["foot_left_xyz_rel"])
        f_old = np.asarray(obs["foot_xyz_rel"] if st == 0 else obs["foot_left_xyz_rel"])
        ddx = f_new[0] - f_old[0]
        if contact[sw] and contact[st] and abs(ddx) > 0.08:
            _S["slope"] += P["SLOPE_LP"] * (float(np.clip((f_new[2] - f_old[2]) / ddx, -0.15, 0.15)) - _S["slope"])
        _S["stance"] = st = sw
        sw = 1 - st
        _S["t"] = 0.0
        _S["kn0"] = float(q[LEG[st][3]])
    t = _S["t"]

    feet_rel = (np.asarray(obs["foot_xyz_rel"]), np.asarray(obs["foot_left_xyz_rel"]))
    d = -feet_rel[st][:2]                     # stance foot -> mass centre (world x, y)
    v_raw = np.asarray(obs["com_velocity_xy"], dtype=float)
    _S["vf"] = v_raw if "vf" not in _S else _S["vf"] + P["V_LP"] * (v_raw - _S["vf"])
    v = _S["vf"]
    tau = np.zeros(17)

    # ---- torso attitude torque (world frame) ----
    up = np.asarray(obs["up_vector"])
    side_st = -1.0 if st == 0 else 1.0        # stance foot side (y sign)
    lean = 0.0 if standing else P["LEAN"] * side_st
    pd = P["TORSO_PITCH"] + P["PITCH_SLOPE"] * _S["slope"]
    up_d = np.array([np.sin(pd), np.sin(lean), 1.0])
    up_d /= np.linalg.norm(up_d)
    e = np.cross(up, up_d)
    w = np.asarray(obs["root_angular_velocity"])
    tau_t = np.array([P["KP_ROLL"] * e[0] - P["KD_ROLL"] * w[0], P["KP_TORSO"] * e[1] - P["KD_TORSO"] * w[1], 0.0])

    # ---- swing leg: Cartesian foot target (capture-point placement) -> leg inverse kinematics ----
    Rp = _D.xmat[B["pelvis"]].reshape(3, 3)
    hx, hz, hy, kn = LEG[sw]
    side = -1.0 if sw == 0 else 1.0           # swing foot side (y sign)
    p_hip = _D.xanchor[JID[hy]]
    c_copy, _ = _com(ALL)
    st_w = c_copy + feet_rel[st]
    sw_w = c_copy + feet_rel[sw]
    if standing:
        target = np.array([p_hip[0], p_hip[1], st_w[2]])
    else:
        if t == 0.0 or "lift" not in _S:
            _S["lift"] = sw_w.copy()
        tc = P["TC"]
        yaw, yaw_rate = obs["yaw"], float(np.asarray(obs["root_angular_velocity"])[2])
        dx = tc * v[0] + P["KV"] * (v[0] - P["V_DES"] - P["V_SLOPE"] * min(_S["slope"], 0.0)) + P["X_OFF"] + P["K_SLOPE_X"] * _S["slope"] - side * (P["K_YAW"] * yaw + P["KD_YAW"] * yaw_rate)
        dy = tc * v[1] + P["KV_LAT"] * v[1] + side * P["STEP_WIDTH"]
        dx = float(np.clip(dx, -0.25, 0.55))
        dy = float(np.clip(dy, -0.35, 0.35))
        s_ph = min(t / P["T_SW"], 1.0)
        lift = _S["lift"]
        goal = np.array([c_copy[0] + dx, c_copy[1] + dy])
        blend = s_ph * s_ph * (3 - 2 * s_ph)
        xy = lift[:2] + (goal - lift[:2]) * min(1.0, 1.5 * blend)
        z_gnd = st_w[2] + _S["slope"] * (goal[0] - st_w[0])     # expected ground (foot-centre) height at the target
        if t <= P["T_SW"]:
            z = st_w[2] + (z_gnd - st_w[2]) * blend + P["CLEAR"] * np.sin(np.pi * s_ph) - P["PRESS"] * s_ph
        else:   # late: keep reaching down until the foot finds the ground
            z = z_gnd - P["PRESS"] - P["DESC"] * (t - P["T_SW"])
        target = np.array([xy[0], xy[1], z])
    r_p = Rp.T @ (target - p_hip)
    L = float(np.clip(np.linalg.norm(r_p), 0.3, L1 + L2 - 1e-3))
    cphi = (L * L - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    phi = float(np.arccos(np.clip(cphi, -1.0, 1.0)))
    if not standing:
        phi = max(phi, P["KNEE_MINFLEX"])
    a_, c_ = -L2 * np.sin(phi), -L1 - L2 * np.cos(phi)
    # thigh rotation = Rx(sx*hx) Rz(sx*hz) Ry(hy) with sx = +1 (right) / -1 (left); hz is the measured hip_z
    sx = HX_SIGN[sw]
    r_p = r_p * (L / max(np.linalg.norm(r_p), 1e-6))
    hz_m = sx * float(q[hz])
    X1 = float(np.clip(r_p[0] / np.cos(hz_m), -0.99 * L, 0.99 * L))
    Z1 = -np.sqrt(max(L * L - X1 * X1, 1e-6))
    pp, qq = X1 * np.sin(hz_m), Z1
    q_hx = sx * float((np.arctan2(r_p[2], r_p[1]) - np.arctan2(qq, pp) + np.pi) % (2 * np.pi) - np.pi)
    q_hy = np.arctan2(X1, Z1) - np.arctan2(a_, c_)
    q_hy = float((q_hy + np.pi) % (2 * np.pi) - np.pi)
    q_kn = -phi
    sw_tau = np.array([P["KP_SWX"] * (q_hx - q[hx]) - P["KD_SWX"] * qd[hx],
                       P["KP_HZ"] * (0.0 - q[hz]) - P["KD_HZ"] * qd[hz],
                       P["KP_SW"] * (q_hy - q[hy]) - P["KD_SW"] * qd[hy],
                       (P["KP_SWK"] if t < P["T_LATE"] else P["KP_SWK_LATE"]) * (q_kn - q[kn]) - P["KD_SWK"] * qd[kn]])
    if not standing:   # gravity feed-forward of the hanging swing leg
        lb = LEG_BODIES[sw]
        for k, bodies in enumerate((lb, lb, lb, lb[1:])):
            c, m = _com(bodies)
            mi = LEG[sw][k]
            sw_tau[k] += -_axis(mi) @ np.cross(c - _D.xanchor[JID[mi]], m * GZ)
    for k, mi in enumerate(LEG[sw]):
        tau[mi] = sw_tau[k]

    # ---- stance leg ----
    shx, shz, shy, skn = LEG[st]
    if standing:
        # both legs share the torso torque and the weight; the "swing" leg above only holds its pose
        for leg in (0, 1):
            lhx, lhz, lhy, lkn = LEG[leg]
            ts = _hip_solve(leg, -0.5 * tau_t)
            tau[lhx] = ts[0]
            tau[lhy] = ts[1]
            tau[lhz] = 10.0 * (0.0 - q[lhz]) - 1.0 * qd[lhz]
            tau[lkn] = P["KP_KNEE"] * (P["STANCE_KNEE"] - q[lkn]) - P["KD_KNEE"] * qd[lkn] + _knee_grav(leg, 0.5)
    else:
        T_sw = sum(sw_tau[k] * _axis(LEG[sw][k]) for k in range(3))
        p_st = _D.xanchor[JID[shx]]
        p_sw = _D.xanchor[JID[hx]]
        c_up, m_up = _com(UPPER)
        gam = np.cross(c_up - p_st, m_up * GZ) + np.cross(p_sw - p_st, M_LEG * GZ)
        T_st = -tau_t - P["SWC"] * T_sw + P["GRAV"] * gam
        ts = _hip_solve(st, T_st)
        tau[shx], tau[shy] = ts
        tau[shz] = P["KP_HZ"] * (0.0 - q[shz]) - P["KD_HZ"] * qd[shz]
        kn0 = _S.get("kn0", P["STANCE_KNEE"])
        kn_goal = P["STANCE_KNEE"] + P["KNEE_SLOPE"] * min(_S["slope"], 0.0)
        kn_t = kn0 + (kn_goal - kn0) * min(1.0, t / P["T_KN"])
        tau[skn] = P["KP_KNEE"] * (kn_t - q[skn]) - P["KD_KNEE"] * qd[skn] + P["KNEE_FF"] * _knee_grav(st)

    # ---- yaw: while both feet are down, the two hip_z joints twist the pelvis toward heading 0 ----
    if not standing and contact[0] and contact[1]:
        yaw_now = obs["yaw"]
        mz = -P["KZ_P"] * np.arctan2(np.sin(yaw_now), np.cos(yaw_now)) - P["KZ_D"] * float(np.asarray(obs["root_angular_velocity"])[2])
        for leg in (0, 1):
            tau[LEG[leg][1]] += -0.5 * mz * (1.0 if leg == 0 else -1.0)

    # ---- abdomen and arms ----
    tau[0:3] = P["KP_AB"] * (0.0 - q[0:3]) - P["KD_AB"] * qd[0:3]
    tau[11:17] = 20.0 * (0.0 - q[11:17]) - 2.0 * qd[11:17]
    return np.clip(np.nan_to_num(tau / GEAR), -0.4, 0.4)
