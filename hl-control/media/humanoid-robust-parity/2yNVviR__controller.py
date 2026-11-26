"""Hand-designed SIMBICON / Raibert-style walker for the MuJoCo Humanoid (no learning; gains found by
derivative-free search over PARAMS on the visible scenarios).

Control law: PD / virtual torques, converted to commands with the *observed* gears (obs["gear"]),
so weaker motors are compensated until the +-0.4 clip.

* Gait clock: two-state machine (right / left stance).  The swing ends on swing-foot touchdown after
  T_MIN or unconditionally at T_MAX (short steps, ~0.2 s, limit the lateral divergence of the point feet).
* Stance leg: hip_y / hip_x / hip_z do not track angles; they apply the torque that holds the torso
  pitch at PITCH_DES (KP_TORSO, KD_TORSO), the pelvis roll at +-ROLL_LEAN (KP_ROLL, KD_ROLL) and the
  heading at 0 (KYAW), minus SWC x the swing-hip torque (SIMBICON torque balance).  Knee holds KNEE_ST.
* Swing foot (Raibert / capture point): the COM state at touchdown is predicted with the linear
  inverted pendulum around the stance foot, and the foot is aimed at
      x = X_V * vx + K_V * (vx - V_DES),   y = side * WIDTH + K_VY * vy   (relative to that COM),
  with v = VMIX * COM velocity + (1 - VMIX) * pelvis velocity (filter VF).  The target is reached on a
  smoothstep (SW_FRAC, SW_FRAC_Y of the step) with a CLEAR-high arc, converted to hip_x/hip_y/knee by
  closed-form 3-joint IK on a kinematic copy of the model, and tracked with PD (KP_SW, KP_SWY, KP_KN).
  The swing hip_z aligns the leg with the world x axis (YAW_ALIGN) and, for lateral targets beyond
  LAT0 +- LAT_DB, turns the leg (K_HZ) so the strong hip_y/knee do the sideways step.
* Abdomen (KP_ABD) leans the torso over the stance hip by ABD_LEAN (the hip_x motors, 40 N m, cannot
  hold the pelvis level alone); arms are held by light PD.
* Disturbance mode: when a fast (0.1 s) and a slow (1 s) low-pass of the COM velocity differ by more
  than D_TH (a push), the walker uses the recovery set R_* (swing stiffness, speed target, capture
  gains, step time) for R_DUR seconds.
"""

import numpy as np
import mujoco
from hl_control.envs.humanoid import XML_PATH

PARAMS = dict(
    K_HZ=0,
    LAT0=-0.07,
    LAT_DB=0.07,
    K_DY=0.0, D_TH=0.25,
    R_DUR=0.6,
    R_KP_SWY=100,
    R_V_DES=0.4491,
    R_X_V=0.1815,
    R_K_VY=0.05269,
    R_T_MAX=0.2232,
    V_DES=0.4491,
    X_V=0.1815,
    K_V=0.07157,
    WIDTH=0.15,
    K_VY=0.05269,
    CLEAR=0.04505,
    T_MIN=0.1292,
    T_MAX=0.2232,
    PITCH_DES=0.0852,
    KNEE_ST=-0.1742,
    KP_TORSO=535.5,
    KD_TORSO=43.57,
    KP_ROLL=312.9,
    KD_ROLL=45.08,
    ROLL_LEAN=0.1581,
    ABD_LEAN=-0.2024,
    KYAW=59.57,
    SWC=0.4984,
    YAW_ALIGN=1,
    VMIX=0.7103,
    KP_SW=374.6,
    KP_SWY=100,
    KP_KN=68.37,
    KP_ABD=272,
    SW_FRAC=0.3833,
    SW_FRAC_Y=0.7026,
    VF=0.7709,
)

_M = mujoco.MjModel.from_xml_path(str(XML_PATH))
_D = mujoco.MjData(_M)
_J = {n: i for i, n in enumerate(("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y",
      "right_knee", "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
      "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"))}
_THIGH = (_M.body("right_thigh").id, _M.body("left_thigh").id)
_PELVIS = _M.body("pelvis").id
L1, L2, RFOOT = 0.383, 0.37, 0.075
_S = {}


def leg_ik(v, side, hz=0.0):
    """Hip-to-foot-centre vector in the pelvis frame and a hip yaw -> (hip_x, hip_y, knee). side 0 right, 1 left.

    Joint chain hip_x (outermost), hip_z, hip_y; the left leg's x and z axes are negated."""
    px, py, pz = v
    ez = hz if side == 0 else -hz
    A, B, C = py, pz, px * np.tan(ez)
    r = np.hypot(A, B)
    dl = np.arctan2(B, A)
    ex = dl + np.arccos(np.clip(C / max(r, 1e-6), -1, 1))          # branch that reduces to atan2(py, -pz) at ez = 0
    ex = (ex + np.pi) % (2 * np.pi) - np.pi
    c, s_ = np.cos(ex), np.sin(ex)
    w = np.array([px, py * c + pz * s_, -py * s_ + pz * c])
    cz, sz = np.cos(ez), np.sin(ez)
    x2, z2 = w[0] * cz + w[1] * sz, w[2]
    d = np.clip(np.hypot(x2, z2), 0.35, L1 + L2 - 0.005)
    kb = np.pi - np.arccos(np.clip((L1 * L1 + L2 * L2 - d * d) / (2 * L1 * L2), -1, 1))
    a = np.arctan2(x2, -z2)
    phi = np.arccos(np.clip((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1))
    return (ex if side == 0 else -ex), -(a + phi), -kb


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    # push / disturbance detector: fast vs slow low-pass of the mass-centre velocity
    vc = np.asarray(obs["com_velocity_xy"], dtype=float)
    if obs["step"] == 0 or "vfast" not in _S:
        _S.clear(); _S.update(stance=0, t=0.0, pel=None, proll=None, vfast=vc.copy(), vslow=vc.copy(), rec=0.0)
    _S["vfast"] += 0.15 * (vc - _S["vfast"]); _S["vslow"] += 0.015 * (vc - _S["vslow"])
    _S["vmed"] = _S.get("vmed", vc.copy()) + 0.05 * (vc - _S.get("vmed", vc.copy()))
    if obs["step"] > 30 and np.linalg.norm(_S["vfast"] - _S["vslow"]) > P["D_TH"]:
        _S["rec"] = P["R_DUR"]
    _S["rec"] = max(_S["rec"] - obs["dt"], 0.0)
    if _S["rec"] > 0:
        P = {**P, "KP_SWY": P["R_KP_SWY"], "V_DES": P["R_V_DES"], "X_V": P["R_X_V"], "K_VY": P["R_K_VY"], "T_MAX": P["R_T_MAX"]}
    dt = obs["dt"]
    q = obs["joint_angles"]; qd = obs["joint_rates"]
    g = obs["gym_obs"]
    _D.qpos[2:] = g[:22]; _D.qpos[:2] = 0
    mujoco.mj_kinematics(_M, _D); mujoco.mj_comPos(_M, _D)
    com = _D.subtree_com[0].copy()
    pel = obs["com_xyz"] + (_D.xpos[_PELVIS] - com)            # absolute pelvis position
    if _S.get("pel") is None:
        _S["vp"] = np.zeros(2)
    else:
        _S["vp"] += P["VF"] * ((pel[:2] - _S["pel"][:2]) / dt - _S["vp"])
    _S["pel"] = pel
    v = P["VMIX"] * obs["com_velocity_xy"] + (1 - P["VMIX"]) * _S["vp"]
    w = obs["root_angular_velocity"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    contact = (obs["foot_contact"], obs["foot_left_contact"])

    st = _S["stance"]; sw = 1 - st
    _S["t"] += dt
    if (_S["t"] > P["T_MIN"] and contact[sw]) or _S["t"] > P["T_MAX"]:
        _S["stance"] = st = sw; sw = 1 - st; _S["t"] = 0.0
        _S["lift"] = None
    feet_abs = (obs["com_xyz"] + obs["foot_xyz_rel"], obs["com_xyz"] + obs["foot_left_xyz_rel"])
    if _S.get("lift") is None:
        _S["lift"] = feet_abs[sw][:2].copy()
    s = min(_S["t"] / P["T_MAX"], 1.0)

    tau = np.zeros(17)
    # swing foot target (world, relative to COM)
    side = 1.0 if sw == 1 else -1.0
    # LIP prediction of the COM state (relative to the stance foot) at the end of the swing
    om = np.sqrt(9.81 / max(obs["com_xyz"][2], 0.5))
    trem = max(P["T_MAX"] - _S["t"], 0.0)
    ch, sh = np.cosh(om * trem), np.sinh(om * trem)
    rel = obs["com_xyz"][:2] - feet_abs[st][:2]
    relp = rel * ch + v / om * sh
    vp = rel * om * sh + v * ch
    xt = P["X_V"] * vp[0] + P["K_V"] * (vp[0] - P["V_DES"])
    yt = side * P["WIDTH"] + P["K_VY"] * vp[1] + P["K_DY"] * _S["vmed"][1]
    zt = RFOOT + P["CLEAR"] * np.sin(np.pi * s) - 0.03 * (s > 0.7)
    tgt = feet_abs[st][:2] + relp + np.array([xt, yt])
    ax = min(_S["t"] / (P["SW_FRAC"] * P["T_MAX"]), 1.0); ax = ax * ax * (3 - 2 * ax)
    ay = min(_S["t"] / (P["SW_FRAC_Y"] * P["T_MAX"]), 1.0); ay = ay * ay * (3 - 2 * ay)
    fxy = np.array([(1 - ax) * _S["lift"][0] + ax * tgt[0], (1 - ay) * _S["lift"][1] + ay * tgt[1]]) - obs["com_xyz"][:2] + com[:2]
    foot_w = np.array([fxy[0], fxy[1], zt])
    hip_w = _D.xpos[_THIGH[sw]]
    Rp = _D.xmat[_PELVIS].reshape(3, 3)
    pyaw = np.arctan2(Rp[1, 0], Rp[0, 0])
    vleg = Rp.T @ (foot_w - hip_w)
    lat = vleg[1] if sw == 0 else -vleg[1]                      # lateral foot offset, + = inward (towards the other leg)
    ex = lat - P["LAT0"]
    hz = (-pyaw if sw == 0 else pyaw) * P["YAW_ALIGN"] + P["K_HZ"] * np.sign(ex) * max(abs(ex) - P["LAT_DB"], 0.0)
    hz = float(np.clip(hz, -0.9, 0.5)) if sw == 0 else float(np.clip(hz, -0.5, 0.9))
    hx, hy, kn = leg_ik(vleg, sw, hz)
    hx = np.clip(hx, -0.40, 0.06)
    names = ("right", "left")[sw]
    jx, jz, jy, jk = _J[names + "_hip_x"], _J[names + "_hip_z"], _J[names + "_hip_y"], _J[names + "_knee"]
    tau[jx] = P["KP_SW"] * (hx - q[jx]) - 0.06 * P["KP_SW"] * qd[jx]
    tau[jz] = 30 * (hz - q[jz]) - 0.7 * qd[jz]
    tau[jy] = P["KP_SWY"] * (hy - q[jy]) - 0.06 * P["KP_SWY"] * qd[jy]
    tau[jk] = P["KP_KN"] * (kn - q[jk]) - 0.03 * P["KP_KN"] * qd[jk]
    # stance leg: torso attitude
    namest = ("right", "left")[st]
    sx, sz, sy, sk = _J[namest + "_hip_x"], _J[namest + "_hip_z"], _J[namest + "_hip_y"], _J[namest + "_knee"]
    t_pitch = P["KP_TORSO"] * (pitch - P["PITCH_DES"]) + P["KD_TORSO"] * w[1]
    tau[sy] = t_pitch - P["SWC"] * tau[jy]
    zp = Rp[:, 2]
    proll = np.arctan2(-zp[1], zp[2])
    prate = 0.0 if _S.get("proll") is None else (proll - _S["proll"]) / dt
    _S["proll"] = proll
    roll_des = P["ROLL_LEAN"] * (1.0 if st == 0 else -1.0)
    t_roll = P["KP_ROLL"] * (proll - roll_des) + P["KD_ROLL"] * prate      # wanted torque about +x on the thigh (right-hip sign)
    t_yaw = P["KYAW"] * yaw + 0.1 * P["KYAW"] * w[2]
    if st == 0:
        tau[sx] = t_roll + tau[jx]
        tau[sz] = t_yaw + tau[jz]
    else:
        tau[sx] = -t_roll + tau[jx]
        tau[sz] = -t_yaw + tau[jz]
    tau[sk] = 300 * (P["KNEE_ST"] - q[sk]) - 5 * qd[sk]
    # abdomen, arms
    abd_t = {"abdomen_y": 0.0, "abdomen_z": 0.0, "abdomen_x": -P["ABD_LEAN"] * (1.0 if st == 0 else -1.0)}
    for n in ("abdomen_y", "abdomen_z", "abdomen_x"):
        tau[_J[n]] = P["KP_ABD"] * (abd_t[n] - q[_J[n]]) - 0.05 * P["KP_ABD"] * qd[_J[n]]
    for i in range(11, 17):
        tau[i] = 10 * (0 - q[i]) - 0.3 * qd[i]
    return np.clip(tau / np.asarray(obs["gear"]), -0.4, 0.4)
