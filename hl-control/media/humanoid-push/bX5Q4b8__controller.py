"""Point-foot humanoid walking by Raibert-style foot placement (LIPM) with Pratt-style torso control.

Control law (all joint torques = gear * command, command clipped to +-0.4):
* A private MuJoCo model gives forward kinematics (hips, feet relative to the mass centre).
* Gait: legs alternate stance/swing every ~T seconds (swing ends early on touchdown).
* Stance leg: hip torques regulate torso pitch/roll/yaw (PD, gains KP_*, KD_*, torque on the
  joint = +gain*error because the reaction acts on the pelvis); the knee holds a fixed leg length LEN.
* Swing leg: the foot follows a smooth path to the Raibert/capture-point target
  x_f = x_c + v*T/2 + K_X*(v - v_des) (lateral: +-WIDTH offset, gain K_Y), with lift height CLEAR;
  joint targets come from analytic leg IK and joint PD (KP_SW, KD_SW).
* Abdomen and arms are held at zero with weak PD.
"""
import numpy as np
import mujoco
from hl_control.envs.humanoid import XML_PATH

PARAMS = dict(
    T0=0.2365, T=0.2053, V_DES=0.5591, K_X=0.0861, A_X=0.3305, K_Y=0.5346, WIDTH=0.304,
    CLEAR=0.1236, LEN=0.7117, PITCH0=-0.0067, KP_P=270.6197, KD_P=34.7261, KP_R=310.5256,
    KD_R=62.8689, KP_Y=47.4245, KD_Y=4.0133, KP_SW=234.4543, KD_SW=2.0594, KP_K=100.0,
    KD_K=11.2406, KP_AB=212.9603, KD_AB=14.8459)

_M = mujoco.MjModel.from_xml_path(str(XML_PATH))
_D = mujoco.MjData(_M)
_ID = {n: _M.body(n).id for n in ("pelvis", "right_thigh", "left_thigh")}
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25.0])
L1, L2 = 0.383, 0.37
S = dict(stance=1, t=0.0, p0=None, side_flip=0)   # stance 0=right, 1=left


def reset():
    S.update(stance=1, t=0.0, p0=None)


def leg_ik(d):
    """Pelvis-frame vector hip->foot centre -> (hip_x, hip_y, knee) joint angles of the right leg convention."""
    n = np.linalg.norm(d)
    n = np.clip(n, abs(L1 - L2) + 1e-3, L1 + L2 - 1e-4)
    ck = (n * n - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    kab = np.arccos(np.clip(ck, -1, 1))
    wx, wz = -L2 * np.sin(kab), -L1 - L2 * np.cos(kab)
    th = np.arctan2(wx, -wz)
    a = np.arctan2(d[1], -d[2])
    uz = -np.hypot(d[1], d[2])
    ux = d[0]
    ux = np.clip(ux, -abs(wz) + 1e-3 if False else -n + 1e-4, n - 1e-4)
    b = th - np.arctan2(ux, -uz)
    return a, b, -kab


def smooth(s):
    s = np.clip(s, 0, 1)
    return s * s * (3 - 2 * s)


def controller(obs):
    P = PARAMS
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    _D.qpos[:2] = 0
    _D.qpos[2:] = obs["gym_obs"][:22]
    mujoco.mj_kinematics(_M, _D)
    mujoco.mj_comPos(_M, _D)
    com_k = _D.subtree_com[0].copy()
    R = _D.xmat[_ID["pelvis"]].reshape(3, 3)
    hip = [_D.xpos[_ID["right_thigh"]] - com_k, _D.xpos[_ID["left_thigh"]] - com_k]
    com = obs["com_xyz"]
    foot = [com + obs["foot_xyz_rel"], com + obs["foot_left_xyz_rel"]]
    contact = [obs["foot_contact"], obs["foot_left_contact"]]
    v = obs["com_velocity_xy"]
    dt = obs["dt"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]

    st = S["stance"]; sw = 1 - st
    if obs["step"] == 0:
        reset(); S["t"] = -P["T0"]; st, sw = S["stance"], 1 - S["stance"]
    if S["p0"] is None:
        S["p0"] = foot[sw].copy()
    S["t"] += dt
    T = P["T"]
    # touchdown -> swap roles
    if S["t"] > 0.5 * T and contact[sw]:
        S["stance"], S["t"] = sw, dt
        st, sw = sw, st
        S["p0"] = foot[sw].copy()
    ph = S["t"] / T

    tau = np.zeros(17)
    # abdomen + arms hold
    tau[0:3] = -P["KP_AB"] * q[0:3] - P["KD_AB"] * qd[0:3]
    tau[11:17] = -10.0 * q[11:17] - 1.0 * qd[11:17]

    sgn = [+1, -1]       # right/left sign for x and z hip axes
    idx = [3, 7]         # hip_x index of each leg (hip_x, hip_z, hip_y, knee follow)
    # ---- stance leg
    i = idx[st]
    tau[i] = sgn[st] * (P["KP_R"] * roll + P["KD_R"] * w[0])
    tau[i + 1] = sgn[st] * (P["KP_Y"] * yaw + P["KD_Y"] * w[2])
    tau[i + 2] = P["KP_P"] * (pitch - P["PITCH0"]) + P["KD_P"] * w[1]
    # knee: hold leg length
    d_st = R.T @ (foot[st] - (com + hip[st]))
    ln = P["LEN"]
    kd = leg_ik(np.array([0, 0, -ln]))[2]
    tau[i + 3] = P["KP_K"] * (kd - q[i + 3]) - P["KD_K"] * qd[i + 3]
    # ---- swing leg
    s = smooth(ph)
    tgt_t = max(T - S["t"], 0.0) + 0.0
    cpred = com[:2] + v * tgt_t
    sidey = -1 if sw == 0 else 1          # swing foot goes to its own side
    vx_err = v[0] - P["V_DES"]
    xf = cpred[0] + P["A_X"] * v[0] + P["K_X"] * vx_err
    yf = cpred[1] + P["K_Y"] * v[1] + sidey * P["WIDTH"] / 2
    zf = 0.075 + P["CLEAR"] * np.sin(np.pi * s) - 0.03 * (ph > 0.9)
    p0 = S["p0"]
    pt = np.array([p0[0] + (xf - p0[0]) * s, p0[1] + (yf - p0[1]) * s, 0.0])
    pt[2] = zf if ph < 1.0 else 0.04
    hip_w = com + hip[sw]
    dsw = R.T @ (pt - hip_w)
    a, b, k = leg_ik(dsw)
    i = idx[sw]
    qt = [sgn[sw] * a, 0.0, b, k]
    for j in range(4):
        kp = P["KP_SW"]; kdp = P["KD_SW"]
        if j == 1:
            kp, kdp = P["KP_Y"], P["KD_Y"]
        tau[i + j] = kp * (qt[j] - q[i + j]) - kdp * qd[i + j]
    cmd = np.clip(tau / GEAR, -0.4, 0.4)
    return cmd
