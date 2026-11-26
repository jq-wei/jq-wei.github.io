"""Model-based SIMBICON / capture-point walker for the MuJoCo Humanoid on proprioceptive sensing.

State estimation (heading frame = world frame rotated by the estimated yaw; x forward, y left)
  * attitude: roll/pitch from the IMU, yaw integrated from the gyro (Euler-rate kinematics).
  * kinematics: the robot's own MuJoCo model is posed with the measured joints and attitude;
    it gives the COM, the foot contact points (sphere bottoms) and their Jacobians.
  * root velocity: accelerometer integrated in the world frame, pulled with gain ``VF`` per
    step toward leg odometry (-velocity of a foot that has been in contact for >= ``NC``
    steps, i.e. a planted foot does not move). COM velocity v = root velocity + kinematic
    COM velocity relative to the root.

Control law (after a ``T0`` s settle with both legs straight)
  * FSM: alternating single support; swing ends after ``T`` s, or on swing-foot touchdown
    after ``TC*T`` s.
  * swing foot placement (capture point + speed regulation), relative to the COM:
        x_f = CP*v_x + KV*(v_x - v_d,x),  y_f = CP*v_y + KVL*(v_y - v_d,y) +- W
    with v_d = ``VD`` m/s along world x, converted into world leg angles and then into hip
    targets through the pelvis attitude and the knee bend; PD ``KP_H``/``KD_H`` on hip x/y.
  * swing knee: flexed to ``KNEE_SW`` for the first ``KPH`` of the swing, then extended;
    PD ``KP_K``/``KD_K``.  Stance knee: PD to ``KNEE_ST`` with ``KP_KS``/``KD_KS``.
  * stance hip: servoes the torso pitch (to ``PITCH_D``, gains ``KP_T``/``KD_T``) and roll
    (``KP_R``/``KD_R``) in the world, minus ``SWC`` times the swing-hip torque (SIMBICON).
  * hip z / abdomen z: PD ``KP_Z``/``KD_Z`` to zero; abdomen x/y and arms: fixed-posture PD.
  Gains are torques (N m/rad, N m s/rad), divided by the gear and clipped to +-0.4. Derivative
  gains are kept small because the light links make a 15 ms discrete PD chatter otherwise.
  The gains were set by hand, then refined by a derivative-free random search on rollouts.
"""

import numpy as np
import mujoco

from hl_control.envs.humanoid import XML_PATH, ACTUATED_JOINTS

PARAMS = dict(
    T=0.2169, PITCH_D=0.0072, KV=0.1383, VD=0.2048, KVL=0.1061, W=0.1048, VF=0.2401,
    KP_T=297.1751, KD_T=18.9433, KP_R=179.2888, KD_R=16.4238, CP=0.2703, KNEE_SW=-0.9613,
    KNEE_ST=-0.2365, KP_H=139.1693, KD_H=2.4187, KP_K=233.156, KD_K=3.0, KP_KS=921.5959,
    KD_KS=10.0, T0=0.232, KPH=0.4702, TC=0.512, SWC=1.0, KP_Z=24.2434, KD_Z=0.5, NC=2,
)

_M = mujoco.MjModel.from_xml_path(str(XML_PATH))
_D = mujoco.MjData(_M)
_QADR = np.array([_M.jnt_qposadr[_M.joint(n).id] for n in ACTUATED_JOINTS])
_VADR = np.array([_M.jnt_dofadr[_M.joint(n).id] for n in ACTUATED_JOINTS])
_FOOT_G = [_M.geom("right_foot").id, _M.geom("left_foot").id]
_FOOT_B = [_M.body("right_foot").id, _M.body("left_foot").id]
_PELVIS = _M.body("pelvis").id
_GEAR = _M.actuator_gear[:, 0].copy()
J = {n: i for i, n in enumerate(ACTUATED_JOINTS)}

_S = {}


def _reset():
    _S.clear()
    _S.update(ncon=[0, 0], vroot=np.zeros(3), yaw=0.0, stance=0, t_sw=0.0)


def _quat(roll, pitch):
    qx = np.array([np.cos(roll / 2), np.sin(roll / 2), 0, 0])
    qy = np.array([np.cos(pitch / 2), 0, np.sin(pitch / 2), 0])
    q = np.zeros(4)
    mujoco.mju_mulQuat(q, qy, qx)
    return q


def _estimate(obs):
    roll, pitch = obs["roll"], obs["pitch"]
    w = np.asarray(obs["gyro_body"])
    dt = obs["dt"]
    _S["yaw"] += dt * (np.sin(roll) * w[1] + np.cos(roll) * w[2]) / max(np.cos(pitch), 0.2)
    _D.qpos[:] = 0
    _D.qpos[3:7] = _quat(roll, pitch)
    _D.qpos[_QADR] = obs["joint_angles"]
    _D.qvel[:] = 0
    _D.qvel[3:6] = w
    _D.qvel[_VADR] = obs["joint_rates"]
    mujoco.mj_kinematics(_M, _D)
    mujoco.mj_comPos(_M, _D)
    com = _D.subtree_com[1].copy()
    jc = np.zeros((3, _M.nv))
    mujoco.mj_jacSubtreeCom(_M, _D, jc, 1)
    vcom = jc @ _D.qvel
    feet, vfeet = [], []
    for g, b in zip(_FOOT_G, _FOOT_B):
        p = _D.geom_xpos[g] - np.array([0, 0, 0.075])
        jp = np.zeros((3, _M.nv))
        mujoco.mj_jac(_M, _D, jp, None, p, b)
        feet.append(p.copy())
        vfeet.append(jp @ _D.qvel)
    R = _D.xmat[1].reshape(3, 3)
    acc = R @ np.asarray(obs["accel_body"]) + np.array([0.0, 0.0, -9.81])
    pel = _D.xmat[_PELVIS].reshape(3, 3)
    p_pitch = np.arctan2(pel[0, 2], pel[2, 2])
    p_roll = np.arctan2(-pel[1, 2], pel[2, 2])
    return dict(roll=roll, pitch=pitch, w=w, com=com, vcom=vcom, feet=feet, vfeet=vfeet, acc=acc,
                p_pitch=p_pitch, p_roll=p_roll)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or not _S:
        _reset()
    dt = obs["dt"]
    e = _estimate(obs)
    q = np.asarray(obs["joint_angles"])
    qd = np.asarray(obs["joint_rates"])
    contact = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]

    # root velocity: accelerometer integration, corrected by leg odometry of the feet in contact
    # (integrated in the world frame through the yaw estimate, read out in the heading frame)
    cy, sy = np.cos(_S["yaw"]), np.sin(_S["yaw"])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    vw = _S["vroot"] + Rz @ e["acc"] * dt
    _S["ncon"] = [(_S["ncon"][k] + 1) if contact[k] else 0 for k in (0, 1)]
    cands = [Rz @ -e["vfeet"][k] for k in (0, 1) if _S["ncon"][k] >= P["NC"]]
    if cands:
        vk = min(cands, key=lambda c: np.linalg.norm(c - vw))
        vw = vw + P["VF"] * (vk - vw)
    _S["vroot"] = vw
    vroot = Rz.T @ vw
    v = vroot + e["vcom"]

    # --- initial settle: both legs straight until the feet are loaded ---
    if obs["time"] < P["T0"]:
        tgt = np.zeros(17)
        tgt[J["right_knee"]] = tgt[J["left_knee"]] = -0.1
        kp = np.where(_GEAR < 50, 8.0, 150.0)
        kd = np.where(_GEAR < 50, 0.2, 3.0)
        tau = kp * (tgt - q) - kd * qd
        _S["dbg"] = dict(d=np.zeros(3), v=np.zeros(3), st=-1)
        return np.clip(tau / _GEAR, -0.4, 0.4)

    # --- finite-state machine ---
    _S["t_sw"] += dt
    sw = 1 - _S["stance"]
    if _S["t_sw"] > P["T"] or (_S["t_sw"] > P["TC"] * P["T"] and contact[sw]):
        _S["stance"] = sw
        _S["t_sw"] = 0.0
        sw = 1 - _S["stance"]
    st = _S["stance"]
    phase = _S["t_sw"] / P["T"]

    d = e["com"] - e["feet"][st]

    tau = np.zeros(17)
    # abdomen & arms posture
    for n, kp, kd, tgt in (("abdomen_y", 150, 3, 0.0), ("abdomen_z", P["KP_Z"], P["KD_Z"], 0.0), ("abdomen_x", 150, 3, 0.0)):
        tau[J[n]] = kp * (tgt - q[J[n]]) - kd * qd[J[n]]
    for n in ("right_shoulder1", "right_shoulder2", "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"):
        tau[J[n]] = 8 * (0.0 - q[J[n]]) - 0.2 * qd[J[n]]

    side = ["right", "left"]
    sgn_x = [1.0, -1.0]   # right hip_x axis +x, left -x
    s, w_ = side[st], side[sw]

    # swing leg targets (world leg angles)
    lat_out = -1.0 if sw == 0 else 1.0       # outward direction (y) of the swing foot
    vd = P["VD"] * np.array([cy, -sy])        # desired world-x velocity seen in the heading frame
    xf = P["CP"] * v[0] + P["KV"] * (v[0] - vd[0])
    yf = P["CP"] * v[1] + P["KVL"] * (v[1] - vd[1]) + lat_out * P["W"]
    phi = np.arcsin(np.clip(xf / 0.9, -0.9, 0.9))
    psi = np.arcsin(np.clip(yf / 0.9, -0.9, 0.9))
    kn_t = P["KNEE_SW"] if phase < P["KPH"] else -0.05
    k_now = q[J[w_ + "_knee"]]
    beta0 = np.arctan2(0.445 * np.sin(-k_now), 0.39 + 0.445 * np.cos(k_now))   # foot lag behind the thigh from the knee bend
    hy_t = -phi - e["p_pitch"] - beta0
    hx_t = sgn_x[sw] * (psi - e["p_roll"])
    i_hy, i_hx, i_hz, i_kn = J[w_ + "_hip_y"], J[w_ + "_hip_x"], J[w_ + "_hip_z"], J[w_ + "_knee"]
    tau[i_hy] = P["KP_H"] * (hy_t - q[i_hy]) - P["KD_H"] * qd[i_hy]
    tau[i_hx] = P["KP_H"] * (hx_t - q[i_hx]) - P["KD_H"] * qd[i_hx]
    tau[i_hz] = P["KP_Z"] * (0.0 - q[i_hz]) - P["KD_Z"] * qd[i_hz]
    tau[i_kn] = P["KP_K"] * (kn_t - q[i_kn]) - P["KD_K"] * qd[i_kn]

    # stance leg: torso servo in the world (no yaw: a point foot transmits no torsional torque)
    wv = e["w"]
    t_pitch = P["KP_T"] * (P["PITCH_D"] - e["pitch"]) - P["KD_T"] * wv[1]
    t_roll = P["KP_R"] * (0.0 - e["roll"]) - P["KD_R"] * wv[0]
    j_hy, j_hx, j_hz, j_kn = J[s + "_hip_y"], J[s + "_hip_x"], J[s + "_hip_z"], J[s + "_knee"]
    tau[j_hy] = -t_pitch - P["SWC"] * tau[i_hy]
    tau[j_hx] = -sgn_x[st] * t_roll - P["SWC"] * tau[i_hx]
    tau[j_hz] = P["KP_Z"] * (0.0 - q[j_hz]) - P["KD_Z"] * qd[j_hz]
    tau[j_kn] = P["KP_KS"] * (P["KNEE_ST"] - q[j_kn]) - P["KD_KS"] * qd[j_kn]

    u = np.clip(tau / _GEAR, -0.4, 0.4)
    _S["dbg"] = dict(d=d, v=v, st=st, hy_t=hy_t, kn_t=kn_t, xf=xf, yf=yf)
    return u


PARAM_BOUNDS = dict(
    T=(0.2, 0.5), PITCH_D=(-0.1, 0.2), KV=(0.0, 0.5), VD=(0.0, 0.5), KVL=(0.0, 0.5), W=(0.0, 0.2), VF=(0.05, 0.5),
    KP_T=(100.0, 800.0), KD_T=(5.0, 60.0), KP_R=(50.0, 500.0), KD_R=(3.0, 40.0), CP=(0.1, 0.5),
    KNEE_SW=(-1.4, -0.3), KNEE_ST=(-0.5, -0.05), KP_H=(100.0, 900.0), KD_H=(1.0, 15.0), KP_K=(50.0, 400.0), KD_K=(0.5, 8.0),
    KP_KS=(100.0, 1200.0), KD_KS=(2.0, 30.0), T0=(0.1, 0.4), KPH=(0.2, 0.8), TC=(0.3, 0.9),
    SWC=(0.0, 1.0), KP_Z=(20.0, 300.0), KD_Z=(0.5, 8.0), NC=(1, 4),
)
