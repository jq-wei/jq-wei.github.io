"""Proprioceptive walking controller for the MuJoCo Humanoid (point feet, no ankles).

State estimation (from encoders + IMU only):
  * attitude = IMU roll/pitch; yaw = gyro integrated through the ZYX Euler-rate relation;
  * the MuJoCo kinematic model, posed with (roll, pitch, 0) and the joint angles, gives the
    mass-centre offset d from each foot in the heading frame;
  * mass-centre velocity v = (J_com - J_contact) qdot with qdot = (0, gyro, joint rates): exact if the
    foot in contact rolls without slipping (held constant in flight), low-pass filtered (VFILT).

Gait (finite-state machine, stance leg alternates):
  * T_SETTLE: initial double support after the 0.1 m drop at reset;
  * T_DS: double support after each touchdown -- torso torque split between both hips, a
    differential hip_y torque for yaw (YAW_DS) and a virtual horizontal force through J^T
    (KDS_X on v_x - V_DES, KDS_Y on v_y);
  * swing (T_STEP, touchdown accepted after T_MIN): Raibert foot placement relative to the mass centre,
        target = v T_STEP/2 + KV (v_x - V_DES), v_y T_STEP/2 + KV_LAT v_y + side*WIDTH,
    reached on a smooth lift-and-place path (CLEAR height, TLAND, DROP), solved by numerical IK on the
    model and tracked by joint PD (KP_SW/KD_SW hip_y, KP_SWX/KD_SWX hip_x, KP_K/KD_K knee;
    SWMAX/SWMAXX torque caps);
  * stance hip: torso attitude PD (KP_T/KD_T to PITCH_DES, KP_R/KD_R to ROLL_LEAN) minus the
    swing-hip reaction (SWC, SWCX; hip_x axes are mirrored), hip_z yaw PD (KP_YAW/KD_YAW);
    stance knee PD to KNEE_ST (KP_KST/KD_KST);
  * abdomen (ABX lean over the stance leg) and arms: joint PD.
Options kept for experimentation: SIMB (SIMBICON joint-space swing instead of IK), CTD (computed-torque
swing tracking), IK4 (hip_z in the IK), FF (IK velocity feedforward), NOFLIGHT.
Torques are divided by the gear and clipped to [-0.4, 0.4].
Active mode: SIMB=1. Swing hip world angles are phi = TH0 + CD*d_x + CV*(v_x - V_DES) and
psi = side*TH_LAT + CDL*d_y + CVL*v_y. The knee is held at KNEE_SW for the first SW_FRAC of the swing,
then at KNEE_END. Gains come from random search on seeds 0-3.
Status: brisk on ~80-85% of seeds (local check on 72 seeds), short of the 0.9 gate; the gait is sensitive
to its parameters (e.g. changing V_DES breaks it).
"""

import numpy as np
import mujoco
from hl_control.envs.humanoid import XML_PATH, ACTUATED_JOINTS

PARAMS = dict(
    T_STEP=0.2004,
    T_MIN=0.1482,
    V_DES=0.8111,
    KV=0.2014,
    KV_LAT=0.2288,
    WIDTH=0.2232,
    CLEAR=0.1672,
    TLAND=0.6502,
    PITCH_DES=0.0986,
    KP_T=213.7716,
    KD_T=17.6253,
    KP_R=299.7151,
    KD_R=15.3511,
    KP_SW=317.7876,
    KD_SW=14.5539,
    KP_SWX=320.1724,
    KD_SWX=17.7981,
    KNEE_ST=-0.2201,
    KP_K=549.5717,
    KD_K=5.7725,
    KP_KST=494.2502,
    KD_KST=7.764,
    T_SETTLE=0.25,
    KP_YAW=53.6029,
    KD_YAW=6.3842,
    VFILT=0.3,
    FF=0.0,
    SWC=0.8163,
    YAW_SW=0.0596,
    SWMAX=442.8185,
    NOFLIGHT=0,
    T_DS=0.1029,
    ROLL_LEAN=-0.2088,
    DROP=-0.1031,
    IK4=0,
    CTD=0,
    WN=33.5698,
    ZETA=0.9,
    KDS_X=0.2367,
    KDS_Y=2.9197,
    SIMB=1,
    TH0=0.5533,
    CD=0.2441,
    CV=0.1708,
    TH_LAT=0.0583,
    CDL=0.3143,
    CVL=0.2752,
    KNEE_SW=-0.8738,
    KNEE_END=-0.087,
    SW_FRAC=0.8013,
    SWCX=1.3048,
    SWMAXX=91.9911,
    ABX=0.0341,
    YAW_DS=3.4907,
)

_model = mujoco.MjModel.from_xml_path(str(XML_PATH))
_data = mujoco.MjData(_model)
_QADR = np.array([_model.jnt_qposadr[_model.joint(n).id] for n in ACTUATED_JOINTS])
_FOOT = [_model.geom("right_foot").id, _model.geom("left_foot").id]
_MASS = _model.body_mass / _model.body_mass.sum()
_GEAR = _model.actuator_gear[:, 0].copy()
J = {n: i for i, n in enumerate(ACTUATED_JOINTS)}
DBG = ""
_S = {}


def reset():
    _S.clear()


def _quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def _kin(q, roll, pitch, yaw):
    _data.qpos[:] = 0.0
    _data.qpos[3:7] = _quat(roll, pitch, yaw)
    _data.qpos[_QADR] = q
    mujoco.mj_kinematics(_model, _data)
    com = _MASS @ _data.xipos
    return com, [_data.geom_xpos[_FOOT[0]].copy(), _data.geom_xpos[_FOOT[1]].copy()]


_JP = np.zeros((3, _model.nv))
_JC = np.zeros((3, _model.nv))
_DADR = np.array([_model.jnt_dofadr[_model.joint(n).id] for n in ACTUATED_JOINTS])
_SHIN = [_model.body("right_foot").id, _model.body("left_foot").id]


def _vel_rel(gyro, qd, k):
    """Mass-centre velocity (world) assuming foot k rolls without slipping (call after _kin)."""
    mujoco.mj_comPos(_model, _data)
    qv = np.zeros(_model.nv)
    qv[3:6] = gyro
    qv[_DADR] = qd
    mujoco.mj_jacSubtreeCom(_model, _data, _JC, 1)
    pt = _data.geom_xpos[_FOOT[k]] - np.array([0.0, 0.0, 0.075])
    mujoco.mj_jac(_model, _data, _JP, None, pt, _SHIN[k])
    return (_JC - _JP) @ qv


_dik = mujoco.MjData(_model)
_JF = np.zeros((3, _model.nv))
_LEG = [[J["right_hip_x"], J["right_hip_y"], J["right_knee"]], [J["left_hip_x"], J["left_hip_y"], J["left_knee"]]]
_LEG4 = [[J["right_hip_x"], J["right_hip_y"], J["right_knee"], J["right_hip_z"]], [J["left_hip_x"], J["left_hip_y"], J["left_knee"], J["left_hip_z"]]]
_RNG = np.array([_model.jnt_range[_model.joint(n).id] for n in ACTUATED_JOINTS])


def _ik(q, roll, pitch, k, target, iters=4):
    """Joint angles (motor order) placing foot k's centre at target (heading frame, root at origin)."""
    qq = q.copy()
    idx = _LEG4[k] if PARAMS["IK4"] else _LEG[k]
    n = len(idx)
    for _ in range(iters):
        _dik.qpos[:] = 0.0
        _dik.qpos[3:7] = _quat(roll, pitch, 0.0)
        _dik.qpos[_QADR] = qq
        mujoco.mj_kinematics(_model, _dik)
        mujoco.mj_comPos(_model, _dik)
        p = _dik.geom_xpos[_FOOT[k]]
        err = target - p
        mujoco.mj_jac(_model, _dik, _JF, None, p, _SHIN[k])
        A = _JF[:, _DADR[idx]]
        dq = np.linalg.solve(A.T @ A + 1e-3 * np.eye(n), A.T @ err)
        qq[idx] = np.clip(qq[idx] + dq, _RNG[idx, 0], _RNG[idx, 1])
    return qq


def _jt_force(k, F):
    """Joint torques (motor order) making foot k push the robot with world force F (call after _kin)."""
    mujoco.mj_comPos(_model, _data)
    pt = _data.geom_xpos[_FOOT[k]] - np.array([0.0, 0.0, 0.075])
    mujoco.mj_jac(_model, _data, _JP, None, pt, _SHIN[k])
    out = np.zeros(17)
    idx = _LEG[k]
    out[idx] = -_JP[:, _DADR[idx]].T @ F
    return out


_MFULL = np.zeros((_model.nv, _model.nv))
_BIAS = np.zeros(_model.nv)


def _swing_ctd(q, qd, gyro, roll, pitch, idx, qdes, wn, zeta):
    """Computed-torque tracking for the swing-leg joints idx (motor order) toward qdes."""
    _data.qpos[:] = 0.0
    _data.qpos[3:7] = _quat(roll, pitch, 0.0)
    _data.qpos[_QADR] = q
    _data.qvel[:] = 0.0
    _data.qvel[3:6] = gyro
    _data.qvel[_DADR] = qd
    mujoco.mj_kinematics(_model, _data)
    mujoco.mj_comPos(_model, _data)
    mujoco.mj_crb(_model, _data)
    mujoco.mj_fullM(_model, _data, _MFULL)
    mujoco.mj_comVel(_model, _data)
    mujoco.mj_rne(_model, _data, 0, _BIAS)
    dof = _DADR[idx]
    qdd = wn * wn * (qdes[idx] - q[idx]) - 2.0 * zeta * wn * qd[idx]
    return _MFULL[np.ix_(dof, dof)] @ qdd + _BIAS[dof]


def controller(obs: dict) -> np.ndarray:
    global DBG
    P = PARAMS
    dt = obs["dt"]
    q = np.asarray(obs["joint_angles"], float)
    qd = np.asarray(obs["joint_rates"], float)
    roll, pitch = obs["roll"], obs["pitch"]
    w = np.asarray(obs["gyro_body"], float)
    contact = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]
    if obs["step"] == 0 or not _S:
        _S.clear()
        _S.update(yaw=0.0, stance=0, tph=0.0, prev=None, v=np.zeros(2), init=True)
    # yaw from gyro (ZYX Euler rates)
    yaw_rate = (np.sin(roll) * w[1] + np.cos(roll) * w[2]) / max(np.cos(pitch), 0.3)
    _S["yaw"] += yaw_rate * dt
    yaw = _S["yaw"]
    com, feet = _kin(q, roll, pitch, 0.0)    # heading frame
    relw = [com[:2] - feet[0][:2], com[:2] - feet[1][:2]]   # world frame

    # FSM
    if obs["time"] >= P["T_SETTLE"]:
        _S["tph"] += dt
    st = _S["stance"]
    sw = 1 - st
    if _S["tph"] > P["T_MIN"] + P["T_DS"] and contact[sw] or _S["tph"] > P["T_STEP"] * 1.3 + P["T_DS"]:
        _S["stance"] = st = sw
        sw = 1 - st
        _S["tph"] = 0.0
    if _S["tph"] == 0.0:
        _S["p0"] = feet[sw] - feet[st]      # swing foot start, relative to the stance foot
    tph = _S["tph"]

    # mass-centre velocity: the pivot foot (in contact, else the lower one) rolls without slipping
    if contact[0] and contact[1]:
        vraw = 0.5 * (_vel_rel(w, qd, 0) + _vel_rel(w, qd, 1))[:2]
    elif contact[0] or contact[1]:
        vraw = _vel_rel(w, qd, 0 if contact[0] else 1)[:2]
    else:
        vraw = _vel_rel(w, qd, _S["stance"])[:2] if P["NOFLIGHT"] else _S["v"]
    a = P["VFILT"]
    _S["v"] = (1 - a) * _S["v"] + a * vraw
    rel = relw
    v = _S["v"]
    d = rel[st]

    tau = np.zeros(17)
    sgn = [1.0, -1.0]  # right / left hip_x and hip_z axis signs
    hx = [J["right_hip_x"], J["left_hip_x"]]
    hz = [J["right_hip_z"], J["left_hip_z"]]
    hy = [J["right_hip_y"], J["left_hip_y"]]
    kn = [J["right_knee"], J["left_knee"]]
    pel_pitch = pitch + q[J["abdomen_y"]]
    pel_roll = roll + q[J["abdomen_x"]]
    tq_p = P["KP_T"] * (P["PITCH_DES"] - pitch) - P["KD_T"] * w[1]
    rs = 1.0 if st == 0 else -1.0           # +1: right stance (lean the upper body over it = positive roll)
    single = obs["time"] >= P["T_SETTLE"] and tph >= P["T_DS"]
    roll_des = rs * P["ROLL_LEAN"] if single else 0.0
    tq_r = P["KP_R"] * (roll_des - roll) - P["KD_R"] * w[0]
    tq_y = P["KP_YAW"] * (0.0 - yaw) - P["KD_YAW"] * w[2]
    settle = obs["time"] < P["T_SETTLE"]
    ds = (not settle) and tph < P["T_DS"]
    if ds:
        _S["p0"] = feet[sw] - feet[st]
    tsw = max(tph - P["T_DS"], 0.0)
    phi = psi = 0.0
    if ds and P["KDS_X"] + P["KDS_Y"] > 0:
        _kin(q, roll, pitch, 0.0)
        F = 42.0 * np.array([-P["KDS_X"] * (v[0] - P["V_DES"]), -P["KDS_Y"] * v[1], 0.0])
        tau += _jt_force(0, 0.5 * F) + _jt_force(1, 0.5 * F)
    if settle or ds:
        for k in (0, 1):
            tau[hy[k]] += -0.5 * tq_p + (0.0 if settle else P["YAW_DS"] * tq_y * (1.0 if k == 0 else -1.0))
            tau[hx[k]] += -sgn[k] * 0.5 * tq_r
            tau[hz[k]] = 100.0 * (0.0 - q[hz[k]]) - 5.0 * qd[hz[k]]
            tau[kn[k]] += P["KP_KST"] * (P["KNEE_ST"] - q[kn[k]]) - P["KD_KST"] * qd[kn[k]]
    else:
        # swing foot: Raibert placement relative to the mass centre, smooth lift-and-place, IK
        side = -1.0 if sw == 0 else 1.0
        tgt = np.array([d[0] + v[0] * P["T_STEP"] / 2 + P["KV"] * (v[0] - P["V_DES"]),
                        d[1] + v[1] * P["T_STEP"] / 2 + P["KV_LAT"] * v[1] + side * P["WIDTH"], 0.0])
        s_ = min(tsw / P["T_STEP"], 1.0)
        b_ = min(s_ / P["TLAND"], 1.0)
        b_ = b_ * b_ * (3 - 2 * b_)
        pdes = _S["p0"] + (tgt - _S["p0"]) * b_
        pdes[2] = P["CLEAR"] * np.sin(np.pi * s_) - P["DROP"] * s_
        target = feet[st] + pdes
        if P["SIMB"]:
            phi_w = P["TH0"] + P["CD"] * d[0] + P["CV"] * (v[0] - P["V_DES"])
            psi_w = side * P["TH_LAT"] + P["CDL"] * d[1] + P["CVL"] * v[1]
            qik = q.copy()
            qik[hy[sw]] = -phi_w - pel_pitch
            qik[hx[sw]] = sgn[sw] * (psi_w - pel_roll)
            qik[kn[sw]] = P["KNEE_SW"] if s_ < P["SW_FRAC"] else P["KNEE_END"]
            qik[[hy[sw], hx[sw], kn[sw]]] = np.clip(qik[[hy[sw], hx[sw], kn[sw]]], _RNG[[hy[sw], hx[sw], kn[sw]], 0], _RNG[[hy[sw], hx[sw], kn[sw]], 1])
            if P["IK4"]:
                qik[hz[sw]] = 0.0
        else:
            qik = _ik(q, roll, pitch, sw, target)
        prev = _S.get("qik")
        qdik = (qik - prev) / dt if (prev is not None and tsw > 0.0) else np.zeros(17)
        _S["qik"] = qik
        phi, psi = pdes[0], pdes[1]
        if P["CTD"]:
            idx = [hx[sw], hz[sw], hy[sw], kn[sw]]
            qdes = qik.copy()
            if not P["IK4"]:
                qdes[hz[sw]] = 0.0
            tau[idx] = _swing_ctd(q, qd, w, roll, pitch, idx, qdes, P["WN"], P["ZETA"])
        else:
            for i, kp, kd in ((hy[sw], P["KP_SW"], P["KD_SW"]), (hx[sw], P["KP_SWX"], P["KD_SWX"]), (kn[sw], P["KP_K"], P["KD_K"])):
                tau[i] = kp * (qik[i] - q[i]) + kd * (P["FF"] * np.clip(qdik[i], -15, 15) - qd[i])
        tau[hy[sw]] = np.clip(tau[hy[sw]], -P["SWMAX"], P["SWMAX"])
        tau[hx[sw]] = np.clip(tau[hx[sw]], -P["SWMAXX"], P["SWMAXX"])
        i = hz[sw]
        if not P["CTD"]:
            tau[i] = (100.0 * (qik[i] - q[i]) - 5.0 * qd[i]) if P["IK4"] else (100.0 * (-sgn[sw] * P["YAW_SW"] * yaw - q[i]) - 5.0 * qd[i])
        # stance hip: torso attitude, minus the swing-hip reaction
        tau[hy[st]] = -tq_p - P["SWC"] * tau[hy[sw]]
        tau[hx[st]] = -sgn[st] * tq_r + P["SWCX"] * tau[hx[sw]]   # hip_x axes are mirrored
        tau[hz[st]] = -sgn[st] * tq_y
        i = kn[st]
        tau[i] = P["KP_KST"] * (P["KNEE_ST"] - q[i]) - P["KD_KST"] * qd[i]

    # abdomen and arms: hold zero
    abx_des = -rs * P["ABX"] if single else 0.0
    for n, qdes in (("abdomen_y", 0.0), ("abdomen_z", 0.0), ("abdomen_x", abx_des)):
        i = J[n]
        tau[i] = 200.0 * (qdes - q[i]) - 10.0 * qd[i]
    for n in ("right_shoulder1", "right_shoulder2", "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"):
        i = J[n]
        tau[i] = 20.0 * (0.0 - q[i]) - 1.0 * qd[i]

    return np.clip(tau / _GEAR, -0.4, 0.4)
