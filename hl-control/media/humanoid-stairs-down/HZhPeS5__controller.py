"""Walking controller for the MuJoCo Humanoid down a 3 cm staircase (point feet, no ankles).

SIMBICON-style virtual torso control with a Cartesian swing foot solved by analytic leg inverse
kinematics. No learning: the gains were picked by a derivative-free search on visible seeds.

Control law (torques in N m, divided by the gear to give motor commands, clipped to +-0.4):

* Kinematics: a private copy of the plant's model is posed from ``gym_obs`` (qpos[2:]) to get the
  hip positions relative to the mass centre and the pelvis orientation (forward kinematics only).
* Start: both legs hold a standing posture until the robot has dropped onto both feet (T_STAND).
* Gait machine: a stance leg and a swing leg. After each swap both feet stay planted for ``t_ds``
  s (double support: with spherical feet and no torsional friction it is the only phase where
  yaw can be corrected), then the swing foot moves. Roles swap when the swing foot touches down
  after ``t_min`` s of swing, or at ``t_max``.
* Swing foot (heading frame, relative to the mass centre): from lift-off to the placement target
  in ``t_swing`` s (ease-out profile), lifted by ``clear`` on a sine arc and ending ``reach``
  below the stance foot, so it keeps reaching for a tread that may be a step lower (the stairs
  are never observed: the touchdown is what finds the next tread). Placement:
      x = a_v * v_x + k_v * (v_x - v_des)
      y = out * width + a_vy * v_y               (out = +1 for the left foot, -1 for the right)
  2-link IK in the pelvis frame turns the target into hip_x, hip_y and knee angles, tracked by PD:
  kp_sw / kd_sw on hip_y (torque capped at sw_lim so the stance hip keeps authority over the
  torso), kp_hx / kd_hx on hip_x, kp_knee / kd_knee on the knee.
* Stance leg: knee PD (kp_kst / kd_kst) to ``knee_stance``; the stance hip produces the torso
  torque tau_t = k_torso * (lean - pitch) - d_torso * pitch_rate (k_roll / d_roll on roll)
  minus the swing hip's reaction: tau_stance_hip = -tau_t - tau_swing_hip.
* Yaw: in double support both hip_z joints apply k_yaw * (0 - yaw) - d_yaw * yaw_rate; in single
  support the stance hip_y adds +-k_ys * (yaw error), a fore-aft push at the foot's lateral
  offset from the mass centre, which turns the body (M_z = -r_y * dF_x).
* Distance rule: once the mass centre has advanced ``x_stop`` m from where it started (its absolute
  x is observed), the speed target drops to ``v_stop`` and the robot steps nearly in place: every
  further step down the stairs adds energy, and the forward rung only needs 3 m.
* Upper body: abdomen PD to zero (kp_ab / kd_ab), arms PD to zero (kp_arm).
* Damping gains are small where the limb is light: the 15 ms zero-order hold destabilises a
  large kd on the knee (0.36 kg m^2) and hip_z (0.03 kg m^2); the stance knee, which carries the
  body, tolerates and needs much more.
"""

import numpy as np

from hl_control.envs import humanoid as _plant   # the plant's own model (forward kinematics only)

mujoco = _plant.mujoco
XML_PATH = _plant.XML_PATH

PARAMS = {
    "v_des": 0.3372,
    "a_v": 0.0768,
    "k_v": 0.0859,
    "width": 0.16,
    "a_vy": 0.3118,
    "t_swing": 0.16,
    "t_min": 0.15,
    "t_max": 0.5,
    "clear": 0.1126,
    "reach": 0.0492,
    "lean": 0.0104,
    "k_torso": 269.1823,
    "d_torso": 37.1403,
    "k_roll": 118.6787,
    "d_roll": 10.4515,
    "knee_stance": -0.2177,
    "kp_sw": 517.6397,
    "kd_sw": 10.2635,
    "kp_hx": 207.487,
    "kd_hx": 5.0,
    "x_stop": 3.4,
    "v_stop": 0.0,
    "kp_knee": 177.0716,
    "kd_knee": 2.0,
    "kp_ab": 200.0,
    "kd_ab": 5.0,
    "k_yaw": 145.1038,
    "sw_lim": 85.5406,
    "kp_kst": 200.6051,
    "kd_kst": 18.986,
    "t_ds": 0.0315,
    "k_ys": 114.7038,
    "kp_arm": 20.0,
    "d_yaw": 19.0819,
    "ease": 1.0,
}

# motor indices
AB_Y, AB_Z, AB_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_K = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_K = 7, 8, 9, 10
ARMS = (11, 12, 13, 14, 15, 16)
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], dtype=float)
LEG = {0: (R_HX, R_HZ, R_HY, R_K), 1: (L_HX, L_HZ, L_HY, L_K)}   # 0 = right, 1 = left
T_STAND = 0.25          # s, standing phase at the start
L1, L2 = 0.383, 0.370   # hip -> knee axis, knee axis -> foot-sphere centre [m]

_M = mujoco.MjModel.from_xml_path(str(XML_PATH))
_D = mujoco.MjData(_M)
_MASS = _M.body_mass.copy()
_THIGH = (_M.body("right_thigh").id, _M.body("left_thigh").id)
_PELVIS = _M.body("pelvis").id

_state = {"stance": 0, "t_phase": 0.0, "last_step": -1, "walking": False, "p0": np.zeros(3), "x0": 0.0}


def _reset():
    _state.update(stance=0, t_phase=0.0, last_step=-1, walking=False, p0=np.zeros(3), x0=0.0)


def _kinematics(obs):
    _D.qpos[:2] = 0.0
    _D.qpos[2:] = np.asarray(obs["gym_obs"][: _M.nq - 2], dtype=float)
    mujoco.mj_kinematics(_M, _D)
    com = _MASS @ _D.xipos / _MASS.sum()
    hips = [_D.xpos[_THIGH[0]] - com, _D.xpos[_THIGH[1]] - com]
    return hips, _D.xmat[_PELVIS].reshape(3, 3).copy()


def _leg_ik(r_world, rot_pelvis, side):
    """Joint targets (hip_x, hip_y, knee) placing the foot-sphere centre at r_world from the hip."""
    rx, ry, rz = rot_pelvis.T @ r_world
    a = np.arctan2(ry, -rz) if side == 0 else np.arctan2(-ry, -rz)
    zp = -np.hypot(ry, rz)
    dist = float(np.clip(np.hypot(rx, zp), 0.35, L1 + L2 - 0.003))
    kappa = np.arccos(np.clip((dist * dist - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1.0, 1.0))
    gamma = np.arctan2(-rx, -zp)
    beta = np.arctan2(L2 * np.sin(kappa), L1 + L2 * np.cos(kappa))
    return float(np.clip(a, -0.43, 0.08)), float(np.clip(gamma - beta, -1.9, 0.35)), float(-kappa)


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    step = int(obs["step"])
    if step == 0 or step < _state["last_step"]:
        _reset()
        _state["x0"] = float(obs["com_xyz"][0])
    _state["last_step"] = step
    dt = float(obs["dt"])
    t = float(obs["time"])

    q = np.asarray(obs["joint_angles"], dtype=float)
    qd = np.asarray(obs["joint_rates"], dtype=float)
    roll, pitch, yaw = float(obs["roll"]), float(obs["pitch"]), float(obs["yaw"])
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot_h = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])   # world -> heading frame
    w = rot_h @ np.asarray(obs["root_angular_velocity"], dtype=float)
    v = rot_h[:2, :2] @ np.asarray(obs["com_velocity_xy"], dtype=float)
    feet = [rot_h @ np.asarray(obs["foot_xyz_rel"], dtype=float), rot_h @ np.asarray(obs["foot_left_xyz_rel"], dtype=float)]
    contact = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]
    hips, rot_pelvis = _kinematics(obs)

    tau = np.zeros(17)
    # ----- upper body: abdomen stiff, arms held -----
    tau[AB_Y] = p["kp_ab"] * (0.0 - q[AB_Y]) - p["kd_ab"] * qd[AB_Y]
    tau[AB_Z] = 60.0 * (0.0 - q[AB_Z]) - 4.0 * qd[AB_Z]
    tau[AB_X] = p["kp_ab"] * (0.0 - q[AB_X]) - p["kd_ab"] * qd[AB_X]
    for j in ARMS:
        tau[j] = p["kp_arm"] * (0.0 - q[j]) - 1.0 * qd[j]
    pelvis_pitch = pitch + q[AB_Y]
    pelvis_roll = roll + q[AB_X]

    t_pitch = p["k_torso"] * (p["lean"] - pitch) - p["d_torso"] * w[1]
    t_roll = p["k_roll"] * (0.0 - roll) - p["d_roll"] * w[0]
    t_yaw = p["k_yaw"] * (0.0 - yaw) - p["d_yaw"] * w[2]

    # ----- standing phase: wait until both feet are down -----
    if not _state["walking"]:
        for leg in (0, 1):
            hx, hz, hy, kn = LEG[leg]
            tau[hy] = p["kp_sw"] * (-pelvis_pitch - q[hy]) - p["kd_sw"] * qd[hy]
            tau[hx] = p["kp_hx"] * (-pelvis_roll * (1 if leg == 0 else -1) - q[hx]) - p["kd_hx"] * qd[hx]
            tau[hz] = 20.0 * (0.0 - q[hz]) - 1.0 * qd[hz]
            tau[kn] = p["kp_knee"] * (p["knee_stance"] - q[kn]) - p["kd_knee"] * qd[kn]
        if contact[0] and contact[1]:
            tau[R_HY] += -0.5 * t_pitch
            tau[L_HY] += -0.5 * t_pitch
        if t >= T_STAND and contact[0] and contact[1]:
            _state["walking"] = True
            _state["stance"] = 0 if feet[0][0] >= feet[1][0] else 1   # swing the rear foot first
            _state["t_phase"] = 0.0
            _state["p0"] = feet[1 - _state["stance"]].copy()
        return np.clip(tau / GEAR, -0.4, 0.4)

    # ----- gait state machine -----
    _state["t_phase"] += dt
    st = _state["stance"]
    sw = 1 - st
    if (_state["t_phase"] > p["t_min"] + p["t_ds"] and contact[sw]) or _state["t_phase"] > p["t_max"] + p["t_ds"]:
        _state["stance"] = st = sw
        sw = 1 - st
        _state["t_phase"] = 0.0
        _state["p0"] = feet[sw].copy()
    tp = _state["t_phase"]

    # ----- swing foot target (heading frame, relative to the mass centre) -----
    out = 1.0 if sw == 1 else -1.0
    # once the mass centre has advanced x_stop, step in place (v_stop): every further step down adds energy
    v_des = p["v_stop"] if float(obs["com_xyz"][0]) - _state["x0"] >= p["x_stop"] else p["v_des"]
    x_t = p["a_v"] * v[0] + p["k_v"] * (v[0] - v_des)
    y_t = out * p["width"] + p["a_vy"] * v[1]
    x_t = float(np.clip(x_t, -0.25, 0.45))
    y_t = float(np.clip(y_t, -0.35, -0.02) if sw == 0 else np.clip(y_t, 0.02, 0.35))
    ts = max(tp - p["t_ds"], 0.0)                      # swing time after the double-support hold
    s = min(ts / p["t_swing"], 1.0)
    sh = (1.0 - p["ease"]) * s * s * (3.0 - 2.0 * s) + p["ease"] * (1.0 - (1.0 - s) ** 2)   # smoothstep / ease-out blend
    p0 = _state["p0"]
    g_z = feet[st][2]                                   # stance foot-centre height = local ground level
    z_t = g_z + p["clear"] * np.sin(np.pi * s) - p["reach"] * s * s - 0.3 * max(ts - p["t_swing"], 0.0)
    if tp < p["t_ds"]:
        target_h = feet[sw].copy()
        _state["p0"] = feet[sw].copy()
    else:
        target_h = np.array([p0[0] + (x_t - p0[0]) * sh, p0[1] + (y_t - p0[1]) * sh, z_t])
    target_w = rot_h.T @ target_h
    a_hx, a_hy, a_kn = _leg_ik(target_w - hips[sw], rot_pelvis, sw)

    hx, hz, hy, kn = LEG[sw]
    tau[hy] = float(np.clip(p["kp_sw"] * (a_hy - q[hy]) - p["kd_sw"] * qd[hy], -p["sw_lim"], p["sw_lim"]))
    tau[hx] = p["kp_hx"] * (a_hx - q[hx]) - p["kd_hx"] * qd[hx]
    tau[hz] = 20.0 * (0.0 - q[hz]) - 1.0 * qd[hz]
    tau[kn] = p["kp_knee"] * (a_kn - q[kn]) - p["kd_knee"] * qd[kn]

    # ----- stance leg: knee PD, hip produces the torso torque -----
    shx, shz, shy, skn = LEG[st]
    tau[skn] = p["kp_kst"] * (p["knee_stance"] - q[skn]) - p["kd_kst"] * qd[skn]
    tau[shz] = 20.0 * (0.0 - q[shz]) - 1.0 * qd[shz]
    # torque on the pelvis about +y: -(tau_RHY + tau_LHY); about +x: -tau_RHX + tau_LHX
    # yaw lever in single support: a fore-aft push dF_x at a foot offset r_y from the mass centre gives M_z = -r_y dF_x
    e_yaw = (0.0 - yaw) - 0.3 * w[2]
    tau[shy] = -t_pitch - tau[hy] + (p["k_ys"] * e_yaw if st == 0 else -p["k_ys"] * e_yaw)
    tau[shx] = (tau[hx] - t_roll) if st == 0 else (t_roll + tau[hx])
    if contact[0] and contact[1]:
        # yaw moment on the pelvis about +z: -tau_RHZ + tau_LHZ
        tau[R_HZ] += -0.5 * t_yaw
        tau[L_HZ] += 0.5 * t_yaw

    return np.clip(tau / GEAR, -0.4, 0.4)
