"""Walking controller for the MuJoCo Humanoid (SIMBICON-style, no learning).

Control law
-----------
Gait clock: the stance leg swaps when the swing foot touches down after ``t_min`` or at
``t_max`` at the latest (``t_first`` for the first step after the ``t_stand`` landing
phase).  With the tuned values ``t_max < t_min``, so the cadence is a fixed clock.

* Swing leg - foot placement + IK.  Target foot offset from the mass centre, in the world
  frame (the reward counts world +x), rotated into the pelvis heading ``psi``:
  ``x = v_x*T/2 + cv*(v_x - v_des) + x_off``, ``y = cvy*v_y + side*width`` (a
  Raibert / capture-point law).  ``kys``/``kyd`` lengthen the step of one side and shorten
  the other in proportion to heading error and yaw rate: the touchdown impulse at the
  laterally offset hip then turns the body back (yaw control for sphere feet, which
  cannot exert a ground yaw torque).  The foot follows a cosine blend from its lift-off
  position to the target with a ``clear`` apex at height ``-leg_z`` (``drop`` keeps it
  descending after phase 0.85); 2-link IK gives hip_y / knee targets, the lateral angle
  gives hip_x; joint PD with ``kp_hip``/``kd_hip``, ``kp_hxs``/``kd_hxs``,
  ``kp_knee``/``kd_knee``.
* Stance leg - torso control.  Knee PD (``kp_kst``) toward ``knee_stance`` (beyond the
  joint limit, i.e. a locked straight knee).  The stance hip applies the torso torque
  ``T = -kp*(angle - target) - kd*rate`` for pitch (``kp_pitch``, ``kd_pitch``,
  ``pitch_des``) and roll (``kp_roll``, ``kd_roll``), minus ``comp`` times the swing-hip
  reaction (SIMBICON virtual torque); hip_y is limited to ``max_st`` N m.  The stance
  hip_z adds a heading twist (``kp_yaw``, ``kd_yaw``); both hip_z keep ``kp_hz``.
* Upper body: abdomen PD (``kp_ab``, ``kd_ab``) to upright, its yaw joint tracking
  ``abz_k*psi``; shoulders counter-swing the legs (``arm_k``, ``arm_k2`` times the
  hip-pitch difference).
Torques are divided by the motor gears in ``obs["gear"]`` and clipped to [-0.4, 0.4].
Gains were tuned by CMA-ES over this structure on the nominal plant and the perturbed
scenarios.  (``knee_swing``, ``knee_land``, ``swing_frac`` are unused legacy entries.)
"""

import numpy as np

PARAMS = dict(
    v_des=0.1611, T=0.3041, t_min=0.3019, t_max=0.2085, x_off=-0.0867, leg_z=0.6711, clear=0.1115,
    cv=0.0634, cvy=0.0647, width=0.2055, pitch_des=-0.113, kp_pitch=132.1872, kd_pitch=21.2793,
    kp_roll=106.9877, kd_roll=73.3484, kp_yaw=25.0391, kd_yaw=7.2249, kp_hip=537.5679, kd_hip=5.8838,
    kp_hipx=200.0, kd_hipx=8.0, kp_knee=340.337, kd_knee=17.1659, kp_hz=12.982, kd_hz=1.0,
    knee_swing=-1.0, knee_land=-0.15, knee_stance=0.202, swing_frac=0.55, kp_ab=208.1637,
    kd_ab=10.3904, leg_len=0.85, t_stand=0.2, comp=1.1419, use_psi=1.0, ds_yaw=0.0, kys=0.2603,
    kyd=0.0333, kp_hxs=319.4691, kd_hxs=8.539, t_first=0.2549, kp_kst=583.6986, max_st=144.0231,
    arm_k=-0.6526, arm_k2=-0.4825, abz_k=-2.0306, drop=-0.1562,
)

_S = dict(stance=0, t=0.0, last_step=-1, x0=None)   # stance 0 = right, 1 = left
# motor indices: hip_x, hip_z, hip_y, knee for right / left
_LEG = ((3, 4, 5, 6), (7, 8, 9, 10))
_SX = (1.0, -1.0)
_L1, _L2 = 0.383, 0.37   # hip-knee, knee-foot-centre lengths   # hip_x / hip_z axis sign (right +x/+z, left -x/-z)


def reset():
    _S.update(stance=0, t=0.0, last_step=-1, x0=None)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or obs["step"] < _S["last_step"]:
        reset()
    _S["last_step"] = obs["step"]
    dt = obs["dt"]
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    gear = obs["gear"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    w = obs["root_angular_velocity"]
    wx, wy = cy * w[0] + sy * w[1], -sy * w[0] + cy * w[1]
    wz = w[2]
    v = obs["com_velocity_xy"]
    vx, vy = cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]

    if obs["time"] < P["t_stand"]:
        return _stand(obs, P, roll, pitch, wx, wy)
    # gait state machine
    _S["t"] += dt
    st = _S["stance"]
    sw = 1 - st
    contacts = (obs["foot_contact"], obs["foot_left_contact"])
    first = obs["time"] < P["t_stand"] + P["t_first"] + 0.01
    if (_S["t"] > (0.6 * P["t_first"] if first else P["t_min"]) and contacts[sw]) or _S["t"] > (P["t_first"] if first else P["t_max"]):
        _S["stance"], _S["t"], _S["x0"] = sw, 0.0, None
        st, sw = sw, st
    phase = _S["t"] / P["T"]

    tau = np.zeros(17)
    # abdomen and arms: upright posture
    tau[0:3] = P["kp_ab"] * (0.0 - q[0:3]) - P["kd_ab"] * qd[0:3]
    tau[11:17] = 20.0 * (0.0 - q[11:17]) - 1.0 * qd[11:17]

    # swing leg: foot trajectory in the heading frame (relative to the hip), then 2-link IK
    hx, hz, hy, kn = _LEG[sw]
    if _S["t"] == 0.0 or _S["x0"] is None:
        f0 = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])[sw]
        ps0 = yaw + q[1]
        _S["x0"] = f0[0] * np.cos(ps0) + f0[1] * np.sin(ps0)
    # desired foot offset from the mass centre: world frame (walk along world +x), then pelvis heading frame
    vxw, vyw = v[0], v[1]
    xw = vxw * P["T"] / 2 + P["cv"] * (vxw - P["v_des"]) + P["x_off"]
    yw = P["cvy"] * vyw
    psi = (yaw + q[1]) * P["use_psi"]
    cp, sp = np.cos(psi), np.sin(psi)
    side = -1.0 if sw == 0 else 1.0
    x_des = cp * xw + sp * yw - side * (P["kys"] * (yaw + q[1]) + P["kyd"] * wz)
    y_des = -sp * xw + cp * yw + side * P["width"]
    s_ph = min(phase, 1.0)
    blend = 0.5 - 0.5 * np.cos(np.pi * s_ph)
    fx = _S["x0"] + (x_des - _S["x0"]) * blend
    fz = -P["leg_z"] + P["clear"] * np.sin(np.pi * s_ph) - P["drop"] * max(phase - 0.85, 0.0)
    r = np.clip(np.hypot(fx, fz), 0.3, _L1 + _L2 - 1e-3)
    ck = np.clip((r * r - _L1 ** 2 - _L2 ** 2) / (2 * _L1 * _L2), -1.0, 1.0)
    k = np.arccos(ck)
    thigh = np.arctan2(fx, -fz) + np.arctan2(_L2 * np.sin(k), _L1 + _L2 * np.cos(k))
    pp = pitch + q[0]                    # pelvis pitch (torso pitch + abdomen_y)
    hy_d = -thigh - pp
    kn_d = -max(k, 0.03)
    beta = np.arcsin(np.clip((y_des - side * 0.1) / P["leg_len"], -0.5, 0.5))   # leg angle toward +y
    hx_d = _SX[sw] * (beta - roll - q[2])
    _S["dbg"] = (x_des, y_des, fx, fz)
    tau[hy] = P["kp_hip"] * (hy_d - q[hy]) - P["kd_hip"] * qd[hy]
    tau[hx] = P["kp_hxs"] * (hx_d - q[hx]) - P["kd_hxs"] * qd[hx]
    tau[hz] = P["kp_hz"] * (0.0 - q[hz]) - P["kd_hz"] * qd[hz]
    tau[kn] = P["kp_knee"] * (kn_d - q[kn]) - P["kd_knee"] * qd[kn]

    # stance leg: knee PD, hip realises the torso torque
    shx, shz, shy, skn = _LEG[st]
    tau[skn] = P["kp_kst"] * (P["knee_stance"] - q[skn]) - P["kd_knee"] * qd[skn]
    tau[shy] = 0.0
    T_y = -P["kp_pitch"] * (pitch - P["pitch_des"]) - P["kd_pitch"] * wy
    T_x = -P["kp_roll"] * roll - P["kd_roll"] * wx
    T_z = -P["kp_yaw"] * yaw - P["kd_yaw"] * wz
    # pelvis receives -s*tau from each hip joint with axis sign s
    tau[shy] = -(T_y + P["comp"] * tau[hy])
    tau[shx] = -(T_x + P["comp"] * _SX[sw] * tau[hx]) / _SX[st]
    # yaw: twist the stance hip against the (inclined) stance leg; differential hip_y in double support
    psi_p = yaw + q[1]
    T_z = -P["kp_yaw"] * psi_p - P["kd_yaw"] * wz
    tau[shz] = P["kp_hz"] * (0.0 - q[shz]) - P["kd_hz"] * qd[shz] - _SX[st] * T_z
    tau[shy] = np.clip(tau[shy], -P["max_st"], P["max_st"])
    # arms counter-swing the legs; abdomen yaw keeps the pelvis heading
    dh = q[5] - q[9]
    tau[11] = 20.0 * (P["arm_k"] * dh - q[11]) - 1.0 * qd[11]
    tau[14] = 20.0 * (P["arm_k2"] * dh - q[14]) - 1.0 * qd[14]
    tau[1] = P["kp_ab"] * (P["abz_k"] * (yaw + q[1]) - q[1]) - P["kd_ab"] * qd[1]
    if contacts[sw]:
        # right foot pushes back / left forward rotates the body to +z: pelvis torque -tau about y per hip
        tau[5] += -P["ds_yaw"] * T_z
        tau[9] += P["ds_yaw"] * T_z

    return np.clip(tau / gear, -0.4, 0.4)


def _stand(obs, P, roll, pitch, wx, wy):
    """Double support: both legs PD to a straight stance, hips correct the torso."""
    q, qd = obs["joint_angles"], obs["joint_rates"]
    tau = np.zeros(17)
    tau[0:3] = P["kp_ab"] * (0.0 - q[0:3]) - P["kd_ab"] * qd[0:3]
    tau[11:17] = 20.0 * (0.0 - q[11:17]) - 1.0 * qd[11:17]
    for k, (hx, hz, hy, kn) in enumerate(_LEG):
        tau[kn] = P["kp_knee"] * (P["knee_stance"] - q[kn]) - P["kd_knee"] * qd[kn]
        tau[hy] = P["kp_hip"] * (0.0 - q[hy]) - P["kd_hip"] * qd[hy]
        tau[hx] = P["kp_hipx"] * (0.0 - q[hx]) - P["kd_hipx"] * qd[hx]
        tau[hz] = P["kp_hz"] * (0.0 - q[hz]) - P["kd_hz"] * qd[hz]
    return np.clip(tau / obs["gear"], -0.4, 0.4)
