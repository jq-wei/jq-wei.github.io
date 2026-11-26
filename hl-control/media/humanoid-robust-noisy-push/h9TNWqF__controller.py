"""Capture-point walking controller for the MuJoCo Humanoid (point feet, torque motors).

Hand-designed model-based law; the numeric gains below were set by a black-box search over the
visible seeds of all scenarios (no learning of any network).  Torques are computed in N m, divided
by the actual gear and clipped to ``lim`` x 0.4 of the nominal command range.

* Gait (state machine): the stance leg alternates every step of nominal duration ``T``; a step ends
  when the swing foot touches down after ``td_min`` T, or unconditionally after ``tmax`` T.  The first
  ``t_stand`` s hold both legs straight while the body drops onto the floor.
* Torso (SIMBICON): the stance hip holds the torso's world pitch at ``pitch_d`` (``kp_t``/``kd_t``)
  and the pelvis roll at 0 (``kp_r``/``kd_r``), absorbing the swing hip's reaction torque; the
  abdomen holds the torso roll upright (``kp_ra``/``kd_ra``); the stance hip yaw steers the heading
  back to +x (``kp_y``/``kd_y``); while both feet are planted, the two hip yaws add a friction
  couple that steers the heading (``kp_yd``/``kd_yd``), since point feet carry no yaw torque alone.
  The stance knee is driven onto its straight limit (``kp_ks``/``kd_ks``).
* Foot placement (balance, capture point): the swing foot lands, relative to the mass centre, at
  X = ``kxv`` v_x + ``kxe`` (v_x - ``v_des``) + ``x0`` and Y = side ``w`` + ``kyv`` v_y, never closer
  than ``sep`` laterally to the stance foot.  It follows a smoothstep path with apex clearance ``clr``
  and keeps reaching down (``ext`` m/s) after T until contact.  Leg inverse kinematics turn the
  foot target into hip/knee targets tracked by PD (``kp_h``/``kd_h`` sagittal, ``kp_hx``/``kd_hx``
  lateral, ``kp_k``/``kd_k`` knee); the swing hip yaw turns the leg by ``align`` x the pelvis heading error.
* Command delay (Smith predictor): if the applied command (``last_action``) lags our output by one
  step, the state is predicted one step ahead on a nominal MuJoCo copy of the plant, replaying the
  pending command, and the law above acts on the prediction (``smith``).
"""

import numpy as np

PARAMS = {
    "lim": 0.9831,
    "smith": 1.0,
    "align": -0.1497,
    "kp_yd": 49.3437,
    "kd_yd": 14.4111,
    "sep": 0.1072,
    "ext": 0.4696,
    "tmax": 1.3213,
    "T": 0.3126,
    "pitch_d": -0.0479,
    "kp_t": 429.7266,
    "kd_t": 41.5864,
    "kp_r": 119.9039,
    "kd_r": 22.3394,
    "kp_y": 13.5325,
    "kp_ra": 244.8377,
    "kd_ra": 2.7797,
    "kd_y": 7.8898,
    "kxv": 0.165,
    "kxe": 0.0485,
    "x0": 0.0373,
    "v_des": 1.1769,
    "w": 0.1648,
    "kyv": 0.2372,
    "clr": 0.0761,
    "hc": 0.15,
    "kp_h": 373.9456,
    "kd_h": 13.7422,
    "kp_hx": 380.7245,
    "kd_hx": 4.0913,
    "kp_k": 50.2031,
    "kd_k": 3.2652,
    "kp_ks": 233.4305,
    "kd_ks": 19.7135,
    "t_stand": 0.3,
    "td_min": 0.2185,
}


A_TH, B_SH = 0.383, 0.37       # hip->knee, knee->foot-centre lengths
HIP = {0: (3, 4, 5, 6), 1: (7, 8, 9, 10)}   # leg 0 = right, 1 = left: (hip_x, hip_z, hip_y, knee)
AY, AZ, AX = 0, 1, 2
GEAR0 = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
_S = {"stance": 0, "t0": 0.0, "lift": np.zeros(2)}


def _pd(kp, kd, err, rate):
    return kp * err - kd * rate


def _control(obs: dict, P: dict) -> np.ndarray:
    t = obs["time"]
    q, dq = obs["joint_angles"], obs["joint_rates"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]
    feet = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])
    contact = (obs["foot_contact"], obs["foot_left_contact"])
    if obs["step"] == 0:
        _S["stance"], _S["t0"] = 0, P["t_stand"]
        _S["lift"] = feet[1][:2].copy()

    st = _S["stance"]
    sw = 1 - st
    ph = t - _S["t0"]
    if t >= P["t_stand"] and ((ph > P["td_min"] * P["T"] and contact[sw]) or ph > P["tmax"] * P["T"]):
        st, sw = sw, st
        _S["stance"], _S["t0"] = st, t
        _S["lift"] = feet[sw][:2].copy()
        ph = 0.0
    s = np.clip(ph / P["T"], 0.0, 1.0)

    vx, vy = obs["com_velocity_xy"]
    tau = np.zeros(17)
    tau[AY] = _pd(200, 10, -q[AY], dq[AY])
    tau[AZ] = _pd(200, 10, -q[AZ], dq[AZ])
    tau[AX] = P["kp_ra"] * roll + P["kd_ra"] * w[0]        # torso roll held upright by the abdomen
    p_pitch, p_roll, p_yaw = pitch + q[AY], roll + q[AX], yaw + q[AZ]

    # ---- swing foot target relative to the mass centre (world frame)
    side = -1.0 if sw == 0 else 1.0
    X = P["kxv"] * vx + P["kxe"] * (vx - P["v_des"]) + P["x0"]
    Y = side * P["w"] + P["kyv"] * vy
    fs_y = feet[st][1]
    Y = fs_y + side * max(side * (Y - fs_y), P["sep"])    # never cross the stance foot
    sm = s * s * (3 - 2 * s)
    fx = (1 - sm) * _S["lift"][0] + sm * X
    fy = (1 - sm) * _S["lift"][1] + sm * Y
    hip_h = obs["height"] - 0.465                      # hip height above ground (approx.)
    fz = -(hip_h - 0.075) + P["clr"] * np.sin(np.pi * s) - P["ext"] * max(ph - P["T"], 0.0)
    # leg vector from hip (hip is ~hc below the mass centre, 0.1 to the side)
    cyw, syw = np.cos(p_yaw), np.sin(p_yaw)
    hx_w, hy_w = -syw * side * 0.1, cyw * side * 0.1
    rx, ry = fx - hx_w, fy - hy_w
    rxp, ryp = cyw * rx + syw * ry, -syw * rx + cyw * ry   # into pelvis heading frame
    rz = fz
    l = np.clip(np.sqrt(rxp ** 2 + ryp ** 2 + rz ** 2), 0.35, A_TH + B_SH - 0.005)
    knee = -(np.pi - np.arccos(np.clip((A_TH ** 2 + B_SH ** 2 - l ** 2) / (2 * A_TH * B_SH), -1, 1)))
    beta = np.arccos(np.clip((A_TH ** 2 + l ** 2 - B_SH ** 2) / (2 * A_TH * l), -1, 1))
    phi = np.arctan2(rxp, -rz) + beta
    psi = np.arctan2(ryp, -rz)
    hx, hz, hy, kn = HIP[sw]
    tau[hy] = _pd(P["kp_h"], P["kd_h"], -phi - p_pitch - q[hy], dq[hy])
    hx_d = (psi - p_roll) if sw == 0 else -(psi - p_roll)
    tau[hx] = _pd(P["kp_hx"], P["kd_hx"], hx_d - q[hx], dq[hx])
    hz_d = P["align"] * np.clip(-p_yaw if sw == 0 else p_yaw, -0.6, 0.6)
    tau[hz] = _pd(60, 3, hz_d - q[hz], dq[hz])
    tau[kn] = _pd(P["kp_k"], P["kd_k"], knee - q[kn], dq[kn])

    # ---- stance leg: torso attitude through the hip, knee straight
    sx, sz, sy_, sk = HIP[st]
    sgn = 1.0 if st == 0 else -1.0
    t_pitch = P["kp_t"] * (pitch - P["pitch_d"]) + P["kd_t"] * w[1]
    t_roll = P["kp_r"] * p_roll + P["kd_r"] * (w[0] + dq[AX])
    t_yaw = P["kp_y"] * yaw + P["kd_y"] * w[2]
    tau[sy_] = t_pitch - tau[hy]
    tau[sx] = sgn * t_roll + tau[hx]
    tau[sz] = sgn * t_yaw + tau[hz]
    if contact[0] and contact[1]:     # both feet planted: steer the heading with a friction couple
        t_yd = P["kp_yd"] * yaw + P["kd_yd"] * w[2]
        tau[sz] += sgn * t_yd
        tau[hz] -= sgn * t_yd
    tau[sk] = _pd(P["kp_ks"], P["kd_ks"], 0.0 - q[sk], dq[sk])

    if t < P["t_stand"]:      # initial drop: both legs straight, torso held by both hips
        for leg in (0, 1):
            hx, hz, hy, kn = HIP[leg]
            sg = 1.0 if leg == 0 else -1.0
            tau[hy] = 0.5 * t_pitch
            tau[hx] = sg * 0.5 * t_roll
            tau[hz] = _pd(60, 3, -q[hz], dq[hz])
            tau[kn] = _pd(P["kp_ks"], P["kd_ks"], 0.0 - q[kn], dq[kn])

    for i in range(11, 17):
        tau[i] = _pd(10, 0.5, -q[i], dq[i])
    lim = 0.4 * P["lim"] * GEAR0 / obs["gear"]
    return np.clip(np.clip(tau / obs["gear"], -lim, lim), -0.4, 0.4)


_PRED = {}


def _predict(obs: dict, pending: np.ndarray) -> dict:
    """One-step-ahead prediction (Smith predictor): replay the pending command on a nominal model."""
    if "env" not in _PRED:
        from hl_control.envs.humanoid import HumanoidConfig, HumanoidEnv
        env = HumanoidEnv(HumanoidConfig(scenario="standard_v0"))
        env.reset(seed=0)
        _PRED["env"] = env
    env = _PRED["env"]
    g = obs["gym_obs"]
    env.data.qpos[:2] = 0.0
    env.data.qpos[2:] = g[:22]
    env.data.qpos[3:7] /= np.linalg.norm(env.data.qpos[3:7])
    env.data.qvel[:] = g[22:45]
    env.data.xfrc_applied[:] = 0.0
    mujoco = __import__("mujoco")
    mujoco.mj_forward(env.model, env.data)
    env.step_count, env.healthy, env.t = 0, True, 0.0
    env.action_queue = []
    env.com_velocity_xy = np.asarray(obs["com_velocity_xy"], float).copy()
    pred, _, _, _ = env.step(pending)
    pred["time"], pred["step"] = obs["time"] + obs["dt"], obs["step"] + 1
    pred["gear"] = obs["gear"]
    return pred


def controller(obs: dict) -> np.ndarray:
    """Delay-aware wrapper: if the applied command lags our output by one step, act on the predicted state."""
    if obs["step"] == 0:
        _S["out"], _S["out2"], _S["delay"] = np.zeros(17), np.zeros(17), False
    elif obs["step"] >= 3 and not _S["delay"]:
        la = obs["last_action"]
        if np.abs(la - _S["out"]).max() > 1e-6 and np.abs(la - _S["out2"]).max() < 1e-6:
            _S["delay"] = True
    if _S["delay"] and PARAMS["smith"] > 0:
        obs = _predict(obs, _S["out"])
    out = _control(obs, PARAMS)
    _S["out2"], _S["out"] = _S["out"], out
    return out
