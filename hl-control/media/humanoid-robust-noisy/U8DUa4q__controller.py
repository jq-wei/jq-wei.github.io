"""SIMBICON-style walking controller for the MuJoCo Humanoid (Gymnasium Humanoid-v5).

Control law
-----------
* Joint PD on every motor: tau = kp (q_d - q) - kd qdot (clipped to the motor's
  torque limit), command = tau / gear.  ``kp_s`` / ``kd_s`` scale the base gains.
* Two-state gait machine: the stance leg alternates.  A swap happens when the
  swing foot touches down after ``t_min`` seconds, or unconditionally at ``t_max``.
* Swing hip (sagittal): the thigh is driven to the world-frame angle
  ``-(phi0 + cd * d + cv * (v - v_d))`` where d is the forward offset of the mass
  centre from the stance foot and v its forward velocity (SIMBICON foot placement).
  Laterally the swing thigh goes to the world roll ``width + cd_lat d_y + cv_lat v_y``.
* Torso: a virtual PD on pitch (target ``lean``; ``kp_t``, ``kd_t``) and on roll
  (``kp_r``, ``kd_r``) is applied through the stance hip, whose torque is
  ``-tau_torso - tau_swing_hip`` (the hip torque that makes the net torque on the
  pelvis equal the desired torso torque).
* Heading: the swing hip_z turns the landing leg by ``-yaw_k * yaw`` so that the next
  stance phase brings the pelvis back towards +x.
* Swing knee flexes to ``knee_swing`` for the first ``swing_frac`` of ``t_max`` and
  extends to ``knee_ext`` afterwards; the stance knee is held at ``knee_stance``.
* Command delay is detected from ``obs["last_action"]`` (it then equals the command
  sent two calls earlier).  From then on the law runs on a one-step-ahead prediction:
  the observed state (from ``gym_obs``) is copied into a nominal MuJoCo model and
  stepped 15 ms under the command that is already in the pipe (a Smith predictor).
  Joint angles feed the predictor through an observer,
  ``q_hat = q_pred + OBS_GAIN[0] (q_meas - q_pred)`` (rates with ``OBS_GAIN[1]``), which
  filters the 0.01 rad angle noise that would otherwise put the predicted feet into the floor.

Gains (PARAMS, tuned by a diagonal evolution strategy over all six scenarios)
-----------------------------------------------------------------------------
phi0, cd, cv, v_d   foot placement: base swing-thigh angle, gains on COM offset / velocity, target speed
lean                torso pitch target (rad, forward positive)
kp_t, kd_t          torso pitch PD through the stance hip;  kp_r, kd_r: the same on roll
t_min, t_max        stance timer (s)
knee_swing, knee_ext, knee_stance, swing_frac   knee schedule
width, cd_lat, cv_lat   lateral placement: step width (rad of thigh roll), gains on COM offset / velocity
yaw_k               heading correction through the swing hip_z
kp_s, kd_s          scale of the base joint gains KP / KD;  ab_kp, ab_kd: abdomen pitch/roll PD
"""

import numpy as np

PARAMS = {
    "phi0": 0.2256,
    "cd": 0.6168,
    "cv": 0.2874,
    "v_d": 0.2929,
    "lean": 0.0581,
    "kp_t": 370.2092,
    "kd_t": 21.8546,
    "kp_r": 150.0751,
    "kd_r": 16.1753,
    "t_min": 0.1461,
    "t_max": 0.5381,
    "knee_swing": -0.3205,
    "knee_ext": -0.1856,
    "knee_stance": -0.0814,
    "swing_frac": 0.5682,
    "width": 0.0308,
    "cd_lat": 0.4547,
    "cv_lat": 0.1894,
    "yaw_k": 0.2702,
    "kp_s": 1.0618,
    "ab_kp": 327.7324,
    "ab_kd": 11.1007,
    "kd_s": 1.2857,
}

# motor indices
AB_Y, AB_Z, AB_X = 0, 1, 2
HX = (3, 7)
HZ = (4, 8)
HY = (5, 9)
KN = (6, 10)
SIDE = (1.0, -1.0)  # hip_x / hip_z axes: +x/+z on the right leg, -x/-z on the left
ARMS = slice(11, 17)

# base joint gains, chosen against the joint-space inertia (hip_z 0.026, knee 0.36, arm
# 0.06-0.18 kg m^2): low-inertia joints get low stiffness and only the passive joint damping,
# otherwise the 15 ms sampled PD chatters (and diverges with one step of delay)
KP = np.array([300, 40, 300, 300, 20, 300, 300, 300, 20, 300, 300, 8, 8, 4, 8, 8, 4], float)
KD = np.array([8, 1, 8, 6, 0, 6, 4, 6, 0, 6, 4, 0.2, 0.2, 0.1, 0.2, 0.2, 0.1], float)

_S = {"stance": 0, "t0": 0.0, "prev": [np.zeros(17), np.zeros(17)], "delay": False}


def _reset(obs):
    _S["stance"] = 0 if obs["foot_xyz_rel"][0] < obs["foot_left_xyz_rel"][0] else 1
    _S["t0"] = 0.0
    _S["prev"] = [np.zeros(17), np.zeros(17)]
    _S["delay"] = False
    _S["xhat"] = None


_PRED = {}
OBS_GAIN = (0.1, 1.0)   # observer gains on joint angles / rates (1 = trust the measurement)


def _predict(obs, u_pending):
    """Simulate the 15 ms that the pending (already sent) command will act for, from the observed state."""
    if not _PRED:
        from hl_control.envs.humanoid import HumanoidEnv
        _PRED["env"] = HumanoidEnv()
    env = _PRED["env"]
    m, d = env.model, env.data
    g = obs["gym_obs"]
    d.qpos[:2] = 0.0
    d.qpos[2:] = g[: m.nq - 2]
    d.qvel[:] = g[m.nq - 2: m.nq - 2 + m.nv]
    xh = _S.get("xhat")
    if xh is not None:   # observer: blend the noisy joint measurement with last step's prediction
        d.qpos[7:] = xh[0] + OBS_GAIN[0] * (d.qpos[7:] - xh[0])
        d.qvel[6:] = xh[1] + OBS_GAIN[1] * (d.qvel[6:] - xh[1])
    import mujoco
    mujoco.mj_forward(m, d)
    com0 = mass_center_xy(m, d)
    d.ctrl[:] = u_pending
    mujoco.mj_step(m, d, nstep=env.config.frame_skip)
    _S["xhat"] = (d.qpos[7:].copy(), d.qvel[6:].copy())
    env.com_velocity_xy = (mass_center_xy(m, d) - com0) / obs["dt"]
    env.last_action = np.asarray(u_pending, float)
    env.step_count = obs["step"] + 1
    env.t = obs["time"] + obs["dt"]
    return env._get_obs()


def mass_center_xy(m, d):
    return (m.body_mass @ d.xipos / m.body_mass.sum())[:2]


def controller(obs: dict) -> np.ndarray:
    if obs["step"] == 0:
        _reset(obs)
    if obs["step"] >= 3 and not _S["delay"]:
        la = obs["last_action"]
        if np.allclose(la, _S["prev"][0], atol=1e-9) and not np.allclose(la, _S["prev"][1], atol=1e-6):
            _S["delay"] = True
    if _S["delay"]:
        obs = _predict(obs, _S["prev"][1])
    u = _law(obs, PARAMS)
    _S["prev"] = [_S["prev"][1], u.copy()]
    return u


def _law(obs, P):
    t = obs["time"]
    dt = obs["dt"]
    q = obs["joint_angles"].copy()
    qd = obs["joint_rates"].copy()
    gear = obs["gear"]
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    w = obs["root_angular_velocity"]

    feet = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])
    contact = (obs["foot_contact"], obs["foot_left_contact"])
    st = _S["stance"]
    sw = 1 - st
    phase_t = t - _S["t0"]
    if (phase_t > P["t_min"] and contact[sw]) or phase_t > P["t_max"]:
        _S["stance"], st, sw = sw, sw, st
        _S["t0"] = t
        phase_t = 0.0

    # heading frame
    cy, sy = np.cos(yaw), np.sin(yaw)
    v = obs["com_velocity_xy"]
    vf, vl = cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]
    fs = feet[st]
    d_f = -(cy * fs[0] + sy * fs[1])
    d_l = -(-sy * fs[0] + cy * fs[1])

    qdes = np.zeros(17)
    qdes[KN[st]] = P["knee_stance"]
    qdes[KN[sw]] = P["knee_swing"] if phase_t < P["swing_frac"] * P["t_max"] else P["knee_ext"]
    thigh = -(P["phi0"] + P["cd"] * d_f + P["cv"] * (vf - P["v_d"]))
    qdes[HY[sw]] = thigh - pitch - q[AB_Y]
    out = -1.0 if sw == 0 else 1.0          # outward direction (+y is left)
    lat = out * P["width"] + P["cd_lat"] * d_l + P["cv_lat"] * vl
    qdes[HX[sw]] = SIDE[sw] * (lat - roll - q[AB_X])
    qdes[HZ[sw]] = -SIDE[sw] * P["yaw_k"] * yaw

    kp, kd = KP * P["kp_s"], KD * P["kd_s"]
    kp[[AB_Y, AB_X]] = P["ab_kp"]
    kd[[AB_Y, AB_X]] = P["ab_kd"]
    tmax = 0.4 * gear
    tau = np.clip(kp * (qdes - q) - kd * qd, -tmax, tmax)

    # torso virtual PD through the stance hip
    tq_p = P["kp_t"] * (P["lean"] - pitch) - P["kd_t"] * w[1]
    tq_r = P["kp_r"] * (0.0 - roll) - P["kd_r"] * w[0]
    tau[HY[st]] = -tq_p - tau[HY[sw]]
    tau[HX[st]] = SIDE[st] * (-tq_r - SIDE[sw] * tau[HX[sw]])

    return np.clip(tau / gear, -0.4, 0.4)
