"""Adroit door: grasp the lever, roll the latch open, pull the door along its arc, let go and let it swing.

Control law
-----------
The controller keeps a private copy of the plant's MuJoCo model and uses it ONLY for forward kinematics and
gravity torques (it is never stepped).  Every 10 ms it

1. estimates what the observation does not report: the arm slide (measured palm point minus the forward
   kinematics of the measured joint angles) and the door-frame position (running mean of the measured handle
   point with the known latch / hinge rotations taken out, which also averages observation noise away);
2. chooses a Cartesian target for a hand-fixed grasp point G (the crook between palm and curled fingers, where
   the lever sits when it is held) from the measured latch angle phi and hinge angle theta:
     APPROACH  G moves on a minimum-jerk line to a point above the lever, hand flat, fingers open, thumb folded;
     GRASP     G descends onto the lever axis and the fingers curl round the lever (knuckles flexed too);
     UNLATCH   G rides the lever's circle about the latch axis and the arm roll follows the lever:
               commanded latch angle = measured phi + lead_phi  (the lead is what pushes the latch round);
     PULL      G rides the handle's circle about the door hinge:
               commanded hinge angle = measured theta + lead_theta + t_ff * hinge rate
               (the lead is the pull; the rate term cancels the arm's tracking lag once the door moves);
     THROW     at theta_open the fingers straighten, the yaw swings to its stop ahead of the lever and the slide
               retracts until the open hand is clear of the handle's circle; the door coasts to its stop and
               stays there on its own hinge friction (DONE: arm frozen in that pose);
3. solves the 3x3 inverse kinematics (slide, pitch, yaw) for G by Newton iteration on the kinematic model, the
   roll being set directly;
4. tracks the joint targets with   ctrl = (k q* + gravity) / g + Kp (q* - q) - Kd qdot   on each arm joint (the
   first term inverts the servo's own law g*ctrl - k*q with gravity fed forward; joint rates are low-passed
   differences), and commands the wrist and finger position servos directly.

Gains (GAINS)
-------------
r_grasp        where the hand holds the lever: distance from the latch axis [m]
h_pre          height of the pre-grasp point above the lever [m]
t_app, t_desc, t_close   approach, descent and finger-closing durations [control steps]
lead_phi       latch-angle lead of the roll / grasp-point target while unlatching [rad] (push on the latch)
phi_go         latch angle at which the pull starts [rad] (the bolt clears the post near 1.2)
phi_top        largest commanded roll [rad]
lead_phi_pull  latch-angle lead kept during the pull so that the latch is not dragged back [rad]
lead_theta     hinge-angle lead of the pull target [rad] (pull strength; too much tears the fingers off)
t_ff           hinge-rate feed-forward time [s] (sets the cruising speed of the door)
theta_top      largest commanded hinge angle [rad]
theta_open     hinge angle at which the hand lets go and gets out of the way [rad]
t_rel, theta_clear   earlier release for a fast door: let go once theta + t_rel * hinge rate exceeds theta_clear
dodge_clear    clearance between the retracted fingertips and the handle's circle [m]
wrist_flex     wrist flexion set point while unlatching [rad] (stiffens the wrist against the latch torque)
j2_closed      knuckle flexion of the closed hand [rad] (wraps the fingers tightly round the lever)
kp_rot, kd_rot extra stiffness / damping on the arm rotations [ctrl per rad, ctrl per rad/s]
kp_sl, kd_sl   extra stiffness / damping on the arm slide [ctrl per m, ctrl per m/s]
"""

import importlib.util
import os

import mujoco
import numpy as np

# A fixed, jointly validated set (deliberately not exposed as a tunable PARAMS block: some gains have a cliff
# nearby: kp_sl below about 1.8 lets the fingers slip off in the pull, lead_theta above 0.2 does on a heavier door).
GAINS = {
    "r_grasp": 0.12,
    "h_pre": 0.04,
    "t_app": 26.0,
    "t_desc": 8.0,
    "t_close": 10.0,
    "lead_phi": 0.5,
    "phi_go": 1.3,
    "phi_top": 2.0,
    "lead_phi_pull": 0.25,
    "lead_theta": 0.15,
    "t_ff": 0.03,
    "theta_top": 1.5,
    "theta_open": 0.6,
    "t_rel": 0.18,
    "theta_clear": 1.0,
    "dodge_clear": 0.10,
    "wrist_flex": 0.3,
    "j2_closed": 1.0,
    "kp_rot": 0.8,
    "kd_rot": 0.0,
    "kp_sl": 2.5,
    "kd_sl": 0.2,
}

G_LOCAL = np.array([0.0, -0.03, 0.105])  # grasp point in the palm frame: the lever axis in the finger crook
YAW_STOP = -0.75                         # arm yaw limit on the hinge side [rad]
HAND_REACH = 0.62                        # arm base to straight fingertips [m]

# ----------------------------------------------------------------------------- private kinematic model


def _load_model():
    """The plant's own model file (kinematics and masses only; the servo gains the plant patches in are not used)."""
    try:
        spec = importlib.util.find_spec("gymnasium_robotics")
        path = os.path.join(os.path.dirname(spec.origin), "envs", "assets", "adroit_hand", "adroit_door.xml")
        return mujoco.MjModel.from_xml_path(path)
    except Exception:  # fall back on the model the benchmark wrapper builds
        from hl_control.envs.adroit import AdroitConfig, AdroitEnv
        return AdroitEnv(AdroitConfig(task="door")).model


_M = _load_model()
_D = mujoco.MjData(_M)
_PALM = _M.body("palm").id
_SGRASP = _M.site("S_grasp").id
_ARM_G = np.full(4, 500.0)   # arm servo gain g
_ARM_K = np.full(4, 200.0)   # arm servo stiffness k
_LO = np.array([-0.3, -0.75, -0.75])   # slide, pitch, yaw ranges
_HI = np.array([0.5, 0.75, 0.75])


def _fk(arm, w1, w0):
    """World position of G for arm = (slide, pitch, yaw, roll) and the two wrist angles."""
    q = _D.qpos
    q[:] = 0.0
    q[0:4] = arm
    q[4] = w1
    q[5] = w0
    mujoco.mj_kinematics(_M, _D)
    return _D.xpos[_PALM] + _D.xmat[_PALM].reshape(3, 3) @ G_LOCAL


def _ik(target, roll, w1, w0, guess):
    """Newton iteration for (slide, pitch, yaw) that puts G on target, inside the joint ranges."""
    x = np.array(guess, dtype=float)
    arm = np.zeros(4)
    arm[3] = roll
    for _ in range(6):
        arm[:3] = x
        p = _fk(arm, w1, w0)
        err = target - p
        if np.linalg.norm(err) < 1e-4:
            break
        J = np.zeros((3, 3))
        for j in range(3):
            arm[:3] = x
            arm[j] += 1e-4
            J[:, j] = (_fk(arm, w1, w0) - p) / 1e-4
        dx = np.linalg.solve(J.T @ J + 1e-6 * np.eye(3), J.T @ err)
        x = np.clip(x + np.clip(dx, -0.3, 0.3), _LO, _HI)
    return x


def _gravity(q30):
    """Gravity torque on the four arm joints at configuration q30."""
    _D.qpos[:] = q30
    _D.qvel[:] = 0.0
    mujoco.mj_forward(_M, _D)
    return _D.qfrc_bias[0:4].copy()


def _slide_from_palm(qobs27, palm_y):
    """The unobserved arm slide: measured palm y minus the palm y the measured joints give at zero slide."""
    q = _D.qpos
    q[:] = 0.0
    q[1:28] = qobs27
    mujoco.mj_kinematics(_M, _D)
    return palm_y - _D.site_xpos[_SGRASP][1]


def _rz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _hinge_to_lever(r, phi, theta):
    """From the door hinge to the point of the lever at distance r from the latch axis (latch phi, hinge theta)."""
    return _rz(theta) @ np.array([-0.46 + r * np.cos(phi), -0.15, -0.025 - r * np.sin(phi)])


def _lever_point(frame, r, phi, theta):
    return frame + np.array([0.6, 0.0, 0.0]) + _hinge_to_lever(r, phi, theta)


def _minjerk(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)


# ----------------------------------------------------------------------------- controller

_S = {}


def _reset():
    _S.clear()
    _S.update(phase="APPROACH", t_phase=0, frame_sum=np.zeros(3), frame_n=0, q_prev=None, s_prev=None,
              sd=0.0, qd=np.zeros(3), om=0.0, th_prev=0.0, ik=np.zeros(3), g0=None, hold=None, phi_cmd=0.0)


def controller(obs: dict) -> np.ndarray:
    P = GAINS
    o = np.asarray(obs["observation"], dtype=float)
    mean, half = np.asarray(obs["ctrl_mean"], dtype=float), np.asarray(obs["ctrl_half_range"], dtype=float)
    if int(obs["step"]) == 0 or not _S:
        _reset()
    S = _S
    qobs, phi, theta = o[0:27], float(o[27]), float(o[28])
    palm, handle = o[29:32], o[32:35]
    dt = float(obs["dt"])

    # --- estimates: door-frame position, arm slide, joint and hinge rates
    S["frame_sum"] += handle - np.array([0.6, 0.0, 0.0]) - _hinge_to_lever(0.15, phi, theta)
    S["frame_n"] += 1
    frame = S["frame_sum"] / S["frame_n"]
    s_est = float(_slide_from_palm(qobs, palm[1]))
    rot = qobs[0:3].copy()  # pitch, yaw, roll
    if S["s_prev"] is None:
        S["s_prev"], S["q_prev"] = s_est, rot.copy()
    S["sd"] = 0.5 * S["sd"] + 0.5 * (s_est - S["s_prev"]) / dt
    S["qd"] = 0.5 * S["qd"] + 0.5 * (rot - S["q_prev"]) / dt
    S["s_prev"], S["q_prev"] = s_est, rot.copy()
    S["om"] = 0.6 * S["om"] + 0.4 * (theta - S["th_prev"]) / dt
    S["th_prev"] = theta
    arm_now = np.array([s_est, rot[0], rot[1], rot[2]])
    if S["g0"] is None:
        S["g0"] = _fk(arm_now, qobs[3], qobs[4])

    # --- phase machine -> grasp-point target, roll, wrist flexion, finger curl (0 open, 1 closed, -1 straight)
    S["t_phase"] += 1
    tp = S["t_phase"]
    rg = P["r_grasp"]
    roll, w0, curl = 0.0, 0.0, 0.0
    if S["phase"] == "APPROACH":
        pre = _lever_point(frame, rg, 0.0, 0.0) + np.array([0.0, 0.0, P["h_pre"]])
        target = S["g0"] + _minjerk(tp / P["t_app"]) * (pre - S["g0"])
        if tp >= P["t_app"] + 2:
            S["phase"], S["t_phase"] = "GRASP", 0
    elif S["phase"] == "GRASP":
        h = P["h_pre"] * (1.0 - _minjerk(tp / P["t_desc"]))
        target = _lever_point(frame, rg, phi, 0.0) + np.array([0.0, 0.0, h])
        roll = phi
        curl = float(np.clip((tp - 0.5 * P["t_desc"]) / 3.0, 0.0, 1.0))
        if tp >= P["t_desc"] + P["t_close"]:
            S["phase"], S["t_phase"] = "UNLATCH", 0
    elif S["phase"] == "UNLATCH":
        S["phi_cmd"] = min(phi + P["lead_phi"] * min(1.0, tp / 5.0), P["phi_top"])
        target = _lever_point(frame, rg, S["phi_cmd"], theta)
        roll, w0, curl = S["phi_cmd"], P["wrist_flex"], 1.0
        if phi >= P["phi_go"]:
            S["phase"], S["t_phase"] = "PULL", 0
    elif S["phase"] == "PULL":
        th_cmd = min(theta + P["lead_theta"] * min(1.0, tp / 5.0) + P["t_ff"] * max(S["om"], 0.0), P["theta_top"])
        S["phi_cmd"] = min(phi + P["lead_phi_pull"], P["phi_top"])
        target = _lever_point(frame, rg, phi, th_cmd)
        roll, curl = S["phi_cmd"], 1.0
        # let go at theta_open, or earlier if the door is fast: the fingers need about t_rel to open, and by
        # then the lever must not have reached the hand waiting at the yaw stop (near theta_clear)
        if theta > P["theta_open"] or (theta > 0.3 and theta + P["t_rel"] * S["om"] > P["theta_clear"]):
            S["phase"] = "THROW"  # from the next step on
    elif S["phase"] == "THROW":
        # let go: yaw to its stop (ahead of the lever), slide back until the straight fingers are clear of the
        # handle's circle (radius 0.484 m about the hinge), pitch and roll kept
        s_dodge = (frame[1] - 0.484 - P["dodge_clear"]) - HAND_REACH * np.cos(YAW_STOP) + 0.7
        S["hold"] = np.array([float(np.clip(s_dodge, -0.3, 0.5)), S["ik"][1], YAW_STOP, S["phi_cmd"]])
        S["phase"] = "DONE"
    if S["phase"] == "DONE":
        q_t, curl = S["hold"], -1.0
    else:
        x = _ik(target, roll, 0.0, w0, S["ik"])
        S["ik"] = x.copy()
        q_t = np.array([x[0], x[1], x[2], roll])

    # --- arm servo: inverse of the servo law with gravity feed-forward, plus PD on the measured joints
    q30 = np.zeros(30)
    q30[0] = s_est
    q30[1:28] = qobs
    kp = np.array([P["kp_sl"], P["kp_rot"], P["kp_rot"], P["kp_rot"]])
    kd = np.array([P["kd_sl"], P["kd_rot"], P["kd_rot"], 0.0])
    vel = np.array([S["sd"], S["qd"][0], S["qd"][1], S["qd"][2]])
    ctrl = np.zeros(28)
    ctrl[0:4] = (_ARM_K * q_t + _gravity(q30)) / _ARM_G + kp * (q_t - arm_now) - kd * vel
    ctrl[5] = w0  # wrist flexion; wrist deviation stays at zero

    # --- fingers: fore/middle/ring (J3, J2, J1, J0), little (J4, J3, J2, J1, J0), thumb folded under the palm
    if curl < 0.0:
        j2, j1, j0 = 0.0, 0.0, 0.0
    else:
        j2, j1, j0 = P["j2_closed"] * curl, 0.5 + 1.1 * curl, 0.2 + 1.4 * curl
    for base in (6, 10, 14):
        ctrl[base:base + 4] = (0.0, j2, j1, j0)
    ctrl[18:23] = (0.0, 0.0, j2, j1, j0)
    ctrl[23:28] = (0.0, 1.3, 0.0, 0.0, 0.0)

    return np.clip((ctrl - mean) / half, -1.0, 1.0)
