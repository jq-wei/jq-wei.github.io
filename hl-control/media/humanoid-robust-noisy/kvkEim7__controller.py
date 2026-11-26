"""SIMBICON-style walking controller for the MuJoCo humanoid.

Two-state machine (right leg swings / left leg swings). In each state

* the swing hip tracks a foot-placement angle  th = th0 + cd*d + cv*(v - vdes)
  (d = COM position relative to the stance foot, v = COM velocity, heading frame),
  the swing knee flexes for clearance and extends before touchdown;
* the stance knee is held nearly straight; the stance hip applies the torque that keeps the
  pelvis/torso at the desired pitch and roll (reaction torque of the swing hip is cancelled);
* the swing foot is placed laterally by an analogous law in the frontal plane;
* abdomen and arms are PD-held at fixed postures.

All PD outputs are torques converted to motor commands through the actuator gear.
"""
import numpy as np

PARAMS = {
    "T": 0.13428,
    "Tmax": 0.68335,
    "th0": 0.41958,
    "cd": 0.79455,
    "cv": 0.50225,
    "vdes": 0.42949,
    "kn_sw": 0.90666,
    "kn_land": 0.25079,
    "kn_st": 0.05986,
    "pitch_des": 0.10474,
    "kp_t": 124.91017,
    "kd_t": 11.54941,
    "kp_h": 214.8238,
    "kd_h": 7.29054,
    "w0": 0.06649,
    "cdy": 0.59727,
    "cvy": 0.32756,
    "roll_bias": -0.01594,
    "kp_ab": 78.69918,
    "kd_ab": 6.25429,
    "kp_arm": 8.0,
    "kd_arm": 1.0,
    "yaw_k": 1.0,
    "dscale": 0.5,    # PD gain scale under command delay
    "lead": 1.0,      # state prediction horizon [control steps] when a command delay is detected
    "t_start": 0.4,
}
PARAM_BOUNDS = {}

# Parameters used once a one-step command delay is detected (tuned separately on the delay scenario).
DELAY_PARAMS = {
    "T": 0.21801,
    "Tmax": 0.68335,
    "th0": 0.69185,
    "cd": 0.71949,
    "cv": 0.66775,
    "vdes": 0.33875,
    "kn_sw": 1.40457,
    "kn_land": 0.21954,
    "kn_st": 0.05731,
    "pitch_des": 0.049,
    "kp_t": 195.49188,
    "kd_t": 18.14089,
    "kp_h": 248.94388,
    "kd_h": 4.05223,
    "w0": 0.10319,
    "cdy": 1.03203,
    "cvy": 0.40605,
    "roll_bias": 0.05989,
    "kp_ab": 154.59711,
    "kd_ab": 3.0,
    "kp_arm": 8.0,
    "kd_arm": 1.0,
    "yaw_k": 1.0,
    "dscale": 0.39779,
    "lead": 1.30257,
    "t_start": 0.4,
}

_S = {"stance": 0, "tau": 0.0, "last_step": -1}


def _pd(kp, kd, q, qd, qdot):
    return kp * (qd - q) - kd * qdot


def _gains(obs):
    """Parameters, with the PD gains of the legs scaled down when a command delay is detected."""
    P = dict(PARAMS)
    if _S.get("delayed"):
        P.update(DELAY_PARAMS)
        for k in ("kp_h", "kd_h", "kp_t", "kd_t", "kp_ab", "kd_ab"):
            P[k] *= P["dscale"]
    return P


def _stand(obs):
    """Double support, legs straight, torso held upright by both hips."""
    P = _gains(obs)
    q, qd = obs["joint_angles"], obs["joint_rates"]
    tq = np.zeros(17)
    tau_t = P["kp_t"] * (P["pitch_des"] - obs["pitch"]) - P["kd_t"] * obs["root_angular_velocity"][1]
    tq[5] = tq[9] = -0.5 * tau_t
    for k in (6, 10):
        tq[k] = _pd(P["kp_h"] * 1.5, P["kd_h"], q[k], -P["kn_st"], qd[k])
    for i in (0, 1, 2):
        tq[i] = _pd(P["kp_ab"], P["kd_ab"], q[i], 0.0, qd[i])
    for i in (3, 4, 7, 8):
        tq[i] = _pd(P["kp_h"], P["kd_h"], q[i], 0.0, qd[i])
    for i in range(11, 17):
        tq[i] = _pd(P["kp_arm"], P["kd_arm"], q[i], 0.0, qd[i])
    return np.clip(tq / obs["gear"], -0.4, 0.4)


def _delayed(obs):
    """True when the command applied last step is not the one returned one step earlier (command delay)."""
    if obs["step"] == 0:
        _S.update(prev=np.zeros(17), delayed=False)
        return False
    if obs["step"] >= 2 and np.max(np.abs(_S["prev"])) > 1e-6:
        if np.max(np.abs(obs["last_action"] - _S["prev"])) > 1e-9:
            _S["delayed"] = True
    return _S["delayed"]


def controller(obs: dict) -> np.ndarray:
    a = _controller(obs)
    _S["prev"] = a
    return a


def _controller(obs: dict) -> np.ndarray:
    _delayed(obs)
    P = _gains(obs)
    if obs["step"] == 0:
        _S.update(stance=0, tau=0.0)
    dt = obs["dt"]
    gear = obs["gear"]
    delayed = _delayed(obs)
    lead = P["lead"] * dt if delayed else 0.0
    qd = obs["joint_rates"]
    q = obs["joint_angles"] + lead * qd
    w = obs["root_angular_velocity"]
    yaw = obs["yaw"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    pitch, roll = obs["pitch"] + lead * w[1], obs["roll"] + lead * w[0]
    wx, wy = w[0], w[1]
    com_v = obs["root_velocity"]
    vx = cy * com_v[0] + sy * com_v[1]
    vy = -sy * com_v[0] + cy * com_v[1]

    if obs["time"] < P["t_start"]:
        return _stand(obs)

    # state machine: stance 0 = right leg on the ground, left swings
    st = _S["stance"]
    contact = (obs["foot_contact"], obs["foot_left_contact"])
    _S["tau"] += dt
    sw_contact = contact[1 - st]
    if (_S["tau"] >= P["T"] and sw_contact) or _S["tau"] >= P["Tmax"]:
        st = 1 - st
        _S["stance"] = st
        _S["tau"] = 0.0
    tau_s = _S["tau"]
    sw = 1 - st

    fr = obs["foot_xyz_rel"] if st == 0 else obs["foot_left_xyz_rel"]
    dx = -(cy * fr[0] + sy * fr[1])
    dy = -(-sy * fr[0] + cy * fr[1])           # COM minus stance foot, heading frame

    hy = (5, 9)     # hip_y indices (right, left)
    hx = (3, 7)
    hz = (4, 8)
    kn = (6, 10)
    a_sign = (1.0, -1.0)   # hip_x axis sign (right +x, left -x)
    tq = np.zeros(17)

    # --- sagittal ---
    th_sw = P["th0"] + P["cd"] * dx + P["cv"] * (vx - P["vdes"])
    th_sw = np.clip(th_sw, -0.3, 0.8)
    pel_pitch = pitch + q[0]
    hy_sw_d = -th_sw - pel_pitch
    tau_sw = _pd(P["kp_h"], P["kd_h"], q[hy[sw]], hy_sw_d, qd[hy[sw]])
    tau_t = P["kp_t"] * (P["pitch_des"] - pitch) - P["kd_t"] * wy
    tq[hy[sw]] = tau_sw
    tq[hy[st]] = -tau_t - tau_sw

    # --- knees ---
    ph = tau_s / P["T"]
    kn_sw_d = -(P["kn_sw"] if ph < 0.6 else P["kn_land"])
    tq[kn[sw]] = _pd(P["kp_h"], P["kd_h"], q[kn[sw]], kn_sw_d, qd[kn[sw]])
    tq[kn[st]] = _pd(P["kp_h"] * 1.5, P["kd_h"], q[kn[st]], -P["kn_st"], qd[kn[st]])

    # --- frontal ---
    sg = 1.0 if sw == 1 else -1.0        # +1 when outward is +y (left swings)
    d_out = sg * dy
    v_out = sg * vy
    psi_out = P["w0"] + P["cdy"] * d_out + P["cvy"] * v_out
    psi_out = np.clip(psi_out, -0.08, 0.4)
    pel_roll = roll + q[2]
    if sw == 0:    # right swings: outward = -y ; hx_R = -psi_out - pel_roll
        hx_sw_d = -psi_out - pel_roll
    else:
        hx_sw_d = pel_roll - psi_out * 1.0
    # left: psi_world(+y) = pel_roll - hx_L ; outward(+y) psi_out => hx_L = pel_roll - psi_out
    tau_swx = _pd(P["kp_h"], P["kd_h"], q[hx[sw]], hx_sw_d, qd[hx[sw]])
    roll_des = P["roll_bias"] * (-sg)    # lean toward stance side (stance is opposite of swing)
    tau_tx = P["kp_t"] * (roll_des - roll) - P["kd_t"] * wx
    tq[hx[sw]] = tau_swx
    tq[hx[st]] = -a_sign[st] * tau_tx + tau_swx

    # --- yaw: hip_z of both legs PD to zero relative angle, abdomen z holds yaw ---
    for i in (0, 1):
        tq[hz[i]] = _pd(P["kp_h"], P["kd_h"], q[hz[i]], -P["yaw_k"] * yaw * (1 if i == 0 else -1) * 0.0, qd[hz[i]])

    # --- abdomen ---
    tq[0] = _pd(P["kp_ab"], P["kd_ab"], q[0], 0.0, qd[0])
    tq[1] = _pd(P["kp_ab"], P["kd_ab"], q[1], 0.0, qd[1])
    tq[2] = _pd(P["kp_ab"], P["kd_ab"], q[2], 0.0, qd[2])
    # --- arms ---
    for i in range(11, 17):
        tq[i] = _pd(P["kp_arm"], P["kd_arm"], q[i], 0.0, qd[i])

    return np.clip(tq / gear, -0.4, 0.4)
