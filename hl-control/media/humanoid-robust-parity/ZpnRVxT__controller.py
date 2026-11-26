"""SIMBICON-style walking controller for the Gymnasium Humanoid (point feet, torque motors).

Control law (joint-space PD; torques in N m, divided by the motor gear and clipped to +-0.4):
  * State machine: one stance leg, one swing leg; roles swap when the swing foot touches down
    after ``t_min`` (or at ``t_max``).  The first ``t_ds`` s is a two-legged landing phase, and
    each touchdown may start ``t_dsw`` s of double support where both hips share torso control.
  * Torso attitude through the stance hip (SIMBICON): stance hip torque = desired torso torque
    minus the swing hip torque actually applied.  Pitch: kp_t/kd_t about ``pitch_des`` (hip_y,
    capped at ``hip_lim``, swing compensation weight ``comp``); roll: kp_r/kd_r (hip_x);
    yaw: kp_y/kd_y (hip_z, effective only in double support: the spherical feet have no
    torsional friction).
  * Swing-foot placement (capture-point feedback, relative to the mass centre):
        x = a_x v_x + k_vx (v_x - v_des) + x_off - side (k_yaw yaw + k_yawd yaw_rate)
        y = side w + a_y v_y - side hip_w   (relative to the swing hip)
    the yaw terms steer by braking on one foot and pushing on the other.  The target is turned
    into thigh angles (leg length ``leg``) and hip targets by removing the pelvis pitch/roll
    (torso angle + ``pel`` x abdomen angle); PD gains kp_sw/kd_sw (hip_y), kp_swx/kd_swx (hip_x).
  * Swing knee folded to ``knee_swing`` for the first ``fold`` x ``T_swing`` s, then extended to
    ``knee_land``; stance knee held at ``knee_stance`` (kp_k/kd_k, extension capped at
    ``knee_push``); abdomen held straight (kp_ab/kd_ab); arms held still.
The numerical values were found by a derivative-free (cross-entropy) search over these gains
on the nominal and perturbed plants.
"""

import numpy as np

PARAMS = {
    "v_des": 1.1908,
    "pitch_des": 0.02,
    "kp_t": 223.5355,
    "kd_t": 14.8469,
    "kp_r": 226.6093,
    "kd_r": 20.3582,
    "kp_y": 1.9678,
    "kd_y": 0.1768,
    "a_x": 0.2366,
    "k_vx": -0.0334,
    "x_off": 0.0445,
    "a_y": 0.2586,
    "w": 0.1551,
    "kp_sw": 555.1509,
    "kd_sw": 5.3522,
    "kp_swx": 381.8067,
    "kd_swx": 7.9086,
    "knee_swing": -0.9468,
    "knee_land": -0.0142,
    "knee_stance": -0.0173,
    "fold": 0.178,
    "kp_k": 604.5112,
    "kd_k": 6.2947,
    "kp_ab": 384.8151,
    "kd_ab": 8.1491,
    "xt_min": 0.0412,
    "hip_w": 0.0115,
    "t_min": 0.1028,
    "t_ds": 0.4106,
    "t_max": 0.3692,
    "T_swing": 0.3167,
    "hf": 0.0,
    "k_yaw": 0.2233,
    "k_yawd": 0.0084,
    "pel": -0.2565,
    "t_dsw": 0.0004,
    "leg": 0.8561,
    "knee_ok": -2.0,
    "knee_push": 0.3727,
    "hip_lim": 139.8213,
    "comp": 0.8481,
    "air": 0.0,
}

# motor indices
AY, AZ, AX = 0, 1, 2
R = dict(x=3, z=4, y=5, k=6)
L = dict(x=7, z=8, y=9, k=10)
ARMS = slice(11, 17)

DEBUG = False
_state = {"stance": "L", "t": 0.0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    dt = obs["dt"]
    if obs["step"] == 0:
        _state["stance"], _state["t"] = "L", 0.0
    q, qd, gear = obs["joint_angles"], obs["joint_rates"], obs["gear"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    pp = pitch + P["pel"] * obs["joint_angles"][AY]  # pelvis pitch / roll estimates (torso + abdomen)
    pr = roll + P["pel"] * obs["joint_angles"][AX]
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cy, sy], [-sy, cy]])            # world -> heading frame
    w = obs["root_angular_velocity"]
    wx, wy = rot @ w[:2]
    wz = w[2]
    v = obs["com_velocity_xy"]                       # world frame (walk toward +x)
    if P["hf"] > 0:                                  # heading frame: walk where the torso faces
        v = rot @ v

    # phase switching
    _state["t"] += dt
    st = _state["stance"]
    sw_contact = obs["foot_contact"] if st == "L" else obs["foot_left_contact"]
    sw_knee = q[6] if st == "L" else q[10]
    if obs["time"] < P["t_ds"]:
        _state["t"] = 0.0
    elif (_state["t"] > P["t_min"] and sw_contact and sw_knee > P["knee_ok"]) or _state["t"] > P["t_max"]:
        st = _state["stance"] = "R" if st == "L" else "L"
        _state["t"] = 0.0
    t = _state["t"]
    S, W = (R, L) if st == "R" else (L, R)
    side_sw = -1.0 if st == "L" else 1.0             # swing foot lateral side (+y = left)

    tau = np.zeros(17)

    # abdomen straight, arms still
    tau[[AY, AZ, AX]] = P["kp_ab"] * (0.0 - q[[AY, AZ, AX]]) - P["kd_ab"] * qd[[AY, AZ, AX]]
    tau[ARMS] = 20.0 * (0.0 - q[ARMS]) - 1.0 * qd[ARMS]

    # swing foot target relative to the mass centre (heading frame)
    xt = P["a_x"] * v[0] + P["k_vx"] * (v[0] - P["v_des"]) + P["x_off"]
    yt = side_sw * P["w"] + P["a_y"] * v[1]
    xt += -side_sw * (P["k_yaw"] * yaw + P["k_yawd"] * wz)   # yaw steering: brake on one side, push on the other
    xt = np.clip(xt, P["xt_min"], 0.5)
    yt = np.clip(yt, -0.4, 0.4) if side_sw > 0 else np.clip(yt, -0.4, 0.4)
    if P["hf"] <= 0:
        xt, yt = rot @ np.array([xt, yt])            # world target -> heading frame
    yt -= side_sw * P["hip_w"]                       # relative to the swing hip, not the mass centre
    alpha = np.arcsin(np.clip(xt / P["leg"], -0.8, 0.8))   # thigh angle, + = forward
    beta = np.arcsin(np.clip(yt / P["leg"], -0.8, 0.8))    # thigh angle, + = toward +y
    # hip_y: world thigh pitch = pitch + q ; foot forward means negative
    qy_des = -alpha - pp
    # hip_x: right foot y-offset ~ roll_pelvis_about_x + q ; obs roll positive tilts toward -y
    if W is R:
        qx_des = beta - pr
    else:
        qx_des = -beta + pr
    tau[W["y"]] = P["kp_sw"] * (qy_des - q[W["y"]]) - P["kd_sw"] * qd[W["y"]]
    tau[W["x"]] = P["kp_swx"] * (qx_des - q[W["x"]]) - P["kd_swx"] * qd[W["x"]]
    tau[W["z"]] = 50.0 * (0.0 - q[W["z"]]) - 1.0 * qd[W["z"]]
    knee_t = P["knee_swing"] if t - P["t_dsw"] < P["fold"] * P["T_swing"] else P["knee_land"]
    tau[W["k"]] = 300.0 * (knee_t - q[W["k"]]) - 3.0 * qd[W["k"]]

    lim = 0.4 * gear
    for k in ("x", "y", "z"):                        # the stance hip compensates the torque actually applied
        tau[W[k]] = np.clip(tau[W[k]], -lim[W[k]], lim[W[k]])

    # stance knee
    tau[S["k"]] = P["kp_k"] * (P["knee_stance"] - q[S["k"]]) - P["kd_k"] * qd[S["k"]]
    tau[S["k"]] = min(tau[S["k"]], P["knee_push"] * gear[S["k"]])   # no violent extension (no hop)

    # torso via stance hip (torque applied on the pelvis is minus the hip torque for +axis joints)
    e_p = pitch - P["pitch_des"]
    tau[S["y"]] = P["kp_t"] * e_p + P["kd_t"] * wy - P["comp"] * tau[W["y"]]
    tau[S["y"]] = np.clip(tau[S["y"]], -P["hip_lim"], P["hip_lim"])   # do not buckle the stance knee
    t_roll = -(P["kp_r"] * roll + P["kd_r"] * wx)   # desired torque on pelvis about +x
    t_yaw = -(P["kp_y"] * yaw + P["kd_y"] * wz)
    if S is R:
        tau[R["x"]] = tau[L["x"]] - t_roll
        tau[R["z"]] = 50.0 * (0.0 - q[R["z"]]) - 1.0 * qd[R["z"]] - t_yaw
    else:
        tau[L["x"]] = t_roll + tau[R["x"]]
        tau[L["z"]] = 50.0 * (0.0 - q[L["z"]]) - 1.0 * qd[L["z"]] + t_yaw

    st_contact = obs["foot_contact"] if S is R else obs["foot_left_contact"]
    if not st_contact and P["air"] > 0:              # stance foot airborne: hold the leg under the hip
        tau[S["y"]] = P["air"] * (-pitch - q[S["y"]]) - 5.0 * qd[S["y"]]
        tau[S["x"]] = 0.5 * P["air"] * (0.0 - q[S["x"]]) - 3.0 * qd[S["x"]]
    if t < P["t_dsw"] and obs["time"] >= P["t_ds"]:  # double support after touchdown: share torso control
        t_pitch = P["kp_t"] * e_p + P["kd_t"] * wy
        for J in (R, L):
            tau[J["y"]] = 0.5 * t_pitch
            tau[J["k"]] = P["kp_k"] * (P["knee_stance"] - q[J["k"]]) - P["kd_k"] * qd[J["k"]]
        tau[R["x"]] = -0.5 * t_roll
        tau[L["x"]] = 0.5 * t_roll
        tau[R["z"]] = 50.0 * (0.0 - q[R["z"]]) - 1.0 * qd[R["z"]] - 0.5 * t_yaw
        tau[L["z"]] = 50.0 * (0.0 - q[L["z"]]) - 1.0 * qd[L["z"]] + 0.5 * t_yaw
    elif obs["foot_contact"] and obs["foot_left_contact"]:   # double support: yaw through both hips
        tau[R["z"]] = 50.0 * (0.0 - q[R["z"]]) - 1.0 * qd[R["z"]] - 0.5 * t_yaw
        tau[L["z"]] = 50.0 * (0.0 - q[L["z"]]) - 1.0 * qd[L["z"]] + 0.5 * t_yaw
    if obs["time"] < P["t_ds"]:                       # initial landing: both legs straight-ish
        for J in (R, L):
            tau[J["y"]] = 300.0 * (-pitch - q[J["y"]]) - 5.0 * qd[J["y"]] + 0.5 * (P["kp_t"] * e_p + P["kd_t"] * wy)
            tau[J["k"]] = P["kp_k"] * (P["knee_stance"] - q[J["k"]]) - P["kd_k"] * qd[J["k"]]
            tau[J["z"]] = 50.0 * (0.0 - q[J["z"]]) - 1.0 * qd[J["z"]]
        tau[R["x"]] = 200.0 * (0.0 - q[R["x"]]) - 5.0 * qd[R["x"]] - 0.5 * t_roll
        tau[L["x"]] = 200.0 * (0.0 - q[L["x"]]) - 5.0 * qd[L["x"]] + 0.5 * t_roll
    if DEBUG:
        print(obs["step"], st, round(t, 3), "h=%.2f p=%.2f r=%.2f y=%.2f v=%.2f,%.2f" % (obs["height"], pitch, roll, yaw, v[0], v[1]),
              "xt=%.2f yt=%.2f" % (xt, yt), "c=%d%d" % (obs["foot_contact"], obs["foot_left_contact"]),
              "q=", np.round(q[3:11], 2), "u=", np.round(tau[3:11] / gear[3:11], 2))
    return np.clip(tau / gear, -0.4, 0.4)
