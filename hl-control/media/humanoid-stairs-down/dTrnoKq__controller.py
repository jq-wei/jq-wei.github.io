"""Foot-placement stepping controller for the MuJoCo Humanoid (point feet, no ankles) on the 3 cm down-staircase.

Control law
-----------
A three-phase state machine per step: DS (short double support after touchdown), LIFT (the swing
leg shortens), LAND (the swing leg lengthens until its foot touches down; then the legs swap roles).
Every joint is driven by a PD servo (torque = kp * error - kd * rate, divided by the gear, clipped to +-0.4):

* swing leg, task space: the foot is steered to a touchdown point relative to the mass centre,
  x = kv_f * v_forward - x0 (capture-point law; the offset x0 > 0 leaves the body falling forward and so
  sets the walking speed) and y = kv_l * v_left +- w0 (step width), in the heading frame, with x0 pointing
  along the world x axis (a yawed body is steered back). Two-link inverse kinematics turns
  (foot offset, leg length) into hip-pitch, hip-roll and knee targets; the leg angles are rate limited
  (sw_rate) and the mass-centre velocity is low-pass filtered (v_filt);
* stance hip torque is a virtual torso servo: it holds the torso at pitch_ref / zero roll and cancels
  a fraction c_sw of the swing hip's reaction (tau_stance = -tau_torso - c_sw * tau_swing);
* stance knee is held bent (knee_stance) so the swing leg has room to reach the ground; a step down is
  felt through the legs: if the swing foot is already drop_eps below the stance foot's level without
  touching anything, or finds no ground t_land after LAND began, the stance knee bends further at
  `sink` rad/s and lowers the body until the foot lands;
* yaw (a point foot cannot resist it): the feet on the ground get opposite fore-aft hip torques
  proportional to yaw and yaw rate (k_yaw, d_yaw, limit yaw_max; scaled by g_ss in single support), and
  the arms are held spread out to raise the yaw inertia;
* abdomen and hip yaw are held at zero.

Speed schedule (uses the observed world x of the mass centre and the task's stated layout: flat start,
first riser 1 m ahead): the offset is x0_fast on the flat start (x < x_fast_until), where a brisker walk
is safe and buys distance margin for the 3 m rung, and the slower, sturdier x0 from there on.

Gains (PARAMS): t_ds / t_lift / t_min / t_land / t_ext / t_max phase timing [s]; kv_f, kv_l [s] and x0, w0 [m]
foot placement; lift [m] swing-leg shortening; pitch_ref torso lean [rad]; kp_* / kd_* servo gains
[N m/rad, N m s/rad]; knee_stance / knee_ext knee targets [rad]; drop_eps [m]. The values come from a random local
search over these gains on the simulator (flat floor, then the staircase, then the staircase with
random pushes); no learning, no gradients.
"""

import numpy as np

PARAMS = {
    "t_ds": 0.0441, "t_lift": 0.063, "t_min": 0.0256, "t_land": 0.1, "t_ext": 0.0739, "t_max": 0.6,
    "kv_f": 0.2919, "x0": 0.0249, "kv_l": 0.4096, "w0": 0.1408,
    "x0_fast": 0.035, "x_fast_until": 0.6,
    "lift": 0.0631, "pitch_ref": 0.0144, "v_filt": 0.2752, "sw_rate": 5.8057,
    "kp_torso": 310.4627, "kd_torso": 30.3327, "kp_roll": 225.9451, "kd_roll": 12.2536,
    "kp_hip": 590.7414, "kd_hip": 14.6131, "kp_hipx": 265.6648, "kd_hipx": 6.1794,
    "kp_knee": 233.5911, "kd_knee": 3.0081, "kp_knee_st": 668.2657, "kd_knee_st": 10.0099,
    "knee_stance": -0.4324, "knee_ext": -0.0975, "sink": 3.9051, "drop_eps": 0.01,
    "kp_abd": 114.1178, "kd_abd": 4.5824, "kp_hipz": 29.3507, "kd_hipz": 1.2719,
    "k_yaw": 73.3337, "d_yaw": 24.9982, "g_ss": 1.0294, "yaw_max": 60.0, "c_sw": 0.45,
}

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], dtype=float)
ABY, ABZ, ABX = 0, 1, 2
HX, HZ, HY, KN = (3, 7), (4, 8), (5, 9), (6, 10)   # index 0 = right leg, 1 = left leg
ARM_POSE = np.array([-0.96, -0.70, -1.57, 0.96, 0.70, -1.57])   # arms spread sideways (yaw inertia)
LA, LB = 0.383, 0.37   # hip-knee and knee-foot lengths [m]

_S = {"stance": 0, "t0": 0.0, "started": False, "phase": 0, "kn_st": -0.4, "vx": 0.0, "vy": 0.0, "gam": 0.0, "alp": 0.0}


def _leg_len(knee):
    """Hip-foot distance for a knee angle (knee <= 0)."""
    return np.sqrt(LA * LA + LB * LB + 2.0 * LA * LB * np.cos(knee))


def _knee_off(knee):
    """Angle between the hip-foot line and the thigh for a knee angle (knee <= 0)."""
    return np.arctan2(LB * np.sin(-knee), LA + LB * np.cos(knee))


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    S = _S
    if obs["step"] == 0:
        S.update(stance=0, t0=0.0, started=False, phase=0, kn_st=P["knee_stance"], vx=0.0, vy=0.0, gam=0.0, alp=0.0)
    t = float(obs["time"])
    dt = float(obs["dt"])
    q = np.asarray(obs["joint_angles"], dtype=float)
    qd = np.asarray(obs["joint_rates"], dtype=float)
    roll, pitch, yaw = float(obs["roll"]), float(obs["pitch"]), float(obs["yaw"])
    c, s = np.cos(yaw), np.sin(yaw)
    w = obs["root_angular_velocity"]
    w_roll = c * w[0] + s * w[1]
    w_pitch = -s * w[0] + c * w[1]
    w_yaw = float(w[2])
    v = obs["com_velocity_xy"]
    S["vx"] += P["v_filt"] * (float(v[0]) - S["vx"])   # low-pass: the swing leg's own momentum makes the raw value spiky
    S["vy"] += P["v_filt"] * (float(v[1]) - S["vy"])
    v_f = c * S["vx"] + s * S["vy"]
    v_l = -s * S["vx"] + c * S["vy"]
    contact = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))
    foot = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])
    com_x = float(obs["com_xyz"][0])

    # ---- state machine ----
    if not S["started"]:
        if contact[0] or contact[1] or t > 0.3:   # the reset drops the robot 10 cm onto its feet
            S.update(started=True, t0=t, phase=0, stance=0)
    else:
        tp = t - S["t0"]
        sw = 1 - S["stance"]
        if S["phase"] == 0 and tp >= P["t_ds"]:
            S["phase"] = 1
            S["gam"] = -(pitch + q[ABY] + q[HY[sw]]) - _knee_off(q[KN[sw]])
            S["alp"] = roll + q[ABX] + (q[HX[sw]] if sw == 0 else -q[HX[sw]])
        elif S["phase"] == 1 and tp >= P["t_ds"] + P["t_lift"]:
            S["phase"] = 2
        elif S["phase"] == 2 and ((contact[sw] and tp >= P["t_ds"] + P["t_lift"] + P["t_min"]) or tp >= P["t_max"]):
            S.update(stance=sw, t0=t, phase=0, kn_st=P["knee_stance"])
    st = S["stance"]
    sw = 1 - st
    phase = S["phase"]
    tp = t - S["t0"]

    tau = np.zeros(17)
    for j in (ABY, ABZ, ABX):
        tau[j] = -P["kp_abd"] * q[j] - P["kd_abd"] * qd[j]
    tau[11:17] = 8.0 * (ARM_POSE - q[11:17]) - 0.4 * qd[11:17]
    for leg in (0, 1):
        tau[HZ[leg]] = -P["kp_hipz"] * q[HZ[leg]] - P["kd_hipz"] * qd[HZ[leg]]

    tau_torso = P["kp_torso"] * (P["pitch_ref"] - pitch) - P["kd_torso"] * w_pitch
    tau_roll = P["kp_roll"] * (0.0 - roll) - P["kd_roll"] * w_roll

    if (not S["started"]) or phase == 0:
        # both legs act as stance legs and share the torso servo
        for leg in (0, 1):
            tau[HY[leg]] = -0.5 * tau_torso
            tau[KN[leg]] = P["kp_knee_st"] * (P["knee_stance"] - q[KN[leg]]) - P["kd_knee_st"] * qd[KN[leg]]
        tau[HX[0]] = -0.5 * tau_roll
        tau[HX[1]] = 0.5 * tau_roll
    else:
        # ---- speed schedule: forward offset of the touchdown point ----
        x0 = P["x0_fast"] if com_x < P["x_fast_until"] else P["x0"]

        # ---- swing leg: foot target relative to the mass centre (heading frame) ----
        side = 1.0 if sw == 1 else -1.0               # the left foot lands to the left (+y)
        x_des = P["kv_f"] * v_f - x0 * c
        y_des = P["kv_l"] * v_l + side * P["w0"] + x0 * s
        fx = c * foot[sw][0] + s * foot[sw][1]
        fy = -s * foot[sw][0] + c * foot[sw][1]
        # current swing leg geometry (hip-foot line: length r, pitch gam (forward +), roll alp (left +))
        kn = q[KN[sw]]
        r_now = _leg_len(kn)
        gam_now = -(pitch + q[ABY] + q[HY[sw]]) - _knee_off(kn)
        alp_now = roll + q[ABX] + (q[HX[sw]] if sw == 0 else -q[HX[sw]])
        hip_x = fx - r_now * np.sin(gam_now)          # swing hip relative to the mass centre
        hip_y = fy - r_now * np.sin(alp_now)
        # leg length schedule: short in LIFT, ramped to full extension in LAND
        r_st = _leg_len(P["knee_stance"])
        r_ext = _leg_len(P["knee_ext"])
        if phase == 1:
            r_des = r_st - P["lift"]
        else:
            u = min(1.0, (tp - P["t_ds"] - P["t_lift"]) / P["t_ext"])
            r_des = (r_st - P["lift"]) + u * (r_ext - r_st + P["lift"])
        gam_t = float(np.arcsin(np.clip((x_des - hip_x) / r_des, -0.45, 0.60)))
        alp_t = float(np.arcsin(np.clip((y_des - hip_y) / r_des, -0.45, 0.45)))
        dmax = P["sw_rate"] * dt
        S["gam"] += float(np.clip(gam_t - S["gam"], -dmax, dmax))   # rate-limited swing: a gentle leg throw
        S["alp"] += float(np.clip(alp_t - S["alp"], -dmax, dmax))
        gam, alp = S["gam"], S["alp"]
        kn_tgt = -float(np.arccos(np.clip((r_des * r_des - LA * LA - LB * LB) / (2.0 * LA * LB), -1.0, 1.0)))
        hy_tgt = -(gam + _knee_off(kn_tgt)) - pitch - q[ABY]
        tau_sw_y = float(np.clip(P["kp_hip"] * (hy_tgt - q[HY[sw]]) - P["kd_hip"] * qd[HY[sw]], -120.0, 120.0))
        tau[HY[sw]] = tau_sw_y
        hx_world = alp - roll - q[ABX]
        hx_tgt = float(np.clip(hx_world if sw == 0 else -hx_world, -0.42, 0.07))
        tau_sw_x = float(np.clip(P["kp_hipx"] * (hx_tgt - q[HX[sw]]) - P["kd_hipx"] * qd[HX[sw]], -40.0, 40.0))
        tau[HX[sw]] = tau_sw_x
        tau[KN[sw]] = P["kp_knee"] * (kn_tgt - q[KN[sw]]) - P["kd_knee"] * qd[KN[sw]]

        # ---- stance leg ----
        late = tp > P["t_ds"] + P["t_lift"] + P["t_land"]
        below = float(foot[sw][2]) < float(foot[st][2]) - P["drop_eps"]   # swing foot already under the stance foot's level
        if phase == 2 and (late or below) and not contact[sw]:
            S["kn_st"] = max(S["kn_st"] - P["sink"] * dt, -1.0)   # no ground yet (a step down): lower the body
        tau[HY[st]] = -tau_torso - P["c_sw"] * tau_sw_y
        tau[HX[st]] = (tau_sw_x - tau_roll) if st == 0 else (tau_roll + tau_sw_x)
        tau[KN[st]] = P["kp_knee_st"] * (S["kn_st"] - q[KN[st]]) - P["kd_knee_st"] * qd[KN[st]]

    # ---- yaw: opposite fore-aft hip torques on the feet that are on the ground ----
    g = 1.0 if (contact[0] and contact[1]) else P["g_ss"]
    dty = g * float(np.clip(P["k_yaw"] * yaw + P["d_yaw"] * w_yaw, -P["yaw_max"], P["yaw_max"]))
    if contact[0]:
        tau[HY[0]] -= dty
    if contact[1]:
        tau[HY[1]] += dty

    return np.clip(tau / GEAR, -0.4, 0.4)
