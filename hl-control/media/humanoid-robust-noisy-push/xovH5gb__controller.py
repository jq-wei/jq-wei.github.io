"""SIMBICON-style stepping controller for the MuJoCo Humanoid (point feet: it has to keep stepping).

Control law, every 15 ms (joint torques -> ctrl = torque / gear, clipped to [-0.4, 0.4]):
  * Two-state machine (left stance / right stance).  The swing ends at touchdown of the swing foot
    (after it has clearly lifted and at least T_MIN elapsed); if no touchdown by T_STEP the foot is
    lowered until contact (forced switch at 1.6 T_STEP).
  * Swing foot: a trajectory from its lift-off position to a capture-point target relative to the mass
    centre (heading frame): x = x_off + k_cp*vx + k_v*(vx - v_des) + side*k_turn*yaw,
    y = side*y_half + k_cpy*vy, with ground clearance h_clear*sin(pi*s).  The target is converted to
    hip_x / hip_y / knee angles by 3-D leg inverse kinematics and tracked with joint PD.
    The side*k_turn*yaw term (k_turn < 0) lands the left foot less far ahead when the body has yawed
    left: the touchdown braking force at a foot beside the mass centre is the dominant yaw moment on this
    low-yaw-inertia body (a point foot transmits no torque), so this steers the heading back to +x.
  * Stance leg: knee held as a stiff strut; stance hip_y / hip_x torques are world-frame PD controllers
    on torso pitch / roll minus the swing-hip torques (so the pelvis receives exactly the balance
    torque); hip_z joints are held at zero (a point foot transmits no yaw torque anyway).
  * Abdomen: PD to zero.  Arms: soft PD to a neutral pose.
PARAMS are the tunable gains (searched with hl_control.harbor.tune); GAINS are fixed.
kp_*/kd_*: joint or torso PD gains [N m/rad, N m s/rad]; k_cp, k_cpy, k_v: capture-point/velocity gains
[s]; x_off, y_half: foot placement offsets [m]; v_des: desired speed [m/s]; pitch_des: torso lean [rad].
"""
import numpy as np

PARAMS = {              # tunable (<= 12 numeric entries)
    "T_STEP": 0.1507,       # s, nominal swing duration
    "v_des": 0.4669,         # m/s desired forward speed
    "k_cp": 0.4494,         # s, sagittal capture-point gain
    "x_off": -0.02454,       # m, sagittal foot offset (negative = behind the capture point)
    "k_v": 0.01246,          # s, extra gain on the forward-speed error
    "k_cpy": 0.8686,        # s, lateral capture-point gain
    "y_half": 0.2357,       # m, lateral half stance width
    "pitch_des": 0.04245,    # rad, desired torso pitch (forward lean)
    "kp_torso": 256.2,    # N m/rad torso pitch PD via stance hip_y
    "kp_roll": 20,     # N m/rad torso roll PD via stance hip_x
    "kp_hip": 286.5,      # N m/rad swing hip PD
    "k_turn": -0.244,      # m/rad fore/aft placement asymmetry steering the yaw to 0 (negative: see docstring)
}
PARAM_BOUNDS = {
    "T_STEP": (0.14, 0.26), "v_des": (0.3, 1.0), "k_cp": (0.3, 0.6), "x_off": (-0.05, 0.12), "k_v": (0.0, 0.3),
    "k_cpy": (0.4, 1.0), "y_half": (0.14, 0.30), "pitch_des": (-0.1, 0.2),
    "kp_torso": (180, 400), "kp_roll": (20, 150), "kp_hip": (150, 350), "k_turn": (-0.5, 0.0),
}
GAINS = {               # fixed
    "T_MIN": 0.10,        # s, earliest touchdown accepted
    "h_clear": 0.10,      # m, swing foot ground clearance
    "lift_thresh": 0.03,  # m, the swing foot counts as lifted above this height over the stance foot
    "kd_torso": 25.0, "kd_roll": 12.0, "kd_hip": 7.0,
    "kp_knee": 200.0, "kd_knee": 5.0, "knee_stance": 0.0,   # stance knee pressed against its -2 deg limit (a locked strut)
    "kp_hz": 40.0, "kd_hz": 5.0,            # hip_z PD (keeps the legs aligned with the pelvis)
    "kp_abd": 150.0, "kd_abd": 10.0,
    "lead": 0.0,          # s, state prediction horizon (compensates one control step of command delay)
    "k_abd_roll": 0.0, "k_abd_pitch": 0.0,  # abdomen assists the torso roll / pitch control (N m/rad)
    "kp_arm": 8.0, "kd_arm": 0.8,
}

# motor indices
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE = 3, 4, 5, 6
L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE = 7, 8, 9, 10
ARMS = slice(11, 17)
L_THIGH, L_SHIN = 0.403, 0.35     # hip->knee, knee->foot centre [m]
HIP_OFF = np.array([-0.015, 0.1, -0.011])   # hip position relative to the mass centre (torso frame), left; right has -y
CTRL_LIMIT = 0.4

STATE = {"stance": None, "t0": 0.0, "last_t": -1.0, "lifted": False, "p0": None, "debug": False}


def reset():
    STATE.update(stance=None, t0=0.0, last_t=-1.0, lifted=False, p0=None)


def _rot(roll, pitch, yaw=0.0):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def leg_ik(f, side):
    """hip_x, hip_y, knee angles that put the foot centre at `f` (pelvis frame, relative to the hip)."""
    D = float(np.clip(np.linalg.norm(f), 0.30, L_THIGH + L_SHIN - 0.005))
    c = np.clip((D * D - L_THIGH ** 2 - L_SHIN ** 2) / (2 * L_THIGH * L_SHIN), -1.0, 1.0)
    phi = np.arccos(c)                                   # knee flexion (0 = straight)
    beta = np.arctan2(L_SHIN * np.sin(phi), L_THIGH + L_SHIN * np.cos(phi))
    u = f / max(np.linalg.norm(f), 1e-6)
    cb, sb = np.cos(beta), np.sin(beta)                  # thigh = virtual leg rotated forward by beta (about pelvis y)
    d = np.array([cb * u[0] - sb * u[2], u[1], sb * u[0] + cb * u[2]])
    hip_y = -np.arcsin(np.clip(d[0], -1.0, 1.0))
    # right leg: hip_x axis +x, dy = sin(hx) cos(hy); left leg: axis -x, dy = -sin(hx) cos(hy)
    hip_x = np.arctan2(d[1], -d[2]) if side < 0 else np.arctan2(-d[1], -d[2])
    return hip_x, hip_y, -phi


def controller(obs):
    P = {**GAINS, **PARAMS}
    t = float(obs["time"])
    if t < STATE["last_t"] or t < 1e-9:
        reset()
    STATE["last_t"] = t
    qd, gear = obs["joint_rates"], obs["gear"]
    q = obs["joint_angles"] + P["lead"] * qd                  # predicted joint angles (delay compensation)
    yaw = float(obs["yaw"])
    Rh = _rot(0.0, 0.0, yaw)                     # heading frame -> world
    v = Rh.T @ np.array([obs["com_velocity_xy"][0], obs["com_velocity_xy"][1], 0.0])
    vx, vy = v[0], v[1]
    w = Rh.T @ obs["root_angular_velocity"]
    wx, wy = w[0], w[1]
    roll, pitch = float(obs["roll"]) + P["lead"] * wx, float(obs["pitch"]) + P["lead"] * wy
    cR, cL = bool(obs["foot_contact"]), bool(obs["foot_left_contact"])
    fR, fL = Rh.T @ obs["foot_xyz_rel"], Rh.T @ obs["foot_left_xyz_rel"]   # feet rel. COM, heading frame

    tau = np.zeros(17)
    for j in (ABD_Y, ABD_Z, ABD_X):
        tau[j] = -P["kp_abd"] * q[j] - P["kd_abd"] * qd[j]
    # positive abdomen_x torque rotates the torso about -x relative to the pelvis (reduces roll); abdomen_y likewise for pitch
    tau[ABD_X] += P["k_abd_roll"] * (roll + 0.1 * wx)
    tau[ABD_Y] += P["k_abd_pitch"] * ((pitch - P["pitch_des"]) + 0.1 * wy)
    tau[ARMS] = -P["kp_arm"] * q[ARMS] - P["kd_arm"] * qd[ARMS]

    # ---- state machine ----
    if STATE["stance"] is None:
        if cR or cL:
            STATE.update(stance="L", t0=t, lifted=False, p0=fR.copy())
        else:   # the initial drop: hold a standing pose
            for hy, kn, hx, hz in ((R_HIP_Y, R_KNEE, R_HIP_X, R_HIP_Z), (L_HIP_Y, L_KNEE, L_HIP_X, L_HIP_Z)):
                tau[hy] = -P["kp_hip"] * q[hy] - P["kd_hip"] * qd[hy]
                tau[kn] = P["kp_knee"] * (P["knee_stance"] - q[kn]) - P["kd_knee"] * qd[kn]
                tau[hx] = -P["kp_hip"] * q[hx] - P["kd_hip"] * qd[hx]
                tau[hz] = -P["kp_hz"] * q[hz] - P["kd_hz"] * qd[hz]
            return np.clip(tau / gear, -CTRL_LIMIT, CTRL_LIMIT)
    stance = STATE["stance"]
    f_sw, f_st = (fR, fL) if stance == "L" else (fL, fR)
    c_sw = cR if stance == "L" else cL
    ts = t - STATE["t0"]
    if f_sw[2] > f_st[2] + P["lift_thresh"]:
        STATE["lifted"] = True
    if (ts >= P["T_MIN"] and c_sw and STATE["lifted"]) or ts >= 1.6 * P["T_STEP"]:
        if STATE["debug"]:
            print(f"  TD t={t:.2f} ts={ts:.2f} sw={'R' if stance == 'L' else 'L'} contact={c_sw} foot=({f_sw[0]:+.3f},{f_sw[1]:+.3f},{f_sw[2]:+.3f}) "
                  f"target=({STATE.get('x_des', 0):+.3f},{STATE.get('y_des', 0):+.3f}) v=({vx:+.2f},{vy:+.2f}) roll={roll:+.2f} pitch={pitch:+.2f} yaw={yaw:+.2f}")
        stance = "R" if stance == "L" else "L"
        f_sw, f_st = f_st, f_sw
        STATE.update(stance=stance, t0=t, lifted=False, p0=f_sw.copy())
        ts = 0.0
    if stance == "L":
        st = dict(hy=L_HIP_Y, kn=L_KNEE, hx=L_HIP_X, hz=L_HIP_Z, side=+1)
        sw = dict(hy=R_HIP_Y, kn=R_KNEE, hx=R_HIP_X, hz=R_HIP_Z, side=-1)
    else:
        st = dict(hy=R_HIP_Y, kn=R_KNEE, hx=R_HIP_X, hz=R_HIP_Z, side=-1)
        sw = dict(hy=L_HIP_Y, kn=L_KNEE, hx=L_HIP_X, hz=L_HIP_Z, side=+1)

    # ---- swing foot trajectory (heading frame, relative to the COM) ----
    s = ts / P["T_STEP"]
    ex = vx - P["v_des"]
    x_des = P["x_off"] + P["k_cp"] * vx + P["k_v"] * ex + sw["side"] * P["k_turn"] * yaw
    y_des = sw["side"] * P["y_half"] + P["k_cpy"] * vy
    STATE["x_des"], STATE["y_des"] = x_des, y_des
    p0 = STATE["p0"]
    blend = 0.5 * (1.0 - np.cos(np.pi * min(s, 1.0)))
    z_ground = f_st[2]
    if s <= 1.0:
        z = z_ground + P["h_clear"] * np.sin(np.pi * s)
    else:
        z = z_ground - 0.15 * (s - 1.0)              # no touchdown yet: lower the foot
    target = np.array([p0[0] + (x_des - p0[0]) * blend, p0[1] + (y_des - p0[1]) * blend, z])
    # convert to the pelvis frame relative to the swing hip
    Rt = _rot(roll, pitch)                            # torso in the heading frame
    Rp = Rt @ _rot(q[ABD_X], q[ABD_Y])                # pelvis (abdomen bend)
    hip = Rt @ (HIP_OFF * np.array([1.0, sw["side"], 1.0]))
    f_pel = Rp.T @ (target - hip)
    hx_d, hy_d, kn_d = leg_ik(f_pel, sw["side"])
    tau[sw["hy"]] = P["kp_hip"] * (hy_d - q[sw["hy"]]) - P["kd_hip"] * qd[sw["hy"]]
    tau[sw["kn"]] = P["kp_knee"] * (kn_d - q[sw["kn"]]) - P["kd_knee"] * qd[sw["kn"]]
    tau[sw["hx"]] = 0.6 * P["kp_hip"] * (hx_d - q[sw["hx"]]) - P["kd_hip"] * qd[sw["hx"]]
    tau[sw["hz"]] = -P["kp_hz"] * q[sw["hz"]] - P["kd_hz"] * qd[sw["hz"]]

    # ---- stance leg ----
    tau[st["kn"]] = P["kp_knee"] * (P["knee_stance"] - q[st["kn"]]) - P["kd_knee"] * qd[st["kn"]]
    # positive stance hip_y torque (foot fixed) pitches the pelvis backward, i.e. reduces pitch
    tau_torso = P["kp_torso"] * (pitch - P["pitch_des"]) + P["kd_torso"] * wy
    tau[st["hy"]] = tau_torso - tau[sw["hy"]]
    # right hip_x (+x axis): positive torque reduces roll; left hip_x (-x axis): increases it
    sgn = 1.0 if st["side"] < 0 else -1.0
    tau_roll = P["kp_roll"] * roll + P["kd_roll"] * wx
    tau[st["hx"]] = sgn * tau_roll - tau[sw["hx"]]
    tau[st["hz"]] = -P["kp_hz"] * q[st["hz"]] - P["kd_hz"] * qd[st["hz"]]
    return np.clip(tau / gear, -CTRL_LIMIT, CTRL_LIMIT)
