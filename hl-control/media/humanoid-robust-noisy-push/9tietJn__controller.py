"""Hand-designed stepping controller for the MuJoCo Humanoid (no learning).

Structure (66.7 Hz, joint torque = gear * command):
  * Gait: after DOUBLE_T s of double support the legs alternate single-support steps.  A
    step ends when the swing foot touches the floor (after T_MIN*T_STEP) or at T_MAX*T_STEP.
  * Swing leg: the foot target relative to the hip, in a yaw-aligned frame, follows the
    Raibert rule  x_f = v*T_STEP/2 + KX*(v - V_DES),  y_f = v_y*T_STEP/2 + KY*v_y (+/- W),
    with a yaw-rate term KYAW*yaw_rate that brakes body spin, and the leg is shortened by
    LIFT mid-swing.  Leg inverse kinematics gives hip/knee targets tracked by PD servos
    (HIP_*, HIPX_*, KNEE_*), ramped in over RAMP of the swing so the foot leaves the floor gently.
  * Stance leg: hip pitch/roll torques servo the pelvis attitude (PITCH_*, ROLL_*, set-point
    PITCH_DES forward lean) and cancel COMP of the swing-hip reaction torque; the knee is a PD
    spring holding leg length L0; hip yaw servos to zero plus a yaw-rate damping term YAW_KD.
  * Abdomen motors keep the torso upright in the world (ABD_*), with a weak joint-space PD
    (ABDJ_*); arms hold a neutral pose (ARM_*).
  * Latency: servos act on the state predicted PRED control steps ahead.  A one-step command
    delay is detected from obs["last_action"] and then the stiff servos are softened and the
    step slowed (DELAY_*).  Actual actuator gears (obs["gear"]) are used, so weaker motors are
    compensated up to the command limit.
  * Derivative gains are small on purpose: at 15 ms steps, kd*dt must stay well below the
    reduced inertia of the light links or the loops chatter.
PARAMS were searched with hl_control.harbor.tune on the nominal plant (seeds 0-7), then KX and
V_DES were lowered by hand for robustness across the perturbation scenarios.
Internal state (phase clock, support leg, delay flag) resets whenever obs["step"] == 0.
"""

import numpy as np

PARAMS = dict(   # tunable gains (searched by hl_control.harbor.tune on the nominal plant)
    T_STEP=0.1458,     # swing duration per step [s]
    L0=0.7266,         # stance leg length, hip to foot-sphere centre [m] (straight = 0.753)
    LIFT=0.1455,       # mid-swing leg shortening for floor clearance [m]
    V_DES=0.25,        # desired walking speed along the current heading [m/s]
    KX=0.2,            # Raibert velocity gain, sagittal foot placement [s]
    KY=0.4015,         # Raibert velocity gain, lateral foot placement [s]
    W=0.0393,          # lateral foot offset per side, + = wider stance [m]
    PITCH_DES=0.0984,  # pelvis pitch set-point, forward lean [rad]
    PITCH_KP=169.299,  # stance hip pitch attitude gain [N m/rad]
    ROLL_KP=111.8705,  # stance hip roll attitude gain [N m/rad]
    HIP_KP=452.6953,   # swing hip pitch servo stiffness [N m/rad]
    KYAW=0.0793,       # yaw-rate term in sagittal foot placement [m s/rad]
)
PARAM_BOUNDS = {'T_STEP': (0.12, 0.3), 'L0': (0.69, 0.75), 'LIFT': (0.06, 0.16), 'V_DES': (0.1, 0.6), 'KX': (0.05, 0.4), 'KY': (0.1, 0.5),
                'W': (-0.07, 0.05), 'PITCH_DES': (-0.05, 0.2), 'PITCH_KP': (60, 350), 'ROLL_KP': (20, 250), 'HIP_KP': (150, 500), 'KYAW': (0.0, 0.12)}
FIXED = dict(   # structural constants and servo gains kept fixed
    DELAY_KNEE_SCALE=0.356, DELAY_HIP_SCALE=0.732, DELAY_ATT_SCALE=0.468, DELAY_PRED=1.397, DELAY_KD_SCALE=0.258, DELAY_T_SCALE=1.132, DELAY_V_SCALE=0.428, DELAY_KYAW_SCALE=1.0,   # applied when a command delay is detected (tuned on noisy_delay_v1)
    WEAK_T_SCALE=1.056, WEAK_V_SCALE=1.023, WEAK_HIP_SCALE=1.018, WEAK_KNEE_SCALE=0.98, WEAK_ATT_SCALE=1.206, WEAK_LIFT_SCALE=0.92, WEAK_KX_SCALE=0.888,   # applied when motors are weaker than nominal (tuned on weak_motors_20)
    DOUBLE_T=0.35,     # initial double support after the drop [s]
    VY_THR=0.25, SHORT=0.7,   # lateral speed threshold [m/s] and swing-time factor for the wrong-side leg
    DS_T=0.0,          # double support between steps [s]
    T_MIN=0.5, T_MAX=1.6,   # swing ends on swing-foot contact after T_MIN*T_STEP, or at T_MAX*T_STEP
    RAMP=0.2,          # fraction of the swing over which swing-leg torques ramp in
    PRED=1.0,          # control steps of state prediction (q + qd*dt*PRED) against the 15 ms latency
    FACE=1.0,          # 1: V_DES along the current heading; 0: along world x
    YAW_KP=0.0, YAW_KD=10.0,   # stance hip yaw torque on body yaw / yaw rate
    X_OFF=0.0,         # constant sagittal foot offset [m]
    LEAN=0.0,          # body roll toward the stance side [rad]
    PITCH_KD=8.0, ROLL_KD=8.0,       # stance hip attitude damping [N m s/rad]
    HIP_KD=8.0,                      # swing hip pitch servo damping
    HIPX_KP=150.0, HIPX_KD=5.0,      # hip roll servo (swing)
    HIPZ_KP=60.0, HIPZ_KD=2.0,       # hip yaw servo to zero
    KNEE_KP=300.0, KNEE_KD=3.0,      # knee servo (stance spring and swing)
    ABD_KP=150.0, ABD_KD=5.0,        # abdomen torque on torso world pitch/roll
    ABDJ_KP=30.0, ABDJ_KD=1.0,       # abdomen joint-space PD (drift prevention)
    ARM_KP=15.0, ARM_KD=0.5,         # arm joint PD to neutral
    COMP=1.0,          # fraction of the swing-hip torque reaction cancelled at the stance hip
)

L1, L2 = 0.383, 0.370          # thigh, shin (knee to foot-sphere centre) [m]
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
# motor indices
AB_Y, AB_Z, AB_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_KNEE = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_KNEE = 7, 8, 9, 10
ARMS = slice(11, 17)

_state = {"support": 1, "t_switch": 0.0, "mode": "swing"}   # support: 0 = right leg, 1 = left leg


def leg_ik(x, y, z, side, pitch, roll):
    """Joint targets (hip_x, hip_y, knee) putting the foot centre at (x, y, z) from the hip (yaw frame)."""
    r = float(np.clip(np.sqrt(x * x + y * y + z * z), 0.35, L1 + L2 - 0.005))
    phi = np.arccos(np.clip((L1 ** 2 + L2 ** 2 - r ** 2) / (2 * L1 * L2), -1, 1))   # interior knee angle
    knee = -(np.pi - phi)
    beta = np.arccos(np.clip((L1 ** 2 + r ** 2 - L2 ** 2) / (2 * L1 * r), -1, 1))
    alpha = np.arctan2(x, -z)
    hip_y = -(alpha + beta + pitch)
    if side == 0:   # right: adduction (+) moves foot to +y
        hip_x = np.arctan2(y, -z) - roll
    else:           # left: adduction (+) moves foot to -y
        hip_x = np.arctan2(-y, -z) + roll
    hip_x = float(np.clip(hip_x, np.radians(-25), np.radians(5)))
    hip_y = float(np.clip(hip_y, np.radians(-110), np.radians(20)))
    knee = float(np.clip(knee, np.radians(-160), np.radians(-2)))
    return hip_x, hip_y, knee


def pd(q, qd, q_des, kp, kd, qd_des=0.0):
    return kp * (q_des - q) + kd * (qd_des - qd)


def controller(obs: dict) -> np.ndarray:
    P = {**FIXED, **PARAMS}
    t = float(obs["time"])
    if int(obs["step"]) == 0:
        _state["support"] = 1
        _state["t_switch"] = 0.0
        _state["mode"] = "swing"
    # command-delay detection: with a delayed plant the applied action lags our previous command
    if int(obs["step"]) == 0:
        _state["delayed"] = False
        _state["prev_cmd"] = None
    elif _state.get("prev_cmd") is not None and np.max(np.abs(np.asarray(obs["last_action"]) - _state["prev_cmd"])) > 1e-9:
        _state["delayed"] = True
    gear = np.asarray(obs.get("gear", GEAR), float)          # actual gears (weak motors are compensated)
    if gear[5] < 0.95 * GEAR[5]:   # weaker motors than nominal: re-scaled gait parameters (WEAK_*)
        P = {**P, "T_STEP": P["T_STEP"] * P["WEAK_T_SCALE"], "V_DES": P["V_DES"] * P["WEAK_V_SCALE"],
             "HIP_KP": P["HIP_KP"] * P["WEAK_HIP_SCALE"], "KNEE_KP": P["KNEE_KP"] * P["WEAK_KNEE_SCALE"],
             "PITCH_KP": P["PITCH_KP"] * P["WEAK_ATT_SCALE"], "ROLL_KP": P["ROLL_KP"] * P["WEAK_ATT_SCALE"],
             "LIFT": P["LIFT"] * P["WEAK_LIFT_SCALE"], "KX": P["KX"] * P["WEAK_KX_SCALE"]}
    qd = np.asarray(obs["joint_rates"])
    if _state.get("delayed"):      # one control step of command delay: predict further, soften stiff servos
        P = {**P, "KNEE_KP": P["KNEE_KP"] * P["DELAY_KNEE_SCALE"], "HIP_KP": P["HIP_KP"] * P["DELAY_HIP_SCALE"],
             "HIPX_KP": P["HIPX_KP"] * P["DELAY_HIP_SCALE"], "PITCH_KP": P["PITCH_KP"] * P["DELAY_ATT_SCALE"],
             "ROLL_KP": P["ROLL_KP"] * P["DELAY_ATT_SCALE"], "ABD_KP": P["ABD_KP"] * P["DELAY_ATT_SCALE"],
             "KNEE_KD": P["KNEE_KD"] * P["DELAY_KD_SCALE"], "HIP_KD": P["HIP_KD"] * P["DELAY_KD_SCALE"],
             "T_STEP": P["T_STEP"] * P["DELAY_T_SCALE"], "V_DES": P["V_DES"] * P["DELAY_V_SCALE"],
             "KYAW": P["KYAW"] * P["DELAY_KYAW_SCALE"]}
    dtp = float(obs["dt"]) * (P["PRED"] + P["DELAY_PRED"] * (1.0 if _state.get("delayed") else 0.0))
    q = np.asarray(obs["joint_angles"]) + qd * dtp
    w = np.asarray(obs["root_angular_velocity"])
    yaw = float(obs["yaw"])
    c0, s0 = np.cos(yaw), np.sin(yaw)
    pitch = float(obs["pitch"]) + (-s0 * w[0] + c0 * w[1]) * dtp
    roll = float(obs["roll"]) + (c0 * w[0] + s0 * w[1]) * dtp
    # yaw-aligned horizontal velocity of the mass centre
    v_world = np.asarray(obs["com_velocity_xy"])
    c, s = np.cos(yaw), np.sin(yaw)
    vx = c * v_world[0] + s * v_world[1]
    vy = -s * v_world[0] + c * v_world[1]
    contacts = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))

    tau = np.zeros(17)
    # body-frame pitch/roll rates of the torso (world rates rotated by yaw)
    pitch_rate = -s * w[0] + c * w[1]
    roll_rate = c * w[0] + s * w[1]
    # pelvis attitude = torso attitude + abdomen joint angles (hips act on the pelvis)
    p_pitch, p_roll = pitch + q[AB_Y], roll + q[AB_X]
    p_pitch_rate, p_roll_rate = pitch_rate + qd[AB_Y], roll_rate + qd[AB_X]
    # abdomen: keep the torso upright in the world (reaction torque sign: + abdomen_y pitches the torso back)
    tau[AB_Y] = P["ABD_KP"] * (pitch - P["PITCH_DES"]) + P["ABD_KD"] * pitch_rate + pd(q[AB_Y], qd[AB_Y], 0.0, P["ABDJ_KP"], P["ABDJ_KD"])
    tau[AB_X] = P["ABD_KP"] * roll + P["ABD_KD"] * roll_rate + pd(q[AB_X], qd[AB_X], 0.0, P["ABDJ_KP"], P["ABDJ_KD"])
    tau[AB_Z] = pd(q[AB_Z], qd[AB_Z], 0.0, P["ABDJ_KP"], P["ABDJ_KD"])
    tau[ARMS] = pd(q[ARMS], qd[ARMS], 0.0, P["ARM_KP"], P["ARM_KD"])
    # stance hip attitude torques on the pelvis

    legs = {0: (R_HX, R_HZ, R_HY, R_KNEE), 1: (L_HX, L_HZ, L_HY, L_KNEE)}
    _, _, knee0 = leg_ik(0.0, 0.0, -P["L0"], 0, 0.0, 0.0)

    if t < P["DOUBLE_T"]:
        stance = (0, 1)
        swing = None
        phase = 0.0
    else:
        elapsed = t - _state["t_switch"]
        sw = 1 - _state["support"]
        # when falling sideways fast, shorten the swing of the leg on the wrong side so the other comes sooner
        wrong_side = (vy > P["VY_THR"] and sw == 0) or (vy < -P["VY_THR"] and sw == 1)
        T_eff = P["T_STEP"] * (P["SHORT"] if wrong_side else 1.0)
        if _state["mode"] == "swing":
            phase = elapsed / T_eff
            if phase >= P["T_MAX"] or (phase > P["T_MIN"] and contacts[sw]):
                _state["mode"] = "ds"
                _state["t_switch"] = t
                elapsed = 0.0
        if _state["mode"] == "ds" and elapsed >= P["DS_T"]:
            _state["mode"] = "swing"
            _state["support"] = sw
            _state["t_switch"] = t
            phase = 0.0
        if _state["mode"] == "ds":
            stance = (0, 1)
            swing = None
        else:
            stance = (_state["support"],)
            swing = 1 - _state["support"]
            phase = (t - _state["t_switch"]) / T_eff

    roll_des = 0.0 if swing is None else (P["LEAN"] if swing == 1 else -P["LEAN"])
    u_pitch = P["PITCH_KP"] * (p_pitch - P["PITCH_DES"]) + P["PITCH_KD"] * p_pitch_rate
    u_roll = P["ROLL_KP"] * (p_roll - roll_des) + P["ROLL_KD"] * p_roll_rate
    tau[AB_X] += P["ABD_KP"] * (-roll_des)     # torso leans with the pelvis
    swing_hy = swing_hx = 0.0
    if swing is not None:
        hx, hz, hy, kn = legs[swing]
        T = P["T_STEP"]
        if P["FACE"] > 0.5:
            vx_des, vy_des = P["V_DES"], 0.0                       # walk along the current heading
        else:
            vx_des, vy_des = P["V_DES"] * c, -P["V_DES"] * s          # desired velocity (world x) in the yaw frame
        side_sign = 1.0 if swing == 1 else -1.0
        xf = vx * T / 2 + P["KX"] * (vx - vx_des) + P["X_OFF"] - side_sign * P["KYAW"] * w[2]
        yf = vy * T / 2 + P["KY"] * (vy - vy_des) + side_sign * P["W"]
        lift = P["LIFT"] * np.sin(np.pi * phase) if phase < 1.0 else -0.03 * min(phase - 1.0, 1.0)
        L = P["L0"] - lift
        zf = -np.sqrt(max(L * L - xf * xf - yf * yf, 0.1))
        qz = q[hz] if swing == 0 else -q[hz]            # thigh yaw relative to pelvis (about +z)
        cz, sz = np.cos(qz), np.sin(qz)
        xt, yt = cz * xf + sz * yf, -sz * xf + cz * yf
        hx_d, hy_d, kn_d = leg_ik(xt, yt, zf, swing, p_pitch, p_roll)
        tau[hy] = pd(q[hy], qd[hy], hy_d, P["HIP_KP"], P["HIP_KD"])
        tau[hx] = pd(q[hx], qd[hx], hx_d, P["HIPX_KP"], P["HIPX_KD"])
        tau[hz] = pd(q[hz], qd[hz], 0.0, P["HIPZ_KP"], P["HIPZ_KD"])
        tau[kn] = pd(q[kn], qd[kn], kn_d, P["KNEE_KP"], P["KNEE_KD"])
        ramp = min(1.0, phase / max(P["RAMP"], 1e-6))
        for j in (hy, hx, kn):
            tau[j] *= ramp
        lim = 0.4 * gear
        swing_hy = float(np.clip(tau[hy], -lim[hy], lim[hy]))
        swing_hx = float(np.clip(tau[hx], -lim[hx], lim[hx])) * (1.0 if swing == 0 else -1.0)  # torque about +x on pelvis

    for leg in stance:
        hx, hz, hy, kn = legs[leg]
        tau[hy] = u_pitch - P["COMP"] * swing_hy
        tau[hx] = (u_roll - P["COMP"] * swing_hx) * (1.0 if leg == 0 else -1.0)
        u_yaw = P["YAW_KP"] * yaw + P["YAW_KD"] * w[2]
        tau[hz] = pd(q[hz], qd[hz], 0.0, P["HIPZ_KP"], P["HIPZ_KD"]) + (u_yaw if leg == 0 else -u_yaw)
        tau[kn] = pd(q[kn], qd[kn], knee0, P["KNEE_KP"], P["KNEE_KD"])

    cmd = np.clip(tau / gear, -0.4, 0.4)
    _state["prev_cmd"] = cmd.copy()
    return cmd
