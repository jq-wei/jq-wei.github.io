"""SIMBICON-style walking controller for the MuJoCo Humanoid (Gymnasium Humanoid-v5 plant).

Structure (Yin et al. 2007, adapted to point feet):
  * Two states, left-stance / right-stance.  A state ends when the swing foot touches the
    floor after at least ``t_min`` of the nominal step time ``T_step``, or when the step has
    lasted ``t_max`` times ``T_step``.
  * Swing leg: the thigh's world-frame angle is set by linear foot placement feedback
        theta_sw = th0 + c_d * d + c_v * v         (sagittal, d = COM - stance foot, v = COM speed)
        a_sw     = a0  + c_dy * d_out + c_vy * v_out  (lateral, "out" = toward the swing side)
    and converted to a hip joint target using the measured torso pitch/roll, so the
    placement is in world coordinates.  The swing knee bends by ``knee_bend`` in the first
    part of the step and straightens for touchdown.
  * Stance leg: the knee is held nearly straight (``knee_stance``); the stance hip is not
    position-controlled but applies  tau = kp_t*(pitch - pitch_des) + kd_t*pitch_rate
    - tau_swing_hip  so that the torso tracks the desired lean regardless of the swing leg
    (the SIMBICON torso rule).  The desired lean is pitch_des + k_lean*(v_des - v): leaning
    further forward when slower than v_des is the speed regulator.  Roll is handled the same way on the hip-x motors (kp_r, kd_r).
  * Abdomen: PD to zero plus the same torso feedback; hip yaw and arms: PD to a fixed pose.
  * Every motor command is a PD law in control units (gear-normalised), clipped to [-0.4, 0.4].
Gains in PARAMS were found with CMA-ES on the visible seeds (no neural network, no gradients).
"""

import numpy as np

PARAMS = {
    "T_step": 0.1981,     # nominal step duration [s]
    "t_min": 0.7377,       # earliest contact-triggered switch (fraction of T_step)
    "t_max": 1.1392,       # latest switch (fraction of T_step)
    "th0": 0.2597,        # swing thigh base angle, world frame, positive = forward [rad]
    "c_d": 0.4102,         # swing thigh gain on COM - stance foot distance [rad/m]
    "c_v": 0.1839,        # swing thigh gain on forward COM speed [rad/(m/s)]
    "a0": 0.1324,         # swing thigh base abduction [rad]
    "c_dy": 1.3113,        # lateral placement gain on outward COM offset [rad/m]
    "c_vy": 0.5211,        # lateral placement gain on outward COM speed [rad/(m/s)]
    "knee_bend": 0.9156,   # swing knee flexion peak [rad]
    "knee_stance": 0.0800,# stance knee flexion [rad]
    "v_des": 1.4207,       # target forward speed for the lean regulator [m/s]
    "k_lean": -0.0345,      # extra torso lean per m/s of speed deficit [rad/(m/s)]
    "pitch_des": 0.0181,  # desired torso lean, positive = forward [rad]
    "kp_t": 0.3079,        # torso pitch gain on stance hip [ctrl/rad]
    "kd_t": 0.0251,       # torso pitch rate gain [ctrl/(rad/s)]
    "kp_r": 0.7581,        # torso roll gain on stance hip-x [ctrl/rad]
    "kd_r": 0.1792,
    "c_comp": 0.2731,      # fraction of the swing hip torque compensated at the stance hip
    "kp_hip": 1.1038,      # swing hip PD gains [ctrl/rad, ctrl/(rad/s)]
    "kd_hip": 0.0097,
    "kp_knee": 0.6272,     # knee PD gains
    "kd_knee": 0.0212,
    "kp_abd": 1.1480,      # abdomen PD gains
    "kd_abd": 0.0939,
    "k_abd_t": -0.7553,     # torso pitch feedback through the abdomen [ctrl/rad]
}

ABD_Y, ABD_Z, ABD_X = 0, 1, 2
R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE = 3, 4, 5, 6
L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE = 7, 8, 9, 10
LEG = {"right": (R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE, -1.0), "left": (L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE, +1.0)}

_state = {"stance": "right", "t0": 0.0, "landed": False}


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    t = obs["time"]
    if obs["step"] == 0:
        _state["stance"] = "right"
        _state["t0"] = 0.0
        _state["landed"] = False
    pitch, roll = obs["pitch"], obs["roll"]
    w = obs["root_angular_velocity"]
    v = obs["root_velocity"]
    contact = {"right": obs["foot_contact"], "left": obs["foot_left_contact"]}
    foot_rel = {"right": obs["foot_xyz_rel"], "left": obs["foot_left_xyz_rel"]}

    if not _state["landed"]:
        # the model starts a few centimetres in the air: hold a straight standing pose until touchdown
        if contact["right"] or contact["left"]:
            _state["landed"] = True
            _state["t0"] = t
        else:
            u = np.zeros(17)
            u[:11] = 0.8 * (0.0 - q[:11]) - 0.04 * qd[:11]
            u[[R_KNEE, L_KNEE]] = 0.8 * (-0.05 - q[[R_KNEE, L_KNEE]]) - 0.03 * qd[[R_KNEE, L_KNEE]]
            return np.clip(u, -0.4, 0.4)
    stance = _state["stance"]
    swing = "left" if stance == "right" else "right"
    s = (t - _state["t0"]) / p["T_step"]
    if (s > p["t_min"] and contact[swing]) or s > p["t_max"]:
        stance, swing = swing, stance
        _state["stance"] = stance
        _state["t0"] = t
        s = 0.0

    u = np.zeros(17)
    sx, sz, sy, sk, side_sw = LEG[swing]
    tx, tz, ty, tk, side_st = LEG[stance]

    # --- swing leg: world-frame foot placement ---
    d = -foot_rel[stance][0]                     # COM ahead of the stance foot [m]
    theta_sw = p["th0"] + p["c_d"] * d + p["c_v"] * v[0]
    q_hip_y_des = -theta_sw - pitch - q[ABD_Y]   # thigh forward angle -> joint angle
    d_out = side_sw * (-foot_rel[stance][1])     # COM offset from the stance foot toward the swing side
    v_out = side_sw * v[1]
    a_sw = p["a0"] + p["c_dy"] * d_out + p["c_vy"] * v_out
    q_hip_x_des = -(a_sw - side_sw * roll)       # abduction angle -> joint angle (negative = abduction)
    knee_des = -0.04 - p["knee_bend"] * max(0.0, np.sin(np.pi * min(s, 1.0)))
    u[sy] = p["kp_hip"] * (q_hip_y_des - q[sy]) - p["kd_hip"] * qd[sy]
    u[sx] = p["kp_hip"] * (q_hip_x_des - q[sx]) - p["kd_hip"] * qd[sx]
    u[sz] = p["kp_hip"] * (0.0 - q[sz]) - p["kd_hip"] * qd[sz]
    u[sk] = p["kp_knee"] * (knee_des - q[sk]) - p["kd_knee"] * qd[sk]

    # --- stance leg: torso attitude through the hip, knee nearly straight ---
    pitch_ref = float(np.clip(p["pitch_des"] + p["k_lean"] * (p["v_des"] - v[0]), -0.3, 0.6))
    tau_pitch = p["kp_t"] * (pitch - pitch_ref) + p["kd_t"] * w[1]
    u[ty] = tau_pitch - p["c_comp"] * np.clip(u[sy], -0.4, 0.4)
    tau_roll = p["kp_r"] * roll + p["kd_r"] * w[0]
    u[tx] = -side_st * tau_roll      # positive right-hip-x torque (axis +x) rolls the pelvis toward -x
    u[tz] = p["kp_hip"] * (0.0 - q[tz]) - p["kd_hip"] * qd[tz]
    u[tk] = p["kp_knee"] * (-p["knee_stance"] - q[tk]) - p["kd_knee"] * qd[tk]

    # --- abdomen ---
    u[ABD_Y] = p["kp_abd"] * (0.0 - q[ABD_Y]) - p["kd_abd"] * qd[ABD_Y] + p["k_abd_t"] * (pitch - pitch_ref)
    u[ABD_X] = p["kp_abd"] * (0.0 - q[ABD_X]) - p["kd_abd"] * qd[ABD_X]
    u[ABD_Z] = p["kp_abd"] * (0.0 - q[ABD_Z]) - p["kd_abd"] * qd[ABD_Z]

    # --- arms: fixed pose ---
    arm_des = np.array([0.0, 0.0, -0.8, 0.0, 0.0, -0.8])
    u[11:] = 0.4 * (arm_des - q[11:]) - 0.02 * qd[11:]
    return np.clip(u, -0.4, 0.4)
