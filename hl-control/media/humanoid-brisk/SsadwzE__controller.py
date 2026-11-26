"""SIMBICON-style stepping controller for the MuJoCo Humanoid (hand-designed, no learning).

Control law (per 15 ms step; outputs normalised motor commands in [-0.4, 0.4]):
  * Settle phase: the model starts 10 cm above the floor, so a plain PD hold of a standing
    pose is applied until both feet touch (or 0.6 s), then the gait machine starts.
  * Two-state gait machine (right-stance / left-stance).  Roles swap when the swing foot
    touches the floor after T_MIN seconds, or after T_MAX seconds at the latest.
  * Every joint runs a PD law  u = KP * (q_des - q) - KD * qdot  (command units per rad;
    KD is small because the loop is sampled at 66.7 Hz).  Desired angles:
      - swing hip y/x: foot placement.  Velocity v and mass-centre offset d (COM minus
        stance foot) are expressed in the torso's yaw frame; the swing foot is sent to
          x_t = step0 + c_v * v_x + c_d * d_x - C_VDES * v_des * goal_x (+ overspeed brake)
          y_t = width (outwards) + c_vy * v_y + 0.3 * (lateral offset) - C_VDES * v_des * goal_y
        where goal is world +x in the torso frame (so progress is along world x even if the
        torso yaws), and v_des ramps in over RAMP seconds.  Targets are converted to hip
        angles with the leg length, the torso pitch/roll and the planned knee flexion.
      - swing knee: sine bump of amplitude knee_swing for floor clearance.
      - stance knee: held at its stop with a constant extension bias (KNEE_BIAS), so the
        stance leg is a rigid strut and the body vaults over the point foot.
      - stance hip y/x (only while the stance foot is in contact): torso attitude servo
        u = kp_t * (pitch - pitch_des) + kd_t * pitch_rate (same for roll), minus the swing
        hip command (SIMBICON torso compensation); ABD_SHARE of it also goes to abdomen_y.
      - hip_z of both legs: twist the pelvis back toward yaw = 0 with gain k_yaw.
  * Abdomen and arms are PD-held at a fixed posture.
PARAMS are the gains searched with hl_control.harbor.tune (bounds in PARAM_BOUNDS).
"""


import numpy as np

PARAMS = {
    "T_MAX": 0.239,       # s, maximum swing duration before forced swap
    "pitch_des": -0.076,   # rad, desired forward torso lean
    "step0": -0.069,       # m, nominal swing-foot target ahead of the hip
    "c_v": 0.099,         # s, foot placement per unit forward velocity
    "c_d": 0.700,          # -, foot placement per unit COM-stance offset
    "v_des": 0.665,        # m/s, desired speed (shifts the foot target backward)
    "k_yaw": 0.089,        # rad/rad, hip_z heading correction toward yaw = 0
    "knee_swing": 1.042,   # rad, swing knee flexion amplitude
    "kp_t": 1.063,         # cmd/rad, torso pitch/roll servo
    "kd_t": 0.143,         # cmd/(rad/s)
    "c_vy": 0.195,        # s, lateral placement per unit lateral velocity
    "width": 0.132,       # m, default lateral swing-foot target (outwards)
}
PARAM_BOUNDS = {
    "T_MAX": (0.14, 0.32), "pitch_des": (-0.15, 0.15), "step0": (-0.15, 0.1), "c_v": (0.0, 0.45),
    "c_d": (0.5, 0.95), "v_des": (0.3, 1.6), "k_yaw": (0.0, 1.5), "knee_swing": (0.5, 1.5),
    "kp_t": (1.0, 2.5), "kd_t": (0.05, 0.35), "c_vy": (0.0, 0.6), "width": (0.08, 0.2),
}

T_MIN = 0.12          # s, no touchdown-triggered swap before this
LEG = 0.85            # m, hip-to-foot length (straight leg)
C_BRAKE = 0.3         # s, extra forward placement per unit overspeed
C_VDES = 0.1          # s, foot-placement shift per unit desired speed
KNEE_BIAS = 0.3       # cmd, stance knee extension bias
RAMP = 2.0            # s, time over which the speed demand ramps in
ABD_SHARE = 0.5       # fraction of the torso pitch command also sent to the abdomen

# motor indices
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_KNEE = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_KNEE = 7, 8, 9, 10
ARMS = [11, 12, 13, 14, 15, 16]

KP = np.array([3.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0] + [0.5] * 6)
KD = np.array([0.02, 0.02, 0.02, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01] + [0.01] * 6)
ARM_POSE = np.array([0.0, 0.0, -0.5, 0.0, 0.0, -0.5])

_state = {"stance": "right", "t0": 0.0, "last_step": -1, "started": False}


def reset():
    _state.update(stance="right", t0=0.0, last_step=-1, started=False)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = float(obs["time"])
    if obs["step"] <= _state["last_step"] or obs["step"] == 0:
        reset()
    _state["last_step"] = obs["step"]
    if not _state["started"]:
        # settle: the model starts 10 cm above the floor; hold a standing pose until it has landed
        if (obs["foot_contact"] and obs["foot_left_contact"]) or t > 0.6:
            _state["started"] = True
            _state["t0"] = t
        else:
            q_des = np.zeros(17)
            q_des[ARMS] = ARM_POSE
            q_des[[R_KNEE, L_KNEE]] = -0.1
            q_des[[R_HX, L_HX]] = -0.05
            return np.clip(KP * (q_des - obs["joint_angles"]) - KD * obs["joint_rates"], -0.4, 0.4)

    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    w = obs["root_angular_velocity"]
    v = obs["com_velocity_xy"]
    com = obs["com_xyz"]

    # ---- gait state machine ----
    stance = _state["stance"]
    ts = t - _state["t0"]
    swing_contact = obs["foot_left_contact"] if stance == "right" else obs["foot_contact"]
    if ts >= P["T_MAX"] or (ts >= T_MIN and swing_contact):
        stance = "left" if stance == "right" else "right"
        _state["stance"] = stance
        _state["t0"] = t
        ts = 0.0
    phase = min(ts / P["T_MAX"], 1.0)

    if stance == "right":
        st_hx, st_hz, st_hy, st_k = R_HX, R_HZ, R_HY, R_KNEE
        sw_hx, sw_hz, sw_hy, sw_k = L_HX, L_HZ, L_HY, L_KNEE
        stance_foot = obs["foot_xyz_rel"]
        s_sw = -1.0   # left leg: +hip_x moves foot toward -y
    else:
        st_hx, st_hz, st_hy, st_k = L_HX, L_HZ, L_HY, L_KNEE
        sw_hx, sw_hz, sw_hy, sw_k = R_HX, R_HZ, R_HY, R_KNEE
        stance_foot = obs["foot_left_xyz_rel"]
        s_sw = 1.0
    d = -stance_foot[:2]            # COM minus stance foot (horizontal, world frame)
    # express velocity, offset and the goal direction (world +x) in the torso's yaw frame
    cy, sy = np.cos(yaw), np.sin(yaw)
    rot = np.array([[cy, sy], [-sy, cy]])
    v = rot @ v
    d = rot @ d
    goal = rot @ np.array([1.0, 0.0])

    # ---- desired joint angles ----
    q_des = np.zeros(17)
    q_des[ARMS] = ARM_POSE
    # swing foot placement (world frame targets relative to the hip)
    v_des = P["v_des"] * min(1.0, t / RAMP)       # ramp the speed demand in from standstill
    x_t = P["step0"] + P["c_v"] * v[0] + P["c_d"] * d[0] - C_VDES * v_des * goal[0]
    x_t += C_BRAKE * max(0.0, v[0] - v_des - 0.25)   # overspeed: reach further forward to brake
    x_t = max(x_t, -0.15)
    y_t = -s_sw * P["width"] + P["c_vy"] * v[1] + 0.3 * (d[1] + s_sw * 0.1) - C_VDES * v_des * goal[1]
    knee_sw = -0.05 - P["knee_swing"] * np.sin(np.pi * phase)
    # foot x relative to hip ~ -0.40 sin(thigh) - 0.45 sin(thigh - knee): flex the thigh more while the knee is bent
    hip_y_sw = (0.45 * knee_sw - x_t) / LEG - pitch - q[ABD_Y]
    hip_x_sw = s_sw * (y_t / LEG - roll - q[ABD_X])
    q_des[sw_hy] = np.clip(hip_y_sw, -1.6, 0.3)
    q_des[sw_hx] = np.clip(hip_x_sw, -0.4, 0.08)
    q_des[sw_k] = knee_sw
    q_des[st_k] = 0.0
    # heading: twist both hips so the pelvis turns back toward yaw = 0 while the feet are planted
    q_des[R_HZ] = np.clip(P["k_yaw"] * yaw, -0.3, 0.3)
    q_des[L_HZ] = -q_des[R_HZ]

    u = KP * (q_des - q) - KD * qd
    u[st_k] += KNEE_BIAS            # press the stance knee onto its stop: a rigid stance leg
    # ---- stance hip: torso attitude servo + swing compensation ----
    stance_contact = obs["foot_contact"] if stance == "right" else obs["foot_left_contact"]
    if stance_contact:
        u_pitch = P["kp_t"] * (pitch - P["pitch_des"]) + P["kd_t"] * w[1]
        u[st_hy] = u_pitch - u[sw_hy]
        u[ABD_Y] += ABD_SHARE * u_pitch       # the abdomen helps right the torso
        s_st = -s_sw
        u_roll = P["kp_t"] * roll + P["kd_t"] * w[0]
        # +hip_x torque on right thigh (axis +x) reacts on pelvis about -x -> roll decreases
        u[st_hx] = s_st * u_roll - u[sw_hx]
    # else: stance foot in the air -> plain PD toward the neutral stance pose (q_des = 0)
    return np.clip(u, -0.4, 0.4)
