"""SIMBICON-style stepping controller for the MuJoCo Humanoid (hand-designed, no learning).

The plant has point feet (spheres) and weak knees (80 N m), so it cannot stand still: the
controller keeps it upright by stepping quickly (T_STEP ~ 0.14 s, ~100 touchdowns per
15 s) and placing each swing foot where the mass-centre velocity says it should land.

Control law (66.7 Hz, all motors are joint PD servos  u = (KP (q* - q) - KD qdot) / gear
clipped to [-0.4, 0.4]; the gains are the PARAMS below, each one commented):
  * Gait machine: two states (right stance / left stance) on a phase clock of period
    T_STEP; a step ends at phase 1, or earlier when the swing foot touches down after
    SWITCH_MIN of the phase with its knee straighter than KNEE_TD_MIN. The first
    T_SETTLE seconds are spent standing on both legs (the model starts 10 cm above the floor).
  * Swing hip_y: the thigh angle is commanded in the WORLD frame (torso pitch subtracted):
    HIP_SWING - CV (v_x - V_DES) - CD d, with v the mass-centre velocity rotated into the
    heading frame and d the distance of the mass centre ahead of the stance foot
    (SIMBICON foot placement). The target ramps in over BLEND of the phase from the
    lift-off angle to avoid torque spikes.
  * Swing knee: a sinusoidal lift bump of depth KNEE_LIFT that is straight again at
    KNEE_EXT_PHASE, with a soft gain KP_KNEE_SW (a stiff knee servo reacts on the thigh
    and cancels the hip flexion).
  * Swing hip_x: lateral foot placement HIPX_BASE +/- (CVY v_y - CROLL roll).
  * Stance hip_y: torso pitch servo (PITCH_KP, PITCH_KD about PITCH_DES) minus COMP times
    the swing-hip command, so the swing leg's reaction does not pitch the pelvis.
  * Stance hip_x: torso roll servo (ROLL_KP, ROLL_KD).
  * Stance knee: PD to STANCE_KNEE (nearly straight, the joint stop carries the load); its
    gain ramps from KP_KNEE_SW to KP_KNEE over KNEE_RAMP of the step so a knee that lands
    bent does not catapult the body.
  * hip_z: a soft PD to zero (KP_HIPZ, KD_HIPZ); a point foot transmits no yaw torque, so
    the heading servo (YAW_KP, YAW_KD, applied in double support) is kept near zero.
  * Abdomen: stiff PD to zero plus 30 % of the pitch servo; arms: soft PD to a fixed pose.
Internal state (stance leg, step start time, lift-off angles) is reset when obs["step"] == 0.
"""
import numpy as np

PARAMS = {
    'T_STEP': 0.13847,  # s, nominal step (swing) duration of the phase clock
    'SWITCH_MIN': 0.229216,  # phase fraction after which a swing-foot touchdown ends the step
    'HIP_SWING': -0.448561,  # rad, base swing-thigh angle (world frame, negative = forward)
    'KNEE_LIFT': -1.28505,  # rad, peak swing-knee flexion of the lift bump
    'CV': 0.484765,  # s/m, swing-thigh correction per m/s of forward speed error
    'CD': 0.613196,  # 1/m, swing-thigh correction per m of mass centre ahead of the stance foot
    'V_DES': 0.649108,  # m/s, desired forward speed
    'PITCH_KP': 137.039,  # N m/rad, stance-hip torso pitch servo
    'PITCH_KD': 11.9915,  # N m s/rad, pitch rate damping
    'PITCH_DES': -0.000103013,  # rad, desired torso pitch
    'ROLL_KP': 471.52,  # N m/rad, stance-hip torso roll servo
    'ROLL_KD': 3.63348,  # N m s/rad, roll rate damping
    'YAW_KP': 0.030036,  # N m/rad, double-support heading servo (both hip_z)
    'YAW_KD': 0.0289608,  # N m s/rad, yaw rate damping
    'YAW_SWING': 0.0446029,  # rad/rad, swing hip_z pre-rotation against heading error
    'HIPX_BASE': -0.191841,  # rad, base swing hip_x (negative = abduction, sets step width)
    'CVY': 0.181985,  # s/m, lateral swing-foot placement per m/s of lateral speed
    'CROLL': 0.26766,  # rad/rad, lateral placement per rad of roll
    'KP_HIP': 466.479,  # N m/rad, swing hip_x/hip_y PD
    'KD_HIP': 4.86394,  # N m s/rad
    'KP_HIPZ': 25.6061,  # N m/rad, soft hip_z PD (a stiff one chatters)
    'KD_HIPZ': 0.366562,  # N m s/rad
    'KP_KNEE': 195.627,  # N m/rad, stance knee PD
    'KD_KNEE': 3.37708,  # N m s/rad
    'KP_KNEE_SW': 129.504,  # N m/rad, swing knee PD (soft: its reaction fights hip flexion)
    'KP_ABD': 193.662,  # N m/rad, abdomen PD to zero
    'KD_ABD': 10.2328,  # N m s/rad
    'KP_ARM': 1.06477,  # N m/rad, arm PD to a fixed pose
    'KD_ARM': 0.339847,  # N m s/rad
    'STANCE_KNEE': -0.0284728,  # rad, stance knee target (nearly straight)
    'COMP': 0.293051,  # fraction of the swing-hip torque cancelled at the stance hip
    'T_SETTLE': 0.307018,  # s, stand on both legs before the first step
    'KNEE_EXT_PHASE': 1.14434,  # phase fraction by which the swing knee is straight again
    'BLEND': 0.151836,  # phase fraction over which the swing-hip target ramps in from lift-off
    'KNEE_RAMP': 0.000903811,  # phase fraction over which the stance-knee gain ramps up after touchdown
    'KNEE_TD_MIN': -0.4,  # rad, touchdown only ends the step if the swing knee is straighter than this
}

# motor indices
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE = 3, 4, 5, 6
L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE = 7, 8, 9, 10
ARMS = [11, 12, 13, 14, 15, 16]
ARM_POSE = np.array([0.0, 0.0, -0.5, 0.0, 0.0, -0.5])

_state = {"stance": 0, "t_switch": 0.0, "hy0": 0.0, "hx0": 0.0}   # stance: 0 = right leg is stance, 1 = left


def _reset():
    _state["stance"] = 0
    _state["t_switch"] = 0.0
    _state["hy0"] = 0.0
    _state["hx0"] = 0.0


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = float(obs["time"])
    if int(obs["step"]) == 0:
        _reset()
    q = obs["joint_angles"]; qd = obs["joint_rates"]; gear = obs["gear"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]
    vw = obs["com_velocity_xy"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    v = np.array([cy * vw[0] + sy * vw[1], -sy * vw[0] + cy * vw[1]])   # heading-frame velocity
    fr, fl = obs["foot_contact"], obs["foot_left_contact"]
    feet = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])

    st = _state["stance"]
    if t < P["T_SETTLE"]:
        # settle on both legs: stiff servo to the straight standing pose
        kp = np.full(17, P["KP_HIP"]); kd = np.full(17, P["KD_HIP"]); qt = np.zeros(17)
        qt[[R_KNEE, L_KNEE]] = P["STANCE_KNEE"]; qt[ARMS] = ARM_POSE; kp[ARMS] = P["KP_ARM"]; kd[ARMS] = P["KD_ARM"]
        _state["t_switch"] = t
        return np.clip((kp * (qt - q) - kd * qd) / gear, -0.4, 0.4)
    phase = (t - _state["t_switch"]) / P["T_STEP"]
    swing_contact = fl if st == 0 else fr
    swing_knee = q[L_KNEE if st == 0 else R_KNEE]
    if phase >= 1.0 or (phase > P["SWITCH_MIN"] and swing_contact and swing_knee > P["KNEE_TD_MIN"]):
        st = 1 - st
        _state["stance"] = st
        _state["t_switch"] = t
        phase = 0.0
        # remember the new swing leg's hip angles to blend smoothly from them
        _state["hy0"] = q[L_HIP_Y if st == 0 else R_HIP_Y]
        _state["hx0"] = q[L_HIP_X if st == 0 else R_HIP_X]

    if st == 0:
        s_hx, s_hz, s_hy, s_kn = R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE
        w_hx, w_hz, w_hy, w_kn = L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE
        s_sign = 1.0           # right hip axes: x -> +x, z -> +z
    else:
        s_hx, s_hz, s_hy, s_kn = L_HIP_X, L_HIP_Z, L_HIP_Y, L_KNEE
        w_hx, w_hz, w_hy, w_kn = R_HIP_X, R_HIP_Z, R_HIP_Y, R_KNEE
        s_sign = -1.0          # left hip axes are mirrored
    stance_foot = feet[st]
    d = -(cy * stance_foot[0] + sy * stance_foot[1])       # mass centre ahead of the stance foot (m, heading frame)

    qt = np.zeros(17); kp = np.zeros(17); kd = np.zeros(17)
    # abdomen
    kp[[ABD_Y, ABD_Z, ABD_X]] = P["KP_ABD"]; kd[[ABD_Y, ABD_Z, ABD_X]] = P["KD_ABD"]
    # arms
    kp[ARMS] = P["KP_ARM"]; kd[ARMS] = P["KD_ARM"]; qt[ARMS] = ARM_POSE
    # swing leg
    kp[[w_hx, w_hy]] = P["KP_HIP"]; kd[[w_hx, w_hy]] = P["KD_HIP"]
    kp[[w_hz, s_hz]] = P["KP_HIPZ"]; kd[[w_hz, s_hz]] = P["KD_HIPZ"]
    kp[w_kn] = P["KP_KNEE_SW"]; kd[w_kn] = P["KD_KNEE"]
    # swing thigh angle commanded in the world frame (SIMBICON): subtract the torso pitch
    hy_target = P["HIP_SWING"] - P["CV"] * (v[0] - P["V_DES"]) - P["CD"] * d - pitch
    blend = min(phase / P["BLEND"], 1.0)                 # smooth ramp from the lift-off angle to the target
    qt[w_hy] = (1 - blend) * _state["hy0"] + blend * hy_target
    ph = min(phase, 1.0)
    qt[w_kn] = -0.1 + (P["KNEE_LIFT"] + 0.1) * np.sin(np.pi * ph / P["KNEE_EXT_PHASE"]) if ph < P["KNEE_EXT_PHASE"] else -0.1   # lift bump, straight before touchdown
    w_sign = -s_sign
    # lateral foot placement: more negative = more abduction; right foot abducts to -y, left to +y
    lat = P["CVY"] * v[1] - P["CROLL"] * roll     # positive when the body moves or leans to +y (left); roll > 0 tilts to -y
    qt[w_hx] = P["HIPX_BASE"] + (lat if w_sign > 0 else -lat)  # w_sign>0: swing is right leg
    qt[w_hz] = w_sign * P["YAW_SWING"] * yaw   # pre-rotate the swing foot against the heading error
    # stance leg
    # stance knee: gain ramps up after touchdown so a knee that lands bent does not catapult the body
    ramp = min(phase / P["KNEE_RAMP"], 1.0)
    kp[s_kn] = P["KP_KNEE_SW"] + (P["KP_KNEE"] - P["KP_KNEE_SW"]) * ramp; kd[s_kn] = P["KD_KNEE"]; qt[s_kn] = P["STANCE_KNEE"]
    u = (kp * (qt - q) - kd * qd) / gear
    pitch_err = pitch - P["PITCH_DES"]
    # stance hip: torso pitch servo, compensating the swing hip's reaction torque on the pelvis
    u[s_hy] = (P["PITCH_KP"] * pitch_err + P["PITCH_KD"] * w[1]) / gear[s_hy] - P["COMP"] * np.clip(u[w_hy], -0.4, 0.4)
    u[s_hx] = s_sign * (P["ROLL_KP"] * roll + P["ROLL_KD"] * w[0]) / gear[s_hx]
    if fr and fl:
        # yaw is only controllable through two planted (point) feet: servo both hip_z in double support
        yaw_u = (P["YAW_KP"] * yaw + P["YAW_KD"] * w[2])
        u[R_HIP_Z] += yaw_u / gear[R_HIP_Z]
        u[L_HIP_Z] -= yaw_u / gear[L_HIP_Z]
    else:
        u[s_hz] = (P["KP_HIPZ"] * (0.0 - q[s_hz]) - P["KD_HIPZ"] * qd[s_hz]) / gear[s_hz]
    u[ABD_Y] += (0.3 * P["PITCH_KP"] * pitch_err + 0.3 * P["PITCH_KD"] * w[1]) / gear[ABD_Y]
    return np.clip(u, -0.4, 0.4)
