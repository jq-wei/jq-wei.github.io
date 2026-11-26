"""SIMBICON-style walking controller for the MuJoCo Humanoid, tuned for a brisk gait.

Control law (all torques in N m, converted to motor commands by dividing by the gear
and clipping to [-0.4, 0.4]):

* Every joint tracks a target angle with a PD law ``tau = kp (q_d - q) - kd qdot``.
* A two-state gait machine alternates the stance leg.  A phase ends when the swing
  foot touches down after ``t_min`` or when ``T_MAX`` elapses.  With ``t_min`` of
  about 0.14 s the gait is a fast shuffle (about 7 steps per second, 8 cm per step),
  which is what keeps this ankle-less, point-footed model balanced.
* Torso attitude (pitch, roll, yaw) is regulated in the world frame by a virtual PD
  torque that the *stance* hip applies (minus the swing hip torque, so the net torque
  on the pelvis is the desired one).  ``pitch_d`` is the forward lean of the torso and
  is the main speed knob: 0.008 rad gives 0.3 m/s, 0.06 rad about 0.65 m/s.
  ``kp_t``/``kd_t`` are the pitch PD gains; ``kd_t`` matters for robustness.
* Swing foot placement (SIMBICON balance feedback): the swing thigh angle in the
  heading frame is ``phi = phi0 + cd * d + cv * (v - v_d)`` sagittally, where ``d`` is
  the mass-centre offset from the stance foot and ``v`` the mass-centre velocity;
  laterally ``alpha = side * lat0 + cd_lat * d_y + cv_lat * v_y`` with ``lat0`` the
  half step width.  The joint targets follow from the world-frame angles by
  subtracting the torso pitch / roll.
* The swing knee is flexed to ``swing_knee`` for the first ``KNEE_FRAC`` of ``t_min``
  for clearance and then extended to ``LAND_KNEE``; the stance knee is held at
  ``STANCE_KNEE``.
* The hip_z (leg rotation) joints use the generic hip PD gains.  Their derivative gain
  is high for the small inertia of a leg about its own axis, which makes the hip_z
  motors chatter between the command limits at the control rate; this dither is part
  of the design handed over (it effectively locks the leg rotation) and the gait was
  found to depend on it, so it is kept deliberately.

``PARAMS`` holds the 12 gains that were searched (``hl_control.harbor.tune`` plus a
hand sweep of ``pitch_d``, ``phi0`` and ``kd_t`` on 120 held-out seeds); the remaining gains are
module constants.
"""

import numpy as np

PARAMS = {
    "pitch_d": 0.06,      # forward lean of the torso [rad] (speed knob)
    "v_d": 0.2596,         # speed offset in the placement law [m/s]
    "phi0": 0.23,        # nominal swing thigh angle [rad]
    "cd": 0.3169,          # sagittal placement gain on mass-centre offset [rad/m]
    "cv": 0.1489,          # sagittal placement gain on mass-centre speed [rad s/m]
    "lat0": 0.0928,        # half step width [rad]
    "cd_lat": 0.4633,      # lateral placement gain on offset [rad/m]
    "cv_lat": 0.2964,      # lateral placement gain on speed [rad s/m]
    "t_min": 0.1379,       # minimum swing time [s]
    "kp_t": 686.0898,      # torso pitch stiffness [N m/rad]
    "kd_t": 45.0,          # torso pitch damping [N m s/rad]
    "swing_knee": -0.4776,  # swing knee flexion [rad]
}
PARAM_BOUNDS = {
    "pitch_d": (0.03, 0.08), "v_d": (0.1, 0.4), "phi0": (0.18, 0.27), "cd": (0.2, 0.6),
    "cv": (0.1, 0.22), "lat0": (0.05, 0.12), "cd_lat": (0.3, 0.7), "cv_lat": (0.2, 0.45),
    "t_min": (0.12, 0.19), "kp_t": (450, 850), "kd_t": (30, 70), "swing_knee": (-0.6, -0.3),
}

# fixed gains
T_MAX = 0.35          # maximum swing time [s]
KNEE_FRAC = 0.6791    # fraction of t_min with the swing knee flexed
STANCE_KNEE = -0.1811
LAND_KNEE = -0.2376
KP_R, KD_R = 250.0, 17.23      # torso roll PD
KP_Y, KD_Y = 123.8, 6.241      # torso yaw PD
KP_HIP, KD_HIP = 355.6, 10.62  # hip joint PD (all three hip axes)
KP_KNEE, KD_KNEE = 246.0, 5.897
KP_ABD, KD_ABD = 197.3, 11.2
KP_ARM, KD_ARM = 20.0, 1.0

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
AB_Y, AB_Z, AB_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_K = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_K = 7, 8, 9, 10
ARMS = slice(11, 17)
HX, HZ, HY, KN = (R_HX, L_HX), (R_HZ, L_HZ), (R_HY, L_HY), (R_K, L_K)
SX = (1.0, -1.0)     # hip_x / hip_z axis sign (+x / +z for right, -x / -z for left)
SIDE = (-1.0, 1.0)   # lateral side of each leg (right is -y)

_state = {"stance": 0, "t_phase": 0.0}   # stance 0 = right, 1 = left


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    dt = obs["dt"]
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    st = _state
    if obs["step"] == 0:
        st["stance"] = 0
        st["t_phase"] = 0.0
    st["t_phase"] += dt

    # --- gait machine ---
    contacts = (obs["foot_contact"], obs["foot_left_contact"])
    swing = 1 - st["stance"]
    if (st["t_phase"] > p["t_min"] and contacts[swing]) or st["t_phase"] > T_MAX:
        st["stance"], swing = swing, st["stance"]
        st["t_phase"] = 0.0
    stance = st["stance"]
    i, j = swing, stance

    # --- kinematic quantities in the heading frame ---
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, s], [-s, c]])  # world -> heading frame
    v = rot @ obs["com_velocity_xy"]
    feet = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])
    d = -(rot @ feet[stance][:2])      # mass centre relative to stance foot

    tau = np.zeros(17)
    # abdomen stiff, arms at rest
    tau[[AB_Y, AB_Z, AB_X]] = KP_ABD * (0 - q[[AB_Y, AB_Z, AB_X]]) - KD_ABD * qd[[AB_Y, AB_Z, AB_X]]
    tau[ARMS] = KP_ARM * (0 - q[ARMS]) - KD_ARM * qd[ARMS]

    # --- swing leg targets (world-frame thigh angles -> joint angles) ---
    phi = p["phi0"] + p["cd"] * d[0] + p["cv"] * (v[0] - p["v_d"])
    hy_sw = -phi - pitch
    alpha = SIDE[i] * p["lat0"] + p["cd_lat"] * d[1] + p["cv_lat"] * v[1]
    hx_sw = SX[i] * (alpha - roll)     # world lateral thigh angle = roll + sx * q_hx
    hz_sw = 0.0
    frac = st["t_phase"] / p["t_min"]
    k_sw = p["swing_knee"] if frac < KNEE_FRAC else LAND_KNEE

    tau[HY[i]] = KP_HIP * (hy_sw - q[HY[i]]) - KD_HIP * qd[HY[i]]
    tau[HX[i]] = KP_HIP * (hx_sw - q[HX[i]]) - KD_HIP * qd[HX[i]]
    tau[HZ[i]] = KP_HIP * (hz_sw - q[HZ[i]]) - KD_HIP * qd[HZ[i]]
    tau[KN[i]] = KP_KNEE * (k_sw - q[KN[i]]) - KD_KNEE * qd[KN[i]]

    # --- stance leg: knee PD, hip applies the torso attitude torque ---
    tau[KN[j]] = KP_KNEE * (STANCE_KNEE - q[KN[j]]) - KD_KNEE * qd[KN[j]]
    t_pitch = p["kp_t"] * (p["pitch_d"] - pitch) - p["kd_t"] * w[1]
    t_roll = KP_R * (0.0 - roll) - KD_R * w[0]
    t_yaw = KP_Y * _wrap(0.0 - yaw) - KD_Y * w[2]
    # hip torques on the pelvis are minus the joint torques (axis-signed)
    tau[HY[j]] = -t_pitch - tau[HY[i]]
    tau[HX[j]] = SX[j] * (-t_roll - SX[i] * tau[HX[i]])
    tau[HZ[j]] = SX[j] * (-t_yaw - SX[i] * tau[HZ[i]])

    return np.clip(tau / GEAR, -0.4, 0.4)
