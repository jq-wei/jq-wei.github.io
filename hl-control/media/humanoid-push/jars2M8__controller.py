"""Stepping controller for the MuJoCo Humanoid (Gymnasium Humanoid-v5 model, point feet, no ankles).

Control law (SIMBICON-style finite-state stepping with Raibert foot placement):

* After a short settle on both feet, a two-state machine alternates the stance leg.  A step
  ends when the swing foot touches down (after 0.6 T_step), when the swing foot is loaded and
  the stance foot is not, or after 1.1 T_step.
* Swing leg: the foot is sent to a Raibert target relative to the mass centre,
  x_f = v_x T_step / 2 + k_fp (v_x - v_cmd) forward and y_f = w_lat + k_lat v_side outward
  (v_side is the lateral velocity toward the swing side), converted to hip angles through the
  two-segment leg geometry (L_SEG, HIP_OFF) and corrected for the pelvis orientation and for
  the shift of the mass centre caused by torso lean (COM_LEAN).  The knee flexes by knee_sw
  (sinusoidal in the step phase) for ground clearance.  All joints then follow a PD law
  (KP, KD in N m/rad and N m s/rad, divided by the motor gears).
* Stance leg: hip_y torque = kp_pitch (pitch - pitch_des) + kd_pitch * pitch rate keeps the
  torso at a small forward lean in the world frame; hip_x likewise regulates torso roll with
  kp_roll / kd_roll (sign flipped for the left leg, whose axis is mirrored); the knee is held
  nearly straight by PD.
* The abdomen adds kp_abd times the torso pitch/roll error so the upper body stays upright.
* The speed command v_cmd ramps from 0 to v_des over V_RAMP seconds.

PARAMS were found with hl_control.harbor.tune (TPE then CMA-ES) on the visible seeds.
"""
import numpy as np

PARAMS = {
    "T_step": 0.3221,      # nominal step duration [s]
    "v_des": 0.5843,        # desired forward speed [m/s]
    "k_fp": 0.0214,        # Raibert foot-placement gain on forward velocity error [s]
    "k_lat": 0.1042,       # lateral foot-placement gain [s]
    "w_lat": 0.2657,       # nominal lateral foot offset from the COM [m]
    "pitch_des": 0.1169,   # desired torso forward lean [rad]
    "kp_pitch": 1.4211,     # stance hip torque per rad of torso pitch error (ctrl units)
    "kd_pitch": 0.1349,
    "kp_roll": 0.5425,
    "kd_roll": 1.0454,
    "kp_abd": 3.6434,       # abdomen torque per rad of torso tilt (ctrl units)
    "knee_sw": 1.2507,      # swing knee flexion [rad]
}
PARAM_BOUNDS = {k: (0.3 * v, 2.0 * v) for k, v in PARAMS.items()}
V_RAMP = 2.0       # time to ramp the speed command up [s]
COM_LEAN = 0.20    # COM shift per rad of torso lean, relative to the pelvis [m/rad]
T_MAX = 1.1        # swing time-out as a multiple of T_step

NAMES = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
         "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
         "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
IDX = {n: i for i, n in enumerate(NAMES)}
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
KP = np.array([150, 80, 150, 250, 80, 250, 150, 250, 80, 250, 150, 10, 10, 5, 10, 10, 5], float)
KD = np.array([5, 3, 5, 8, 3, 10, 4, 8, 3, 10, 4, 0.5, 0.5, 0.3, 0.5, 0.5, 0.3], float)
L_SEG = 0.40      # thigh and shin length [m]
HIP_OFF = 0.10    # lateral hip offset from the pelvis centre [m]

_state = {"stance": 0, "t_sw": 0.0, "last_t": -1.0, "started": False, "t0": 0.0}


def reset():
    _state.update(stance=0, t_sw=0.0, last_t=-1.0, started=False, t0=0.0)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = obs["time"]
    if t < _state["last_t"] or t == 0.0:
        reset()
    _state["last_t"] = t
    dt = obs["dt"]
    q = obs["joint_angles"]; qd = obs["joint_rates"]
    pitch, roll = obs["pitch"], obs["roll"]
    wx, wy = obs["root_angular_velocity"][0], obs["root_angular_velocity"][1]
    vx, vy = obs["com_velocity_xy"]
    contacts = (obs["foot_contact"], obs["foot_left_contact"])   # right, left
    pelvis_pitch = pitch + q[IDX["abdomen_y"]]
    pelvis_roll = roll + q[IDX["abdomen_x"]]

    q_des = np.zeros(17)
    q_des[IDX["right_shoulder1"]] = 0.3; q_des[IDX["right_shoulder2"]] = -0.3; q_des[IDX["right_elbow"]] = -0.5
    q_des[IDX["left_shoulder1"]] = -0.3; q_des[IDX["left_shoulder2"]] = 0.3; q_des[IDX["left_elbow"]] = -0.5

    # ---------- initial settle: both feet down, PD stand ----------
    if not _state["started"]:
        if contacts[0] and contacts[1] and t > 0.1:
            _state["started"] = True
            _state["t_sw"] = 0.0
            _state["t0"] = t
        else:
            q_des[IDX["right_knee"]] = q_des[IDX["left_knee"]] = -0.1
            q_des[IDX["right_hip_y"]] = q_des[IDX["left_hip_y"]] = -pelvis_pitch
            u = (KP * (q_des - q) - KD * qd) / GEAR
            return np.clip(u, -0.4, 0.4)
    v_cmd = P["v_des"] * min(1.0, (t - _state["t0"]) / max(V_RAMP, 1e-3))

    # ---------- finite state machine: 0 = right stance (left swings), 1 = left stance ----------
    st = _state["stance"]; sw = 1 - st
    _state["t_sw"] += dt
    tau = _state["t_sw"]
    if ((tau > 0.6 * P["T_step"] and contacts[sw]) or tau > T_MAX * P["T_step"]
            or (tau > 0.3 * P["T_step"] and contacts[sw] and not contacts[st])):
        _state["stance"] = sw; st, sw = sw, st; _state["t_sw"] = 0.0; tau = 0.0
    phase = min(tau / P["T_step"], 1.0)
    sw_name = "left" if sw == 1 else "right"
    st_name = "right" if sw == 1 else "left"
    side = 1.0 if sw == 1 else -1.0           # +y is the left side

    # ---------- swing leg: Raibert foot placement, foot-space targets -> hip angles ----------
    x_f = vx * P["T_step"] / 2 + P["k_fp"] * (vx - v_cmd)
    x_f = np.clip(x_f, -0.25, 0.40) + COM_LEAN * pitch          # COM-relative -> pelvis-relative
    knee_des = -0.1 - P["knee_sw"] * np.sin(np.pi * phase)
    k = knee_des
    # thigh world angle so the foot (knee at k) sits x_f ahead of the COM: x = L sin(phi) + L sin(phi + k)
    phi = np.arcsin(np.clip(x_f / (2 * L_SEG * np.cos(k / 2)), -0.95, 0.95)) - k / 2
    q_des[IDX[sw_name + "_hip_y"]] = -phi - pelvis_pitch
    q_des[IDX[sw_name + "_knee"]] = knee_des
    v_side = vy * side
    y_f = P["w_lat"] + P["k_lat"] * v_side                  # lateral foot offset from COM toward the swing side
    y_f = np.clip(y_f, 0.05, 0.45) - side * COM_LEAN * roll     # COM-relative -> pelvis-relative
    theta_out = np.arcsin(np.clip((y_f - HIP_OFF) / (2 * L_SEG * np.cos(k / 2)), -0.95, 0.95))
    q_des[IDX[sw_name + "_hip_x"]] = -theta_out + side * pelvis_roll
    q_des[IDX[sw_name + "_hip_z"]] = 0.0

    # ---------- stance leg ----------
    q_des[IDX[st_name + "_knee"]] = -0.12
    q_des[IDX[st_name + "_hip_x"]] = q[IDX[st_name + "_hip_x"]]
    q_des[IDX[st_name + "_hip_z"]] = 0.0
    q_des[IDX[st_name + "_hip_y"]] = q[IDX[st_name + "_hip_y"]]
    q_des[IDX["abdomen_y"]] = 0.0; q_des[IDX["abdomen_x"]] = 0.0; q_des[IDX["abdomen_z"]] = 0.0

    u = (KP * (q_des - q) - KD * qd) / GEAR
    # stance hip controls torso pitch and roll in the world frame (verified signs)
    u[IDX[st_name + "_hip_y"]] = P["kp_pitch"] * (pitch - P["pitch_des"]) + P["kd_pitch"] * wy
    st_sign = 1.0 if st == 0 else -1.0
    u[IDX[st_name + "_hip_x"]] = st_sign * (P["kp_roll"] * roll + P["kd_roll"] * wx) - 0.05 * qd[IDX[st_name + "_hip_x"]]
    # abdomen keeps the torso upright as well
    u[IDX["abdomen_y"]] += P["kp_abd"] * (pitch - P["pitch_des"]) + 0.1 * wy
    u[IDX["abdomen_x"]] += P["kp_abd"] * roll + 0.1 * wx
    return np.clip(u, -0.4, 0.4)
