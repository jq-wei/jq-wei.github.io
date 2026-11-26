"""SIMBICON-style walking controller for the Humanoid (point feet, so it must keep stepping).

Control law
-----------
* A finite-state machine alternates stance / swing legs.  A swing phase ends when the swing
  foot touches down after at least ``t_min`` s, or unconditionally after ``t_max`` s.
* Swing leg (joint PD): the desired foot position relative to the mass centre, in the heading
  frame, follows a Raibert / capture-point rule
      x_f = kx_v * vx + kx_e * (vx - v_des),    y_f = side * width + ky * vy
  plus a SIMBICON "d" term (kx_d, ky_d) on the mass-centre offset from the stance foot, and is
  converted to world leg angles, then to hip joint targets by subtracting the *pelvis* attitude
  (torso attitude plus the abdomen joints; world angle = joint angle + pelvis angle).  The swing
  hip_z can optionally undo part of the pelvis yaw (yaw_align).  The swing knee is flexed (``knee_lift``)
  during the first part of the swing for clearance and extended (``knee_land``) for landing.
* Stance leg: the stance hip produces the torque that keeps the pelvis at (pitch_des, 0 roll,
  0 yaw) with PD gains (kp_t, kd_t), (kp_r, kd_r), (kp_yaw, kd_yaw), minus the swing hip torque
  (SIMBICON: tau_stance = -tau_torso - tau_swing).  The stance-hip pitch torque is capped at
  ``ty_max`` (50 N m): larger torques make the stance leg push the body off the ground (short
  ballistic hops that end in falls); the cap was the single most important robustness gain.
  The stance knee holds ``knee_stance`` (stiffness kp_sknee).
* ``t_hard`` (= t_max, i.e. plain timeout) and ``flight_hold`` (0 = off) are variants that were
  tried and found worse; they are kept for experiments.
* Abdomen and arms are held at zero by joint PD.
Gains were found by a derivative-free random search over PARAMS on held-out seeds (no learning);
derivative gains are kept small because, at 15 ms control, larger ones make the light joints chatter.
Torques are divided by the motor gears and clipped to [-0.4, 0.4].
"""

import numpy as np

PARAMS = {
    "v_des": 0.4,        # m/s, target forward speed
    "kx_v": 0.2,        # s, foot placement per unit forward velocity
    "kx_e": 0.08853,        # s, extra foot placement per unit speed error
    "kx_d": 0.5,        # foot placement per metre of mass-centre offset ahead of the stance foot (SIMBICON d)
    "ky_d": 0.1,        # same, lateral
    "ky": 0.4,          # s, lateral foot placement per unit lateral velocity
    "width": 0.1,       # m, nominal lateral foot offset from the mass centre
    "leg_h": 0.8393,       # m, nominal hip-to-ground height used to turn foot offsets into leg angles
    "pitch_des": 0.02913,   # rad, torso forward lean
    "t_min": 0.1629,       # s, minimum swing time
    "t_max": 0.2804,       # s, maximum swing time
    "t_hard": 0.2804,      # s, swap legs even without touchdown (after t_max only if the stance foot is airborne)
    "flight_hold": 0.0,  # 1 = in flight the stance hip holds the leg vertical instead of controlling the torso
    "ty_max": 50.0,     # N m, cap on the stance hip pitch torque
    "lift_frac": 0.5,    # fraction of the nominal swing with the knee flexed
    "knee_lift": 0.9,    # rad, swing knee flexion
    "knee_land": 0.1,   # rad, swing knee flexion at landing
    "knee_stance": 0.25, # rad, stance knee flexion
    "kp_t": 405.1, "kd_t": 16.43,     # torso pitch PD through the stance hip [N m/rad, N m s/rad]
    "kp_r": 300, "kd_r": 10.05,     # torso roll PD through the stance hip
    "kp_yaw": 95.39, "kd_yaw": 2, # heading PD through the stance hip
    "kp_hip": 300, "kd_hip": 10, # swing hip pitch joint PD
    "kp_hx": 150, "kd_hx": 4,   # swing hip roll joint PD
    "yaw_align": 0,   # fraction of the pelvis yaw undone by the swing hip_z
    "kp_hz": 62.02, "kd_hz": 1,     # swing hip yaw joint PD
    "kp_knee": 250, "kd_knee": 7.669,   # swing knee PD (kd also used in stance)
    "kp_sknee": 250,                # stance knee stiffness
    "kp_abd": 300, "kd_abd": 1.619,
    "kp_arm": 19.93, "kd_arm": 0.5,
}

# motor indices
AY, AZ, AX = 0, 1, 2
R_HX, R_HZ, R_HY, R_K = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_K = 7, 8, 9, 10
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)

_state = {"stance": 1, "t": 0.0}   # stance: 0 = right, 1 = left


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    dt = obs["dt"]
    if obs["step"] == 0:
        _state["stance"] = 1
        _state["t"] = 0.0
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    # pelvis attitude (hip joints act between pelvis and thigh): torso angles plus abdomen joints
    roll = obs["roll"] + q[AX]
    pitch = obs["pitch"] + q[AY]
    yaw = obs["yaw"] + q[AZ]
    w = obs["root_angular_velocity"] + qd[[AX, AY, AZ]]
    cy, sy = np.cos(obs["yaw"]), np.sin(obs["yaw"])
    vw = obs["com_velocity_xy"]
    vx = cy * vw[0] + sy * vw[1]
    vy = -sy * vw[0] + cy * vw[1]

    # --- state machine
    _state["t"] += dt
    st = _state["stance"]
    swing_contact = obs["foot_contact"] if st == 1 else obs["foot_left_contact"]
    stance_contact = obs["foot_left_contact"] if st == 1 else obs["foot_contact"]
    if (_state["t"] > p["t_min"] and swing_contact) or _state["t"] > p["t_hard"] or \
            (_state["t"] > p["t_max"] and not stance_contact):
        st = 1 - st
        _state["stance"] = st
        _state["t"] = 0.0
    t = _state["t"]
    sw = 1 - st
    side = -1.0 if sw == 0 else 1.0   # lateral side of the swing leg (+y = left)

    tau = np.zeros(17)

    # --- swing leg
    fs = obs["foot_xyz_rel"] if st == 0 else obs["foot_left_xyz_rel"]          # stance foot rel. mass centre (world)
    dx = -(cy * fs[0] + sy * fs[1])                                              # mass centre ahead of stance foot
    dy = -(-sy * fs[0] + cy * fs[1])
    x_f = p["kx_v"] * vx + p["kx_e"] * (vx - p["v_des"]) + p["kx_d"] * dx
    y_f = side * p["width"] + p["ky"] * vy + p["ky_d"] * dy
    dy_hip = y_f - side * 0.1
    phi_x = -np.arctan2(x_f, p["leg_h"])       # world leg angle about +y (negative = foot forward)
    phi_y = np.arctan2(dy_hip, p["leg_h"])      # world leg angle about +x (positive = foot to +y)
    knee = p["knee_lift"] if t < p["lift_frac"] * p["t_max"] else p["knee_land"]
    hy_des = phi_x - pitch - 0.5 * knee
    if sw == 0:
        ihx, ihz, ihy, ik = R_HX, R_HZ, R_HY, R_K
        hx_des = phi_y - roll
        sgn_x = 1.0
    else:
        ihx, ihz, ihy, ik = L_HX, L_HZ, L_HY, L_K
        hx_des = -(phi_y - roll)
        sgn_x = -1.0
    hx_des = float(np.clip(hx_des, -0.43, 0.08))
    hy_des = float(np.clip(hy_des, -1.9, 0.34))
    tau[ihy] = p["kp_hip"] * (hy_des - q[ihy]) - p["kd_hip"] * qd[ihy]
    tau[ihx] = p["kp_hx"] * (hx_des - q[ihx]) - p["kd_hx"] * qd[ihx]
    hz_des = float(np.clip(-sgn_x * _wrap(yaw) * p["yaw_align"], -1.0, 0.6))   # point the swing leg along the world heading
    tau[ihz] = p["kp_hz"] * (hz_des - q[ihz]) - p["kd_hz"] * qd[ihz]
    tau[ik] = p["kp_knee"] * (-knee - q[ik]) - p["kd_knee"] * qd[ik]
    tau_sw_y = float(np.clip(tau[ihy], -120.0, 120.0))
    tau_sw_x = sgn_x * float(np.clip(tau[ihx], -40.0, 40.0))

    # --- stance leg: torso attitude
    if st == 0:
        jhx, jhz, jhy, jk, s_x = R_HX, R_HZ, R_HY, R_K, 1.0
    else:
        jhx, jhz, jhy, jk, s_x = L_HX, L_HZ, L_HY, L_K, -1.0
    t_y = p["kp_t"] * (pitch - p["pitch_des"]) + p["kd_t"] * w[1] - tau_sw_y
    t_x = p["kp_r"] * roll + p["kd_r"] * w[0] - tau_sw_x
    t_z = p["kp_yaw"] * _wrap(yaw) + p["kd_yaw"] * w[2]
    if not stance_contact and p["flight_hold"] > 0:
        # airborne: no ground to push on, so hold the stance leg vertical (ready to land) instead of torso control
        t_y = p["kp_hip"] * (-pitch - q[jhy]) - p["kd_hip"] * qd[jhy]
    tau[jhy] = float(np.clip(t_y, -p["ty_max"], p["ty_max"]))
    tau[jhx] = s_x * t_x
    tau[jhz] = s_x * t_z
    tau[jk] = p["kp_sknee"] * (-p["knee_stance"] - q[jk]) - p["kd_knee"] * qd[jk]

    # --- abdomen and arms
    for i in (AY, AZ, AX):
        tau[i] = p["kp_abd"] * (0.0 - q[i]) - p["kd_abd"] * qd[i]
    for i in range(11, 17):
        tau[i] = p["kp_arm"] * (0.0 - q[i]) - p["kd_arm"] * qd[i]

    return np.clip(tau / GEAR, -0.4, 0.4)
