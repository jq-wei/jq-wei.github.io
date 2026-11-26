"""Open-loop gait generator + joint PD + torso-lean feedback for the MuJoCo Humanoid.

Control law
-----------
Each leg follows a periodic joint-angle reference, the left leg half a period behind
the right:  q_des = off + amp * sin(2 pi f t + phase)  for hip_y (swing), knee, hip_x
(sideways sway) with a start-up ramp.  The abdomen has a lean offset and a small sway.
Arms swing in anti-phase with the legs.  Each actuated joint is driven by a PD law on
the reference, u = kp (q_des - q) - kd qdot  (units of the normalised command, clipped
to +-0.4 by the plant).  Torso pitch/roll feedback (k_pitch, k_pitchd, k_roll, k_rolld)
shifts the hip and abdomen references to keep the trunk upright.

All numbers live in PARAMS (tuned offline by CMA-ES on the visible seeds).
"""

import numpy as np

PARAMS = {
    "f": 3.1211,
    "hy_off": -0.53769,
    "hy_amp": 0.57751,
    "hy_ph": 0.10656,
    "kn_off": -0.54738,
    "kn_amp": 0.67622,
    "kn_ph": -0.91223,
    "hx_off": -0.06757,
    "hx_amp": 0.03039,
    "hx_ph": -0.07861,
    "hz_off": -0.02693,
    "ab_y": 0.17478,
    "ab_x_amp": 0.05986,
    "ab_x_ph": -0.0587,
    "arm_amp": 0.57357,
    "kp_hy": 0.68758,
    "kd_hy": -0.00028,
    "kp_kn": 0.91262,
    "kd_kn": 0.03879,
    "kp_hx": 4.0748,
    "kd_hx": 0.01072,
    "kp_ab": 0.94401,
    "kd_ab": 0.06126,
    "k_pitch": 0.5967,
    "k_pitchd": 0.06546,
    "k_roll": 0.01256,
    "k_rolld": -0.07265,
    "k_vx": -0.08496,
    "k_vy": -0.06422,
    "k_yaw": 0.09277,
    "k_yawd": 0.01823,
    "vdes": 1.15583,
    "k_abp": 0.03916,
    "k_abx": 0.01658,
    "k_abz": -0.01261,
    "ramp": 1.02328,
}


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    t = obs["time"]
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    ramp = min(1.0, t / max(p["ramp"], 1e-3))
    pitch = obs["pitch"]
    roll = obs["roll"]
    pitchd = obs["root_angular_velocity"][1]
    rolld = obs["root_angular_velocity"][0]
    yaw = obs["yaw"]
    vx, vy = obs["com_velocity_xy"]
    vfx = np.cos(yaw) * vx + np.sin(yaw) * vy      # heading-frame mass-centre velocity
    vfy = -np.sin(yaw) * vx + np.cos(yaw) * vy
    fb_p = p["k_pitch"] * pitch + p["k_pitchd"] * pitchd + p["k_vx"] * (vfx - p["vdes"])
    fb_r = p["k_roll"] * roll + p["k_rolld"] * rolld + p["k_vy"] * vfy
    fb_z = p["k_yaw"] * yaw + p["k_yawd"] * obs["root_angular_velocity"][2]

    des = np.zeros(17)
    des[0] = p["ab_y"] + p["k_abp"] * pitch
    des[1] = p["k_abz"] * fb_z
    des[2] = p["k_abx"] * roll + p["ab_x_amp"] * ramp * np.sin(2 * np.pi * p["f"] * t + p["ab_x_ph"])
    for side, (ihx, ihz, ihy, ikn, ish1) in enumerate(((3, 4, 5, 6, 11), (7, 8, 9, 10, 14))):
        ph = 2 * np.pi * p["f"] * t + np.pi * side
        s = 1.0 if side == 0 else -1.0
        des[ihy] = p["hy_off"] + ramp * p["hy_amp"] * np.sin(ph + p["hy_ph"]) + fb_p
        des[ikn] = p["kn_off"] + ramp * p["kn_amp"] * np.sin(ph + p["kn_ph"])
        des[ihx] = p["hx_off"] + ramp * p["hx_amp"] * np.sin(ph + p["hx_ph"]) + s * fb_r
        des[ihz] = p["hz_off"] + s * fb_z
        des[ish1] = ramp * p["arm_amp"] * np.sin(ph + np.pi) * (1 if side == 0 else -1)
    kp = np.array([p["kp_ab"]] * 3 + [p["kp_hx"], p["kp_hx"], p["kp_hy"], p["kp_kn"]] * 2 + [0.5] * 6)
    kd = np.array([p["kd_ab"]] * 3 + [p["kd_hx"], p["kd_hx"], p["kd_hy"], p["kd_kn"]] * 2 + [0.02] * 6)
    u = kp * (des - q) - kd * qd
    return np.clip(u, -0.4, 0.4)
