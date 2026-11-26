"""Humanoid walking controller: open-loop phase oscillator + joint PD + state feedback.

Gait: a clock phase th = 2*pi*FREQ*t drives right-leg joint targets as low-order Fourier series
(hip_y, knee, hip_x, hip_z); the left leg runs the same series half a cycle later (the XML's mirrored
joint axes make equal angles mirror-symmetric). The amplitude ramps in over the first RAMP seconds.
Feedback offsets on the targets stabilise the torso: pitch / pitch-rate / forward-speed error into
hip_y and abdomen_y, roll / roll-rate / lateral speed into hip_x (odd between legs) and abdomen_x,
yaw / yaw-rate into hip_z (odd) and abdomen_z. Joint torques come from PD laws
tau = kp (q* - q) - kd qdot, divided by the motor gear to give the command.
All numbers live in THETA (see NAMES), tuned by derivative-free search (opt.py, CMA-ES).
"""

import numpy as np

NAMES = [
    "freq", "ramp", "vdes",
    "H0", "H1", "H2", "K0", "K1", "K2", "K3", "K4", "X0", "X1", "X2", "Z0", "Z1", "Z2",
    "A0", "B0", "B1", "B2", "C1", "C2", "SW", "EL",
    "hy_p", "hy_d", "hy_v", "ay_p", "ay_d", "ay_v",
    "hx_p", "hx_d", "hx_v", "ax_p", "ax_d",
    "hz_p", "hz_d", "az_p", "az_d",
    "kp_hy", "kp_kn", "kp_hxz", "kp_ab", "kd_leg", "kd_ab",
]
THETA = [1.1326, 1.5779, 0.58054, -0.22714, 0.001215, 0.0038569, -0.28131, 0.024352, -0.0062811, 0.00027639, -0.014564, -0.40509, -0.0096835, -0.046205, -0.06427, 0.25424, -0.065412, 0.41275, -0.0016113, -0.074749, -0.02453, -0.039124, 0.039534, 0.66629, -0.098114, 0.76082, 0.14992, -0.34661, 2.6665, -0.088822, 1.3442, -0.29143, 0.31441, 0.079189, 1.7166, 0.24786, -0.23172, 0.39254, 0.6506, 0.042121, 277.9, 135.15, 38.194, 61.315, 4.8903, 5.003]
PARAMS = dict(zip(NAMES, THETA))  # informational only

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25.0])


def _leg(p, th):
    """Targets (hip_x, hip_z, hip_y, knee) of one leg at phase th."""
    s, c, s2, c2 = np.sin(th), np.cos(th), np.sin(2 * th), np.cos(2 * th)
    hy = p["H0"] + p["H1"] * s + p["H2"] * c
    kn = p["K0"] + p["K1"] * s + p["K2"] * c + p["K3"] * s2 + p["K4"] * c2
    hx = p["X0"] + p["X1"] * s + p["X2"] * c
    hz = p["Z0"] + p["Z1"] * s + p["Z2"] * c
    return hx, hz, hy, kn


def controller(obs: dict):
    p = dict(zip(NAMES, THETA))
    t = obs["time"]
    ramp = min(1.0, t / max(p["ramp"], 1e-3))
    th = 2 * np.pi * p["freq"] * t
    q, qd = obs["joint_angles"], obs["joint_rates"]
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    w = obs["root_angular_velocity"]
    vx, vy = obs["com_velocity_xy"]
    # smooth velocity estimate: use com velocity from the root (less noisy than per-step difference)
    v = obs["root_velocity"]
    vx, vy = v[0], v[1]

    tgt = np.zeros(17)
    Rx, Rz, Ry, Rk = _leg(p, th)
    Lx, Lz, Ly, Lk = _leg(p, th + np.pi)
    # gait terms ramp in; offsets (H0, K0, X0, Z0) are posture and apply from the start
    def ramped(val, off):
        return off + ramp * (val - off)
    Rx, Lx = ramped(Rx, p["X0"]), ramped(Lx, p["X0"])
    Rz, Lz = ramped(Rz, p["Z0"]), ramped(Lz, p["Z0"])
    Ry, Ly = ramped(Ry, p["H0"]), ramped(Ly, p["H0"])
    Rk, Lk = ramped(Rk, p["K0"]), ramped(Lk, p["K0"])

    ev = vx - p["vdes"]
    hy_fb = p["hy_p"] * pitch + p["hy_d"] * w[1] + p["hy_v"] * ev
    ay_fb = p["ay_p"] * pitch + p["ay_d"] * w[1] + p["ay_v"] * ev
    hx_fb = p["hx_p"] * roll + p["hx_d"] * w[0] + p["hx_v"] * vy
    ax_fb = p["ax_p"] * roll + p["ax_d"] * w[0]
    hz_fb = p["hz_p"] * yaw + p["hz_d"] * w[2]
    az_fb = p["az_p"] * yaw + p["az_d"] * w[2]

    tgt[0] = p["A0"] + ay_fb
    tgt[1] = ramp * (p["C1"] * np.sin(th) + p["C2"] * np.cos(th)) + az_fb
    tgt[2] = p["B0"] + ramp * (p["B1"] * np.sin(th) + p["B2"] * np.cos(th)) + ax_fb
    tgt[3], tgt[4], tgt[5], tgt[6] = Rx + hx_fb, Rz + hz_fb, Ry + hy_fb, Rk
    tgt[7], tgt[8], tgt[9], tgt[10] = Lx - hx_fb, Lz - hz_fb, Ly + hy_fb, Lk
    # arms: swing opposite to the same-side leg, elbows held
    tgt[11] = ramp * p["SW"] * np.sin(th + np.pi)
    tgt[14] = ramp * p["SW"] * np.sin(th)
    tgt[13] = tgt[16] = p["EL"]

    kp = np.array([p["kp_ab"]] * 3 + [p["kp_hxz"], p["kp_hxz"], p["kp_hy"], p["kp_kn"]] * 2 + [10.0] * 6)
    kd = np.array([p["kd_ab"]] * 3 + [p["kd_leg"]] * 8 + [1.0] * 6)
    kp[[3, 4, 7, 8]] = p["kp_hxz"]
    tau = kp * (tgt - q) - kd * qd
    return np.clip(tau / GEAR, -0.4, 0.4)
