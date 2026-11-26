"""Humanoid gait controller: CPG-driven joint references tracked by joint PD, plus IMU feedback.

Control law
-----------
A clock phase p = 2*pi*f*t drives the two legs in antiphase (right leg p, left leg p+pi).
Each leg's joint references are
  hip_y  = h0 + Ah*sin(p)                      (swing / stance sweep; negative = leg forward)
  knee   = k0 + Ak*max(0, sin(p+kpsi))         (flexion during swing)
  hip_x  = x0 + Ax*sin(p+xpsi)                 (lateral weight shift)
Torque = KP*(ref - q) - KD*qdot per joint group, divided by the motor gear to give the command.
IMU feedback adds to the references: torso pitch/gyro -> hip_y and abdomen_y, roll/gyro -> hip_x and
abdomen_x. Arms hold a fixed pose. PARAMS are tuned by CMA-ES on the visible seeds.
"""
import numpy as np

PARAMS = {
    "f": 1.6353,
    "h0": -0.26916,
    "Ah": 0.59389,
    "k0": -0.32357,
    "Ak": 0.77619,
    "kpsi": -1.0989,
    "x0": -0.21714,
    "Ax": 0.058178,
    "xpsi": 1.0143,
    "a0": -0.096851,
    "kp_hip": 200.24,
    "kp_knee": 112.79,
    "kp_abd": 144.82,
    "kd_ratio": 0.021255,
    "Kph": -0.31038,
    "Kpa": 0.63237,
    "Kgh": 0.090176,
    "Krh": 0.33075,
    "Kra": 1.7311,
    "Kgr": 0.21486,
    "Kpk": -1.211,
    "arm": 0.017145,
    "armsw": 0.20412,
    "Az": -0.11751,
}

_RIGHT = dict(x=3, z=4, y=5, k=6)
_LEFT = dict(x=7, z=8, y=9, k=10)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    gear = obs["gear"]
    t = obs["time"]
    pitch = obs["pitch"]
    roll = obs["roll"]
    gyro = obs["gyro_body"]
    p = 2 * np.pi * P["f"] * t
    ref = np.zeros(17)
    # ramp the gait in over the first second
    ramp = min(1.0, t / 0.8)
    ref[0] = P["a0"] + P["Kpa"] * pitch + 0.0
    ref[2] = P["Kra"] * roll + P["Kgr"] * gyro[0]
    ref[1] = 0.0
    for side, ph, sgn in ((_RIGHT, p, 1.0), (_LEFT, p + np.pi, -1.0)):
        s = np.sin(ph)
        ref[side["y"]] = P["h0"] + ramp * P["Ah"] * s + P["Kph"] * pitch + P["Kgh"] * gyro[1]
        ref[side["k"]] = P["k0"] + ramp * P["Ak"] * max(0.0, np.sin(ph + P["kpsi"])) + P["Kpk"] * pitch
        ref[side["x"]] = P["x0"] + ramp * P["Ax"] * np.sin(ph + P["xpsi"]) + sgn * P["Krh"] * roll * 0 + P["Krh"] * roll
        ref[side["z"]] = sgn * P["Az"] * np.sin(ph) * ramp
    ref[11] = ref[14] = P["arm"]
    ref[11] += P["armsw"] * np.sin(p + np.pi)
    ref[14] += P["armsw"] * np.sin(p)
    kp = np.zeros(17)
    kp[0:3] = P["kp_abd"]
    kp[3:6] = P["kp_hip"]; kp[7:10] = P["kp_hip"]
    kp[6] = kp[10] = P["kp_knee"]
    kp[11:] = 10.0
    kd = kp * P["kd_ratio"]
    tau = kp * (ref - q) - kd * qd
    return np.clip(tau / gear, -0.4, 0.4)
