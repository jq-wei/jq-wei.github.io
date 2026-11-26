"""Foot-placement walking controller for the MuJoCo Humanoid (hand-designed structure,
gains found by a black-box random search over PARAMS; no neural network, no gradients).

Every joint command is a PD torque divided by the motor gear, clipped to [-0.4, 0.4].

* Gait state machine: one leg in stance, the other swinging; the roles swap when the
  swing foot touches down after ``t_min`` s of swing, or ``t_sw + 0.1`` s at the latest.
* Torso balance (SIMBICON stance-hip rule): a virtual torque on the pelvis,
  tau = -kp (angle - target) - kd rate, for pitch (``kp_t``/``kd_t``, target ``lean``,
  which sets the forward drive) and roll (``kp_roll``/``kd_roll``, target 0), is
  produced by the stance hip after subtracting what the swing hip applies to the pelvis.
* Foot placement (balance and speed, Raibert / capture-point style, world frame): the
  swing foot is steered to, relative to the mass centre,
      x_d = px_v * v_x + px_k * (v_x - v_des) + px_0
      y_d = side * width + py_v * v_y
  reaching x_d at ``x_frac`` of ``t_sw``, lifted by ``lift`` and pushed ``z_push`` below
  the floor after ``z_frac`` of ``t_sw`` to force touchdown. Targets are converted to
  hip/knee angles by two-link inverse kinematics in the pelvis frame (hip_x capped at
  ``hx_max``, legs cannot cross) and tracked by PD (``kp_sw``/``kd_sw`` hip pitch,
  ``kp_swx``/``kd_swx`` hip roll, ``kp_knee``/``kd_knee`` knee).
* Stance knee held at ``knee_st`` (``kp_kst``/``kd_kst``); hip yaw joints softly held at 0
  (``kp_z``/``kd_z``; the spherical feet have no torsional friction, so the stance hip
  cannot steer the heading).
* Heading: the abdomen yaw joint adds ``k_ayaw * yaw + d_ayaw * yaw_rate`` to its PD
  (``kp_abd``/``kd_abd``), twisting the pelvis/legs relative to the torso so the gait
  stays pointed along +x.
* Arms held by a light joint PD.
"""

import numpy as np

PARAMS = {
    "yaw_frame": 0.0,  # 0: foot targets in the world frame, 1: in the heading frame
    "t_min": 0.15, "t_sw": 0.3304,
    "lean": 0.07528, "kp_t": 137.4, "kd_t": 11.55, "kp_roll": 277.1, "kd_roll": 11.1,
    "px_v": 0.06405, "px_k": 0.05852, "px_0": -0.02586, "v_des": 1.055, "width": 0.08779, "py_v": 0.1868,
    "lift": 0.2929, "x_frac": 0.8594, "z_frac": 0.8805, "z_push": 0.1695, "hx_max": 0.02894,
    "kp_sw": 259.2, "kd_sw": 11.77, "kp_swx": 679.4, "kd_swx": 9.04, "kp_knee": 170.7, "kd_knee": 5.784,
    "knee_st": -0.01383, "kp_kst": 203.8, "kd_kst": 3.733, "kp_z": 32.66, "kd_z": 0.9075,
    "k_ayaw": -50.0, "d_ayaw": -3.5, "kp_abd": 123.7, "kd_abd": 2.424,
}

J = {n: i for i, n in enumerate((
    "abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
    "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
    "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"))}
LEGS = ("right", "left")
SX = {"right": 1.0, "left": -1.0}   # hip_x / hip_z axis sign (world x / z) per leg
OUT = {"right": -1.0, "left": 1.0}  # outward lateral direction per leg
L1, L2, R_FOOT = 0.383, 0.37, 0.075

_S = {"stance": 0, "t": 0.0, "p0": np.zeros(2)}


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _leg_fk(q, leg):
    """Foot centre relative to the hip joint, pelvis frame."""
    sx = SX[leg]
    R = _rx(sx * q[J[f"{leg}_hip_x"]]) @ _rz(sx * q[J[f"{leg}_hip_z"]]) @ _ry(q[J[f"{leg}_hip_y"]])
    return R @ (np.array([0, 0, -L1]) + _ry(-q[J[f"{leg}_knee"]]) @ np.array([0, 0, -L2]))


def _leg_ik(p, leg):
    """Hip x, hip y, knee angles placing the foot centre at p (pelvis frame, from the hip)."""
    x, y, z = p
    hx = np.arctan2(y, -z)
    rp = np.hypot(y, z)
    r = np.clip(np.hypot(x, rp), 0.3, L1 + L2 - 0.005)
    kb = np.arccos(np.clip((r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2), -1, 1))
    beta = np.arctan2(L2 * np.sin(kb), L1 + L2 * np.cos(kb))
    alpha = np.arctan2(x, rp)
    return SX[leg] * hx, -(alpha + beta), -kb


def _pd(kp, kd, qd, q, dq):
    return kp * (qd - q) - kd * dq


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    dt = obs["dt"]
    q, dq = obs["joint_angles"], obs["joint_rates"]
    tau = np.zeros(17)
    contact = {"right": obs["foot_contact"], "left": obs["foot_left_contact"]}
    foot_rel = {"right": np.asarray(obs["foot_xyz_rel"]), "left": np.asarray(obs["foot_left_xyz_rel"])}

    yaw = obs["yaw"]
    c, s = np.cos(P["yaw_frame"] * yaw), np.sin(P["yaw_frame"] * yaw)
    Rh = np.array([[c, s], [-s, c]])  # world -> placement frame (identity when yaw_frame = 0)
    if obs["step"] == 0:
        _S["stance"], _S["t"] = 0, 0.0
        _S["p0"] = Rh @ foot_rel["left"][:2]

    # state machine
    _S["t"] += dt
    st, sw = LEGS[_S["stance"]], LEGS[1 - _S["stance"]]
    if (_S["t"] > P["t_min"] and contact[sw]) or _S["t"] > P["t_sw"] + 0.1:
        _S["stance"] = 1 - _S["stance"]
        _S["t"] = 0.0
        st, sw = sw, st
        _S["p0"] = Rh @ foot_rel[sw][:2]
    sph = min(_S["t"] / P["t_sw"], 1.0)

    v = Rh @ obs["com_velocity_xy"]
    w = obs["root_angular_velocity"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    wh = np.array([cy * w[0] + sy * w[1], -sy * w[0] + cy * w[1], w[2]])
    pitch, roll = obs["pitch"], obs["roll"]
    ay, az, ax = q[J["abdomen_y"]], q[J["abdomen_z"]], q[J["abdomen_x"]]
    pitch_p = pitch + ay
    roll_p = roll + ax
    Rp = _rz(yaw) @ _ry(pitch) @ _rx(roll) @ _rz(az) @ _ry(ay) @ _rx(ax)  # pelvis -> world

    # abdomen, arms
    for n in ("abdomen_y", "abdomen_z", "abdomen_x"):
        tau[J[n]] = _pd(P["kp_abd"], P["kd_abd"], 0.0, q[J[n]], dq[J[n]])
    tau[J["abdomen_z"]] += P["k_ayaw"] * yaw + P["d_ayaw"] * w[2]  # turn the torso back to +x
    for n in ("right_shoulder1", "right_shoulder2", "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow"):
        tau[J[n]] = _pd(20.0, 2.0, 0.0, q[J[n]], dq[J[n]])

    # swing foot target relative to the mass centre (placement frame)
    x_d = P["px_v"] * v[0] + P["px_k"] * (v[0] - P["v_des"]) + P["px_0"]
    y_d = OUT[sw] * P["width"] + P["py_v"] * v[1]
    sx = min(sph / P["x_frac"], 1.0)
    sx = sx * sx * (3 - 2 * sx)
    tgt_h = _S["p0"] + (np.array([x_d, y_d]) - _S["p0"]) * sx
    clear = P["lift"] * np.sin(np.pi * sph / P["z_frac"]) if sph < P["z_frac"] else -P["z_push"]
    com_z = obs["com_xyz"][2]
    # swing hip position relative to the mass centre (world), from the measured foot and FK
    hip_rel = foot_rel[sw] - Rp @ _leg_fk(q, sw)
    tgt_w = np.array([*(Rh.T @ tgt_h), R_FOOT + clear - com_z])
    hx_d, hy_d, k_d = _leg_ik(Rp.T @ (tgt_w - hip_rel), sw)
    hx_d = np.clip(hx_d, -0.43, P["hx_max"])
    iy, ix, iz, ik = (J[f"{sw}_hip_y"], J[f"{sw}_hip_x"], J[f"{sw}_hip_z"], J[f"{sw}_knee"])
    tau[iy] = _pd(P["kp_sw"], P["kd_sw"], hy_d, q[iy], dq[iy])
    tau[ix] = _pd(P["kp_swx"], P["kd_swx"], hx_d, q[ix], dq[ix])
    tau[iz] = _pd(P["kp_z"], P["kd_z"], 0.0, q[iz], dq[iz])
    tau[ik] = _pd(P["kp_knee"], P["kd_knee"], k_d, q[ik], dq[ik])

    # stance leg: torso balance torques
    tp = -P["kp_t"] * (pitch_p - P["lean"]) - P["kd_t"] * (wh[1] + dq[J["abdomen_y"]])
    tr = -P["kp_roll"] * roll_p - P["kd_roll"] * (wh[0] + dq[J["abdomen_x"]])
    jy, jx, jz, jk = (J[f"{st}_hip_y"], J[f"{st}_hip_x"], J[f"{st}_hip_z"], J[f"{st}_knee"])
    tau[jy] = -tp - tau[iy]
    tau[jx] = -SX[st] * (tr + SX[sw] * tau[ix])
    tau[jz] = _pd(P["kp_z"], P["kd_z"], 0.0, q[jz], dq[jz])
    tau[jk] = _pd(P["kp_kst"], P["kd_kst"], P["knee_st"], q[jk], dq[jk])

    return np.clip(tau / obs["gear"], -0.4, 0.4)
