"""Walking controller for the MuJoCo Humanoid climbing 5 cm stairs (stairs_up_5): SIMBICON-style torso control
with Raibert/capture-point foot placement, swing foot placed by inverse kinematics.

Control law. Every motor gets tau = kp (q_d - q) - kd dq, sent as tau / gear and clipped to +-0.4. Each joint's
gains respect the 15 ms sample time (kd dt / I_reflected < ~0.5). Above that, the low-inertia joints chatter:
hip_z (leg axial rotation), arms and abdomen.
  * Forward kinematics of a copy of the plant (hl_control's humanoid.xml fed with gym_obs' qpos) gives hips, feet,
    mass centre and pelvis orientation each step.
  * Stance/swing state machine. Roles swap at swing-foot touchdown after T_min (or at T_max).
  * Torso: a virtual PD torque (pitch -> pitch_d with kp_t/kd_t, pelvis roll -> 0 with kp_r/kd_r) is produced by the
    STANCE hip: tau_stance = -tau_torso - c_sw * tau_swing (SIMBICON). abdomen_x leans the trunk by `lean`.
  * Stance knee: driven into its extension limit (knee_st, kp_ks, low kd_ks), so the stance leg is a locked strut.
    The 3 ms physics handles the limit, which avoids the discrete-time instability of a PD on a loaded straight knee.
    A bent landing on a higher tread is straightened by the same PD, which lifts the body up the riser.
  * Swing foot target relative to the mass centre (world axes; side = +1 when the left leg swings):
        x = kx v + kv (v - v_des) + x_off - side kpsi yaw     (Raibert placement + yaw steering)
        y = side w + ky vy                                    (lateral capture)
    Height above the stance foot: raised to h_clear (clears the 5 cm risers) during the first s_up of the swing.
    Held there until the foot is within ex_tol of the x target (after s_down). Then lowered at v_down until
    touchdown. Two-link inverse kinematics in the pelvis frame converts the target to hip_x, hip_y and knee targets
    (swing PD kp_hx, kp_hy (torque-limited by sw_lim), kp_k).
  * Point feet transmit no yaw torque. Yaw is steered by asymmetric fore-aft placement (kpsi): a right foot placed
    further forward brakes in right stance and turns the body right.
  * Arms and abdomen y/z are held near zero with gains limited by the sample time.
Gains were set by hand from the analysis above, then refined by random search (flat ground, then stairs_up_5 seeds).
Status (2026-11-24): stays on its feet for the full 15 s on stairs_up_5 seeds 0-4 (alive rung), but steps nearly in
place (x ~ -0.8 m), so it never reaches the risers. With x_off=-0.02, v_des=0.7 it walks about 5 m on flat ground, but on the
stairs it falls at the first risers: short steps land on the 2.5 cm riser ramps and the stance foot slides off.
"""

import numpy as np
import mujoco
from hl_control.envs.humanoid import XML_PATH

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)

PARAMS = {
    "T_sw": 0.35, "T_min": 0.20, "T_max": 0.60,
    "kx": 0.20, "kv": 0.15, "v_des": 0.4, "x_off": 0.0,
    "w": 0.10, "ky": 0.25,
    "h_clear": 0.12, "s_up": 0.3, "s_down": 0.65, "z_end": -0.04, "v_down": 0.6, "ex_tol": 0.04,
    "knee_st": -0.15,
    "pitch_d": 0.0,
    "kp_t": 400.0, "kd_t": 40.0, "kp_r": 300.0, "kd_r": 30.0, "kp_y": 100.0, "kd_y": 10.0,
    "kp_hy": 250.0, "kd_hy": 15.0, "sw_lim": 100.0, "kp_hx": 200.0, "kd_hx": 10.0, "kp_hz": 30.0, "kd_hz": 1.0,
    "kp_k": 300.0, "kd_k": 6.0, "kp_ks": 300.0, "kd_ks": 1.0, "kp_ab": 300.0, "kd_ab": 30.0,
    "T_stand": 0.25, "lean": 0.0, "pel": 0.0, "c_sw": 1.0, "ff_x": 0.0, "swx_lim": 40.0, "zal": 0.0, "zc": 0.0, "kpsi": 0.0, "kpsid": 0.0, "early": 0.0, "omega": 3.3, "x_desc": 0.0, "T_ds": 0.08, "yaw_ss": 0.0, "kp_ds": 200.0, "kd_ds": 20.0,
}
# tuned values (random search on flat ground, 2026-11-24)
PARAMS.update({"h_clear": 0.06, "v_des": 0.3, "T_min": 0.08, "knee_st": 0.0, "kd_ks": 1, "kp_ks": 300, "s_up": 0.3, "s_down": 0.5, "ex_tol": 0.1, "v_down": 1.0, "kp_k": 400, "T_max": 0.357, "w": 0.117, "ky": 0.074, "kp_hy": 400, "sw_lim": 52.705, "kd_t": 8.256, "kd_r": 14.196, "kd_hy": 10, "kd_k": 6, "T_sw": 0.277, "kp_t": 255.821, "c_sw": 0.416, "T_ds": 0.0, "zal": 0, "kpsi": 0.162, "kpsid": 0, "kx": 0.077, "kv": 0.021, "kp_r": 325.365, "kp_hx": 393.79, "lean": -0.098, "pitch_d": 0.065})

_M = mujoco.MjModel.from_xml_path(str(XML_PATH))
_D = mujoco.MjData(_M)
_MASS = _M.body_mass.copy()
_MTOT = _MASS.sum()
_PELVIS = _M.body("pelvis").id
_THIGH = (_M.body("right_thigh").id, _M.body("left_thigh").id)
_FOOT = (_M.geom("right_foot").id, _M.geom("left_foot").id)
L1, L2 = 0.403, 0.35
_HX_LO, _HX_HI = np.radians(-25.0), np.radians(5.0)

# joint-space gains limited by the 15 ms sample time (kd dt / I < ~0.5 for each joint's reflected inertia)
KP_AB = np.array([150.0, 60.0, 150.0])   # abdomen y, z, x
KD_AB = np.array([8.0, 2.0, 6.0])
KP_ARM = np.array([15.0, 15.0, 5.0, 15.0, 15.0, 5.0])
KD_ARM = np.array([0.5, 0.5, 0.2, 0.5, 0.5, 0.2])

_S = {"stance": None, "t": 0.0, "last_step": -1, "p0": None}


def _reset():
    _S.update(stance=None, t=0.0, last_step=-1, p0=None)


def _ik(d, right):
    """Foot-centre offset d from the hip in the pelvis frame -> (hip_x, hip_y, knee)."""
    X, Y, Z = d
    Z = min(Z, -0.2)
    qx = np.arctan2(Y, -Z)
    qx = float(np.clip(qx if right else -qx, _HX_LO, _HX_HI))
    Zp = -np.hypot(Y, Z)
    D = float(np.clip(np.hypot(X, Zp), 0.25, L1 + L2 - 0.003))
    ck = (D * D - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    kap = float(np.arccos(np.clip(ck, -1.0, 1.0)))
    phi = np.arctan2(-X, -Zp)
    cdl = (L1 * L1 + D * D - L2 * L2) / (2 * L1 * D)
    dl = float(np.arccos(np.clip(cdl, -1.0, 1.0)))
    return qx, phi - dl, -kap


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    step = int(obs["step"])
    if step == 0 or step <= _S["last_step"]:
        _reset()
    _S["last_step"] = step
    dt = float(obs["dt"])
    q = np.asarray(obs["joint_angles"], float)
    dq = np.asarray(obs["joint_rates"], float)

    # forward kinematics of the plant copy
    _D.qpos[0:2] = 0.0
    _D.qpos[2:] = np.asarray(obs["gym_obs"], float)[:22]
    mujoco.mj_kinematics(_M, _D)
    com = _MASS @ _D.xipos / _MTOT
    Rp = _D.xmat[_PELVIS].reshape(3, 3)
    hips = (_D.xpos[_THIGH[0]].copy(), _D.xpos[_THIGH[1]].copy())
    feet = (_D.geom_xpos[_FOOT[0]].copy(), _D.geom_xpos[_FOOT[1]].copy())

    yaw = float(obs["yaw"])
    c, s = np.cos(yaw), np.sin(yaw)
    Rh = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # world -> heading
    up = Rh @ np.asarray(obs["up_vector"], float)
    th = np.arctan2(up[0], up[2])
    ph = -np.arctan2(up[1], up[2])
    w = Rh @ np.asarray(obs["root_angular_velocity"], float)
    thd, phd, psd = w[1], w[0], w[2]
    vx, vy = float(obs["com_velocity_xy"][0]), float(obs["com_velocity_xy"][1])
    cR, cL = bool(obs["foot_contact"]), bool(obs["foot_left_contact"])

    upp = Rh @ Rp[:, 2]
    ph_pel = -np.arctan2(upp[1], upp[2])
    th_pel = np.arctan2(upp[0], upp[2])
    if P["pel"] > 0:
        tau_p = P["kp_t"] * (P["pitch_d"] - th_pel) - P["kd_t"] * (thd + dq[0])
    else:
        tau_p = P["kp_t"] * (P["pitch_d"] - th) - P["kd_t"] * thd
    tau_r = P["kp_r"] * (0.0 - ph_pel) - P["kd_r"] * (phd + dq[2])
    tau_y = P["kp_y"] * (0.0 - yaw) - P["kd_y"] * psd

    tau = np.zeros(17)
    tau[0:3] = KP_AB * (0.0 - q[0:3]) - KD_AB * dq[0:3]
    tau[11:17] = KP_ARM * (0.0 - q[11:17]) - KD_ARM * dq[11:17]

    t = obs["time"]
    if _S["stance"] is None:
        for (ix, iz, iy, ik) in ((3, 4, 5, 6), (7, 8, 9, 10)):
            tau[ik] = P["kp_ks"] * (P["knee_st"] - q[ik]) - P["kd_ks"] * dq[ik]
            tau[iy] = -0.5 * tau_p
            tau[ix] = P["kp_hx"] * (0.0 - q[ix]) - P["kd_hx"] * dq[ix]
            tau[iz] = P["kp_hz"] * (0.0 - q[iz]) - P["kd_hz"] * dq[iz]
        if t >= P["T_stand"]:
            _S["stance"] = "R" if abs(feet[0][1] - com[1]) < abs(feet[1][1] - com[1]) else "L"
            _S["t"] = 0.0
            _S["p0"] = None
        return np.clip(tau / GEAR, -0.4, 0.4)

    _S["t"] += dt
    st = _S["stance"]
    tp = _S["t"]
    sw_contact = cL if st == "R" else cR
    if (tp > P["T_min"] + P["T_ds"] and sw_contact) or tp > P["T_max"] + P["T_ds"]:
        st = "L" if st == "R" else "R"
        _S["stance"] = st
        _S["t"] = 0.0
        _S["p0"] = None
        _S["desc"] = None
        tp = 0.0

    if tp < P["T_ds"]:
        # double support: both legs stance-like, torso + yaw torques shared by both hips
        for (ix, iz, iy, ik) in ((3, 4, 5, 6), (7, 8, 9, 10)):
            tau[ik] = P["kp_ks"] * (P["knee_st"] - q[ik]) - P["kd_ks"] * dq[ik]
            tau[iy] = -0.5 * tau_p
        tau[3] = -0.5 * tau_r
        tau[7] = 0.5 * tau_r
        tau[4] = -0.5 * tau_y
        tau[8] = 0.5 * tau_y
        _S["p0"] = None
        return np.clip(tau / GEAR, -0.4, 0.4)
    tp = tp - P["T_ds"]

    si, wi = (0, 1) if st == "R" else (1, 0)
    lam = P["lean"] if st == "R" else -P["lean"]   # torso roll target (+ = towards -y, the right foot)
    tau[2] = KP_AB[2] * (-lam - q[2]) - KD_AB[2] * dq[2]
    S = (3, 4, 5, 6) if si == 0 else (7, 8, 9, 10)
    W = (3, 4, 5, 6) if wi == 0 else (7, 8, 9, 10)
    side = 1.0 if wi == 1 else -1.0  # swing foot side (+y for left)
    fs, fw, hw = feet[si], feet[wi], hips[wi]
    if _S["p0"] is None:
        _S["p0"] = (fw - com)[:2].copy()

    # swing foot target (world axes, relative to the mass centre / stance foot height)
    sph = tp / P["T_sw"]
    xt = P["kx"] * vx + P["kv"] * (vx - P["v_des"]) + P["x_off"]
    # yaw steering: a right foot placed further forward brakes in right stance (yaw moment -), the left one propels
    xt += -side * (P["kpsi"] * yaw + P["kpsid"] * psd)
    yt = side * P["w"] + P["ky"] * vy
    a = float(np.clip(sph / P["s_down"], 0.0, 1.0))
    a = a * a * (3 - 2 * a)
    pxy = _S["p0"] + (np.array([xt, yt]) - _S["p0"]) * a
    ex = (xt - (fw - com)[0])
    if _S.get("desc") is None and sph >= P["s_down"] and (ex < P["ex_tol"] or sph >= 1.0):
        _S["desc"] = tp
    # lateral emergency: the mass centre heads out over the stance foot -> put the swing foot down now,
    # so that the other leg (on the side the body is falling to) becomes the swing leg
    ys = (com - fs)[1]
    if _S.get("desc") is None and P["early"] > 0 and tp > 0.05 and (-side) * (ys * P["omega"] + vy) > P["early"]:
        _S["desc"] = tp
    if _S.get("desc") is None:
        hz = P["h_clear"] * np.sin(0.5 * np.pi * min(1.0, sph / P["s_up"]))
    else:
        hz = P["h_clear"] - P["v_down"] * (tp - _S["desc"])
    if _S.get("desc") is not None:
        pxy = pxy + np.array([P["x_desc"], 0.0])   # keep the foot moving forward while it comes down (riser edges)
    target = np.array([com[0] + pxy[0], com[1] + pxy[1], fs[2] + hz])
    d = Rp.T @ (target - hw)
    # thigh yaw relative to the pelvis (about +z): right hip_z axis +z, left -z
    qz = q[W[1]] if wi == 0 else -q[W[1]]
    cz, sz = np.cos(qz), np.sin(qz)
    d = np.array([cz * d[0] + sz * d[1], -sz * d[0] + cz * d[1], d[2]])
    qx_d, qy_d, qk_d = _ik(d, wi == 0)
    psi_p = float(np.arctan2(Rp[1, 0], Rp[0, 0]))
    qz_d = float(np.clip(P["zal"] * (-psi_p if wi == 0 else psi_p), -0.9, 0.5))

    tau[W[0]] = np.clip(P["kp_hx"] * (qx_d - q[W[0]]) - P["kd_hx"] * dq[W[0]], -P["swx_lim"], P["swx_lim"])
    tau[W[1]] = P["kp_hz"] * (qz_d - q[W[1]]) - P["kd_hz"] * dq[W[1]]
    tau[W[2]] = np.clip(P["kp_hy"] * (qy_d - q[W[2]]) - P["kd_hy"] * dq[W[2]], -P["sw_lim"], P["sw_lim"])
    tau[W[3]] = P["kp_k"] * (qk_d - q[W[3]]) - P["kd_k"] * dq[W[3]]

    # stance leg
    tau[S[3]] = P["kp_ks"] * (P["knee_st"] - q[S[3]]) - P["kd_ks"] * dq[S[3]]
    tau[S[2]] = -tau_p - P["c_sw"] * tau[W[2]]
    if st == "R":
        tau[S[0]] = tau[W[0]] - tau_r + P["ff_x"]
        tau[S[1]] = P["zc"] * tau[W[1]] - P["yaw_ss"] * tau_y
    else:
        tau[S[0]] = tau_r + tau[W[0]] + P["ff_x"]
        tau[S[1]] = P["yaw_ss"] * tau_y + P["zc"] * tau[W[1]]

    _S["dbg"] = (target - com, fw - com)
    return np.clip(tau / GEAR, -0.4, 0.4)
