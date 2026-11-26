"""Humanoid walking controller: SIMBICON-style state machine with model-based swing-foot placement.

No learning: a private copy of the MuJoCo model (hl_control's humanoid.xml) is set to the
measured state (gym_obs = qpos[2:], qvel) every step to get foot positions, joint axes,
Jacobians and gravity/bias torques. All gains below are in N m (or N) and divided by the gear.

Control law
* Settle (t < t_settle): both legs hold the torso upright, knees at knee_st.
* Gait: one leg in stance, the other swings; the swing ends on swing-foot touchdown once
  phase > ph_min and the foot is within td_tol of its target, or at phase ph_max (phase = time/T).
* Swing foot (Raibert/capture-point placement, relative to the mass centre, in a heading
  frame rotated by hfrac*yaw):  x = x_off + kx*vx + kvdes*(vx - vdes),
  y = +-w + kvy*(vy - hsteer*vdes_y).  The foot follows a smoothstep from lift-off to the
  target (reached at phase s_end) by Jacobian-transpose force kp_f/kd_f (lateral x kfy), a weak
  vertical term (fz_on, kp_fz, kd_fz, clearance `clear`) and a knee flex/extend profile
  (knee_sw until phase ph_k, then knee_ext; kp_k, kd_k).  Swing hip_z holds the leg along
  the world heading (kp_z, kd_z, yaw_ff).
* Stance leg: hip joints apply a world-frame torso attitude torque (lean pitch_d; kp_t, kd_t;
  yaw kp_yaw, kd_yaw) minus the swing-hip reaction (SIMBICON); stance knee PD to knee_st.
* Abdomen PD to zero (kp_ab, kd_ab); arms PD to rest (kp_arm, kd_arm) plus a yaw-correcting
  reaction torque (ka_yaw, kda_yaw).
Gains were chosen by random search on visible seeds (standard_v0 and push_v1).
"""
import numpy as np, mujoco, os
import hl_control.envs.humanoid as _H

_m = mujoco.MjModel.from_xml_path(str(_H.XML_PATH))
_d = mujoco.MjData(_m)
MOTOR = _H.ACTUATED_JOINTS
GEAR = _m.actuator_gear[:, 0].copy()
DOF = {n: int(_m.jnt_dofadr[_m.joint(n).id]) for n in MOTOR}
JID = {n: _m.joint(n).id for n in MOTOR}
MIDX = {n: i for i, n in enumerate(MOTOR)}
LEG = {s: [f"{s}_hip_x", f"{s}_hip_z", f"{s}_hip_y", f"{s}_knee"] for s in ("right", "left")}
FOOT = {s: _m.geom(f"{s}_foot").id for s in ("right", "left")}

P = dict(
    T=0.40, pitch_d=0.05,
    kp_t=300., kd_t=30., kp_yaw=60., kd_yaw=6.,
    kx=0.30, kvdes=0.15, vdes=0.4, x_off=0.0,
    w=0.11, kvy=0.25, clear=0.10,
    kp_f=2000., kd_f=80., kp_fz=600., kd_fz=30.,
    knee_st=-0.15, hfrac=0.0, hsteer=0.0, td_tol=1.0, kfy=1.0, s_end=0.7, fz_on=0.0, fz_knee=0.0, ka_yaw=0.0, kda_yaw=0.0, t_settle=0.3, ph_max=1.3, ph_min=0.45, kp_z=60., kd_z=4., yaw_ff=1.0, knee_sw=-1.1, knee_ext=-0.2, ph_k=0.55, kp_k=200., kd_k=10., kp_ks=300., kd_ks=15.,
    kp_ab=200., kd_ab=15., kp_arm=20., kd_arm=2.,
)
BOUNDS = dict(T=(0.25, 0.6), pitch_d=(-0.1, 0.25), kp_t=(50, 600), kd_t=(2, 60), kp_yaw=(0, 150), kd_yaw=(0, 20),
    kx=(0.0, 0.6), kvdes=(0.0, 0.5), vdes=(0.1, 0.45), x_off=(-0.1, 0.15), w=(0.0, 0.2), kvy=(0.0, 0.6), clear=(0.0, 0.2),
    kp_f=(300, 5000), kd_f=(5, 200), knee_st=(-0.6, -0.05), kp_ks=(100, 800), kd_ks=(2, 40), knee_sw=(-1.5, -0.3),
    knee_ext=(-0.6, -0.05), ph_k=(0.3, 0.8), kp_k=(50, 400), kd_k=(2, 30), kp_z=(0, 150), kd_z=(0, 15), yaw_ff=(0, 1.5),
    ph_min=(0.2, 0.9), ph_max=(0.9, 1.6), kp_ab=(50, 400), kd_ab=(2, 40), ka_yaw=(0, 60), kda_yaw=(0, 10), kp_arm=(0, 40), kd_arm=(0, 5), fz_on=(0, 1), hfrac=(0, 1), hsteer=(0, 1), td_tol=(0.03, 0.5), kfy=(0.5, 4.0), s_end=(0.3, 1.0), fz_knee=(0, 1), kp_fz=(200, 3000), kd_fz=(5, 150))
P.update({"T": 0.3582, "pitch_d": -0.0148, "kp_t": 505.6317, "kd_t": 48.6379, "kp_yaw": 0.0, "kd_yaw": 8.5316, "kx": 0.1771, "kvdes": 0.054, "vdes": 0.3732, "x_off": -0.0666, "w": 0.0589, "kvy": 0.2976, "clear": 0.0878, "kp_f": 1834.6469, "kd_f": 195.4507, "knee_st": -0.463, "kp_ks": 146.2001, "kd_ks": 12.2075, "knee_sw": -1.4705, "knee_ext": -0.05, "ph_k": 0.4155, "kp_k": 110.3403, "kd_k": 20.0707, "kp_z": 115.0363, "kd_z": 3.6425, "yaw_ff": 1.0932, "ph_min": 0.3066, "ph_max": 0.9919, "kp_ab": 304.1323, "kd_ab": 17.7251, "ka_yaw": 48.0551, "kda_yaw": 1.5302, "kp_arm": 20.4061, "kd_arm": 2.7818, "fz_on": 0.576, "hfrac": 0.1475, "hsteer": 0.225, "td_tol": 0.5, "kfy": 2.3773, "s_end": 0.4523, "fz_knee": 0.149, "kp_fz": 1044.5361, "kd_fz": 10.3269})
_S = {}

def reset():
    _S.clear()

def controller(obs):
    p = P
    t = obs["time"]
    g = obs["gym_obs"]
    _d.qpos[:2] = 0.0; _d.qpos[2:] = g[:22]; _d.qvel[:] = g[22:45]
    mujoco.mj_forward(_m, _d)
    if obs["step"] == 0 or "st" not in _S:
        _S.update(st="right", t0=t, lift=None)
    q = np.asarray(obs["joint_angles"]); dq = np.asarray(obs["joint_rates"])
    com = _d.subtree_com[0].copy()
    v = np.asarray(obs["com_velocity_xy"])
    st = _S["st"]; sw = "left" if st == "right" else "right"
    ph = (t - _S["t0"]) / p["T"]
    swc = obs["foot_left_contact"] if sw == "left" else obs["foot_contact"]
    if t < p["t_settle"]:
        _S["t0"] = t
        return _stand(obs, q, dq, p)
    if ph >= p["ph_max"] or (ph > p["ph_min"] and swc and _S.get("terr", 0.0) < p["td_tol"]):
        st, sw = sw, st; _S["st"] = st; _S["t0"] = t; ph = 0.0; _S["lift"] = None
    pf = _d.geom_xpos[FOOT[sw]].copy()
    if _S["lift"] is None:
        _S["lift"] = pf[:2] - com[:2]
    tau = np.zeros(17)
    # swing foot target relative to COM
    vx, vy = v
    # placement in a heading frame rotated by hfrac * yaw (keeps step width perpendicular to the legs)
    psi = p["hfrac"] * obs["yaw"]; c, sn = np.cos(psi), np.sin(psi)
    vh = np.array([c * vx + sn * vy, -sn * vx + c * vy])
    vdh = p["vdes"] * np.array([c, -sn])
    side = 1.0 if sw == "left" else -1.0
    xh = p["x_off"] + p["kx"] * vh[0] + p["kvdes"] * (vh[0] - vdh[0])
    yh = side * p["w"] + p["kvy"] * (vh[1] - p["hsteer"] * vdh[1])
    xt, yt = c * xh - sn * yh, sn * xh + c * yh
    s = np.clip(ph / p["s_end"], 0, 1); s = s * s * (3 - 2 * s)
    tgt = (1 - s) * _S["lift"] + s * np.array([xt, yt])
    zt = 0.075 + p["clear"] * np.sin(np.pi * min(ph / 0.8, 1.0)) - (0.03 if ph > 0.8 else 0.0)
    jacp = np.zeros((3, _m.nv)); mujoco.mj_jacGeom(_m, _d, jacp, None, FOOT[sw])
    dofs = [DOF[n] for n in LEG[sw]]
    J = jacp[:, dofs]
    fvel = jacp @ _d.qvel
    err = np.array([tgt[0] - (pf[0] - com[0]), tgt[1] - (pf[1] - com[1]), zt - pf[2]])
    _S["terr"] = float(np.hypot(xt - (pf[0] - com[0]), yt - (pf[1] - com[1])))
    verr = np.array([-(fvel[0] - vx), -(fvel[1] - vy), -fvel[2]])
    F = np.array([p["kp_f"], p["kp_f"] * p["kfy"], p["kp_fz"]]) * err + np.array([p["kd_f"], p["kd_f"] * p["kfy"], p["kd_fz"]]) * verr
    F[2] *= p["fz_on"]
    tsw = J.T @ F + _d.qfrc_bias[dofs]
    # swing hip_z: keep the swing leg pointing along world x (axis +z right, -z left)
    zq = MIDX[f"{sw}_hip_z"]; sgz = 1.0 if sw == "right" else -1.0
    tsw[1] = p["kp_z"] * (-sgz * obs["yaw"] * p["yaw_ff"] - q[zq]) - p["kd_z"] * dq[zq]
    kq = MIDX[f"{sw}_knee"]
    kd_ = p["knee_sw"] if ph < p["ph_k"] else p["knee_ext"]
    tsw[3] = p["fz_knee"] * tsw[3] + p["kp_k"] * (kd_ - q[kq]) - p["kd_k"] * dq[kq] + _d.qfrc_bias[dofs[3]]
    for n, tv in zip(LEG[sw], tsw):
        tau[MIDX[n]] = tv
    # torso attitude torque (world)
    up = np.asarray(obs["up_vector"]); w = np.asarray(obs["root_angular_velocity"])
    zd = np.array([np.sin(p["pitch_d"]), 0.0, np.cos(p["pitch_d"])])
    tw = p["kp_t"] * np.cross(up, zd) - p["kd_t"] * np.array([w[0], w[1], 0.0])
    tw[2] += -p["kp_yaw"] * obs["yaw"] - p["kd_yaw"] * w[2]
    need = tw.copy()
    for n, tv in zip(LEG[sw][:3], tsw[:3]):
        need += tv * _d.xaxis[JID[n]]
    for n in LEG[st][:3]:
        tau[MIDX[n]] = -_d.xaxis[JID[n]] @ need
    k = MIDX[f"{st}_knee"]
    tau[k] = p["kp_ks"] * (p["knee_st"] - q[k]) - p["kd_ks"] * dq[k]
    for n in ("abdomen_y", "abdomen_z", "abdomen_x"):
        i = MIDX[n]; tau[i] = -p["kp_ab"] * q[i] - p["kd_ab"] * dq[i]
    # arms: PD to rest plus a world-frame yaw torque on the torso by reaction
    tyaw = np.array([0.0, 0.0, -p["ka_yaw"] * obs["yaw"] - p["kda_yaw"] * w[2]])
    for i in range(11, 17):
        tau[i] = -p["kp_arm"] * q[i] - p["kd_arm"] * dq[i] - _d.xaxis[JID[MOTOR[i]]] @ tyaw
    return np.clip(tau / GEAR, -0.4, 0.4)


def _stand(obs, q, dq, p):
    tau = np.zeros(17)
    up = np.asarray(obs["up_vector"]); w = np.asarray(obs["root_angular_velocity"])
    tw = p["kp_t"] * np.cross(up, [0, 0, 1.0]) - p["kd_t"] * np.array([w[0], w[1], 0.0])
    for s in ("right", "left"):
        for n in LEG[s][:3]:
            tau[MIDX[n]] = -0.5 * _d.xaxis[JID[n]] @ tw
        k = MIDX[f"{s}_knee"]; tau[k] = p["kp_ks"] * (p["knee_st"] - q[k]) - p["kd_ks"] * dq[k]
    for i in range(3):
        tau[i] = -p["kp_ab"] * q[i] - p["kd_ab"] * dq[i]
    for i in range(11, 17):
        tau[i] = -p["kp_arm"] * q[i] - p["kd_arm"] * dq[i]
    return np.clip(tau / GEAR, -0.4, 0.4)
