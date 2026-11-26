"""Whole-body inverse-dynamics shuffle-walker for the MuJoCo Humanoid (no learning).

State and model
---------------
The full state is rebuilt from ``gym_obs`` (qpos[2:], qvel) in a private copy of the MuJoCo
model.  Observable plant changes are identified at reset: motor gears from ``obs["gear"]``,
torso mass from the torso row of ``cinert`` in ``gym_obs``, gravity from the mass-centre
free-fall acceleration during the first steps (before the feet touch down).  If the
applied command (``last_action``) lags the command we sent, a one-step command delay is
assumed and the private model is stepped forward 15 ms with the in-flight command, so the
new command is computed for the state at which it will act.

Control law (each step, bounded least squares, scipy ``lsq_linear``)
------------------------------------------------------------------
Unknowns: joint accelerations qdd and stance-foot friction-pyramid weights beta >= 0
(friction ``mu``).  Weighted rows:
  floating-base rows of  M qdd + h - passive = J_c^T f          (weight 1e3: physics)
  J_c qdd = -k_c J_c qd                                         (stance foot does not slip)
  torso angular acc = kp_ori * R-error - kd_ori * omega, axes scaled (k_roll, 1, k_yaw);
      desired torso: pitched forward by ``lean``, zero yaw         (weight w_ori)
  CoM vertical acc  = kp_z (z0 - z) - kd_z vz                    (w_z)
  CoM horizontal acc = -k_vxy (v_com - (v_des, 0))               (w_xy)
  swing-foot acc    = kp_sw (p_des - p) + kd_sw (v_ref - v)       (w_sw)
  joint posture PD (kp_post, kd_post) and torque / force regularisers (w_post, w_tau, w_f)
Motor commands are the actuated rows of M qdd + h - passive - J_c^T f, divided by the gears.

Gait: finite-state machine with one stance leg; stance swaps when the swing foot touches down
after tmin_f * t_sw or at tmax_f * t_sw.  Short, quick, low steps (``t_sw``, ``step_h``) with
Raibert/capture-point placement relative to the mass centre:
  x = x0 + cv * vx + ce * (vx - v_des),   y = +-width + cvy * vy,
reached along a cosine blend (``sw_speed``).  The first ``t_stand`` s are double support
(dropping onto the floor).  ``alpha`` blends integrated joint rates with the measured angles
(a complementary filter against angle noise).  Entries ``d_<name>`` replace ``<name>`` (all but ``d_alpha``, unused) once a
command delay has been detected.  The numbers were set by a derivative-free search on the
visible seeds; the gait is a fast, low shuffle (~0.3 m/s).
"""

import numpy as np
import mujoco
from scipy.optimize import lsq_linear
from hl_control.envs.humanoid import XML_PATH, ACTUATED_JOINTS

PARAMS = dict(
    v_des=0.5317, lean=0.0767, z_des=0.0,
    kp_ori=336.4644, kd_ori=21.8614, kp_z=114.1525, kd_z=20.0, kp_sw=584.2242, kd_sw=35.8951, kp_post=50.0, kd_post=10.0,
    w_ori=10.0, w_z=10.0, w_xy=23.8681, w_sw=10.0, w_post=0.7096, w_tau=0.183, w_f=0.001, k_c=10.0, mu=0.5, pred_delay=1,
    d_x0=0.0632, d_alpha=0.6, d_kp_sw=524.6174, d_kd_sw=30.6176, d_ce=0.1456, d_kp_ori=322.7843, d_kd_ori=23.0766, d_v_des=0.5254, d_lean=0.0662, d_t_sw=0.1406, alpha=0.3386, sw_speed=1.4327, k_yaw=0.4948, k_roll=0.9433, k_vxy=2.0804,
    x0=0.058, cv=0.0, ce=0.1456, width=0.1394, cvy=0.1265,
    step_h=0.014, t_stand=0.3, shift=0.6, tmin_f=0.6597, tmax_f=1.4123, t_sw=0.14, press=0.02,
)

_m = mujoco.MjModel.from_xml_path(str(XML_PATH))
_d = mujoco.MjData(_m)
NV = _m.nv
_DADR = np.array([int(_m.jnt_dofadr[_m.joint(n).id]) for n in ACTUATED_JOINTS])   # dof of each motor
_QADR = np.array([int(_m.jnt_qposadr[_m.joint(n).id]) for n in ACTUATED_JOINTS])
_GEAR = _m.actuator_gear[:, 0].copy()
_FOOT = {s: _m.geom(f"{s}_foot").id for s in ("right", "left")}
_FOOTB = {s: _m.body(f"{s}_foot").id for s in ("right", "left")}
_TORSO = _m.body("torso").id
_TORSO_M0 = float(_m.body_mass[_TORSO])
_TORSO_I0 = _m.body_inertia[_TORSO].copy()
_MTOT = float(_m.body_mass.sum())   # nominal; used only for the force regulariser
_R = 0.075
_S = {}


def _reset():
    _S.clear()
    _S.update(stance="left", t_state=0.0, p0=None, tgt_prev=None, z0=None, u_prev=np.zeros(17), delayed=False)


def _jac_point(body, point):
    jp = np.zeros((3, NV))
    mujoco.mj_jac(_m, _d, jp, None, point, body)
    return jp


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or not _S:
        _reset()
    dt = obs["dt"]
    g = obs["gym_obs"]
    if obs["step"] == 0:
        # identify what is observable of the plant: motor gears and the torso mass (from cinert)
        _GEAR[:] = obs["gear"]
        _m.actuator_gear[:, 0] = obs["gear"]
        ratio = g[54] / _TORSO_M0
        _m.body_mass[_TORSO] = _TORSO_M0 * ratio
        _m.body_inertia[_TORSO] = _TORSO_I0 * ratio
        _m.opt.gravity[2] = -9.81
    # joint angles: complementary filter (integrate the rates, correct with the measured angles)
    qj, qdj = g[5:22].copy(), g[28:45]
    if obs["step"] > 0:
        qj = P["alpha"] * (_S["qj"] + 0.5 * (_S["qdj"] + qdj) * dt) + (1 - P["alpha"]) * qj
    _S["qj"], _S["qdj"] = qj, qdj.copy()
    _d.qpos[:] = np.concatenate([[0.0, 0.0], g[0:5], qj])
    _d.qvel[:] = g[22:45]
    if obs["step"] >= 2 and not np.allclose(obs["last_action"], _S["u_prev"]):
        _S["delayed"] = True
    if _S.get("delayed") and P["pred_delay"]:
        # one step of command delay: predict the state at which this command will act
        _d.ctrl[:] = _S["u_prev"]
        mujoco.mj_step(_m, _d, nstep=5)
    mujoco.mj_forward(_m, _d)
    qd = _d.qvel.copy()
    com = _d.subtree_com[1].copy()
    jcom = np.zeros((3, NV))
    mujoco.mj_jacSubtreeCom(_m, _d, jcom, 1)
    vcom = jcom @ qd
    # gravity from the initial free fall (no contact yet): the mass centre accelerates at -g
    if not (obs["foot_contact"] or obs["foot_left_contact"]) and not _S.get("delayed"):
        if obs["step"] == 1:
            _S["vz1"] = vcom[2]
        elif obs["step"] == 5 and "vz1" in _S:
            gest = -(vcom[2] - _S["vz1"]) / (4 * dt)
            if 8.0 < gest < 12.0:
                _m.opt.gravity[2] = -gest
    if _S.get("delayed"):   # retuned gains for a plant with one step of command delay (d_* entries)
        P = {**P, **{k[2:]: v for k, v in P.items() if k.startswith("d_")}}
    if _S["z0"] is None:
        _S["z0"] = com[2] - 0.12 + P["z_des"]

    # ---- gait state machine ----
    _S["t_state"] += dt
    if obs["time"] < P["t_stand"]:
        _S["t_state"] = 0.0
    st = _S["stance"]
    sw = "right" if st == "left" else "left"
    sw_contact = obs["foot_contact"] if sw == "right" else obs["foot_left_contact"]
    if (_S["t_state"] > P["tmin_f"] * P["t_sw"] and sw_contact) or _S["t_state"] > P["tmax_f"] * P["t_sw"]:
        st, sw = sw, st
        _S.update(stance=st, t_state=0.0, p0=None, tgt_prev=None)
    standing = obs["time"] < P["t_stand"]
    if _S["p0"] is None:
        _S["p0"] = _d.geom_xpos[_FOOT[sw]] - com
    s = min(_S["t_state"] / P["t_sw"], 1.0)

    # ---- contacts ----
    touch = {"right": obs["foot_contact"], "left": obs["foot_left_contact"]}
    contacts = [k for k in ([st] + ([sw] if standing else [])) if touch[k]]
    if contacts:
        Jc = np.vstack([_jac_point(_FOOTB[k], _d.geom_xpos[_FOOT[k]] - np.array([0, 0, _R])) for k in contacts])
    else:
        Jc = np.zeros((0, NV))
    nc = len(contacts)
    nf = 3 * nc
    mu = P["mu"]
    D = np.zeros((nf, 4 * nc))        # contact force = D @ beta, beta >= 0 (friction pyramid)
    for i in range(nc):
        D[3 * i:3 * i + 3, 4 * i:4 * i + 4] = np.array([[mu, -mu, 0, 0], [0, 0, mu, -mu], [1, 1, 1, 1]])
    Jf = D.T @ Jc
    nb = 4 * nc
    nx = NV + nb
    M = np.zeros((NV, NV))
    mujoco.mj_fullM(_m, _d, M)
    h = _d.qfrc_bias - _d.qfrc_passive
    rows, rhs = [], []

    def add(A, b, w):
        rows.append(w * A)
        rhs.append(w * b)

    # physics of the unactuated floating base
    A = np.zeros((6, nx)); A[:, :NV] = M[:6]; A[:, NV:] = -Jf[:, :6].T
    add(A, -h[:6], 1e3)
    # stance foot fixed
    A = np.zeros((nf, nx)); A[:, :NV] = Jc
    add(A, -P["k_c"] * (Jc @ qd), 1e2)
    # torso orientation
    jr = np.zeros((3, NV))
    mujoco.mj_jacBody(_m, _d, None, jr, _TORSO)
    w_ang = jr @ qd
    R = _d.xmat[_TORSO].reshape(3, 3)
    cl, sl = np.cos(P["lean"]), np.sin(P["lean"])
    Rd = np.array([[cl, 0, sl], [0, 1, 0], [-sl, 0, cl]])      # desired: lean forward about world y, zero yaw
    E = Rd @ R.T
    err = 0.5 * np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    A = np.zeros((3, nx)); A[:, :NV] = jr
    kyaw = np.array([P["k_roll"], 1.0, P["k_yaw"]])
    add(A, kyaw * (P["kp_ori"] * err - P["kd_ori"] * w_ang), P["w_ori"])
    # CoM height (and optional horizontal damping)
    A = np.zeros((1, nx)); A[0, :NV] = jcom[2]
    add(A, np.array([P["kp_z"] * (_S["z0"] - com[2]) - P["kd_z"] * vcom[2]]), P["w_z"])
    if standing and len(contacts) == 2:      # shift the mass centre over the first stance foot
        tgt_xy = 0.5 * (_d.geom_xpos[_FOOT["left"]][:2] + _d.geom_xpos[_FOOT["right"]][:2])
        tgt_xy[1] += P["shift"] * (_d.geom_xpos[_FOOT[st]][1] - tgt_xy[1])
        tgt_xy[0] = com[0]
        A = np.zeros((2, nx)); A[:, :NV] = jcom[:2]
        add(A, 30.0 * (tgt_xy - com[:2]) - 10.0 * vcom[:2], 10.0)
    if P["w_xy"] > 0:
        A = np.zeros((2, nx)); A[:, :NV] = jcom[:2]
        add(A, -P["k_vxy"] * (vcom[:2] - np.array([P["v_des"], 0.0])), P["w_xy"])
    # swing foot
    if not standing:
        sign = -1.0 if sw == "right" else 1.0
        vx, vy = vcom[0], vcom[1]
        px = P["x0"] + P["cv"] * vx + P["ce"] * (vx - P["v_des"])
        py = sign * P["width"] + P["cvy"] * vy
        p0 = _S["p0"]
        _S["tgt"] = (round(px, 2), round(py, 2))
        wgt = 0.5 - 0.5 * np.cos(np.pi * min(1.0, s * P["sw_speed"]))
        tgt = np.array([com[0] + p0[0] + (px - p0[0]) * wgt, com[1] + p0[1] + (py - p0[1]) * wgt,
                        _R + P["step_h"] * np.sin(np.pi * s) - P["press"] * (s >= 1.0)])
        vt = np.zeros(3) if _S["tgt_prev"] is None else (tgt - _S["tgt_prev"]) / dt
        _S["tgt_prev"] = tgt
        fp = _d.geom_xpos[_FOOT[sw]]
        Js = _jac_point(_FOOTB[sw], fp)
        A = np.zeros((3, nx)); A[:, :NV] = Js
        add(A, P["kp_sw"] * (tgt - fp) + P["kd_sw"] * (vt - Js @ qd), P["w_sw"])
        _S["dbg"] = np.round(tgt - fp, 2)
    # posture of the joints (abdomen, arms, hip yaw, knees mildly bent)
    q0 = np.zeros(17)
    q0[[6, 10]] = -0.3
    A = np.zeros((17, nx)); A[np.arange(17), _DADR] = 1.0
    add(A, P["kp_post"] * (q0 - _d.qpos[_QADR]) - P["kd_post"] * qd[_DADR], P["w_post"])
    # regularisation: contact force ~ weight share, motor torques small
    A = np.zeros((nf, nx)); A[:, NV:] = D
    fz = np.zeros(nf); fz[2::3] = _MTOT * 9.81 / max(len(contacts), 1)
    add(A, fz, P["w_f"])
    A = np.zeros((17, nx)); A[:, :NV] = M[_DADR] / (_GEAR[:, None] * 0.4); A[:, NV:] = -Jf[:, _DADR].T / (_GEAR[:, None] * 0.4)
    add(A, -h[_DADR] / (_GEAR * 0.4), P["w_tau"])

    Aall = np.vstack(rows)
    ball = np.concatenate(rhs)
    if nb:
        lo = np.concatenate([-np.inf * np.ones(NV), np.zeros(nb)])
        x = lsq_linear(Aall, ball, bounds=(lo, np.inf * np.ones(nx)), method="bvls").x
    else:
        x = np.linalg.lstsq(Aall, ball, rcond=None)[0]
    qdd, beta = x[:NV], x[NV:]
    _S["qdd"] = qdd
    f = D @ beta
    tau = (M @ qdd + h - Jc.T @ f)[_DADR]
    _S["f"] = np.round(f, 0)
    u = np.clip(tau / _GEAR, -0.4, 0.4)
    _S["u_prev"] = u
    return u
