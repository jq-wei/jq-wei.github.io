"""Adroit relocate: pick up the ball and hold it at the target, at joint level.

Structure: state machine (APPROACH -> CLOSE -> CARRY) + model-based inverse kinematics + joint PD servo.
No learning; a private copy of the plant's MuJoCo model is used only for kinematics (palm-site position
and its Jacobian with respect to the six arm joints) and for the gravity load on the arm joints.

Control law
-----------
1. Estimation. Joint angles are measured; joint velocities are finite differences. Forward kinematics of
   the palm site (S_grasp) gives its world position, hence the world positions of the ball and of the
   target, which are observed only relative to the palm. If the very first observation shows measurement
   noise (the hand starts exactly at the zero pose), the static quantities (target, resting ball) are
   averaged and the moving ones (joint velocities, palm-ball offset while carrying) are low-pass filtered
   (nz_av, nz_ar); without noise the raw values are used.
2. Task space -> joint space. A bounded, damped least-squares IK turns the desired palm-site position into
   targets for the six arm joints. The weights w_* decide how the redundant joints share the motion (the
   light, strongly actuated pitch/yaw rotations do most of it, the y slide gives the forward reach, roll
   stays at zero).
3. Arm servo. Each arm servo applies g*ctrl - k*q, so
       ctrl = (k*q_ref + gravity + Kp*(q_ref - q) - Kd*qdot) / g
   holds q_ref with gravity compensated and adds stiffness Kp and damping Kd to the servo's own spring.
   kp_s/kd_s act on the three slides (5 kg, lightly damped by themselves: they need both), kp_r on the
   three rotations (already well damped by the joint damping, so kd_r = 0: finite-difference damping on
   these low-inertia joints only adds chatter). Wrist and fingers are plain position servos: set points.
4. Phases.
   APPROACH: hand open, palm site driven to the grasp point = ball + (gx, gy, gz). The reference is raised
     by funnel*(horizontal distance - r0), capped at hmax, so the palm comes down only once it is above
     the ball. A ball beyond the reach of the y slide is grasped from as far forward as the palm gets.
   CLOSE: starts when the palm site is within close_tol of the grasp point, i.e. during the last ~2 cm of
     the descent, so the fingers (f_j2, f_j1, f_j0) wrap the ball while the palm settles on it and the
     thumb (th4..th0) swings under it; lasts close_steps.
   CARRY: palm reference = target + measured (palm - ball), which servoes the *ball* onto the target;
     the joint reference is low-passed (alpha_carry) to keep jerk off the grasp. The wrist is flexed
     (wr_carry) so the ball rests in the hook of the fingers, and the palm is arched (lf4_c) to close the
     little-finger side. If the ball gets farther than `lost` from the palm near the table, the hand
     re-opens and approaches again.
"""

import numpy as np

from hl_control.envs.adroit import AdroitConfig, AdroitEnv, mujoco

PARAMS = {
    # grasp geometry: palm site (S_grasp) relative to the ball centre when closing [m]
    "gx": 0.0, "gy": -0.012, "gz": 0.010,
    # funnel: extra height per metre of horizontal distance beyond r0, capped at hmax
    "funnel": 1.0, "r0": 0.018, "hmax": 0.08,
    # closing trigger (palm-site error to the grasp point [m]) and closing duration [steps]; v_tol is an optional
    # palm-speed limit for the trigger (10 m/s = off: closing during the descent grasps best); after app_timeout
    # steps of approach the hand closes anyway if it is within 5 cm
    "close_tol": 0.025, "close_steps": 10, "v_tol": 10.0, "app_timeout": 45,
    # finger set points when closed [rad]: knuckle, middle, tip; thumb joints
    "f_j2": 1.6, "f_j1": 1.2, "f_j0": 0.8,
    "th4": 0.0, "th3": 1.3, "th2": 0.0, "th1": -0.5, "th0": -0.8,
    # palm arch (little-finger metacarpal) once the ball is lifted: closes the escape route on the little-finger side
    "lf4_c": 0.7,
    # pre-shape while approaching
    "pre_j2": 0.15, "pre_th4": 0.3, "pre_th3": 0.0,
    # wrist flexion set point while approaching/grasping and while carrying
    "wr_grasp": 0.0, "wr_carry": 0.5,
    # arm PD gains (physical units: N/m, N s/m for slides; N m/rad, N m s/rad for rotations)
    "kp_s": 2500.0, "kd_s": 150.0, "kp_r": 200.0, "kd_r": 0.0,
    # reference filter (fraction of the remaining joint error taken per step) per phase
    "alpha_app": 1.0, "alpha_carry": 0.5,
    # IK weights (larger = joint used less): slide x, slide up, slide y, pitch, yaw, roll
    "w_tx": 0.02, "w_ty": 0.02, "w_tz": 0.002, "w_rx": 0.002, "w_ry": 0.004, "w_rz": 1.0,
    # ball considered lost when farther than this from the palm site during carry [m]
    "lost": 0.09,
    # carry-phase IK: weights of the up-slide and the pitch, and the preferred pitch (nose-down keeps the ball in the finger hook)
    "w_ty_c": 0.02, "w_rx_c": 0.002, "pitch_nom_c": 0.0,
    # noisy-measurement mode: low-pass factors for the joint velocities and for the palm-ball offset while carrying
    "nz_av": 0.35, "nz_ar": 0.2,
}

_env = AdroitEnv(AdroitConfig(task="relocate"))
_m = _env.model
_d = mujoco.MjData(_m)
_SITE = _m.site("S_grasp").id
_LO = _m.jnt_range[:30, 0].copy()
_HI = _m.jnt_range[:30, 1].copy()
_G = 500.0   # arm servo gain
_K = 200.0   # arm servo stiffness
_jacp = np.zeros((3, _m.nv))
_bias = np.zeros(_m.nv)

_S = {}


def _reach_y():
    """Farthest forward position of the palm site at grasp height (y slide at its end)."""
    q = np.zeros(30)
    q[2] = _HI[2]
    best = 0.0
    for pitch in np.linspace(0.1, 0.35, 26):
        q[3] = pitch
        p = _fk(q)
        if p[2] <= 0.047:
            return float(p[1])
        best = float(p[1])
    return best


def _fk(q30):
    _d.qpos[:30] = q30
    _d.qpos[30:33] = (0.0, 0.0, -5.0)
    mujoco.mj_kinematics(_m, _d)
    mujoco.mj_comPos(_m, _d)
    return _d.site_xpos[_SITE].copy()


def _ik(q30, p_des, w, q_nom, iters=4):
    """Arm joint targets (6,) that put the palm site at p_des, regularised toward the zero posture.
    Bounded damped least squares: a joint that hits its range is frozen for the remaining iterations."""
    q = q30.copy()
    free = np.ones(6, dtype=bool)
    for _ in range(iters):
        p = _fk(q)
        mujoco.mj_jacSite(_m, _d, _jacp, None, _SITE)
        J = _jacp[:, :6][:, free]
        A = J.T @ J + np.diag(w[free])
        b = J.T @ (p_des - p) - w[free] * (q[:6][free] - q_nom[free])
        dq = np.zeros(6)
        dq[free] = np.linalg.solve(A, b)
        qn = q[:6] + dq
        hit = (qn < _LO[:6]) | (qn > _HI[:6])
        q[:6] = np.clip(qn, _LO[:6], _HI[:6])
        free &= ~hit
        if not free.any():
            break
    return q[:6]


_Y_MAX = None


def _reset(q):
    global _Y_MAX
    if _Y_MAX is None:
        _Y_MAX = _reach_y() - 0.002
    _S.clear()
    _S.update(phase=0, q_prev=q.copy(), q_ref=q[:6].copy(), t_close=0, t_app=0, palm_prev=_fk(q), retries=0)
    # the hand starts exactly at the zero pose: any non-zero finger reading at step 0 is measurement noise
    _S.update(noisy=bool(np.sqrt(np.mean(q[6:30] ** 2)) > 1e-5), tgt_sum=np.zeros(3), tgt_n=0,
              ball_sum=np.zeros(3), ball_n=0, rel_f=np.zeros(3), qd_f=np.zeros(30))


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    q = o[:30]
    step = int(obs["step"])
    if step == 0 or not _S:
        _reset(q)
    dt = float(obs["dt"])
    qd = (q - _S["q_prev"]) / dt
    _S["q_prev"] = q.copy()

    palm = _fk(q)
    # gravity load on the arm joints at the measured posture
    _d.qvel[:] = 0.0
    _d.qacc[:] = 0.0
    mujoco.mj_comVel(_m, _d)
    mujoco.mj_rne(_m, _d, 0, _bias)
    grav = _bias[:6].copy()

    ball = palm - o[30:33]
    target = palm - o[33:36]
    rel = o[30:33].copy()  # palm site minus ball centre
    if _S["noisy"]:
        # noisy measurements: average what is static, low-pass what moves
        _S["tgt_sum"] += target
        _S["tgt_n"] += 1
        target = _S["tgt_sum"] / _S["tgt_n"]
        _S["qd_f"] += P["nz_av"] * (qd - _S["qd_f"])
        qd = _S["qd_f"].copy()
        if _S["phase"] == 2:
            _S["rel_f"] += P["nz_ar"] * (rel - _S["rel_f"])
            rel = _S["rel_f"].copy()
            ball = palm - rel
        else:
            _S["rel_f"] += 0.5 * (rel - _S["rel_f"])
            if _S["ball_n"] > 0 and np.linalg.norm(ball - _S["ball_sum"] / _S["ball_n"]) > 0.03:
                _S["ball_sum"] = np.zeros(3)
                _S["ball_n"] = 0
            _S["ball_sum"] += ball
            _S["ball_n"] += 1
            ball = _S["ball_sum"] / _S["ball_n"]
    g_off = np.array([P["gx"], P["gy"], P["gz"]])

    cm = np.asarray(obs["ctrl_mean"], dtype=float)
    ch = np.asarray(obs["ctrl_half_range"], dtype=float)
    ctrl = np.zeros(30)

    phase = _S["phase"]
    v_palm = (palm - _S["palm_prev"]) / dt
    _S["palm_prev"] = palm.copy()
    # ---------------- phase logic ----------------
    grasp_pt = ball + g_off
    # the palm site cannot go farther forward than the end of the y slide: grasp from as far as it reaches
    grasp_pt[1] = min(grasp_pt[1], _Y_MAX)
    if phase == 0:
        err = grasp_pt - palm
        settled = np.linalg.norm(err) < P["close_tol"] and np.linalg.norm(v_palm) < P["v_tol"]
        timeout = step - _S["t_app"] > P["app_timeout"] and np.linalg.norm(err) < 0.05
        if settled or timeout:
            phase = 1
            _S["t_close"] = step
    elif phase == 1:
        if step - _S["t_close"] >= P["close_steps"]:
            phase = 2
    elif phase == 2:
        if np.linalg.norm(rel) > P["lost"] and ball[2] < 0.06:
            phase = 0
            _S["t_app"] = step
            _S["retries"] += 1
            _S["ball_n"] = 0
            _S["ball_sum"] = np.zeros(3)
    _S["phase"] = phase

    # ---------------- arm reference ----------------
    if phase == 0:
        dxy = np.linalg.norm((grasp_pt - palm)[:2])
        p_des = grasp_pt + np.array([0.0, 0.0, min(P["hmax"], P["funnel"] * max(0.0, dxy - P["r0"]))])
        alpha = P["alpha_app"]
        wrist = P["wr_grasp"]
    elif phase == 1:
        p_des = grasp_pt.copy()
        p_des[2] = 0.035 + P["gz"]
        alpha = P["alpha_app"]
        wrist = P["wr_grasp"]
    else:
        p_des = target + rel
        alpha = P["alpha_carry"]
        wrist = P["wr_carry"]

    w = np.array([P["w_tx"], P["w_ty"], P["w_tz"], P["w_rx"], P["w_ry"], P["w_rz"]])
    q_ik = q.copy()
    q_ik[6] = 0.0
    q_ik[7] = wrist
    q_nom = np.zeros(6)
    if phase == 2:
        w[1] = P["w_ty_c"]
        w[3] = P["w_rx_c"]
        q_nom[3] = P["pitch_nom_c"]
    q_t = _ik(q_ik, p_des, w, q_nom)
    q_ref = _S["q_ref"] + alpha * (q_t - _S["q_ref"])
    _S["q_ref"] = q_ref

    kp = np.array([P["kp_s"]] * 3 + [P["kp_r"]] * 3)
    kd = np.array([P["kd_s"]] * 3 + [P["kd_r"]] * 3)
    ctrl[:6] = (_K * q_ref + grav + kp * (q_ref - q[:6]) - kd * qd[:6]) / _G
    ctrl[6] = 0.0 + _bias[6] / 10.0
    ctrl[7] = wrist + _bias[7] / 10.0

    # ---------------- fingers ----------------
    if phase == 0:
        for b in (8, 12, 16, 21):
            ctrl[b + 1] = P["pre_j2"]
        ctrl[25] = P["pre_th4"]
        ctrl[26] = P["pre_th3"]
    else:
        for b in (8, 12, 16, 21):
            ctrl[b + 1] = P["f_j2"]
            ctrl[b + 2] = P["f_j1"]
            ctrl[b + 3] = P["f_j0"]
        ctrl[25] = P["th4"]
        ctrl[26] = P["th3"]
        ctrl[27] = P["th2"]
        ctrl[28] = P["th1"]
        ctrl[29] = P["th0"]
        if phase == 2 and step - _S["t_close"] > P["close_steps"] + 8:
            ctrl[20] = P["lf4_c"]  # arch the palm (little-finger metacarpal) around the ball once it is lifted

    a = (ctrl - cm) / ch
    return np.clip(a, -1.0, 1.0)
