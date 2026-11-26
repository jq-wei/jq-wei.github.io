"""Computed-torque controller with payload estimation, delay prediction and lag inversion.

Pipeline, per 10 ms control step:

1. **Delay predictor.** The commanded torque acts one control step late (10 ms pure
   delay) through a 20 ms first-order lag. Both are known exactly, so the controller
   keeps a model of the applied torque ``y`` and of the command still in flight, and
   integrates the rigid-body model ``PRED`` steps forward from the (filtered) measured
   state. The servo then acts on the *predicted* state against the reference evaluated
   at the same future time, which removes the pure delay from the loop and restores
   phase margin.
2. **Reference in joint space.** Closed-form IK (elbow-up branch) maps the future
   task-space reference to ``q_ref``; a damped Jacobian inverse maps the reference
   velocity; the reference acceleration is a clean finite difference (the reference is
   noise-free). The reference radius is clamped to [R_MIN, 0.98] m so mid-move chords
   can never command the elbow into its mechanical stop.
3. **Inverse dynamics.** Desired acceleration ``qdd_ref + KP e + KD ed`` is pushed
   through the full model ``M qdd + C + G + Fv qd`` (known link parameters + payload
   estimate), plus an integral torque (gain KI, clamp INT_LIM) for residuals. During
   the dwell tail the servo gains are scaled by HOLD_SCALE to stop the loop from
   chasing sensor noise while the hold criterion is being measured.
4. **Payload estimator.** The payload (0.3-1 kg) attaches at the start of segment 2,
   so the estimate is pinned to 0 before that and jumps to a 0.6 kg prior there; then,
   in a short window early in each dwell (while the integrator is frozen - from one
   equilibrium only the *total* holding torque is identifiable, so only one of the two
   may adapt at a time), the quasi-static torque balance gives the mass directly from
   the gravity regressor ``r = g [l1 c1 + l2 c12, l2 c12]``:
   ``m = r . (y - G_nominal - friction - tau_int) / (r . r)``, low-pass filtered at
   EST_ALPHA per step. The integrator itself (KI, clamp INT_LIM) only winds during
   dwells, outside the estimation window, so move transients cannot bias it.
5. **Lag inversion.** Over one step the lag maps command u to applied torque
   ``y+ = d y + (1-d) u`` with ``d = (1 - h/tau)^10``; commanding
   ``u = y + LEAD (tau_des - y)`` (LEAD = 1/(1-d) ~ 2.49 would be exact) cancels most
   of it. POS_BETA / VEL_BETA are the measurement low-pass blends (1 = raw).
"""

import math

import numpy as np

from hl_control.envs.two_link_arm import inverse_kinematics, jacobian, min_jerk

PARAMS = {
    "KP": 313.8403706824282,
    "KD": 19.57167073245429,
    "KI": 23.74538630696114,
    "INT_LIM": 6.552141596526809,
    "LEAD": 2.488101739132253,
    "EST_ALPHA": 0.09060546247111236,
    "VEL_BETA": 0.53887281442875,
    "POS_BETA": 0.9505999614936416,
    "R_MIN": 0.17,
    "PRED": 0.9387512181941804,
    "HOLD_SCALE": 0.6237522068383548,
}

PARAM_BOUNDS = {
    "KP": (120.0, 700.0),
    "KD": (18.0, 70.0),
    "KI": (10.0, 150.0),
    "INT_LIM": (1.0, 8.0),
    "LEAD": (1.0, 2.49),
    "EST_ALPHA": (0.05, 0.4),
    "VEL_BETA": (0.3, 1.0),
    "POS_BETA": (0.3, 1.0),
    "PRED": (0.0, 1.6),
    "HOLD_SCALE": (0.3, 1.0),
}

ACT_TAU = 0.02  # s, actuator lag (scenario constant)

_state: dict = {}


def _reset_state(obs):
    _state.clear()
    _state["q_f"] = np.asarray(obs["q"], dtype=float).copy()
    _state["qd_f"] = np.asarray(obs["q_dot"], dtype=float).copy()
    _state["qd_ref_prev"] = None
    _state["segment_prev"] = -1
    _state["p0"] = np.asarray(obs["ref_xy"], dtype=float).copy()
    _state["tau_int"] = np.zeros(2)
    _state["m_hat"] = 0.0
    _state["prior_set"] = False
    _state["y"] = np.zeros(2)            # modelled applied torque at the obs time
    _state["cmd_queue"] = [np.zeros(2)]  # one control step of pure delay


def _params_of(obs, payload):
    """a1, a2, a3, g1, g2 of the standard two-link model, payload included."""
    m1, m2 = obs["link_mass"]
    l1, l2 = obs["link_length"]
    c1, c2 = obs["link_com"]
    i1, i2 = obs["link_inertia"]
    g = obs["gravity"]
    a1 = i1 + m1 * c1 * c1 + i2 + m2 * (l1 * l1 + c2 * c2) + payload * (l1 * l1 + l2 * l2)
    a2 = i2 + m2 * c2 * c2 + payload * l2 * l2
    a3 = m2 * l1 * c2 + payload * l1 * l2
    g1 = (m1 * c1 + m2 * l1 + payload * l1) * g
    g2 = (m2 * c2 + payload * l2) * g
    return a1, a2, a3, g1, g2


def _dynamics(obs, q, qd, payload):
    """Mass matrix and the sum of Coriolis + gravity + viscous friction torques."""
    a1, a2, a3, g1, g2 = _params_of(obs, payload)
    c2 = math.cos(q[1])
    s2 = math.sin(q[1])
    M = np.array([[a1 + 2.0 * a3 * c2, a2 + a3 * c2], [a2 + a3 * c2, a2]])
    C = a3 * s2 * np.array([-(2.0 * qd[0] * qd[1] + qd[1] ** 2), qd[0] ** 2])
    G = np.array([g1 * math.cos(q[0]) + g2 * math.cos(q[0] + q[1]),
                  g2 * math.cos(q[0] + q[1])])
    Fv = np.asarray(obs["viscous_friction"], dtype=float) * qd
    return M, C + G + Fv


def _future_reference(obs, s, t_ahead):
    """Task-space reference position and velocity ``t_ahead`` seconds from now."""
    tau = float(obs["time_in_segment"]) + t_ahead
    T = float(obs["move_time"])
    p1 = np.asarray(obs["target_xy"], dtype=float)
    if tau >= T:
        return p1.copy(), np.zeros(2)
    p0 = s["p0"]
    sj, sjd = min_jerk(tau / T)
    return p0 + (p1 - p0) * sj, (p1 - p0) * sjd / T


def controller(obs: dict) -> np.ndarray:
    if obs["step"] == 0 or not _state:
        _reset_state(obs)
    s = _state
    p = PARAMS
    dt = float(obs["sample_dt"])
    L = obs["link_length"]
    lim = obs["torque_limit"]

    q_raw = np.asarray(obs["q"], dtype=float)
    qd_raw = np.asarray(obs["q_dot"], dtype=float)
    s["q_f"] = p["POS_BETA"] * q_raw + (1.0 - p["POS_BETA"]) * s["q_f"]
    s["qd_f"] = p["VEL_BETA"] * qd_raw + (1.0 - p["VEL_BETA"]) * s["qd_f"]
    q, qd = s["q_f"], s["qd_f"]

    seg = int(obs["segment"])
    if seg != s["segment_prev"]:
        s["p0"] = np.asarray(obs["ref_xy"], dtype=float).copy()

    # --- payload estimate ---------------------------------------------------------
    # The payload attaches at the start of segment 2 (task guarantee): before that the
    # estimate is pinned to zero; at segment 2 it jumps to the mid-range prior so only
    # +-0.35 kg is left for the estimator. Estimation runs in a short window early in
    # each dwell DURING WHICH THE INTEGRATOR IS FROZEN: from one equilibrium only the
    # total holding torque is identifiable, so letting both adapt at once lets them
    # chase each other (observed as a slow drift of the estimate).
    if seg < 2:
        s["m_hat"] = 0.0
    elif not s["prior_set"]:
        s["prior_set"] = True
        s["m_hat"] = max(s["m_hat"], 0.6)
    tis = float(obs["time_in_segment"])
    mt = float(obs["move_time"])
    est_window = seg >= 2 and (not obs["in_move"]) and (mt + 0.15 < tis <= mt + 0.55)
    g = obs["gravity"]
    c1v, c12v = math.cos(q[0]), math.cos(q[0] + q[1])
    reg = g * np.array([L[0] * c1v + L[1] * c12v, L[1] * c12v])  # d(gravity)/d(payload)
    if est_window and float(np.max(np.abs(qd_raw))) < 0.15 and float(reg @ reg) > 4.0:
        _, _, _, g1n, g2n = _params_of(obs, 0.0)
        g_nom = np.array([g1n * c1v + g2n * c12v, g2n * c12v])
        fric = np.asarray(obs["viscous_friction"], dtype=float) * qd_raw
        m_inst = float(reg @ (s["y"] - g_nom - fric - s["tau_int"])) / float(reg @ reg)
        m_inst = min(max(m_inst, 0.0), 1.5)
        s["m_hat"] += p["EST_ALPHA"] * (m_inst - s["m_hat"])

    # --- delay predictor: roll the model forward PRED control steps ----------------
    t_ahead = p["PRED"] * dt
    qp, qdp = q.copy(), qd.copy()
    if t_ahead > 0.0:
        n_sub = 3
        h = t_ahead / n_sub
        yp = s["y"].copy()
        u_inflight = s["cmd_queue"][0]
        lag_step = 1.0 - math.exp(-h / ACT_TAU)
        for _ in range(n_sub):
            M, b = _dynamics(obs, qp, qdp, s["m_hat"])
            qdd = np.linalg.solve(M, yp - b)
            qp = qp + h * qdp + 0.5 * h * h * qdd
            qdp = qdp + h * qdd
            yp = yp + lag_step * (u_inflight - yp)

    # --- joint-space reference at the predicted time -------------------------------
    ref, ref_v = _future_reference(obs, s, t_ahead)
    r = float(np.hypot(ref[0], ref[1]))
    r_cl = min(max(r, p["R_MIN"]), 0.98)
    if r_cl != r:
        ref = ref * (r_cl / max(r, 1e-9))
    q_ref = inverse_kinematics(ref, L, elbow_up=True)
    J_ref = jacobian(q_ref, L)
    qd_ref = J_ref.T @ np.linalg.solve(J_ref @ J_ref.T + 1e-4 * np.eye(2), ref_v)
    if s["qd_ref_prev"] is None or seg != s["segment_prev"]:
        qdd_ref = np.zeros(2)
    else:
        qdd_ref = (qd_ref - s["qd_ref_prev"]) / dt
    s["qd_ref_prev"] = qd_ref.copy()
    s["segment_prev"] = seg

    # --- inverse dynamics servo on the predicted state ------------------------------
    hold = (not obs["in_move"]) and obs["time_in_segment"] > obs["move_time"] + 0.5
    scale = p["HOLD_SCALE"] if hold else 1.0
    e = q_ref - qp
    ed = qd_ref - qdp
    qdd_cmd = qdd_ref + scale * p["KP"] * e + math.sqrt(scale) * p["KD"] * ed
    M, b = _dynamics(obs, qp, qdp, s["m_hat"])
    tau_des = M @ qdd_cmd + b + s["tau_int"]

    # integral on joint error: dwells only (move-phase tracking transients would just
    # wind in garbage), frozen while the payload estimator is adapting, hard clamp
    e_now = q_ref - q
    if (not obs["in_move"]) and float(np.max(np.abs(e_now))) < 0.15 and not est_window:
        s["tau_int"] = np.clip(s["tau_int"] + p["KI"] * e_now * dt, -p["INT_LIM"], p["INT_LIM"])

    # --- actuator lag inversion and applied-torque bookkeeping ----------------------
    u = s["y"] + p["LEAD"] * (tau_des - s["y"])
    u = np.clip(u, -lim, lim)
    s["cmd_queue"].append(u.copy())
    eff = s["cmd_queue"].pop(0)
    d = (1.0 - dt / 10.0 / ACT_TAU) ** 10  # (1 - h/tau)^substeps, the plant's discretisation
    s["y"] = d * s["y"] + (1.0 - d) * eff

    return u
