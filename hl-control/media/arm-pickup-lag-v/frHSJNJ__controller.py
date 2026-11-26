"""Two-link arm: computed-torque joint servo with an adaptive payload estimate.

Control law (joint space, all model terms evaluated with the payload estimate m_hat):

    q_d, qd_d, qdd_d  = IK of the task-space min-jerk reference (elbow-up branch), differentiated exactly
    tau = M(q; m_hat) (qdd_d + KP e + KD e_dot) + C(q, q_dot; m_hat) + G(q; m_hat) + F q_dot + KI * int(e)
          + feedforward lead: M qdd_d + C + G evaluated LEAD seconds ahead on the reference
            (compensates the 10 ms delay + 20 ms first-order actuator lag)

    e = q_d - q, e_dot = qd_d - q_dot

Payload estimation (the only unknown parameter, EST = 1, default): the rigid-body model is linear in the
payload mass m. Every step the torque that actually acted is reconstructed from the command history
(one step of delay, 20 ms lag) and the one-step residual tau_avg - (M0 qdd + C0 + G0 + F qd) = m Ym is fed to a
recursive instrumental-variable least-squares fit with forgetting factor LS_FORGET (the instrument uses the
reference acceleration, so the differentiated velocity noise does not bias m_hat low). m_hat is clipped to
[0, M_MAX]. It picks up the 0.3-1 kg payload within ~0.1-0.2 s of its attachment.
EST = 0 selects the fallback gradient law m_hat_dot = GAMMA * Y_g(q)^T (e_dot + LAM e) on the gravity regressor.
When obs["payload_known"] is True the reported value is used directly.

Gains: KP, KD  - servo stiffness/damping (rad/s^2 per rad, per rad/s), applied through M so both joints
                 get the same closed-loop bandwidth (wn = 15 rad/s; stiffer loses margin to the 30 ms lag);
       KI      - optional joint integral (N m per rad s) for residual static error (0: not needed);
       LS_FORGET, LS_PRIOR - forgetting factor (~0.2 s memory) and initial weight of the m = 0 prior;
       GAMMA, LAM - rate and position weight of the fallback gradient estimator (EST = 0);
       LEAD    - feedforward time advance [s].
"""

import math

import numpy as np

PARAMS = {
    "KP": 225.0,
    "KD": 28.0,
    "KI": 0.0,
    "GAMMA": 1.0,
    "LAM": 5.0,
    "LEAD": 0.03,
    "EST": 1,
    "LS_FORGET": 0.95,
    "LS_PRIOR": 50.0,
}
M_MAX = 1.5
ACT_TAU = 0.02  # s, actuator lag assumed by the payload estimator

_state = {}


def _reset(obs):
    _state.clear()
    _state["p0"] = np.asarray(obs["ref_xy"], float).copy()
    _state["seg"] = int(obs["segment"])
    _state["prev_target"] = np.asarray(obs["target_xy"], float).copy()
    _state["m_hat"] = 0.0
    _state["integ"] = np.zeros(2)
    _state["cmds"] = []
    _state["ta"] = np.zeros(2)
    _state["num"] = 0.0
    _state["den"] = PARAMS["LS_PRIOR"]
    _state["q_prev"] = None


def _ls_update(obs, q, qd, dt, fric, qdd_ref):
    """Recursive least squares (with forgetting) for the payload mass m from the one-step dynamics residual.

    The torque that acted over the last step is reconstructed from the command history through the actuator
    model (one step of delay, 20 ms first-order lag); the rigid-body model is linear in m:
        tau_avg - (M0 qdd + C0 + G0 + F qd) = m * Ym(q, qd, qdd).
    """
    if _state["q_prev"] is None:
        return 0.0
    cmds = _state["cmds"]
    u = cmds[-2] if len(cmds) >= 2 else np.zeros(2)
    ta, acc = _state["ta"], np.zeros(2)
    a = (dt / 10) / ACT_TAU
    for _ in range(10):
        ta = ta + a * (u - ta)
        acc += ta
    _state["ta"] = ta
    tau_avg = acc / 10
    qm = 0.5 * (q + _state["q_prev"])
    qdm = 0.5 * (qd + _state["qd_prev"])
    qdd = (qd - _state["qd_prev"]) / dt
    M0, C0, G0 = _dyn(obs, 0.0, qm, qdm)
    M1, C1, G1 = _dyn(obs, 1.0, qm, qdm)
    r = tau_avg - (M0 @ qdd + C0 + G0 + fric * qdm)
    Ym = (M1 - M0) @ qdd + (C1 - C0) + (G1 - G0)
    # instrument: the same regressor with the noise-free reference acceleration (removes the noise bias)
    Z = (M1 - M0) @ qdd_ref + (C1 - C0) + (G1 - G0)
    lam = PARAMS["LS_FORGET"]
    _state["num"] = lam * _state["num"] + float(Z @ r)
    _state["den"] = lam * _state["den"] + float(Z @ Ym)
    return min(M_MAX, max(0.0, _state["num"] / _state["den"]))


def _model(obs, m):
    m1, m2 = obs["link_mass"]
    l1, l2 = obs["link_length"]
    c1, c2 = obs["link_com"]
    i1, i2 = obs["link_inertia"]
    g = obs["gravity"]
    a1 = i1 + m1 * c1 * c1 + i2 + m2 * (l1 * l1 + c2 * c2) + m * (l1 * l1 + l2 * l2)
    a2 = i2 + m2 * c2 * c2 + m * l2 * l2
    a3 = m2 * l1 * c2 + m * l1 * l2
    g1 = (m1 * c1 + m2 * l1 + m * l1) * g
    g2 = (m2 * c2 + m * l2) * g
    return a1, a2, a3, g1, g2


def _dyn(obs, m, q, qd):
    a1, a2, a3, g1, g2 = _model(obs, m)
    c2, s2 = math.cos(q[1]), math.sin(q[1])
    M = np.array([[a1 + 2 * a3 * c2, a2 + a3 * c2], [a2 + a3 * c2, a2]])
    C = a3 * s2 * np.array([-(2 * qd[0] * qd[1] + qd[1] ** 2), qd[0] ** 2])
    c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
    G = np.array([g1 * c1 + g2 * c12, g2 * c12])
    return M, C, G


def _ref_task(t_seg, p0, p1, T):
    """Task-space position, velocity, acceleration of the min-jerk segment at time t_seg."""
    if t_seg >= T:
        return p1.copy(), np.zeros(2), np.zeros(2)
    s = max(0.0, t_seg / T)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / T
    acc = (60 * s - 180 * s**2 + 120 * s**3) / (T * T)
    d = p1 - p0
    return p0 + d * pos, d * vel, d * acc


def _ref_joint(xy, v, a, L):
    l1, l2 = L
    x, y = xy
    c2 = (x * x + y * y - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    c2 = min(1.0, max(-1.0, c2))
    q2 = math.atan2(-math.sqrt(max(0.0, 1 - c2 * c2)), c2)
    q1 = math.atan2(y, x) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
    q = np.array([q1, q2])
    s1, cc1 = math.sin(q1), math.cos(q1)
    s12, c12 = math.sin(q1 + q2), math.cos(q1 + q2)
    J = np.array([[-l1 * s1 - l2 * s12, -l2 * s12], [l1 * cc1 + l2 * c12, l2 * c12]])
    qd = np.linalg.solve(J, v)
    w1, w12 = qd[0], qd[0] + qd[1]
    Jdq = np.array([-l1 * cc1 * w1 * w1 - l2 * c12 * w12 * w12, -l1 * s1 * w1 * w1 - l2 * s12 * w12 * w12])
    qdd = np.linalg.solve(J, a - Jdq)
    return q, qd, qdd


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    dt = float(obs["sample_dt"])
    if int(obs["step"]) == 0 or not _state:
        _reset(obs)
    seg = int(obs["segment"])
    if seg != _state["seg"]:
        _state["p0"] = _state["prev_target"].copy()
        _state["seg"] = seg
    _state["prev_target"] = np.asarray(obs["target_xy"], float).copy()

    L = np.asarray(obs["link_length"], float)
    p0, p1 = _state["p0"], np.asarray(obs["target_xy"], float)
    T = float(obs["move_time"])
    tau = float(obs["time_in_segment"])
    q = np.asarray(obs["q"], float)
    qd = np.asarray(obs["q_dot"], float)

    xr, vr, ar = _ref_task(tau, p0, p1, T)
    q_d, qd_d, qdd_d = _ref_joint(xr, vr, ar, L)
    e = q_d - q
    e[0] = (e[0] + math.pi) % (2 * math.pi) - math.pi
    ed = qd_d - qd

    # payload estimate
    fric = np.asarray(obs["viscous_friction"], float)
    if obs["payload_known"]:
        m_hat = float(obs["payload_mass"])
    elif P["EST"] == 1:
        m_hat = _ls_update(obs, q, qd, dt, fric, qdd_d)
    else:
        l1, l2 = L
        g = obs["gravity"]
        c1, c12 = math.cos(q[0]), math.cos(q[0] + q[1])
        Y = g * np.array([l1 * c1 + l2 * c12, l2 * c12])
        m_hat = _state["m_hat"] + dt * P["GAMMA"] * float(Y @ (ed + P["LAM"] * e))
        m_hat = min(M_MAX, max(0.0, m_hat))
        _state["m_hat"] = m_hat

    _state["integ"] = np.clip(_state["integ"] + dt * e, -0.5, 0.5)

    M, C, G = _dyn(obs, m_hat, q, qd)
    tau_fb = M @ (P["KP"] * e + P["KD"] * ed) + P["KI"] * _state["integ"]

    # feedforward on the reference, advanced by LEAD to cover delay + lag
    xl, vl, al = _ref_task(tau + P["LEAD"], p0, p1, T)
    ql, qdl, qddl = _ref_joint(xl, vl, al, L)
    Ml, Cl, Gl = _dyn(obs, m_hat, ql, qdl)
    tau_ff = Ml @ qddl + Cl + Gl + fric * qdl

    tau_cmd = tau_fb + tau_ff
    lim = np.asarray(obs["torque_limit"], float)
    u = np.clip(tau_cmd, -lim, lim)
    _state["cmds"].append(u)
    del _state["cmds"][:-3]
    _state["q_prev"], _state["qd_prev"] = q, qd
    return u
