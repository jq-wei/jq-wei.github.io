"""Gate-course controller for the rotor-thrust-level quadrotor (min_time task).

The lap clock only starts when the vehicle leaves the start sphere, so the first 3.5 s are spent hovering 0.14 m
behind the start point and identifying the vehicle and the wind; then a minimum-time plan is solved for this
seed's vehicle and wind and tracked, and re-solved on the way.

1. Estimator.  Attitude R is integrated from the gyro.  In hover a fixed-gain observer on the delayed 50 Hz tracker
   corrects position, velocity and tilt (a tilt error shows as a position residual across the specific force) and the
   heading; the mean correction applied over a window is the gyro bias.  In flight only position and velocity are
   corrected (L_P, L_V), against the estimate stored 30 ms earlier.  The rotor thrusts are tracked through the known
   30 ms lag; with them the accelerometer gives `kappa` (thrust acceleration per newton of command: the hidden
   thrust coefficient over the hidden mass) and the external specific force, i.e. the drag.
   Wind: in hover the heading is swung out to YAW_AMP and back (YAW_T0..YAW_T1), which separates the body-fixed
   accelerometer bias from the world-fixed wind drag by least squares; in flight the wind estimate is corrected
   from the residual between measured and modelled drag (step MU_W, along the principal axes of the drag box).

2. Planner.  Point mass + attitude model (rate command K_ATT * tilt error, rate lag TAU_W, angular acceleration
   cap ALPHA_MAX, rotor differential taken from the collective: K_DELTA), thrust level ETA of the estimated maximum
   (ETA_LAST in the last segment), MuJoCo's box drag in the estimated wind.  The thrust direction follows the
   time-optimal "bilinear tangent" law n(t) ~ q(t), q piecewise linear in time with knots at the gates; SLSQP finds
   the knots and the segment durations (minimum total time; pass within a box of R_PLAN / R_3 / R_LAST around the
   gate centres), warm-started from a solution computed at import.  The plan is re-solved from the current state and
   wind estimate at each gate passage and once part-way through the first segment (N_MID, F_REPLAN).

3. Tracking.  thrust vector = planned thrust + planned drag - modelled drag at the current state - low-passed drag
   residual (TAU_D) + KP * position error + KV * velocity error (feedback capped at FB_MAX); the reference clock
   slows down when the vehicle is behind along the path (WARP).  Its direction (the planned axis keeps a weight of
   at least THR_DIR, so that flips follow the plan) is the attitude command: body rates = K_ATT * tilt error +
   planned rates, rate loop K_RATE (yaw K_RATE_Z, heading held by K_YAW in hover and free in flight) + planned
   angular acceleration; mixer with priority roll/pitch > small yaw differential (Y_MAX) > collective > rest of
   yaw; lead compensation LEAD of the motor lag.  Hover uses the same loops with KP_H, KV_H.
"""

import math

import numpy as np
from scipy.optimize import minimize

from hl_control.envs.quadrotor import nominal_parameters

GAINS = {
    "ETA": 1.0,         # fraction of the maximum collective thrust used by the plan
    "K_ATT": 10.0,      # 1/s, tilt error -> body rate (plan model and controller)
    "TAU_W": 0.05,      # s, rate-loop lag assumed by the plan
    "ALPHA_MAX": 55.0,  # rad/s^2, angular acceleration cap assumed by the plan
    "K_DELTA": 15.5,    # rad/s^2 per newton of rotor differential (worst axis), couples turning and thrust in the plan
    "MU_W": 0.06,       # step of the in-flight wind estimator (per control step)
    "FB_MAX": 15.0,     # m/s^2, cap of the feedback part of the thrust vector
    "THR_DIR": 25.0,    # m/s^2, minimum weight of the planned thrust axis in the attitude command
    "R_PLAN": 0.18,     # m, the plan may pass this far from the centre of gates 1 and 2 (the gate radius is 0.3 m)
    "R_3": 0.12,        # m, the same for gate 3 (the hardest to hit: the top of the course)
    "R_LAST": 0.05,     # m, the same for the last gate
    "ETA_LAST": 0.92,   # thrust level of the last segment relative to ETA (keeps authority for feedback in the final dive)
    "REPLAN": 1,        # 1: re-solve the plan at each gate passage
    "N_MID": 1,         # number of leading segments that also get a re-solve part-way (at F_REPLAN of their duration)
    "F_REPLAN": 0.65,   # fraction of the segment's planned duration at which that happens
    "IT_RE": 25,        # SLSQP iteration cap of a re-solve (40 for the first plan)
    "WARP": 0.5,        # 1/m, slows the reference clock when the vehicle is behind along the path
    "KP": 30.0,         # 1/s^2, position gain while racing
    "KV": 9.0,          # 1/s, velocity gain while racing
    "KP_H": 14.0,       # position gain in hover
    "KV_H": 6.5,        # velocity gain in hover
    "K_RATE": 30.0,     # 1/s, roll/pitch rate loop
    "K_RATE_Z": 8.0,    # 1/s, yaw rate loop
    "K_YAW": 3.0,       # 1/s, heading error -> yaw rate (hover; while racing only the yaw rate is damped)
    "YAW_T0": 0.5,      # s, start of the heading swing during the hover
    "YAW_T1": 3.3,      # s, end of it (back at heading zero)
    "YAW_AMP": 1.8,     # rad, amplitude of the swing
    "Y_MAX": 0.4,       # N per rotor reserved for yaw, taken from the collective when needed
    "LEAD": 2.5,        # motor-lag lead factor (1 = none)
    "TAU_D": 0.06,      # s, low-pass of the external-force residual while racing
    "L_P": 0.5,         # tracker observer gain on position (per frame)
    "L_V": 6.0,         # tracker observer gain on velocity (1/s per frame)
    "T_LAUNCH": 3.5,    # s, launch time (the clock starts at 4 s at the latest)
    "BACK": 0.14,       # m, hover point behind the start along the launch direction
}

_NP = nominal_parameters()
M_NOM = float(_NP["mass"])
J_NOM = np.array(_NP["inertia"], dtype=float)
G = 9.81
DT = 0.005
TMAX = 13.0
MOTOR_TAU = 0.03
LAG_STEP = 1.0 - math.exp(-DT / MOTOR_TAU)
_RP = np.array(_NP["rotor_positions"], dtype=float)
_YC = np.array(_NP["yaw_coeff"], dtype=float)
# rows: collective, roll torque, pitch torque, yaw torque (per newton of rotor thrust)
MIX = np.array([np.ones(4), _RP[:, 1], -_RP[:, 0], _YC])
MIX_INV = np.linalg.inv(MIX)
_FR = np.array(_NP["inertia_frame"], dtype=float)
_CD = np.array(_NP["drag_linear_coeff"], dtype=float)
GATES = np.array([[8.0, 3.0, 3.0], [14.0, -3.0, 1.5], [20.0, 2.0, 3.5], [26.0, 0.0, 2.0]])
START = np.array([0.0, 0.0, 2.0])
N_SEG = 16           # RK4 steps per segment in the plan
N_VAR = 19


# ----------------------------------------------------------------------------- planner

def _cross(a, b):
    return np.stack((a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1], a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2],
                     a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]), axis=1)


_HEAD = [1.0]   # +1: the plan's drag box has the body x axis along world +x, -1: along world -x (heading at launch)


def _model(V, Z, W, Q, c, wind, minv):
    """Batch point-mass model: returns acceleration, dZ/dt, dW/dt, thrust level, drag acceleration."""
    k_att, tau_w, a_max = GAINS["K_ATT"], GAINS["TAU_W"], GAINS["ALPHA_MAX"]
    n = Q / np.sqrt((Q * Q).sum(1))[:, None]
    ca = (n * Z).sum(1)
    yb = np.stack((np.zeros(len(Z)), Z[:, 2], -Z[:, 1]), axis=1)
    yb /= np.sqrt((yb * yb).sum(1))[:, None]
    xb = _cross(yb, Z)
    e0 = (_HEAD[0] * _FR[0, 0]) * xb + _FR[2, 0] * Z
    e1 = (_HEAD[0] * _FR[0, 1]) * xb + _FR[2, 1] * Z
    u = wind - V
    u0, u1, u2 = (u * e0).sum(1), (u * e1).sum(1), (u * yb).sum(1)
    drag = ((_CD[0] * np.abs(u0) * u0)[:, None] * e0 + (_CD[1] * np.abs(u1) * u1)[:, None] * e1
            + (_CD[2] * np.abs(u2) * u2)[:, None] * yb) * minv
    err = _cross(Z, n)
    mag = np.sqrt((err * err).sum(1))
    scale = np.where(ca < 0.0, 1.0 / np.maximum(mag, 1e-6), 1.0)
    dW = (k_att * err * scale[:, None] - W) / tau_w
    amag = np.sqrt((dW * dW).sum(1))
    dW = dW * np.minimum(1.0, a_max / np.maximum(amag, 1e-9))[:, None]
    dZ = _cross(W, Z)
    # the rotors that turn the vehicle are taken from (or added to) the collective: per-rotor differential delta
    delta = np.minimum(amag, a_max) / (GAINS["K_DELTA"] * minv * M_NOM)
    base = np.minimum(TMAX - delta, np.maximum(delta, c * np.maximum(ca, 0.0) / (4.0 * minv)))
    thr = (4.0 * minv) * base
    acc = thr[:, None] * Z + drag
    acc[:, 2] -= G
    return acc, dZ, dW, thr, drag


def _rollout(X, p0, v0, z0, w0, c, wind, minv, n_seg, nodes=None, tail=0.0):
    """Integrate the plan for a batch of decision vectors X (B, 4 n + 3), n gates left: n + 1 thrust-direction knots
    and n segment durations; returns the positions at the segment ends."""
    B = X.shape[0]
    n = (X.shape[1] - 3) // 4
    Qk = X[:, :3 * n + 3].reshape(B, n + 1, 3)
    T = X[:, 3 * n + 3:]
    P, V, Z, W = np.tile(p0, (B, 1)), np.tile(v0, (B, 1)), np.tile(z0, (B, 1)), np.tile(w0, (B, 1))
    ends = []
    t = 0.0
    n_phase = n + 1 if tail > 0.0 else n
    for k in range(n_phase):
        if k < n:
            h = (T[:, k] / n_seg)[:, None]
            qa = Qk[:, k]
            dq = (Qk[:, k + 1] - qa) / n_seg
        else:
            h = np.full((B, 1), tail / n_seg)
            qa = Qk[:, n]
            dq = np.zeros((B, 3))
        ck = c if k < n - 1 else c * GAINS["ETA_LAST"]      # the last segment keeps thrust in hand for feedback
        for i in range(n_seg):
            q0 = qa + dq * i
            qm = q0 + 0.5 * dq
            q1 = q0 + dq
            a1, z1, w1, thr, drag = _model(V, Z, W, q0, ck, wind, minv)
            if nodes is not None:
                nodes.append((t, P[0].copy(), V[0].copy(), a1[0].copy(), Z[0].copy(), W[0].copy(), w1[0].copy(), float(thr[0]), drag[0].copy()))
            a2, z2, w2, _, _ = _model(V + 0.5 * h * a1, Z + 0.5 * h * z1, W + 0.5 * h * w1, qm, ck, wind, minv)
            a3, z3, w3, _, _ = _model(V + 0.5 * h * a2, Z + 0.5 * h * z2, W + 0.5 * h * w2, qm, ck, wind, minv)
            a4, z4, w4, _, _ = _model(V + h * a3, Z + h * z3, W + h * w3, q1, ck, wind, minv)
            P = P + h * (V + (h / 6.0) * (a1 + a2 + a3))
            V = V + (h / 6.0) * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
            Z = Z + (h / 6.0) * (z1 + 2.0 * z2 + 2.0 * z3 + z4)
            Z /= np.sqrt((Z * Z).sum(1))[:, None]
            W = W + (h / 6.0) * (w1 + 2.0 * w2 + 2.0 * w3 + w4)
            t += float(h[0, 0])
        if k < n:
            ends.append(P)
    if nodes is not None:
        a1, z1, w1, thr, drag = _model(V, Z, W, Qk[:, n], c * GAINS["ETA_LAST"], wind, minv)
        nodes.append((t, P[0].copy(), V[0].copy(), a1[0].copy(), Z[0].copy(), W[0].copy(), w1[0].copy(), float(thr[0]), drag[0].copy()))
    return ends


def _solve(x0, p0, v0, z0, w0, c, wind, minv, n_seg, maxiter, gates):
    """Minimum total time through the remaining gates; the passing points may lie in a box of half-size R_PLAN / sqrt(3)
    around the gate centres (offsets o, extra variables). Returns (x, converged, largest constraint violation)."""
    cache = {}
    eps = 1e-5
    nv = len(x0)
    n = (nv - 3) // 4
    eye = np.eye(nv)
    grad = np.concatenate((np.zeros(3 * n + 3), np.ones(n), np.zeros(3 * n)))
    tol = [GAINS["R_PLAN"], GAINS["R_PLAN"], GAINS["R_3"], GAINS["R_LAST"]][4 - n:]
    bounds = [(-8.0, 8.0)] * (3 * n + 3) + [(0.1, 2.5)] * n
    for r in tol:
        bounds += [(-r / math.sqrt(3.0), r / math.sqrt(3.0))] * 3
    minus_eye = -np.eye(3 * n)

    def batch(y):
        x = y[:nv]
        key = x.tobytes()
        if key not in cache:
            X = np.tile(x, (nv + 1, 1))
            X[1:] += eps * eye
            ends = _rollout(X, p0, v0, z0, w0, c, wind, minv, n_seg)
            res = np.concatenate([e - g for e, g in zip(ends, gates)], axis=1)
            cache.clear()
            cache[key] = (res[0], np.hstack(((res[1:] - res[0]).T / eps, minus_eye)))
        return cache[key][0] - y[nv:], cache[key][1]

    cons = [{"type": "eq", "fun": lambda y: batch(y)[0], "jac": lambda y: batch(y)[1]}]
    y0 = np.concatenate((np.asarray(x0, dtype=float), np.zeros(3 * n)))
    try:
        out = minimize(lambda y: float(y[3 * n + 3:nv].sum()), y0, jac=lambda y: grad, bounds=bounds,
                       constraints=cons, method="SLSQP", options={"maxiter": maxiter, "ftol": 2e-4})
        y = np.asarray(out.x, dtype=float)
        ok = bool(out.success)
    except Exception:
        y, ok = y0, False
    viol = float(np.max(np.abs(batch(y)[0])))
    return y[:nv].copy(), ok, viol


def _reference(x, p0, v0, z0, w0, c, wind, minv):
    """Fine (5 ms) reference tables from the plan's nodes: Hermite interpolation of position and velocity."""
    nodes = []
    _rollout(x[None, :], p0, v0, z0, w0, c, wind, minv, N_SEG * 2, nodes=nodes, tail=0.5)
    tn = np.array([n[0] for n in nodes])
    Pn, Vn, An, Zn, Wn, dWn = (np.array([n[j] for n in nodes]) for j in (1, 2, 3, 4, 5, 6))
    thrn = np.array([n[7] for n in nodes])
    Dn = np.array([n[8] for n in nodes])
    tf = np.arange(0.0, tn[-1], DT)
    idx = np.clip(np.searchsorted(tn, tf, side="right") - 1, 0, len(tn) - 2)
    h = (tn[idx + 1] - tn[idx])[:, None]
    s = ((tf - tn[idx]) / h[:, 0])[:, None]
    h00, h10, h01, h11 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s, -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
    lin = lambda Y: Y[idx] * (1.0 - s) + Y[idx + 1] * s
    Z = lin(Zn)
    Z /= np.sqrt((Z * Z).sum(1))[:, None]
    return {
        "p": h00 * Pn[idx] + h10 * h * Vn[idx] + h01 * Pn[idx + 1] + h11 * h * Vn[idx + 1],
        "v": h00 * Vn[idx] + h10 * h * An[idx] + h01 * Vn[idx + 1] + h11 * h * An[idx + 1],
        "z": Z, "w": lin(Wn), "dw": lin(dWn), "thr": lin(thrn[:, None])[:, 0], "d": lin(Dn),
        "T": float(x[3 * ((len(x) - 3) // 4) + 3:].sum()),
    }


def _hover_point():
    d = GATES[0] - START
    d = d / np.linalg.norm(d)
    return START - GAINS["BACK"] * np.array([d[0], d[1], 0.0])


# warm start: thrust-direction knots (start, four gates) and segment durations
X_NOM = np.array([1.037, 1.072, 0.53, -0.268, -1.083, -0.147, -0.024, 2.127, 1.11, 0.82, -1.971, -0.401, -1.058, -2.431, -1.079,
                  1.064, 0.751, 0.837, 0.462])


def _nominal_plan():
    c = GAINS["ETA"] * 4.0 * TMAX / M_NOM
    x, ok, viol = _solve(X_NOM, _hover_point(), np.zeros(3), np.array([0.0, 0.0, 1.0]), np.zeros(3), c, np.zeros(3), 1.0 / M_NOM, N_SEG, 150, GATES)
    return x if viol < 0.02 else X_NOM


X_PLAN = _nominal_plan()


# ----------------------------------------------------------------------------- estimator and control

_S = {}
_I3 = np.eye(3)


def _rot(w):
    """Rotation matrix of the rotation vector w (Rodrigues)."""
    th2 = float(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th2 < 1e-12:
        return _I3 + K
    th = math.sqrt(th2)
    return _I3 + (math.sin(th) / th) * K + ((1.0 - math.cos(th)) / th2) * (K @ K)


def _drag(v, R, wind, kappa):
    """Drag acceleration of the box model at the estimated state (same model as the plan's), and per principal axis
    its sensitivity to the wind along that axis."""
    u = wind - v
    e0 = _FR[0, 0] * R[:, 0] + _FR[2, 0] * R[:, 2]
    e1 = _FR[0, 1] * R[:, 0] + _FR[2, 1] * R[:, 2]
    yb = R[:, 1]
    u0, u1, u2 = float(u @ e0), float(u @ e1), float(u @ yb)
    g0, g1, g2 = kappa * _CD[0] * abs(u0), kappa * _CD[1] * abs(u1), kappa * _CD[2] * abs(u2)
    return (g0 * u0) * e0 + (g1 * u1) * e1 + (g2 * u2) * yb, ((2.0 * g0, e0), (2.0 * g1, e1), (2.0 * g2, yb))


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _reset(obs):
    S = _S
    S.clear()
    yaw = float(obs["tracker_yaw"])
    cy, sy = math.cos(yaw), math.sin(yaw)
    S["R"] = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    S["p"] = np.array(obs["tracker_pos"], dtype=float)
    S["v"] = np.zeros(3)
    S["bg"] = np.zeros(3)
    S["gyro"] = np.array(obs["gyro"], dtype=float)
    S["acc"] = np.array(obs["accel"], dtype=float)
    S["hp"] = np.tile(S["p"], (16, 1))
    S["hy"] = np.full(16, yaw)
    S["T"] = np.full(4, M_NOM * G / 4.0)
    S["kappa"] = 1.0 / M_NOM
    S["d"] = np.zeros(3)
    S["dh"] = np.zeros(3)
    S["dres"] = np.zeros(3)
    S["csum"] = np.zeros(3)
    S["ref"] = None
    S["launch"] = -1
    S["hover"] = _hover_point()
    S["wind"] = np.zeros(3)
    S["N"] = 1e-3 * np.eye(4)
    S["rhs"] = np.zeros(4)


def _launch(S):
    """Estimate the wind from the hover drag, plan the lap from the current state."""
    R, kappa = S["R"], S["kappa"]
    sol = np.linalg.solve(S["N"], S["rhs"])          # world drag (2), body accelerometer bias (2)
    S["d"][0], S["d"][1] = sol[0], sol[1]
    fb = (R.T @ np.array([sol[0], sol[1], 0.0])) / kappa
    wb = np.array([math.copysign(math.sqrt(abs(fb[0]) / _CD[1]), fb[0]), math.copysign(math.sqrt(abs(fb[1]) / _CD[2]), fb[1]), 0.0])
    wind = R @ wb
    wind[2] = 0.0
    S["wind"] = wind
    S["mid"] = [False] * 5
    S["gi"] = 0
    _HEAD[0] = 1.0 if R[0, 0] >= 0.0 else -1.0
    _plan(S, X_PLAN, 0, np.zeros(3), 40)


def _plan(S, x0, gi, w0, maxiter):
    """Plan from the current estimated state through the gates gi..3; keep the old plan if the solve fails."""
    kappa = S["kappa"]
    c = GAINS["ETA"] * 4.0 * TMAX * kappa
    p0, v0, z0, wind = S["p"].copy(), S["v"].copy(), S["R"][:, 2].copy(), S["wind"].copy()
    x, ok, viol = _solve(x0, p0, v0, z0, w0, c, wind, kappa, N_SEG, maxiter, GATES[gi:])
    S["plan_viol"] = viol
    if viol < 0.1 or S["ref"] is None:
        if viol >= 0.1:
            x = np.asarray(x0, dtype=float)
        n = (len(x) - 3) // 4
        S["x"] = x
        S["cum"] = np.concatenate(([0.0], np.cumsum(x[3 * n + 3:])))
        S["plan_gi"] = gi
        S["ref"] = _reference(x, p0, v0, z0, w0, c, wind, kappa)
        S["s"] = 0.0


def _replan(S, gi, w0):
    """Re-solve the rest of the lap from the current state and wind estimate (at each gate passage, and part-way
    through the first N_MID segments), warm-started from the running plan: its thrust-direction knot interpolated
    at the current plan time."""
    x = S["x"]
    n = (len(x) - 3) // 4
    i = gi - S["plan_gi"]
    if i < 0 or i >= n:
        return
    q = x[:3 * n + 3].reshape(n + 1, 3)
    T = x[3 * n + 3:]
    f = min(0.9, max(0.0, (S["s"] * DT - S["cum"][i]) / T[i]))
    q_now = q[i] + f * (q[i + 1] - q[i])
    x0 = np.concatenate((q_now, q[i + 1:].ravel(), [(1.0 - f) * T[i]], T[i + 1:]))
    _plan(S, x0, gi, w0, int(GAINS["IT_RE"]))


def controller(obs: dict) -> np.ndarray:
    """Rotor thrust commands [N], shape (4,). Any internal failure falls back to the hover thrust."""
    try:
        u = _control(obs)
        if u.shape == (4,) and bool(np.all(np.isfinite(u))):
            return u
    except Exception:
        pass
    return np.full(4, min(TMAX, G / (4.0 * _S.get("kappa", 1.0 / M_NOM))))


def _control(obs: dict) -> np.ndarray:
    step = int(obs["step"])
    if step == 0 or not _S:
        _reset(obs)
    S = _S
    P = GAINS
    gyro = np.array(obs["gyro"], dtype=float)
    acc = np.array(obs["accel"], dtype=float)
    R = S["R"]
    racing = S["ref"] is not None
    t = step * DT

    # ---- propagate attitude, position, velocity and rotor thrusts from the previous step
    if step > 0:
        R_new = R @ _rot((0.5 * (gyro + S["gyro"]) - S["bg"]) * DT)
        a_w = 0.5 * (R @ S["acc"] + R_new @ acc)
        a_w[2] -= G
        S["p"] = S["p"] + S["v"] * DT + (0.5 * DT * DT) * a_w
        S["v"] = S["v"] + DT * a_w
        R = R_new
        S["T"] = S["T"] + (np.asarray(obs["last_action"], dtype=float) - S["T"]) * LAG_STEP
    S["gyro"], S["acc"] = gyro, acc

    # ---- tracker frame: delayed position and heading
    i = step % 16
    S["hp"][i] = S["p"]
    S["hy"][i] = math.atan2(R[1, 0], R[0, 0])
    if obs["tracker_fresh"]:
        j = max(step - 6, 0) % 16
        e = np.asarray(obs["tracker_pos"], dtype=float) - S["hp"][j]
        lp, lv = (P["L_P"], P["L_V"]) if racing else (0.36, 2.16)
        S["p"] = S["p"] + lp * e
        S["v"] = S["v"] + lv * e
        S["hp"] += lp * e
        if not racing:
            # tilt error shows as a position residual across the specific force (third observer state)
            fw = R @ acc
            dth = (4.32 / float(fw @ fw)) * np.array([fw[1] * e[2] - fw[2] * e[1], fw[2] * e[0] - fw[0] * e[2], fw[0] * e[1] - fw[1] * e[0]])
            R = _rot(dth) @ R
            S["csum"] += R.T @ dth
        if not racing:
            # heading from the tracker (hover only: near a vertical body x axis the heading is ill-defined)
            ang = 0.15 * _wrap(float(obs["tracker_yaw"]) - S["hy"][j])
            ca, sa = math.cos(ang), math.sin(ang)
            R = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]]) @ R
            S["hy"] += ang
            S["csum"] += R[2] * ang
    S["R"] = R
    p, v = S["p"], S["v"]
    zb = R[:, 2]
    kappa = S["kappa"]
    T_sum = float(S["T"].sum())

    # ---- gyro bias from the mean correction applied in hover, thrust coefficient, external force
    if not racing:
        if step == 200:
            S["csum"][:] = 0.0
        elif step == 400:
            S["bg"] = S["bg"] - S["csum"] / (200 * DT)
            S["csum"][:] = 0.0
        if step > 30 and T_sum > 1.0:
            S["kappa"] = kappa = min(0.95, max(0.6, kappa + (acc[2] / T_sum - kappa) * (DT / 0.3)))
    d_meas = R @ acc - (kappa * T_sum) * zb

    if not racing and t >= P["T_LAUNCH"]:
        S["bg"] = S["bg"] - S["csum"] / ((step - 400) * DT)
        _launch(S)
        S["launch"] = step
        S["dres"] = np.zeros(3)
        racing = True

    w_ff = None
    if racing:
        gi = int(obs.get("gate_index", 0))
        if P["REPLAN"] and gi < 4:
            i = gi - S["plan_gi"]
            if gi != S["gi"]:
                S["gi"] = gi
                _replan(S, gi, R @ (gyro - S["bg"]))
            elif gi < P["N_MID"] and not S["mid"][gi] and 0 <= i < len(S["cum"]) - 1:
                cum = S["cum"]
                if S["s"] * DT - cum[i] >= P["F_REPLAN"] * (cum[i + 1] - cum[i]):
                    S["mid"][gi] = True
                    _replan(S, gi, R @ (gyro - S["bg"]))
        ref = S["ref"]
        k = min(int(S["s"]), len(ref["thr"]) - 1)
        ep = ref["p"][k] - p
        vr = ref["v"][k]
        along = float(ep @ vr) / max(math.sqrt(float(vr @ vr)), 1.0)
        S["s"] += min(1.2, max(0.5, 1.0 - P["WARP"] * along))
        d_mod, sens = _drag(v, R, S["wind"], kappa)
        res = d_meas - d_mod
        om_now = gyro - S["bg"]
        if float(om_now @ om_now) < 25.0:
            # (not while flipping: the accelerometer then also sees the lever arm to the centre of mass)
            S["dres"] = S["dres"] + (res - S["dres"]) * (DT / P["TAU_D"])
            # wind from the drag residual: normalised step along each principal axis of the drag box
            for g, e in sens:
                S["wind"] = S["wind"] + (P["MU_W"] * g / (g * g + 2.0) * float(res @ e)) * e
        fb = ref["d"][k] - d_mod - S["dres"] + P["KP"] * ep + P["KV"] * (vr - v)
        fbn = math.sqrt(float(fb @ fb))
        if fbn > P["FB_MAX"]:
            fb *= P["FB_MAX"] / fbn
        tv = ref["thr"][k] * ref["z"][k] + fb
        # direction command: the planned axis keeps a minimum weight, so that the feedback cannot swing the
        # attitude command around while the planned thrust is low (flips)
        tv_dir = max(ref["thr"][k], P["THR_DIR"]) * ref["z"][k] + fb
        w_ff, dw_ff = ref["w"][k], ref["dw"][k]
    else:
        S["dh"] = S["dh"] + (d_meas - S["dh"]) * (DT / 0.3)
        if t > 0.6:
            # d_meas (horizontal) = drag in world axes + Rz(yaw) * accelerometer bias: accumulate the normal equations
            psi = math.atan2(R[1, 0], R[0, 0])
            cp, sp = math.cos(psi), math.sin(psi)
            A = np.array([[1.0, 0.0, cp, -sp], [0.0, 1.0, sp, cp]])
            S["N"] += A.T @ A
            S["rhs"] += A.T @ d_meas[:2]
        tv = P["KP_H"] * (S["hover"] - p)
        tvn = math.sqrt(float(tv @ tv))
        if tvn > 2.0:
            tv *= 2.0 / tvn
        tv -= P["KV_H"] * v
        tv[0] -= S["dh"][0]
        tv[1] -= S["dh"][1]
        tv[2] += G
        tv_dir = tv

    # ---- attitude: tilt error of the thrust axis, heading held at zero
    zd = tv_dir / max(math.sqrt(float(tv_dir @ tv_dir)), 1e-6)
    cosang = float(zd @ zb)
    fc = max(float(tv @ zb), 1.0)
    ew = np.array([zb[1] * zd[2] - zb[2] * zd[1], zb[2] * zd[0] - zb[0] * zd[2], zb[0] * zd[1] - zb[1] * zd[0]])
    if cosang < 0.0:
        ew = ew / max(math.sqrt(float(ew @ ew)), 1e-6)
    wc = P["K_ATT"] * ew
    if w_ff is not None:
        wc = wc + w_ff
    wc_b = R.T @ wc
    hscale = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if racing:
        wc_b[2] = 0.0
    else:
        # hover: swing the heading out to YAW_AMP and back (YAW_T0 .. YAW_T1) so that the accelerometer bias
        # (body-fixed) separates from the wind drag (world-fixed); the lap is flown from heading zero
        frac = min(1.0, max(0.0, (t - P["YAW_T0"]) / (P["YAW_T1"] - P["YAW_T0"])))
        slope = 2.0 * P["YAW_AMP"] / (P["YAW_T1"] - P["YAW_T0"])
        rate = 0.0 if frac <= 0.0 or frac >= 1.0 else (slope if frac < 0.5 else -slope)
        psi_ref = P["YAW_AMP"] * (1.0 - abs(2.0 * frac - 1.0))
        wc_b[2] = rate + P["K_YAW"] * hscale * max(-0.6, min(0.6, _wrap(psi_ref - math.atan2(R[1, 0], R[0, 0]))))
    om = gyro - S["bg"]
    alpha = np.array([P["K_RATE"] * (wc_b[0] - om[0]), P["K_RATE"] * (wc_b[1] - om[1]), P["K_RATE_Z"] * (wc_b[2] - om[2])])
    if w_ff is not None:
        a_b = R.T @ dw_ff
        alpha[0] += a_b[0]
        alpha[1] += a_b[1]
    Jw = J_NOM @ om
    tau = (J_NOM @ alpha + np.array([om[1] * Jw[2] - om[2] * Jw[1], om[2] * Jw[0] - om[0] * Jw[2], om[0] * Jw[1] - om[1] * Jw[0]])) / (kappa * M_NOM)

    # ---- mixer: roll/pitch torque first, then a small yaw differential (Y_MAX), then the collective;
    #      the rest of the yaw demand only uses what headroom is left
    t_yaw = MIX_INV[:, 3] * tau[2]
    ymax = abs(float(t_yaw[0]))
    f_y = min(1.0, P["Y_MAX"] / max(ymax, 1e-9))
    t_d = MIX_INV[:, 1] * tau[0] + MIX_INV[:, 2] * tau[1] + f_y * t_yaw
    lo, hi = float(t_d.min()), float(t_d.max())
    if hi - lo > TMAX:
        t_d = t_d * (TMAX / (hi - lo))
        lo, hi = float(t_d.min()), float(t_d.max())
    T_des = min(TMAX - hi, max(-lo, fc / kappa / 4.0)) + t_d
    if f_y < 1.0:
        t_more = (1.0 - f_y) * t_yaw
        s = 1.0
        for q in range(4):
            if t_more[q] > 1e-9:
                s = min(s, (TMAX - T_des[q]) / t_more[q])
            elif t_more[q] < -1e-9:
                s = min(s, (0.0 - T_des[q]) / t_more[q])
        T_des = T_des + max(0.0, s) * t_more
    u = S["T"] + P["LEAD"] * (T_des - S["T"])
    return np.minimum(TMAX, np.maximum(0.0, u))
