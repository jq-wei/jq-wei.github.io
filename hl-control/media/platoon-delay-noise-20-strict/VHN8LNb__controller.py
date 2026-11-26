"""Delay-compensated constant-time-headway controller with an IMM Kalman predictor.

Structure
---------
1. Own state.  Every observation carries the exact own velocity/acceleration at its
   (delayed) sample time; the commands issued since are kept in ``memory`` and replayed
   through the first-order actuator model to get the present own state.
2. Predecessor state ``[gap, v_pred, a_pred, u_pred, rel_v_bias]`` is estimated by a
   two-mode IMM Kalman filter that knows the sample index of every delayed,
   out-of-order measurement.  The filter is checkpointed at the newest index whose
   measurements have all arrived (MAX_DELAY steps back), re-run over the open window
   each step and predicted to the present.  Mode "calm" models the predecessor command
   as a slow random walk (Q_U_SLOW); mode "manoeuvre" (Q_U_FAST) lets it jump, so a
   sudden leader reversal is picked up from the relative-velocity residuals even when
   acceleration readings are stale or dropped.  Readings of exactly 0.0 are treated as
   acceleration dropouts.
3. Control law (present-time estimates, h = time headway)::

       e      = gap - d0 - h v                    spacing error
       a_des  = (dv + LAMBDA e) / h               gives  e_dot = -LAMBDA e
       a_ref  = a_des   (vehicle 1: a_ref = a_des low-passed with time constant TF1)
       u      = a_ref + TAU d(a_ref)/dt + K_A (a_ref - a)

   With exact estimates the follower's acceleration is its predecessor's passed through
   1/(1 + h s): disturbances are attenuated along the string and the spacing error only
   reacts to (decaying) estimation error.  The first follower is the only one behind a
   step-commanded vehicle; filtering its target acceleration costs it a larger (but
   < 5 m) spacing error and makes its motion band-limited, hence predictable over the
   observation delay for everybody behind it.

Gains
-----
LAMBDA      spacing-error convergence rate [1/s].
K_A         acceleration-tracking feedback on top of actuator-lag inversion (TAU term).
TF1         target-acceleration smoothing of vehicle 1 [s] (TF: other vehicles, 0 = off).
Q_U_*       per-step std of the predecessor-command random walk in each IMM mode.
P_SF, P_FS  IMM mode-switch probabilities per step (calm->manoeuvre, manoeuvre->calm).
Q_W, Q_B    process noise on predecessor velocity and on the relative-velocity bias.
R_*         measurement variances (gap, relative velocity, predecessor acceleration).
"""

from __future__ import annotations

import numpy as np

TAU = 0.45        # nominal actuator lag [s]
MAX_DELAY = 6     # base delay 3 + jitter 3 [steps]
LAMBDA = 0.4
K_A = 1.0
TF1 = 0.5
TF = 0.0
Q_U_SLOW = 0.1
Q_U_FAST = 1.0
P_SF = 0.03
P_FS = 0.15
Q_W = 0.01
Q_B = 0.002
R_GAP = 0.09 ** 2
R_RV = 0.09 ** 2
R_A = 1e-5

_H_G = np.array([1.0, 0, 0, 0, 0])
_H_RV = np.array([0, 1.0, 0, 0, 1.0])
_H_A = np.array([0, 0, 1.0, 0, 0])


def _transition(dt: float) -> np.ndarray:
    al = dt / TAU
    F = np.eye(5)  # state: gap, v_pred, a_pred, u_pred, rel_v bias
    F[2, 2] = 1.0 - al
    F[2, 3] = al
    F[1, 2] = dt * (1.0 - al)
    F[1, 3] = dt * al
    F[0, :] = dt * F[1, :]
    F[0, 0] = 1.0
    return F


def _update_ll(x, P, H, z, r):
    Ph = P @ H
    s = H @ Ph + r
    y = z - H @ x
    k = Ph / s
    return x + k * y, P - np.outer(k, Ph), -0.5 * (y * y / s + np.log(s))


def _predict(st, F, Qs, v_own_next, dt):
    """IMM mixing + per-mode prediction. st = (filters, mode_probs)."""
    filters, mu = st
    T = np.array([[1.0 - P_SF, P_SF], [P_FS, 1.0 - P_FS]])
    cj = mu @ T
    out = []
    for j, Q in enumerate(Qs):
        wts = T[:, j] * mu / cj[j]
        x0 = wts[0] * filters[0][0] + wts[1] * filters[1][0]
        P0 = np.zeros((5, 5))
        for i in range(2):
            dx = filters[i][0] - x0
            P0 += wts[i] * (filters[i][1] + np.outer(dx, dx))
        x = F @ x0
        x[0] -= dt * v_own_next  # g' = g + dt * (w' - v_own')
        out.append((x, F @ P0 @ F.T + Q))
    return out, cj


def _correct(st, meas_list, v_own):
    filters, mu = st
    out, ll = [], np.zeros(2)
    for j, (x, P) in enumerate(filters):
        for g, rv, ap in meas_list:
            if ap is not None:
                x, P, l = _update_ll(x, P, _H_A, ap, R_A)
                ll[j] += l
            x, P, l = _update_ll(x, P, _H_G, g, R_GAP)
            ll[j] += l
            x, P, l = _update_ll(x, P, _H_RV, rv + v_own, R_RV)
            ll[j] += l
        out.append((x, P))
    lik = mu * np.exp(ll - ll.max())
    return out, np.clip(lik / lik.sum(), 1e-6, 1.0)


def controller(obs: dict, memory: dict | None = None):
    dt = float(obs["dt"])
    k = int(obs["step"])
    m = k - int(obs["obs_delay_steps"])
    h = float(obs["time_headway"])
    d0 = float(obs["d0"])
    al = dt / TAU

    if memory is None:
        memory = {
            "F": _transition(dt),
            "Q": [np.diag([0.0, Q_W ** 2, 0.0, q ** 2, Q_B ** 2]) for q in (Q_U_SLOW, Q_U_FAST)],
            "u": {}, "own": {}, "meas": {}, "ckpt": None,
        }
    F, Qs = memory["F"], memory["Q"]
    u_hist, own, meas = memory["u"], memory["own"], memory["meas"]

    # ---- own state: exact (delayed) reading, replayed through the actuator model
    own[m] = (float(obs["own_velocity"]), float(obs["own_acceleration"]))
    cache: dict[int, float] = {}

    def own_v(j):
        if j not in cache:
            base = max(i for i in own if i <= j)
            vv, aa = own[base]
            for i in range(base, j):
                aa = aa + al * (u_hist.get(i, 0.0) - aa)
                vv = vv + dt * aa
            cache[j] = vv
        return cache[j]

    ap = float(obs["predecessor_acceleration"])
    meas.setdefault(m, []).append(
        (float(obs["gap_front"]), float(obs["relative_velocity_front"]), None if ap == 0.0 else ap)
    )

    # ---- checkpoint at the newest index whose measurements have all arrived
    if memory["ckpt"] is None:
        j0 = min(meas)
        g0, rv0, ap0 = meas.pop(j0)[0]
        x0 = np.array([g0, own_v(j0) + rv0, ap0 or 0.0, ap0 or 0.0, 0.0])
        P0 = np.diag([R_GAP, R_RV, 0.5, 1.0, 0.08 ** 2])
        memory["ckpt"] = (j0, ([(x0, P0), (x0.copy(), P0.copy())], np.array([0.9, 0.1])))
    c, filt = memory["ckpt"]
    while c < k - MAX_DELAY:
        filt = _predict(filt, F, Qs, own_v(c + 1), dt)
        c += 1
        if c in meas:
            filt = _correct(filt, meas.pop(c), own_v(c))
    memory["ckpt"] = (c, filt)
    for j in [j for j in meas if j <= c]:
        del meas[j]
    keep = max([i for i in own if i <= c], default=min(own))
    for j in [j for j in own if j < keep]:
        del own[j]
    for j in [j for j in u_hist if j < keep]:
        del u_hist[j]

    # ---- re-run the open window and predict to the present
    for j in range(c, k):
        filt = _predict(filt, F, Qs, own_v(j + 1), dt)
        if j + 1 in meas:
            filt = _correct(filt, meas[j + 1], own_v(j + 1))
    (xs, _), (xf, _) = filt[0]
    mu = filt[1]
    x = mu[0] * xs + mu[1] * xf

    # ---- control at present time k
    v = own_v(k)
    a = own[max(own)][1]
    base = max(own)
    for i in range(base, k):
        a = a + al * (u_hist.get(i, 0.0) - a)
    g, w, a_p = x[0], x[1], x[2]
    e = g - d0 - h * v
    dv = w - v
    a_des = (dv + LAMBDA * e) / h
    a_des_dot = (a_p - a + LAMBDA * (dv - h * a)) / h
    tf = TF1 if int(obs["vehicle_index"]) == 1 else TF
    if tf > 0.0:
        a_ref = memory.get("a_ref", a_des)
        a_ref_dot = (a_des - a_ref) / tf
        memory["a_ref"] = a_ref + dt * a_ref_dot
        a_des, a_des_dot = a_ref, a_ref_dot
    u = a_des + TAU * a_des_dot + K_A * (a_des - a)
    u = float(np.clip(u, -float(obs["brake_limit"]), float(obs["accel_limit"])))
    u_hist[k] = u

    return u, memory
