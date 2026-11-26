"""Delay-compensated CTH controller with a predecessor Kalman filter.

Control law (evaluated at the *current* time, using predicted states):

    e   = gap - d0 - h * v                      spacing error
    a*  = (dv + K_E * e) / h                    acceleration that makes  de/dt = -K_E * e
    a*' = (a_p - a + K_E * (dv - h * a)) / h    its time derivative
    u   = a* + TAU * a*' + K_A * (a* - a)       lag compensation (+ extra tracking gain)

With perfect information this gives e(t) -> 0 independently for every follower and
each vehicle's acceleration is the predecessor's low-passed with time constant h, so
disturbances cannot amplify down the string.

Information handling (all local, per vehicle, kept in ``memory``):

* Own state: the actuator model ``a += dt (u - a) / TAU`` is replayed from the latest
  (delayed, exact) own reading using our own stored command history, so own v and a are
  known at the current step despite the observation delay.
* Predecessor: a Kalman filter over z = [gap, dv, a_p, u_p, b_v] (u_p = predecessor's
  actuator command, b_v = constant rel-velocity sensor bias), driven by our own known
  acceleration. Measurements are filed by their timestamp ``step - obs_delay_steps``;
  a committed filter is advanced only once a timestamp can no longer receive late
  measurements, and the current estimate is re-run from it each step. That handles
  jittered / out-of-order delays. A predecessor acceleration reading of exactly 0.0
  is treated as a dropout. Prediction holds u_p, so a predecessor brake step is
  extrapolated through its actuator lag.

Gains: K_E is the spacing-error convergence rate [1/s]; K_A speeds up tracking of a*;
Q_U is the random-walk std of the predecessor command per step (filter agility);
R_GAP / R_DV / R_ACC are measurement noise stds.
"""

from __future__ import annotations

import numpy as np

TAU = 0.45        # nominal actuator lag [s]
K_E = 0.5         # spacing-error convergence rate [1/s]
K_A = 0.0         # extra a* tracking gain
Q_U = 0.35        # predecessor command random-walk std per step [m/s^2]
Q_A = 0.02        # predecessor acceleration model noise per step [m/s^2]
R_GAP = 0.09
R_DV = 0.09
R_ACC = 0.01
COMMIT_LAG = 6    # steps after which no late measurement is expected
B_V0 = 0.08       # prior std of the rel-velocity bias


def _model(dt):
    al = dt / TAU
    # z' = F z + G * a_own_next
    F = np.array([
        [1.0, dt, dt * dt * (1 - al), dt * dt * al, 0.0],
        [0.0, 1.0, dt * (1 - al), dt * al, 0.0],
        [0.0, 0.0, 1 - al, al, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
    ])
    G = np.array([-dt * dt, -dt, 0.0, 0.0, 0.0])
    Q = np.diag([1e-6, 1e-6, Q_A ** 2, Q_U ** 2, 0.0])
    return F, G, Q


def _update(z, P, i, y, r, bias=False):
    """Scalar KF update for measurement z[i] (+ z[4] when ``bias``)."""
    PH = P[:, i] + P[:, 4] if bias else P[:, i]
    s = PH[i] + (PH[4] if bias else 0.0) + r * r
    k = PH / s
    z = z + k * (y - z[i] - (z[4] if bias else 0.0))
    P = P - np.outer(k, PH)
    return z, P


def _apply(z, P, meas):
    for g, dv, ap in meas:
        z, P = _update(z, P, 0, g, R_GAP)
        z, P = _update(z, P, 1, dv, R_DV, bias=True)
        if ap is not None:
            z, P = _update(z, P, 2, ap, R_ACC)
    return z, P


def _own_step(v, a, u, dt, vmax):
    a = a + dt * (u - a) / TAU
    v = min(max(v + dt * a, 0.0), vmax)
    return v, a


def controller(obs: dict, memory: dict | None = None, params: dict | None = None):
    del params
    dt = float(obs["dt"])
    h = float(obs["time_headway"])
    d0 = float(obs["d0"])
    vmax = float(obs["speed_limit"])
    umax = float(obs["accel_limit"])
    umin = -float(obs["brake_limit"])
    k = int(obs["step"])
    ts = k - int(obs["obs_delay_steps"])

    ap_raw = float(obs["predecessor_acceleration"])
    meas = (float(obs["gap_front"]), float(obs["relative_velocity_front"]),
            None if ap_raw == 0.0 else ap_raw)

    if memory is None:
        F, G, Q = _model(dt)
        z0 = np.array([meas[0], meas[1], meas[2] or 0.0, meas[2] or 0.0, 0.0])
        P0 = np.diag([R_GAP ** 2, R_DV ** 2, 0.5 ** 2, 1.0 ** 2, B_V0 ** 2])
        memory = {
            "F": F, "G": G, "Q": Q,
            "own": {ts: (float(obs["own_velocity"]), float(obs["own_acceleration"]))},
            "cmd": {},
            "meas": {},
            "T": ts, "z": z0, "P": P0,  # committed filter = prior at timestamp T
        }
    F, G, Q = memory["F"], memory["G"], memory["Q"]
    own, cmd, buf = memory["own"], memory["cmd"], memory["meas"]

    # --- own state: overwrite with exact delayed reading, replay commands up to now
    if ts >= memory.setdefault("own_ts", ts):
        memory["own_ts"] = ts
        own[ts] = (float(obs["own_velocity"]), float(obs["own_acceleration"]))
    for j in range(memory["own_ts"], k):
        own[j + 1] = _own_step(*own[j], cmd.get(j, 0.0), dt, vmax)
    v_now, a_now = own[k]

    # --- predecessor filter
    if ts >= memory["T"]:
        buf.setdefault(ts, []).append(meas)
    T, z, P = memory["T"], memory["z"], memory["P"]
    while T < k - COMMIT_LAG:
        z, P = _apply(z, P, buf.pop(T, ()))
        z, P = F @ z + G * own[T + 1][1], F @ P @ F.T + Q
        T += 1
    memory["T"], memory["z"], memory["P"] = T, z, P
    for j in range(T, k):
        z, P = _apply(z, P, buf.get(j, ()))
        z, P = F @ z + G * own[j + 1][1], F @ P @ F.T + Q
    z, P = _apply(z, P, buf.get(k, ()))

    gap, dv, a_p = float(z[0]), float(z[1]), float(z[2])

    # --- control law
    e = gap - d0 - h * v_now
    a_star = (dv + K_E * e) / h
    a_star_dot = (a_p - a_now + K_E * (dv - h * a_now)) / h
    u = a_star + TAU * a_star_dot + K_A * (a_star - a_now)
    u = float(np.clip(u, umin, umax))

    cmd[k] = u
    for d in (own, cmd):
        for j in [j for j in d if j < T - 2]:
            del d[j]
    return u, memory
