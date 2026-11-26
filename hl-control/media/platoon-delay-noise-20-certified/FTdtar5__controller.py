"""Delay-compensated CACC: exact own-state Smith predictor + Kalman predecessor predictor.

Control law (one follower i, step k, observation delayed by d steps, timestamp m = k-d):

1. Own state.  Own velocity/acceleration in the observation are exact but old.  Each
   vehicle stores the commands it issued and replays the plant model
   a+ = a + dt/tau (u - a),  v+ = v + dt a+,  x+ = x + dt v+  up to step k.  This gives
   its current a_i, v_i and an odometer x_i, so the own feedback loop is delay-free.
2. Predecessor.  Every observation is an out-of-order sample at timestamp m of the
   predecessor in the own odometer frame: position (gap + odometer, noisy, biased),
   speed (rel. speed + own speed, noisy, biased by b) and acceleration (exact unless
   dropped out, which reads 0.0).  A Kalman filter with state [p, v, a, u_p, b] and the
   same plant model (u_p = predecessor command, random walk; b = rel.-speed bias) runs
   at a fixed lag of KF_LAG steps (all samples of that timestamp have arrived), and is
   re-run forward each step over the fresher samples, then predicted to step k.
   Output: predicted current gap g^, relative speed r^ and predecessor accel a^_p.
3. Spacing law.  eps = g^ - d0 - h v_i,  eps_dot = r^ - h a_i.  We demand
       a_i + h a_i' = w,   w = a^_p + KP * eps + KD * eps_dot,
   so eps obeys eps'' + KD eps' + KP eps = 0 and the string transfer is
   Gamma(s) ~ e^{-theta s} Q(s) / (1 + h s) with Q ~ e^{theta s}, i.e. |Gamma| < 1.
   The first-order actuator is inverted exactly: u = a_i + (tau / h) (w - a_i).

Gains:
  KP  stiffness of the residual spacing-error loop [1/s^2]
  KD  damping of that loop [1/s]
  SIGMA_U  per-step random-walk std of the predecessor command in the filter: larger
      reacts faster to the predecessor's command changes, smaller rejects more noise.
"""

from __future__ import annotations

import numpy as np

TAU = 0.45  # nominal actuator lag [s]
KP = 1.0
KD = 3.0
KF_LAG = 6  # max observation delay: timestamps older than this are complete
SIGMA_GAP = 0.09  # gap noise std [m]
SIGMA_RELV = 0.09  # rel. speed noise std [m/s]
SIGMA_ACC = 0.01  # accel reading std (exact up to dropout) [m/s^2]
SIGMA_U = 0.5  # predecessor command random walk per step [m/s^2]
KEEP = 16  # steps of history kept per vehicle

_H_P = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
_H_V = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
_H_A = np.array([0.0, 0.0, 1.0, 0.0, 0.0])


def _plant(dt: float) -> np.ndarray:
    al = dt / TAU
    # a+ = (1-al) a + al u ; v+ = v + dt a+ ; p+ = p + dt v+
    F = np.eye(5)
    F[2, 2], F[2, 3] = 1.0 - al, al
    F[1] = F[1] + dt * F[2]
    F[0] = F[0] + dt * F[1]
    return F


def _kf_step(x, P, F, Q, samples):
    x = F @ x
    P = F @ P @ F.T + Q
    for H, z, r in samples:
        PH = P @ H
        K = PH / (H @ PH + r)
        x = x + K * (z - H @ x)
        P = P - np.outer(K, PH)
    return x, P


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    dt = float(obs["dt"])
    h = float(obs["time_headway"])
    step = int(obs["step"])
    m = step - int(obs["obs_delay_steps"])
    if memory is None:
        F = _plant(dt)
        G = F[:, 3]  # command noise enters through u
        memory = {"cmds": {}, "own": {}, "meas": {}, "kf": None,
                  "F": F, "Q": np.outer(G, G) * SIGMA_U ** 2 + np.diag([0, 0, 0, 0, 1e-8])}
    cmds, own, meas = memory["cmds"], memory["own"], memory["meas"]
    F, Q = memory["F"], memory["Q"]

    # 1. replay own commands from the observation's timestamp to now
    a = float(obs["own_acceleration"])
    v = float(obs["own_velocity"])
    x = own[m][0] if m in own else 0.0
    own[m] = (x, v, a)
    vmax = float(obs["speed_limit"])
    for k in range(m, step):
        a = a + dt / TAU * (cmds.get(k, 0.0) - a)
        v = min(max(v + dt * a, 0.0), vmax)
        x = x + dt * v
        own[k + 1] = (x, v, a)

    # 2. predecessor sample at timestamp m in the own odometer frame
    p_meas = float(obs["gap_front"]) + own[m][0]
    v_meas = float(obs["relative_velocity_front"]) + own[m][1]
    pa = float(obs["predecessor_acceleration"])
    sample = meas.setdefault(m, [])
    sample.append((_H_P, p_meas, SIGMA_GAP ** 2))
    sample.append((_H_V, v_meas, SIGMA_RELV ** 2))
    if pa != 0.0:  # exactly 0.0 means dropout
        sample.append((_H_A, pa, SIGMA_ACC ** 2))

    if memory["kf"] is None:
        x0 = np.array([p_meas, v_meas, pa, pa, 0.0])
        P0 = np.diag([SIGMA_GAP ** 2, 0.15 ** 2, 0.1 ** 2, 1.0, 0.1 ** 2])
        memory["kf"] = (m, x0, P0)
    t, xf, Pf = memory["kf"]
    while t < step - KF_LAG:  # advance the lagged filter over complete timestamps
        t += 1
        xf, Pf = _kf_step(xf, Pf, F, Q, meas.get(t, ()))
    memory["kf"] = (t, xf, Pf)
    while t < step:  # fresh, possibly incomplete timestamps, then predict to now
        t += 1
        xf, Pf = _kf_step(xf, Pf, F, Q, meas.get(t, ()))
    px, pv, pa_now = xf[0], xf[1], xf[2]

    # 3. spacing law with exact actuator inversion
    eps = (px - x) - float(obs["d0"]) - h * v
    eps_dot = (pv - v) - h * a
    w = pa_now + KP * eps + KD * eps_dot
    u = a + (TAU / h) * (w - a)
    u = min(max(u, -float(obs["brake_limit"])), float(obs["accel_limit"]))

    cmds[step] = u
    for store in (cmds, own, meas):
        for k in [k for k in store if k < step - KEEP]:
            del store[k]
    return u, memory
