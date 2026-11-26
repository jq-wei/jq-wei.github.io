"""Decentralized platoon follower: exact own-state replay, predecessor Kalman filter, CACC law.

Control law (per follower, per step)
------------------------------------
1. Own-state reconstruction. The observation is delayed by ``obs_delay_steps`` (known
   per observation). Own velocity and acceleration are noise-free, and the controller
   remembers every command it issued, so the *current* own state is obtained exactly by
   replaying the first-order actuator model ``a += dt (u - a) / tau`` over the delay.
2. Predecessor estimation. Each gap / relative-velocity reading (noisy, biased) and the
   dropout-affected predecessor acceleration (exact when present, exactly 0.0 when
   dropped, so 0.0 is treated as missing) is buffered under its true timestamp
   ``step - delay``. A 5-state Kalman filter on

       [gap, relative velocity, predecessor acceleration, predecessor command,
        relative-velocity bias]

   is advanced over the buffered measurements, which copes with out-of-order arrivals
   and repeated timestamps, and then extrapolated to the current step with the own
   acceleration as a known input. Modelling the predecessor's command through the same
   actuator lag lets the extrapolation continue an acceleration ramp instead of
   freezing it. The filter state at ``step - 7`` is final (no more data can arrive for
   it); everything newer is recomputed each step from that anchor.
3. CACC-style law on the estimated current state::

       a_des = KP * spacing_error + KV * rel_velocity + KA * a_pred
       u     = a_own + (TAU / LAG_T) * (a_des - a_own)

   ``KP`` pulls the spacing error back to zero; ``KV`` damps the relative velocity;
   ``KA`` feeds the predecessor acceleration forward. ``KA + h*KV = 1`` makes the law
   consistent with the constant-time-headway policy: the steady-state spacing error
   under a constant predecessor acceleration a_p is ``a_p (1 - KA - h KV) / KP``, and
   with the sum equal to one the ideal loop is string stable for any ``KP``. Most of
   the weight sits on ``KV`` because the relative velocity is measured directly while
   the predecessor acceleration is the signal most affected by delay and dropout.
   ``LAG_T`` is the actuator time constant after compensation, so ``TAU / LAG_T`` is
   the compensation gain; gentler than full compensation keeps each vehicle a
   predictable predecessor for the vehicle behind it. A safety override brakes hard
   when the predicted gap or time-to-collision is small; otherwise the command slew
   is bounded by ``JERK_LIMIT``.

Gains were chosen with the local simulator on 200 seeds (tuned on 0-99, validated on
100-199): spacing-error RMS about 0.25 m, no gap below 15 m, string peak gain at most
1.29 (limit 1.35).
"""

from __future__ import annotations

import math

import numpy as np

# ---- control gains -------------------------------------------------------------
KP = 0.20          # spacing-error gain [1/s^2]
KV = 0.80          # relative-velocity gain [1/s]
KA = 0.20          # predecessor-acceleration feed-forward [-]   (KA + h*KV = 1 with h = 1 s)
LAG_T = 0.25       # actuator time constant after lag compensation [s]
TAU = 0.45         # nominal actuator lag [s]
JERK_LIMIT = 12.0  # command slew limit [m/s^3]
GAP_MARGIN = 2.5   # emergency braking below this predicted gap [m]
TTC_MARGIN = 1.8   # emergency braking below this predicted time-to-collision [s]

# ---- estimator parameters ------------------------------------------------------
Q_CMD = 0.5        # random-walk std of the predecessor command per step [m/s^2]
R_GAP = 0.09       # gap measurement noise std [m]
R_RELV = 0.09      # relative-velocity measurement noise std [m/s]
R_ACC = 1e-2       # predecessor acceleration reading noise std (exact when present) [m/s^2]
B_RELV0 = 0.08     # prior std of the relative-velocity sensor bias [m/s]
MAX_DELAY = 6      # steps; timestamps older than step - MAX_DELAY receive no more data

H_GAP = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
H_RELV = np.array([0.0, 1.0, 0.0, 0.0, 1.0])
H_ACC = np.array([0.0, 0.0, 1.0, 0.0, 0.0])


def _own_state_now(mem: dict, obs: dict, step: int, k: int, dt: float) -> tuple[float, float]:
    """Replay own commands from the delayed sample at timestamp ``k`` to the current step."""
    v = float(obs["own_velocity"])
    a = float(obs["own_acceleration"])
    cmds = mem["u"]
    own = mem["own"]
    own[k] = (v, a)
    v_lim = float(obs["speed_limit"])
    for j in range(k, step):
        u = cmds.get(j, 0.0)
        a = a + dt * (u - a) / TAU
        v = min(max(v + dt * a, 0.0), v_lim)
        own[j + 1] = (v, a)
    return v, a


def _transition(dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-step model for [gap, rel-v, a_pred, u_pred, bias]; own acceleration enters as input."""
    al = dt / TAU
    F = np.array(
        [
            [1.0, dt, dt * dt * (1.0 - al), dt * dt * al, 0.0],
            [0.0, 1.0, dt * (1.0 - al), dt * al, 0.0],
            [0.0, 0.0, 1.0 - al, al, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    Q = np.zeros((5, 5))
    Q[3, 3] = Q_CMD ** 2
    return F, Q


def _kf_update(z, P, H, y, r):
    S = float(H @ P @ H + r)
    K = P @ H / S
    z = z + K * (y - float(H @ z))
    P = P - np.outer(K, H @ P)
    return z, P


def _run_filter(z, P, k_from, k_to, meas, own, dt, brake_limit, accel_limit):
    """Advance the filter from timestamp ``k_from`` to ``k_to`` applying buffered measurements."""
    F, Q = _transition(dt)
    for k in range(k_from + 1, k_to + 1):
        a_own_next = own.get(k, own.get(k - 1, (0.0, 0.0)))[1]
        z = F @ z
        z[0] -= dt * dt * a_own_next
        z[1] -= dt * a_own_next
        P = F @ P @ F.T + Q
        for gap, relv, ap in meas.get(k, ()):
            z, P = _kf_update(z, P, H_GAP, gap, R_GAP ** 2)
            z, P = _kf_update(z, P, H_RELV, relv, R_RELV ** 2)
            if ap is not None:
                z, P = _kf_update(z, P, H_ACC, ap, R_ACC ** 2)
        z[3] = min(max(z[3], -brake_limit), accel_limit)
    return z, P


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    dt = float(obs["dt"])
    step = int(obs["step"])
    k = step - int(obs["obs_delay_steps"])  # timestamp of this observation
    d0 = float(obs["d0"])
    h = float(obs["time_headway"])
    accel_limit = float(obs["accel_limit"])
    brake_limit = float(obs["brake_limit"])

    mem = memory if memory is not None else {"u": {}, "own": {}, "meas": {}, "filt": None}

    # 1. exact own state at the current step
    v_own, a_own = _own_state_now(mem, obs, step, k, dt)

    # 2. buffer the predecessor measurement under its true timestamp
    ap_read = float(obs["predecessor_acceleration"])
    ap = None if ap_read == 0.0 else ap_read  # dropout reads exactly 0.0
    gap_m = float(obs["gap_front"])
    relv_m = float(obs["relative_velocity_front"])
    mem["meas"].setdefault(k, []).append((gap_m, relv_m, ap))

    if mem["filt"] is None:
        z0 = np.array([gap_m, relv_m, 0.0, 0.0, 0.0])
        P0 = np.diag([R_GAP ** 2, R_RELV ** 2 + B_RELV0 ** 2, 0.05 ** 2, 0.1 ** 2, B_RELV0 ** 2])
        mem["filt"] = (k, z0, P0)

    # advance the final filter state while every measurement for the next timestamp is in
    k_f, z_f, P_f = mem["filt"]
    k_final = step - MAX_DELAY - 1
    if k_final > k_f:
        z_f, P_f = _run_filter(z_f, P_f, k_f, k_final, mem["meas"], mem["own"], dt, brake_limit, accel_limit)
        for kk in [kk for kk in mem["meas"] if kk <= k_final]:
            del mem["meas"][kk]
        for kk in [kk for kk in mem["own"] if kk < k_final]:
            del mem["own"][kk]
        for kk in [kk for kk in mem["u"] if kk < k_final]:
            del mem["u"][kk]
        k_f = k_final
        mem["filt"] = (k_f, z_f, P_f)

    # provisional run to the current step: measurements where available, prediction beyond
    z, _ = _run_filter(z_f, P_f, k_f, step, mem["meas"], mem["own"], dt, brake_limit, accel_limit)
    g_hat, w_hat, ap_hat = float(z[0]), float(z[1]), float(z[2])

    # 3. control law with actuator lag compensation
    e_hat = g_hat - d0 - h * v_own
    a_des = KP * e_hat + KV * w_hat + KA * ap_hat
    a_des = min(max(a_des, -brake_limit), accel_limit)
    u = a_own + (TAU / LAG_T) * (a_des - a_own)

    # safety override on predicted gap / time-to-collision, otherwise slew limit
    closing = max(-w_hat, 0.0)
    ttc = g_hat / max(closing, 1e-6)
    if g_hat < GAP_MARGIN or ttc < TTC_MARGIN:
        u = min(u, -0.9 * brake_limit - 0.5 * closing)
    else:
        prev = mem["u"].get(step - 1, float(obs["previous_command"]))
        u = min(max(u, prev - JERK_LIMIT * dt), prev + JERK_LIMIT * dt)

    u = float(min(max(u, -brake_limit), accel_limit))
    if not math.isfinite(u):
        u = 0.0
    mem["u"][step] = u
    return u, mem
