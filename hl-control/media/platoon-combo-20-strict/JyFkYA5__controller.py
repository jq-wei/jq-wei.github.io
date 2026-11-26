"""Delay-compensated constant-time-headway (CTH) platoon controller.

Control law for one follower (all quantities predicted to the current step):

    a_des = (dv_hat + KP * e_hat) / h + kff * a_pred_hat
    u     = a_own + (tau_hat / TC) * (a_des - a_own)        # actuator-lag inversion
    u     is clipped to [-min(brake_limit, cap), accel_limit]

With e_dot = dv - h * a_own, the ideal CTH law a_own = (dv + KP*e)/h gives e_dot = -KP*e:
no overshoot, and no amplification along the string.  The pieces that make it work
under delay, noise, dropout and heterogeneity:

* Own state (a_own, v_own) is reconstructed exactly from the delayed own-state reading
  and the stored command history, using the actuator lag tau_hat identified from pairs
  of consecutive own-acceleration samples (own state is noise-free in this scenario).
* Predecessor state (gap, velocity, acceleration, relative-velocity bias) is tracked by a
  4-state Kalman filter fed with the delayed noisy readings (dropped acceleration
  readings, which arrive as exactly 0.0, are skipped) and propagated forward over the
  known observation delay with constant predecessor acceleration.
* KP (1/s): spacing-error gain. Larger values pull transient errors back faster; too
  large amplifies measurement noise through the lag inversion.
* TC (s): closed-loop time constant of the own-acceleration loop; the inversion makes
  every vehicle respond alike despite a 3.4x spread in actuator lag.
* BRAKE_CAP / BRAKE_CAP_1 (m/s^2): caps on commanded braking. The first follower acts as
  the shock absorber for the leader's -4 m/s^2 pulses (its peak error stays below the
  5 m bound), and the uniform downstream cap equalizes heterogeneous brake limits so
  that error peaks shrink along the string.  E_RELIEF / RELIEF_GAIN relax the cap when
  the predicted spacing error drops below -E_RELIEF, protecting against collisions.
* KFF_WEAK / WEAK_THR: anticipatory braking for vehicles whose own brake limit is small
  compared with the predecessor's deceleration (e.g. vehicle 10, 45 % brakes): they
  start braking before the relative velocity builds up.
* T_DECAY (s): optional decay of the predicted predecessor acceleration (0 = none).
* Safety override: full braking when the predicted gap is < 3 m or time-to-collision
  is < 1.5 s.
"""
from __future__ import annotations

import numpy as np

KP = 0.6
KFF = 0.0          # feed-forward of predicted predecessor acceleration
KFF_WEAK = 0.4     # extra feed-forward when own brake limit is weak relative to |a_pred|
TC = 0.15
TAU_DEFAULT = 0.6
BRAKE_CAP = 2.8    # uniform cap on commanded braking [m/s^2] (equalizes heterogeneous vehicles)
ACCEL_CAP = 99.0   # uniform cap on commanded acceleration [m/s^2]
BRAKE_CAP_1 = 3.0  # brake cap for the first follower (absorbs the leader's pulse)
E_RELIEF = 2.0     # brake cap is relaxed by RELIEF_GAIN*(−e − E_RELIEF) when the gap closes too far
RELIEF_GAIN = 1.0
WEAK_THR = 0.5     # pre-braking starts when |a_pred| exceeds WEAK_THR * own brake limit
T_DECAY = 0.0      # time constant [s] with which the predicted predecessor acceleration decays

# Kalman filter noise parameters
R_GAP = 0.08 ** 2
R_DV = 0.08 ** 2
R_AP = 0.01 ** 2
Q_JERK = 0.5 ** 2      # per-step variance of predecessor acceleration change
Q_BIAS = 1e-6


def _own_forward(a, v, cmds, m_from, m_to, tau, dt, vmax):
    for j in range(m_from, m_to):
        u = cmds.get(j, 0.0)
        a = a + dt * (u - a) / tau
        v = min(max(v + dt * a, 0.0), vmax)
    return a, v


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    dt = float(obs["dt"])
    h = float(obs["time_headway"])
    d0 = float(obs["d0"])
    vmax = float(obs["speed_limit"])
    k = int(obs["step"])
    d = int(obs["obs_delay_steps"])
    m_obs = k - d

    mem = memory if memory is not None else {}
    cmds: dict = mem.setdefault("cmds", {})
    a_hist: dict = mem.setdefault("a_hist", {})
    v_own_hist: dict = mem.setdefault("v_hist", {})
    taus: list = mem.setdefault("taus", [])

    a_m = float(obs["own_acceleration"])
    v_m = float(obs["own_velocity"])
    a_hist[m_obs] = a_m

    # ---- actuator lag identification from consecutive own-acceleration samples
    for mm in (m_obs - 1, m_obs):
        if mm in a_hist and (mm + 1) in a_hist and mm in cmds:
            da = a_hist[mm + 1] - a_hist[mm]
            du = cmds[mm] - a_hist[mm]
            if abs(da) > 1e-7 and abs(du) > 1e-4:
                t_est = dt * du / da
                if 0.1 < t_est < 2.0:
                    taus.append(t_est)
    tau = float(np.median(taus[-15:])) if taus else TAU_DEFAULT

    # ---- own state reconstruction to the current step
    a_now, v_now = _own_forward(a_m, v_m, cmds, m_obs, k, tau, dt, vmax)
    # store own velocity history for the predecessor filter propagation
    a_t, v_t = a_m, v_m
    v_own_hist[m_obs] = v_m
    for j in range(m_obs, k):
        a_t = a_t + dt * (cmds.get(j, 0.0) - a_t) / tau
        v_t = min(max(v_t + dt * a_t, 0.0), vmax)
        v_own_hist[j + 1] = v_t

    # ---- predecessor Kalman filter: state [gap, v_pred, a_pred, dv_bias]
    gap_meas = float(obs["gap_front"])
    dv_meas = float(obs["relative_velocity_front"])
    ap_meas = float(obs["predecessor_acceleration"])
    if "x" not in mem:
        mem["x"] = np.array([gap_meas, v_m + dv_meas, 0.0, 0.0])
        mem["P"] = np.diag([R_GAP, R_DV, 1.0, 0.08 ** 2])
        mem["m_f"] = m_obs
    x = mem["x"]
    P = mem["P"]
    m_f = mem["m_f"]

    def predict(x, P, m_from, m_to):
        F = np.array([[1.0, dt, dt * dt, 0.0],
                      [0.0, 1.0, dt, 0.0],
                      [0.0, 0.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0, 1.0]])
        Q = np.diag([1e-4, 1e-4, Q_JERK, Q_BIAS])
        for j in range(m_from, m_to):
            x = F @ x
            x[0] -= dt * v_own_hist.get(j + 1, v_now)
            x[1] = max(x[1], 0.0)
            P = F @ P @ F.T + Q
        return x, P

    if m_obs > m_f:
        x, P = predict(x, P, m_f, m_obs)
        # measurement update
        if ap_meas != 0.0:
            H = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0],
                          [0.0, 0.0, 1.0, 0.0]])
            z = np.array([gap_meas, dv_meas + v_m, ap_meas])
            R = np.diag([R_GAP, R_DV, R_AP])
        else:
            H = np.array([[1.0, 0.0, 0.0, 0.0],
                          [0.0, 1.0, 0.0, 1.0]])
            z = np.array([gap_meas, dv_meas + v_m])
            R = np.diag([R_GAP, R_DV])
        y = z - H @ x
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ H) @ P
        mem["x"], mem["P"], mem["m_f"] = x, P, m_obs
        m_f = m_obs

    # propagate to now with decaying predecessor acceleration (hedge against over-braking)
    gap_hat, v_pred_hat, a_pred_hat = x[0], x[1], x[2]
    dec = np.exp(-dt / T_DECAY) if T_DECAY > 0 else 1.0
    for j in range(m_f, k):
        a_pred_hat *= dec
        v_pred_hat = max(v_pred_hat + dt * a_pred_hat, 0.0)
        gap_hat += dt * (v_pred_hat - v_own_hist.get(j + 1, v_now))
    dv_hat = v_pred_hat - v_now
    e_hat = gap_hat - d0 - h * v_now

    # ---- control law
    bl = float(obs["brake_limit"])
    kff = KFF
    if a_pred_hat < 0.0:
        kff += KFF_WEAK * min(max((-a_pred_hat - WEAK_THR * bl) / bl, 0.0), 1.0)
    a_des = (dv_hat + KP * e_hat) / h + kff * a_pred_hat
    u = a_now + (tau / TC) * (a_des - a_now)

    # safety: predicted gap too small / closing fast
    closing = max(-dv_hat, 0.0)
    if gap_hat < 3.0 or gap_hat / max(closing, 1e-6) < 1.5:
        u = min(u, -float(obs["brake_limit"]))

    cap = BRAKE_CAP_1 if int(obs["vehicle_index"]) == 1 else BRAKE_CAP
    cap += RELIEF_GAIN * max(0.0, -e_hat - E_RELIEF)
    lo = -min(float(obs["brake_limit"]), cap)
    hi = min(float(obs["accel_limit"]), ACCEL_CAP)
    if gap_hat < 3.0 or gap_hat / max(closing, 1e-6) < 1.5:
        lo = -float(obs["brake_limit"])
    u = float(np.clip(u, lo, hi))
    cmds[k] = u

    # prune old history
    if k % 50 == 0:
        for dct in (cmds, a_hist, v_own_hist):
            for key in [kk for kk in dct if kk < k - 40]:
                del dct[key]
    return u, mem
