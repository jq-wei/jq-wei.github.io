"""Delay-compensating, lag-compensating constant-time-headway follower.

Control law (per follower, stateful form ``controller(obs, memory)``):

1. Own-state prediction.  The observation is ``d`` steps old, but own
   velocity/acceleration readings are exact and every issued command is
   remembered, so the first-order actuator model is rolled forward ``d`` steps
   to the present.  The actuator lag ``tau`` is identified online from
   consecutive (a, previous_command) samples.  Out-of-order (older) samples
   are ignored in favour of the freshest one seen.
2. Predecessor estimation.  A Kalman filter on [v_p, a_p, jerk_p] fuses the
   noisy v_p = v + relative_velocity - b every step with the exact
   predecessor-acceleration reading when it is not dropped out (0.0), so
   acceleration changes are still seen through velocity during dropouts.
   ``b`` (relative-velocity sensor bias) is estimated by comparing the
   integral of the relative-velocity readings with the change in measured gap.
   The filter state is extrapolated over the delay (jerk decaying with
   SLOPE_DECAY, capped by SLOPE_MAX) to predict the present gap and dv.
3. Reference acceleration.  With e = gap - d0 - h v, choosing
   a_ref = (dv + LAMBDA * e) / h  gives  de/dt = -LAMBDA * e, i.e. the
   follower's velocity is the ideal 1/(1 + h s) low-pass of its predecessor's
   (string stable).  LAMBDA is the error-recovery rate.
4. Envelope caps.  a_ref is clipped to the peak accel/decel an ideal 1 s CTH
   cascade needs at this index (IDEAL_*) plus margins M_ACC/M_DEC, so no
   vehicle demands more than weaker vehicles behind it can deliver.  Follower 1
   additionally brakes no harder than DEC_CAP_1: it absorbs part of the
   leader's brake pulse as its own (allowed, < 5 m) spacing error and hands a
   gentler wave to the rest.  Caps relax by CAP_RELAX per metre of |e| beyond
   CAP_E0 so no follower drifts past the 5 m bound.
5. Lag inversion.  u = a + tau * da_ref/dt + (tau / T_TRACK) * (a_ref - a)
   makes the actual acceleration track a_ref with time constant T_TRACK.
"""
import numpy as np

LAMBDA = 0.6        # spacing-error recovery rate [1/s]
T_TRACK = 0.25      # acceleration tracking time constant of the lag inversion [s]
SLOPE_DECAY = 0.25  # decay time of the predecessor-jerk state when extrapolating [s]
SLOPE_MAX = 0.5     # max extrapolated predecessor jerk [m/s^2 per step]
TAU_NOM = 0.45      # actuator lag used until it has been identified [s]
Q_JERK = 8.0        # Kalman process noise on the predecessor jerk
R_V = 0.0064        # relative-velocity measurement variance (0.08^2)
M_ACC = 0.3         # margin above the ideal-cascade accel envelope [m/s^2]
M_DEC = 0.5         # margin beyond the ideal-cascade decel envelope [m/s^2]
# Peak accel / decel an ideal 1 s CTH cascade needs at follower i for this leader profile.
IDEAL_ACC = [1.43, 1.10, 0.84, 0.66, 0.55, 0.47, 0.41, 0.36, 0.32, 0.29,
             0.26, 0.24, 0.22, 0.21, 0.19, 0.18, 0.17, 0.15, 0.14]
IDEAL_DEC = [3.58, 2.89, 2.37, 2.00, 1.75, 1.56, 1.41, 1.27, 1.17, 1.08,
             1.01, 0.95, 0.89, 0.84, 0.80, 0.76, 0.73, 0.70, 0.67]
DEC_CAP_1 = 2.3     # follower 1 brakes at most this hard (shapes the wave for the rest) [m/s^2]
CAP_E0 = 2.0        # |spacing error| beyond which the caps relax [m]
CAP_RELAX = 0.5     # cap relaxation per metre of error beyond CAP_E0 [m/s^2 / m]
SETTLE = 8          # samples older than k - SETTLE have all arrived (max delay 7)


def controller(obs: dict, memory: dict | None = None):
    if memory is None:
        memory = {"u": {}, "a": {}, "ap": {}, "rv": {}, "gp": {}, "tau": TAU_NOM, "tau_n": 0,
                  "cur": None, "smp": {}, "kf": None, "I": 0.0, "c0": None, "c0n": 0, "s0": None, "b": 0.0}
    dt = obs["dt"]
    h = obs["time_headway"]
    d0 = obs["d0"]
    k = obs["step"]
    d = obs["obs_delay_steps"]
    s = k - d
    U, A, AP = memory["u"], memory["a"], memory["ap"]

    # --- actuator-lag identification from exact own-acceleration samples ---
    a_s = obs["own_acceleration"]
    A[s] = a_s
    U.setdefault(s, obs["previous_command"])
    U[s] = obs["previous_command"]
    for s0 in (s, s + 1):
        if s0 in A and s0 - 1 in A and s0 in U:
            da = A[s0] - A[s0 - 1]
            if abs(da) > 0.02:
                tau_i = dt * (U[s0] - A[s0 - 1]) / da
                if 0.1 < tau_i < 2.0:
                    n = min(memory["tau_n"], 20)
                    memory["tau"] = (memory["tau"] * n + tau_i) / (n + 1)
                    memory["tau_n"] += 1
    tau = memory["tau"]

    # --- relative-velocity bias: sum(rv * dt) - delta(gap) = b * T ---
    memory["rv"][s] = obs["relative_velocity_front"]
    memory["gp"][s] = obs["gap_front"]
    fin = k - SETTLE
    if memory["cur"] is None:
        if fin >= 0 and 0 in memory["rv"]:
            memory["cur"] = 0
    if memory["cur"] is not None:
        RV, GP = memory["rv"], memory["gp"]
        c = memory["cur"]
        for j in sorted(x for x in RV if c < x <= fin):
            memory["I"] += dt * 0.5 * (RV[j] + RV[c]) * (j - c)
            c = j
            # offset = I - gap; average its first samples as the reference
            off = memory["I"] - GP[j]
            if memory["c0n"] < 20:
                memory["c0"] = off if memory["c0"] is None else (memory["c0"] * memory["c0n"] + off) / (memory["c0n"] + 1)
                memory["c0n"] += 1
                memory["s0"] = j - 10 if memory["c0n"] == 20 else None
            elif memory["s0"] is not None and j - memory["s0"] > 30:
                b_i = (off - memory["c0"]) / ((j - memory["s0"]) * dt)
                memory["b"] += 0.1 * (b_i - memory["b"])
        memory["cur"] = c
    b = memory["b"]

    # --- freshest sample: observations arrive with jittered delay, so an older
    # sample than one already seen is replaced by the stored freshest one ---
    S = memory["smp"]
    S[s] = (obs["own_velocity"], a_s, obs["relative_velocity_front"], obs["gap_front"])
    s = max(S)
    v, a, rv, gap = S[s]

    # --- predecessor Kalman filter, state [v_p, a_p, jerk_p], advanced to the
    # freshest anchor; fuses v_p = v + rv - b (noisy) and a_p (exact unless dropout) ---
    ap_obs = obs["predecessor_acceleration"]
    sd = 1.0 - dt / SLOPE_DECAY
    F = np.array([[1.0, dt, 0.0], [0.0, 1.0, dt], [0.0, 0.0, sd]])
    Q = np.diag([1e-6, 1e-6, Q_JERK])
    if memory["kf"] is None:
        memory["kf"] = (s, np.array([v + rv - b, 0.0, 0.0]), np.diag([0.1, 0.5, 0.5]))
    ks, x, P = memory["kf"]
    if s > ks or s == k - d:
        for _ in range(s - ks):
            x = F @ x
            P = F @ P @ F.T + Q
        if s == k - d:   # this observation is the anchor: measurement update
            Hs = [np.array([1.0, 0.0, 0.0])]
            zs = [v + rv - b]
            Rs = [R_V]
            if ap_obs != 0.0:
                Hs.append(np.array([0.0, 1.0, 0.0])); zs.append(ap_obs); Rs.append(1e-4)
            for Hm, z, r in zip(Hs, zs, Rs):
                Ph = P @ Hm
                g = Ph / (Hm @ Ph + r)
                x = x + g * (z - Hm @ x)
                P = P - np.outer(g, Ph)
        memory["kf"] = (s, x, P)
    vp_kf, ap, slope = x
    slope = max(-SLOPE_MAX, min(SLOPE_MAX, slope * dt))

    # --- roll forward from the anchor to the present step k ---
    vp = vp_kf
    for j in range(s + 1, k + 1):
        u_j = U.get(j, a)
        a = a + dt * (u_j - a) / tau
        v = v + dt * a
        ap += slope
        slope *= sd
        vp = vp + dt * ap
        gap = gap + dt * (vp - v)

    e = gap - d0 - h * v
    dv = vp - v
    a_ref = (dv + LAMBDA * e) / h
    de = dv - h * a
    a_ref_dot = (ap - a + LAMBDA * de) / h
    i = obs["vehicle_index"] - 1
    hi, lo = IDEAL_ACC[i] + M_ACC, -(IDEAL_DEC[i] + M_DEC)
    if i == 0:   # follower 1 shapes the wave: gentle braking unless it gets too close
        lo = max(lo, -DEC_CAP_1 - CAP_RELAX * max(0.0, -e - CAP_E0))
    hi += CAP_RELAX * max(0.0, e - CAP_E0)
    if a_ref > hi:
        a_ref, a_ref_dot = hi, min(a_ref_dot, 0.0)
    elif a_ref < lo:
        a_ref, a_ref_dot = lo, max(a_ref_dot, 0.0)
    u = a + tau * a_ref_dot + (tau / T_TRACK) * (a_ref - a)

    u = max(-obs["brake_limit"], min(obs["accel_limit"], u))
    U[k + 1] = u
    if k % 50 == 0:
        for D in (U, A, AP, memory["rv"], memory["gp"], memory["smp"]):
            for j in [j for j in D if j < k - 30]:
                del D[j]
    return float(u), memory
