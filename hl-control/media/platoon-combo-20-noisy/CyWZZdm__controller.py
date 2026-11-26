"""Delay-compensated constant-time-headway (CTH) platoon follower.

Control law (per follower, evaluated at the *current* step after undoing the delay):

    e      = gap - d0 - h*v                 spacing error
    dv     = v_pred - v                     relative speed
    a_des  = (dv + LAM*e) / HE              CTH law: drives  de/dt = -LAM*e  when HE = h,
                                            so each car low-pass filters its predecessor's
                                            speed with 1/(1 + h s)  (string attenuation)
    u      = a + (tau/TC)*(a_des - a) + tau*d(a_des)/dt
                                            actuator-lag inversion: own acceleration follows
                                            a_des with time constant TC instead of tau

Gains:
    LAM  = 0.6  spacing-error convergence rate [1/s].
    HE   = 1.0  dynamic headway used in a_des (= h: exact CTH behaviour).
    HE1  = 1.6  dynamic headway of follower 1 only: a softer "shock absorber" that
                attenuates the leader's hard brake/recovery pulses (its own spacing-error
                peak stays ~2-4 m, well inside the 5 m bound) so the cars behind see a
                smoother, reachable profile.
    TC   = 0.1  closed-loop acceleration time constant [s] of the lag inversion.

State estimation (all from the local observation, kept in per-vehicle memory):
  * Own speed: `desired_gap = d0 + h*v_own` is computed from the noise-free (delayed) own
    speed, so v_own(step - delay) = (desired_gap - d0)/h exactly.  The actuator lag tau is
    re-identified every step by a 1-D search that best explains the last WIN known speeds
    given our own (exact) command history; the same fit gives the own acceleration, which
    is then propagated to the current step through the lag model.
  * Predecessor: 2-state Kalman filter on (speed, acceleration).  Accel samples are exact
    when not dropped (dropout reads exactly 0.0), speed samples come from relative speed
    (RV = measurement std), QA = accel random-walk std per step.  The gap is integrated
    from the speed difference and pulled to the measured gap with KG / KGV; a slowly
    adapted bias (KB) removes the constant relative-speed sensor bias.  The filter is run
    on settled data (steps <= k-7 have all arrived) and re-propagated to the current
    step every call, so out-of-order/jittered delays are handled exactly.
"""

LAM = 0.6
TC = 0.1
HE = 1.0
HE1 = 1.6
LAM1 = 0.6     # spacing-error rate of follower 1
QA = 0.3       # predecessor accel random-walk std per step [m/s^2]
RV = 0.1       # predecessor speed measurement std [m/s]
KG = 0.10      # gap residual -> gap estimate
KGV = 0.03     # gap residual -> predecessor speed
KB = 0.01      # gap residual -> relative-speed bias estimate
WIN = 25       # own-lag identification window [steps]


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _fit_own(vown, cmd, lo_step, hi_step, umin, umax, dt, tau):
    """Least-squares fit of a(lo) for given tau to the known speeds in [lo, hi]."""
    g = dt / tau
    ks = [i for i in range(lo_step, hi_step + 1) if i in vown]
    p0 = ks[0]; v0 = vown[p0]
    al, be, A, B = 1.0, 0.0, 0.0, 0.0
    sAA = sAy = syy = sAB = sBy = sBB = 0.0
    rows = []
    for i in range(p0 + 1, hi_step + 1):
        u = _clip(cmd.get(i - 1, 0.0), umin, umax)
        al, be = al * (1 - g), be * (1 - g) + g * u
        A += al; B += be
        if i in vown:
            y = (vown[i] - v0) / dt - B
            sAA += A * A; sAy += A * y; syy += y * y
    if sAA < 1e-9:
        return 0.0, None, p0
    a0 = sAy / sAA
    return syy - sAy * a0, a0, p0


def controller(obs, memory=None):
    dt = obs["dt"]; h = obs["time_headway"]; d0 = obs["d0"]
    k = obs["step"]; j = k - obs["obs_delay_steps"]
    umax = obs["accel_limit"]; umin = -obs["brake_limit"]
    if memory is None:
        memory = {"cmd": {}, "v": {}, "gap": {}, "vp": {}, "ap": {}, "tau": 0.45, "F": None, "bv": 0.0}
    m = memory
    cmd, vown, apd = m["cmd"], m["v"], m["ap"]
    vj = (obs["desired_gap"] - d0) / h          # exact own speed at step j
    vown[j] = vj
    m["gap"][j] = obs["gap_front"]
    m["vp"][j] = vj + obs["relative_velocity_front"]
    if obs["predecessor_acceleration"] != 0.0:
        apd[j] = obs["predecessor_acceleration"]

    # --- own lag: 1-D search on the speed-fit residual over a recent window
    L = max(vown)
    lo = max(min(vown), L - WIN)
    tau = m["tau"]
    if len([i for i in range(lo, L + 1) if i in vown]) >= 4:
        best = None
        cands = [tau * f for f in (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25)]
        if k % 10 == 0:
            cands += [0.25, 0.35, 0.5, 0.7, 0.9, 1.2]
        for t in cands:
            t = _clip(t, 0.12, 1.8)
            c = _fit_own(vown, cmd, lo, L, umin, umax, dt, t)[0]
            c += 1e-4 * (t - tau) ** 2 / tau ** 2
            if best is None or c < best[0]:
                best = (c, t)
        tau = best[1]
    m["tau"] = tau
    g = dt / tau
    _, a0, p0 = _fit_own(vown, cmd, lo, L, umin, umax, dt, tau)
    if a0 is None:
        a0 = 0.0
    a, v = a0, vown[p0]
    vmod = {p0: v}
    for i in range(p0, k):
        a = a + g * (_clip(cmd.get(i, 0.0), umin, umax) - a)
        v = v + dt * a
        vmod[i + 1] = v
    a_now, v_now = a, v

    def v_own(i):
        if i in vown:
            return vown[i]
        if i in vmod:
            return vmod[i]
        lo_ = max((s for s in vown if s < i), default=None)
        hi_ = min((s for s in vown if s > i), default=None)
        if lo_ is None:
            return vown[hi_]
        if hi_ is None:
            return vown[lo_]
        return vown[lo_] + (vown[hi_] - vown[lo_]) * (i - lo_) / (hi_ - lo_)

    # --- predecessor Kalman filter (speed, accel) + gap integrator
    def kf_step(st, i, use_meas=True, settle=False):
        gh, vh, ah, P11, P12, P22 = st
        vh += dt * ah
        P11, P12, P22 = P11 + 2 * dt * P12 + dt * dt * P22 + 1e-4, P12 + dt * P22, P22 + QA * QA
        gh += dt * (vh - v_own(i))
        if use_meas:
            if i in apd:  # exact accel sample
                R = 1e-4; s = P22 + R; k1, k2 = P12 / s, P22 / s; r = apd[i] - ah
                vh += k1 * r; ah += k2 * r
                P11, P12, P22 = P11 - k1 * P12, P12 - k1 * P22, P22 - k2 * P22
            if i in m["vp"]:
                R = RV * RV; s = P11 + R; k1, k2 = P11 / s, P12 / s; r = m["vp"][i] - m["bv"] - vh
                vh += k1 * r; ah += k2 * r
                P11, P12, P22 = P11 - k1 * P11, P12 - k1 * P12, P22 - k2 * P12
                rg = m["gap"][i] - gh
                gh += KG * rg; vh += KGV * rg
                if settle:
                    m["bv"] -= KB * rg
        return (gh, vh, ah, P11, P12, P22)

    S = k - 7
    if m["F"] is None:
        j0 = min(m["gap"])
        m["F"] = j0
        m["st"] = (m["gap"][j0], m["vp"][j0], apd.get(j0, 0.0), 0.04, 0.0, 1.0)
    while m["F"] < S:
        m["F"] += 1
        m["st"] = kf_step(m["st"], m["F"], settle=True)
    st = m["st"]
    for i in range(m["F"] + 1, k + 1):
        st = kf_step(st, i)
    gap, vp, ap_h = st[0], st[1], st[2]

    # --- control law
    e = gap - d0 - h * v_now
    dv = vp - v_now
    he = HE1 if obs["vehicle_index"] == 1 else HE
    lam = LAM1 if obs["vehicle_index"] == 1 else LAM
    a_des = (dv + lam * e) / he
    a_des_dot = (ap_h - a_now + lam * (dv - h * a_now)) / he
    u = a_now + (tau / TC) * (a_des - a_now) + tau * a_des_dot
    u = _clip(u, umin, umax)
    cmd[k] = u
    if k % 50 == 0:
        for dct in (cmd, vown, m["gap"], m["vp"], apd):
            for s in [s for s in dct if s < k - 60]:
                del dct[s]
    return u, m
