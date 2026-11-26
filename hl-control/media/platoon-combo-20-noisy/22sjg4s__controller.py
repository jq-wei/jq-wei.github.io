"""Delay-compensating, lag-inverting CTH controller for the noisy heterogeneous platoon.

Control law (evaluated on a prediction of the *current* state, step k):

    a* = (kv * dv + k * e) / h                       target acceleration (ideal CTH)
    u  = a* + tau * da*/dt + (tau / Tc - 1) * (a* - a)

* ``a*`` is the acceleration that makes the spacing error obey  de/dt = -k e  (with kv = 1);
  under perfect actuation the follower's speed is the predecessor's speed low-passed with
  time constant h, which is string stable.  ``kv`` (< 1) softens the relative-speed term,
  ``k`` sets how fast spacing errors are removed.
* The actuator is a first-order lag with unknown, drifting ``tau``.  The second line inverts
  it: the ``tau * da*/dt`` feed-forward (computed analytically from the predecessor's
  acceleration) cancels the lag, and the last term pulls the acceleration onto the target
  with time constant ``Tc``.

State estimation (all local):

* ``desired_gap`` is computed from the exact own speed, so (desired_gap - d0) / h gives the
  exact own speed at the observation's source step; differences of consecutive samples give
  the exact own acceleration.  Every observation is filed under its source step
  (step - obs_delay_steps) so delay jitter is handled.
* ``tau`` is identified online by forgetting-factor least squares on
  a_j - a_i = (dt / tau) * sum(u - a) between pairs of exact acceleration samples
  (``lam`` = forgetting factor), and its log is extrapolated along its recent trend to
  cover the estimator latency (``trend_n`` / ``trend_L`` steps).
* The newest sample is propagated to the present: own acceleration/speed by simulating the
  actuator model with the stored command history, predecessor speed and gap with the last
  valid predecessor-acceleration reading.  A reading of exactly 0.0 is treated as a dropout;
  when the last valid reading is stale, it is blended (weight ``wdrop``) with the mean
  acceleration implied by the measured predecessor speed since then.
"""

import math

P = dict(
    kv=0.85,      # relative-speed gain (1.0 = exact CTH law)
    k=0.9,        # spacing-error convergence rate [1/s]
    Tc=0.1,       # closed-loop actuator time constant after lag inversion [s]
    tau0=0.45,    # prior actuator lag [s]
    lam=0.4,      # forgetting factor of the tau estimator (per sample)
    npair=6,      # max step distance between paired acceleration samples
    xth=0.01,     # minimum excitation sum(u - a) for a tau sample
    trend_n=20,   # window [steps] for the tau drift trend
    trend_L=8,    # extrapolation horizon [steps] for the tau drift trend
    dmin=2,       # staleness [steps] above which a dropped predecessor accel is re-estimated
    wdrop=0.6,    # weight of the speed-derived predecessor acceleration during dropouts
)
TAU_MIN, TAU_MAX = 0.15, 1.6


def _clip_theta(theta, dt):
    return min(max(theta, dt / TAU_MAX), dt / TAU_MIN)


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    dt, h, d0 = obs["dt"], obs["time_headway"], obs["d0"]
    m = memory if memory is not None else {
        "u": {}, "v": {}, "meas": {}, "ap": {}, "a_known": {}, "lt": {},
        "smax": -1, "Pt": 1.0, "Qt": dt / P["tau0"],
    }
    k = obs["step"]
    s = k - obs["obs_delay_steps"]                      # source step of this observation

    # ---- file the observation under its source step ------------------------------------
    m["v"][s] = (obs["desired_gap"] - d0) / h           # exact own speed
    m["meas"][s] = (obs["gap_front"], obs["relative_velocity_front"])
    if obs["predecessor_acceleration"] != 0.0:          # 0.0 == dropout
        m["ap"][s] = obs["predecessor_acceleration"]
    if s == 0:
        m["a_known"][0] = 0.0

    # ---- exact own acceleration and actuator-lag identification ------------------------
    new_a = []
    for j in (s, s + 1):
        if j in m["v"] and (j - 1) in m["v"] and j not in m["a_known"]:
            m["a_known"][j] = (m["v"][j] - m["v"][j - 1]) / dt
            new_a.append(j)
    th0 = _clip_theta(m["Qt"] / m["Pt"], dt)
    for j in new_a:
        j1 = max((i for i in m["a_known"] if j - P["npair"] <= i < j), default=None)
        if j1 is None or any(i not in m["u"] for i in range(j1, j)):
            continue
        ah, x = m["a_known"][j1], 0.0
        for i in range(j1, j):                          # model-interpolated sum(u - a)
            x += m["u"][i] - ah
            ah += th0 * (m["u"][i] - ah)
        y = m["a_known"][j] - m["a_known"][j1]
        if abs(x) > P["xth"]:
            m["Pt"] = P["lam"] * m["Pt"] + x * x
            m["Qt"] = P["lam"] * m["Qt"] + x * y
    theta = _clip_theta(m["Qt"] / m["Pt"], dt)         # theta = dt / tau
    m["lt"][k] = math.log(theta)
    if (k - P["trend_n"]) in m["lt"]:                   # follow the slow drift of tau
        trend = (m["lt"][k] - m["lt"][k - P["trend_n"]]) / P["trend_n"]
        theta = _clip_theta(math.exp(m["lt"][k] + trend * P["trend_L"]), dt)
    tau_hat = dt / theta

    # ---- newest information ------------------------------------------------------------
    m["smax"] = sm = max(m["smax"], s)
    sa = max((j for j in m["a_known"] if j <= sm), default=None)
    if sa is None:
        sa, a = sm, 0.0
    else:
        a = m["a_known"][sa]
    v = m["v"][sa]
    g, rv = m["meas"][sm]
    vp = m["v"][sm] + rv
    sp = max((j for j in m["ap"] if j <= sm), default=None)
    ap = m["ap"][sp] if sp is not None else 0.0
    if sp is not None and sm - sp >= P["dmin"] and sp in m["meas"]:
        avg = (vp - m["v"][sp] - m["meas"][sp][1]) / ((sm - sp) * dt)
        ap += P["wdrop"] * min(1.0, (sm - sp) / 4.0) * (avg - ap)

    # ---- propagate to the present ------------------------------------------------------
    for j in range(sa, k):
        a += theta * (m["u"].get(j, 0.0) - a)
        v += dt * a
        if j >= sm:
            vp += dt * ap
            g += dt * (vp - v)
        elif (j + 1) in m["v"]:
            v = m["v"][j + 1]

    # ---- control law ---------------------------------------------------------------------
    e = g - d0 - h * v
    dv = vp - v
    a_des = (P["kv"] * dv + P["k"] * e) / h
    a_des_dot = (P["kv"] * (ap - a) + P["k"] * (dv - h * a)) / h
    u = a_des + tau_hat * a_des_dot + (tau_hat / P["Tc"] - 1.0) * (a_des - a)
    u = min(max(u, -obs["brake_limit"]), obs["accel_limit"])
    m["u"][k] = u

    if k % 50 == 0:                                     # bound memory
        for key in ("u", "v", "meas", "ap", "a_known", "lt"):
            m[key] = {j: x for j, x in m[key].items() if j > k - 30}
    return u, m
