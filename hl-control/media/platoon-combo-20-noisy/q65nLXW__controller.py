"""Delay-compensated constant-time-headway (CTH) follower with on-line actuator-lag estimation.

Control law, evaluated once per follower per step (stateful form, memory per vehicle):

  1. Predecessor acceleration: a dropped reading is exactly 0.0, so zeros are replaced by the
     last valid value (`ap`).  The reading itself is noise-free.
  2. Actuator-lag estimation: a bank of first-order models with candidate lags TAUS is driven by
     the vehicle's own command history.  Each model's acceleration, delayed by the known
     `obs_delay_steps`, is compared with the noisy own-acceleration reading and scored with an
     exponentially forgotten squared residual (rate BETA).  The best model gives the lag
     estimate `tau_est` (smoothed with TAU_SMOOTH) and the model-based own acceleration.
  3. Delay compensation: the delayed spacing error e and relative velocity dv are propagated
     forward over the known delay with  e' = dv - h*a,  dv' = ap - a,  a' = (u - a)/tau_est,
     using the stored commands, so the law acts on an estimate of the *current* state.
  4. CTH law:  a_des = KP*e + KD*dv.
       KD close to 1/h makes the follower's acceleration a low-passed copy of the
       predecessor's (the exact CTH response keeps e = 0); it is reduced by KD_TAU_REF/tau_est
       for slow actuators to keep phase margin.  KP pulls the spacing error back to zero and
       is the main lever for string attenuation of the residual error peaks.
  5. Lag inversion:  u = a_des + LEAD_FRAC*tau_est*d(a_des)/dt  with
       d(a_des)/dt = KP*(dv - h*a) + KD*(ap - a) evaluated from the model, which cancels most
       of the first-order actuator lag (LEAD_FRAC > 1 over-compensates slightly to make up for
       the lag of the estimator).
  6. Weak-brake anticipation: a vehicle whose brake limit is below WEAK_BRAKE simulates
       ANT_HORIZON s ahead assuming the predecessor keeps decelerating for ANT_HOLD s while it
       brakes at its own limit; if e would still fall below -ANT_ETHRESH it brakes at the limit
       now, because it cannot catch up later.
  7. Safety override: if the stopping deceleration implied by the predicted gap and closing
       speed exceeds SAFE_FRAC of the brake limit, brake at the limit.
  The command is finally clipped to the vehicle's own limits.
"""

KP = 0.60          # spacing-error gain [1/s^2]
KD = 0.90          # relative-velocity gain [1/s]  (~1/h)
KD_TAU_REF = 0.6   # KD is scaled by KD_TAU_REF/tau_est when the estimated lag exceeds this [s]
LEAD_FRAC = 1.3    # lag inversion strength, fraction of the estimated lag [-]
A_BLEND = 0.2      # weight of the noisy own-acceleration reading vs. the model at obs time
BETA = 0.03        # forgetting rate of the lag-model residual scores (per step)
TAU_SMOOTH = 0.1   # smoothing of the lag estimate (per step)
WEAK_BRAKE = 2.6   # brake limit below which the anticipation rule is active [m/s^2]
ANT_HORIZON = 4.0  # prediction horizon of the anticipation check [s]
ANT_HOLD = 1.5     # time the predecessor is assumed to keep its current deceleration [s]
ANT_ETHRESH = 1.0  # predicted spacing error (under full braking) that triggers it [m]
SAFE_MARGIN = 2.0  # gap reserve for the emergency check [m]
SAFE_FRAC = 0.7    # fraction of the brake limit that triggers full braking
TAUS = (0.18, 0.26, 0.36, 0.5, 0.7, 0.95, 1.3)   # candidate actuator lags [s]
MAX_HIST = 12      # stored steps of command / model history (> max delay)


def _init():
    return {
        "cmds": [],                                # applied commands, oldest first
        "models": [[0.0] for _ in TAUS],           # per candidate lag: modelled accel, newest first
        "scores": [0.0 for _ in TAUS],
        "tau_est": 0.45,
        "ap_hold": 0.0,
    }


def controller(obs: dict, memory: dict | None = None):
    dt = float(obs["dt"])
    h = float(obs["time_headway"])
    d = int(obs["obs_delay_steps"])
    mem = memory or _init()
    brake = float(obs["brake_limit"])
    accel = float(obs["accel_limit"])

    # 1. predecessor acceleration with dropout hold
    ap = float(obs["predecessor_acceleration"])
    if ap == 0.0:
        ap = mem["ap_hold"]
    else:
        mem["ap_hold"] = ap

    # 2. actuator-lag estimation from the model bank
    a_meas = float(obs["own_acceleration"])
    scores, models = mem["scores"], mem["models"]
    for c in range(len(TAUS)):
        hist = models[c]
        r = a_meas - hist[min(d, len(hist) - 1)]
        scores[c] = (1.0 - BETA) * scores[c] + BETA * r * r
    best = min(range(len(TAUS)), key=lambda c: scores[c])
    tau_est = (1.0 - TAU_SMOOTH) * mem["tau_est"] + TAU_SMOOTH * TAUS[best]
    mem["tau_est"] = tau_est
    lo = max([c for c in range(len(TAUS)) if TAUS[c] <= tau_est] or [0])
    hi = min(lo + 1, len(TAUS) - 1)
    w = 0.0 if hi == lo else (tau_est - TAUS[lo]) / (TAUS[hi] - TAUS[lo])

    def a_model_at(k):  # modelled own acceleration k steps ago, interpolated between models
        return (1.0 - w) * models[lo][min(k, len(models[lo]) - 1)] + w * models[hi][min(k, len(models[hi]) - 1)]

    a_now = a_model_at(0)
    a = (1.0 - A_BLEND) * a_model_at(d) + A_BLEND * a_meas

    # 3. forward prediction of e, dv over the known delay
    cmds = mem["cmds"]
    n = len(cmds)
    e = float(obs["spacing_error"])
    dv = float(obs["relative_velocity_front"])
    for i in range(d):
        idx = n - d + i
        u_i = cmds[idx] if 0 <= idx < n else float(obs["previous_command"])
        e += dt * (dv - h * a)
        dv += dt * (ap - a)
        a += dt * (u_i - a) / tau_est

    # 4./5. CTH law with lag inversion
    kd = KD * min(1.0, KD_TAU_REF / max(tau_est, 1e-3))
    a_des = KP * e + kd * dv
    a_des_dot = KP * (dv - h * a_now) + kd * (ap - a_now)
    u = a_des + LEAD_FRAC * tau_est * a_des_dot

    # 6. weak-brake anticipation
    if brake < WEAK_BRAKE and ap < 0.0:
        ee, vv, aa = e, dv, a_now
        e_min = ee
        hold_steps = int(round(ANT_HOLD / dt))
        for k in range(int(round(ANT_HORIZON / dt))):
            ap_k = ap if k < hold_steps else 0.0
            ee += dt * (vv - h * aa)
            vv += dt * (ap_k - aa)
            aa += dt * (-brake - aa) / tau_est
            e_min = min(e_min, ee)
        if e_min < -ANT_ETHRESH:
            u = min(u, -brake)

    # 7. safety override on the predicted gap
    if dv < 0.0:
        gap_pred = float(obs["gap_front"]) + (e - float(obs["spacing_error"]))
        room = max(gap_pred - SAFE_MARGIN, 0.3)
        if dv * dv / (2.0 * room) > SAFE_FRAC * brake:
            u = min(u, -brake)
    u = max(-brake, min(accel, u))

    # update the model bank and the command history
    for c, tau_c in enumerate(TAUS):
        hist = models[c]
        models[c] = ([hist[0] + dt * (u - hist[0]) / tau_c] + hist)[:MAX_HIST]
    cmds.append(u)
    mem["cmds"] = cmds[-MAX_HIST:]
    return u, mem
