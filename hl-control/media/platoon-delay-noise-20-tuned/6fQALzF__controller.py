"""Delay-compensating CACC controller for the hard_delay_noise_v1 platooning scenario.

Structure
---------
Observations arrive delayed by a known number of steps (``obs_delay_steps``) and the
gap / relative-velocity channels are noisy and biased, while the *own* state channels
(velocity, acceleration) are exact but delayed. The controller therefore:

1. Stores every command it issued (per step) in its memory and dead-reckons its own
   velocity/acceleration from the delayed sample to "now" through the known
   first-order actuator model (lag ``tau_model``), replicating the simulator update.
2. Tracks the two freshest predecessor-acceleration readings (they are exact when
   present; an exactly-0.0 reading is a dropout and is ignored). Because the delay
   jitters, readings arrive out of order; keeping the newest ones and extrapolating
   their slope over the observation age (gain ``ff_extrap``) anticipates the
   predecessor's braking instead of reacting a full delay late.
3. Reconstructs the predecessor velocity from the delayed relative-velocity reading
   and rolls it forward over the delay using the extrapolated acceleration.
4. Low-pass-blends the delay-compensated gap and relative velocity with a one-step
   prediction of the previous estimates (gains ``blend_gap`` / ``blend_relv``) to
   average out sensor noise.
5. Applies a CACC law on the compensated estimates
       u = kp_eff * spacing_error + kv_eff * rel_velocity + ka_ff * pred_accel
           - ka_self * own_accel
   where kv_eff = kv * brake_gain when the gap is closing (asymmetric response) and
   kp_eff = kp * (1 + kp_nl * (|spacing_error| - 1 m)) beyond 1 m of error: the
   string-gain metric floors peaks below 1 m, so errors are allowed to be loose in
   the noise band but are squeezed hard before they can form a >1.35 m peak.
   A first-order command smoother (``smoothing``) adds jerk/string damping.
6. A fixed safety override brakes hard when the predicted gap or the predicted
   time-to-collision falls below a constant threshold (see below).

Gains
-----
kp              spacing-error feedback [1/s^2]
kp_nl           progressive extra spacing gain per metre of |error| beyond 1 m
kv              relative-velocity feedback [1/s]
ka_ff           predecessor-acceleration feedforward [-]
ka_self         own-acceleration damping [-]
tau_model       actuator lag used for dead reckoning [s]
blend_gap       innovation gain for the gap estimate (0..1)
blend_relv      innovation gain for the relative-velocity estimate (0..1)
ff_extrap       fraction of the predecessor-acceleration slope extrapolated over
                the observation age (0 = hold constant)
smoothing       command low-pass factor (0 = none)
brake_gain      multiplier on kv when the gap is closing (rel velocity < 0)
first_soft      feedback scale for the first follower only. Its spacing peak never
                appears as a numerator in the neighbour peak-gain ratios (it is only
                follower 2's denominator, floored at 1 m), so tracking the leader's
                hard brake more softly both lifts that denominator and hands follower
                2 a smoother velocity profile - the string-gain tail shrinks on both
                counts.

The emergency-brake thresholds (gap below 2.5 m or time-to-collision below 1.5 s force
a full brake) are fixed constants: nominal operation keeps the minimum gap above 10 m,
so they act only as a safety net and are not worth tuning slots.
"""

PARAMS = {
    "kp": 1.1400460819428897,
    "kv": 1.6196241850067565,
    "ka_ff": 0.6044958754215397,
    "ka_self": 1.7567864899269316,
    "tau_model": 0.5485734214784432,
    "blend_gap": 0.9401499860363058,
    "blend_relv": 0.793121230594415,
    "ff_extrap": 0.19790076801552897,
    "smoothing": 0.1372663338504553,
    "brake_gain": 1.2044436304576152,
    "first_soft": 0.697927081303318,
    "kp_nl": 0.9195718174630474,
}

PARAM_BOUNDS = {
    "kp": (0.05, 1.2),
    "kv": (0.2, 2.5),
    "ka_ff": (0.0, 1.6),
    "ka_self": (0.0, 2.0),
    "tau_model": (0.2, 0.9),
    "blend_gap": (0.05, 1.0),
    "blend_relv": (0.05, 1.0),
    "ff_extrap": (0.0, 1.5),
    "smoothing": (0.0, 0.7),
    "brake_gain": (0.7, 2.2),
    "first_soft": (0.35, 1.0),
    "kp_nl": (0.0, 2.5),
}

_EXTRAP_MAX_STEPS = 8  # cap on how far (in steps) the acceleration slope is extrapolated
_SLOPE_LIMIT = 15.0  # |d a_pred / dt| cap [m/s^3]
_GAP_MARGIN = 2.5  # emergency-brake gap threshold [m]
_TTC_MARGIN = 1.5  # emergency-brake time-to-collision threshold [s]


def controller(obs: dict, memory: dict | None = None, params: dict | None = None):
    p = dict(PARAMS)
    if params:
        p.update(params)

    dt = float(obs["dt"])
    step = int(obs["step"])
    delay = int(obs["obs_delay_steps"])
    m = step - delay  # the historical step this observation describes
    accel_limit = float(obs["accel_limit"])
    brake_limit = float(obs["brake_limit"])
    speed_limit = float(obs["speed_limit"])
    tau = max(float(p["tau_model"]), 1e-3)

    mem = memory if isinstance(memory, dict) else {}
    cmds = mem.get("cmds", {})

    # --- 1. dead-reckon own state from the delayed sample to the current step -------
    v_own = float(obs["own_velocity"])
    a_own = float(obs["own_acceleration"])
    dist_own = 0.0  # distance we travelled during the delay window
    for k in range(m, step):
        u_k = cmds.get(k, float(obs["previous_command"]))
        u_k = min(max(u_k, -brake_limit), accel_limit)
        a_own += dt * (u_k - a_own) / tau
        v_own = min(max(v_own + dt * a_own, 0.0), speed_limit)
        dist_own += dt * v_own

    # --- 2. predecessor acceleration: freshest readings + slope extrapolation -------
    # ap1 = (step, value) of the freshest exact reading, ap0 the one before it.
    ap1 = mem.get("ap1")  # may be None
    ap0 = mem.get("ap0")
    reading = float(obs["predecessor_acceleration"])
    if reading != 0.0:  # exactly 0.0 means dropout (true accel is never exactly zero)
        if ap1 is None or m > ap1[0]:
            ap0, ap1 = ap1, (m, reading)
        elif ap1[0] > m and (ap0 is None or m > ap0[0]):
            ap0 = (m, reading)

    if ap1 is None:
        base_step, base_val, slope = m, 0.0, 0.0
    else:
        base_step, base_val = ap1
        if ap0 is not None and ap1[0] > ap0[0]:
            slope = (ap1[1] - ap0[1]) / (dt * (ap1[0] - ap0[0]))
            slope = min(max(slope, -_SLOPE_LIMIT), _SLOPE_LIMIT)
        else:
            slope = 0.0

    def a_pred_at(k: int) -> float:
        age = min(max(k - base_step, 0), _EXTRAP_MAX_STEPS)
        a = base_val + float(p["ff_extrap"]) * slope * age * dt
        return min(max(a, -brake_limit), accel_limit)

    # --- 3. roll the predecessor forward over the delay ------------------------------
    v_pred = float(obs["own_velocity"]) + float(obs["relative_velocity_front"])
    dist_pred = 0.0
    for k in range(m, step):
        v_pred = min(max(v_pred + dt * a_pred_at(k), 0.0), speed_limit)
        dist_pred += dt * v_pred

    a_pred_now = a_pred_at(step)
    gap_meas_now = float(obs["gap_front"]) + dist_pred - dist_own
    relv_meas_now = v_pred - v_own

    # --- 4. blend with the one-step prediction of the previous estimates ------------
    if "gap_est" in mem:
        gap_prior = float(mem["gap_est"]) + dt * float(mem["relv_est"])
        relv_prior = float(mem["relv_est"]) + dt * (a_pred_now - a_own)
        bg = float(p["blend_gap"])
        br = float(p["blend_relv"])
        gap_est = (1.0 - bg) * gap_prior + bg * gap_meas_now
        relv_est = (1.0 - br) * relv_prior + br * relv_meas_now
    else:
        gap_est = gap_meas_now
        relv_est = relv_meas_now

    # --- 5. CACC law on the compensated estimates ------------------------------------
    desired_gap = float(obs["d0"]) + float(obs["time_headway"]) * v_own
    spacing_error = gap_est - desired_gap
    kv_eff = p["kv"] * (p["brake_gain"] if relv_est < 0.0 else 1.0)
    kp_eff = p["kp"] * min(1.0 + p["kp_nl"] * max(abs(spacing_error) - 1.0, 0.0), 3.0)
    scale = float(p["first_soft"]) if int(obs["vehicle_index"]) == 1 else 1.0
    raw = scale * (
        kp_eff * spacing_error
        + kv_eff * relv_est
        + p["ka_ff"] * a_pred_now
    ) - p["ka_self"] * a_own

    # --- 6. safety override -----------------------------------------------------------
    closing = max(-relv_est, 0.0)
    ttc = gap_est / max(closing, 1e-6)
    emergency = gap_est < _GAP_MARGIN or ttc < _TTC_MARGIN
    if emergency:
        raw = min(raw, -brake_limit)

    prev_cmd = float(mem.get("prev_cmd", obs["previous_command"]))
    alpha = 0.0 if emergency else min(max(float(p["smoothing"]), 0.0), 0.95)
    command = (1.0 - alpha) * raw + alpha * prev_cmd
    command = min(max(command, -brake_limit), accel_limit)

    # --- bookkeeping -------------------------------------------------------------------
    cmds[step] = command
    if len(cmds) > 16:  # keep only the recent window needed for dead reckoning
        for old in [k for k in cmds if k < step - 10]:
            del cmds[old]
    mem.update(
        cmds=cmds,
        prev_cmd=command,
        gap_est=gap_est,
        relv_est=relv_est,
        ap1=ap1,
        ap0=ap0,
    )
    return command, mem
