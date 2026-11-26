"""Delay-compensated CACC with actuator-lag inversion for a heterogeneous platoon.

Control law (per follower, every quantity reconstructed at the *current* step):

    e     = gap_hat - d0 - h * v_own           spacing error (constant time headway)
    e_dot = (v_pred_hat - v_own) - h * a_own   its exact time derivative
    a_des = KP * e + KD * e_dot + KFF * a_pred_hat,   clipped to [-B_CAP, A_CAP]
    u     = a_own + (tau_hat / T_LAG) * (a_des - a_own)   lag inversion, then clipped

Gains and their roles
    KP      spacing stiffness: pulls the gap back to d0 + h * v_own.
    KD      damping on e_dot. Using e_dot (not the bare relative velocity) makes the
            steady-state spacing error zero while the predecessor holds a constant
            acceleration, which is what keeps peak errors small during the brake pulses.
    KFF     feedforward of the predecessor's acceleration (< 1 so that acceleration
            amplitude is attenuated vehicle by vehicle: peak-sense string stability).
    T_LAG   effective actuator time constant that the inversion emulates for every
            vehicle regardless of its real lag (identified as tau_hat), so the fleet
            behaves homogeneously despite tau in [0.25, 0.83] s.
    A_CAP   shared soft cap on acceleration, just below the smallest possible
            accel_limit (0.7 * 2.0): no vehicle accelerates faster than any follower can.
    B_CAP   shared soft cap on deceleration, below the smallest non-weak brake_limit
            (0.65 * 4.5): the leader's -4 m/s^2 pulses are absorbed by vehicle 1 and
            then arrive at every follower as a profile it can actually track.
    ALPHA   blend weight of each new delay-compensated measurement against the
            one-step prediction of the previous estimate (light noise filtering).
    TTC_MARGIN, GAP_MARGIN   emergency override: full braking (ignoring B_CAP) when
            the predicted time to collision or the predicted gap is too small.
    BIAS_TC ramp-in time of the relative-velocity bias estimate.

State handling (all in the per-vehicle ``memory`` dict)
    * Own velocity and acceleration are exact but delayed by obs_delay_steps, so they
      are rolled forward to the current step with the stored command history and the
      identified lag tau_hat (median of dt*(u_k - a_{k-1})/(a_k - a_{k-1}) over pairs of
      own-acceleration samples one step apart).
    * The predecessor's acceleration reading is exact apart from dropout; a reading of
      exactly 0.0 is treated as a dropout and the last valid value is held.
    * The predecessor's velocity and the gap are rolled forward over the delay with the
      held acceleration and the reconstructed own-velocity trajectory.
    * The constant per-vehicle relative-velocity bias is estimated from the fact that
      the integral of the relative velocity must equal the change of the measured gap,
      and subtracted; without this the bias leaves a standing error of KD*bias/KP.
"""

from __future__ import annotations

PARAMS = {
    "KP": 0.6,
    "KD": 1.5,
    "KFF": 0.75,
    "T_LAG": 0.25,
    "A_CAP": 1.35,      # shared soft cap on commanded acceleration [m/s^2]
    "B_CAP": 2.3,       # shared soft cap on commanded deceleration [m/s^2]
    "ALPHA": 0.5,       # blend weight of the new delay-compensated measurement
    "TTC_MARGIN": 2.0,  # emergency braking below this time to collision [s]
    "GAP_MARGIN": 2.5,
    "BIAS_TC": 6.0,     # ramp-in time of the relative-velocity bias estimate [s]
}


def _tau_update(mem: dict, k: int, a_k: float, u_k: float, dt: float) -> None:
    """Identify the actuator lag from own-acceleration samples at steps k-1 and k."""
    hist = mem["a_hist"]
    hist[k] = a_k
    if k - 1 in hist:
        da = a_k - hist[k - 1]
        if abs(da) > 1e-4:
            est = dt * (u_k - hist[k - 1]) / da
            if 0.15 <= est <= 1.2:
                mem["tau_samples"].append(est)
                mem["tau_samples"] = mem["tau_samples"][-9:]
                s = sorted(mem["tau_samples"])
                mem["tau"] = s[len(s) // 2]
    for key in [kk for kk in hist if kk < k - 12]:
        del hist[key]


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    p = PARAMS
    dt = float(obs["dt"])
    s = int(obs["step"])
    d = int(obs["obs_delay_steps"])
    k = s - d
    d0 = float(obs["d0"])
    h = float(obs["time_headway"])
    amax = float(obs["accel_limit"])
    bmax = float(obs["brake_limit"])
    vmax = float(obs["speed_limit"])

    mem = memory if memory is not None else {
        "cmds": {}, "tau": 0.45, "tau_samples": [], "a_hist": {},
        "ap": 0.0, "ap_idx": -1, "gap": None, "vp": None,
        "k0": None, "gap0": 0.0, "k_last": None, "relv_last": 0.0, "I": 0.0, "bias": 0.0,
    }

    # --- own state: exact but delayed -> roll forward with known commands ------------
    v_k = float(obs["own_velocity"])
    a_k = float(obs["own_acceleration"])
    u_k = float(obs["previous_command"])
    _tau_update(mem, k, a_k, u_k, dt)
    tau = mem["tau"]
    cmds = mem["cmds"]
    v_own_traj = [v_k]
    v, a = v_k, a_k
    for j in range(k, s):
        u = cmds.get(j, u_k)
        a += dt * (u - a) / tau
        v = min(max(v + dt * a, 0.0), vmax)
        v_own_traj.append(v)
    v_own, a_own = v, a

    # --- predecessor acceleration: hold last valid reading (0.0 == dropout) ------------
    ap_read = float(obs["predecessor_acceleration"])
    if ap_read != 0.0 and k >= mem["ap_idx"]:
        mem["ap"], mem["ap_idx"] = ap_read, k
    ap = mem["ap"]

    # --- relative-velocity bias: integral of relv must match the change of the gap -----
    gap = float(obs["gap_front"])
    relv = float(obs["relative_velocity_front"])
    if mem["k0"] is None:
        mem["k0"], mem["gap0"], mem["k_last"], mem["relv_last"] = k, gap, k, relv
    elif k > mem["k_last"]:
        mem["I"] += (k - mem["k_last"]) * dt * 0.5 * (relv + mem["relv_last"])
        mem["k_last"], mem["relv_last"] = k, relv
        span = (k - mem["k0"]) * dt
        if span > 1.0:
            raw_bias = (mem["I"] - (gap - mem["gap0"])) / span
            mem["bias"] = raw_bias * span / (span + p["BIAS_TC"])
    relv -= mem["bias"]

    # --- predecessor state rolled forward over the delay -------------------------------
    vp = v_k + relv
    for j in range(1, d + 1):
        vp = max(vp + dt * ap, 0.0)
        gap += dt * (vp - v_own_traj[j])
    if mem["gap"] is not None:
        vp_pred = max(mem["vp"] + dt * ap, 0.0)
        gap_pred = mem["gap"] + dt * (vp_pred - v_own)
        al = p["ALPHA"]
        gap = al * gap + (1.0 - al) * gap_pred
        vp = al * vp + (1.0 - al) * vp_pred
    mem["gap"], mem["vp"] = gap, vp

    # --- control law -------------------------------------------------------------------
    e = gap - d0 - h * v_own
    dv = vp - v_own
    e_dot = dv - h * a_own
    a_des = p["KP"] * e + p["KD"] * e_dot + p["KFF"] * ap
    a_des = min(max(a_des, -p["B_CAP"]), p["A_CAP"])

    # emergency braking on short predicted time to collision
    if dv < 0.0 and (gap < p["GAP_MARGIN"] or gap / (-dv) < p["TTC_MARGIN"]):
        a_des = min(a_des, -bmax)

    u = a_own + (tau / p["T_LAG"]) * (a_des - a_own)
    u = min(max(u, -bmax), amax)

    cmds[s] = u
    for key in [kk for kk in cmds if kk < s - 12]:
        del cmds[key]
    return u, mem
