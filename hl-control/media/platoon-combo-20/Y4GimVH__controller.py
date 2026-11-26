"""Predictive constant-time-headway controller for one platoon follower (hard_combo_v1).

Control law. Every quantity is first dead-reckoned to the *current* time, because the
observation refers to a time ``obs_delay_steps`` in the past:

    a_des = (KV * rv + LAM * e) / h  +  (KFF + KFF_TAU * tau_hat) * (a_pred - a_own)
    u     = a_own + (tau_hat / T_ACT) * (a_des - a_own)

with e = gap - d0 - h * v_own the spacing error and rv = v_pred - v_own.

Why this form. For the constant-time-headway policy the kinematic follower
a = (rv + LAM * e) / h has velocity transfer exactly 1/(h s + 1): string stable for any
LAM, and it needs no feed-forward of the predecessor acceleration. Feeding that
acceleration forward with gain near one (classic CACC) makes followers over-brake
under this spacing policy, which is what produced large positive errors early on.

Roles of the gains
    LAM      spacing-error stiffness (1/s); higher corrects faster, more control effort
    KV_NEG   relative-velocity gain while closing (rv < 0); below 1 deliberately
             under-brakes, because brake authority is large and accel authority is
             small, so over-braking at the end of a pulse is far costlier than
             closing a little during it
    KV_POS   relative-velocity gain while opening (rv > 0)
    KV_FIRST relative-velocity gain used by vehicle 1 only. Vehicle 1 faces the raw
             leader pulse; a softer gain filters it before it reaches the slow or
             weak-braking vehicles behind, whose closing peaks were the main failure
    KFF      lag compensation: T_eff * d(a_des)/dt, approximated by (a_pred - a_own)
    KFF_TAU  extra lag compensation per second of identified actuator lag, since the
             effective lag equals tau (not T_ACT) whenever the command saturates
    T_ACT    target actuator time constant; the identified lag tau_hat is inverted
             down to this value so heterogeneous actuators respond alike
    DECAY_NEG decay rate (1/s) applied to a held *negative* predecessor acceleration
             while predicting through the delay window; pulses end, so a held hard
             brake over-predicts closing. Positive accelerations are held as-is.

State handling
  * Samples are stored by time index (step - obs_delay_steps), so delay jitter and
    out-of-order observations are handled correctly.
  * Own velocity and acceleration are exact, so the actuator lag is identified from
    two consecutive-time samples and own state is propagated to the present with the
    stored command history. This is exact when tau_hat is exact.
  * The predecessor acceleration reads exactly 0.0 on dropout; such readings are
    ignored and the latest valid value is held, decaying slowly once stale.
  * A safety layer commands full braking when predicted gap or time-to-collision
    are small. It has not triggered in any evaluated seed.

Stateful signature: controller(obs, memory) -> (u, memory); memory is per vehicle.
"""

from __future__ import annotations

import math

# ---- gains -----------------------------------------------------------------------
LAM = 1.5
KV_NEG = 0.8
KV_POS = 0.8
KV_FIRST = 0.3
KFF = 0.3
KFF_TAU = 0.4
T_ACT = 0.25
DECAY_NEG = 1.0
TAU_DEFAULT = 0.45
MAX_TAU_RATIO = 4.0        # cap on the tau_hat / T_ACT lead gain
# safety layer
SAFE_GAP = 2.5
SAFE_TTC = 1.6
# memory horizon (steps)
MAX_HIST = 40


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def controller(obs: dict, memory: dict | None = None):
    dt = float(obs["dt"])
    step = int(obs["step"])
    delay = int(obs["obs_delay_steps"])
    m_idx = step - delay                      # time index this observation refers to
    d0 = float(obs["d0"])
    h = float(obs["time_headway"])
    vmax = float(obs["speed_limit"])
    a_lim = float(obs["accel_limit"])
    b_lim = float(obs["brake_limit"])

    mem = memory if memory is not None else {}
    samples: dict = mem.setdefault("samples", {})
    cmds: dict = mem.setdefault("cmds", {})   # step -> clipped command issued at that step
    taus: list = mem.setdefault("taus", [])

    # ---- store this observation by its time index ----------------------------------
    a_pred_raw = float(obs["predecessor_acceleration"])
    samples[m_idx] = {
        "gap": float(obs["gap_front"]),
        "rv": float(obs["relative_velocity_front"]),
        "v": float(obs["own_velocity"]),
        "a": float(obs["own_acceleration"]),
        "u": float(obs["previous_command"]),
        "ap": None if a_pred_raw == 0.0 else a_pred_raw,   # 0.0 is the dropout value
    }
    for k in [k for k in samples if k < step - MAX_HIST]:
        del samples[k]
    for k in [k for k in cmds if k < step - MAX_HIST]:
        del cmds[k]

    # ---- identify actuator lag from two consecutive-time samples -------------------
    prev = samples.get(m_idx - 1)
    if prev is not None:
        cur = samples[m_idx]
        da = cur["a"] - prev["a"]
        du = cur["u"] - prev["a"]
        if abs(da) > 1e-4 and abs(du) > 1e-2:
            t_est = dt * du / da
            if 0.15 <= t_est <= 1.2:
                taus.append(t_est)
                if len(taus) > 15:
                    del taus[0]
    if taus:
        srt = sorted(taus)
        tau_hat = srt[len(srt) // 2]
    else:
        tau_hat = TAU_DEFAULT

    # ---- newest sample and latest valid predecessor acceleration -------------------
    newest = max(samples)
    s = samples[newest]
    ap_hat = 0.0
    ap_age = None
    for k in sorted(samples, reverse=True):
        if samples[k]["ap"] is not None:
            ap_hat = samples[k]["ap"]
            ap_age = step - k
            break
    if ap_age is not None and ap_age > 12:
        ap_hat *= math.exp(-(ap_age - 12) * dt / 1.0)

    # ---- dead-reckon own state and the predecessor to the current step -------------
    v_o = s["v"]
    a_o = s["a"]
    v_p = s["rv"] + s["v"]
    gap = s["gap"]
    ap_k = ap_hat
    for j in range(newest + 1, step + 1):
        u_j = cmds.get(j - 1, s["u"])
        a_o += dt * (u_j - a_o) / tau_hat
        v_o = _clip(v_o + dt * a_o, 0.0, vmax)
        v_p = _clip(v_p + dt * ap_k, 0.0, vmax)
        if ap_k < 0.0:
            ap_k *= math.exp(-DECAY_NEG * dt)
        gap += dt * (v_p - v_o)
    rv = v_p - v_o

    # ---- control law ---------------------------------------------------------------
    e = gap - d0 - h * v_o
    kv = KV_NEG if rv < 0.0 else KV_POS
    if int(obs["vehicle_index"]) == 1:
        kv = KV_FIRST
    a_des = (kv * rv + LAM * e) / h + (KFF + KFF_TAU * tau_hat) * (ap_hat - a_o)
    lead = min(tau_hat / T_ACT, MAX_TAU_RATIO)
    u = a_o + lead * (a_des - a_o)

    # ---- safety layer --------------------------------------------------------------
    closing = max(-rv, 0.0)
    ttc = gap / max(closing, 1e-6)
    if gap < SAFE_GAP or ttc < SAFE_TTC:
        u = min(u, -b_lim)

    u = _clip(u, -b_lim, a_lim)
    cmds[step] = u
    return u, mem
