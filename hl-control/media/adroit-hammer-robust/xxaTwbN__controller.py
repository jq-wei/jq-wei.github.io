"""Adroit hammer: scripted joint-space grasp, lift and yaw-swing strike, with feedback on the hammer-head height
and on the nail depth.  No learning, no simulator in the loop: a phase machine on obs["step"] plus three small loops.

Every 10 ms the law builds a joint-angle target for each of the 26 servos and converts it to an action.

1. GRASP (steps 0 .. t_down + t_close + t_squeeze)
   The arm pitches down (ARRx past its limit, so the palm rests on the handle) with a small yaw offset `y_grasp`
   that puts the hand near the butt end of the handle (the head then reaches further).  The thumb is pre-shaped
   as a post pointing down *behind* the handle (`th*_pre`); the four fingers then curl around the far side
   (`f2, f1, f0`, commanded past the handle so the servos keep squeezing, `lf4` arches the palm) and pull the
   handle against that post; finally the thumb swings forward under the handle (`th*`).  The post is what keeps
   the handle from sliding out toward the wrist when the head is loaded.
2. LIFT + WIND-UP (t_lift steps)
   Arm pitch: feed-forward target for the observed nail height (head height ~ z0 - z_slope * ARRx), then a
   trim that tracks, with gain `kz`, the pitch that would put the *observed hammer head* (reconstructed from the
   hammer pose in the observation) at the nail height + z_off.  The pitch servo gets a gravity feed-forward
   `tau_g` and an outer proportional term `kp_x` on the measured pitch, so that a heavier hammer or a softer
   servo shifts the hand less.  Meanwhile the arm yaws to `y_wind` and the wrist deviates to `w1_strike`
   (hand turned clockwise, the head reaches out to the nail's radius).
3. STRIKE (t_swing steps + t_hold)
   A smoothstep yaw sweep from `y_wind` to `y_end` (beyond the joint limit, so the arm is still driving at
   impact) swings the head along the nail axis; the nail is driven by the impact.
4. JUDGE / REPEAT
   The nail depth channel decides: noise-free observations use the exact threshold `q_done` (a driven nail
   rests at 0.0816 m, success needs 0.081 m); noisy observations (recognised during the grasp, where the true
   depth is exactly zero) use a mean over several samples against `q_done_noisy`.  If the nail is not driven,
   the arm winds up again (t_back) and strikes again, as long as a strike can still land before the last
   25 steps (`t_last`); once it is driven the arm stays put.

Angles in rad, lengths in m, times in control steps (10 ms).
"""

import numpy as np

PARAMS = {
    # --- grasp ---
    "t_down": 12.0,      # steps to lower the open hand onto the handle
    "t_close": 14.0,     # steps over which the fingers close
    "t_squeeze": 8.0,   # steps to let the grasp settle before lifting
    "x_down": 0.60,      # ARRx target while grasping (beyond the joint limit: the hand presses on the handle)
    "y_grasp": -0.1,     # ARRy during the grasp (shifts the hand along the handle)
    "w1_grasp": 0.0,     # wrist deviation during the grasp (skews the hammer in the hand)
    "w0_grasp": 0.0,     # wrist flexion during the grasp
    "f2": 1.6, "f1": 1.6, "f0": 1.6,          # finger flexion set points of the closed hand
    "lf4": 0.3,          # little-finger metacarpal (palm arch) of the closed hand
    "th4_pre": -0.3, "th3_pre": 1.05, "th1_pre": 0.0, "th0_pre": -0.6,   # thumb pre-shape: a post behind the handle
    "th4": 0.8, "th3": 1.3, "th2": 0.0, "th1": 0.0, "th0": -0.8,   # thumb set points of the closed hand (squeeze)
    "t_thumb": 8.0,      # steps over which the thumb squeezes, starting when the fingers are closed
    # --- lift / wind-up ---
    "t_lift": 28.0,      # steps for lift + wind-up
    "y_wind": 0.20,      # ARRy at wind-up (hand toward -x)
    "w1_strike": -0.524, # wrist deviation during the strike (clockwise: head reaches outward)
    "w0_strike": 0.0,    # wrist flexion during the strike
    "z_off": 0.0,         # head height target relative to the nail head [m]
    "z0": 0.115, "z_slope": 0.63,   # feed-forward: head height ~ z0 - z_slope * ARRx while the hammer is held
    "lift_frac": 0.7,    # fraction of t_lift used for the feed-forward pitch ramp (the trim starts after it)
    "kz": 0.25,          # per-step gain with which the pitch target tracks the head-height estimate
    "tau_g": 16.6,       # gravity torque of arm + hammer about ARRx [N m] (feed-forward)
    "kp_x": 0.4,         # outer proportional gain on the arm pitch error [set point rad per rad]
    # --- strike ---
    "t_swing": 26.0,     # steps of the yaw sweep
    "y_end": -0.45,      # ARRy target at the end of the sweep (beyond what the nail allows: it keeps pushing)
    "t_hold": 14.0,      # steps to keep pushing before judging the nail
    "t_back": 20.0,      # steps to wind up again
    "q_done": 0.0812,    # nail depth [m] counted as driven when the observation is noise-free
    "q_done_noisy": 0.078,   # ... threshold on the sample mean when the observation is noisy
    "n_avg_noisy": 12.0,     # samples averaged for the verdict right after a strike (noisy case)
    "wait_noisy": 8.0,       # extra steps before that verdict, so the averaged samples see the nail at rest
    "n_recheck_noisy": 30.0, # samples averaged for the later re-check while resting (noisy case)
    "t_last": 150.0,     # no strike is started after this step (it could not hold for the last 25 steps)
}

_ARM_RATIO = 2.5         # arm servo: equilibrium joint angle = 2.5 * set point (g = 500, k = 200)
_FINGERS_J2 = (5, 9, 13, 18)
_FINGERS_J1 = (6, 10, 14, 19)
_FINGERS_J0 = (7, 11, 15, 20)
_S = {}


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _head_position(o):
    """Hammer head centre from the observed hammer frame position and Euler angles (head is 0.24 m along -x)."""
    ai, aj, ak = -o[41], -o[40], -o[39]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    xaxis = np.array([cj * ci, sj * ci * sk - si * ck, sj * ci * ck + si * sk])
    return o[36:39] - 0.24 * xaxis


def _hand_pose(p, grip, thumb=1.0):
    """Joint targets of the fingers (grip = 0 open, 1 closed) and thumb (0 pre-shaped post, 1 squeezing)."""
    q = np.zeros(26)
    for i in _FINGERS_J2:
        q[i] = grip * p["f2"]
    for i in _FINGERS_J1:
        q[i] = grip * p["f1"]
    for i in _FINGERS_J0:
        q[i] = grip * p["f0"]
    q[16] = grip * p["lf4"]
    q[21] = p["th4_pre"] + thumb * (p["th4"] - p["th4_pre"])
    q[22] = p["th3_pre"] + thumb * (p["th3"] - p["th3_pre"])
    q[23] = thumb * p["th2"]
    q[24] = p["th1_pre"] + thumb * (p["th1"] - p["th1_pre"])
    q[25] = p["th0_pre"] + thumb * (p["th0"] - p["th0_pre"])
    return q


def _driven(S, p, n):
    """Is the nail driven?  Noise-free observations: exact test on the last sample.  Noisy observations: mean of
    the last n samples against a lower threshold."""
    if not S["noisy"]:
        return S["buf"][-1] > p["q_done"]
    return float(np.mean(S["buf"][-int(n):])) > p["q_done_noisy"]


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    t = int(obs["step"])
    mean = np.asarray(obs["ctrl_mean"], dtype=float)
    half = np.asarray(obs["ctrl_half_range"], dtype=float)
    S = _S
    if t == 0 or not S:
        S.clear()
        S.update(h_sum=0.0, h_n=0, xi=0.0, phase="lift", t0=0.0, y_from=0.0, strikes=0, n2=0.0, buf=[], noisy=False)
    # nail height: running mean of the (possibly noisy) measurement
    S["h_sum"] += o[44]
    S["h_n"] += 1
    h = S["h_sum"] / S["h_n"]
    S["buf"].append(float(o[26]))          # nail depth history (for the "driven" test)
    head = _head_position(o)

    t1 = p["t_down"]
    t2 = t1 + p["t_close"]
    t3 = t2 + p["t_squeeze"]
    if t < t3:
        # ---------------- grasp ----------------
        grip = _smooth((t - t1) / p["t_close"])
        q = _hand_pose(p, grip, _smooth((t - t2) / p["t_thumb"]))
        qx = p["x_down"]
        qy = p["y_grasp"]
        q[2] = p["w1_grasp"]
        q[3] = p["w0_grasp"]
        S["phase"], S["t0"], S["y_from"] = "lift", t3, p["y_grasp"]
        S["x0"] = S["xb"] = float(o[0])
        S["n2"] += float(o[26]) ** 2
        S["noisy"] = bool(np.sqrt(S["n2"] / (t + 1.0)) > 5e-4)   # the true nail depth is exactly 0 here
    else:
        q = _hand_pose(p, 1.0)
        ph, tau = S["phase"], t - S["t0"]
        # head-height loop: feed-forward pitch for the nail height plus a trim on the observed head height
        z_des = h + p["z_off"]
        x_ff = (p["z0"] - z_des) / p["z_slope"]
        # arm pitch that would put the head on z_des, assuming the hammer moves rigidly with the arm from here
        x_est = float(o[0]) + (head[2] - z_des) / p["z_slope"]
        trim = p["kz"] * (x_est - (S["xb"] + S["xi"]))
        w1, w0 = p["w1_strike"], p["w0_strike"]
        if ph == "lift":
            s = _smooth(tau / p["t_lift"])
            qy = S["y_from"] + (p["y_wind"] - S["y_from"]) * s
            w1 = p["w1_grasp"] + (p["w1_strike"] - p["w1_grasp"]) * s
            w0 = p["w0_grasp"] + (p["w0_strike"] - p["w0_grasp"]) * s
            S["xb"] = S["x0"] + (x_ff - S["x0"]) * _smooth(tau / (p["lift_frac"] * p["t_lift"]))
            if tau >= p["lift_frac"] * p["t_lift"]:
                S["xi"] += trim
            if tau >= p["t_lift"]:
                S["phase"], S["t0"] = "swing", t
        elif ph == "swing":
            s = _smooth(tau / p["t_swing"])
            qy = p["y_wind"] + (p["y_end"] - p["y_wind"]) * s
            if tau >= p["t_swing"] + p["t_hold"] + (p["wait_noisy"] if S["noisy"] else 0.0):
                S["strikes"] += 1
                if _driven(S, p, p["n_avg_noisy"]) or t > p["t_last"]:
                    S["phase"], S["t0"] = "done", t
                else:
                    S["phase"], S["t0"], S["y_from"] = "back", t, float(o[1])
        elif ph == "back":
            s = _smooth(tau / p["t_back"])
            qy = S["y_from"] + (p["y_wind"] - S["y_from"]) * s
            S["xi"] += trim
            if tau >= p["t_back"]:
                S["phase"], S["t0"] = "swing", t
        else:  # done: keep the arm where the strike ended; strike again only if the nail is clearly not driven
            qy = p["y_end"]
            S["xi"] += trim
            if tau >= p["n_recheck_noisy"] and t <= p["t_last"] and not _driven(S, p, p["n_recheck_noisy"]):
                S["phase"], S["t0"], S["y_from"] = "back", t, float(o[1])
        S["xi"] = float(np.clip(S["xi"], -0.25, 0.25))
        qx = S["xb"] + S["xi"]
        q[2], q[3] = w1, w0

    ctrl = q.copy()
    # arm pitch servo: 500 ctrl - 200 q + tau_gravity = 0 at rest (gravity pushes ARRx positive, i.e. down).
    # Gravity feed-forward plus an outer proportional loop on the measured pitch (stiffens the servo, so a wrong
    # gravity / servo-gain guess moves the hand less).
    if t >= t3:
        ctrl[0] = (200.0 * qx - p["tau_g"] * np.cos(o[0])) / 500.0 + p["kp_x"] * (qx - o[0])
    else:
        ctrl[0] = 200.0 * qx / 500.0
    ctrl[1] = qy / _ARM_RATIO
    return np.clip((ctrl - mean) / half, -1.0, 1.0)
