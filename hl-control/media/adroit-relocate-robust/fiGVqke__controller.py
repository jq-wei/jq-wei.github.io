"""Adroit relocate: joint-level pick, carry and hold of the ball (state machine + servo loops, nothing learned).

Control law
-----------
Only four arm joints are actively steered: the slides ARTx (moves the hand along world -x), ARTz (world +y),
ARTy (world +z, upward only) and the pitch ARRx (positive lowers the hand; the palm point sits 0.5 m ahead of and
0.05 m below the pitch axis).  ARRy, ARRz and the wrist deviation stay at zero; the wrist flexion is held at a fixed
small extension.  Each step the task-space errors of the observation are converted into joint targets ("where must
this joint be for the error to vanish"; exact for the slides, first order for the pitch, including the forward
shift of the palm point when the arm pitches).  The targets of a ball at rest are constants, so they are low-pass
filtered, which removes measurement noise without adding lag.  Every joint is then servoed to its target r with
feed-forward of the servo spring, the gravity load and the target velocity, plus PD feedback:

    ctrl = (k r + c r' + G) / g  +  Kp (r - q_hat)  +  Kd (r' - v_hat)

q_hat, v_hat come from a small per-joint observer (plant model + output correction): the observation carries no
velocities, and differencing noisy positions would be useless.  Fingers, thumb and wrist are position servos that
simply receive set points (open pre-shape or the closed grasp shape).

Phases
------
0 approach  Hover the palm point above the ball, and pitch down onto it once it is horizontally aligned.
3 rake      Only for a ball beyond the forward travel of the arm: stop at the travel limit, curl the fingers
            slowly over the ball so that it rolls in under the palm, reopen and return to phase 0.
1 close     Fingers, then thumb, wrap the ball against the palm (power grasp); the arm keeps tracking the ball.
2 carry     Joint targets now come from (ball - target): lift with ARTy and by pitching back, slide in x and y,
            then hold.  If the ball leaves the hand at any time, go back to phase 0 and pick it up again.
After a rake or a drop the ball may be rolling; the horizontal targets are then tracked with an alpha-beta filter
(position and velocity), whose velocity feeds the servo feed-forward so the palm follows the ball without lag.

The grasp shape and the grasp offset were found by random search in simulation, scored on low friction, heavy
ball and weak servos (hold reached and little slip of the ball in the hand); they are not hand-derived.
"""

import numpy as np

PARAMS = {
    # --- arm servo feedback (set-point units per unit of joint error / of joint velocity) ---
    "kp_slide": 1.2,      # P gain of the slides: stiffens the 200 N/m servo spring to 800 N/m (13 rad/s)
    "kd_slide": 0.18,     # D gain of the slides: brings the 30 %-damped slide to about 90 % damping
    "kp_rot": 0.5,        # P gain of the arm pitch
    "kd_rot": 0.02,       # D gain of the arm pitch (the pitch joint is nearly critically damped on its own)
    "tgt_alpha": 0.35,    # low-pass factor of the joint targets (noise rejection)
    "trk_a": 0.5,         # alpha-beta tracker of a rolling ball: position gain
    "trk_b": 0.1,         # ... and velocity gain
    # --- approach: desired palm-point minus ball-centre offset at the grasp [m] ---
    "off_x": 0.017,
    "off_y": -0.012,
    "off_z": -0.002,
    "hover": 0.045,       # extra palm height while the palm is not yet above the ball (fingers clear the ball)
    "xy_gate": 0.015,     # horizontal error below which the hand is fully down
    "gate_w": 0.03,       # width of the blend between hover height and grasp height [m]
    "grasp_tol": 0.015,   # position error below which the fingers start closing [m]
    # --- grasp: closed-hand set points [rad] and timing [steps] ---
    "mcp": 1.06, "pip": 1.377, "dip": 0.798,          # knuckle, middle and tip flexion of the four fingers
    "abd_ff": -0.129, "abd_lf": 0.188, "lf4": 0.019,  # side-to-side of fore / little finger, little-finger arch
    "th4": 0.372, "th3": 1.085, "th2": -0.183, "th1": 0.314, "th0": -1.098,   # closed thumb
    "th4_pre": -0.398, "th3_pre": 0.386,              # thumb pre-shape while the hand is open
    "w0": -0.221,         # wrist flexion set point, held throughout (slight extension)
    "t_f": 4,             # delay of the finger closing after the grasp starts
    "t_th": 10,           # delay of the thumb closing (the thumb closes after the fingers)
    "close_steps": 19,    # duration of the closing phase before the lift
    "regrasp_tol": 0.045, # palm-to-ball error above which a closed hand reopens instead of lifting [m]
    # --- far balls: beyond the forward travel of the arm ---
    "far_tol": 0.012,     # missing forward travel above which the ball is raked in first [m]
    "rake_h": 0.04,       # palm height above the grasp height while raking [m]
    "rake_ramp": 14,      # steps over which the raking fingers curl (slow, so the ball is not kicked away)
    "rake_steps": 20,     # duration of the rake
    # --- carry ---
    "th_carry": 0.109,    # preferred arm pitch while carrying [rad]; pitches up further for high targets
    "lift_first": 0.03,   # ball height above the table at which the horizontal carry reaches full speed [m]
    "drop_tol": 0.08,     # palm-to-ball error that counts as a dropped ball [m]
}

# arm plant (joints ARTx, ARTy, ARTz, ARRx in observation / action order)
_G, _K, _C = 500.0, 200.0, 20.0                    # servo gain, servo stiffness, joint damping
_M = np.array([4.912, 4.912, 4.912, 0.555])        # moved mass [kg] of the slides, inertia [kg m^2] of the pitch
_BIAS = np.array([0.0, 48.2, 0.0, -15.5])          # gravity load each servo has to carry [N, N m]
_QLO = np.array([-0.25, 0.0, -0.3, -0.75])         # joint ranges
_QHI = np.array([0.25, 0.2, 0.5, 0.75])
_YMAX = 0.488                                      # forward slide position the servo can hold at its limit
_ZMAX = 0.17                                       # vertical slide travel that is safe to ask for (weak servos)
_L, _H = 0.5, 0.05                                 # palm point: ahead of / below the pitch axis [m]
_DT = 0.01
_FINGERS = (8, 12, 16, 21)                         # index of the side-to-side joint of each finger


def _reach(th):
    """Forward distance of the palm point from the pitch axis at arm pitch th."""
    return _L * np.cos(th) - _H * np.sin(th)


class _State:
    pass


_S = _State()


def _reset(q):
    _S.phase = 0                  # 0 approach, 3 rake, 1 close, 2 carry
    _S.t_phase = 0                # steps spent in phases 1, 2, 3
    _S.t_close = 0                # steps since the hand started closing
    _S.ok_count = 0               # consecutive steps with the palm in place
    _S.touched = False            # the ball has been raked or grasped at least once
    _S.moving = False             # the ball may be rolling (after a rake or a drop)
    _S.qh = q[:4].copy()          # observer: joint positions
    _S.vh = np.zeros(4)           # observer: joint velocities
    _S.u_prev = np.zeros(4)       # last arm set points
    _S.tgt = q[:4].copy()         # filtered joint targets
    _S.trk_on = False             # alpha-beta tracker of the horizontal targets
    _S.tx = np.zeros(2)
    _S.tv = np.zeros(2)
    _S.tz_sum, _S.tz_n, _S.target_z = 0.0, 0, 0.25   # target height, averaged while the ball rests on the table


def _observe(q_meas):
    """Per-joint observer: integrate the servo model with the last set point, then correct with the measurement."""
    qh, vh = _S.qh, _S.vh
    for _ in range(5):                                   # the five 2 ms physics steps of one control step
        acc = (_G * _S.u_prev - _K * qh - _C * vh - _BIAS) / _M
        vh = vh + 0.002 * acc
        qh = qh + 0.002 * vh
    err = q_meas - qh
    qh = qh + 0.5 * err
    vh = vh + 12.0 * err
    lo, hi = qh < _QLO, qh > _QHI                        # joint stops
    qh = np.clip(qh, _QLO, _QHI)
    vh = np.where((lo & (vh < 0)) | (hi & (vh > 0)), 0.0, vh)
    _S.qh, _S.vh = qh, vh


def _reopen():
    """Back to the approach with an open hand (after a rake, a failed grasp or a drop)."""
    _S.phase, _S.t_phase, _S.t_close, _S.ok_count, _S.moving = 0, 0, 0, 0, True


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    mean = np.asarray(obs["ctrl_mean"], dtype=float)
    half = np.asarray(obs["ctrl_half_range"], dtype=float)
    if int(obs["step"]) == 0 or not hasattr(_S, "phase"):
        _reset(o[:30])
    else:
        _observe(o[:4])
    qh, vh = _S.qh, _S.vh
    theta = qh[3]
    lever = _L * np.cos(theta)                           # palm-point height change per radian of pitch

    pb, pt, bt = o[30:33], o[33:36], o[36:39]            # palm - ball, palm - target, ball - target
    bt = 0.5 * (bt + (pt - pb))                          # two readings of the same quantity: average them
    off = np.array([P["off_x"], P["off_y"], P["off_z"]])
    ex, ey = pb[0] - off[0], pb[1] - off[1]              # horizontal error of the palm w.r.t. the grasp position

    # ----- phase logic: joint targets `des` for ARTx, ARTy, ARTz, ARRx -----
    if _S.phase == 0:
        if not _S.touched:                               # ball at rest at z = 0.035: this fixes the target height
            _S.tz_sum += 0.035 - bt[2]
            _S.tz_n += 1
            _S.target_z = _S.tz_sum / _S.tz_n
        th_g = theta + (pb[2] - off[2]) / lever          # pitch that puts the palm point at grasp height
        short = (qh[2] - ey - (_reach(th_g) - _reach(theta))) - _YMAX   # forward travel missing to get there
        far = short > P["far_tol"]
        if far:                                          # out of reach: stop at the limit, above the ball
            ey_gate, oz_base = qh[2] - _YMAX, off[2] + P["rake_h"]
        else:
            ey_gate, oz_base = ey, off[2]
        exy = np.hypot(ex, ey_gate)
        oz = oz_base + P["hover"] * np.clip((exy - P["xy_gate"]) / P["gate_w"], 0.0, 1.0)
        th_d = theta + (pb[2] - oz) / lever
        des = np.array([qh[0] + ex, 0.0, qh[2] - ey - (_reach(th_d) - _reach(theta)), th_d])
        in_place = exy < P["grasp_tol"] and abs(pb[2] - oz_base) < P["grasp_tol"]
        _S.ok_count = _S.ok_count + 1 if in_place else 0
        if _S.ok_count >= 2:
            _S.phase, _S.t_phase, _S.touched = (3 if far else 1), 0, True
    elif _S.phase == 3:
        th_d = theta + (pb[2] - (off[2] + P["rake_h"])) / lever
        des = np.array([qh[0] + ex, 0.0, _YMAX, th_d])
        _S.t_phase += 1
        _S.t_close += 1
        if _S.t_phase >= P["rake_steps"]:
            _reopen()
    elif _S.phase == 1:
        th_d = theta + (pb[2] - off[2]) / lever
        des = np.array([qh[0] + ex, 0.0, qh[2] - ey - (_reach(th_d) - _reach(theta)), th_d])
        _S.t_phase += 1
        _S.t_close += 1
        if _S.t_phase >= P["close_steps"]:
            if np.linalg.norm(pb - off) > P["regrasp_tol"]:
                _reopen()                                # the ball is not in the hand: try again
            else:
                _S.phase, _S.t_phase = 2, 0
    else:
        _S.t_phase += 1
        _S.t_close += 1
        ball_h = _S.target_z + bt[2] - 0.035             # ball height above the table
        gain_xy = np.clip(ball_h / P["lift_first"], 0.15, 1.0)   # lift first, then carry sideways
        dz = -bt[2]                                      # rise still needed
        th_des = P["th_carry"]                           # the rise is shared by the pitch and the vertical slide
        z_des = qh[1] + dz - _L * (np.sin(theta) - np.sin(th_des))
        if z_des > _ZMAX:                                # slide at its top: pitch up for the rest
            th_des = np.arcsin(np.clip(np.sin(theta) - (dz - (_ZMAX - qh[1])) / _L, -0.65, 0.65))
            z_des = _ZMAX
        elif z_des < 0.0:                                # slide at its bottom: pitch down for the rest
            th_des = np.arcsin(np.clip(np.sin(theta) - (dz + qh[1]) / _L, -0.65, 0.65))
            z_des = 0.0
        des = np.array([qh[0] + gain_xy * bt[0], z_des,
                        qh[2] - gain_xy * bt[1] - (_reach(th_des) - _reach(theta)), th_des])
        if np.linalg.norm(pb - off) > P["drop_tol"] and _S.t_phase > 5:
            _reopen()                                    # dropped: pick it up again

    # ----- target filtering -----
    des = np.clip(des, _QLO, _QHI)
    r = _S.tgt + P["tgt_alpha"] * (des - _S.tgt)
    rdot = np.zeros(4)
    if _S.moving and _S.phase != 2:                      # rolling ball: alpha-beta tracking of the slide targets
        d2 = des[[0, 2]]
        if not _S.trk_on:
            _S.tx, _S.tv, _S.trk_on = d2.copy(), np.zeros(2), True
        else:
            _S.tx = _S.tx + _DT * _S.tv
            res = d2 - _S.tx
            _S.tx = _S.tx + P["trk_a"] * res
            _S.tv = np.clip(_S.tv + P["trk_b"] / _DT * res, -0.6, 0.6)
        r[[0, 2]] = _S.tx
        rdot[[0, 2]] = _S.tv
    else:
        _S.trk_on = False
    _S.tgt = r

    # ----- arm servo loops -----
    kp = np.array([P["kp_slide"]] * 3 + [P["kp_rot"]])
    kd = np.array([P["kd_slide"]] * 3 + [P["kd_rot"]])
    u = (_K * r + _C * rdot + _BIAS) / _G + kp * (r - qh) + kd * (rdot - vh)
    u = np.clip(u, mean[:4] - half[:4], mean[:4] + half[:4])
    _S.u_prev = u.copy()

    # ----- set points (physical units); everything not mentioned stays at 0 -----
    ctrl = np.zeros(30)
    ctrl[:4] = u
    ctrl[7] = P["w0"]
    ctrl[25], ctrl[26] = P["th4_pre"], P["th3_pre"]      # thumb pre-shape, fingers straight
    if _S.phase != 0 and _S.t_close >= P["t_f"]:
        # the rake curls the fingers slowly, the grasp at once
        ramp = min(1.0, (_S.t_close - P["t_f"] + 1) / P["rake_ramp"]) if _S.phase == 3 else 1.0
        for j in _FINGERS:
            ctrl[j + 1], ctrl[j + 2], ctrl[j + 3] = P["mcp"] * ramp, P["pip"] * ramp, P["dip"] * ramp
        ctrl[8], ctrl[21], ctrl[20] = P["abd_ff"], P["abd_lf"], P["lf4"]
    if _S.phase in (1, 2) and _S.t_close >= P["t_th"]:
        ctrl[25:30] = P["th4"], P["th3"], P["th2"], P["th1"], P["th0"]

    return np.clip((ctrl - mean) / half, -1.0, 1.0)
