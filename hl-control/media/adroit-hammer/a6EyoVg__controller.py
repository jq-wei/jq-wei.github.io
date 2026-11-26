"""Adroit hammer (AdroitHandHammer-v1, joint level): scripted "pinch, lift, wind up, swing" controller.

No learning: a fixed sequence of joint set points with two measurement-driven pieces (the lift height and the
squeeze). The sequence, in episode time:

1. reach   (t < t_reach)   The arm pitches down (`arm_down`) so the palm rests on the hammer handle, fingers half
                           open (`f*_open`), thumb rotated into opposition (`th4`). The arm is yawed by `yaw_g` and the
                           wrist deviated by `dev_g` *before* the grasp: the hammer is therefore held rotated clockwise
                           in the hand, which is what later lets the head reach the nail (see 4).
2. pinch   (t < t_hook)    Fingers curl in front of / under the handle (`f*_hook`), the thumb closes on its head-side
                           end (`th*`): the handle is clamped against the palm and comes off the table.
3. lift + wind up          For `t_windup` seconds: the arm pitch follows `p_des`, the pitch at which the *measured*
                           centre of the striking face (hammer position + Euler angles from the observation) would sit
                           at the nail height, rate-limited by `lift_rate`. The wrist flexes to keep the hand pitched
                           down by `alpha`, deviates fully (`dev_hit`), and the arm yaws back to `yaw_back`. The knuckle
                           set points are frozen at the measured angles + `f2_margin` (a constant squeeze torque; a
                           larger set point rolls the handle out of the pinch).
4. swing                   The arm yaws to its stop (`yaw_hit`) at `swing_rate`. With the wrist fully deviated this
                           carries the face of the head almost exactly along the nail axis (0.93, 0.36, 0), drives the
                           nail home and leaves the face leaning on the nail head. That residual push matters: the
                           nail's soft end stop pushes a free nail back to the very edge of the 0.01 m tolerance.

Arm servos (g = 500, k = 200) sit at 2.5 x set point, so a desired angle q is commanded as q / 2.5 plus a
proportional term `kp_arm` on the measured error and a constant `g_comp` against the arm's weight (pitch only).
Wrist and finger servos are position servos: the desired angle is the set point.

Gains / parameters (PARAMS): times `t_reach`, `t_hook`, `t_windup` [s]; grasp pose `arm_down`, `wr_flex`,
`yaw_g`, `dev_g`; finger targets `f2/f1/f0_open`, `f2/f1/f0_hook` and squeeze `f2_margin` [rad]; thumb targets
`th4..th0`; hand pitch `alpha`; height offset `z_off` [m] and lever `l_arm` [m] of the lift law; rates
`lift_rate`, `yaw_rate`, `dev_rate`, `swing_rate` [rad/s]; wind-up / strike angles `yaw_back`, `dev_hit`, `yaw_hit`.
"""

import numpy as np

JOINTS = ("ARRx ARRy WRJ1 WRJ0 FFJ3 FFJ2 FFJ1 FFJ0 MFJ3 MFJ2 MFJ1 MFJ0 RFJ3 RFJ2 RFJ1 RFJ0 "
          "LFJ4 LFJ3 LFJ2 LFJ1 LFJ0 THJ4 THJ3 THJ2 THJ1 THJ0").split()
J = {name: i for i, name in enumerate(JOINTS)}     # joint order of the observation and of the action
FINGERS = ("FF", "MF", "RF", "LF")

PARAMS = dict(
    # phase lengths [s]
    t_reach=0.22, t_hook=0.45, t_windup=0.16,
    # grasp pose
    arm_down=0.25, wr_flex=0.25, yaw_g=-0.02, dev_g=0.175,
    f2_open=0.0, f1_open=0.9, f0_open=0.3,
    f2_hook=1.0, f1_hook=1.6, f0_hook=1.6, f2_margin=0.45,
    th4=1.0, th3=1.2, th2=0.0, th1=0.5, th0=-0.8,
    # lift
    alpha=0.25, z_off=-0.004, l_arm=0.45, lift_rate=2.75,
    # wind up and swing
    yaw_back=0.22, dev_hit=-0.524, yaw_hit=-0.3, yaw_rate=4.0, dev_rate=6.0, swing_rate=4.0,
    # arm servo shaping
    kp_arm=0.35, g_comp=0.03,
)
PARAM_BOUNDS = dict(
    t_windup=(0.14, 0.18), yaw_g=(-0.04, 0.0), alpha=(0.1, 0.5), z_off=(-0.012, 0.006),
    lift_rate=(2.4, 3.1), yaw_back=(0.15, 0.28), swing_rate=(3.6, 4.5),
)

FACE_IN_HAMMER = np.array([-0.24, 0.0, -0.04])     # centre of the striking face of the head, hammer body frame [m]
_state = {}


def _approach(current, target, rate, dt):
    """Move `current` toward `target` by at most rate * dt."""
    return float(current + np.clip(target - current, -rate * dt, rate * dt))


def _euler2mat(e):
    """Rotation matrix of the observation's Euler angles (the convention of gymnasium-robotics' quat2euler)."""
    ai, aj, ak = -e[2], -e[1], -e[0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs, sc, ss = ci * ck, ci * sk, si * ck, si * sk
    return np.array([[cj * ci, cj * si, -sj],
                     [sj * cs - sc, sj * ss + cc, cj * sk],
                     [sj * cc + ss, sj * sc - cs, cj * ck]])


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    t, dt = float(obs["time"]), float(obs["dt"])
    if obs["step"] == 0 or not _state:           # new episode: the nail height is read once, at rest
        _state.clear()
        _state.update(pitch=0.0, yaw=0.0, dev=0.0, h=float(o[44]), f2_lock=None)
    S = _state
    q = np.zeros(26)                             # desired joint angles [rad]

    if t < P["t_reach"]:                         # 1. reach: open hand down onto the handle
        S["pitch"], S["yaw"], S["dev"] = P["arm_down"], P["yaw_g"], P["dev_g"]
        q[J["WRJ0"]] = P["wr_flex"]
        fingers = (P["f2_open"], P["f1_open"], P["f0_open"])
        q[J["THJ4"]] = P["th4"]
    else:
        for name in ("THJ4", "THJ3", "THJ2", "THJ1", "THJ0"):
            q[J[name]] = P["th" + name[-1]]
        if t < P["t_hook"]:                      # 2. pinch: fingers in front of the handle, thumb behind it
            q[J["WRJ0"]] = P["wr_flex"]
            fingers = (P["f2_hook"], P["f1_hook"], P["f0_hook"])
        else:                                    # 3. lift and wind up, 4. swing
            if S["f2_lock"] is None:             # freeze the knuckles where the pinch formed (+ a squeeze margin)
                S["f2_lock"] = float(np.mean([o[J[f + "J2"]] for f in FINGERS])) + P["f2_margin"]
            fingers = (S["f2_lock"], P["f1_hook"], P["f0_hook"])
            q[J["WRJ0"]] = float(np.clip(P["alpha"] - S["pitch"], -0.75, 0.6))
            # arm pitch at which the measured striking face would sit at the nail height (hammer rigid in the hand)
            face_z = o[38] + _euler2mat(o[39:42])[2] @ FACE_IN_HAMMER
            z_err = (S["h"] + P["z_off"]) - face_z
            p_des = float(np.arcsin(np.clip(np.sin(o[0]) - z_err / P["l_arm"], -0.39, 0.25)))
            S["pitch"] = _approach(S["pitch"], p_des, P["lift_rate"], dt)
            S["dev"] = _approach(S["dev"], P["dev_hit"], P["dev_rate"], dt)
            if t < P["t_hook"] + P["t_windup"]:
                S["yaw"] = _approach(S["yaw"], P["yaw_back"], P["yaw_rate"], dt)
            else:
                S["yaw"] = _approach(S["yaw"], P["yaw_hit"], P["swing_rate"], dt)

    for f in FINGERS:
        q[J[f + "J2"]], q[J[f + "J1"]], q[J[f + "J0"]] = fingers
    q[J["ARRx"]], q[J["ARRy"]], q[J["WRJ1"]] = S["pitch"], S["yaw"], S["dev"]

    ctrl = q.copy()                              # wrist and fingers: position servos
    ctrl[0] = q[0] / 2.5 + P["kp_arm"] * (q[0] - o[0]) - P["g_comp"]
    ctrl[1] = q[1] / 2.5 + P["kp_arm"] * (q[1] - o[1])
    action = (ctrl - np.asarray(obs["ctrl_mean"], dtype=float)) / np.asarray(obs["ctrl_half_range"], dtype=float)
    return np.clip(np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)
