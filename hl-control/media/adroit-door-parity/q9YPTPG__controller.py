"""Adroit door: reach the lever, turn the latch, fling the door open, get out of its way.

Control law
-----------
Arm (slide, pitch, yaw, roll).  Every arm servo applies 500*ctrl - 200*q, so commanding
ctrl = (200*q + u)/500 cancels the servo spring and leaves a pure force/torque u.  u is a PD law:
the *measured* palm-point error (obs 29-31 against a target built from the measured handle point,
latch angle and door angle) is mapped to slide/pitch/yaw errors with the inverse of the analytic arm
Jacobian, u = Kp*dq - Kd*qdot (+ gravity feed-forward on the pitch).  Joint rates are finite differences
of the observation; the unobserved slide is recovered from the palm y coordinate through the forward
kinematics.  The roll follows the latch angle.  Wrist and fingers are plain set points.

Phases (a state machine driven by the measured latch and door angles):
  0 REACH   palm point -> hover point above the lever (REACH_*), fingers pre-curled into a hook.
  1 TURN    the palm target is the hover pose rotated about the latch axis by (latch + TURN_LEAD) and
            the forearm rolls with the latch (ROLL_LEAD): the palm pushes the lever down along its arc
            while the fingers close behind it and the wrist flexes (WR_FLEX) to keep the hook shut.
  2 PULL    latch past LATCH_OPEN.  While the door is below TH_A the target leads the palm straight
            out of the door plane by PULL_LEAD (ramped over PULL_RAMP steps) and holds the palm against
            the lever (XOFF) with the roll keeping the latch at LATCH_MAX; afterwards the target leads
            the palm along the handle's swing circle by PULL_LEAD2, PULL_AHEAD rad ahead of the door.
            The leads act as a virtual spring that drags the door open and follows it.
            Past TH_OPEN the fingers are told to open (they need ~60 ms) while the arm keeps following.
  3 RELEASE door past TH_REL: the door now coasts to its stop on its own momentum; the arm pulls
            straight back out of the door's swing circle to the park pose (*_RETREAT) and stays there.

Gains
-----
KP_S/KD_S     slide PD [N/m, N s/m]            KP_R/KD_R  pitch and yaw PD [N m/rad, N m s/rad]
KP_ROLL       roll P gain [N m/rad] (the roll joint is damping dominated), clipped at ROLL_TMAX
MAX_STEP      clip [m] on the palm error fed to the PD (bounds the commanded force)
REACH_TOL     distance [m] to the hover point at which TURN starts
PRESS_Z       palm-point offset [m] across the lever while pressing (how hard the palm bears on it)
FING_PRE/FING_PIP  knuckle / middle-joint pre-curl of the fingers during the reach [rad]
THUMB_LATCH   latch angle [rad] past which the thumb closes
The values below were tuned by random search on seeds 0-47 and checked on > 1000 other seeds.
"""

import numpy as np

PARAMS = {
    "KP_S": 3416.7359, "KD_S": 148.0453,
    "KP_R": 1058.18261, "KD_R": 12.0,
    "KP_ROLL": 704.90546, "ROLL_TMAX": 400.0,
    "MAX_STEP": 0.2,
    # reach
    "REACH_X": 0.01816, "REACH_Y": 0.01, "REACH_Z": 0.0398, "REACH_TOL": 0.03494,
    "FING_PRE": 0.2782, "FING_PIP": 1.57631,
    # turn
    "PRESS_Z": -0.03291, "TURN_LEAD": 0.5, "ROLL_LEAD": 0.2, "LATCH_MAX": 1.64268, "LATCH_OPEN": 1.30575,
    "THUMB_LATCH": 1.0, "WR_FLEX": 0.6,
    # pull
    "PULL_LEAD": 0.21546, "PULL_RAMP": 1.69705, "XOFF": -0.00047, "TH_A": 0.21934,
    "PULL_LEAD2": 0.15421, "PULL_AHEAD": 0.01128, "TH_OPEN": 0.71, "TH_REL": 0.82966,
    # release / park pose
    "S_RETREAT": -0.26999, "PITCH_RETREAT": 0.32536, "YAW_RETREAT": -0.75, "ROLL_RETREAT": 1.84263,
}

# ---- model constants ----
_BASE = np.array([0.0, -0.7, 0.2])      # forearm mount [m]
_K_ARM, _G_ARM = 200.0, 500.0           # arm servo stiffness and gain
_GRAV_PITCH = 15.0                      # N m, weight of forearm + hand about the pitch axis (arm horizontal)
_LEVER_R = 0.15                         # m, handle point to latch axis
_TANGENT = (0.31, 0.95)                 # direction the handle point starts to move in, door frame (x toward hinge, y out of the door)
_FINGER_BLOCKS = (0, 4, 8, 13)          # index of J3 of the fore, middle, ring, little finger in the 22 hand set points
_THUMB_WRAP = (0.6, 1.0, 0.0, 0.3, -0.8)


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_R0 = _rx(-1.57) @ _ry(0.0) @ _rz(3.14)   # forearm frame in the world (the model's euler angles)


def palm_fk(s, a, b, c, w1, w0):
    """World position of the palm point S_grasp: slide s, arm pitch/yaw/roll (a, b, c), wrist (w1, w0)."""
    hand = np.array([0.0, 0.0, 0.396]) + _ry(w1) @ (np.array([0.0, 0.0, 0.034]) + _rx(w0) @ np.array([0.007, -0.05, 0.07]))
    return _BASE + _R0 @ (np.array([0.0, 0.0, s]) + _rx(a) @ _ry(b) @ _rz(c) @ hand)


class _State:
    def __init__(self):
        self.phase = 0          # 0 reach, 1 turn, 2 pull, 3 release
        self.t_phase = 0        # control steps spent in the current phase
        self.prev_arm = None    # last (slide, pitch, yaw, roll), for the rate estimate


_S = _State()


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    if obs["step"] == 0:
        _S.__init__()
    a, b, c, w1, w0 = o[0], o[1], o[2], o[3], o[4]      # arm pitch, yaw, roll; wrist deviation, flexion
    latch, door = o[27], o[28]
    palm, handle = o[29:32], o[32:35]

    # arm state: slide from the palm y coordinate, rates by finite differences
    fk0 = palm_fk(0.0, a, b, c, w1, w0)
    s = float(np.clip(palm[1] - fk0[1], -0.35, 0.55))
    arm = np.array([s, a, b, c])
    if _S.prev_arm is None:
        _S.prev_arm = arm.copy()
    varm = (arm - _S.prev_arm) / obs["dt"]
    _S.prev_arm = arm.copy()

    # door frame in the world and the latch axis point in the lever's plane
    cd, sd = np.cos(door), np.sin(door)
    ex = np.array([cd, sd, 0.0])          # along the door toward the hinge
    ey = np.array([-sd, cd, 0.0])         # into the door
    axis = handle - _LEVER_R * (np.cos(latch) * ex) + np.array([0.0, 0.0, _LEVER_R * np.sin(latch)])

    # ---- phase logic: palm target and roll target ----
    roll_t = 0.0
    if _S.phase == 0:
        target = handle + np.array([P["REACH_X"], P["REACH_Y"], P["REACH_Z"]])
        if np.linalg.norm(target - palm) < P["REACH_TOL"]:
            _S.phase, _S.t_phase = 1, 0
    if _S.phase == 1:
        phi = min(latch + P["TURN_LEAD"], P["LATCH_MAX"])
        dx, dz = _LEVER_R + P["REACH_X"], P["PRESS_Z"]
        cp, sp = np.cos(phi), np.sin(phi)
        target = axis + (dx * cp + dz * sp) * ex + P["REACH_Y"] * ey + np.array([0.0, 0.0, -dx * sp + dz * cp])
        roll_t = min(latch + P["ROLL_LEAD"], P["LATCH_MAX"])
        if latch > P["LATCH_OPEN"]:
            _S.phase, _S.t_phase = 2, 0
    if _S.phase == 2:
        if door < P["TH_A"]:
            # the bolt has only just cleared: pull straight out of the door plane, palm held against the lever
            ramp = min(1.0, (_S.t_phase + 1) / max(P["PULL_RAMP"], 1.0))
            ex_err = float((handle - palm) @ ex) + P["XOFF"]
            target = palm - P["PULL_LEAD"] * ramp * ey + ex_err * ex
            roll_t = P["LATCH_MAX"]
        else:
            # lead the palm along the handle's swing circle
            ang = door + P["PULL_AHEAD"]
            tdir = np.array([_TANGENT[0] * np.cos(ang) + _TANGENT[1] * np.sin(ang),
                             _TANGENT[0] * np.sin(ang) - _TANGENT[1] * np.cos(ang), 0.0])
            target = palm + P["PULL_LEAD2"] * tdir
            roll_t = c
        target[2] = palm[2]
        if door > P["TH_REL"]:
            _S.phase, _S.t_phase = 3, 0
    if _S.phase == 3:
        target = palm.copy()
        roll_t = c
    _S.t_phase += 1

    # ---- wrist and finger set points [rad] ----
    hand = np.zeros(22)                   # FFJ3..0, MFJ3..0, RFJ3..0, LFJ4..0, THJ4..0 (zero = open, flat)
    wrist = np.zeros(2)                   # deviation, flexion
    if _S.phase == 0:
        for k in _FINGER_BLOCKS:          # "L" hook: knuckle nearly straight, middle joint bent
            hand[k + 1], hand[k + 2], hand[k + 3] = P["FING_PRE"], P["FING_PIP"], 0.6
    elif _S.phase in (1, 2) and not (_S.phase == 2 and door > P["TH_OPEN"]):
        for k in _FINGER_BLOCKS:          # close around the lever
            hand[k + 1], hand[k + 2], hand[k + 3] = 1.6, 1.6, 1.0
        if _S.phase == 2 or latch > P["THUMB_LATCH"]:
            hand[17:22] = _THUMB_WRAP
        wrist[1] = P["WR_FLEX"]
    elif _S.phase == 2:
        wrist[1] = P["WR_FLEX"]           # fingers opening ahead of the release

    # ---- arm servo forces ----
    if _S.phase < 3:
        err = target - palm
        n = np.linalg.norm(err)
        if n > P["MAX_STEP"]:
            err *= P["MAX_STEP"] / n
        eps = 1e-4
        J = np.zeros((3, 3))              # d(palm)/d(slide, pitch, yaw)
        J[:, 0] = _R0[:, 2]
        J[:, 1] = (palm_fk(0.0, a + eps, b, c, w1, w0) - fk0) / eps
        J[:, 2] = (palm_fk(0.0, a, b + eps, c, w1, w0) - fk0) / eps
        try:
            dq = np.linalg.solve(J, err)
        except np.linalg.LinAlgError:
            dq = np.zeros(3)
        droll = float(np.clip(P["KP_ROLL"] * (roll_t - c), -P["ROLL_TMAX"], P["ROLL_TMAX"]))
    else:
        # park pose in joint space: straight back out of the swing circle
        dq = np.array([P["S_RETREAT"] - s, P["PITCH_RETREAT"] - a, P["YAW_RETREAT"] - b])
        droll = P["KP_ROLL"] * (P["ROLL_RETREAT"] - c)
    u = np.zeros(4)
    u[0] = _K_ARM * s + P["KP_S"] * dq[0] - P["KD_S"] * varm[0]
    u[1] = _K_ARM * a + P["KP_R"] * dq[1] - P["KD_R"] * varm[1] - _GRAV_PITCH * np.cos(a)
    u[2] = _K_ARM * b + P["KP_R"] * dq[2] - P["KD_R"] * varm[2]
    u[3] = _K_ARM * c + droll

    ctrl = np.concatenate([u / _G_ARM, wrist, hand])
    act = (ctrl - np.asarray(obs["ctrl_mean"], dtype=float)) / np.asarray(obs["ctrl_half_range"], dtype=float)
    return np.clip(act, -1.0, 1.0)
