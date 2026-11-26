"""Get-up controller for the lying MuJoCo Humanoid.

STATUS: reaches the `sit` rung (root held at about 0.51 m) from every start pose; it does NOT reach `crouch`
or `stand`, so it does not pass the task's gate.  Nothing here balances the robot on its two point feet.

Control law
-----------
Joint-space PD on pose targets,

    torque = KP * (q_target - q) - KD * qdot,      command = clip(torque / gear, -0.4, 0.4),

with the gear taken from the observation, so weaker motors get the same torque request.  The targets are blended
smoothly (smoothstep) from the joint angles at the start of a phase to the phase's key pose.  Phases:

  flatten : legs spread, arms out sideways, everything else straight.  A body lying on a side or propped on an
            arm tips onto its back or its front.
  roll    : (face down) one knee is drawn up under the body and the waist is twisted, which levers the trunk
            over that knee onto its side and beyond.  If nothing moves the other knee is tried.
  situp   : hips flexed to 105 deg with straight, slightly spread legs and the waist bent forward.  The legs
            outweigh the forward-folded trunk about the seat, so the body rocks onto the seat with the legs down.
  sit     : opens the fold to an upright long-sit (trunk within about 20 deg of vertical, root at 0.51 m), hands
            down beside the hips.  If the trunk drops again the sequence restarts.

Gains
-----
KP [N m/rad] asks for a joint's full torque at about 0.25 rad of error (hip pitch 120 N m, knee 80 N m, other
leg and waist axes 40 N m), less for the leg-twist axis and the arms.
KD [N m s/rad] is kept below roughly I/dt for each joint (I: the inertia that joint moves on its own, dt = 15 ms).
The leg-twist (hip_z) and arm joints carry very little inertia; with a derivative gain proportional to KP the
sampled loop chatters, the contacts flicker and a seated robot spins and topples.
PARAMS holds the phase timings and thresholds.
"""

import numpy as np

JOINTS = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
          "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
          "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
IDX = {n: i for i, n in enumerate(JOINTS)}

KP = np.array([160, 160, 160, 160, 40, 480, 320, 160, 40, 480, 320, 30, 30, 30, 30, 30, 30.0])
KD = np.array([6, 4, 6, 5, 0.5, 20, 8, 5, 0.5, 20, 8, 0.6, 0.6, 0.4, 0.6, 0.6, 0.4])

PARAMS = {
    "flat_t": 1.0,        # s, length of the flatten phase
    "roll_t": 0.7,        # s, ramp into the roll-over pose
    "roll_switch": 1.0,   # s after the ramp: still flat on the front -> draw up the other knee
    "roll_giveup": 3.0,   # s after the ramp: stalled -> flatten and start again on the other side
    "face_up_done": 0.5,  # roll is finished when the chest's upward component exceeds this
    "situp_t": 1.2,       # s, ramp into the folded sit-up pose
    "situp_h": 0.38,      # m, root height at which the folded sit counts as reached
    "situp_giveup": 3.0,  # s after the ramp without reaching it -> start over
    "situp_stall": 1.2,   # s after the ramp: still below 0.25 m (stuck on a side) -> start over early
    "sit_t": 1.5,         # s, ramp from the fold to the upright sit
    "retry_h": 0.30,      # m, root height below which a seated robot is taken to have fallen
    "retry_until": 11.0,  # s, no restarts after this time (a restart needs up to 4 s to sit again)
}


def pose(**kw):
    """Joint targets [rad] from degrees: ay/az/ax waist, hx/hz/hy/kn both legs, s1/s2/el both arms (mirrored)."""
    q = np.zeros(17)
    for k, v in kw.items():
        r = np.radians(v)
        if k in ("ay", "az", "ax"):
            q[IDX["abdomen_" + k[1]]] = r
        elif k in ("hx", "hz", "hy"):
            q[IDX["right_hip_" + k[1]]] = r
            q[IDX["left_hip_" + k[1]]] = r
        elif k == "kn":
            q[IDX["right_knee"]] = r
            q[IDX["left_knee"]] = r
        elif k in ("s1", "s2"):
            q[IDX["right_shoulder" + k[1]]] = r
            q[IDX["left_shoulder" + k[1]]] = -r
        elif k == "el":
            q[IDX["right_elbow"]] = r
            q[IDX["left_elbow"]] = r
        else:
            raise KeyError(k)
    return q


ARMS_OUT = dict(s1=-45, s2=-75, el=-20)      # hands out sideways at shoulder height, clear of the thighs
ARMS_PROP = dict(s1=55, s2=-85, el=-40)      # hands down and slightly behind the hips, as props
FLAT = pose(hx=-25, kn=-5, **ARMS_OUT)
ROLL_R = pose(az=40)                         # right knee drawn up, waist twisted, arms folded in front of the chest
ROLL_R[[IDX["right_hip_y"], IDX["right_knee"], IDX["left_hip_x"]]] = np.radians([-100, -100, -20])
ROLL_L = pose(az=-40)                        # mirror image
ROLL_L[[IDX["left_hip_y"], IDX["left_knee"], IDX["right_hip_x"]]] = np.radians([-100, -100, -20])
SIT_UP = pose(hx=-15, hy=-105, kn=-5, ay=-40, **ARMS_OUT)
SIT = pose(hx=-20, hy=-97, kn=-5, ay=-8, **ARMS_PROP)

_S = {}   # phase, its start time and start joint angles; reset at step 0


def _smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3.0 - 2.0 * s)


def _face_up(obs):
    """Vertical component of the chest's forward axis: +1 lying face up, -1 face down."""
    w, x, y, z = obs["gym_obs"][1:5]
    return 2.0 * (x * z - w * y)


def _start(obs, phase):
    _S.update(phase=phase, t0=float(obs["time"]), q_from=np.array(obs["joint_angles"], dtype=float))


def _blend(goal, tau, ramp):
    return _S["q_from"] + _smooth(tau / ramp) * (goal - _S["q_from"])


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    t = float(obs["time"])
    if obs["step"] == 0 or "phase" not in _S:
        _S.clear()
        _S["side"] = 0                              # which knee the roll draws up (even: right, odd: left)
        # flat on the front: roll straight away (spread limbs would block it); otherwise flatten first
        _start(obs, "flatten" if _face_up(obs) > -0.8 else "roll")
    phase, tau = _S["phase"], t - _S["t0"]

    if phase == "flatten":
        target = _blend(FLAT, tau, p["flat_t"])
        if tau >= p["flat_t"]:
            _start(obs, "situp" if _face_up(obs) > 0.0 else "roll")
    elif phase == "roll":
        target = _blend(ROLL_R if _S["side"] % 2 == 0 else ROLL_L, tau, p["roll_t"])
        face = _face_up(obs)
        if face > p["face_up_done"] or (tau > p["roll_t"] + 1.5 and face > 0.2):
            _start(obs, "situp")                    # past the side: the sit-up fold finishes the turn onto the seat
        elif tau > p["roll_t"] + p["roll_switch"] and face < -0.8 and not _S.get("switched"):
            _S["side"] += 1                         # no progress at all: draw up the other knee instead
            _start(obs, "roll")
            _S["switched"] = True
        elif tau > p["roll_t"] + p["roll_giveup"]:
            _S["side"] += 1
            _S["switched"] = False
            _start(obs, "flatten")
    elif phase == "situp":
        target = _blend(SIT_UP, tau, p["situp_t"])
        if tau >= p["situp_t"] + 0.5 and obs["height"] > p["situp_h"]:
            _start(obs, "sit")
        elif tau >= p["situp_t"] + p["situp_giveup"] or (tau >= p["situp_t"] + p["situp_stall"] and obs["height"] < 0.25):
            _start(obs, "flatten")                  # did not come up (or is stuck low on a side): flatten and retry
    else:  # sit
        target = _blend(SIT, tau, p["sit_t"])
        if obs["height"] < p["retry_h"] and t < p["retry_until"]:
            _start(obs, "flatten")

    torque = KP * (target - obs["joint_angles"]) - KD * obs["joint_rates"]
    return np.clip(torque / obs["gear"], -0.4, 0.4)
