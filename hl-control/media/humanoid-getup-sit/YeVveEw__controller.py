"""Sit-up controller for the MuJoCo Humanoid that starts lying on the floor.

Control law: a small state machine chooses joint-angle targets, a joint-space PD turns them into
torques, the torques are divided by the motor gears, clipped to [-0.4, 0.4] and low-pass filtered.
No model, no learning; the only feedback besides the joint encoders is the direction the chest and
the pelvis face (from the root quaternion in ``gym_obs`` and the three abdomen angles) and the root
height.

Phases (targets not mentioned are zero, i.e. the model's neutral pose; knees are kept at -0.05):

``roll``    Bring the body onto its back.  The side that is higher (sign of the torso's lateral axis
            in world z) is the "top" side; the code is written for top = left and mirrored otherwise.
            * strut: the top leg is flexed in front of the body (``strut_hip``) and driven across the
              midline with saturated adduction torque (``strut_hip_z``, ``strut_hip_x`` lie beyond
              the joint ranges on purpose), knee bent (``push_knee``) for the first ``strut_bend``
              seconds of every ``strut_cycle`` and straight afterwards: the knee, then the foot,
              presses on the floor and levers the pelvis over the lower side.  The waist is twisted
              (``push_twist``) so that the chest leads and the lower arm is tucked along the body
              (``low_arm``) so that it does not prop the chest.  Repeating the bend/straighten cycle
              frees poses that are propped on an arm.
            * release: once the pelvis faces up by ``release_pf`` (or the chest by 0.8) the leg is
              laid back and the top arm is swung behind the body (``arm_back``); gravity finishes
              the turn.  Falling back below ``strut_pf`` re-arms the strut; after
              ``release_timeout`` seconds without getting flat the machine moves on to ``settle``.
``settle``  ``settle_time`` seconds stretched out on the back (chest facing up by ``supine_fz``);
            if the chest does not face up by 0.8 at its end, back to ``roll``.
``fold``    Sit-up: curl the abdomen (``fold_abd``) and flex both hips (``fold_hip``) with straight
            legs for ``fold_time`` seconds.  Curled, the trunk is lighter about the hips than the
            straight legs, so the body jack-knifes and comes to rest folded forward over the legs.
``rise``    Unfold over ``rise_time`` seconds to the seated posture: abdomen ``sit_abd``, hips
            ``sit_hip`` (pelvis leaning ~10 degrees forward, so the legs stay loaded on the floor and
            the hip springs cannot tip the trunk backwards).  The slow ramp avoids overshooting.
``hold``    Keep that posture (root at about 0.52 m).  If the root stays below ``fail_height`` for
            ``fail_time`` seconds during rise/hold, the machine starts over from ``roll``.

Gains: ``KP`` [N m/rad] and ``KD`` [N m s/rad] per joint in motor order.  They are deliberately soft
(with the 15 ms control step, stiffer gains chatter between the torque limits and shake the robot
off the floor); the 0.5/0.5 first-order filter on the command removes what chatter is left.
"""
import numpy as np

NAMES = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
         "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
         "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
I = {n: i for i, n in enumerate(NAMES)}
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
KP = np.array([120, 60, 60, 60, 60, 250, 150, 60, 60, 250, 150, 10, 10, 10, 10, 10, 10], float)
KD = np.array([8, 5, 5, 5, 5, 14, 8, 5, 5, 14, 8, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6], float)
ARM_SIGN = np.array([-1.0, -1.0, 1.0])     # shoulder angles change sign under left/right mirroring, the elbow does not
KNEE_STRAIGHT = -0.05                      # rad, just inside the knee's range

PARAMS = {
    # roll (prone / on a side -> supine)
    "strut_hip": -1.5, "strut_hip_z": 1.5, "strut_hip_x": 0.5, "push_knee": -1.6, "push_twist": -0.7,
    "strut_cycle": 1.5, "strut_bend": 0.5, "low_arm": 1.0, "release_pf": 0.3, "strut_pf": -0.1, "arm_back": 1.4,
    "release_timeout": 2.0, "supine_fz": 0.9, "settle_time": 0.4,
    # sit-up
    "fold_abd": -1.3, "fold_hip": -1.9, "fold_time": 1.2,
    "rise_time": 2.0, "sit_hip": -1.75, "sit_abd": 0.0,
    # retry
    "fail_height": 0.35, "fail_time": 0.6,
}

_S = {}     # per-episode state: phase, phase start time, roll direction, strut/release flag, last command


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 1:
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def mirror(tgt):
    """Left/right mirror image of a target vector (motor order)."""
    out = tgt.copy()
    out[1], out[2] = -tgt[1], -tgt[2]                         # abdomen twist and side bend
    out[3:7], out[7:11] = tgt[7:11], tgt[3:7]                 # legs
    out[11:14], out[14:17] = tgt[14:17] * ARM_SIGN, tgt[11:14] * ARM_SIGN
    return out


def _enter(phase, t):
    _S["phase"] = phase
    _S["t0"] = t


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    q = np.asarray(obs["joint_angles"], float)
    qd = np.asarray(obs["joint_rates"], float)
    t = float(obs["time"])
    h = float(obs["height"])
    if obs["step"] == 0 or "t0" not in _S:                    # new episode
        _S.clear()
        _S.update(phase="roll", t0=t, d=1.0, low_since=None, released=False, rel_t0=t)

    # which way the chest (torso x axis) and the pelvis face, and which side is up (torso y axis)
    R = quat_to_mat(np.asarray(obs["gym_obs"][1:5], float))
    chest_up, left_up = R[2, 0], R[2, 1]
    pelvis_front_up = (R @ rot(2, q[1]) @ rot(1, q[0]) @ rot(0, q[2]))[2, 0]
    if abs(left_up) > 0.05 and _S["phase"] == "roll" and not _S["released"]:
        _S["d"] = 1.0 if left_up > 0 else -1.0                # roll over the lower side

    # ---- phase transitions ----
    ph, tp = _S["phase"], t - _S["t0"]
    if ph == "roll" and (chest_up > P["supine_fz"] or (_S["released"] and t - _S["rel_t0"] > P["release_timeout"])):
        _enter("settle", t)
    elif ph == "settle" and tp > P["settle_time"]:
        _S["released"] = False
        _enter("fold" if chest_up > 0.8 else "roll", t)
    elif ph == "fold" and tp > P["fold_time"]:
        _enter("rise", t)
    elif ph == "rise" and tp > P["rise_time"]:
        _enter("hold", t)
    ph, tp = _S["phase"], t - _S["t0"]
    if ph in ("rise", "hold"):                                # fell over while sitting up: start again
        if h < P["fail_height"] and (ph == "hold" or tp > 1.0):
            if _S["low_since"] is None:
                _S["low_since"] = t
            elif t - _S["low_since"] > P["fail_time"]:
                _S["low_since"] = None
                _S["released"] = False
                _enter("roll", t)
        else:
            _S["low_since"] = None
        ph, tp = _S["phase"], t - _S["t0"]

    # ---- joint targets ----
    tgt = np.zeros(17)
    if ph == "roll":                                          # written for top side = left, mirrored below
        if _S["released"] and pelvis_front_up < P["strut_pf"] and chest_up < P["strut_pf"]:
            _S["released"] = False
        elif not _S["released"] and (pelvis_front_up > P["release_pf"] or chest_up > 0.8):
            _S["released"] = True
            _S["rel_t0"] = t
        if not _S["released"]:
            tgt[I["left_hip_y"]] = P["strut_hip"]
            tgt[I["left_hip_z"]] = P["strut_hip_z"]
            tgt[I["left_hip_x"]] = P["strut_hip_x"]
            tgt[I["left_knee"]] = P["push_knee"] if tp % P["strut_cycle"] < P["strut_bend"] else KNEE_STRAIGHT
            tgt[I["abdomen_z"]] = P["push_twist"]
            tgt[I["right_shoulder1"]] = P["low_arm"]
        else:
            tgt[I["left_shoulder1"]] = P["arm_back"]
            tgt[I["left_shoulder2"]] = P["arm_back"]
        if _S["d"] < 0:
            tgt = mirror(tgt)
    elif ph == "fold":
        tgt[I["abdomen_y"]] = P["fold_abd"]
        tgt[I["left_hip_y"]] = tgt[I["right_hip_y"]] = P["fold_hip"]
    elif ph in ("rise", "hold"):
        a = min(tp / P["rise_time"], 1.0) if ph == "rise" else 1.0
        tgt[I["abdomen_y"]] = (1 - a) * P["fold_abd"] + a * P["sit_abd"]
        tgt[I["left_hip_y"]] = tgt[I["right_hip_y"]] = (1 - a) * P["fold_hip"] + a * P["sit_hip"]
    tgt[I["left_knee"]] = min(tgt[I["left_knee"]], KNEE_STRAIGHT)
    tgt[I["right_knee"]] = min(tgt[I["right_knee"]], KNEE_STRAIGHT)

    # ---- PD, gear, clip, low-pass ----
    cmd = np.clip((KP * (tgt - q) - KD * qd) / GEAR, -0.4, 0.4)
    cmd = 0.5 * cmd + 0.5 * _S.get("cmd", cmd)
    _S["cmd"] = cmd
    return cmd
