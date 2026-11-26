"""Humanoid get-up: from lying in any pose to sitting on the floor (long sitting), a three-mode state machine.

Orientation comes from the root quaternion (obs["gym_obs"][1:5]) as the world-z components of the torso axes:
fz (chest, torso x), sz (left side, torso y), uz (head, torso z); lying on the back fz ~ +1, on the front fz ~ -1.
Every joint not commanded otherwise is held by the joint PD
    a = clip(KP * (q_target - q) - KD * qdot, -0.4, 0.4)      (KP in command units per rad; x gear = N m / rad).

1. ``roll`` (start not face up, or fallen over): a cyclic two-phase posture pattern (ROLL_P1 for roll_T1 s, then
   ROLL_P2 for roll_T2 s, repeated), tracked by the joint PD, rolls the body over its left side onto its back. If the
   right side is the lower one when the roll starts, the pattern is mirrored (left/right limbs swapped, lateral joints
   sign-flipped). A roll that has not reached the back within roll_flip s restarts, in the other direction if the
   chest still points down (fz < flip_fz). The pattern was found offline by a
   cross-entropy search over its 36 numbers on 11 visible seeds (scored by how far and how fast the chest turned up).
   -> ``sit`` once fz > supine_fz and |sz| < supine_sz.
2. ``sit``: a sit-up with trunk-pitch feedback on both hip-pitch motors,
       u_hip = clip(-sit_kp * (pb - sit_pbt) - sit_kd * pb_rate, -sit_umax, 0.4),   pb = atan2(fz, uz)
   (pb = +90 deg lying on the back, 0 upright). The flexion side is capped (sit_umax, ~78 N m): more than the trunk's
   gravity moment about the hips, less than that of the straight legs, so the trunk comes up while the legs stay on the
   floor as a counterweight. A leg whose thigh points upward (leftover from the roll; sine of elevation > leg_elev) is
   first brought down with leg_down hip extension. While the trunk leans back more than ab_curl the abdomen curls at
   full command. The target sit_pbt < 0 keeps the trunk slightly forward over the legs, so the seat is held by the hip
   extensors, which press the legs into the floor (statically stable, root ~0.52 m). Knees nearly straight (knee),
   legs spread wide for a broad base (sit_hz: with the hips flexed 90 deg, hip_z swings the thighs sideways; sit_hx
   turns them about their long axis); once the trunk is upright (uz > lat_uz) the abdomen side-bend also leans
   against sideways tilt: a_abx += lat_kp * sz + lat_kd * (roll rate about the chest axis).
   -> ``roll`` again if the robot stays low (< 0.3 m) on its front or a side for bad_steps steps.
"""
import numpy as np

# motor order: ab_y ab_z ab_x | rhx rhz rhy rk | lhx lhz lhy lk | rs1 rs2 re | ls1 ls2 le
KP = np.array([2, 1, 1, 1, 1, 1.5, 1, 1, 1, 1.5, 1, 1, 1, 1, 1, 1, 1.0])
KD = 0.1 * KP
PERM = [0, 1, 2, 7, 8, 9, 10, 3, 4, 5, 6, 14, 15, 16, 11, 12, 13]       # left <-> right
SIGN = np.array([1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, 1, -1, -1, 1.0])

PARAMS = dict(
    sit_kp=0.6,       # hip command per rad of trunk-pitch error
    sit_kd=0.1,       # hip command per rad/s of trunk-pitch rate
    sit_umax=0.26,    # largest hip-flexion command (keeps the straight legs on the floor)
    sit_pbt=-0.15,    # trunk-pitch target [rad]; negative = leaning forward over the legs
    ab_curl=0.6,      # abdomen curls at full command while the trunk leans back more than this [rad]
    knee=-0.05,       # knee target while sitting [rad]
    roll_T1=0.6169, roll_T2=0.9404,   # durations of the two roll phases [s]
    supine_fz=0.85,   # roll -> sit when the chest points up at least this much ...
    supine_sz=0.5,    # ... and the body is turned to a side by less than this
    bad_steps=30,     # steps lying low on the front/side before rolling again
    sit_hx=-0.4,      # hip_x target while sitting [rad] (with the hip flexed 90 deg: turns the knees outward)
    sit_hz=-0.5,      # hip_z target while sitting [rad] (with the hip flexed 90 deg: negative spreads the legs)
    lat_kp=1.0, lat_kd=0.1, lat_uz=0.7,   # lateral balance on the abdomen side-bend once uz > lat_uz
    roll_flip=4.0,    # a roll that has not reached the back after this long [s] restarts, and is tried the other
    flip_fz=0.3,      # way round if the chest still points down (fz < flip_fz)
    leg_elev=0.5,     # a thigh pointing up more than this (sine of elevation) is first brought down ...
    leg_down=0.3,     # ... with this hip-extension command
)

ROLL_P1 = np.array([-0.0169,  0.6029,  0.1138, -0.0466,  0.3043, -1.8615, -1.6910,  0.0833,  0.6100, -0.6913, -0.5732,  0.3511,  1.0500, -0.1210, -0.5448, -0.3453,  0.3671])
ROLL_P2 = np.array([ 0.0264,  0.7800,  0.1731,  0.0699,  0.5996, -1.7338, -0.3349, -0.0798,  0.0357, -0.0579, -0.7424,  0.1655,  0.1932, -0.6507,  0.5297,  1.0471, -1.2996])

_S = {}


def _mirror(v):
    return SIGN * v[PERM]


def _quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _pd(tgt, q, qd):
    return np.clip(KP * (tgt - q) - KD * qd, -0.4, 0.4)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0 or "mode" not in _S:      # new episode
        _S.clear()
        _S["mode"] = None
    q = np.asarray(obs["joint_angles"], float)
    qd = np.asarray(obs["joint_rates"], float)
    t = float(obs["time"])
    R = _quat2mat(np.asarray(obs["gym_obs"][1:5], float))
    fz, sz, uz = R[2, 0], R[2, 1], R[2, 2]
    h = float(obs["height"])

    if _S["mode"] is None:
        _S["mode"] = "sit" if fz > 0.5 else "roll"
        _S["t0"] = t
        _S["mir"] = sz > 0
    if _S["mode"] == "roll" and fz > P["supine_fz"] and abs(sz) < P["supine_sz"]:
        _S["mode"] = "sit"; _S["t0"] = t
    if _S["mode"] == "sit":
        if h < 0.3 and (fz < 0.5 or abs(sz) > 0.6):
            _S["bad"] = _S.get("bad", 0) + 1
        else:
            _S["bad"] = 0
        if _S["bad"] > P["bad_steps"]:
            _S["mode"] = "roll"; _S["t0"] = t; _S["mir"] = sz > 0; _S["bad"] = 0

    if _S["mode"] == "roll" and t - _S["t0"] > P["roll_flip"]:
        if fz < P["flip_fz"]:
            _S["mir"] = not _S["mir"]                     # no progress this way: roll the other way
        _S["t0"] = t
    if _S["mode"] == "roll":
        tau = t - _S["t0"]
        T1, T2 = P["roll_T1"], P["roll_T2"]
        tgt = ROLL_P1 if (tau % (T1 + T2)) < T1 else ROLL_P2
        if _S["mir"]:
            return _mirror(_pd(tgt, _mirror(q), _mirror(qd)))
        return _pd(tgt, q, qd)

    # sit up / hold the seat
    pb = np.arctan2(fz, uz)
    pbr = -float(np.asarray(obs["root_angular_velocity"], float) @ R[:, 1])
    tgt = np.zeros(17)
    tgt[6] = tgt[10] = P["knee"]
    tgt[3] = tgt[7] = P["sit_hx"]
    tgt[4] = tgt[8] = P["sit_hz"]
    a = _pd(tgt, q, qd)
    uh = np.clip(-P["sit_kp"] * (pb - P["sit_pbt"]) - P["sit_kd"] * pbr, -P["sit_umax"], 0.4)
    for j in (5, 9):
        elev = -np.cos(-pb + q[0] + q[j])          # thigh pointing up (> 0) or lying along the floor (~ 0)
        a[j] = P["leg_down"] if elev > P["leg_elev"] else uh
    if pb > P["ab_curl"]:
        a[0] = -0.4
    if uz > P["lat_uz"]:   # lateral balance: bend the abdomen against the torso's sideways lean (sz < 0: leaning left)
        lr = float(np.asarray(obs["root_angular_velocity"], float) @ R[:, 0])
        a[2] = np.clip(a[2] + P["lat_kp"] * sz + P["lat_kd"] * lr, -0.4, 0.4)
    return a
