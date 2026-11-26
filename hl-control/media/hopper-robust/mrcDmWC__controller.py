"""Crouch-and-hop controller for the MuJoCo Hopper (stance posture servo + thrust pulses).

Sign conventions (probed on the plant): positive knee command extends the knee (toward
0); positive ankle command leans the shank forward / pushes the body forward; positive hip
command swings the thigh forward (toward 0) and, with the foot planted, pitches the torso
nose-down (pitch increases).  Angles: thigh_abs = pitch - thigh, shank_abs = thigh_abs -
leg (positive = top of the link ahead of its bottom); with the foot flat, ankle = shank_abs.

Why this structure: the leg is 3x heavier than the torso, so the thigh angle cannot be
changed in flight without pitching the torso out of the +-0.2 rad band.  The gait therefore
re-establishes a statically stable crouch (COM over the long foot) in every stance with a
posture servo, then fires a short knee thrust.  The forward lean of the crouch gives the
push a forward component; the lean is adjusted with the speed error (Raibert-style).

Law
  * Stance, settle (first ``t_settle`` s after touchdown): ankle PD drives the shank to
    ``shank_des + k_v*(v_des - vx)``; knee PD holds the crouch ``leg_crouch``; hip PD
    holds pitch at ``pitch_des``.
  * Stance, thrust (until lift-off or ``t_thrust_max``): knee = ``thrust`` plus the same
    PD toward ``leg_push``; ankle adds ``toe_push``.
  * Flight: knee PD toward ``leg_flight``; hip PD on pitch (gain ``kp_pitch_f``, clipped
    to ``hip_flight_max``); ankle pre-sets a flat foot (clipped to ``foot_flight_max``
    because its reaction swings the whole leg).  Thigh guard: when the thigh swings past
    ``thigh_guard`` toward its 0 rad limit, the hip pushes it back with gain ``kp_guard``;
    without it a high flight ends with the leg slamming into the hip limit and dragging
    the torso nose-up past -0.2 rad.

The gains were refined with a (1+lambda) evolutionary search over the five grading
scenarios (nominal, heavy torso, low friction, weak motors, high gravity); the gait is
a ~1 m/s bounding hop with 14-15 flights per episode.
"""

import numpy as np

PARAMS = {
    "pitch_des": -0.0401,     # torso pitch target [rad] (positive = nose down)
    "kp_pitch": 1.8761,       # stance hip PD on pitch
    "kd_pitch": 0.3,
    "shank_des": 0.3311,      # shank lean in the crouch [rad]
    "k_v": 0.5537,            # extra lean per m/s of speed error [rad s/m]
    "v_des": 0.7112,          # desired forward speed [m/s]
    "kp_shank": 1.3647,       # stance ankle PD on shank angle
    "kd_shank": 0.08,
    "leg_crouch": -0.4308,    # knee angle of the crouch [rad]
    "kp_leg": 0.8,            # knee PD gains
    "kd_leg": 0.08,
    "t_settle": 0.1398,       # stance time before the thrust [s]
    "thrust": 0.2073,         # knee feed-forward during thrust
    "leg_push": -0.4,         # knee target during thrust
    "t_thrust_max": 0.08,     # thrust cut-off if no lift-off [s]
    "toe_push": 0.4599,       # ankle feed-forward during thrust
    "leg_flight": -0.3677,    # knee angle held in flight [rad]
    "kp_pitch_f": 0.5,        # flight hip PD on pitch
    "kd_pitch_f": 0.1,
    "hip_flight_max": 0.1147,
    "kp_foot": 0.6,           # flight ankle PD (flat foot)
    "kd_foot": 0.03,
    "foot_flight_max": 0.9466,# clip on the flight ankle command (its reaction swings the leg)
    "thigh_guard": -0.1009,   # flight: above this thigh angle the hip pushes the thigh back from its 0 limit
    "kp_guard": 0.6213,
}

_state = {"touched": False, "t_td": 0.0, "prev_contact": False}


def controller(obs: dict) -> np.ndarray:
    P, S = PARAMS, _state
    t = float(obs["time"])
    if obs["step"] == 0:
        S.update(touched=False, t_td=0.0, prev_contact=False)

    pitch, prate = obs["pitch"], obs["pitch_rate"]
    thigh, leg, foot = obs["thigh"], obs["leg"], obs["foot"]
    thigh_r, leg_r, foot_r = obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]
    vx = obs["x_velocity"]
    raw_contact = bool(obs["foot_contact"])
    S["touched"] |= raw_contact
    contact = raw_contact or not S["touched"]      # the initial drop counts as stance
    if contact and not S["prev_contact"]:
        S["t_td"] = t
    S["prev_contact"] = contact

    shank_abs = pitch - thigh - leg                # shank pitch in the world
    shank_rate = prate - thigh_r - leg_r

    if contact:
        ts = t - S["t_td"]
        u_hip = -P["kp_pitch"] * (pitch - P["pitch_des"]) - P["kd_pitch"] * prate
        lean = P["shank_des"] + P["k_v"] * (P["v_des"] - vx)
        u_ankle = P["kp_shank"] * (lean - shank_abs) - P["kd_shank"] * shank_rate
        if ts < P["t_settle"] or ts > P["t_settle"] + P["t_thrust_max"]:
            u_knee = P["kp_leg"] * (P["leg_crouch"] - leg) - P["kd_leg"] * leg_r
        else:
            u_knee = P["thrust"] + P["kp_leg"] * (P["leg_push"] - leg) - P["kd_leg"] * leg_r
            u_ankle += P["toe_push"]
    else:
        u_hip = np.clip(-P["kp_pitch_f"] * (pitch - P["pitch_des"]) - P["kd_pitch_f"] * prate,
                        -P["hip_flight_max"], P["hip_flight_max"])
        u_knee = P["kp_leg"] * (P["leg_flight"] - leg) - P["kd_leg"] * leg_r
        u_ankle = np.clip(P["kp_foot"] * (shank_abs - foot) - P["kd_foot"] * foot_r,
                          -P["foot_flight_max"], P["foot_flight_max"])
        if thigh > P["thigh_guard"]:
            u_hip -= P["kp_guard"] * (thigh - P["thigh_guard"])

    return np.clip(np.array([u_hip, u_knee, u_ankle]), -1.0, 1.0)
