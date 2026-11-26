"""Hand-designed hopping controller for the MuJoCo Hopper (design V: rail -> balance -> free).

Control law
-----------
A three-phase state machine driven by the foot-contact flag.

* ``settle`` (first ``settle_time`` s): stand on a straight knee while the ankle rotates the
  whole body until the thigh is ``lean0`` rad off its 0 rad limit, so the hip has authority
  in both directions before the first hop.
* ``stance`` (foot on the floor):
  - hip      : PD on torso pitch -> ``pitch_ref`` (``kp_pitch``, ``kd_pitch``). Above the
               desired speed the set-point is lowered (``kv_pitch_s``, floor ``pitch_s_min``):
               a nose-up torso puts the centre of mass behind the foot and decelerates.
  - ankle    : PD on the thigh "lean" (world angle of the thigh, positive = foot behind the
               hip) -> ``lean_ref`` (``ka_p``, ``ka_d``, saturation ``ank_max``). This is the
               inverted-pendulum balance about the foot; holding lean > 0 with pitch ~ 0 keeps
               the thigh negative. ``lean_ref = lean0 - kv * (v - v_des)``.
  - knee     : while descending, a stiff PD holds the landing angle (``kp_knee``, ``kd_knee``,
               the knee must not fold under load); once the body rises, constant ``thrust``
               until the knee reaches ``knee_stop``, then a PD hold just short of the 0 stop.
* ``flight`` (foot in the air):
  - hip      : PD on torso pitch (``kp_pitch_f``, ``kd_pitch_f``). In flight hip torque mostly
               rotates the light torso (the leg is three times heavier), so the thigh angle is
               not commanded directly (``kp_hip_f`` = 0). Above the desired speed the target is
               lowered (``kv_pitch``, floor ``pitch_f_min``) so the foot lands further forward.
  - knee     : soft PD (``kp_j``, ``kd_j``) to the landing angle ``lg_land``, straightened
               with forward speed (``kv_knee``, cap ``lg_land_max``): a straighter knee is the
               only way the joint limits allow the foot to be placed further forward, which
               shortens the vault over the foot and regulates speed.
  - ankle    : soft PD that keeps the sole horizontal for a flat landing.

Speed is set indirectly (``v_des``); the free system settles near 1.2 m/s (about 9.5 m
per episode), the rail rigs hop in place at 12-20 flights per episode.
"""
import numpy as np

PARAMS = dict(
    pitch_ref=0.03,    # torso pitch set-point [rad] (positive = nose down)
    lean0=0.12,        # nominal thigh world angle in stance (positive = foot behind hip) [rad]
    kv=0.15,           # lean_ref change per m/s of speed error
    v_des=0.8,         # desired forward speed [m/s]
    kv_pitch=0.3, pitch_f_min=-0.12,     # flight pitch target lowering with excess speed
    kv_pitch_s=0.3, pitch_s_min=-0.1,    # stance pitch target lowering with excess speed
    lg_land=-0.3, kv_knee=0.2, lg_land_max=-0.1,   # knee landing angle and its speed dependence
    kp_j=0.4, kd_j=0.02,                 # flight joint PD (knee, ankle)
    kp_hip_f=0.0, kp_pitch_f=0.6, kd_pitch_f=0.1,  # flight hip law
    kp_pitch=2.5, kd_pitch=0.5,          # stance hip pitch loop
    ka_p=1.5, ka_d=0.15, ank_max=1.0,    # stance ankle lean loop
    thrust=0.4, knee_stop=-0.08, kp_knee=3.0, kd_knee=0.1,   # stance knee
    settle_time=0.6,                     # [s]
)
P = PARAMS
_state = {"phase": "settle", "t_td": 0.0}


def controller(obs):
    th, lg, ft = obs["thigh"], obs["leg"], obs["foot"]
    thr, lgr, ftr = obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]
    pitch, pr = obs["pitch"], obs["pitch_rate"]
    t = obs["time"]
    v = obs["x_velocity"]
    if obs["step"] == 0:
        _state["phase"] = "settle"; _state["t_td"] = t
    contact = obs["foot_contact"]
    ph = _state["phase"]
    if ph == "settle":
        if t > P["settle_time"]:
            ph = "stance"
    elif contact and ph == "flight":
        ph = "stance"; _state["t_td"] = t
    elif not contact and ph == "stance":
        ph = "flight"
    _state["phase"] = ph

    u = np.zeros(3)
    flat = pitch - th - lg          # ankle angle that makes the sole horizontal
    lean = pitch - th               # world angle of the thigh (positive = foot behind hip)
    excess = max(v - P["v_des"], 0.0)
    lean_ref = P["lean0"] - P["kv"] * (v - P["v_des"])

    if ph == "flight":
        th_land = P["pitch_ref"] - lean_ref
        pitch_ref_f = max(P["pitch_f_min"], P["pitch_ref"] - P["kv_pitch"] * excess)
        u[0] = P["kp_hip_f"] * (th_land - th) - P["kp_pitch_f"] * (pitch - pitch_ref_f) - P["kd_pitch_f"] * pr
        lg_land = min(P["lg_land_max"], P["lg_land"] + P["kv_knee"] * max(v, 0.0))
        u[1] = P["kp_j"] * (lg_land - lg) - P["kd_j"] * lgr
        u[2] = P["kp_j"] * (flat - ft) - P["kd_j"] * ftr
    else:
        pitch_ref_s = max(P["pitch_s_min"], P["pitch_ref"] - P["kv_pitch_s"] * excess)
        u[0] = -P["kp_pitch"] * (pitch - pitch_ref_s) - P["kd_pitch"] * pr
        u[2] = np.clip(P["ka_p"] * (lean_ref - lean) - P["ka_d"] * (pr - thr), -P["ank_max"], P["ank_max"])
        if ph == "settle":
            u[1] = P["kp_knee"] * (0.0 - lg) - P["kd_knee"] * lgr
        elif obs["z_velocity"] < 0:
            u[1] = P["kp_knee"] * (P["lg_land"] - lg) - P["kd_knee"] * lgr
        elif lg < P["knee_stop"]:
            u[1] = P["thrust"]
        else:
            u[1] = P["kp_knee"] * (P["knee_stop"] - lg) - P["kd_knee"] * lgr
    return np.clip(u, -1.0, 1.0)
