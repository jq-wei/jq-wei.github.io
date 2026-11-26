"""Flat-foot knee hopper with heel-push / toe-off ankle law (hand-designed, no learning).

Phase machine (reset when obs["step"] == 0):
  FLIGHT   -> COMPRESS on foot contact.
  COMPRESS -> THRUST when the knee has reached its crouch angle and the body has stopped sinking
              (or the torso pitches nose-up past pitch_trig, or after t_stance_max steps).
  THRUST   -> FLIGHT at lift-off (or back to COMPRESS after t_stance_max steps without lift-off).

Control law per phase (commands in [-1, 1], gear 200 N m):
  Hip (thigh):  torso pitch PD in every phase (the torso is ~30x lighter in inertia than the leg, so hip
                torque re-pitches the torso), plus a weak thigh-posture PD toward th_st / th_fl.
  Knee (leg):   COMPRESS: PD toward the crouch angle kn_c (kn_c0 for the very first stance, which starts from a
                straight leg); THRUST: stiff PD toward kn_e plus feedforward u_ff, reduced by k_uv when faster
                than v_des; FLIGHT: soft PD toward kn_fl (knee nearly straight: a deep crouch would force the
                torso nose-up because the hip cannot swing forward).
  Ankle (foot): COMPRESS: compliant spring toward a flat foot, torque bounded by a_dorsi / a_plant so the
                centre of pressure stays inside the foot. THRUST: push = u_ffa + k_push * (thigh - th_push):
                leg under the body -> positive (heel push, body rolls forward over the foot), leg trailing ->
                negative (toe-off; pivoting on the 0.26 m toe moves the ankle forward and un-trails the leg).
                FLIGHT: PD toward a world-flat foot plus ft_fl, with a toe-down bias k_ft * (vx - v_des) that
                brakes when fast; once the dead-reckoned distance (integrated x_velocity) exceeds d_cruise the
                target speed drops to v_slow to reduce the late-episode speed run-away. On the rails
                (obs["locked_joints"]) the foot cannot tip or the body cannot translate, so the vertical rail
                gets a pure toe-off thrust (u_ffa_rail) and the in-place rigs land flat.

Gains (PARAMS): kp_*/kd_* are command per rad and per rad/s; k_push, k_ft, k_uv in command (or rad) per rad
or per m/s as noted above; t_stance_max in control steps (8 ms).
Visible-seed results: rail and balance contracts 10/10, forward on 187/200 seeds of the free system.
"""
import numpy as np

PARAMS = dict(
    kn_fl=-0.10, ft_fl=0.1, kp_ft_fl=1.5,    # flight knee target, landing foot bias (toe-up) [rad], foot PD gain
    kp_fl=0.6, kd_fl=0.04, kp_fl_th=0.3, th_fl=-0.12,   # flight joint PD, weak thigh placement
    kp_pf=3.0, kd_pf=0.30,                   # hip pitch PD in flight
    kn_c=-0.35, kp_kc=1.5, kd_kc=0.06,       # compression knee spring target / gains
    k_s=0.8, b_s=0.03, a_dorsi=0.10, a_plant=0.30,   # ankle spring to flat foot, torque bounds
    kn_e=-0.02, kp_ke=2.0, kd_ke=0.05, u_ff=0.5,     # thrust knee PD + feedforward
    u_ffa=0.15, k_push=3.0, th_push=-0.11,   # thrust ankle push = u_ffa + k_push*(thigh - th_push): heel push / toe-off
    u_ffa_min=-0.70, u_ffa_max=0.40,         # bounds of that push (negative = toe-off)
    u_ffa_rail=-0.7,                         # pure toe-off thrust on the vertical rail (pitch locked)
    k_ft=0.8, k_uv=0.3, v_des=0.65,          # speed loop: toe-down landing bias and thrust cut per m/s above v_des
    d_cruise=3.4, v_slow=0.5,                # cruise target speed once the dead-reckoned distance exceeds d_cruise [m]
    kp_p=2.0, kd_p=0.20, pitch_des=-0.05,    # hip pitch PD in stance
    kp_th=0.8, th_st=-0.12,                  # weak thigh posture term in stance
    pitch_trig=-0.15,                        # thrust early if the torso pitches nose-up past this
    t_stance_max=30,
)
_S = {"phase": "FLIGHT", "t": 0, "dist": 0.0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0:
        _S["phase"] = "FLIGHT"; _S["t"] = 0; _S["dist"] = 0.0
    contact = bool(obs["foot_contact"])
    _S["dist"] += obs["x_velocity"] * obs["dt"]                 # dead-reckoned distance (x is not observed)
    v_des = P["v_slow"] if _S["dist"] > P["d_cruise"] else P["v_des"]   # cruise slower once the distance is banked
    rail = "rooty" in tuple(obs.get("locked_joints", ()))   # pitch locked: foot cannot tip, toe-off is free
    q = np.asarray(obs["joint_angles"]); qd = np.asarray(obs["joint_rates"])
    pitch, prate, vz = obs["pitch"], obs["pitch_rate"], obs["z_velocity"]
    ph = _S["phase"]
    if ph == "FLIGHT":
        if contact:
            ph = "COMPRESS"; _S["t"] = 0
    elif ph == "COMPRESS":
        if not contact and _S["t"] > 2:
            ph = "FLIGHT"
        elif ((_S["t"] > 3 and vz >= 0.0 and q[1] < P["kn_c"] + 0.12) or q[1] < P["kn_c"] + 0.03
              or pitch < P["pitch_trig"] or _S["t"] > P["t_stance_max"]):
            ph = "THRUST"; _S["t"] = 0
    elif ph == "THRUST":
        if not contact:
            ph = "FLIGHT"
        elif _S["t"] > P["t_stance_max"]:
            ph = "COMPRESS"; _S["t"] = 0
    _S["phase"] = ph; _S["t"] += 1

    u = np.zeros(3)
    pitch_err = pitch - P["pitch_des"]
    if ph == "FLIGHT":
        u[0] = P["kp_fl_th"] * (P["th_fl"] - q[0]) - P["kp_pf"] * pitch_err - P["kd_pf"] * prate
        u[1] = P["kp_fl"] * (P["kn_fl"] - q[1]) - P["kd_fl"] * qd[1]
        if "rootx" in tuple(obs.get("locked_joints", ())):     # translation locked (hop in place): flat landing
            ft_fl = 0.0
        else:
            ft_fl = P["ft_fl"] - P["k_ft"] * (obs["x_velocity"] - v_des)  # fast -> toe-down landing (brake)
        ft_des = np.clip(pitch - q[0] - q[1] + ft_fl, -0.7, 0.7)
        u[2] = P["kp_ft_fl"] * (ft_des - q[2]) - P["kd_fl"] * qd[2]
    else:
        u[0] = P["kp_th"] * (P["th_st"] - q[0]) - P["kp_p"] * pitch_err - P["kd_p"] * prate
        spring = P["k_s"] * (0.0 - q[2]) - P["b_s"] * qd[2]
        if ph == "COMPRESS":
            u[1] = P["kp_kc"] * (P["kn_c"] - q[1]) - P["kd_kc"] * qd[1]
            u[2] = np.clip(spring, -P["a_plant"], P["a_dorsi"])
        else:
            u_ff = P["u_ff"] - P["k_uv"] * max(0.0, obs["x_velocity"] - v_des)   # fast -> weaker thrust
            u[1] = P["kp_ke"] * (P["kn_e"] - q[1]) - P["kd_ke"] * qd[1] + u_ff
            if rail:
                u[2] = P["u_ffa_rail"]
            else:
                push = np.clip(P["u_ffa"] + P["k_push"] * (q[0] - P["th_push"]), P["u_ffa_min"], P["u_ffa_max"])   # trailing leg -> toe-off
                u[2] = push - P["b_s"] * qd[2]
    return np.clip(u, -1, 1)
