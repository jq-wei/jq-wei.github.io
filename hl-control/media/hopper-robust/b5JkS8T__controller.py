"""Hopper controller: Raibert-style three-phase state machine, gains found by CMA-ES on visible seeds.

Stance (foot in contact): phase 0 compresses the knee (-crouch_u) until the knee reaches crouch_depth or the body
rises; phase 1 thrusts the knee (+thrust) until it is straight. The hip holds a thigh angle hs0 - hk*knee (kps/kds)
kept off the 0 limit, plus a torso-pitch loop (kpp/kdp toward pitch_des); the ankle runs its own pitch loop (ka/kad)
and a phase-dependent angle bias (ab0/ab1, kpa).
Flight: knee folds to fl_leg (light leg, so hip torque swings it instead of pitching the torso) and extends to
ext_leg once descending faster than zv_ext; the thigh goes to fl_thigh + kv*(x_velocity - vdes) (foot placement),
foot to fl_foot, with soft gains kpf/kdf.
"""
import numpy as np

PARAMS = {
 "crouch_u": 0.09358363946772724,
 "crouch_depth": 0.3676291681100672,
 "thrust": 0.35091058895037847,
 "hs0": -0.16476569257508422,
 "hk": 0.17006589043745235,
 "kps": 1.302410709710048,
 "kds": 0.7338703074672839,
 "kpp": 0.8774816293714284,
 "kdp": 0.8205767061750227,
 "pitch_des": -0.06109106475622441,
 "fl_leg": -0.10195771852336177,
 "fl_thigh": -0.052716257331427596,
 "kpf": 0.491707127672619,
 "kdf": 0.16530583961783427,
 "zv_ext": 0.605818114986571,
 "ext_leg": -0.4259580208124174,
 "ka": 1.8755589406184936,
 "kad": 0.4811279223313507,
 "kv": -0.34429666506303547,
 "vdes": 1.6055847456952197,
 "ab0": 0.2941552713674257,
 "ab1": -0.32249903869481905,
 "kpa": 0.44486990041760216,
 "fl_foot": 0.31703442451205155
}
PARAM_BOUNDS = {"crouch_u": (0.0, 1.0), "crouch_depth": (0.1, 1.2), "thrust": (0.2, 1.0),
                "hs0": (-0.6, -0.02), "hk": (-1.0, 1.0), "kps": (0.0, 5.0), "kds": (0.0, 1.0), "kpp": (0.0, 5.0),
                "kdp": (0.0, 1.0), "pitch_des": (-0.1, 0.15), "fl_leg": (-1.5, 0.0), "fl_thigh": (-0.8, 0.0),
                "kpf": (0.0, 2.0), "kdf": (0.0, 0.3), "zv_ext": (0.0, 2.0), "ext_leg": (-0.8, 0.0),
                "ka": (-2.0, 2.0), "kad": (-0.5, 0.5), "kv": (-0.5, 0.5), "vdes": (0.0, 2.0),
                "ab0": (-0.7, 0.7), "ab1": (-0.7, 0.7), "kpa": (0.0, 2.0), "fl_foot": (-0.7, 0.7)}
_state = {"phase": 0}

def reset():
    _state["phase"] = 0

def controller(obs):
    P = PARAMS
    pitch, pr = obs["pitch"], obs["pitch_rate"]
    th, lg, ft = obs["thigh"], obs["leg"], obs["foot"]
    thr, lgr, ftr = obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]
    if obs["foot_contact"]:
        ph = _state["phase"]
        if ph >= 10: ph = 0          # just landed
        if ph == 0 and (lg < -P["crouch_depth"] or (obs["z_velocity"] > 0.1 and lg < -0.1)): ph = 1
        if ph == 1 and lg > -0.02: ph = 0
        _state["phase"] = ph
        ths = P["hs0"] - P["hk"] * lg
        u_th = P["kps"] * (ths - th) - P["kds"] * thr + P["kpp"] * (P["pitch_des"] - pitch) - P["kdp"] * pr
        u_lg = -P["crouch_u"] if ph == 0 else P["thrust"]
        u_ft = P["ka"] * (P["pitch_des"] - pitch) - P["kad"] * pr + P["kpa"] * ((P["ab0"] if ph == 0 else P["ab1"]) - ft) - 0.02 * ftr
    else:
        _state["phase"] = 10
        u_th = P["kpf"] * (P["fl_thigh"] + P["kv"] * (obs["x_velocity"] - P["vdes"]) - th) - P["kdf"] * thr
        tgt = P["ext_leg"] if obs["z_velocity"] < -P["zv_ext"] else P["fl_leg"]
        u_lg = P["kpf"] * (tgt - lg) - P["kdf"] * lgr
        u_ft = 0.5 * (P["fl_foot"] - ft) - 0.05 * ftr
    return np.clip(np.array([u_th, u_lg, u_ft]), -1, 1)
