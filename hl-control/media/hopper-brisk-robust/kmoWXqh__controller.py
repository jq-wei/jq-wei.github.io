"""Hand-designed hopping controller for the MuJoCo Hopper (Hopper-v5 plant).

Structure (one stance/flight state machine, no learning):

Stance (foot in contact)
  * hip   : PD on torso pitch about ``pitch_st`` (positive hip torque pitches the torso nose-down),
            gains ``pitch_kp``/``pitch_kd``, saturated at ``hip_max``.
  * knee  : "comp" phase = a soft spring (``comp_kp``) pulling the knee to ``stance_knee``;
            "thrust" phase (entered when the knee is bent enough or after ``comp_time``) = constant
            torque ``thrust`` until the knee is straight, then only a light hold.  The very first
            stance squats slowly to ``first_knee`` with the torso level (``first_pitch``).
  * ankle : balances the leg over the foot.  phi = pitch - thigh is the thigh angle from vertical
            (positive = hip ahead of the foot); the ankle torque -phi_kp (phi - phi_des) - phi_kd phi_dot
            shifts the centre of pressure along the 0.39 m foot (positive torque = toe up = rocks the
            hip forward).  ``k_vank`` adds a toe-down push when faster than ``v_des`` (the push also
            injects nose-up angular momentum, which brings the leg forward during the next flight);
            ``toe_off`` is a constant ankle bias during the thrust phase.

Flight
  * hip   : PD on torso pitch about ``pitch_fl`` (``fp_kp``/``fp_kd``), plus a pull toward ``thigh_f``
            (``hip_fkp``) only when the thigh is above that angle, so the hip never sits on its upper
            limit at touchdown; saturated at ``hip_fmax``.  The leg is three times heavier than the
            torso, so the hip cannot swing the leg in flight without pitching the torso out of the
            +-0.2 rad band; attitude is held instead and foot placement is left to the stance ankle.
  * knee  : PD to ``land_knee`` (pre-bent so the next thrust has range).
  * ankle : PD to the angle that puts the sole parallel to the ground plus ``foot_f`` (toe-up offset).

Before the first touchdown the controller outputs zero (the plant starts 4 cm above the floor).
``PARAMS`` were found with a multi-scenario (1+1) evolution strategy on seeds 0-2 of the nominal,
heavy-torso, low-friction, weak-motor and high-gravity plants; ``C`` holds fixed constants.
"""
import numpy as np

PARAMS = dict(   # found by the multi-scenario search described above
    pitch_st=0, stance_knee=-0.5, thrust=0.6, phi_des=0.08,
    phi_kp=1, phi_kd=0.1, toe_off=0, k_vank=0.6,
    land_knee=-0.4, foot_f=0.2, fp_kp=2, pitch_kp=2,
    v_des=1.3, comp_time=0.06, pitch_fl=0, thigh_f=-0.15,
    hip_fkp=1.5, ankle_kp=1.5,
)
PARAM_BOUNDS = dict(
    pitch_st=(-0.12, 0.12), stance_knee=(-1.0, -0.2), thrust=(0.25, 1.0), phi_des=(0.0, 0.4), phi_kp=(0.2, 4.0), phi_kd=(0.0, 0.5),
    toe_off=(-0.5, 0.5), k_vank=(-0.5, 0.5), land_knee=(-0.9, -0.1), foot_f=(-0.3, 0.6), fp_kp=(0.0, 6.0), pitch_kp=(0.5, 6.0),
    v_des=(0.9, 2.2), comp_time=(0.0, 0.15), pitch_fl=(-0.12, 0.12), thigh_f=(-0.5, -0.03), hip_fkp=(0.0, 4.0), ankle_kp=(0.3, 4.0),
)
C = dict(   # fixed constants
    pitch_kd=0.1, hip_fkd=0.05, hip_fmax=0.6, fp_kd=0.1, ankle_kd=0.05, swing_all=0,
    hip_max=1.0, comp_kp=0.3, knee_kp=1.5, knee_kd=0.05, comp_tol=0.1, ank_max=1.0, first_knee=-0.5, first_pitch=0.0,
)
S = dict(prev_contact=True, t_td=0.0, t_lo=0.0, step=-1, touched=False, phase="comp", hops=0)

def controller(obs):
    P = {**C, **PARAMS}
    if obs["step"] <= S["step"]:  # new episode
        S.update(prev_contact=True, t_td=0.0, t_lo=0.0, touched=False, phase="comp", hops=0)
    S["step"] = obs["step"]
    t = obs["time"]; c = obs["foot_contact"]
    pitch, pr = obs["pitch"], obs["pitch_rate"]
    th, lg, ft = obs["thigh"], obs["leg"], obs["foot"]
    thr, lgr, ftr = obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]
    vx = obs["x_velocity"]
    if c and not S["prev_contact"]:
        S["t_td"] = t; S["phase"] = "comp"
    if (not c) and S["prev_contact"]:
        S["t_lo"] = t
        if S["touched"]:
            S["hops"] += 1
    S["prev_contact"] = c
    if c:
        S["touched"] = True
    if not S["touched"]:
        return np.zeros(3)
    u = np.zeros(3)
    phi = pitch - th            # leg (thigh) angle from vertical, positive = hip ahead of the knee/foot
    phi_rate = pr - thr
    foot_flat = pitch - th - lg  # ankle angle that puts the sole parallel to the ground
    if c:
        ts = t - S["t_td"]
        first = S["hops"] == 0
        e = pitch - (P["first_pitch"] if first else P["pitch_st"])
        sk = P["first_knee"] if first else P["stance_knee"]
        u[0] = np.clip(-P["pitch_kp"] * e - P["pitch_kd"] * pr, -P["hip_max"], P["hip_max"])
        if S["phase"] == "comp" and (lg <= sk + P["comp_tol"] or (ts > P["comp_time"] and not first)):
            S["phase"] = "thrust"
        if S["phase"] == "thrust" and lg > -0.03 and ts > 0.3:
            S["phase"] = "comp"; S["t_td"] = t  # stuck extended on the ground: squat again
        u[2] = -P["phi_kp"] * (phi - P["phi_des"]) - P["phi_kd"] * phi_rate - P["k_vank"] * (vx - P["v_des"])  # + ankle = toe up = rocks the hip forward
        if S["phase"] == "comp":
            u[1] = P["comp_kp"] * (sk - lg) - P["knee_kd"] * lgr
        else:
            u[1] = P["thrust"] if lg < -0.03 else 0.1
            u[2] += P["toe_off"]
        u[2] = np.clip(u[2], -P["ank_max"], P["ank_max"])
    else:
        u[0] = -P["fp_kp"] * (pitch - P["pitch_fl"]) - P["fp_kd"] * pr        # torso attitude in flight
        if P["swing_all"] or th > P["thigh_f"]:                                                  # keep the thigh off its upper limit
            u[0] += P["hip_fkp"] * (P["thigh_f"] - th) - P["hip_fkd"] * thr
        u[0] = np.clip(u[0], -P["hip_fmax"], P["hip_fmax"])
        u[1] = P["knee_kp"] * (P["land_knee"] - lg) - P["knee_kd"] * lgr
        u[2] = P["ankle_kp"] * (foot_flat + P["foot_f"] - ft) - P["ankle_kd"] * ftr
    return np.clip(u, -1, 1)
