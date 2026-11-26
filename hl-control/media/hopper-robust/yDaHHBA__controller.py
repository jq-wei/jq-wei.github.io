"""MuJoCo Hopper: heel-roll-toe hopping gait (stance/flight state machine, no learning).

Physics behind the design.  The leg (12 kg) carries almost all of the angular momentum, so
in flight the hip can only re-orient the light torso, never the leg: the leg angle at
landing is set by the leg's rotation at take-off.  A late-stance *toe push* (ankle toe-down,
like a human push-off) both thrusts the body up/forward and applies a nose-up moment to the
leg that cancels its forward rotation, so the foot lands roughly under the body again.

Control law (motor commands in [-1, 1], 1.0 = full gear).  phi1 = pitch - thigh is the
world angle of the thigh behind the vertical; v is a low-pass filtered forward speed and
ev = v - v_des the speed error.
  * Before the first touchdown (the model starts 4 cm in the air): light PD hold at zero.
  * STANCE (foot contact):
      hip   u1 = -Kp*(pitch - th_des) - Dp*pitch_rate      torso attitude loop
      knee  u2 = Kk*(q2_rest - knee) - Dk*knee_rate        stiff spring near full extension
      ankle - heel/roll phase (phi1 < phi_toe - Kphi*ev):  u3 = clip(A_h - Kv*ev, 0, 0.9)
                (toe-up torque rolls the body forward; weaker when too fast)
            - toe push (phi_toe.. <= phi1 < phi_end):       u3 = -A_t, knee gets +thrust
                (fires earlier when too fast, which brakes because the COM is then behind the toe)
            - past phi_end: PD toward a flat foot.
  * FLIGHT:
      hip   u1 = -Kfl*(pitch - th_fl) - Dfl*pitch_rate     keep the torso level
      knee  PD (Kk_fl) toward q2_fl (shortened) while rising, q2_rest while descending
      ankle PD (Ka, Da) toward the flat-landing angle pitch - thigh - knee.
Gains in PARAMS were found with a (1+lambda) evolution strategy across the five grading
scenarios (standard, heavy torso, low friction, weak motors, high gravity) on seeds 0-13;
FIXED holds the hand-set secondary gains.
"""
import numpy as np

PARAMS = {
              "A_h": -0.04383228857684418,
              "A_t": 0.7065741646733191,
              "phi_toe": 0.33819595104377886,
              "phi_end": 1.0705115361964377,
              "Kk": 3.5465959073598237,
              "q2_rest": -0.02824970510147229,
              "thrust": 0.1251447074404528,
              "th_des": -0.10091485673279565,
              "q2_fl": -0.1861493706698163,
              "Kv": 0.6230930516708618,
              "v_des": 0.8351321764620375,
              "Kphi": 0.2087640964094576
}
PARAM_BOUNDS = dict(A_h=(-0.6, 0.9), A_t=(0.0, 1.0), phi_toe=(0.0, 0.5), phi_end=(0.2, 1.2), Kk=(0.5, 4.0),
                    q2_rest=(-0.6, 0.0), thrust=(0.0, 0.8), th_des=(-0.2, 0.2), q2_fl=(-1.4, -0.1),
                    Kv=(0.0, 1.0), v_des=(0.4, 1.6), Kphi=(0.0, 1.5))
FIXED = dict(Dk=0.05, Kp=2.0, Dp=0.3, Kfl=2.0, Dfl=0.2, th_fl=0.0, Kk_fl=1.0, Ka=0.6, Da=0.03, Kz=0.0, vz_ref=1.0)
S = {}


def controller(obs: dict) -> np.ndarray:
    p = {**FIXED, **PARAMS}
    if obs["step"] == 0:
        S.clear(); S.update(landed=False, v=0.0, c_prev=False, vz_land=1.0)
    th, thd = obs["pitch"], obs["pitch_rate"]
    q1, q2, q3 = obs["joint_angles"]; qd1, qd2, qd3 = obs["joint_rates"]
    c = obs["foot_contact"]; vz = obs["z_velocity"]
    S["v"] = 0.9 * S["v"] + 0.1 * obs["x_velocity"]
    ev = S["v"] - p["v_des"]
    if c and not S["c_prev"]:
        S["vz_land"] = abs(min(vz, 0.0))
    S["c_prev"] = c
    if c:
        S["landed"] = True
    if not S["landed"]:
        return np.clip([0.3 * (0 - q1) - 0.02 * qd1, 0.3 * (0 - q2) - 0.02 * qd2, 0.3 * (0 - q3) - 0.02 * qd3], -1, 1)
    flat = np.clip(th - q1 - q2, -0.75, 0.75)
    phi1 = th - q1
    if c:
        u1 = -p["Kp"] * (th - p["th_des"]) - p["Dp"] * thd
        u2 = p["Kk"] * (p["q2_rest"] - q2) - p["Dk"] * qd2
        if phi1 < p["phi_toe"] - p["Kphi"] * ev:
            u3 = np.clip(p["A_h"] - p["Kv"] * ev, 0.0, 0.9)
        elif phi1 < p["phi_end"]:
            u3 = -np.clip(p["A_t"] - p["Kz"] * (S["vz_land"] - p["vz_ref"]), 0.0, 1.0)
            u2 += p["thrust"]
        else:
            u3 = p["Ka"] * (flat - q3) - p["Da"] * qd3
    else:
        u1 = -p["Kfl"] * (th - p["th_fl"]) - p["Dfl"] * thd
        q2t = p["q2_fl"] if vz > 0 else p["q2_rest"]
        u2 = p["Kk_fl"] * (q2t - q2) - p["Dk"] * qd2
        u3 = p["Ka"] * (flat - q3) - p["Da"] * qd3
    return np.clip([u1, u2, u3], -1, 1)
