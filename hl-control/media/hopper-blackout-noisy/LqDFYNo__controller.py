"""Hopper controller: lean-and-thrust hopping with a stale-sensor (blackout) filter.

Control law (phases, clock-driven where possible so that 96 ms sensing blackouts cannot derail it):
  stand   : hip PD holds torso pitch at pitch_ref; ankle PD (gain ankle_phi_kp/kd, limits ankle_min/max) lets
            the body lean forward to the absolute leg angle phi_ref; when the lean passes phi_trigger -> thrust.
  stance  : after touchdown, hold the posture for comp_time (shorter when fast: k_comp), then thrust.
  thrust  : knee extension `thrust` plus toe push `toe_pulse` (both reduced by k_thrust/k_toe when faster than
            v_des); ends when the heel has lifted (foot angle < foot_lift), on contact loss or after thrust_max.
  tuck    : knee flexion `tuck` for tuck_time pulls the foot off the ground.
  flight  : hip = w_pitch * pitch PD (fpitch_kp/kd towards pitch_ref_flight) + (1-w_pitch) * thigh PD towards the
            landing thigh angle thigh_land (+ k_v speed correction), clipped to hip_flight_max; knee folds to
            flight_knee while rising and extends to land_knee while falling; ankle keeps the foot level.
  stop    : (optional, x_stop) land with a bent knee and stand still; disabled with x_stop = 99.
Blackout filter: a blackout holds the noise-free sample, so the per-step change of the six measured rates collapses
to noise; when their summed absolute change is below bo_thresh the reading is treated as stale and the state is
dead-reckoned from the last live sample (constant joint rates, ballistic height, predicted touchdown at z_td).
A reading counts as stale only after bo_min_k consecutive mismatches (an isolated one is noise); from bo_hip_k stale
steps on, the hip command in stance/thrust is scaled by bo_hip_scale (0 = no hip torque on a stale pitch).
Gains were tuned by a (1+lambda) evolution search on the blackout_noisy_v1 plant, seeds 0-8.
"""
import numpy as np

PARAMS = {
    'thrust': 1.27032,
    'toe_pulse': -0.5,
    'thrust_max': 0.16948,
    'foot_lift': -0.42963,
    'tuck': -0.24106,
    'tuck_time': 0.05115,
    'flight_knee': -0.47095,
    'land_knee': -0.05712,
    'comp_time': 0.04143,
    'k_comp': 0.05,
    'comp_max': 0.1,
    'pitch_kp': 1.07063,
    'pitch_kd': 0.21204,
    'pitch_ref': 0.07276,
    'pitch_ref_flight': -0.12181,
    'fpitch_kp': 0.4108,
    'fpitch_kd': 0.27344,
    'hip_flight_max': 0.15334,
    'v_des': 0.68324,
    'k_thrust': 0.51129,
    'k_toe': 0.58306,
    'thigh_land': -0.06786,
    'k_v': -0.00338,
    'thigh_kp': 0.72847,
    'thigh_kd': 0.06695,
    'w_pitch': 0.92139,
    'thigh_max': -0.03,
    'phi_ref': 0.45382,
    'phi_trigger': 0.22146,
    'ankle_phi_kp': 1.09248,
    'ankle_phi_kd': 0.18683,
    'ankle_min': -0.11797,
    'ankle_max': 0.09109,
    'knee_kp': 0.66313,
    'knee_kd': 0.06082,
    'ankle_kp': 0.36563,
    'ankle_kd': 0.02569,
    't0': 0.3,
    'knee_rest': -0.04375,
    'toe_hop': -0.29737,
    'x_stop': 99.0,
    'stop_land_knee': -0.4499,
    'stop_pitch_flight': -0.1596,
    'stop_knee0': -0.47006,
    'stop_knee1': -0.33439,
    'stop_ramp': 0.95324,
    'stop_knee_kp': 1.07505,
    'stop_knee_kd': 0.19183,
    'stop_phi_ref': 0.06117,
    'stop_ankle_min': -0.19525,
    'stop_pitch_kp': 3.52211,
    'stop_pitch_kd': 0.2032,
    'bo_thresh': 0.19526,
    'bo_hip_scale': 0.0,
    'z_td': 1.17,
    'bo_min_k': 3,
    'bo_hip_k': 3,
    'stop_knee': -0.2,
}

G = 9.81


class Estimator:
    """Detect held (stale) sensor samples and dead-reckon through them."""

    def __init__(s, dt=0.008):
        s.dt = dt
        s.reset()

    def reset(s):
        s.prev = None
        s.hold = None
        s.k = 0

    def __call__(s, obs):
        if obs["step"] == 0:
            s.reset()
        rates = np.array([obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"], obs["pitch_rate"], obs["z_velocity"], obs["x_velocity"]])
        angs = np.array([obs["thigh"], obs["leg"], obs["foot"], obs["pitch"], obs["height"]])
        stale = s.prev is not None and float(np.sum(np.abs(rates - s.prev[0]))) < PARAMS["bo_thresh"]
        o = dict(obs)
        if stale:
            if s.hold is None:
                s.hold = (s.prev[0].copy(), s.prev[1].copy(), bool(s.prev[2]))
                s.k = 1
            s.k += 1
        else:
            s.hold = None
            s.k = 0
        if stale and s.k < PARAMS["bo_min_k"]:
            stale = False          # an isolated mismatch is noise, not a blackout: use the raw reading
        if stale:
            r0, a0, c0 = s.hold
            T = s.k * s.dt
            q = a0[:3] + r0[:3] * T
            q[:2] = np.clip(q[:2], -2.6, 0.0)
            q[2] = np.clip(q[2], -0.78, 0.78)
            pitch = a0[3] + r0[3] * T
            if c0:
                z, zd, contact = a0[4], r0[4], True
            else:
                z = a0[4] + r0[4] * T - 0.5 * G * T * T
                zd = r0[4] - G * T
                contact = bool(z < PARAMS["z_td"] and zd < 0)
            o.update(thigh=q[0], leg=q[1], foot=q[2], pitch=pitch, height=z, z_velocity=zd, x_velocity=r0[5],
                     thigh_rate=r0[0], leg_rate=r0[1], foot_rate=r0[2], pitch_rate=r0[3], foot_contact=contact)
        s.prev = (rates, angs, obs["foot_contact"])
        o["stale"] = stale
        return o


class Ctl:
    def __init__(s): s.reset()
    def reset(s): s.phase = "stand"; s.t_ph = 0.0; s.comp = 0.03; s.first = True; s.x_int = 0.0
    def __call__(s, obs):
        if obs["step"] == 0: s.reset()
        p = PARAMS; t = obs["time"]
        q = np.array([obs["thigh"], obs["leg"], obs["foot"]]); qd = np.array([obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]])
        pitch, pr, xd, zd = obs["pitch"], obs["pitch_rate"], obs["x_velocity"], obs["z_velocity"]
        c = bool(obs["foot_contact"])
        s.x_int += xd * 0.008
        a = np.zeros(3)
        if s.phase == "stop":
            a[0] = p["stop_pitch_kp"] * (p["pitch_ref"] - pitch) - p["stop_pitch_kd"] * pr
            kd_ = min(p["stop_knee1"], p["stop_knee0"] + p["stop_ramp"] * (t - s.t_ph))
            a[1] = p["stop_knee_kp"] * (kd_ - q[1]) - p["stop_knee_kd"] * qd[1]
            phi = pitch - q[0]; phid = pr - qd[0]
            a[2] = np.clip(-p["ankle_phi_kp"] * (phi - p["stop_phi_ref"]) - p["ankle_phi_kd"] * phid, p["stop_ankle_min"], p["ankle_max"])
            return np.clip(a, -1, 1)
        phi = pitch - q[0]; phid = pr - qd[0]
        foot_des = np.clip(pitch - q[0] - q[1], -0.7, 0.7)
        hip_stance = p["pitch_kp"] * (p["pitch_ref"] - pitch) - p["pitch_kd"] * pr
        thigh_des = min(p["thigh_land"] + p["k_v"] * (xd - p["v_des"]), p["thigh_max"])
        hip_flight = (1 - p["w_pitch"]) * (p["thigh_kp"] * (thigh_des - q[0]) - p["thigh_kd"] * qd[0]) \
                   + p["w_pitch"] * (p["fpitch_kp"] * (p["pitch_ref_flight"] - pitch) - p["fpitch_kd"] * pr)
        hip_flight = np.clip(hip_flight, -p["hip_flight_max"], p["hip_flight_max"])
        ankle_bal = np.clip(-p["ankle_phi_kp"] * (phi - p["phi_ref"]) - p["ankle_phi_kd"] * phid, p["ankle_min"], p["ankle_max"])
        ankle_level = p["ankle_kp"] * (foot_des - q[2]) - p["ankle_kd"] * qd[2]
        knee_hold = p["knee_kp"] * (p["knee_rest"] - q[1]) - p["knee_kd"] * qd[1]
        if s.phase == "stand":
            a[0] = hip_stance; a[2] = ankle_bal; a[1] = knee_hold
            if t > p["t0"] and phi > p["phi_trigger"]:
                s.phase = "thrust"; s.t_ph = t
        elif s.phase == "stance":
            a[0] = hip_stance; a[2] = ankle_bal; a[1] = knee_hold
            if t - s.t_ph > s.comp:
                s.phase = "thrust"; s.t_ph = t
        if s.phase == "thrust":
            ev = max(0.0, xd - p["v_des"])
            a[0] = hip_stance
            a[1] = np.clip(p["thrust"] - p["k_thrust"] * ev, 0.2, 1.0)
            a[2] = np.clip((p["toe_pulse"] if s.first else p["toe_hop"]) - p["k_toe"] * ev, -1.0, 0.0)
            if q[2] < p["foot_lift"] or t - s.t_ph > p["thrust_max"] or not c:
                s.phase = "tuck"; s.t_ph = t; s.first = False
        if s.phase == "tuck":
            a[0] = hip_flight; a[1] = p["tuck"]; a[2] = ankle_level
            if t - s.t_ph > p["tuck_time"]: s.phase = "flight"; s.t_ph = t
        elif s.phase == "flight":
            a[0] = hip_flight
            stopping = s.x_int >= p["x_stop"]
            knee_des = p["flight_knee"] if zd > 0 else (p["stop_land_knee"] if stopping else p["land_knee"])
            a[1] = p["knee_kp"] * (knee_des - q[1]) - p["knee_kd"] * qd[1]
            if stopping:
                a[0] = np.clip(p["fpitch_kp"] * (p["stop_pitch_flight"] - pitch) - p["fpitch_kd"] * pr, -p["hip_flight_max"], p["hip_flight_max"])
            a[2] = ankle_level
            if c and t - s.t_ph > 0.03:
                s.phase = "stance" if s.x_int < p["x_stop"] else "stop"; s.t_ph = t
                s.comp = float(np.clip(p["comp_time"] + p["k_comp"] * (p["v_des"] - xd), 0.0, p["comp_max"]))
        return np.clip(a, -1, 1)

_core = Ctl()
_est = Estimator()


def controller(obs: dict) -> np.ndarray:
    o = _est(obs)
    a = _core(o)
    if o["stale"] and _est.k >= PARAMS["bo_hip_k"] and _core.phase in ("stance", "thrust", "stand", "stop"):
        a[0] *= PARAMS["bo_hip_scale"]
    return np.clip(a, -1.0, 1.0)
