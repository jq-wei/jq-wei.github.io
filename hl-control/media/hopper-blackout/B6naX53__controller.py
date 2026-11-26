"""Hopper controller: slide-hop gait (quasi-static forward slide over the planted foot, then a hop to reset the foot).

Cycle:
  SETTLE  (t < 0.4 s)   stiff joint PD on the straight stand, pitch trimmed with the ankle.
  SLIDE   ramp the joints toward the crouch (TH_SL, KN_SL, foot flat) at rate SL_RATE; with the foot planted the
          body moves forward over the foot.  Hip holds torso pitch (KH, KHD).  Leaves when the centre of mass is
          E_TRIG ahead of the foot centre (planar kinematics) or after T_SL_MAX.
  THRUST  knee +U_TH and ankle plantarflexion -U_ANK until the foot leaves the ground (or T_TH_MAX).
  FLIGHT  PD toward the landing posture (TH_LAND, KN_LAND, foot flat) with gains KQ_FL: the straightening leg brings
          the foot forward under the body.  Hip torque is small because it pitches the torso in the air.
  LAND    for T_LAND after touchdown: stiff PD on the landing posture, hip pitch loop, ankle balance of the centre
          of mass over the foot centre (KA, KAD; the toe push brakes the body); then SLIDE again.
          (A distance-based stop into a static stand exists but is disabled, X_STOP = 1e9: braking from the
          cruising speed toppled the robot more often than continuing the gait.)
Blackouts (every measured channel repeated exactly, 12 steps = 96 ms): joint angles are dead-reckoned from the
frozen rates over at most BO_HORIZON, pitch is not extrapolated, contact-based transitions and the hop trigger
are suspended (time-based transitions continue), and the hip command is zeroed (BO_HIP in flight, BO_HIP_ST on the
ground): a stale hip torque is what toppled the torso in the blind intervals.
Gains: KQ_ST/KD_ST stance joint PD, KH/KHD hip pitch PD, KH_TH hip pitch gain during the thrust, KA/KAD ankle
balance, KQ_FL flight PD, T_FL_MAX blind-flight timeout, T_SL_MAX slide timeout.  PARAMS were tuned with
hl_control.harbor.tune on blackout_v1 seeds 0-35 (36/36 forward), revalidated on seeds 36-59 (24/24) and
checked on fresh seeds 60-119 (55/60); the nominal plant standard_v0 passes seeds 0-7 as well.
"""
import numpy as np, os

PARAMS = {"TH_SL": -0.1287, "KN_SL": -0.3385, "SL_RATE": 0.4004, "E_TRIG": 0.0365, "U_TH": 0.7084, "T_TH_MAX": 0.0737, "TH_LAND": -0.2361, "KN_LAND": -0.0937, "KQ_FL": 0.8869, "T_LAND": 0.337, "KH": 5.7346, "KA": 1.1595}
PARAM_BOUNDS = {"TH_SL": [-0.253, -0.1084], "KN_SL": [-0.7133, -0.3057], "SL_RATE": [0.2687, 0.6271], "E_TRIG": [0.0358, 0.0834], "U_TH": [0.4979, 1.0], "T_TH_MAX": [0.05, 0.0945], "TH_LAND": [-0.3, -0.1481], "KN_LAND": [-0.1369, -0.0587], "KQ_FL": [0.6648, 1.5], "T_LAND": [0.2009, 0.4689], "KH": [2.6434, 6.168], "KA": [0.6426, 1.4994]}
FIXED = dict(T_SL_MAX=1.5, U_ANK=0.0, KHD=0.4, KAD=0.3, KQ_ST=np.array([3.0, 4.0, 2.0]), KD_ST=0.2, KFOOT=1.0, T_FL_MAX=0.6, BO_HIP=0.0, FLAT_TOL=0.3, A_CLIP=0.3, KN_TOL=0.05, KH_TH=2.0, KN_MIN_TH=-0.1, BO_HORIZON=0.05, T_SL_MIN=0.0, X_STOP=1e9, BO_HIP_ST=0.0, TH_STOP=-0.02, KA_STOP=6.0, KAD_STOP=1.0)
for _k in list(PARAMS) + list(FIXED):  # optional environment overrides for quick experiments
    if _k in os.environ:
        (PARAMS if _k in PARAMS else FIXED)[_k] = float(os.environ[_k])

MASS = np.array([3.665, 4.058, 2.781, 5.316])


def _rot(a, x, z):
    c, s = np.cos(a), np.sin(a)
    return np.array([x * c + z * s, -x * s + z * c])


def com_over_foot(pitch, th, kn, ft):
    a1 = pitch - th; a2 = a1 - kn; a3 = a2 - ft
    hip = _rot(pitch, 0, -0.2); knee = hip + _rot(a1, 0, -0.45); ankle = knee + _rot(a2, 0, -0.5)
    foot = ankle + _rot(a3, 0.065, 0)
    com = (MASS[1] * (hip + _rot(a1, 0, -0.225)) + MASS[2] * (knee + _rot(a2, 0, -0.25)) + MASS[3] * foot) / MASS.sum()
    return com[0] - foot[0]


S = {}


def controller(obs: dict) -> np.ndarray:
    P = {**FIXED, **PARAMS}
    t = obs["time"]; dt = obs["dt"]
    meas = np.concatenate([obs["gym_obs"], obs["foot_xz_rel"], [float(obs["foot_contact"])]])
    if obs["step"] == 0:
        S.clear(); S.update(phase="SETTLE", t0=0.0, last_meas=None, bo=0, tgt=np.zeros(3), stop=False, x_est=0.0)
    blackout = S["last_meas"] is not None and np.array_equal(meas, S["last_meas"])
    S["last_meas"] = meas
    S["bo"] = S["bo"] + 1 if blackout else 0
    dt_bo = min(S["bo"] * dt, P["BO_HORIZON"])  # dead-reckoning horizon
    q = obs["joint_angles"] + obs["joint_rates"] * dt_bo
    qd = obs["joint_rates"] * 0.9 ** S["bo"]
    pitch = obs["pitch"]; prate = 0.0 if blackout else obs["pitch_rate"]   # no pitch extrapolation while blind
    c = bool(obs["foot_contact"]); vx = obs["x_velocity"]
    S["x_est"] = S.get("x_est", 0.0) + vx * dt      # dead-reckoned distance (x itself is not observed)
    e = com_over_foot(pitch, q[0], q[1], q[2])
    tp = t - S["t0"]
    ph = S["phase"]

    def go(new):
        S["phase"] = new; S["t0"] = t

    if ph == "SETTLE" and t >= 0.4:
        go("SLIDE"); S["tgt"] = q.copy()
    elif ph == "SLIDE" and not blackout and tp >= P["T_SL_MIN"] and ((e >= P["E_TRIG"] and q[1] <= P["KN_MIN_TH"]) or tp >= P["T_SL_MAX"]):
        go("THRUST")
    elif ph == "THRUST" and ((not c and not blackout) or tp >= P["T_TH_MAX"]):
        go("FLIGHT")
    elif ph == "FLIGHT" and ((c and not blackout and tp > 0.03) or tp >= P["T_FL_MAX"]):
        go("LAND")
    elif ph == "LAND" and tp >= P["T_LAND"] and not S["stop"]:
        go("SLIDE"); S["tgt"] = q.copy()
    if S["x_est"] >= P["X_STOP"] and ph in ("FLIGHT", "LAND"):
        S["stop"] = True   # far enough: the next landing brakes and holds (blackout-proof static stance)
    ph = S["phase"]; tp = t - S["t0"]

    u = np.zeros(3)
    flat = float(np.clip(pitch - q[0] - q[1], -0.75, 0.75))
    if ph == "SETTLE":
        u = 8.0 * (0 - q) - 0.4 * qd
        if c:
            u[2] += -1.0 * pitch - 0.1 * prate
        else:
            u[0] = 0.0
    elif ph == "SLIDE":
        goal = np.array([P["TH_SL"], P["KN_SL"], 0.0]); goal[2] = float(np.clip(-(goal[0] + goal[1]), -0.75, 0.75))
        step = P["SL_RATE"] * dt
        S["tgt"] = S["tgt"] + np.clip(goal - S["tgt"], -step, step)
        S["tgt"][2] = float(np.clip(-(S["tgt"][0] + S["tgt"][1]), -0.75, 0.75))  # foot flat for the current targets
        u = P["KQ_ST"] * (S["tgt"] - q) - P["KD_ST"] * qd
        u[0] += -P["KH"] * pitch - P["KHD"] * prate
    elif ph == "THRUST":
        u[1] = P["U_TH"]; u[2] = -P["U_ANK"]
        u[0] = -P["KH_TH"] * pitch - P["KHD"] * prate
    elif ph == "FLIGHT":
        tgt = np.array([P["TH_STOP"] if S["stop"] else P["TH_LAND"], P["KN_LAND"], flat])
        u = P["KQ_FL"] * (tgt - q) - 0.1 * qd
        u[0] *= 0.5
    elif ph == "STAND":
        goal = np.array([P["TH_STAND"], P["KN_STAND"], 0.0]); goal[2] = float(np.clip(-(goal[0] + goal[1]), -0.75, 0.75))
        step = 0.5 * dt
        S["tgt"] = S["tgt"] + np.clip(goal - S["tgt"], -step, step)
        u = P["KQ_ST"] * (S["tgt"] - q) - P["KD_ST"] * qd
        u[0] += -P["KH"] * pitch - P["KHD"] * prate
        u[2] += -P["KA_STAND"] * (e - 0.02) - 0.5 * vx
    else:  # LAND
        tgt = np.array([P["TH_STOP"] if S["stop"] else P["TH_LAND"], P["KN_LAND"], 0.0])
        u = P["KQ_ST"] * (tgt - q) - P["KD_ST"] * qd
        u[0] += -P["KH"] * pitch - P["KHD"] * prate
        if S["stop"]:
            u[2] = -P["KA_STOP"] * e - P["KAD_STOP"] * vx
        else:
            u[2] = -P["KA"] * e - P["KAD"] * vx
    if blackout:  # blind: no hip torque in the air, reduced hip torque on the ground
        u[0] *= P["BO_HIP"] if ph == "FLIGHT" else P["BO_HIP_ST"]
    return np.clip(u, -1, 1)
