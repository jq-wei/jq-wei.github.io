"""Bang-bang hopping controller for the MuJoCo Hopper with ternary motors (bangbang_v1).

Every motor command is quantised to -1 / 0 / +1 (200 N m pulses), so the law is written
directly in pulses: each joint runs a sliding-mode rule ``bang(s, band)`` that fires a full
pulse against the sign of a surface ``s = error + T * error_rate`` whenever ``|s|`` leaves a
deadband.  One 8 ms pulse changes the torso pitch rate by about 1 rad/s and a joint rate by
about 1.2 rad/s, which sets the deadbands.

Gait (Raibert's parts rearranged for this plant, whose leg is three times heavier than the
torso so the leg angle is frozen in flight and foot placement is impossible):

* before the first touchdown nothing is commanded (the robot is dropped from 4 cm);
* stance, hip: holds torso pitch on ``s = (pitch - p_ref) + Tp*pitch_rate + kth*(thigh - th_ref)``;
* stance, knee: regulates the leg lean ``a1 = pitch - thigh`` (positive = foot behind the hip)
  with ``(a1 - a1_ref) + Ta1*a1_rate`` -- folding the knee rotates the leg forward, extending
  sends it back -- inside hard knee bounds ``kn_min``/``kn_max``; after ``t_comp`` seconds of
  stance, if the lean is at least ``a1_thr`` and the speed is at most ``vx_max``, it fires
  ``n_thrust`` extension pulses (the hop and the forward push);
* stance, ankle: balances the torso ``dx_ref`` metres ahead of the foot with ``(dx - dx_ref) + Ta*vx``;
* flight, hip: pitch to ``p_ref_fl``; knee: fold to ``kn_fold`` while rising, ``kn_fl`` while
  falling, but a knee pulse that would push pitch away from target is vetoed, and when the pitch
  error exceeds ``bp_big`` the knee becomes the pitch actuator (the hip has no authority when
  the thigh sits at its 0 rad limit); ankle: keeps the foot level for landing;
* any knee extension pulse is cancelled when the knee would reach ``kn_stop`` within ``Tstop``
  seconds: slamming the extension stop kicks the torso pitch.

PARAMS were found by an evolutionary search on seeds 0-15 (objective: capped distance plus
survival); FIXED holds the hand-set deadbands and time constants.
"""
import numpy as np

PARAMS = dict(
    p_ref=-0.001071,
    th_ref=-0.07428,
    kth=0,
    a1_ref=0.3,
    Ta1=0.077092,
    t_comp=0.13514,
    n_thrust=7.679,
    a1_thr=-0.17263,
    kn_fold=-0.3856,
    kn_fl=-0.22444,
    dx_ref=0.2271,
    vx_max=0.8783,
    p_ref_fl=-0.08,
    bp_big=0.06,
    kn_stop=-0.08,
)
# deadbands (b*), sliding-surface time constants (T*), knee bounds and thrust shape
FIXED = dict(bp=0.03, Tp=0.035, Tstop=0.03, Tp_fl=0.03, bp_fl=0.03, Tk=0.05, Tf=0.05, bf=0.08, thrust_on=1.0,
             Ta=0.4, ba=0.05, bk=0.03, ba1=0.03, ft_off=0.0, ank_thr=0.0, kne_thr=1.0, kn_min=-0.9, kn_max=-0.1,
             dx_thr=-9.0, vx_thr=-9.0)


class _P(dict):
    def __getitem__(self, k):
        return PARAMS[k] if k in PARAMS else FIXED[k]


P = _P()
S = {"t_td": 0.0, "was_contact": False, "n_thr": 0, "landed": False, "thr_now": False}


def bang(s, band):
    """Ternary sliding-mode pulse: push against the surface s once it leaves the deadband."""
    return -1.0 if s > band else (1.0 if s < -band else 0.0)


def controller(obs: dict) -> np.ndarray:
    p, pr = obs["pitch"], obs["pitch_rate"]
    th, kn, ft = obs["joint_angles"]
    thr, knr, ftr = obs["joint_rates"]
    vx, vz, t, contact = obs["x_velocity"], obs["z_velocity"], obs["time"], obs["foot_contact"]
    if obs["step"] == 0:
        S.update(t_td=0.0, was_contact=False, n_thr=0, landed=False)
    if contact and not S["was_contact"]:
        S["t_td"] = t
        S["n_thr"] = 0
        S["landed"] = True
    S["was_contact"] = contact
    u = np.zeros(3)
    S["thr_now"] = False
    if not S["landed"]:
        return u
    if contact:
        ts = t - S["t_td"]
        # hip: torso attitude
        u[0] = bang((p - P["p_ref"]) + P["Tp"] * pr + P["kth"] * (th - P["th_ref"]), P["bp"])
        # knee: leg lean regulation, then the thrust burst
        a1 = p - th
        a1r = pr - thr
        go = (ts >= P["t_comp"] and a1 >= P["a1_thr"] and P["vx_thr"] <= vx <= P["vx_max"]) or S["n_thr"] > 0
        if P["thrust_on"] > 0.5 and go and S["n_thr"] < P["n_thrust"]:
            u[1] = P["kne_thr"]
            S["n_thr"] += 1
            S["thr_now"] = True
        elif S["n_thr"] == 0:
            u[1] = bang((a1 - P["a1_ref"]) + P["Ta1"] * a1r, P["ba1"])   # fold (-1) when the leg is too far back
            if kn < P["kn_min"]:
                u[1] = 1.0
            elif kn > P["kn_max"]:
                u[1] = -1.0
        # ankle: centre-of-pressure balance of the torso over the foot
        dx = -obs["foot_xz_rel"][0]          # torso ahead of the foot centre [m]
        u[2] = P["ank_thr"] if S["thr_now"] else bang((dx - P["dx_ref"]) + P["Ta"] * vx, P["ba"])
    else:
        sp = (p - P["p_ref_fl"]) + P["Tp_fl"] * pr
        u[0] = bang(sp, P["bp_fl"])
        u[1] = bang((kn - (P["kn_fold"] if vz > 0 else P["kn_fl"])) + P["Tk"] * knr, P["bk"])
        # knee pulses pitch the light torso (+1 nose down, -1 nose up): veto pulses that push pitch away
        # from target, and use the knee as the pitch actuator when the error is large (hip may be at its limit)
        if (u[1] < 0 and sp < 0) or (u[1] > 0 and sp > 0):
            u[1] = 0.0
        if sp < -P["bp_big"] and kn < -0.05:
            u[1] = 1.0
        elif sp > P["bp_big"] and kn > -1.2:
            u[1] = -1.0
        ft_des = p - th - kn + P["ft_off"]   # foot level with the ground
        u[2] = bang((ft - ft_des) + P["Tf"] * ftr, P["bf"])
    # never drive the knee into its extension stop: the impact kicks the torso pitch
    if u[1] > 0 and kn + P["Tstop"] * knr > P["kn_stop"]:
        u[1] = 0.0
    return u


PARAM_BOUNDS = dict(
    p_ref=(-0.15, 0.1), vx_max=(0.3, 2.0), th_ref=(-0.6, -0.05), kth=(0.0, 0.6), a1_ref=(-0.2, 0.3), Ta1=(0.02, 0.3),
    t_comp=(0.02, 0.3), n_thrust=(3, 12), a1_thr=(-0.2, 0.3), kn_fold=(-1.0, -0.1), kn_fl=(-0.7, -0.05),
    dx_ref=(-0.05, 0.3), p_ref_fl=(-0.15, 0.1), bp_big=(0.02, 0.15), kn_stop=(-0.3, -0.02),
)
