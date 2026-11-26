"""SIMBICON-style finite-state walking controller for Walker2d (hand tuned).

Kinematic facts used (checked in simulation): hip and knee ranges are
[-150 deg, 0], positive pitch is nose down, and the thigh's world angle
(positive = forward) is ``hip - pitch``.  The legs can therefore only be
ahead of the torso axis when the torso leans back, so the gait holds the
torso at a nose-up pitch ``pitch_d`` with a *soft* PD (the torso is light,
3.7 kg, so a stiff torso loop chatters and converts swing-leg reaction
torque into body acceleration).

Two mirrored states (right stance / left stance) alternate; a state ends
after ``T`` seconds, or when the swing foot touches down after having
lifted (not before 30 % of ``T``).  Per state:
  * swing hip: PD (``kp_h``, ``kd_h``, bounded by ``hip_max``) to the world
    thigh angle ``sw_th0 + c_d*d + c_v*(v - v_d)``, where d is how far the
    torso is ahead of the stance ankle and v the forward speed; this is
    the foot-placement balance feedback and sets the walking speed.
  * swing knee: smooth flexion bump from ``knee_st`` to ``knee_sw`` over the
    step (ground clearance).
  * swing ankle: holds the foot at ``ankle_sw`` rad toe-up in the world.
  * stance hip: torso-pitch PD torque (``kp_t``, ``kd_t``) minus the swing
    hip torque, so the swing leg's reaction does not disturb the torso.
  * stance knee: PD to ``knee_st`` (slightly bent strut).
  * stance ankle: PD to a foot angle ``ankle_st`` in the world, plus
    ``k_va`` times the speed error (toe-up brakes) and, in the last 30 % of
    the step, a constant ``-push`` added to the command (so a negative
    ``push`` is a toe-up braking torque, a positive one a toe-down push-off).
Gains are in motor command units per rad and per rad/s (1.0 = 100 N m).
Parameters were found by random search over seeds 0-5 of the nominal and
the four perturbed plants; the module-level state resets at step 0.
"""
import numpy as np

PARAMS = dict(
    T=0.409,                 # nominal step duration [s]
    v_d=0.526,               # reference forward speed [m/s] for the speed feedback terms
    sw_th0=0.391,            # swing thigh world angle at the reference speed (positive = forward)
    c_d=0.0,                 # foot placement feedback from torso position ahead of the stance ankle
    c_v=0.137,               # foot placement feedback from the speed error
    knee_sw=-1.2,            # swing knee flexion bump depth [rad]
    knee_st=-0.43,           # stance knee set point [rad] (also the swing knee at lift-off/touchdown)
    ankle_sw=0.3,            # swing foot toe-up angle in the world [rad]
    ankle_st=0.102,          # stance foot world angle set point [rad]
    push=-0.2,               # late-stance ankle offset; negative = toe-up braking (see docstring)
    k_va=0.0,                # stance ankle command per m/s of speed error (toe up brakes)
    pitch_d=-0.433,          # desired torso pitch [rad] (negative = nose up)
    kp_t=1.159, kd_t=0.205,  # torso pitch PD
    kp_h=2.498, kd_h=0.212,  # swing hip PD
    hip_max=1.0,             # swing hip command bound
    kp_k=1.198, kd_k=0.03,   # knee PD
    kp_a=0.585, kd_a=0.071,  # ankle PD
)

_state = dict(phase=0, t0=0.0, lifted=False)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = obs["time"]
    if obs["step"] == 0:
        _state["phase"], _state["t0"], _state["lifted"] = 0, 0.0, False
    q = obs["joint_angles"]; qd = obs["joint_rates"]
    p, pd_ = obs["pitch"], obs["pitch_rate"]
    v = obs["x_velocity"]
    contacts = (obs["foot_contact"], obs["foot_left_contact"])
    feet = (obs["foot_xz_rel"], obs["foot_left_xz_rel"])

    st = _state["phase"]          # stance leg index (0 right, 1 left)
    sw = 1 - st
    el = t - _state["t0"]
    if not contacts[sw]:
        _state["lifted"] = True
    if el >= P["T"] or (el > 0.3 * P["T"] and _state["lifted"] and contacts[sw]):
        _state["phase"], _state["t0"], _state["lifted"] = sw, t, False
        st, sw = sw, st
        el = 0.0
    s = min(el / P["T"], 1.0)

    i_st, i_sw = 3 * st, 3 * sw
    u = np.zeros(6)

    # ----- swing leg -----
    d = -feet[st][0] + 0.1                 # torso ahead of stance ankle
    th_w = P["sw_th0"] + P["c_d"] * d + P["c_v"] * (v - P["v_d"])
    h_d = np.clip(th_w + p, -2.3, -0.02)   # joint = world + pitch
    u_sw_hip = np.clip(P["kp_h"] * (h_d - q[i_sw]) - P["kd_h"] * qd[i_sw], -P["hip_max"], P["hip_max"])
    k_d = P["knee_st"] + (P["knee_sw"] - P["knee_st"]) * np.sin(np.pi * s)
    a_sw = np.clip(p - q[i_sw] - q[i_sw + 1] + P["ankle_sw"], -0.7, 0.7)
    u[i_sw] = u_sw_hip
    u[i_sw + 1] = P["kp_k"] * (k_d - q[i_sw + 1]) - P["kd_k"] * qd[i_sw + 1]
    u[i_sw + 2] = P["kp_a"] * (a_sw - q[i_sw + 2]) - P["kd_a"] * qd[i_sw + 2]

    # ----- stance leg -----
    u_torso = P["kp_t"] * (P["pitch_d"] - p) - P["kd_t"] * pd_
    u[i_st] = u_torso - u_sw_hip
    u[i_st + 1] = P["kp_k"] * (P["knee_st"] - q[i_st + 1]) - P["kd_k"] * qd[i_st + 1]
    a_st = np.clip(p - q[i_st] - q[i_st + 1] + P["ankle_st"], -0.7, 0.7)
    a_cmd = P["kp_a"] * (a_st - q[i_st + 2]) - P["kd_a"] * qd[i_st + 2]
    a_cmd += P["k_va"] * (v - P["v_d"])
    if s > 0.7:
        a_cmd -= P["push"]
    u[i_st + 2] = a_cmd
    return np.clip(u, -1.0, 1.0)
