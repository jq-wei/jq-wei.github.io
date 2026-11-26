"""Walking controller for the MuJoCo Humanoid on stairs / rough ground (SIMBICON + Raibert style).

The humanoid has point (sphere) feet: it cannot stand still and must keep stepping.

State machine: after a short settle (``t_init``, both legs hold a standing posture
while the robot drops onto the ground), one leg is stance and the other swings; they
swap when the swing foot touches down after half a step period ``T`` (or at 1.6 T).

* Torso attitude through the stance hip (SIMBICON): pitch is servoed to ``lean`` (``kt``,
  ``dt_``), roll to a small tilt ``lat_t`` towards the stance foot (``kr``, ``dr``; the tilt
  shifts weight over the stance foot, relieving the 40 N m hip abductors); the stance hip
  also cancels the swing hip's reaction torque.  ``ky``/``dy``: heading term on hip_z.
* Swing foot (Raibert / capture-point placement in a world-aligned frame, relative to the
  CoM):  x_f = x0 + xi + cv*vx + kv*(vx - vdes),  y_f = side*w0 + cvl*vy, converted to
  hip angles with leg length ``H`` and the current torso yaw/pitch/roll.  ``xi`` is a slow
  integral of the speed error (gain ``ki``), so the walking speed settles at ``vdes``.
* Swing knee flexes to ``sw_knee`` (with ``hip_lift`` x that angle of extra hip flexion)
  for the first ``knee_frac`` of the step to clear the 3 cm risers, then extends to
  ``sw_knee_end``; stance knee holds ``st_knee``.
* All joints: PD (``kp_*``, ``kd_*`` in N m/rad, N m s/rad) divided by the gear and
  clipped to the [-0.4, 0.4] command range; small derivative gains avoid chatter at the
  15 ms control period.  Gains were found by a random local search on visible seeds.
"""

import numpy as np

PARAMS = dict(
    yaw_mode=0,
    ki=0.2094,
    t_init=0.3,
    init_hip=0.0,
    T=0.3576,
    vdes=0.25,
    x0=0.0609,
    cv=0.0,
    kv=0.1467,
    w0=0.05,
    cvl=0.2459,
    kvl=0.1,
    H=0.893,
    sw_knee=-1.2779,
    sw_knee_end=-0.3086,
    knee_frac=0.361,
    hip_lift=0.4825,
    st_knee=-0.315,
    lean=0.0126,
    kt=177.7773,
    dt_=6.9123,
    kr=320,
    dr=18.6203,
    ky=7.2195,
    dy=1.5132,
    kp_hip=170.7178,
    kd_hip=5.0,
    kp_knee=200.0,
    kd_knee=3.0,
    kp_hx=177.0376,
    kd_hx=3.0,
    kp_ab=210.6543,
    kd_ab=5.0,
    kp_arm=10.0,
    kd_arm=1.0,
    lat_t=0.0934,
    lat_a=-0.111,
)

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
AY, AZ, AX = 0, 1, 2
LEGS = [(3, 4, 5, 6), (7, 8, 9, 10)]   # (hip_x, hip_z, hip_y, knee) for right, left
SGN = [1.0, -1.0]                      # left hip_x / hip_z axes are mirrored

_state = {"stance": 0, "t0": 0.0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = obs["time"]
    if obs["step"] == 0:
        _state["stance"] = 0
        _state["t0"] = 0.0
        _state["xi"] = 0.0
    q, qd = obs["joint_angles"], obs["joint_rates"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]
    v = obs["com_velocity_xy"]
    feet = [obs["foot_xyz_rel"], obs["foot_left_xyz_rel"]]
    contact = [obs["foot_contact"], obs["foot_left_contact"]]

    st = _state["stance"]
    phase = (t - _state["t0"]) / P["T"]
    if (phase > 0.5 and contact[1 - st]) or phase > 1.6:
        st = _state["stance"] = 1 - st
        _state["t0"] = t
        phase = 0.0
    sw = 1 - st

    tau = np.zeros(17)
    if t < P["t_init"]:
        # settle: both legs hold a standing posture while the robot drops onto the ground
        _state["t0"] = t
        tq = np.zeros(17)
        tq[[6, 10]] = P["st_knee"]
        tq[[5, 9]] = -pitch + P["init_hip"]
        kp = np.array([P["kp_ab"]] * 3 + [P["kp_hx"], P["kp_hx"], P["kp_hip"], P["kp_knee"]] * 2 + [P["kp_arm"]] * 6)
        kd = np.array([P["kd_ab"]] * 3 + [P["kd_hx"], P["kd_hx"], P["kd_hip"], P["kd_knee"]] * 2 + [P["kd_arm"]] * 6)
        return np.clip((kp * (tq - q) - kd * qd) / GEAR, -0.4, 0.4)
    tau[0:3] = P["kp_ab"] * (0 - q[0:3]) - P["kd_ab"] * qd[0:3]
    s_st = 1.0 if st == 1 else -1.0   # y side of the stance foot
    tau[2] = P["kp_ab"] * (s_st * P.get("lat_a", 0.0) - q[2]) - P["kd_ab"] * qd[2]
    roll_d = -s_st * P.get("lat_t", 0.0)
    tau[11:] = P["kp_arm"] * (0 - q[11:]) - P["kd_arm"] * qd[11:]

    # ---- swing foot target, world-aligned frame, relative to the CoM ----
    side = -1.0 if sw == 0 else 1.0
    if t >= P["t_init"]:
        _state["xi"] = float(np.clip(_state["xi"] + P["ki"] * (v[0] - P["vdes"]) * obs["dt"], -0.15, 0.15))
    fx = P["x0"] + _state["xi"] + P["cv"] * v[0] + P["kv"] * (v[0] - P["vdes"])
    fy = side * P["w0"] + P["cvl"] * v[1]
    # rotate into the heading frame
    c, s = np.cos(yaw), np.sin(yaw)
    hx_f = c * fx + s * fy
    hy_f = -s * fx + c * fy
    hip_y_off = side * 0.1
    th = -np.arctan2(hx_f, P["H"])                 # thigh world pitch (negative = forward)
    ph = np.arctan2(hy_f - hip_y_off, P["H"])      # lateral foot angle (+ = towards +y)
    hx, hz, hy, kn = LEGS[sw]
    hy_t = th - pitch + (P["hip_lift"] * P["sw_knee"] if phase < P["knee_frac"] else 0.0)
    hx_t = SGN[sw] * (ph - roll)
    knee_t = P["sw_knee"] if phase < P["knee_frac"] else P["sw_knee_end"]
    hz_t = 0.0
    tau[hy] = P["kp_hip"] * (hy_t - q[hy]) - P["kd_hip"] * qd[hy]
    tau[hx] = P["kp_hx"] * (hx_t - q[hx]) - P["kd_hx"] * qd[hx]
    tau[hz] = P["kp_hx"] * (hz_t - q[hz]) - P["kd_hx"] * qd[hz]
    tau[kn] = P["kp_knee"] * (knee_t - q[kn]) - P["kd_knee"] * qd[kn]

    # ---- stance leg: torso attitude ----
    shx, shz, shy, skn = LEGS[st]
    tau[shy] = P["kt"] * (pitch - P["lean"]) + P["dt_"] * w[1] - tau[hy]
    tau[shx] = SGN[st] * (P["kr"] * (roll - roll_d) + P["dr"] * w[0]) + tau[hx]
    if P["yaw_mode"] > 0.5:
        # point feet cannot react yaw torque: hold the stance hip_z, steer the heading only in double support
        tau[shz] = P["kp_hx"] * (0 - q[shz]) - P["kd_hx"] * qd[shz]
        if contact[0] and contact[1]:
            tau[shz] += SGN[st] * (P["ky"] * yaw + P["dy"] * w[2])
            tau[hz] += SGN[sw] * (P["ky"] * yaw + P["dy"] * w[2])
    else:
        tau[shz] = SGN[st] * (P["ky"] * yaw + P["dy"] * w[2]) + tau[hz]
    tau[skn] = P["kp_knee"] * (P["st_knee"] - q[skn]) - P["kd_knee"] * qd[skn]
    return np.clip(tau / GEAR, -0.4, 0.4)
