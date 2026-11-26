"""SIMBICON-style walking controller for the MuJoCo Humanoid (torque motors, point feet).

Control law
-----------
After a short double-support settle (`t_stand` s, `_stand`), a two-state machine
alternates the stance leg every `T` s (earlier on swing-foot touchdown only if
`T_min` < `T`; with the tuned values the switch is purely timed). The tuned gait is a
fast, short-stride shuffle (~9.5 steps/s) at about 0.6-0.7 m/s.

* Swing hip (sagittal y, frontal x): PD (`kph`, `kdh`) to a *world-frame* leg angle
  given by the SIMBICON balance feedback
      phi   = th0 + cd  * d_x + cv  * v_x          (forward placement)
      alpha = side*w0 + cdl * d_y + cvl * v_y     (lateral placement)
  with d = mass centre - stance foot and v = mass-centre velocity; the pelvis
  orientation (torso + abdomen angle) is subtracted to get the joint target.
* Stance hip: the torque that holds the torso at pitch `pitch_d`, roll 0 and yaw 0
  (PD gains `kpt/kdt`, `kpr/kdr`, `kpy/kdy`), minus the swing hip torque, so the
  pelvis only feels the net torso-correction torque.
* Knees (`kpk`, `kdk`): stance knee to `knee_st` (nearly straight); swing knee
  to `knee_sw` for the first `lift_frac` of the swing, then `knee_land`.
* Abdomen (`kpa`, `kda`) and arms: stiff PD to a neutral posture.
Torques are divided by the actual gear (so weaker motors are compensated up to
the [-0.4, 0.4] clamp). Gains were tuned by random search over all five scenarios.
"""

import numpy as np

PARAMS = dict(
    T=0.1051,
    T_min=0.1179,
    th0=0.2411,
    cd=1.1034,
    cv=0.3182,
    w0=0.0075,
    cdl=1.0011,
    cvl=0.2017,
    pitch_d=-0.041,
    t_stand=0.3,
    kpt=350.4335,
    kdt=46.3816,
    kpr=385.9747,
    kdr=23.9447,
    kpy=150.0,
    kdy=15.0,
    kph=346.2916,
    kdh=11.6215,
    knee_st=-0.0912,
    knee_sw=-0.762,
    knee_land=-0.1921,
    lift_frac=1.1672,
    kpk=200.0,
    kdk=5.0,
    kpa=200.0,
    kda=8.0,
)

_S = {"stance": 0, "t": 0.0}

# motor indices
AY, AZ, AX = 0, 1, 2
HIP = {0: (3, 4, 5, 6), 1: (7, 8, 9, 10)}   # (hip_x, hip_z, hip_y, knee) for right(0)/left(1)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    dt = obs["dt"]
    if obs["step"] == 0:
        _S["stance"] = 0
        _S["t"] = 0.0
    _S["t"] += dt
    st = _S["stance"]
    sw = 1 - st
    contact = (obs["foot_contact"], obs["foot_left_contact"])
    if obs["time"] < P["t_stand"]:
        return _stand(obs)
    if (_S["t"] > P["T_min"] and contact[sw]) or _S["t"] > P["T"]:
        _S["stance"], st, sw = sw, sw, st
        _S["t"] = 0.0
    s = min(_S["t"] / P["T"], 1.0)

    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    w = obs["root_angular_velocity"]
    vel = obs["com_velocity_xy"]
    feet = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])
    d = -feet[st][:2]                     # mass centre relative to stance foot
    tau = np.zeros(17)

    def pd(i, target, kp, kd):
        return kp * (target - q[i]) - kd * qd[i]

    # abdomen and arms
    tau[AY] = pd(AY, 0.0, P["kpa"], P["kda"])
    tau[AZ] = pd(AZ, 0.0, P["kpa"], P["kda"])
    tau[AX] = pd(AX, 0.0, P["kpa"], P["kda"])
    for i in range(11, 17):
        tau[i] = pd(i, 0.0, 20.0, 1.0)

    # swing leg targets (world-frame leg angles)
    side = 1.0 if sw == 1 else -1.0          # left swing goes to +y
    phi = P["th0"] + P["cd"] * d[0] + P["cv"] * vel[0]
    alpha = side * P["w0"] + P["cdl"] * d[1] + P["cvl"] * vel[1]
    hx, hz, hy, kn = HIP[sw]
    ppitch = pitch + q[AY]                   # pelvis orientation (torso + abdomen)
    proll = roll + q[AX]
    hy_d = -phi - ppitch
    hx_d = (alpha - proll) if sw == 0 else (proll - alpha)
    tau[hy] = pd(hy, hy_d, P["kph"], P["kdh"])
    tau[hx] = pd(hx, hx_d, P["kph"], P["kdh"])
    tau[hz] = pd(hz, 0.0, 100.0, 10.0)
    tau[kn] = pd(kn, P["knee_sw"] if s < P["lift_frac"] else P["knee_land"], P["kpk"], P["kdk"])

    # stance leg: torso control
    T_pitch = -P["kpt"] * (pitch - P["pitch_d"]) - P["kdt"] * w[1]
    T_roll = -P["kpr"] * roll - P["kdr"] * w[0]
    T_yaw = -P["kpy"] * yaw - P["kdy"] * w[2]
    sx, sz, sy, sk = HIP[st]
    tau[sy] = -T_pitch - tau[hy]
    if st == 0:
        tau[sx] = tau[hx] - T_roll
        tau[sz] = tau[hz] - T_yaw
    else:
        tau[sx] = T_roll + tau[hx]
        tau[sz] = T_yaw + tau[hz]
    tau[sk] = pd(sk, P["knee_st"], P["kpk"], P["kdk"])

    return np.clip(tau / obs["gear"], -0.4, 0.4)


def _stand(obs):
    """Initial double support: both legs share the torso torque, knees nearly straight."""
    P = PARAMS
    q, qd, w = obs["joint_angles"], obs["joint_rates"], obs["root_angular_velocity"]
    tau = np.zeros(17)
    for i in (AY, AZ, AX):
        tau[i] = P["kpa"] * (0.0 - q[i]) - P["kda"] * qd[i]
    for i in range(11, 17):
        tau[i] = 20.0 * (0.0 - q[i]) - 1.0 * qd[i]
    T_pitch = -P["kpt"] * (obs["pitch"] - 0.0) - P["kdt"] * w[1]
    T_roll = -P["kpr"] * obs["roll"] - P["kdr"] * w[0]
    for leg in (0, 1):
        hx, hz, hy, kn = HIP[leg]
        tau[hy] = -0.5 * T_pitch
        tau[hx] = (-0.5 * T_roll if leg == 0 else 0.5 * T_roll) + 100.0 * (0.0 - q[hx])
        tau[hz] = 100.0 * (0.0 - q[hz]) - 10.0 * qd[hz]
        tau[kn] = P["kpk"] * (P["knee_st"] - q[kn]) - P["kdk"] * qd[kn]
    return np.clip(tau / obs["gear"], -0.4, 0.4)
