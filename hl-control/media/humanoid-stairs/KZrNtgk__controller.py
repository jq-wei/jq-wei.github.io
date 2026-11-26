"""SIMBICON-style walking controller for the MuJoCo Humanoid (point feet, torque motors).

Control law
-----------
A two-state machine alternates the stance leg: switch when the swing foot touches down after
``t_min`` (or once the swing has lasted ``t_abort``). Every torque is a PD law divided by the
gear and clipped to the command range. Derivative gains stay below I_free/dt (15 ms control
period), otherwise the light limbs chatter bang-bang.

All horizontal quantities (foot positions relative to the mass centre, mass-centre velocity,
roll/pitch rates) are rotated into the torso's heading frame. A yawed body then still steps
within the hip joint ranges.

* Torso: the stance hip holds torso pitch at ``pitch_d`` and roll at 0 (``kp_tp, kd_tp,
  kp_tr, kd_tr``). The swing hip torque is subtracted so the pelvis sees only the torso torque
  (SIMBICON).
* Swing foot (Cartesian, relative to the mass centre): target
  ``x_t = cv*vx + ck*(vx - v_d) + x0`` (capture point, speed regulation towards world +x,
  offset ``x0``, capped at ``x_max``) and ``y_t = side*w0 + cvy*vy``. The swing hip y/x torques
  are ``kc_x, kc_y`` times the foot error over the leg length, damped by ``kd_sw, kd_swx`` and
  clamped to ``sw_max, swx_max`` so the stance hip keeps authority over the torso.
* Knees: the stance knee is held near straight (``knee_st``; ``kp_ks, kd_ks``). The swing knee
  folds to ``knee_lift`` for the first ``lift_frac*t_max`` of the swing, for riser clearance,
  then extends to ``knee_land`` (``kp_k, kd_k``).
* Yaw: a point foot cannot resist torsion, so the stance hip z joint is only held relative to
  the pelvis (``kp_shz, kd_shz``). World-yaw feedback there just winds the leg up and twists
  the hip axes. Swing hip z is held at 0 (``kp_hz, kd_hz``). Abdomen (``kp_ab, kd_ab``) and arms
  are held stiff at zero.

Parameters not mentioned above (``kp_j``, ``jmode``, ``t_hold``, ``td_tol``, ``ff_hip``, ...)
belong to variants that were tried and switched off; at their defaults they have no effect.
"""

import numpy as np

PARAMS = dict(
    v_d=0.35,
    pitch_d=-0.0145,
    kp_tp=235.1434,
    kd_tp=15.0,
    kp_tr=185.7488,
    kd_tr=11.524,
    cv=0.25,
    ck=0.15,
    x0=0.0546,
    w0=0.1155,
    cvy=0.2294,
    kc_x=114.0368,
    kd_sw=3.1299,
    kc_y=300.0,
    kd_swx=8.0,
    knee_st=-0.1,
    knee_lift=-1.1,
    knee_land=-0.2941,
    lift_frac=0.5,
    kp_k=100.0,
    kd_k=2.0,
    kp_ks=168.2673,
    kd_ks=3.0,
    kp_hz=30.0,
    kd_hz=0.5,
    kp_ab=150.0,
    kd_ab=3.0,
    t_min=0.1,
    t_max=0.2,
    t_abort=0.7,
    kp_yaw=0,
    kd_yaw=0,
    t_ds=0.06,
    k_arm=0.0,
    k_arm_y=0.0,
    kp_sty=0.0,
    kd_sty=0.0,
    x_max=0.35,
    sw_max=61.7659,
    swx_max=21.2961,
    ff_hip=5.1569,
    ff_t=0.1,
    t_hold=0.0,
    ext_tol=1.0,
    kp_shz=30.0,
    kd_shz=0.0006,
    jmode=0.0,
    kp_j=385.0397,
    kd_j=6.0,
    kp_jx=363.6106,
    kd_jx=8.0,
    hip_w=0.1,
    knee_drop=0.0,
    cky=0.0,
    steer=1.0,
    td_tol=9.0,
    scuff_t=0.08,
    scuff_k=0.3,
)

PARAM_BOUNDS = dict(
    v_d=(0.0, 0.8), pitch_d=(-0.1, 0.2), kp_tp=(50, 600), kd_tp=(2, 40), kp_tr=(50, 600), kd_tr=(2, 40),
    cv=(0.0, 0.6), ck=(-0.3, 0.3), x0=(-0.1, 0.15), w0=(0.0, 0.15), cvy=(0.0, 0.6),
    kc_x=(50, 400), kd_sw=(1, 8), kc_y=(50, 500), kd_swx=(1, 15),
    knee_st=(-0.5, -0.04), knee_lift=(-1.5, -0.3), knee_land=(-0.6, -0.04), lift_frac=(0.2, 0.8),
    kp_k=(30, 250), kd_k=(0.5, 4), kp_ks=(100, 400), kd_ks=(1, 4), t_min=(0.08, 0.3), t_max=(0.2, 0.5),
    kp_sty=(0, 300), kd_sty=(0, 4), kp_yaw=(0, 300), kd_yaw=(0, 15), t_ds=(0, 0.1), x_max=(0.1, 0.5), knee_drop=(0.0, 1.5), sw_max=(20, 120), swx_max=(5, 40), ff_hip=(0, 60), ff_t=(0.02, 0.2), t_hold=(0.0, 0.12), ext_tol=(-0.05, 0.3), k_arm=(0, 10), k_arm_y=(0, 20), cky=(0, 0.4), td_tol=(0.0, 0.3), scuff_t=(0.0, 0.15), scuff_k=(0.0, 0.6), kp_shz=(0, 100), kd_shz=(0, 1.5), kp_j=(50, 450), kd_j=(1, 10), kp_jx=(50, 600), kd_jx=(1, 15), hip_w=(0.0, 0.15),
)

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
NAMES = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
         "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
         "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
IDX = {n: i for i, n in enumerate(NAMES)}
LEG = (("right_hip_x", "right_hip_z", "right_hip_y", "right_knee"), ("left_hip_x", "left_hip_z", "left_hip_y", "left_knee"))
SGN = (1.0, -1.0)    # hip_x / hip_z joint sign relative to the world axes (left leg axes are mirrored)
LEG_L = 0.8
_S = {"stance": 0, "t": 0.0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    if obs["step"] == 0:
        _S["stance"], _S["t"], _S["reached"] = 0, 0.0, False   # 0: right stance, 1: left stance
    q, qd = obs["joint_angles"], obs["joint_rates"]
    tau = np.zeros(17)
    for n in ("abdomen_y", "abdomen_z", "abdomen_x"):
        tau[IDX[n]] = P["kp_ab"] * (0 - q[IDX[n]]) - P["kd_ab"] * qd[IDX[n]]
    tau[11:] = 10.0 * (0 - q[11:]) - 1.0 * qd[11:]
    arm_yaw = P["k_arm"] * obs["root_angular_velocity"][2] + P["k_arm_y"] * obs["yaw"]
    tau[12] += arm_yaw; tau[15] += arm_yaw                                 # equal shoulder2 torques: pure yaw on the torso

    _S["t"] += obs["dt"]
    st = _S["stance"]
    contacts = (obs["foot_contact"], obs["foot_left_contact"])
    if (_S["t"] > P["t_hold"] + P["t_min"] and contacts[1 - st] and _S.get("reached", True)
            and (_S.get("ex", 0.0) < P["td_tol"] or _S["t"] > P["t_max"])) or _S["t"] > P["t_abort"]:
        _S["stance"], _S["t"], _S["reached"], _S["scuff"] = 1 - st, 0.0, False, -1.0
        st = 1 - st
    sw = 1 - st
    yaw = obs["yaw"]
    c, s_ = np.cos(yaw), np.sin(yaw)
    rot = lambda v: np.array([c * v[0] + s_ * v[1], -s_ * v[0] + c * v[1]])   # world -> heading frame
    feet = (rot(obs["foot_xyz_rel"]), rot(obs["foot_left_xyz_rel"]))
    roll, pitch = obs["roll"], obs["pitch"]
    w = obs["root_angular_velocity"]
    wr, wp = rot(w[:2])                                                     # roll and pitch rates in the heading frame
    vx, vy = rot(obs["com_velocity_xy"])

    # ---- swing leg: Cartesian foot placement relative to the mass centre
    hx, hz, hy, kn = (IDX[n] for n in LEG[sw])
    vdx, vdy = rot((P["v_d"], 0.0)) if P["steer"] > 0.5 else (P["v_d"], 0.0)                                         # desired velocity (world +x) in the heading frame
    x_t = min(P["cv"] * vx + P["ck"] * (vx - vdx) + P["x0"], P["x_max"])
    y_t = (-1.0 if sw == 0 else 1.0) * P["w0"] + P["cvy"] * vy + P["cky"] * (vy - vdy)
    ex, ey = x_t - feet[sw][0], y_t - feet[sw][1]
    _S["ex"] = ex
    if contacts[sw] and _S["t"] > P["t_min"] and ex >= P["td_tol"]:          # scuff on a riser: fold the knee again
        _S["scuff"] = _S["t"]
    tau_sw_y = -P["kc_x"] * ex / LEG_L - P["kd_sw"] * qd[hy]                 # +hip_y swings the foot backwards
    if P["jmode"] > 0.5:                                                      # joint mode: aim the thigh in the world frame
        phi = pitch + q[IDX["abdomen_y"]] + q[hy]                             # thigh pitch (+ = foot behind)
        phi_d = -np.arcsin(np.clip(x_t / LEG_L, -0.7, 0.7))
        tau_sw_y = P["kp_j"] * (phi_d - phi) - P["kd_j"] * qd[hy]
        lat_hip = (-1.0 if sw == 0 else 1.0) * P["hip_w"]
        psi = roll + q[IDX["abdomen_x"]] + SGN[sw] * q[hx]                   # thigh roll (+ = foot to +y)
        psi_d = np.arcsin(np.clip((y_t - lat_hip) / LEG_L, -0.4, 0.4))
        tau_sw_x_j = P["kp_jx"] * (psi_d - psi) - P["kd_jx"] * SGN[sw] * qd[hx]
    if _S["t"] < P["ff_t"]:
        tau_sw_y -= P["ff_hip"]                                                 # feedforward kick: start the swing forward
    tau_sw_y = float(np.clip(tau_sw_y, -P["sw_max"], P["sw_max"]))               # leave the stance hip room for the torso
    tau_sw_x = P["kc_y"] * ey / LEG_L - P["kd_swx"] * SGN[sw] * qd[hx]       # world sign: + moves the foot to +y
    if P["jmode"] > 0.5:
        tau_sw_x = tau_sw_x_j
    tau_sw_x = float(np.clip(tau_sw_x, -P["swx_max"], P["swx_max"]))
    tau_sw_z = SGN[sw] * (P["kp_hz"] * (0 - q[hz]) - P["kd_hz"] * qd[hz])
    tau[hy], tau[hx], tau[hz] = tau_sw_y, SGN[sw] * tau_sw_x, SGN[sw] * tau_sw_z
    late = max(0.0, _S["t"] - P["t_hold"] - P["t_max"])                                   # swing overdue: reach for the ground
    reached = ex < P["ext_tol"]                                              # foot has (nearly) reached its forward target
    _S["reached"] = reached or late > 0
    lifting = _S["t"] < P["t_hold"] + P["lift_frac"] * P["t_max"] or not _S["reached"]
    kt = P["knee_lift"] if lifting else (P["knee_land"] if late == 0 else -0.05)
    if _S["t"] - _S.get("scuff", -1.0) < P["scuff_t"]:
        kt = P["knee_lift"] - P["scuff_k"]
    tau[kn] = P["kp_k"] * (kt - q[kn]) - P["kd_k"] * qd[kn]
    yaw_t = P["kp_yaw"] * obs["yaw"] + P["kd_yaw"] * w[2]                   # world-sign hip_z torque that turns the pelvis back
    if _S["t"] < P["t_hold"]:                                                # double support: trailing foot stays planted
        tau_sw_y = 0.5 * (P["kp_tp"] * (pitch - P["pitch_d"]) + P["kd_tp"] * wp)
        tau_sw_x = 0.0
        tau[hy], tau[hx] = tau_sw_y, 0.0
        tau[kn] = P["kp_ks"] * (P["knee_st"] - q[kn]) - P["kd_ks"] * qd[kn]
    if _S["t"] < P["t_ds"] and contacts[sw]:                                 # both feet down: steer the yaw with both hips
        tau_sw_z = 0.0
        tau[hz] = SGN[sw] * yaw_t * 0.5

    # ---- stance leg: torso attitude in the world frame
    hx, hz, hy, kn = (IDX[n] for n in LEG[st])
    tau[hy] = P["kp_tp"] * (pitch - P["pitch_d"]) + P["kd_tp"] * wp - tau_sw_y
    tau[hx] = SGN[st] * (P["kp_tr"] * roll + P["kd_tr"] * wr - tau_sw_x)
    tau[hz] = SGN[st] * (P["kp_sty"] * obs["yaw"] + P["kd_sty"] * w[2] - tau_sw_z) + P["kp_shz"] * (0 - q[hz]) - P["kd_shz"] * qd[hz]
    if _S["t"] < P["t_ds"] and contacts[sw]:
        tau[hz] = SGN[st] * yaw_t * 0.5
    tau[kn] = P["kp_ks"] * (P["knee_st"] - P["knee_drop"] * late / 0.2 - q[kn]) - P["kd_ks"] * qd[kn]
    return np.clip(tau / GEAR, -0.4, 0.4)
