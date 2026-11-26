"""Hand-designed SIMBICON-style stepping controller for the MuJoCo Humanoid (no learning).

Control law
-----------
* Two-state machine alternating the stance leg: the swing ends when the swing foot touches
  down after ``t_min_frac*T`` s, or at ``t_max_frac*T`` s at the latest.  The tuned gaits
  take short, quick steps (T ~ 0.1-0.2 s), which keeps the 40 N m hip_x motors out of
  long single-support phases (sphere feet: no ankle torque, no torsional friction).
* Every joint is a PD servo, torque = kp (q_d - q) - kd qdot + passive-spring compensation,
  converted to commands with the *observed* gears (so weakened motors are compensated).
* Torso pitch/roll are regulated through the stance hip (torque = desired torso torque minus
  the swing-hip torque, SIMBICON's virtual ankle); stance hip_z is a joint PD, and when both
  feet are down both hip_z motors also correct the heading (``ky_c``), since only the
  two-foot friction couple can resist yaw.
* Swing foot servo (balance): the swing foot, measured relative to the mass centre in the
  heading frame, is driven to the capture-point-like target
  ``x_d = x0 + cv*v_x + kv*(v_x - v_d)``, ``y_d = +-w0 + cvl*v_y`` through the hip angles
  (gains ``kpos``, ``kpos_y``); the knee flexes for ``knee_frac`` of the swing for clearance.
  ``v_d`` ramps up over ``ramp`` s from standstill, after ``t_settle`` s of two-leg stance.
* Three gain sets: ``PARAMS`` (nominal), ``PARAMS_DELAY`` (selected once the controller
  sees that the applied command, ``obs["last_action"]``, lags the one it sent: one-step
  delay; it also extrapolates the measured state by ``pred_d`` steps), and ``PARAMS_PUSH``
  (selected once a push is felt as a non-zero external force on the torso in the
  ``cfrc_ext`` block of ``gym_obs``; ``kff`` adds the push's expected velocity change to the
  placement feedback while it acts).

Gains (PARAMS)
--------------
pitch_d, kp_t, kd_t: torso pitch set-point and PD; kp_r, kd_r, roll_st: torso roll PD;
kp_y, kd_y, ky_c, ky_ss: heading PD and its double/single-support weights; kpz, kp_hz, kd_hz,
ysw: hip_z servos; x0, cv, kv, v_d, w0, cvl, kpos, kpos_y: foot placement; comp: share of the
swing hip_x torque reacted by the stance hip; sw_x_max: swing hip_x torque cap; sw_knee,
knee_frac, st_knee, kp_k, kd_k, kp_sk, kd_sk: knees; kp_h, kd_h: swing hip servo; kp_ab, kd_ab,
kp_abz, kd_abz, abx_st: abdomen; kp_arm, kd_arm: arms; T, t_min_frac, t_max_frac: step timing;
yaw_frac: fraction of the yaw used for the heading frame.  Unused switches: t_ds (timed double
support), tol/lift_tol/t_max_far (placement-gated touchdown), zs_d/kds_d (delay gain scaling).
The numeric values were found by random search on the sandbox simulator.
"""

import numpy as np

PARAMS = dict(
    pitch_d=0.0524,
    kp_t=205.9932,
    kd_t=10.5575,
    kp_r=57.6814,
    kd_r=15.9625,
    kp_y=100.0,
    kd_y=10.0,
    x0=0.0215,
    cv=0.2241,
    w0=0.1592,
    cvl=0.6879,
    kpos=0.9549,
    kpos_y=1.4003,
    comp=0.5811,
    roll_st=0.0082,
    sw_knee=-0.8322,
    st_knee=-0.1472,
    knee_frac=0.3079,
    kp_h=259.05,
    kd_h=5.8483,
    kp_k=91.8285,
    kd_k=4.2498,
    T=0.1881,
    t_min_frac=0.4685,
    t_max_frac=1.3447,
    v_d=0.35,
    kv=0.2089,
    yaw_frac=1.1034,
    ysw=-0.314,
    kpz=59.604,
    kp_ab=192.7699,
    kd_ab=6.9051,
    kp_abz=36.9605,
    kd_abz=1.2102,
    kp_hz=117.2164,
    kd_hz=1.2274,
    t_ds=0.0,
    ky_ss=0.0817,
    abx_st=0.0,
    sw_x_max=42.4648,
    tol=10.0,
    lift_tol=10.0,
    t_max_far=0.35,
    kp_sk=300.0,
    kd_sk=6.0,
    pred_d=0.7964,
    zs_d=1.0,
    kp_arm=10.0,
    kd_arm=0.5,
    kds_d=1.0,
    kff=0.0,
    ky_c=1.0,
    ramp=1.5,
    first=0,
    t_settle=0.2,
)

# gain set used once a push has been felt (tuned on the push plant)
PARAMS_PUSH = dict(PARAMS)
PARAMS_PUSH.update(
    pitch_d=0.0168,
    kp_t=161.1888,
    kd_t=14.3802,
    kp_r=63.1159,
    kd_r=33.1886,
    kp_y=100.0,
    kd_y=10.0,
    x0=0.0225,
    cv=0.2915,
    w0=0.1592,
    cvl=0.3025,
    kpos=0.6039,
    kpos_y=1.6556,
    comp=0.7891,
    roll_st=0.0106,
    sw_knee=-0.9018,
    st_knee=-0.2375,
    knee_frac=0.6939,
    kp_h=253.5817,
    kd_h=5.1695,
    kp_k=173.4136,
    kd_k=2.8147,
    T=0.1389,
    t_min_frac=0.3495,
    t_max_frac=0.7906,
    v_d=0.3,
    kv=0.1335,
    yaw_frac=0.9681,
    ysw=-0.1469,
    kpz=74.0734,
    kp_ab=223.7075,
    kd_ab=13.0062,
    kp_abz=31.871,
    kd_abz=1.1957,
    kp_hz=110.6106,
    kd_hz=1.6142,
    t_ds=0.0,
    ky_ss=0.113,
    abx_st=0.0,
    sw_x_max=37.6478,
    tol=10.0,
    lift_tol=10.0,
    t_max_far=0.35,
    kp_sk=300.0,
    kd_sk=3.6175,
    pred_d=0.7964,
    zs_d=1.0,
    kp_arm=9.6885,
    kd_arm=0.4195,
    kds_d=1.0,
    kff=1.5,
    ky_c=0.5964,
)

# gain set used once a command delay has been detected (tuned on the delayed plant)
PARAMS_DELAY = dict(PARAMS)
PARAMS_DELAY.update(
    t_settle=0.25,
    pitch_d=0.0449,
    kp_t=248.8194,
    kd_t=19.926,
    kp_r=88.2974,
    kd_r=10.9332,
    kp_y=100.0,
    kd_y=10.0,
    x0=0.0162,
    cv=0.1749,
    w0=0.4021,
    cvl=0.706,
    kpos=1.2517,
    kpos_y=0.6685,
    comp=0.5811,
    roll_st=0.0123,
    sw_knee=-0.8439,
    st_knee=-0.1363,
    knee_frac=0.3307,
    kp_h=259.05,
    kd_h=5.7764,
    kp_k=91.8285,
    kd_k=4.8787,
    T=0.1347,
    t_min_frac=0.5928,
    t_max_frac=0.6473,
    v_d=0.3,
    kv=0.2089,
    yaw_frac=1.0466,
    ysw=-0.298,
    kpz=56.9754,
    kp_ab=180.8219,
    kd_ab=2.5059,
    kp_abz=29.7423,
    kd_abz=1.2102,
    kp_hz=117.2164,
    kd_hz=1.1835,
    t_ds=0.0,
    ky_ss=0.0666,
    abx_st=0.0,
    sw_x_max=41.8298,
    tol=10.0,
    kp_sk=104.1175,
    kd_sk=6.0,
    pred_d=0.7964,
    zs_d=0.7214,
    kp_arm=10.0,
    kd_arm=0.4689,
    kds_d=1.0892,
)

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
STIFF = np.array([10, 20, 10, 10, 10, 20, 0, 10, 10, 20, 1, 1, 1, 0, 1, 1, 0], float)
# indices: hip_x, hip_z, hip_y, knee for right (0) and left (1)
LEG = [(3, 4, 5, 6), (7, 8, 9, 10)]

_state = {"stance": 0, "t0": 0.0, "prev": None, "delayed": False, "pushed": False}


def _reset(obs):
    _state["stance"] = int(PARAMS["first"])
    _state["t0"] = 0.0
    _state["prev"] = None
    _state["delayed"] = False
    _state["pushed"] = False


def controller(obs: dict) -> np.ndarray:
    if obs["step"] == 0:
        _reset(obs)
    t = obs["time"]
    # command delay detection: the applied command differs from the one we sent last step
    if _state["prev"] is not None and not np.allclose(obs["last_action"], _state["prev"], atol=1e-9):
        _state["delayed"] = True
    # felt push: the torso's external force (cfrc_ext row of gym_obs) is non-zero only under a push
    f_push = obs["gym_obs"][273:275]
    if np.linalg.norm(f_push) > 1.0:
        _state["pushed"] = True
    P = PARAMS_DELAY if _state["delayed"] else (PARAMS_PUSH if _state["pushed"] else PARAMS)
    la = obs["dt"] * (P["pred_d"] if _state["delayed"] else 0.0)   # look-ahead to compensate the delay
    qd = obs["joint_rates"]
    q = obs["joint_angles"] + la * qd
    w = obs["root_angular_velocity"]
    roll, pitch, yaw = obs["roll"], obs["pitch"], obs["yaw"]
    c0, s0 = np.cos(yaw), np.sin(yaw)
    roll += la * (c0 * w[0] + s0 * w[1])
    pitch += la * (-s0 * w[0] + c0 * w[1])
    yaw += la * w[2]
    c, s = np.cos(P["yaw_frac"] * yaw), np.sin(P["yaw_frac"] * yaw)
    R = np.array([[c, s], [-s, c]])           # world -> heading frame
    wh = R @ w[:2]
    wz = w[2]
    v = R @ (obs["com_velocity_xy"] + P["kff"] * 0.15 * f_push / 42.0)   # anticipate the push's velocity change
    feet = [R @ obs["foot_xyz_rel"][:2], R @ obs["foot_left_xyz_rel"][:2]]
    contact = [obs["foot_contact"], obs["foot_left_contact"]]

    if t < P["t_settle"]:                                # land on both feet before the first step
        _state["t0"] = t
    st = _state["stance"]
    sw = 1 - st
    phase_t = t - _state["t0"]
    # where the swing foot should land (capture-point style, relative to the mass centre)
    side = -1.0 if sw == 0 else 1.0                     # swing side direction in y (right = -y)
    v_d = P["v_d"] * min(1.0, t / P["ramp"]) if P["ramp"] > 0 else P["v_d"]   # speed ramp from standstill
    x_d = P["x0"] + P["cv"] * v[0] + P["kv"] * (v[0] - v_d)
    y_d = side * P["w0"] + P["cvl"] * v[1]
    err = np.hypot(x_d - feet[sw][0], y_d - feet[sw][1])
    near = err < P["tol"]
    far = err > P["lift_tol"]                            # big placement error (e.g. after a push): real step
    t_max = P["t_max_far"] if (far and phase_t < P["t_max_far"]) else P["t_max_frac"] * P["T"]
    if (phase_t > P["t_min_frac"] * P["T"] and contact[sw] and near and not far) or phase_t > t_max:
        _state["stance"], _state["t0"] = sw, t
        st, sw = sw, st
        phase_t = 0.0
        side = -side
        x_d = P["x0"] + P["cv"] * v[0] + P["kv"] * (v[0] - v_d)
        y_d = side * P["w0"] + P["cvl"] * v[1]
    ph = phase_t / P["T"]

    tau = np.zeros(17)
    kp = np.zeros(17)
    kd = np.zeros(17)
    qt = np.zeros(17)

    # abdomen: hold straight
    kp[0:3], kd[0:3] = P["kp_ab"], P["kd_ab"]
    kp[1], kd[1] = P["kp_abz"], P["kd_abz"]
    # arms: hang
    kp[11:17], kd[11:17] = P["kp_arm"], P["kd_arm"]
    qt[11:17] = [0.0, 0.0, -0.4, 0.0, 0.0, -0.4]

    # swing leg: servo the foot (relative to the mass centre) to a capture-point target
    sx, sz, sy, sk = LEG[sw]
    fx, fy = feet[sw]
    qt[sy] = q[sy] - P["kpos"] * (x_d - fx)
    qt[sx] = q[sx] - side * P["kpos_y"] * (y_d - fy)
    qt[sz] = -P["ysw"] * yaw * (1.0 if sw == 0 else -1.0)
    qt[sk] = P["sw_knee"] if (ph < P["knee_frac"] or far) else -0.1
    kp[[sx, sy]] = P["kp_h"]
    kd[[sx, sy]] = P["kd_h"]
    kp[sz], kd[sz] = P["kp_hz"], P["kd_hz"]
    kp[sk], kd[sk] = P["kp_k"], P["kd_k"]

    # stance knee
    ax, az, ay, ak = LEG[st]
    qt[ak] = P["st_knee"]
    kp[ak], kd[ak] = P["kp_sk"], P["kd_sk"]

    # abdomen_x: lean the upper torso over the stance hip (offloads the stance hip_x motor)
    qt[2] = -P["abx_st"] if st == 0 else P["abx_st"]

    if _state["delayed"]:                                # low-inertia twist loops lose phase margin under delay
        for j in (1, 4, 8):
            kp[j] *= P["zs_d"]
            kd[j] *= P["zs_d"]
        kd *= P["kds_d"]
    tau = kp * (qt - q) - kd * qd + STIFF * q
    tau[sx] = np.clip(tau[sx], -P["sw_x_max"], P["sw_x_max"])

    # stance hip: torso orientation
    tp = P["kp_t"] * (pitch - P["pitch_d"]) + P["kd_t"] * wh[1]
    roll_d = (P["roll_st"] + P["abx_st"]) * (1.0 if st == 0 else -1.0)      # lean toward the stance side
    tr = P["kp_r"] * (roll - roll_d) + P["kd_r"] * wh[0]
    ty = P["kp_y"] * yaw + P["kd_y"] * wz
    zs = P["zs_d"] if _state["delayed"] else 1.0
    sgn = (1.0, -1.0)                                    # right/left hip_x and hip_z axes are mirrored
    if phase_t < P["t_ds"] or t < P["t_settle"]:
        # double support: both legs share the torso torque; friction couple lets hip_z correct yaw
        for leg in (0, 1):
            lx, lz, ly, lk = LEG[leg]
            tau[ly] = 0.5 * tp + STIFF[ly] * q[ly]
            tau[lx] = sgn[leg] * 0.5 * tr + STIFF[lx] * q[lx]
            tau[lz] = sgn[leg] * ty + zs * (P["kpz"] * (0.0 - q[lz]) - P["kd_hz"] * qd[lz]) + STIFF[lz] * q[lz]
            tau[lk] = P["kp_sk"] * (P["st_knee"] - q[lk]) - P["kd_sk"] * qd[lk]
    else:
        tau[ay] = tp - tau[sy] + STIFF[ay] * q[ay]
        tau[ax] = sgn[st] * tr + P["comp"] * tau[sx] + STIFF[ax] * q[ax]
        tau[az] = sgn[st] * P["ky_ss"] * ty + zs * (P["kpz"] * (0.0 - q[az]) - P["kd_hz"] * qd[az]) + STIFF[az] * q[az]
        if contact[0] and contact[1]:                    # both feet down: friction couple can correct yaw
            tau[az] += sgn[st] * P["ky_c"] * ty
            tau[sz] += sgn[sw] * P["ky_c"] * ty
    out = np.clip(tau / obs["gear"], -0.4, 0.4)          # actual gears: compensates weakened motors
    _state["prev"] = out
    return out
