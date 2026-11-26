"""Foot-placement stepping controller for the MuJoCo Humanoid on uneven, tilted ground.

Control law (hand-designed feedback, no learning):

* Step machine: right stance / left stance. The swing foot follows a short Cartesian trajectory
  (lift from where it stands, travel, lower) through two-link leg inverse kinematics in the
  gravity-aligned heading frame. After the nominal swing time the foot keeps reaching down and
  the stance knee yields until the foot touches; the touch-down swaps the legs. Touch-down is
  the only thing the controller learns about the terrain.
* Foot placement balances the point-footed robot (inverted-pendulum capture rule). Relative to
  the mass centre the touch-down target is
      ahead:    x = cx * vx + kx * (vx - vx_des) + integral(ki * (vx - vx_des))
      sideways: y = +-width + cy * vy + ky * (vy - vy_des)
  where the desired velocity is v_des along world +x, rotated into the heading frame, so the
  robot keeps moving towards +x whatever way it happens to face (heading is not controlled:
  a ball foot has no torsional grip).
* The stance hip holds the pelvis level in the world (PD on pelvis pitch and roll, minus the
  swing-hip torque so the swing leg does not tip the light pelvis, plus a constant abduction
  torque carrying the swing side). The abdomen keeps the torso upright in the world. The stance
  knee is a spring around a crouch. Hip yaw and arms are joint PD loops to zero.
* All rates are two-sample averages: the pelvis is a light link between strong hip motors and a
  weak abdomen, and raw-rate damping chatters at the 15 ms control period.

Gains are in N m/rad and N m s/rad; commands are torque / gear, clipped to +-0.4.

Known limits (measured on `uneven_v2`): about half of the seeds reach the `forward` rung.
Downhill seeds do; uphill seeds mostly stall where the slope begins, turn sideways and slide back.
"""

import numpy as np

PARAMS = {
    # gait timing
    "t_settle": 0.20,     # s, initial double-support hold after the drop
    "t_swing": 0.26,      # s, nominal swing duration
    "t_max": 0.55,        # s, forced stance swap if no touch-down came
    "s_td": 0.55,         # swing-phase fraction after which a touch-down swaps stance
    # swing foot trajectory
    "clear": 0.15,        # m, foot lift at mid swing (large: the ground under the swing foot may be higher)
    "reach_rate": 1.0,    # m/s, how fast the foot reaches down after the nominal swing
    "knee_stance": -0.35,  # rad, stance knee set point (crouch, leaves leg length to reach the ground)
    "yield_rate": 1.5,    # rad/s, stance knee flexion rate after the nominal swing time
    # foot placement
    "cx": 0.14,           # s, neutral sagittal placement per m/s of forward speed
    "kx": 0.12,           # s, speed-error gain of the sagittal placement
    "ki": 0.10,           # integral gain of the sagittal placement (m per m of speed-error integral)
    "i_max": 0.12,        # m, cap of the integral term (about what a 5 degree slope needs)
    "x0": 0.0,            # m, constant sagittal offset
    "width": 0.06,        # m, lateral placement offset from the mass centre
    "cy": 0.30,           # s, lateral placement per m/s of lateral speed
    "ky": 0.10,           # s, speed-error gain of the lateral placement
    "v_des": 0.45,        # m/s, target speed along world +x
    "vy_max": 0.15,       # m/s, cap of the side-stepping speed asked for when facing away from +x
    "pitch_des": 0.03,    # rad, forward torso lean
    # PD gains (N m/rad, N m s/rad)
    "kp_pel": 200.0, "kd_pel": 6.0,        # pelvis pitch through the stance hip_y
    "kp_pelx": 200.0, "kd_pelx": 6.0,      # pelvis roll through the stance hip_x
    "ff_abd": 28.0,                        # N m, stance hip abduction feed-forward
    "couple": 1.0,                         # share of the swing hip_y torque cancelled at the stance hip
    "kp_tor": 200.0, "kd_tor": 10.0,       # torso pitch/roll through the abdomen
    "kp_swy": 250.0, "kd_swy": 6.0,        # swing hip_y (tracks the inverse-kinematics angle)
    "kp_swx": 200.0, "kd_swx": 5.0,        # swing hip_x (task-space lateral foot position)
    "kp_hz": 60.0, "kd_hz": 3.0,           # hip_z (both legs, to zero)
    "kp_knee": 250.0, "kd_knee": 4.0,      # knees (stance uses twice the damping)
    "kp_arm": 12.0, "kd_arm": 1.0,         # arms (to zero)
}

ABY, ABZ, ABX, RHX, RHZ, RHY, RK, LHX, LHZ, LHY, LK, RS1, RS2, RE, LS1, LS2, LE = range(17)
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], dtype=float)
L1, L2 = 0.383, 0.370          # hip-knee and knee-foot-centre lengths [m]
LMAX = L1 + L2 - 0.004

_S = {"t_last": 1e9}


def _reset():
    _S.clear()
    _S.update(t_last=-1.0, stance=None, t_switch=0.0, x_int=0.0, v_f=0.0, vy_f=0.0,
              x_lo=0.0, y_lo=0.0, z_lo=-0.72, dq_prev=None, w_prev=None)


def _fk(a, k):
    """Foot centre relative to the hip in the leg plane; a = thigh angle forward of vertical, k = knee flexion."""
    return L1 * np.sin(a) + L2 * np.sin(a - k), -(L1 * np.cos(a) + L2 * np.cos(a - k))


def _ik(x, z):
    """Thigh angle forward of vertical and knee flexion that put the foot centre at (x, z) below the hip."""
    r = float(np.clip(np.hypot(x, z), 0.35, LMAX))
    k = np.pi - np.arccos(np.clip((L1 * L1 + L2 * L2 - r * r) / (2 * L1 * L2), -1.0, 1.0))
    a = np.arctan2(x, -z) + np.arccos(np.clip((L1 * L1 + r * r - L2 * L2) / (2 * L1 * r), -1.0, 1.0))
    return float(a), float(k)


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = float(obs["time"])
    if obs["step"] == 0 or t < _S["t_last"]:
        _reset()                                  # new episode
    dt = float(obs["dt"])
    _S["t_last"] = t
    q = np.asarray(obs["joint_angles"], dtype=float)
    dq_raw = np.asarray(obs["joint_rates"], dtype=float)
    w_raw = np.asarray(obs["root_angular_velocity"], dtype=float)
    if _S.get("dq_prev") is None:
        _S["dq_prev"], _S["w_prev"] = dq_raw, w_raw
    # two-sample average of the rates: removes the sample-rate chatter of the light pelvis link
    dq = 0.5 * (dq_raw + _S["dq_prev"])
    w_avg = 0.5 * (w_raw + _S["w_prev"])
    _S["dq_prev"], _S["w_prev"] = dq_raw, w_raw

    # heading frame (x forward along the torso heading, y to the left)
    yaw = float(obs["yaw"])
    c, s = np.cos(yaw), np.sin(yaw)

    def hf(v):
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])

    up = hf(obs["up_vector"])
    pitch = float(np.arcsin(np.clip(up[0], -1, 1)))     # torso forward lean
    roll = float(-np.arcsin(np.clip(up[1], -1, 1)))     # torso rotation about heading x
    w = hf(w_avg)
    roll_rate, pitch_rate = float(w[0]), float(w[1])
    v = hf(obs["com_velocity_xy"])
    foot_r, foot_l = hf(np.asarray(obs["foot_xyz_rel"])), hf(np.asarray(obs["foot_left_xyz_rel"]))
    con_r, con_l = bool(obs["foot_contact"]), bool(obs["foot_left_contact"])

    a_f = 0.35                                          # speed filter (the raw value jumps at touch-down)
    _S["v_f"] += a_f * (float(v[0]) - _S["v_f"])
    _S["vy_f"] += a_f * (float(v[1]) - _S["vy_f"])
    vx, vy = _S["v_f"], _S["vy_f"]

    # pelvis attitude in the world
    pel_p, pel_pr = pitch + q[ABY], pitch_rate + dq[ABY]
    pel_r, pel_rr = roll + q[ABX], roll_rate + dq[ABX]

    tau = np.zeros(17)
    # abdomen keeps the torso upright in the world
    tau[ABY] = P["kp_tor"] * (pitch - P["pitch_des"]) + P["kd_tor"] * pitch_rate
    tau[ABX] = P["kp_tor"] * roll + P["kd_tor"] * roll_rate
    tau[ABZ] = -100.0 * q[ABZ] - 8.0 * dq[ABZ]
    for j in (RS1, RS2, RE, LS1, LS2, LE):
        tau[j] = -P["kp_arm"] * q[j] - P["kd_arm"] * dq[j]
    for j in (RHZ, LHZ):
        tau[j] = -P["kp_hz"] * q[j] - P["kd_hz"] * dq[j]

    t_pel = P["kp_pel"] * pel_p + P["kd_pel"] * pel_pr
    t_pelx = P["kp_pelx"] * pel_r + P["kd_pelx"] * pel_rr

    if _S["stance"] is None:
        # initial double support: both legs hold, both hips carry the pelvis
        for hy, hx, kn, sg in ((RHY, RHX, RK, 1.0), (LHY, LHX, LK, -1.0)):
            tau[hy] = 0.5 * t_pel - 60.0 * q[hy] - 6.0 * dq[hy]
            tau[hx] = 0.5 * sg * t_pelx - 100.0 * q[hx] - 10.0 * dq[hx]
            tau[kn] = P["kp_knee"] * (-0.12 - q[kn]) - 2.0 * P["kd_knee"] * dq[kn]
        if t >= P["t_settle"]:
            _S["stance"] = "L"       # becomes "R" in the swap below
            _S["t_switch"] = t - 10.0
        else:
            return np.clip(tau / GEAR, -0.4, 0.4)

    # stance swap on touch-down of the swing foot (late in the swing) or on time-out
    tp = t - _S["t_switch"]
    sw_con = con_l if _S["stance"] == "R" else con_r
    if (tp > P["s_td"] * P["t_swing"] and sw_con) or tp > P["t_max"]:
        _S["stance"] = "L" if _S["stance"] == "R" else "R"
        _S["t_switch"] = t
        tp = 0.0
        new_sw = foot_l if _S["stance"] == "R" else foot_r
        _S["x_lo"], _S["y_lo"] = float(new_sw[0]), float(new_sw[1])

    if _S["stance"] == "R":
        st_hy, st_hx, st_kn, sw_hy, sw_hx, sw_kn = RHY, RHX, RK, LHY, LHX, LK
        st_sg, sw_sg, side = 1.0, -1.0, 1.0
        foot_sw = foot_l
    else:
        st_hy, st_hx, st_kn, sw_hy, sw_hx, sw_kn = LHY, LHX, LK, RHY, RHX, RK
        st_sg, sw_sg, side = -1.0, 1.0, -1.0
        foot_sw = foot_r

    # leg geometry in the gravity-aligned heading frame
    a_sw, k_sw = -(pel_p + q[sw_hy]), -q[sw_kn]
    a_st, k_st = -(pel_p + q[st_hy]), -q[st_kn]
    x_sw, z_sw = _fk(a_sw, k_sw)
    _, z_st = _fk(a_st, k_st)
    r_sw = max(0.4, -z_sw)

    # speed regulation: the desired velocity is along world x, expressed in the heading frame
    vdx, vdy = P["v_des"] * c, -P["v_des"] * s
    ev = vx - vdx
    _S["x_int"] = float(np.clip(_S["x_int"] + P["ki"] * ev * dt, -P["i_max"], P["i_max"]))
    x_t = P["x0"] + P["cx"] * vx + P["kx"] * ev + _S["x_int"]
    x_t = float(np.clip(x_t, -0.25, 0.32))
    y_t = side * P["width"] + P["cy"] * vy + P["ky"] * (vy - np.clip(vdy, -P["vy_max"], P["vy_max"]))
    y_t = side * max(side * y_t, 0.02)            # never step across the mass centre
    y_t = float(np.clip(y_t, -0.26, 0.26))

    # swing foot trajectory
    sp = tp / P["t_swing"]
    blend = float(np.clip(sp / 0.8, 0.0, 1.0))
    blend = blend * blend * (3.0 - 2.0 * blend)
    x_des = _S["x_lo"] + (x_t - _S["x_lo"]) * blend      # relative to the mass centre
    y_des = _S["y_lo"] + (y_t - _S["y_lo"]) * blend
    if tp == 0.0:
        _S["z_lo"] = z_sw                                # hip-relative height the swing foot lifts off from
    if sp <= 1.0:
        # lift from where the foot actually stands (on a slope the trailing foot is higher or lower
        # than the stance foot) and blend to the stance-foot level at the end of the swing
        z_des = _S["z_lo"] + (z_st - _S["z_lo"]) * blend + P["clear"] * np.sin(np.pi * sp)
    else:
        z_des = z_st - P["reach_rate"] * (tp - P["t_swing"])
    hip_x_rel = float(foot_sw[0]) - x_sw                 # swing hip relative to the mass centre
    a_des, k_des = _ik(x_des - hip_x_rel, min(z_des, -0.35))
    q_hy_des = float(np.clip(-a_des - pel_p, -1.7, 0.3))
    tau_sw_y = float(np.clip(P["kp_swy"] * (q_hy_des - q[sw_hy]) - P["kd_swy"] * dq[sw_hy], -120.0, 120.0))
    tau[sw_hy] = tau_sw_y
    tau[sw_kn] = P["kp_knee"] * (-k_des - q[sw_kn]) - P["kd_knee"] * dq[sw_kn]
    tau[sw_hx] = sw_sg * P["kp_swx"] * (y_des - float(foot_sw[1])) / r_sw - P["kd_swx"] * dq[sw_hx]

    # stance leg: pelvis level in the world, knee spring
    tau[st_hy] = t_pel - P["couple"] * tau_sw_y
    tau[st_hx] = st_sg * t_pelx - P["ff_abd"]
    # the stance knee yields once the nominal swing is over, lowering the body onto the swing foot
    k_st_des = P["knee_stance"] - P["yield_rate"] * max(0.0, tp - P["t_swing"])
    tau[st_kn] = P["kp_knee"] * (k_st_des - q[st_kn]) - 2.0 * P["kd_knee"] * dq[st_kn]

    return np.clip(tau / GEAR, -0.4, 0.4)
