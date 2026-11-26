"""SIMBICON-style walking controller for the MuJoCo Humanoid (17 torque motors, point feet).

Control law
-----------
* Every motor is a joint PD servo, tau = kp (q_d - q) - kd qdot, converted to a command by
  dividing by the motor gear and clipping to [-0.4, 0.4].  The pelvis is light (6.6 kg) and
  hangs on a soft abdomen, so a hip torque sees only ~0.2 kg m^2 in the air; with a 15 ms
  zero-order hold a damper is stable only for kd*dt < 2 I, hence the small kd values.
* For the first ``t_settle`` seconds the robot stands on both legs (the model is spawned
  10 cm above the floor and has to land).  Then a two-state gait machine (right stance /
  left stance) runs, switching when the swing foot touches down after ``t_min`` seconds
  or at ``t_max`` at the latest.
* Foot placement (capture-point style): the swing foot is sent to a point relative to the
  mass centre, ``x0 + cd*d + cv*(v - v_des)`` ahead and ``y0 + cd_lat*d_lat + cv_lat*(v_lat -
  v_lat_des)`` outward, where d / d_lat are the mass-centre offsets from the stance foot and
  v the mass-centre velocity (heading frame).  The desired velocity is ``v_d`` along the
  *world* x axis: point feet transmit no yaw torque and the body slowly turns, so the legs
  step toward world +x whatever the body's heading (a crab walk when yawed).  The target is
  converted to swing hip pitch / roll angles with the pelvis orientation and the stance-leg
  geometry (hip position estimated from the stance foot), so a lean does not shift the
  landing point; the reach is capped at ``max_reach`` of the leg length.
* The swing knee flexes to ``swing_knee`` by ``t_flex``, holds until ``t_ext0`` and extends
  to ``land_knee`` by ``t_ext1``; the stance knee ramps from its touchdown angle to
  ``stance_knee`` over ``knee_ramp`` (nearly straight: the knee motor is only 80 N m).
* The stance hip does not servo a joint angle: it applies a virtual PD on the pelvis pitch
  (toward ``pitch_d``, gains kp_t/kd_t) and roll (kp_r/kd_r), minus ``comp`` times the
  swing-hip torque so the swing leg's reaction goes into the ground instead of the pelvis
  (SIMBICON's tau_A = -tau_torso - tau_B).  The pelvis orientation is the root (torso)
  orientation plus the abdomen joint angles; the abdomen is servoed so the torso stays
  upright in the world.  Hip yaw joints and arms hold zero with soft servos.

PARAMS (gait, tuned with hl_control.harbor.tune on seeds 0-11): v_d target speed [m/s];
t_min/t_max stance timing [s]; x0 [m], cd [-], cv [s] sagittal placement; y0 [m], cd_lat [-],
cv_lat [s] lateral placement; pitch_d pelvis lean [rad]; swing_knee, land_knee [rad].
GAINS: servo gains per joint group, the virtual pelvis PD gains, the knee schedule times,
the placement geometry (leg_len, hip_half, max_reach), t_settle and comp.  The remaining
GAINS entries (kp_y/kd_y yaw reaction wheel, k_y world-y pull, k_lean, settle_pitch,
cv_back, pick_stance) are optional variants that are switched off / neutral; they were
tried and did not improve the success rate.
"""

import numpy as np

PARAMS = {
    "v_d": 0.105,         # target forward speed [m/s]
    "t_max": 0.318,      # stance duration cap [s]
    "t_min": 0.117,      # earliest touchdown switch [s]
    "x0": 0.036,         # nominal landing point ahead of the mass centre [m]
    "cd": 0.163,          # sagittal placement gain on d [-]
    "cv": 0.243,         # sagittal placement gain on v - v_d [s]
    "y0": 0.200,         # nominal landing point outward of the mass centre [m]
    "cd_lat": 0.574,      # lateral placement gain on d_lat [-]
    "cv_lat": 0.117,      # lateral placement gain on v_lat [s]
    "pitch_d": 0.027,     # desired pelvis pitch (forward lean) [rad]
    "swing_knee": -0.445, # swing knee flexion target [rad]
    "land_knee": -0.258,  # swing knee target before landing [rad]
}
PARAM_BOUNDS = {
    "v_d": (0.0, 0.8), "t_max": (0.2, 0.6), "t_min": (0.05, 0.3), "x0": (-0.1, 0.2),
    "cd": (0.0, 1.0), "cv": (0.0, 0.8), "y0": (0.0, 0.3), "cd_lat": (0.0, 1.5), "cv_lat": (0.0, 1.0),
    "pitch_d": (-0.1, 0.3), "swing_knee": (-1.8, -0.4), "land_knee": (-0.8, -0.05),
}
GAINS = {
    "kp_hy": 400.0, "kd_hy": 8.0,         # hip pitch servo (gear 300)
    "kp_hx": 200.0, "kd_hx": 5.0,         # hip roll servo (gear 100)
    "kp_hz": 30.0, "kd_hz": 1.0,          # hip yaw servo (gear 100)
    "kp_knee": 200.0, "kd_knee": 5.0,     # knee servo (gear 200)
    "kp_abd": 200.0, "kd_abd": 5.0,       # abdomen pitch / roll servo (gear 100)
    "kp_abz": 50.0, "kd_abz": 1.0,        # abdomen yaw servo
    "kp_arm": 10.0, "kd_arm": 0.5, "kp_elb": 5.0, "kd_elb": 0.2,
    "kp_t": 450.0, "kd_t": 15.0,          # virtual pelvis pitch PD through the stance hip
    "kp_r": 200.0, "kd_r": 8.0,           # virtual pelvis roll PD through the stance hip
    "kp_y": 0.0, "kd_y": 0.0,             # body yaw PD through the stance hip_z (0 = off)
    "pick_stance": 0.0, "pick_tc": 0.3,   # 1: choose the first stance leg from the lateral lean at the end of settle
    "settle_pitch": 0.0, # extra pelvis lean during the settle phase, on top of pitch_d
    "cv_back": 0.243,    # sagittal placement gain when the mass centre is slower than v_d (backward falls)
    "k_lean": 0.0,       # extra backward pelvis lean per m/s of excess forward speed [rad s/m]
    "k_y": 0.0, "v_y_max": 0.15,          # pull toward world y = 0: desired lateral speed [1/s], cap [m/s]
    "stance_knee": -0.05, "knee_ramp": 0.15,
    "t_flex": 0.06, "t_ext0": 0.12, "t_ext1": 0.22,   # swing knee schedule times [s]
    "max_reach": 0.5,    # cap on the swing foot's horizontal offset from the hip, as a fraction of leg_len
    "leg_len": 0.75,     # hip-to-foot distance used by the placement geometry [m]
    "hip_half": 0.10,    # lateral hip offset from the pelvis centre [m]
    "t_settle": 0.3,     # hold a standing pose this long after reset [s]
    "comp": 0.7,         # fraction of the swing-hip torque fed back to the stance hip
}

# motor indices
AB_Y, AB_Z, AB_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_KNEE = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_KNEE = 7, 8, 9, 10
SHOULDERS = [11, 12, 14, 15]
ELBOWS = [13, 16]
HIPS_X, HIPS_Z, HIPS_Y, KNEES = [R_HX, L_HX], [R_HZ, L_HZ], [R_HY, L_HY], [R_KNEE, L_KNEE]
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], dtype=float)

_state = {"stance": "right", "t_switch": 0.0, "last_step": -1, "knee_land": -0.05, "knee_sw0": -0.05}


def _reset():
    _state["stance"] = "right"
    _state["t_switch"] = 0.0
    _state["last_step"] = -1
    _state["knee_land"] = GAINS["stance_knee"]
    _state["knee_sw0"] = GAINS["stance_knee"]


def _asin(x):
    return float(np.arcsin(np.clip(x, -0.95, 0.95)))


def controller(obs: dict) -> np.ndarray:
    P, G = PARAMS, GAINS
    step = int(obs["step"])
    t = float(obs["time"])
    if step == 0 or step <= _state["last_step"]:
        _reset()
    _state["last_step"] = step

    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    yaw = obs["yaw"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    w = obs["root_angular_velocity"]
    # torso rates in the heading frame (world rotated by -yaw)
    wx_t, wy_t = cy * w[0] + sy * w[1], -sy * w[0] + cy * w[1]
    # pelvis orientation = torso orientation + abdomen joints (abdomen_y about +y, abdomen_x about +x)
    pitch = obs["pitch"] + q[AB_Y]
    roll = obs["roll"] + q[AB_X]
    wy = wy_t + qd[AB_Y]
    wx = wx_t + qd[AB_X]
    v_w = obs["com_velocity_xy"]
    vx, vy = cy * v_w[0] + sy * v_w[1], -sy * v_w[0] + cy * v_w[1]

    # ---- servo setup (every joint; the stance hip is overridden below) ----
    q_d = np.zeros(17)
    kp = np.zeros(17)
    kd = np.zeros(17)
    kp[[AB_Y, AB_X]] = G["kp_abd"]; kd[[AB_Y, AB_X]] = G["kd_abd"]
    kp[AB_Z] = G["kp_abz"]; kd[AB_Z] = G["kd_abz"]
    kp[HIPS_Y] = G["kp_hy"]; kd[HIPS_Y] = G["kd_hy"]
    kp[HIPS_X] = G["kp_hx"]; kd[HIPS_X] = G["kd_hx"]
    kp[HIPS_Z] = G["kp_hz"]; kd[HIPS_Z] = G["kd_hz"]
    kp[KNEES] = G["kp_knee"]; kd[KNEES] = G["kd_knee"]
    kp[SHOULDERS] = G["kp_arm"]; kd[SHOULDERS] = G["kd_arm"]
    kp[ELBOWS] = G["kp_elb"]; kd[ELBOWS] = G["kd_elb"]
    # desired pelvis lean: nominal pitch_d, reduced when the mass centre runs faster than v_d
    pitch_ref = P["pitch_d"] - G["k_lean"] * float(np.clip(vx - P["v_d"], -0.5, 1.0))
    # torso pitch = pelvis pitch - abdomen_y: abdomen targets keep the torso upright in the world
    q_d[AB_Y] = np.clip(pitch - P["pitch_d"], -1.0, 0.5)
    q_d[AB_X] = np.clip(roll, -0.5, 0.5)
    q_d[KNEES] = G["stance_knee"]

    # ---- settle: stand on both legs until the drop onto the floor is absorbed ----
    if t < G["t_settle"]:
        _state["t_switch"] = t
        if G["pick_stance"] > 0.5:
            # first stance leg = the loaded one: the side the mass centre is falling toward (heading frame)
            mid_y = 0.5 * (obs["foot_xyz_rel"][1] + obs["foot_left_xyz_rel"][1])
            lean = -(cy * mid_y - sy * 0.5 * (obs["foot_xyz_rel"][0] + obs["foot_left_xyz_rel"][0])) + G["pick_tc"] * vy
            _state["stance"] = "left" if lean > 0.0 else "right"
        q_d[HIPS_X] = -0.05
        tau = kp * (q_d - q) - kd * qd
        tau[HIPS_Y] += 0.5 * (G["kp_t"] * (pitch - P["pitch_d"] - G["settle_pitch"]) + G["kd_t"] * wy)
        return np.clip(tau / GEAR, -0.4, 0.4)

    # ---- gait machine ----
    stance = _state["stance"]
    t_in = t - _state["t_switch"]
    swing_contact = obs["foot_left_contact"] if stance == "right" else obs["foot_contact"]
    if (t_in >= P["t_min"] and swing_contact) or t_in >= P["t_max"]:
        stance = "left" if stance == "right" else "right"
        _state["stance"] = stance
        _state["t_switch"] = t
        _state["knee_land"] = float(q[R_KNEE if stance == "right" else L_KNEE])
        _state["knee_sw0"] = float(q[L_KNEE if stance == "right" else R_KNEE])
        t_in = 0.0
    if stance == "right":
        st_foot = obs["foot_xyz_rel"]
        sHX, sHZ, sHY, sK = R_HX, R_HZ, R_HY, R_KNEE
        wHX, wHZ, wHY, wK = L_HX, L_HZ, L_HY, L_KNEE
        sigma = 1.0      # stance-side mirror sign of the hip_x axis
        s_out = 1.0      # swing (left) leg: outward is +y
    else:
        st_foot = obs["foot_left_xyz_rel"]
        sHX, sHZ, sHY, sK = L_HX, L_HZ, L_HY, L_KNEE
        wHX, wHZ, wHY, wK = R_HX, R_HZ, R_HY, R_KNEE
        sigma = -1.0
        s_out = -1.0
    L = G["leg_len"]
    # stance foot -> mass centre, heading frame
    dx_w, dy_w = -st_foot[0], -st_foot[1]
    d = cy * dx_w + sy * dy_w                      # mass centre ahead of the stance foot
    d_lat = (-sy * dx_w + cy * dy_w) * s_out       # mass centre toward the swing side
    v_lat = vy * s_out
    # stance thigh world angles -> stance hip position relative to the mass centre
    alpha_st = -(pitch + q[sHY])                   # stance leg forward tilt (foot ahead of hip)
    gamma_st = -s_out * roll - q[sHX]              # stance leg outward tilt (stance side)
    hip_x = -d - L * np.sin(alpha_st)              # stance hip ahead of the mass centre
    hip_y = -d_lat + L * np.sin(gamma_st) + 2.0 * G["hip_half"]   # swing hip toward the swing side

    # ---- swing leg targets: capture-point style placement relative to the mass centre ----
    # desired mass-centre velocity: v_d along world +x (the body may be yawed), plus a pull toward the
    # world x axis (y = 0) so the path stays straight; expressed in the heading frame
    vy_w_des = float(np.clip(-G["k_y"] * obs["com_xyz"][1], -G["v_y_max"], G["v_y_max"]))
    vx_des = cy * P["v_d"] + sy * vy_w_des
    vy_des = -sy * P["v_d"] + cy * vy_w_des
    dv = vx - vx_des
    x_des = P["x0"] + P["cd"] * d + (P["cv"] if dv >= 0.0 else G["cv_back"]) * dv
    y_des = P["y0"] + P["cd_lat"] * d_lat + P["cv_lat"] * (v_lat - s_out * vy_des)
    reach = G["max_reach"] * L
    phi = _asin(np.clip(x_des - hip_x, -reach, reach) / L)    # swing thigh forward angle (world)
    psi = _asin(np.clip(y_des - hip_y, -reach, reach) / L)    # swing thigh outward angle (world)
    q_d[wHY] = -phi - pitch
    q_d[wHX] = s_out * roll - psi
    # swing knee schedule (time based, so the leg is extended before any touchdown): flex during
    # [0, t_flex], hold until t_ext0, then extend to land_knee by t_ext1
    if t_in < G["t_flex"]:
        s_k = t_in / max(G["t_flex"], 1e-3)
        q_d[wK] = _state["knee_sw0"] + s_k * (P["swing_knee"] - _state["knee_sw0"])
    elif t_in < G["t_ext0"]:
        q_d[wK] = P["swing_knee"]
    else:
        s_k = min(1.0, (t_in - G["t_ext0"]) / max(G["t_ext1"] - G["t_ext0"], 1e-3))
        q_d[wK] = P["swing_knee"] + s_k * (P["land_knee"] - P["swing_knee"])
    # stance knee: ramp from the touchdown angle to the stance target so the leg does not snap straight
    ramp = min(1.0, t_in / max(G["knee_ramp"], 1e-3))
    q_d[sK] = _state["knee_land"] + ramp * (G["stance_knee"] - _state["knee_land"])

    tau = kp * (q_d - q) - kd * qd
    # ---- stance hip: virtual pelvis PD minus the swing-hip reaction ----
    tau[sHY] = G["kp_t"] * (pitch - pitch_ref) + G["kd_t"] * wy - G["comp"] * tau[wHY]
    tau[sHX] = sigma * (G["kp_r"] * roll + G["kd_r"] * wx) + G["comp"] * tau[wHX]
    # yaw: the stance hip_z as a reaction wheel (a point foot transmits no yaw torque; the leg spins
    # about its own axis while the body receives the opposite torque, bounded by the joint range)
    if G["kp_y"] > 0.0:
        tau[sHZ] = sigma * (G["kp_y"] * (yaw + q[AB_Z]) + G["kd_y"] * (w[2] + qd[AB_Z]))
    return np.clip(tau / GEAR, -0.4, 0.4)
