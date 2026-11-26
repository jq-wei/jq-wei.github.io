"""Hand-designed stepping controller for the MuJoCo Humanoid (no learning).

Structure: a two-state gait machine (which leg is in stance) with a fast cadence
(step period T ~ 0.17 s) drives joint-space PD targets; balance feedback sits on top.
All lateral/forward quantities are expressed in the heading (yaw) frame.
  * stance leg : hip_y torque = PD on torso pitch (kp_pitch, kd_pitch), hip_x torque =
                 PD on torso roll (kp_roll, kd_roll), knee PD to a nearly straight angle
                 (knee_st); if the swing foot is late the stance knee bends (k_late) to
                 lower the hips.
  * swing leg  : foot-placement target relative to the mass centre, Raibert style
                 (x_f = vx*T/2 + k_v*(vx - vx_des) + x_off, lateral distance
                 d = width + k_y * v_out, never below d_min so the legs do not cross);
                 the foot follows a smoothstep path (s_reach) with a lift of h_lift and is
                 turned into hip_y / knee targets by two-link inverse kinematics (A, B);
                 hip_x places the foot laterally, with torso-roll compensation.
  * hip_z      : heading PD (kp_yaw, kd_yaw) -- the feet are points, so yaw can only be
                 corrected in double support or through the swing leg's reaction.
  * abdomen    : stiff PD to zero (kp_abd); arms: soft PD to a neutral pose.
Steps switch when the swing foot touches down (after T/2) or at 2 T.
Gains are in N m / rad and divided by the motor gear to give commands in [-0.4, 0.4].
Status: steps in place / drifts forward slowly and survives the nominal plant on most
seeds; it does not yet reject the push_v1 pushes or reach the 3 m forward rung.
"""
import numpy as np

PARAMS = {              # tunable (<= 12 for hl_control.harbor.tune)
    "T": 0.17,           # nominal step period [s]
    "vx_des": 0.2,      # desired forward speed [m/s]
    "k_v": 0.08,        # Raibert velocity gain [s]
    "x_off": -0.01,       # forward foot placement offset [m]
    "k_y": 0.45,         # lateral capture gain [s]
    "width": 0.16,      # nominal lateral foot distance from the mass centre [m]
    "h_lift": 0.101,     # swing foot lift [m]
    "kp_pitch": 307.56,  # stance hip pitch gain [N m/rad]
    "kp_roll": 199.0,   # stance hip roll gain [N m/rad]
    "kp_hip": 427.2,    # swing hip PD [N m/rad]
    "kd_hip": 6.63,
    "pitch_des": -0.007,  # torso lean target [rad]
}
CONST = {               # fixed structure gains
    "t_settle": 0.15,   # initial double-support time [s]
    "s_reach": 0.7,     # fraction of the step period in which the swing foot reaches its target
    "z_reach": 0.1,     # extra downward reach per unit of lateness [m]
    "d_min": 0.05,      # minimum lateral foot distance from the mass centre [m]
    "knee_st": 0.08,    # stance knee flexion [rad]
    "k_late": 0.8,      # extra stance-knee flexion per unit of swing lateness [rad]
    "kd_pitch": 40.0,
    "kd_roll": 25.0,
    "kp_knee": 300.0,   # stance knee PD [N m/rad]
    "kd_knee": 8.0,
    "kp_knee_sw": 150.0, # swing knee PD [N m/rad]
    "kp_yaw": 150.0,    # heading PD through the hip_z motors [N m/rad]
    "kd_yaw": 20.0,
    "kp_abd": 150.0,    # abdomen PD [N m/rad]
    "kd_abd": 10.0,
}
PARAM_BOUNDS = {
    "T": (0.15, 0.3), "vx_des": (0.0, 0.5), "k_v": (0.0, 0.35), "x_off": (-0.1, 0.1), "k_y": (0.1, 0.45),
    "width": (0.05, 0.2), "h_lift": (0.03, 0.15), "kp_pitch": (200.0, 800.0), "kp_roll": (50.0, 400.0),
    "kp_hip": (300.0, 900.0), "kd_hip": (2.0, 20.0), "pitch_des": (-0.1, 0.2),
}

LEG_LEN = 0.85
A, B = 0.383, 0.33      # thigh (hip to knee joint) and shin (knee joint to foot-sphere centre) lengths [m]
G = 9.81
_state = {"stance": 0, "t_sw": 0.0, "last_t": -1.0, "started": False}


def _reset():
    _state.update(stance=0, t_sw=0.0, started=False)


def controller(obs: dict) -> np.ndarray:
    P = {**CONST, **PARAMS}
    t = float(obs["time"]); dt = float(obs["dt"])
    if t < _state["last_t"] or t == 0.0:
        _reset()
    _state["last_t"] = t
    q = obs["joint_angles"]; qd = obs["joint_rates"]; gear = obs["gear"]
    pitch = obs["pitch"]; roll = obs["roll"]
    w = obs["root_angular_velocity"]
    yaw = obs["yaw"]; cy, sy = np.cos(yaw), np.sin(yaw)
    def to_body(v):                                   # world xy -> heading (yaw) frame
        return cy * v[0] + sy * v[1], -sy * v[0] + cy * v[1]
    vx, vy = to_body(obs["com_velocity_xy"])
    vx_w = obs["com_velocity_xy"][0]
    contact = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]

    # ---- gait state machine ----
    T = P["T"]
    if t < P["t_settle"]:             # settle in double support first
        _state["t_sw"] = 0.0
        single = False
    else:
        single = True
        _state["t_sw"] += dt
        sw = 1 - _state["stance"]
        if (_state["t_sw"] > 0.5 * T and contact[sw]) or _state["t_sw"] > 2.0 * T:
            _state["stance"] = sw
            _state["t_sw"] = 0.0
    st = _state["stance"]; sw = 1 - st
    s = _state["t_sw"] / T

    tau = np.zeros(17)   # torques [N m]
    HX = [3, 7]; HZ = [4, 8]; HY = [5, 9]; KN = [6, 10]
    side = [1.0, -1.0]   # right, left sign for roll torques

    # ---- abdomen: stiff PD to zero ----
    for j in (0, 1, 2):
        tau[j] = -P["kp_abd"] * q[j] - P["kd_abd"] * qd[j]
    # ---- arms: soft PD ----
    arm_t = np.array([0.0, 0.0, -0.5, 0.0, 0.0, -0.5])
    for k, j in enumerate(range(11, 17)):
        tau[j] = -8.0 * (q[j] - arm_t[k]) - 0.5 * qd[j]

    late = max(0.0, s - 0.9) if single else 0.0            # swing foot late -> lower the hips
    knee_st = P["knee_st"] + P["k_late"] * late

    def stance_leg(i, scale=1.0):
        tau[HY[i]] = scale * (P["kp_pitch"] * (pitch - P["pitch_des"]) + P["kd_pitch"] * w[1]) - 5.0 * qd[HY[i]]
        tau[HX[i]] = scale * side[i] * (P["kp_roll"] * roll + P["kd_roll"] * w[0]) - 3.0 * qd[HX[i]]
        tau[HZ[i]] = -20.0 * q[HZ[i]] - 3.0 * qd[HZ[i]] + side[i] * (P["kp_yaw"] * yaw + P["kd_yaw"] * w[2])
        tau[KN[i]] = P["kp_knee"] * (-knee_st - q[KN[i]]) - P["kd_knee"] * qd[KN[i]]

    if not single:
        stance_leg(0, 0.5); stance_leg(1, 0.5)
    else:
        stance_leg(st)
        # ---- swing leg: foot-placement target + smooth trajectory + two-link inverse kinematics ----
        fx, fy = to_body(obs["foot_xyz_rel"] if sw == 0 else obs["foot_left_xyz_rel"])
        fsx, _ = to_body(obs["foot_xyz_rel"] if st == 0 else obs["foot_left_xyz_rel"])
        pp = pitch + q[0]                                                    # pelvis pitch (torso + abdomen_y)

        def fk(hy, kn):                                                      # foot centre relative to the hip: (forward, down)
            th = hy + pp
            return -(A * np.sin(th) + B * np.sin(th - kn)), A * np.cos(th) + B * np.cos(th - kn)

        if _state["t_sw"] <= dt + 1e-9:                                      # liftoff: remember where the foot starts
            _state["x0"], _state["y0"] = fx, fy
        xf = vx * T / 2 + P["k_v"] * (vx - P["vx_des"]) + P["x_off"]          # target foot x relative to the mass centre
        xf = np.clip(xf, -0.3, 0.45)
        if late > 0:                                                          # late: blend toward the touchdown geometry
            b = min(late / 0.5, 1.0)
            xf = (1 - b) * xf + b * (-fsx)
        v_out = -side[sw] * vy                                               # lateral COM speed toward the swing side
        d = np.clip(P["width"] + P["k_y"] * v_out, P["d_min"], 0.4)         # target foot distance outward of the mass centre (never cross the legs)
        yf = -side[sw] * d                                                   # right foot: -y, left foot: +y
        sm = 3 * min(s / P["s_reach"], 1.0) ** 2 - 2 * min(s / P["s_reach"], 1.0) ** 3   # smoothstep reach
        x_t = _state["x0"] + (xf - _state["x0"]) * sm
        y_t = _state["y0"] + (yf - _state["y0"]) * sm
        px, _ = fk(q[HY[sw]], q[KN[sw]])
        x_hip = fx - px                                                      # hip x relative to the mass centre
        _, z_st = fk(q[HY[st]], q[KN[st]])                                   # hip height above the stance foot centre
        lift = P["h_lift"] * np.sin(np.pi * min(s, 1.0)) if s < 1.0 else 0.0
        z_t = z_st - lift + P["z_reach"] * late                              # reach down when late
        xr = x_t - x_hip
        r = min(np.hypot(xr, z_t), A + B - 0.005)
        phi = np.arccos(np.clip((r * r - A * A - B * B) / (2 * A * B), -1.0, 1.0))   # knee flexion
        alpha = np.arctan2(xr, max(z_t, 0.05)); beta = np.arctan2(B * np.sin(phi), A + B * np.cos(phi))
        hy_t = -(alpha + beta) - pp
        kn_t = -max(phi, P["knee_st"])
        hx_t = -np.arcsin(np.clip((abs(y_t) - 0.1) / LEG_LEN, -0.5, 0.5)) - side[sw] * roll
        hx_t = np.clip(hx_t, -0.42, 0.08)
        tau[KN[sw]] = P["kp_knee_sw"] * (kn_t - q[KN[sw]]) - 0.05 * P["kp_knee_sw"] * qd[KN[sw]]
        tau[HY[sw]] = P["kp_hip"] * (hy_t - q[HY[sw]]) - P["kd_hip"] * qd[HY[sw]]
        tau[HX[sw]] = P["kp_hip"] * 0.6 * (hx_t - q[HX[sw]]) - P["kd_hip"] * qd[HX[sw]]
        tau[HZ[sw]] = -40.0 * q[HZ[sw]] - 3.0 * qd[HZ[sw]] + side[sw] * (P["kp_yaw"] * yaw + P["kd_yaw"] * w[2])

    u = tau / gear
    return np.clip(u, -0.4, 0.4)
