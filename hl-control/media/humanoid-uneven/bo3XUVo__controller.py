"""Dynamic stepping controller for the MuJoCo Humanoid on uneven, tilted ground.

The feet are point spheres with no ankles, so the robot cannot stand still: it walks with
short steps (T_step = 0.27 s) and balances by foot placement.  Every quantity is expressed
in the heading (yaw) frame, because the point feet let the body spin freely about the
vertical and the hips act in the body frame.

Control law, once per 15 ms step (stance leg alternates at each confirmed touchdown):
  * Stance hip_y / hip_x: PD on the *pelvis* lean (torso up-vector plus abdomen angles) and
    its rate (kp_att, kd_att) keeps the pelvis upright over the stance foot; a weak position
    term pulls the hip toward neutral.  Stance knee: stiff PD to almost straight
    (knee_stance, kp_knee_st) so the body vaults over a rigid leg.
  * Swing leg: Raibert foot placement.  Target foot position relative to the mass centre is
    x_f = v*T/2 + k_raibert*(v - v_des) (+ x_off), y_f = v_y*T/2 + k_lat*v_y +/- width/2.
    The hip targets are closed on the *measured* swing-foot position (k_foot per unit of
    error over leg length) with a velocity lead (k_fd); the knee flexes to knee_swing and
    re-extends after clear_frac of the step.  Touchdown is accepted after 0.6 T when the foot
    has touched for two steps (or when the stance foot has left the ground).
  * Heading: hip_z of both legs is driven to +/- k_yaw * yaw (clipped to hz_clip), which
    turns the pelvis back toward world +x whenever both feet are planted; the speed demand
    is scaled by cos(yaw) so a turned robot marches in place instead of drifting sideways.
  * Abdomen: stiff PD to neutral (kp_abd); arms: PD to neutral (arm_amp = 0 disables the
    optional counter-swing).
Joint PD gains are deliberately small on the light shin and pelvis because derivative
feedback at 15 ms sampling chatters above ~0.03 ctrl per rad/s.  All gains are in units of
ctrl (-0.4 .. 0.4) per rad or per rad/s.  PARAMS holds the 12 gains searched with
hl_control.harbor.tune; FIXED holds the joint-level gains.
"""

import numpy as np

PARAMS = {              # tunable gains (searched by hl_control.harbor.tune)
    "v_des": 0.46,       # m/s desired forward speed (ramped in over "ramp" seconds)
    "T_step": 0.27,      # s, nominal step period
    "k_raibert": 0.16,   # s, foot placement gain on forward velocity error
    "k_lat": 0.25,       # s, foot placement gain on lateral velocity
    "width": 0.30,       # m, nominal lateral distance between feet
    "kp_att": 2.0,       # stance hip gain on pelvis lean (ctrl per unit up-vector)
    "k_foot": 1.5,       # swing hip command per unit of (foot position error / leg length)
    "k_fd": 0.12,        # s, swing foot velocity lead in the placement loop
    "knee_swing": -0.8,  # rad, swing knee flexion
    "k_yaw": 3.0,        # hip_z target per rad of heading error
    "x_off": 0.0,        # m, constant forward offset of the foot target
    "arm_amp": 0.0,      # rad, arm swing amplitude (right arm backward during right-leg swing)
}
PARAM_BOUNDS = {
    "v_des": (0.15, 0.7), "T_step": (0.25, 0.45), "k_raibert": (0.05, 0.4), "k_lat": (0.05, 0.4),
    "width": (0.18, 0.4), "kp_att": (1.0, 4.0), "k_foot": (0.5, 3.0),
    "k_fd": (0.02, 0.3), "knee_swing": (-1.3, -0.4), "k_yaw": (0.0, 4.0), "x_off": (-0.1, 0.1), "arm_amp": (-1.2, 1.2),
}
FIXED = {               # joint-level gains, chosen for stability at the 15 ms control period
    "lean_des": 0.03,    # desired forward lean of the pelvis (up-vector x component)
    "kd_att": 0.10,      # stance hip gain on pelvis angular velocity (ctrl per rad/s)
    "kp_leg": 0.8,       # hip joint position gain (ctrl per rad)
    "kd_leg": 0.015,     # hip joint rate gain
    "kp_knee": 0.5,      # swing knee position gain (light shin: small damping)
    "kd_knee": 0.01,
    "knee_stance": -0.05,  # rad, stance knee target (almost straight)
    "kp_knee_st": 1.0,   # stance knee gain
    "kp_swing": 1.0,     # swing hip position gain
    "clear_frac": 0.4,   # fraction of swing after which the knee re-extends
    "kp_abd": 2.5, "kd_abd": 0.01,
    "arm_tau": 0.08,     # s, arm swing smoothing time constant
    "kp_arm": 0.5,       # shoulder/elbow position gain
    "ramp": 1.5,         # s, speed demand ramp duration
    "yaw_scale": 1.0,    # 1: scale the speed demand by cos(yaw); 0: full demand while |yaw| < yaw_stop
    "yaw_stop": 1.3,     # rad
    "T_ds": 0.0,         # s, double-support window after touchdown (yaw correction acts here)
    "kp_hz_ds": 1.5,     # hip_z gain during double support
    "hz_clip": 0.6,      # rad, limit of the hip_z yaw-correction target
    "kp_hz": 0.8,        # hip_z position gain
}

LEG_LEN = 0.75   # m, hip to foot centre
HIP_Y_OFF = 0.1  # m, hip lateral offset from the pelvis centre
R, L = 0, 1
# motor indices
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
HX = (3, 7); HZ = (4, 8); HY = (5, 9); KNEE = (6, 10)
ARMS = slice(11, 17)

_state = {}


def _reset():
    _state.update(stance=R, t_phase=0.0, started=False, hy_lo=[0.0, 0.0], hx_lo=[0.0, 0.0], v_filt=np.zeros(2),
                  feet_prev=None, feet_vel=np.zeros((2, 3)), arm_phase=0.0, sw_contact_n=0, wrong_foot_n=0)


def _pd(q, qd, q_des, kp, kd):
    return kp * (q_des - q) - kd * qd


def controller(obs: dict) -> np.ndarray:
    P = {**FIXED, **PARAMS}
    if obs["step"] == 0 or not _state:
        _reset()
    dt = obs["dt"]
    t = obs["time"]
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    up = obs["up_vector"]
    w = obs["root_angular_velocity"]
    v = obs["com_velocity_xy"]
    _state["v_filt"] = 0.7 * _state["v_filt"] + 0.3 * v
    # everything horizontal is expressed in the yaw (heading) frame: the hips act in the body frame and the
    # point feet let the robot spin freely about the vertical
    yaw = obs["yaw"]
    cy, sy = np.cos(yaw), np.sin(yaw)
    Ryaw = np.array([[cy, sy], [-sy, cy]])
    vf = Ryaw @ _state["v_filt"]
    up = np.concatenate([Ryaw @ up[:2], up[2:]])
    w = np.concatenate([Ryaw @ w[:2], w[2:]])
    contacts = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))
    feet = np.stack([obs["foot_xyz_rel"], obs["foot_left_xyz_rel"]])
    if _state["feet_prev"] is not None:
        _state["feet_vel"] = 0.5 * _state["feet_vel"] + 0.5 * (feet - _state["feet_prev"]) / dt
    _state["feet_prev"] = feet.copy()
    fvel = _state["feet_vel"]
    u = np.zeros(17)

    # ---- torso: abdomen and arms to neutral ----
    u[ABD_Y] = _pd(q[ABD_Y], qd[ABD_Y], 0.0, P["kp_abd"], P["kd_abd"])
    u[ABD_Z] = _pd(q[ABD_Z], qd[ABD_Z], 0.0, P["kp_abd"], P["kd_abd"])
    u[ABD_X] = _pd(q[ABD_X], qd[ABD_X], 0.0, P["kp_abd"], P["kd_abd"])
    # arms: counter-swing to the legs (left arm forward with the right leg) to cancel the yaw momentum of
    # the swinging leg. A rotation th about the pelvis y axis maps onto shoulder1/shoulder2 as
    # (0.41 th, -0.71 th) for both arms (mirror-symmetric axes); th > 0 moves the right arm backward.
    if _state["started"]:
        _state["arm_phase"] += (dt / P["arm_tau"]) * ((1.0 if _state["stance"] == L else -1.0) - _state["arm_phase"])
    th = P["arm_amp"] * _state["arm_phase"]
    arm_tgt = np.array([0.41 * th, -0.71 * th, 0.0, 0.41 * th, -0.71 * th, 0.0])
    u[ARMS] = _pd(q[ARMS], qd[ARMS], arm_tgt, P["kp_arm"], 0.02)

    # ---- settle phase: symmetric crouch until the feet have landed ----
    if not _state["started"]:
        if t > 0.25 and (contacts[0] or contacts[1]):
            _state["started"] = True
            _state["stance"] = L
            _state["t_phase"] = 0.0
            _state["hy_lo"] = [q[HY[0]], q[HY[1]]]
            _state["hx_lo"] = [q[HX[0]], q[HX[1]]]
        for s in (R, L):
            u[HY[s]] = _pd(q[HY[s]], qd[HY[s]], -0.25 if s == L else 0.1, P["kp_leg"], P["kd_leg"])
            u[KNEE[s]] = _pd(q[KNEE[s]], qd[KNEE[s]], P["knee_stance"], P["kp_knee"], P["kd_knee"])
            u[HX[s]] = _pd(q[HX[s]], qd[HX[s]], -0.07, P["kp_leg"], P["kd_leg"])
            u[HZ[s]] = _pd(q[HZ[s]], qd[HZ[s]], 0.0, P["kp_leg"], P["kd_leg"])
        return np.clip(u, -0.4, 0.4)

    # ---- gait phase bookkeeping ----
    st = _state["stance"]
    sw = 1 - st
    _state["t_phase"] += dt
    tp = _state["t_phase"]
    T = P["T_step"]
    # touchdown of the swing foot must be confirmed on two consecutive steps; if the stance foot has left the
    # ground while the swing foot carries the robot (a bounce), hand over regardless of the timer
    _state["sw_contact_n"] = _state["sw_contact_n"] + 1 if contacts[sw] else 0
    _state["wrong_foot_n"] = _state["wrong_foot_n"] + 1 if (contacts[sw] and not contacts[st]) else 0
    if (tp > 0.6 * T and _state["sw_contact_n"] >= 2) or tp > 1.5 * T or (tp > 0.3 * T and _state["wrong_foot_n"] >= 3):
        st, sw = sw, st
        _state["stance"] = st
        _state["t_phase"] = tp = 0.0
        _state["sw_contact_n"] = 0
        _state["wrong_foot_n"] = 0
        _state["hy_lo"] = [q[HY[0]], q[HY[1]]]
        _state["hx_lo"] = [q[HX[0]], q[HX[1]]]
    phase = min(tp / T, 1.0)

    # ---- desired velocity (ramp) ----
    # walk along the body's forward axis (no lateral demand: the hips can hardly adduct); the yaw
    # correction turns the body back toward world +x, and the demand shrinks while it is turned away
    v_des_w = P["v_des"] * min(1.0, max(0.0, (t - 1.0) / P["ramp"]))
    if P["yaw_scale"] > 0.5:
        v_des = np.array([v_des_w * max(np.cos(yaw), 0.0), 0.0])
    else:
        v_des = np.array([v_des_w if abs(yaw) < P["yaw_stop"] else 0.0, 0.0])

    # ---- double support right after touchdown: both feet planted, hip_z torques turn the pelvis back to heading 0
    # (on a single point foot a hip_z torque only spins the leg) ----
    hz_tgt = float(np.clip(P["k_yaw"] * yaw, -P["hz_clip"], P["hz_clip"]))   # right +, left -
    if tp < P["T_ds"]:
        u[HY[sw]] = _pd(q[HY[sw]], qd[HY[sw]], _state["hy_lo"][sw], P["kp_leg"], P["kd_leg"])
        u[HX[sw]] = _pd(q[HX[sw]], qd[HX[sw]], _state["hx_lo"][sw], P["kp_leg"], P["kd_leg"])
        u[KNEE[sw]] = _pd(q[KNEE[sw]], qd[KNEE[sw]], P["knee_stance"], P["kp_knee_st"], P["kd_knee"])
        u[HZ[sw]] = _pd(q[HZ[sw]], qd[HZ[sw]], hz_tgt if sw == R else -hz_tgt, P["kp_hz_ds"], P["kd_leg"])
        u[HZ[st]] = _pd(q[HZ[st]], qd[HZ[st]], hz_tgt if st == R else -hz_tgt, P["kp_hz_ds"], P["kd_leg"])
        _state["dbg"] = (0, 0, 0, 0, 0, 0)
        return _stance_leg(u, q, qd, up, w, st, P, hz_tgt, skip_hz=True)

    # ---- swing leg: Raibert foot placement, closed on the measured foot position ----
    # desired foot position relative to the mass centre at touchdown (world frame)
    xf = vf[0] * T / 2 + P["k_raibert"] * (vf[0] - v_des[0]) + P["x_off"]
    yf = vf[1] * T / 2 + P["k_lat"] * (vf[1] - v_des[1]) + (-0.5 * P["width"] if sw == R else 0.5 * P["width"])
    xf = float(np.clip(xf, -0.3, 0.45))
    yf = float(np.clip(yf, -0.5, 0.5))
    foot = Ryaw @ feet[sw][:2]
    fv = Ryaw @ fvel[sw][:2]
    # the foot moves ~LEG_LEN * dtheta per rad of hip angle: forward = negative hip_y,
    # outward (+y for left, -y for right) = negative hip_x
    ex = xf - foot[0] - P["k_fd"] * fv[0]
    ey = yf - foot[1] - P["k_fd"] * fv[1]
    hy_cmd = q[HY[sw]] - P["k_foot"] * ex / LEG_LEN
    hx_cmd = q[HX[sw]] + (1.0 if sw == R else -1.0) * P["k_foot"] * ey / LEG_LEN
    hx_cmd = np.clip(hx_cmd, -0.45, 0.1)
    hy_cmd = np.clip(hy_cmd, -1.3, 0.3)
    if phase < P["clear_frac"]:
        knee_cmd = P["knee_swing"]
    else:
        b = min(1.0, (phase - P["clear_frac"]) / (1.0 - P["clear_frac"]))
        knee_cmd = (1 - b) * P["knee_swing"] + b * P["knee_stance"]
    _state['dbg'] = (xf, yf, foot[0], foot[1], hy_cmd, hx_cmd)
    u[HY[sw]] = _pd(q[HY[sw]], qd[HY[sw]], hy_cmd, P["kp_swing"], P["kd_leg"])
    u[HX[sw]] = _pd(q[HX[sw]], qd[HX[sw]], hx_cmd, P["kp_swing"], P["kd_leg"])
    u[KNEE[sw]] = _pd(q[KNEE[sw]], qd[KNEE[sw]], knee_cmd, P["kp_knee"], P["kd_knee"])
    u[HZ[sw]] = _pd(q[HZ[sw]], qd[HZ[sw]], hz_tgt if sw == R else -hz_tgt, P["kp_hz"], P["kd_leg"])

    return _stance_leg(u, q, qd, up, w, st, P, hz_tgt)


def _stance_leg(u, q, qd, up, w, st, P, hz_tgt, skip_hz=False):
    """Stance leg: pelvis attitude through the hip, stiff knee, yaw correction through hip_z."""
    # ---- stance leg: torso attitude through the hip, stiff knee ----
    # forward lean rate = w_y (world); lean toward +y rate = -w_x
    # the hip acts on the pelvis: pelvis lean = torso lean + abdomen angles
    lean_x, lean_y = up[0] + q[ABD_Y], up[1] - q[ABD_X]
    u[HY[st]] = P["kp_att"] * (lean_x - P["lean_des"]) + P["kd_att"] * w[1] \
        + 0.3 * _pd(q[HY[st]], qd[HY[st]], 0.0, P["kp_leg"], P["kd_leg"])
    # positive stance hip_x torque leans the torso toward the swing side (+y for right stance)
    sgn = 1.0 if st == R else -1.0
    u[HX[st]] = -sgn * (P["kp_att"] * lean_y + P["kd_att"] * (-w[0])) \
        + 0.3 * _pd(q[HX[st]], qd[HX[st]], -0.07, P["kp_leg"], P["kd_leg"])
    u[KNEE[st]] = _pd(q[KNEE[st]], qd[KNEE[st]], P["knee_stance"], P["kp_knee_st"], P["kd_knee"])
    if not skip_hz:
        u[HZ[st]] = _pd(q[HZ[st]], qd[HZ[st]], hz_tgt if st == R else -hz_tgt, P["kp_hz"], P["kd_leg"])

    return np.clip(u, -0.4, 0.4)
