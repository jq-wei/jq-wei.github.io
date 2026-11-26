"""Walking controller for the MuJoCo Humanoid on rough, tilted ground (hand-designed, no learning).

A clocked stepping controller of the Raibert / SIMBICON family with capture-point foot placement.
The feet are balls and there are no ankles, so balance is kept by where and when the feet land.

1. Gait clock. Single support lasts about ``T`` s and ends when the swing foot touches the ground
   after phase ``PHI_MIN`` (at ``PHI_MAX`` at the latest); a short double support ``T_DS`` follows and
   the legs swap. The robot starts 10 cm in the air: it lands for ``T_INIT`` s on both legs, then
   takes a first half step of ``T_FIRST`` s.

2. Foot placement (the balance law), in world axes (the goal is +x). The swing foot is steered to
   ``foot - com = v_td / omega + offset``: the capture point of a linear inverted pendulum
   (omega^2 = g / height), with the mass-centre velocity ``v_td`` extrapolated to touchdown from
   the present offset to the stance foot (``PRED``). Forward offset: landing short of the capture
   point by ``BX_GAIN * V_DES * T / (exp(omega T) - 1)`` makes the robot walk at ``V_DES``; an integral
   term on the speed error (``KI_V``, limit ``BXI_MAX``) removes the bias of the swing tracking.
   Lateral offset: ``BY`` to the outside (step width), never closer than ``YMIN`` to the mass centre.
   (``KV``, ``K_SL``, ``K_VYI``, ``K_SLY``, ``K_YX``: optional extra placement terms; 0 = off.)

3. Swing leg. The foot path is defined relative to the mass centre: forward and lateral blends
   finished by phases ``PHI_X`` and ``PHI_Y``; height ``H`` at phase ``PHI_UP``, back to the expected
   ground at ``PHI_TD`` (``DIG`` below it), then descending at ``V_DOWN`` until contact. The expected
   ground is a plane fitted to the last ``N_HOLDS`` footholds (``SLOPE``): all the controller knows about the
   terrain, learned through the legs. Planar two-link inverse kinematics about the hip (its position
   recovered each step from the observed foot, which closes the loop in task space) gives hip and
   knee targets, tracked by PD on the world-frame thigh angles (``KP_HY/KD_HY`` pitch, ``KP_HX/KD_HX``
   roll) and on the knee (``KP_KN/KD_KN``), with target-rate feed-forward (``FF``) and cancellation of
   the knee torque's reaction on the thigh (``FF_K``). Hip twist is held at zero (``KP_HZ/KD_HZ``).

4. Stance leg. Knee: PD plus gravity feed-forward (``KP_KS/KD_KS``, ``FF_KS``) around the flexion
   ``K_ST``; the target relaxes from the touchdown angle at ``KS_RATE``, yields at ``Y_RATE`` while the
   swing foot is late, and on downhill ground sinks through the step by ``K_DH`` x the estimated slope
   so that the lower foothold ahead can be reached (without it the robot runs away downhill).
   Hip: not position controlled. Its torques hold the pelvis pitch and roll (``KP_T/KD_T``, lean
   ``PITCH``; ``KP_R/KD_R``) and cancel the reaction of the swing hip (share ``C_SW``), so the swing leg
   can be driven stiffly without shaking the light pelvis.

5. Trunk and heading. Abdomen pitch and roll hold the torso upright in the world frame
   (``KP_AB/KD_AB``, ``KP_AX/KD_AX``). Ball feet transmit no yaw torque, and the yaw inertia is small
   against the reaction of a swinging leg, so: the arms are stretched sideways (``ARMS_OUT``,
   ``KP_ARM/KD_ARM``); the abdomen twist holds the pelvis heading against the upper body
   (``K_PZ/K_DZ``, joint centring ``KP_AZ/KD_AZ``); and a differential fore-aft hip torque of the
   stance legs turns the body back to +x (``K_YP/K_YD``; in double support, and with share ``K_YS``
   on the single stance hip). ``K_YAW`` (stance hip twist) is off.

Units: ``KP_*`` N m/rad, ``KD_*`` N m s/rad. The PD gains respect the 15 ms control period given the
low joint-space inertia of the floating body (checked against the mass matrix). The numbers were
set with a derivative-free search (hill climbing on simulated rollouts) and one-at-a-time scans.
"""

import math

import numpy as np

PARAMS = {
    "T": 0.42,          # s, nominal swing duration
    "T_INIT": 0.22,     # s, initial double-stance phase (the robot drops 10 cm at the start)
    "T_FIRST": 0.222,    # s, duration of the first (half) step
    "PHI_MIN": 0.5,     # earliest phase at which a touchdown ends the step
    "PHI_MAX": 1.6,     # latest phase: swap legs even without contact
    "PHI_TD": 0.8446,     # phase at which the swing foot is back at ground level
    "PHI_UP": 0.3,     # phase of the highest point of the swing foot
    "PHI_X": 0.7,      # phase by which the forward target is reached
    "PHI_Y": 0.6856,      # phase by which the lateral target is reached
    "H": 0.0504,          # m, swing foot clearance
    "DIG": 0.02,         # m, depth below the expected ground the swing foot aims at (earlier, firmer touchdown)
    "V_DOWN": 0.9,      # m/s, descent speed of the swing foot when the ground comes late
    "Y_RATE": 1.7167,      # rad/s, stance knee yield rate when the swing foot is late
    "K_ST": 0.2981,       # rad, stance knee flexion
    "K_LAND_MAX": 1.2,  # rad, largest stance knee flexion target taken over from the touchdown angle
    "YIELD_MAX": 0.9,   # rad, largest yield of the stance knee
    "KS_RATE": 0.5615,     # rad/s, rate at which the stance knee target returns to K_ST after touchdown
    "PITCH": 0.0,      # rad, torso / pelvis forward lean
    "V_DES": 0.5388,      # m/s, target speed
    "BX_GAIN": 0.8005,     # scale of the LIPM offset that produces V_DES
    "KV": 0.0,         # s, extra forward placement per m/s of speed error
    "BY": 0.05,        # m, lateral capture-point offset (step width)
    "PRED": 0.7,        # fraction of the remaining swing time used in the LIPM prediction
    "KP_HY": 583.8655, "KD_HY": 15.802,   # swing thigh pitch, world frame (reaction taken by the stance hip)
    "KP_KN": 77.1016, "KD_KN": 2.8099,     # knee (swing)
    "KP_KS": 164.8135, "KD_KS": 3.4045,    # knee (stance)
    "FF_KS": 80.0,      # N m, stance knee gravity feed-forward per unit sin(knee flexion)
    "KP_HX": 471.5668, "KD_HX": 20.0,   # swing thigh roll, world frame
    "KP_HZ": 20.0, "KD_HZ": 0.3,     # hip twist
    "KP_T": 227.0661, "KD_T": 5.6324,      # pelvis pitch through the stance hip
    "KP_R": 321.5967, "KD_R": 8.6041,     # pelvis roll through the stance hip
    "KP_AB": 132.7503, "KD_AB": 5.0,    # torso pitch (world frame) through the abdomen
    "KP_AX": 182.4864, "KD_AX": 10.1813,   # torso roll (world frame) through the abdomen
    "KP_AZ": 11.2053, "KD_AZ": 0.5,     # abdomen twist: centring of the joint
    "K_PZ": 38.9926, "K_DZ": 2.7324,       # abdomen twist: pelvis heading held against the upper body
    "ARMS_OUT": 1.0,    # 1: arms stretched sideways, 0: neutral
    "KP_ARM": 25.0, "KD_ARM": 0.8,    # arms
    "FF": 0.5678,          # weight of the target-rate feed-forward in the swing PD
    "T_DS": 0.0501,       # s, double-support phase after each touchdown
    "K_YP": 180.0,       # N m/rad, differential fore-aft stance-hip torque against the heading error
    "K_YD": 23.8245,       # N m s/rad, the same against the yaw rate
    "N_HOLDS": 6,       # number of recent footholds in the ground-plane fit
    "SLOPE": 0.5538,       # weight of the estimated ground slope in the swing-foot height target
    "K_YS": 1.0,        # share of the heading torque applied by the single stance hip (it disturbs the pitch)
    "K_YX": 0.0,       # m/rad, differential step length against the heading error
    "K_DH": 4.5,        # rad per unit slope: extra stance knee flexion through the step on downhill ground
    "K_VYI": 0.0,       # s, lateral foothold shift per m/s of average lateral drift
    "K_SLY": 0.0,       # m per unit cross slope: lateral foothold shift towards the downhill side
    "K_SL": 0.0,        # m per unit slope: extra forward placement on estimated downhill ground (braking)
    "YMIN": 0.05,       # m, smallest lateral distance of the landing foot from the mass centre
    "BXI_MAX": 0.08,    # m, upper limit of the integral placement offset
    "KI_V": 0.03,       # m per (m/s) per step: integral action of the placement on the speed error
    "FF_K": 1.0456,        # share of the swing knee torque reaction cancelled at the swing hip
    "K_YAW": 0.0,       # stance hip twist per rad of heading error
    "C_SW": 1.0,        # share of the swing hip reaction cancelled by the stance hip
}

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], dtype=float)
L1, L2 = 0.383, 0.370          # hip-knee and knee-foot-centre lengths [m]
G = 9.81
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
HX, HZ, HY, KN = 3, 4, 5, 6    # right leg; the left leg is +4
RIGHT, LEFT = 0, 1
ARM_POSE = np.radians([-50.0, -30.0, -90.0, 50.0, 30.0, -90.0])   # arms stretched sideways: more yaw inertia


class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode = "init"
        self.swing = LEFT
        self.t_ds = 0.0
        self.holds = []           # recent footholds (world x, y, z of the foot centre) for the ground-slope estimate
        self.gx = 0.0             # estimated ground slope dz/dx, dz/dy
        self.gy = 0.0
        self.vy_avg = 0.0         # slow average of the lateral velocity (drift)
        self.t0 = 0.0
        self.T = 0.3
        self.x0 = 0.0             # swing foot position relative to the mass centre at lift-off (world axes)
        self.y0 = 0.1
        self.dz0 = 0.0            # height of the swing foot above the stance foot at lift-off
        self.k_lead = 0.3
        self.k_st = 0.3           # stance knee flexion target (relaxes to K_ST after touchdown)
        self.bx_i = 0.0           # integral placement offset (speed error)
        self.v_sum, self.v_n = 0.0, 0
        self.yield_k = 0.0        # extra stance knee flexion while the swing foot looks for the ground
        self.rate = (0.0, 0.0, 0.0)
        self.prev_tgt = None      # previous swing joint targets (hy, kn, hx) for the velocity feed-forward
        self.last_step = -1


_S = _State()
DEBUG = {}   # last internal values (phase, targets), for inspection only


def _ik(x, z):
    """Planar two-link IK: foot centre at (x forward, z up) from the hip -> thigh flexion A, knee flexion K."""
    r = math.hypot(x, z)
    rmax, rmin = L1 + L2 - 0.003, 0.40
    if r > rmax:
        x, z, r = x * rmax / r, z * rmax / r, rmax
    elif r < rmin:
        x, z, r = x * rmin / max(r, 1e-6), z * rmin / max(r, 1e-6), rmin
    ck = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)
    k = math.acos(max(-1.0, min(1.0, ck)))
    a = math.atan2(x, -z) + math.atan2(L2 * math.sin(k), L1 + L2 * math.cos(k))
    return a, k


def _smooth(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    st = _S
    step = int(obs["step"])
    if step == 0 or step < st.last_step:
        st.reset()
    st.last_step = step
    t = float(obs["time"])
    dt = float(obs["dt"])
    q = np.asarray(obs["joint_angles"], dtype=float)
    dq = np.asarray(obs["joint_rates"], dtype=float)
    roll, pitch, yaw = float(obs["roll"]), float(obs["pitch"]), float(obs["yaw"])
    w = obs["root_angular_velocity"]
    cy, sy = math.cos(yaw), math.sin(yaw)
    w_roll = cy * w[0] + sy * w[1]           # torso roll / pitch rates in its heading frame
    w_pitch = -sy * w[0] + cy * w[1]
    v = obs["com_velocity_xy"]
    vx, vy = float(v[0]), float(v[1])        # world axes: +x is the goal direction
    feet = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])   # relative to the mass centre, world axes
    contact = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))

    pp = pitch + q[ABD_Y]          # pelvis pitch (small-angle composition)
    pr = roll + q[ABD_X]           # pelvis roll
    yaw_p = yaw + q[ABD_Z]         # pelvis heading
    cp, sp_ = math.cos(yaw_p), math.sin(yaw_p)

    tau = np.zeros(17)

    # ---- trunk and arms: torso upright in the world frame through the abdomen, arms stretched sideways ----
    # (terms like 10.0 * q[...] / 20.0 * q[...] below cancel the passive joint springs of the model)
    wp_pelvis = w_pitch + dq[ABD_Y]
    wr_pelvis = w_roll + dq[ABD_X]
    tau[ABD_Y] = P["KP_AB"] * (pitch - P["PITCH"]) + P["KD_AB"] * w_pitch + 10.0 * q[ABD_Y]
    tau[ABD_X] = P["KP_AX"] * roll + P["KD_AX"] * w_roll + 10.0 * q[ABD_X]
    # abdomen twist: the upper body is the reaction mass that holds the pelvis heading against the
    # swing leg's yaw reaction (the feet are balls: no yaw torque from the ground in single support)
    wz_p = float(w[2]) + dq[ABD_Z]
    tau[ABD_Z] = _clip(-P["K_PZ"] * _clip(yaw_p, -0.6, 0.6) - P["K_DZ"] * wz_p - P["KP_AZ"] * q[ABD_Z] - P["KD_AZ"] * dq[ABD_Z], -40.0, 40.0)
    tau[11:17] = P["KP_ARM"] * (ARM_POSE * P["ARMS_OUT"] - q[11:17]) - P["KD_ARM"] * dq[11:17]

    # pelvis attitude torques (as hip-joint torques on the thigh side), shared by the stance hip(s)
    t_pitch = P["KP_T"] * (pp - P["PITCH"]) + P["KD_T"] * wp_pelvis
    t_roll = P["KP_R"] * pr + P["KD_R"] * wr_pelvis

    # heading: differential fore-aft ground force of the two stance legs (right hip +u, left hip -u)
    yaw_b = yaw + 0.4 * q[ABD_Z]              # body heading: torso and pelvis weighted by their yaw inertia
    u_yaw = -_clip(P["K_YP"] * _clip(yaw_b, -0.5, 0.5) + P["K_YD"] * (float(w[2]) + 0.4 * dq[ABD_Z]), -60.0, 60.0)

    def stance_knee(o, k_tgt):
        k = -q[KN + o]
        return P["KP_KS"] * (k - k_tgt) - P["KD_KS"] * dq[KN + o] + P["FF_KS"] * math.sin(min(k, 1.5))

    def start_step(new_swing, T):
        st.swing = new_swing
        st.T = T
        st.t0 = t
        f, g = feet[new_swing], feet[1 - new_swing]
        st.x0, st.y0 = float(f[0]), float(f[1])
        st.dz0 = float(f[2] - g[2])
        st.k_st = _clip(-q[KN + 4 * (1 - new_swing)], P["K_ST"], P["K_LAND_MAX"])
        st.yield_k = 0.0
        st.prev_tgt = None
        if st.v_n > 0:
            st.bx_i = _clip(st.bx_i + P["KI_V"] * (st.v_sum / st.v_n - P["V_DES"]), -0.12, P["BXI_MAX"])
        st.v_sum, st.v_n = 0.0, 0

    if st.mode == "init":
        for leg in (RIGHT, LEFT):
            o = 4 * leg
            sgn = 1.0 if leg == RIGHT else -1.0
            tau[KN + o] = stance_knee(o, P["K_ST"]) - 0.5 * P["FF_KS"] * math.sin(min(-q[KN + o], 1.5))
            tau[HY + o] = 0.5 * t_pitch + 20.0 * q[HY + o] + sgn * u_yaw
            tau[HX + o] = 100.0 * (0.0 - q[HX + o]) - 4.0 * dq[HX + o] + 0.5 * sgn * t_roll
            tau[HZ + o] = -P["KP_HZ"] * q[HZ + o] - P["KD_HZ"] * dq[HZ + o]
        if t >= P["T_INIT"]:
            st.mode = "walk"
            start_step(LEFT if vy >= 0.0 else RIGHT, P["T_FIRST"])
        return np.clip(tau / GEAR, -0.4, 0.4)

    # ---- step clock ----
    phi = (t - st.t0) / st.T
    if st.mode == "walk" and ((phi > P["PHI_MIN"] and contact[st.swing]) or phi > P["PHI_MAX"]):
        st.mode = "ds"
        st.t_ds = t
        c = obs["com_xyz"]
        fl = feet[st.swing]
        st.holds.append((float(c[0] + fl[0]), float(c[1] + fl[1]), float(c[2] + fl[2])))
        if len(st.holds) > int(P["N_HOLDS"]):
            st.holds.pop(0)
        if len(st.holds) >= 3:   # ridge-regularised plane through the last footholds
            hp = np.array(st.holds)
            d = hp - hp.mean(axis=0)
            a = d[:, :2].T @ d[:, :2] + 0.004 * np.eye(2)
            g = np.linalg.solve(a, d[:, :2].T @ d[:, 2])
            st.gx, st.gy = _clip(float(g[0]), -0.15, 0.15), _clip(float(g[1]), -0.10, 0.10)
        st.k_lead = _clip(-q[KN + 4 * st.swing], P["K_ST"], P["K_LAND_MAX"])
    if st.mode == "ds":
        if t - st.t_ds >= P["T_DS"] - 1e-9:
            st.mode = "walk"
            start_step(1 - st.swing, P["T"])
            phi = 0.0
        else:
            # double support: both legs carry; hips share the pelvis attitude and steer the heading
            lead, trail = st.swing, 1 - st.swing
            st.k_lead = max(P["K_ST"], st.k_lead - P["KS_RATE"] * dt)
            for leg, k_tgt in ((lead, st.k_lead), (trail, st.k_st + st.yield_k + P["K_DH"] * max(0.0, -st.gx))):
                o = 4 * leg
                sgn = 1.0 if leg == RIGHT else -1.0
                tau[KN + o] = stance_knee(o, k_tgt)
                tau[HY + o] = 0.5 * t_pitch + 20.0 * q[HY + o] + sgn * u_yaw
                tau[HX + o] = 0.5 * sgn * t_roll + 10.0 * q[HX + o] - 3.0 * dq[HX + o]
                tau[HZ + o] = -P["KP_HZ"] * q[HZ + o] - P["KD_HZ"] * dq[HZ + o]
            DEBUG.update(phi=1.0, sw=st.swing, vf=vx, vl=vy)
            return np.clip(tau / GEAR, -0.4, 0.4)
    st.v_sum += vx
    st.v_n += 1
    sw, sp = st.swing, 1 - st.swing
    T = st.T
    osw, osp = 4 * sw, 4 * sp
    side = 1.0 if sw == LEFT else -1.0           # lateral sign of the swing foot (+y is left)
    sgn_sw = 1.0 if sw == RIGHT else -1.0        # hip_x axis sign
    sgn_sp = -sgn_sw

    # ---- foot placement: LIPM capture point predicted at touchdown (world axes) ----
    fs = feet[sp]
    zc = max(0.6, -float(fs[2]))
    om = math.sqrt(G / zc)
    t_r = max(0.0, P["PHI_TD"] - phi) * T * P["PRED"]
    sh, ch = math.sinh(om * t_r), math.cosh(om * t_r)
    vx_td = om * (-fs[0]) * sh + vx * ch
    vy_td = om * (-fs[1]) * sh + vy * ch
    e_wt = math.exp(om * P["T"])
    bx = -P["BX_GAIN"] * P["V_DES"] * P["T"] / (e_wt - 1.0) + P["KV"] * (vx - P["V_DES"]) - P["K_SL"] * min(st.gx, 0.0)
    x_t = _clip(vx_td / om + bx + st.bx_i + side * P["K_YX"] * _clip(yaw_p, -0.4, 0.4), -0.30, 0.40)
    st.vy_avg += (vy - st.vy_avg) * dt / 1.0          # one-second average: the drift, not the sway
    y_shift = P["K_VYI"] * st.vy_avg - P["K_SLY"] * st.gy
    y_t = side * _clip(side * (vy_td / om + side * P["BY"] + P["KV"] * vy + y_shift), P["YMIN"], 0.36)

    # ---- swing foot trajectory (relative to the mass centre, world axes) ----
    x_c = st.x0 + (x_t - st.x0) * _smooth(phi / P["PHI_X"])
    y_c = st.y0 + (y_t - st.y0) * _smooth(phi / P["PHI_Y"])
    if phi <= P["PHI_UP"]:
        h = P["H"] * math.sin(0.5 * math.pi * phi / P["PHI_UP"])
    elif phi <= P["PHI_TD"]:
        h = (P["H"] + P["DIG"]) * 0.5 * (1.0 + math.cos(math.pi * (phi - P["PHI_UP"]) / (P["PHI_TD"] - P["PHI_UP"]))) - P["DIG"]
    else:
        h = max(-0.15, -P["DIG"] - P["V_DOWN"] * (phi - P["PHI_TD"]) * T)
    ux = _smooth(phi / P["PHI_X"])
    dz_t = P["SLOPE"] * (st.gx * (x_t - float(fs[0])) + st.gy * (y_t - float(fs[1])))   # ground at the target vs stance foot
    z_c = float(fs[2]) + st.dz0 * (1.0 - ux) + dz_t * ux + h

    # into the pelvis heading frame, where the leg kinematics are (nearly) planar
    f = feet[sw]
    fpx, fpy = cp * f[0] + sp_ * f[1], -sp_ * f[0] + cp * f[1]
    cpx, cpy = cp * x_c + sp_ * y_c, -sp_ * x_c + cp * y_c
    # hip of the swing leg relative to the mass centre, from the observed foot and the leg's own kinematics
    a_now = -(pp + q[HY + osw])
    k_now = -q[KN + osw]
    fx = L1 * math.sin(a_now) + L2 * math.sin(a_now - k_now)
    fz = -L1 * math.cos(a_now) - L2 * math.cos(a_now - k_now)
    th_now = pr + sgn_sw * q[HX + osw]
    fy = -fz * math.sin(th_now)
    hip = (fpx - fx, fpy - fy, float(f[2]) - fz)
    a_des, k_des = _ik(cpx - hip[0], z_c - hip[2])
    th_des = math.atan2(cpy - hip[1], -(z_c - hip[2]))
    hy_des = _clip(-a_des - pp, -1.8, 0.33)
    kn_des = _clip(-k_des, -2.6, -0.04)
    hx_des = _clip(sgn_sw * (th_des - pr), -0.42, 0.08)
    tgt = (hy_des, kn_des, hx_des)
    if st.prev_tgt is None:
        st.rate = (0.0, 0.0, 0.0)
    else:
        raw = tuple(_clip((a - b) / dt, -6.0, 6.0) for a, b in zip(tgt, st.prev_tgt))
        st.rate = tuple(0.5 * (a + b) for a, b in zip(raw, st.rate))
    rate = tuple(P["FF"] * a for a in st.rate)
    st.prev_tgt = tgt

    # swing PD on the world-frame thigh angles (the joint target already carries the pelvis attitude)
    w_th_p = wp_pelvis + dq[HY + osw]
    w_th_r = sgn_sw * wr_pelvis + dq[HX + osw]
    tau_hy_sw = P["KP_HY"] * (hy_des - q[HY + osw]) + P["KD_HY"] * (rate[0] - w_th_p) + 20.0 * q[HY + osw]
    tau_kn_sw = P["KP_KN"] * (kn_des - q[KN + osw]) + P["KD_KN"] * (rate[1] - dq[KN + osw])
    tau_hx_sw = P["KP_HX"] * (hx_des - q[HX + osw]) + P["KD_HX"] * (rate[2] - w_th_r) + 10.0 * q[HX + osw]
    tau_hy_sw = _clip(tau_hy_sw - P["FF_K"] * _clip(tau_kn_sw, -80.0, 80.0), -120.0, 120.0)
    tau_hx_sw = _clip(tau_hx_sw, -40.0, 40.0)
    # stance hip: pelvis attitude minus the swing hip's reaction; if it saturates, the swing hip gives way
    tau_hy_st = _clip(t_pitch - P["C_SW"] * tau_hy_sw, -120.0, 120.0)
    if P["C_SW"] > 0.5:
        tau_hy_sw = _clip((t_pitch - tau_hy_st) / P["C_SW"], -120.0, 120.0)
    tau[HY + osw] = tau_hy_sw
    tau[KN + osw] = tau_kn_sw
    tau[HX + osw] = tau_hx_sw
    tau[HZ + osw] = -P["KP_HZ"] * q[HZ + osw] - P["KD_HZ"] * dq[HZ + osw]

    # ---- stance leg ----
    st.k_st = max(P["K_ST"], st.k_st - P["KS_RATE"] * dt)
    if phi > P["PHI_TD"] and not contact[sw]:
        st.yield_k = min(P["YIELD_MAX"], st.yield_k + P["Y_RATE"] * dt)
    k_slope = P["K_DH"] * max(0.0, -st.gx) * _smooth(phi)   # downhill: sink onto the lower ground ahead
    tau[KN + osp] = stance_knee(osp, st.k_st + st.yield_k + k_slope)
    tau[HY + osp] = _clip(tau_hy_st + (1.0 if sp == RIGHT else -1.0) * P["K_YS"] * u_yaw, -120.0, 120.0)
    tau[HX + osp] = sgn_sp * t_roll + P["C_SW"] * tau_hx_sw
    hz_des = (1.0 if sp == RIGHT else -1.0) * P["K_YAW"] * _clip(yaw_p, -0.3, 0.3)
    tau[HZ + osp] = P["KP_HZ"] * (hz_des - q[HZ + osp]) - P["KD_HZ"] * dq[HZ + osp]

    DEBUG.update(phi=phi, sw=sw, x_t=x_t, y_t=y_t, vf=vx, vl=vy, hip=hip, x_c=x_c, y_c=y_c, z_c=z_c)
    return np.clip(tau / GEAR, -0.4, 0.4)
