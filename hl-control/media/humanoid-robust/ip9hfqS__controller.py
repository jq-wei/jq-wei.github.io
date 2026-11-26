"""SIMBICON-style walking controller for the MuJoCo Humanoid (no learning, hand-structured).

Control law
-----------
A two-state machine alternates the stance leg (right / left).  A state lasts at least
`Tmin` and ends when the swing foot touches down after `Tmin`, or at `Tmax`.
All joints are PD servos, tau = kp*(q_des - q) - kd*qdot, sent as tau/gear (clipped).

* Stance leg: knee held near `kn_st`; the hip pitch/roll torque is a feedback on the torso
  pitch/roll (`tp, tpd, tr, trd`) so the stance leg keeps the torso upright over a point foot.
* Swing leg: hip pitch target `hy_sw + cd*d + cv*v` (d = COM distance ahead of the stance foot,
  v = forward COM speed error: SIMBICON foot placement), knee flexed `kn_sw` then extended
  `kn_sw2` late in the swing; hip roll target `hx_sw + cdl*dy + cvl*vy` places the foot laterally.
* Abdomen: PD to a forward-lean offset `ay0`; yaw is held with hip_z/abdomen_z (`gy`).
* Arms: weak PD to a fixed posture.
All numbers live in PARAMS.
"""
import numpy as np

PARAMS = {
    'Tmin': 0.16073,
    'Tmax': 0.30887,
    'kp': 159.15,
    'kd': 5.0886,
    'kp_hx': 110.14,
    'vtarget': 0.45001,
    'hy_sw': -0.43539,
    'cd': -0.87563,
    'cv': -0.54955,
    'kn_sw': -0.55544,
    'kn_sw2': -0.47409,
    'swsplit': 0.30073,
    'kn_st': -0.059455,
    'hy_st': -0.084643,
    'tp': 216.86,
    'tpd': 32.173,
    'tr': -43.349,
    'trd': 8.257,
    'hx_sw': -0.12403,
    'cdl': 1.9367,
    'cvl': 0.56079,
    'hx_st': -0.055426,
    'ay0': 0.022785,
    'ap': -70.72,
    'apd': 0.98311,
    'ar': 70.847,
    'ard': 19.948,
    'gy': -0.3193,
    'gyd': -0.70809,
    'arm': -0.23642,
}

STATE = {"side": 0, "t0": 0.0, "last_t": -1.0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = obs["time"]
    if t <= STATE["last_t"] or t < 1e-9:
        STATE.update(side=0, t0=t)
    STATE["last_t"] = t
    q, qd = obs["joint_angles"], obs["joint_rates"]
    yaw, pitch, roll = obs["yaw"], obs["pitch"], obs["roll"]
    w = obs["root_angular_velocity"]
    c, s = np.cos(yaw), np.sin(yaw)
    def heading(v):
        return np.array([c * v[0] + s * v[1], -s * v[0] + c * v[1]])
    vcom = heading(obs["com_velocity_xy"])
    # stance side: 0 -> right stance (left swings), 1 -> left stance
    side = STATE["side"]
    tin = t - STATE["t0"]
    sw_contact = obs["foot_left_contact"] if side == 0 else obs["foot_contact"]
    if tin >= P["Tmax"] or (tin >= P["Tmin"] and sw_contact):
        side = STATE["side"] = 1 - side
        STATE["t0"] = t
        tin = 0.0
    ph = min(1.0, tin / P["Tmax"])
    st_idx = (3, 4, 5, 6) if side == 0 else (7, 8, 9, 10)
    sw_idx = (7, 8, 9, 10) if side == 0 else (3, 4, 5, 6)
    m_st = 1.0 if side == 0 else -1.0
    m_sw = -m_st
    foot_st = obs["foot_xyz_rel"] if side == 0 else obs["foot_left_xyz_rel"]
    f = heading(foot_st)       # stance foot relative to COM
    d, dy = -f[0], -f[1]       # COM relative to stance foot
    des = np.zeros(17)
    kp = np.full(17, P["kp"]); kd = np.full(17, P["kd"])
    kp[11:] = 8.0; kd[11:] = 1.0
    extra = np.zeros(17)       # direct torques [N m]
    # torso
    des[0] = P["ay0"]
    extra[0] = P["ap"] * (pitch) + P["apd"] * w[1]
    extra[2] = P["ar"] * roll + P["ard"] * w[0]
    des[1] = -(P["gy"] * yaw + P["gyd"] * w[2])
    # stance leg
    ix, iz, iy, ik = st_idx
    des[iy] = P["hy_st"]; kp[iy] = 0.0
    extra[iy] = P["tp"] * pitch + P["tpd"] * w[1]
    des[ik] = P["kn_st"]
    des[ix] = P["hx_st"]; kp[ix] = P["kp_hx"]
    extra[ix] = m_st * (P["tr"] * roll + P["trd"] * w[0])
    des[iz] = 0.0
    # swing leg
    ix, iz, iy, ik = sw_idx
    des[iy] = P["hy_sw"] + P["cd"] * d + P["cv"] * (vcom[0] - P["vtarget"])
    des[ik] = P["kn_sw"] if ph < P["swsplit"] else P["kn_sw2"]
    des[ix] = P["hx_sw"] + m_sw * (P["cdl"] * dy + P["cvl"] * vcom[1])
    kp[ix] = P["kp_hx"]
    des[iz] = 0.0
    des[11:] = P["arm"]
    tau = kp * (des - q) - kd * qd + extra
    return np.clip(tau / obs["gear"], -0.4, 0.4)
