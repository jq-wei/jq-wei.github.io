"""Walking controller for the MuJoCo Humanoid: finite-state stepping + Raibert foot placement.

Control law (torques computed in N m, divided by the gear, clipped to +-0.4):
  * Gait state machine: one leg is stance, the other swings for about T; roles swap at
    swing-foot touchdown (after 0.5 T) or after 1.5 T. (An optional double-support phase
    of length Tds is supported; the tuned gait uses Tds = 0.)
  * Stance leg (point foot, no ankle): its hip holds the trunk attitude (SIMBICON):
    hip_y realises a pitch PD (kpt, kdt) towards a forward lean pitch_d, hip_x a roll PD
    (kpr, kdr) towards a lean r0 over the stance foot (the 40 N m abductors cannot hold
    the trunk otherwise; abdomen_x bends by ax0 the same way). The swing hip's reaction
    torque is subtracted. Stance knee: stiff PD (kpks, kdks) to kst.
  * Yaw: the spherical feet have no torsional friction, so hip_z cannot yaw the body.
    A yaw PD torque Ty = -kpy yaw - kdy yaw_rate is produced instead by adding
    kyp * Ty * (stance side) to the stance pitch torque: the extra fore-aft ground force
    acts at the stance foot's lateral offset (budget limyaw). hip_z (kpz, kdz) only aims
    the legs along +x.
  * Swing leg: foot target relative to the mass centre in the world frame (Raibert):
        x_f = kx0 + v_x T/2 + kvx (v_x - v_des),     y_f = side*w + v_y T/2 + kvy v_y
    with at least wmin between the feet, reached along a smooth path from lift-off with a
    clearance bump `clear`. The hip position comes from 3D forward kinematics (torso
    Euler angles, abdomen and leg joints); an analytic 3D inverse kinematics gives
    hip_x, hip_y, knee targets, tracked by joint PD (kps/kds, kpk/kdk) within torque
    budgets limx/limy so that the stance hip keeps authority over the trunk.
  * Abdomen and arms are held by PD. v_des ramps up during the first second.
The gains were hand-structured, then refined by a (1+lambda) evolution strategy on seeds
0-15 (validated on 176 further seeds: all walk > 8 m).
"""

import numpy as np

PARAMS = {
    "T": 0.1956,                            # nominal swing duration [s]
    "Tds": 0.0,                             # double-support duration after touchdown [s]
    "vdes": 0.3445,                         # desired forward speed [m/s]
    "kx0": -0.0166,                         # sagittal foot offset [m]
    "kvx": 0.0911,                          # sagittal velocity-error placement gain [s]
    "w": 0.1019,                            # lateral half step width [m]
    "wmin": 0.1259,                         # minimum lateral distance between the feet [m]
    "kvy": 0.1402,                          # lateral velocity placement gain [s]
    "clear": 0.11,                          # swing foot clearance [m]
    "pitch_d": 0.0747,                      # trunk forward lean target [rad]
    "r0": 0.121,                            # trunk roll towards the stance side [rad]
    "ax0": 0.1068,                          # abdomen_x bend towards the stance side [rad]
    "kpt": 150.0, "kdt": 12.1739,           # trunk pitch PD via stance hip
    "kpr": 271.1002, "kdr": 27.2591,        # trunk roll PD via stance hip_x
    "kpy": 132.6391, "kdy": 14.7705,        # yaw PD [N m/rad, N m s/rad], steered via kyp
    "kyp": 1.7485,                          # pitch torque per unit yaw torque, times stance side
    "limx": 19.8348, "limy": 119.3028,      # swing hip_x / hip_y torque budgets [N m]
    "limyaw": 69.7146,                      # budget of the yaw-steering pitch torque [N m]
    "kpz": 62.3951, "kdz": 1.0,             # hip_z PD aiming the legs along +x
    "kps": 158.2293, "kds": 3.0,            # swing hip_x / hip_y PD
    "kpk": 112.2974, "kdk": 3.3169,         # swing knee PD
    "kpks": 592.0426, "kdks": 5.3516,       # stance knee PD
    "kst": -0.0388,                         # stance knee target [rad]
}

AY, AZ, AX, RHX, RHZ, RHY, RK, LHX, LHZ, LHY, LK = range(11)
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
L1, L2 = 0.383, 0.37   # hip-knee, knee-foot centre [m]
HX_RANGE = (-0.436, 0.087)
HZ_RANGE = (-1.047, 0.611)

_S = {"stance": 1, "t": 0.0, "lift": np.zeros(2)}


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def fk(phi, knee):
    """Foot centre relative to hip in the thigh's sagittal plane (x forward, z up)."""
    sh = phi - knee
    return np.array([-L1 * np.sin(phi) - L2 * np.sin(sh), -L1 * np.cos(phi) - L2 * np.cos(sh)])


def ik(x, z):
    """Hip_y and knee angles putting the foot centre at (x, z) in the thigh's sagittal plane."""
    r = np.clip(np.hypot(x, z), 0.3, L1 + L2 - 1e-3)
    beta = np.arctan2(-x, -z)
    gam = np.arccos(np.clip((L1 ** 2 + r ** 2 - L2 ** 2) / (2 * L1 * r), -1, 1))
    kap = np.pi - np.arccos(np.clip((L1 ** 2 + L2 ** 2 - r ** 2) / (2 * L1 * L2), -1, 1))
    return beta - gam, -kap


def leg_vec(s, hxq, hzq, hyq, kq):
    """Foot centre relative to hip in the pelvis frame."""
    f = fk(hyq, kq)
    return Rx(s * hxq) @ Rz(s * hzq) @ np.array([f[0], 0.0, f[1]])


def leg_ik(s, d, hzq):
    """hip_x, hip_y, knee for foot vector d (pelvis frame) with hip_z fixed to hzq."""
    b = s * hzq
    vx = d[0] / np.cos(b)
    ys = vx * np.sin(b)
    rr = max(d[1] ** 2 + d[2] ** 2 - ys ** 2, 0.09)
    vz = -np.sqrt(rr)
    a = np.arctan2(d[2], d[1]) - np.arctan2(vz, ys)
    a = (a + np.pi) % (2 * np.pi) - np.pi
    hxq = np.clip(a * s, *HX_RANGE)
    hyq, kq = ik(vx, vz)
    return hxq, hyq, kq


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    dt = obs["dt"]
    q = obs["joint_angles"]; dq = obs["joint_rates"]
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    Rt = Rz(yaw) @ Ry(pitch) @ Rx(roll)
    wb = Rt.T @ obs["root_angular_velocity"]          # body-frame angular velocity
    Rp = Rt @ Rz(q[AZ]) @ Ry(q[AY]) @ Rx(q[AX])       # pelvis orientation
    pyaw = np.arctan2(Rp[1, 0], Rp[0, 0])
    v = obs["com_velocity_xy"]
    feet = [obs["foot_xyz_rel"], obs["foot_left_xyz_rel"]]
    contact = [obs["foot_contact"], obs["foot_left_contact"]]
    hx = (RHX, LHX); hz = (RHZ, LHZ); hy = (RHY, LHY); kn = (RK, LK)
    sgn = (1.0, -1.0)  # hip_x / hip_z axis sign per leg

    if obs["step"] == 0:
        _S["stance"] = 1; _S["t"] = 0.0; _S["lift"] = feet[0][:2].copy()
    _S["t"] += dt
    st = _S["stance"]; sw = 1 - st
    if (_S["t"] > P["Tds"] + 0.5 * P["T"] and contact[sw]) or _S["t"] > P["Tds"] + 1.5 * P["T"]:
        _S["stance"] = st = sw; sw = 1 - st; _S["t"] = 0.0; _S["lift"] = feet[sw][:2].copy()
    t = _S["t"]
    vdes = P["vdes"] * min(1.0, obs["time"] / 1.0)
    sst = -1.0 if st == 0 else 1.0  # stance side (+1 = left, +y)

    tau = np.zeros(17)
    ab_d = (0.0, 0.0, -sst * P["ax0"])
    for j in (AY, AZ, AX):
        tau[j] = 200 * (ab_d[j] - q[j]) - 4 * dq[j]
    for j in range(11, 17):
        tau[j] = 20 * (0.0 - q[j]) - 1 * dq[j]

    # swing foot target relative to the mass centre (world frame)
    side = -sst
    xf = P["kx0"] + v[0] * P["T"] / 2 + P["kvx"] * (v[0] - vdes)
    yf = side * P["w"] + v[1] * P["T"] / 2 + P["kvy"] * v[1]
    yf = min(yf, feet[st][1] - P["wmin"]) if sw == 0 else max(yf, feet[st][1] + P["wmin"])
    ds = t < P["Tds"]
    sph = np.clip((t - P["Tds"]) / P["T"], 0.0, 1.0)
    blend = np.clip(1.6 * sph, 0, 1); blend = blend * blend * (3 - 2 * blend)
    x_t = _S["lift"][0] + (xf - _S["lift"][0]) * blend
    y_t = _S["lift"][1] + (yf - _S["lift"][1]) * blend
    z_t = feet[st][2] + P["clear"] * np.sin(np.pi * sph) - (0.03 if sph > 0.8 else 0.0)

    # swing hip from forward kinematics, then 3D IK with hip_z aiming the leg along +x
    s_sw = sgn[sw]
    hip = feet[sw] - Rp @ leg_vec(s_sw, q[hx[sw]], q[hz[sw]], q[hy[sw]], q[kn[sw]])
    hz_d = np.clip(-pyaw / s_sw, *HZ_RANGE)
    d = Rp.T @ (np.array([x_t, y_t, z_t]) - hip)
    hx_d, hy_d, kn_d = leg_ik(s_sw, d, hz_d)
    tau_sw = {
        hx[sw]: np.clip(P["kps"] * (hx_d - q[hx[sw]]) - P["kds"] * dq[hx[sw]], -P["limx"], P["limx"]),
        hy[sw]: np.clip(P["kps"] * (hy_d - q[hy[sw]]) - P["kds"] * dq[hy[sw]], -P["limy"], P["limy"]),
        hz[sw]: P["kpz"] * (hz_d - q[hz[sw]]) - P["kdz"] * dq[hz[sw]],
        kn[sw]: P["kpk"] * (kn_d - q[kn[sw]]) - P["kdk"] * dq[kn[sw]],
    }

    # trunk attitude torques (body frame), realised through the stance hip(s)
    Tp = P["kpt"] * (P["pitch_d"] - pitch) - P["kdt"] * wb[1]
    Tr = P["kpr"] * (-sst * P["r0"] - roll) - P["kdr"] * wb[0]
    Ty = P["kpy"] * (0.0 - yaw) - P["kdy"] * obs["root_angular_velocity"][2]
    Tp_st = Tp + np.clip(P["kyp"] * Ty, -P["limyaw"], P["limyaw"]) * sst  # yaw through the GRF at the stance foot's lateral offset
    if ds:  # double support: both legs share the trunk torques, both knees stiff
        for i in range(2):
            tau[hy[i]] = -0.5 * Tp
            tau[hx[i]] = -0.5 * Tr / sgn[i]
            tau[hz[i]] = -0.5 * Ty / sgn[i]
            tau[kn[i]] = P["kpks"] * (P["kst"] - q[kn[i]]) - P["kdks"] * dq[kn[i]]
    else:
        for j, v_ in tau_sw.items():
            tau[j] = v_
        tau[hy[st]] = -Tp_st - tau[hy[sw]]
        tau[hx[st]] = -(Tr + sgn[sw] * tau[hx[sw]]) / sgn[st]
        hz_st = np.clip(-pyaw / sgn[st], *HZ_RANGE)
        tau[hz[st]] = 0.5 * P["kpz"] * (hz_st - q[hz[st]]) - P["kdz"] * dq[hz[st]]
        tau[kn[st]] = P["kpks"] * (P["kst"] - q[kn[st]]) - P["kdks"] * dq[kn[st]]

    return np.clip(tau / GEAR, -0.4, 0.4)
