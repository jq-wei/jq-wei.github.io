"""Slung-payload quadrotor through a window lower than the hanging vehicle, then settle the swing.

Idea: a conical pendulum. With the payload circling the vehicle at ~62 deg from the vertical, it rides at a constant
height L cos(62 deg) = 0.47 m below the vehicle and 0.88 m to the side. Put the vehicle at window_z + 0.30 and every
point of the circle lies inside the 0.9 m x 2.0 m opening: the vehicle, the payload and the straight cable then cross
the wall plane inside the window whatever the phase of the swing, so the passage needs no timing.

Estimation (200 Hz)
- Attitude: gyro integration (bias estimated), yaw corrected by the delayed tracker, roll/pitch corrected from the
  acceleration residual of the translational filter (tilt error shows up as an unexplained horizontal acceleration).
- Position/velocity: a [p, v, d] Kalman filter per axis driven by R f + g (accelerometer), with the delayed tracker
  frame compared with the estimate of six steps earlier; d absorbs the accelerometer bias.
- Payload: a virtual pendulum r = p_payload - p_vehicle driven by the estimated vehicle acceleration (it starts at rest,
  as the payload does), corrected from the accelerometer's body-x,y components, which see the cable force because the
  thrust is purely along body z. A slow offset (dl, time constant dl_tau) takes the drag and bias; after the passage a
  quadratic drag model with the wind estimated at the start explains the drag change due to the vehicle's own motion.
  obs_kr / obs_kv are the correction gains on r and dr/dt (phase 0-1, and *2 in phase 2).
- Thrust-to-mass ratio alpha (hidden mass and thrust coefficient) from body-z specific force vs modelled rotor thrust.

Reference (relative to the window, built at the first step)
- The payload path (flat output) is a spiral whose radius grows smoothly (7th-order step over Ts) to L sin(theta),
  at the conical-pendulum rate sqrt(g / (L cos)), around a centre that moves from the start to just before the wall
  (Ts), then on through the window at constant speed; the vehicle path is payload + L * (a_payload - g)/|a_payload - g|.
- Phase 0 (t < T0): hold the start. Phase 1: track the vehicle path. Phase 2 (vehicle at x_end beyond the wall):
  regulate to the goal.

Control
- Phase 1: a = a_ref(t + ff_adv) + kp e_p + kv e_v + integral (ki); ff_adv compensates the attitude/motor lag (it sets
  the cone angle actually reached); cone_fb * LQR_K adds feedback on the payload's deviation from its planned motion.
- Phase 2: horizontal LQR (LQR_K2) on vehicle position/velocity and on the high-passed (r_hp) payload swing; vertical PD.
- Thrust vector F = m_v (a - g) - T r / L (cable force feedforward from the virtual pendulum), divided by alpha;
  geometric attitude control (att_w natural frequency, att_z damping, yaw_w for yaw) through the rotor mixer,
  with a motor-lag lead (lead) on the modelled 30 ms rotor response.
"""

import math
import numpy as np

G = 9.81
M_V = 1.325
M_P = 0.15
L_C = 1.0
DT = 0.005
GVEC = np.array([0.0, 0.0, -G])
J = np.array([[0.03665, 0, -0.0021], [0, 0.02541, 0], [-0.0021, 0, 0.06053]])
ROTOR = np.array([[-0.14, -0.18], [-0.14, 0.18], [0.14, 0.18], [0.14, -0.18]])
KYAW = np.array([-0.0201, 0.0201, -0.0201, 0.0201])
MIX = np.vstack([np.ones(4), ROTOR[:, 1], -ROTOR[:, 0], KYAW])
MIX_INV = np.linalg.inv(MIX)
CDRAG = np.array([0.0254, 0.0211, 0.1656])   # quadratic drag coefficients of the vehicle along body x, y, z [N/(m/s)^2]

PARAMS = {
    "kp": 5.0, "kv": 4.0, "ki": 1.5,
    "att_w": 14.0, "att_z": 0.8, "yaw_w": 5.0,
    "lead": 1.0, "alpha_rate": 1.0,
    "ff_adv": 0.085, "cone_fb": 0.3,
    "obs_kr": 1.0, "obs_kv": 1.0, "obs_kr2": 1.0, "obs_kv2": 1.0, "dl_tau": 6.0, "dl_tau2": 6.0, "drag_model": 1.0,
    "r_hp": 2.0,
}


def skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def expso3(w):
    th = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    K = skew(w)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (math.sin(th) / th) * K + ((1 - math.cos(th)) / (th * th)) * (K @ K)


class State:
    pass


S = State()


CONE = {"theta": 62.0, "zv": 0.30, "Ts": 3.0, "T0": 0.5, "tr": 1.5, "x_end": 1.0}
T_M = CONE["T0"]
# horizontal gains on [position, velocity, payload relative position, payload relative velocity]
LQR_K = np.array([2.0, 3.05, -10.6, -2.96])    # LQR of the hanging pendulum; its swing part corrects the cone during the passage
LQR_K2 = np.array([3.0, 4.0, -6.0, -1.0])      # final regulation to the goal with swing damping


def smooth7(s):
    s = np.clip(s, 0.0, 1.0)
    return s**4 * (35 - 84 * s + 70 * s**2 - 20 * s**3)


def build_plan():
    """Conical swing: the payload path (flat output) is a spiral of growing radius around a centre moving toward the
    window, at the height that puts the vehicle at window_z + zv; the vehicle path follows from the cable direction."""
    C = CONE
    wc = S.wc
    th = math.radians(C["theta"])
    rho_f = L_C * math.sin(th)
    Vp0 = (-rho_f - 0.1 - S.p0[0]) / (C["Ts"] - 0.5 * C["tr"])
    Tend = C["Ts"] + (C["x_end"] - (-rho_f - 0.1)) / Vp0 + 0.5
    t = np.arange(-0.5, Tend + 0.5 + 1e-9, DT)
    rho = rho_f * smooth7(t / C["Ts"])
    cth = np.sqrt(1.0 - (rho / L_C) ** 2)
    Om = np.sqrt(G / (L_C * cth))
    phi = np.cumsum(Om) * DT
    p0 = S.p0
    Vp = (-rho_f - 0.1 - p0[0]) / (C["Ts"] - 0.5 * C["tr"])
    S.Vp = Vp
    vc = Vp * smooth7(t / C["tr"])
    xc = p0[0] + np.cumsum(vc) * DT
    xc = xc - xc[int(round(0.5 / DT))] + p0[0]
    lat = smooth7(1.5 * t / C["Ts"])
    yc = p0[1] + (wc[1] - p0[1]) * lat
    zvn = p0[2] + (wc[2] + C["zv"] - p0[2]) * lat
    pp = np.stack([xc + rho * np.cos(phi), yc + rho * np.sin(phi), zvn - L_C * cth], 1)
    vp = np.gradient(pp, DT, axis=0)
    ap = np.gradient(vp, DT, axis=0)
    n = ap - GVEC
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    pv = pp + L_C * n
    vv = np.gradient(pv, DT, axis=0)
    av = np.gradient(vv, DT, axis=0)
    k0 = int(round(0.5 / DT))
    S.mp, S.mv, S.ma, S.mr, S.mpa = pv[k0:], vv[k0:], av[k0:], (pp - pv)[k0:], ap[k0:]
    S.mrd = np.gradient(S.mr, DT, axis=0)
    S.H = Tend
    S.pm = pv[k0].copy()
    S.pm[1] = wc[1]


def reference(t):
    """Returns (p_ref, v_ref, a_ref, phase)."""
    if t < T_M:
        return S.p0.copy(), np.zeros(3), np.zeros(3), 0
    tm = t - T_M
    i = min(int(round(tm / DT)), len(S.mp) - 1)
    if tm < S.H and S.mp[i, 0] < CONE["x_end"]:
        j = min(int(round((tm + PARAMS["ff_adv"]) / DT)), len(S.mp) - 1)
        a = S.ma[j].copy()
        lam = PARAMS["cone_fb"]
        if lam > 0:
            dr = S.r[0:2] - S.mr[i, 0:2]
            drd = S.rd[0:2] - S.mrd[i, 0:2]
            a[0:2] += lam * (-LQR_K[2] * dr - LQR_K[3] * drd)
        return S.mp[i].copy(), S.mv[i].copy(), a, 1
    return S.goal.copy(), np.zeros(3), np.zeros(3), 2


def reset(obs):
    S.R = np.eye(3)
    f = np.asarray(obs["accel"], float)
    # initial tilt from the accelerometer (payload hanging at rest), yaw from the tracker
    fz = f / np.linalg.norm(f)
    roll = math.atan2(fz[1], fz[2])
    pitch = math.atan2(-fz[0], math.hypot(fz[1], fz[2]))
    yaw = float(obs["tracker_yaw"])
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    S.R = Rz @ Ry @ Rx
    S.bg = np.zeros(3)
    S.p = np.asarray(obs["tracker_pos"], float).copy()
    S.v = np.zeros(3)
    S.d = np.zeros(3)
    S.P = np.diag([1e-4, 1e-3, 0.05])
    S.phist = [S.p.copy()] * 8
    S.yhist = [yaw] * 8
    S.r = np.array([0.0, 0.0, -L_C])
    S.rd = np.zeros(3)
    S.av = np.zeros(3)
    S.ip = np.zeros(3)
    S.T = M_P * G
    S.wc = np.asarray(obs["window_centre"], float).copy()
    S.goal = np.asarray(obs["target_pos"], float).copy()
    S.p0 = S.p.copy()
    S.f = None
    S.dl = np.zeros(2)
    S.k = 0
    S.wind = np.zeros(3)
    S.dlm = np.zeros(2)
    S.wprev = np.asarray(obs["gyro"], float).copy()
    S.wdot = np.zeros(3)
    S.fo = np.asarray(obs["accel"], float).copy()
    S.phase = 0
    S.rlp = np.zeros(2)
    S.alpha = 1.0 / M_V
    S.Tm = np.full(4, (M_V + M_P) * G / 4)
    S.ulast = S.Tm.copy()
    build_plan()


def yaw_of(R):
    return math.atan2(R[1, 0], R[0, 0])


def estimate(obs):
    gyro = np.asarray(obs["gyro"], float)
    f = np.asarray(obs["accel"], float)
    w = gyro - S.bg
    S.fo = f
    S.R = S.R @ expso3(w * DT)
    fw = S.R @ f
    a = fw + GVEC + S.d
    S.av = a
    S.p = S.p + S.v * DT + 0.5 * a * DT * DT
    S.v = S.v + a * DT
    F = np.array([[1, DT, 0.5 * DT * DT], [0, 1, DT], [0, 0, 1]])
    Q = np.diag([1e-9, (0.2 * DT) ** 2, (0.08 ** 2) * DT])
    S.P = F @ S.P @ F.T + Q
    S.phist.append(S.p.copy())
    S.phist.pop(0)
    S.yhist.append(yaw_of(S.R))
    S.yhist.pop(0)
    if obs["tracker_fresh"]:
        z = np.asarray(obs["tracker_pos"], float)
        innov = z - S.phist[-7]
        Rm = 0.003 ** 2
        Sv = S.P[0, 0] + Rm
        K = S.P[:, 0] / Sv
        S.p = S.p + K[0] * innov
        S.v = S.v + K[1] * innov
        S.d = S.d + K[2] * innov
        S.P = S.P - np.outer(K, S.P[0, :])
        # correct the history so later innovations are consistent
        for i in range(len(S.phist)):
            S.phist[i] = S.phist[i] + K[0] * innov
        ey = (float(obs["tracker_yaw"]) - S.yhist[-7] + math.pi) % (2 * math.pi) - math.pi
        phz = 0.15 * ey
        S.R = expso3(np.array([0, 0, phz])) @ S.R
        S.bg = S.bg - 0.02 * (S.R.T @ np.array([0, 0, phz])) / (4 * DT)
        S.yhist = [y + phz for y in S.yhist]
    # tilt correction from the acceleration residual
    n2 = float(fw @ fw)
    if n2 > 4.0:
        phi = 0.5 * DT * np.cross(fw, S.d) / n2
        phi[2] = 0.0
        S.R = expso3(phi) @ S.R
        S.d = S.d - np.cross(phi, fw)
        S.bg = S.bg - 0.3 * (S.R.T @ phi) / DT * DT


def pendulum(a_v):
    """Virtual payload: relative position r = p_payload - p_vehicle driven by the vehicle acceleration."""
    r, rd = S.r, S.rd
    T = M_P * (float(r @ (GVEC - a_v)) + float(rd @ rd)) / L_C
    T = max(T, 0.0)
    rdd = -T * r / (M_P * L_C) + GVEC - a_v
    rd = rd + rdd * DT
    r = r + rd * DT
    n = np.linalg.norm(r)
    if n > L_C:
        u = r / n
        r = u * L_C
        vr = float(rd @ u)
        if vr > 0:
            rd = rd - vr * u
    # correction from the accelerometer: its body-x,y components see the cable force (thrust is along body z)
    if T > 0.5 and S.f is not None:
        fb = S.R.T @ (T * r / L_C) / M_V
        e0 = S.f[0:2] - fb[0:2] - S.dl          # residual without a drag model
        if S.k < 80:
            # payload hanging at rest at the start: learn the drag and bias offset first
            S.dl = S.dl + e0 / (S.k + 1)
            S.k += 1
            if S.k == 80:
                # mean wind from the drag the vehicle feels at rest (bias included), body axes -> world
                wb = np.zeros(3)
                wb[0:2] = np.sign(S.dl) * np.sqrt(np.abs(S.dl) * M_V / CDRAG[0:2])
                S.wind = S.R @ wb
                S.wind[2] = 0.0
                va = S.R.T @ (S.v - S.wind)
                S.dlm = S.dl - (-CDRAG * np.abs(va) * va / M_V)[0:2]
            S.r, S.rd, S.T = r, rd, T
            return
        va = S.R.T @ (S.v - S.wind)
        e1 = S.f[0:2] - fb[0:2] + (CDRAG * np.abs(va) * va / M_V)[0:2] - S.dlm   # residual with the drag model
        tau = PARAMS["dl_tau2"] if S.phase == 2 else PARAMS["dl_tau"]
        S.dl = S.dl + (DT / tau) * e0
        S.dlm = S.dlm + (DT / tau) * e1
        e = e1 if (S.phase == 2 and PARAMS["drag_model"] > 0) else e0
        ew = S.R[:, 0:2] @ e
        u = r / L_C
        ew = ew - float(ew @ u) * u
        dr = ew * M_V * L_C / T
        kr, kv = (PARAMS["obs_kr2"], PARAMS["obs_kv2"]) if S.phase == 2 else (PARAMS["obs_kr"], PARAMS["obs_kv"])
        r = r + kr * DT * dr
        rd = rd + kv * DT * dr
        r = r * (L_C / np.linalg.norm(r))
        rd = rd - float(rd @ r) / (L_C * L_C) * r
    S.r, S.rd, S.T = r, rd, T
    S.rlp = S.rlp + (DT / PARAMS["r_hp"]) * (r[0:2] - S.rlp)


def control(obs, pref, vref, aref, yaw_ref=0.0, lqr=False):
    P = PARAMS
    ep = pref - S.p
    ev = vref - S.v
    S.ip = np.clip(S.ip + P["ki"] * ep * DT, -3.0, 3.0)
    a_des = aref + P["kp"] * ep + P["kv"] * ev + S.ip
    if lqr:
        K = LQR_K2
        e_xy = np.clip(-ep[0:2], -1.5, 1.5)
        rh = S.r[0:2] - S.rlp
        a_xy = -(K[0] * e_xy + K[1] * (-ev[0:2]) + K[2] * rh + K[3] * S.rd[0:2]) + S.ip[0:2]
        nrm = np.linalg.norm(a_xy)
        if nrm > 6.0:
            a_xy *= 6.0 / nrm
        a_des[0:2] = a_xy
    # feedforward of the cable force with the commanded acceleration
    r, rd = S.r, S.rd
    T = max(0.0, M_P * (float(r @ (GVEC - a_des)) + float(rd @ rd)) / L_C)
    Fd = M_V * (a_des - GVEC) - T * r / L_C
    if Fd[2] < 2.0:
        Fd[2] = 2.0
    b3 = Fd / np.linalg.norm(Fd)
    c1 = np.array([math.cos(yaw_ref), math.sin(yaw_ref), 0.0])
    b2 = np.cross(b3, c1)
    b2 /= np.linalg.norm(b2)
    b1 = np.cross(b2, b3)
    Rd = np.column_stack([b1, b2, b3])
    R = S.R
    E = Rd.T @ R - R.T @ Rd
    eR = 0.5 * np.array([E[2, 1], E[0, 2], E[1, 0]])
    w = np.asarray(obs["gyro"], float) - S.bg
    wn, z = P["att_w"], P["att_z"]
    kR = np.array([wn * wn, wn * wn, P["yaw_w"] ** 2])
    kW = np.array([2 * z * wn, 2 * z * wn, 2 * 0.9 * P["yaw_w"]])
    tau = J @ (-kR * eR - kW * w) + np.cross(w, J @ w)
    thrust = float(Fd @ R[:, 2]) / (M_V * S.alpha)
    tau = tau / (M_V * S.alpha)
    u = MIX_INV @ np.array([thrust, tau[0], tau[1], tau[2]])
    # motor lag pre-compensation on the modelled rotor thrusts
    u = u + P["lead"] * (u - S.Tm)
    u = np.clip(u, 0.0, 13.0)
    return u


MOTOR_A = 1.0 - math.exp(-DT / 0.03)


def controller(obs: dict):
    if obs["step"] == 0:
        reset(obs)
    else:
        S.Tm = S.Tm + (S.ulast - S.Tm) * MOTOR_A
        estimate(obs)
        S.f = S.fo
        # thrust-to-mass ratio from the accelerometer along body z
        f = np.asarray(obs["accel"], float)
        Fcz = float((S.R.T @ (S.T * S.r / L_C))[2])
        st = float(np.sum(S.Tm))
        if st > 3.0:
            a_obs = (f[2] - Fcz / M_V) / st
            S.alpha += PARAMS["alpha_rate"] * DT * (a_obs - S.alpha)
    pendulum(S.av)
    t = float(obs["time"])
    pref, vref, aref, phase = reference(t)
    S.phase = phase
    if phase == 2:
        u = control(obs, pref, vref, aref, lqr=True)
    else:
        u = control(obs, pref, vref, aref)
    S.ulast = u.copy()
    return u
