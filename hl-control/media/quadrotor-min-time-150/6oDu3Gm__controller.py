"""Gate-race controller: EKF state estimation, a precomputed near-time-optimal trajectory, INDI-style tracking.

Estimation (error-state EKF, 15 states): position and velocity of the IMU point, attitude, gyro bias and accelerometer
bias. The IMU (gyro + specific force) drives the prediction at 200 Hz; each 50 Hz tracker frame (position and yaw of
the body origin, 30 ms late) is compared with the stored prediction of the step it shows. Roll and pitch become
observable through the tracker (a tilt error turns into an acceleration error). The rotor thrusts are not measured, but
their lag state (command units) is simulated exactly from the commands; the hidden thrust-per-mass ratio ``eta`` is
estimated from the accelerometer during the hover before the launch.

Reference: a point-mass trajectory with piecewise-constant jerk, from a run-up point inside the start sphere (at rest)
through the four gate centres, optimised offline (scratch plan2.py, SLSQP) for minimum time under: thrust acceleration
(with the drag model) <= 27 m/s^2, jerk <= 150 m/s^3, tilt <= 75 deg, vertical thrust >= 6 m/s^2 and thrust-direction
rotation <= 3 rad/s. Before ``t_launch`` the vehicle hovers and moves to the run-up point (the clock does not run).
The reference clock tau advances at rate 1 + k_s * (along-track lead in s), clipped to [rate_min, rate_max], so the
vehicle follows the path (and the gates) even when wind or thrust margin make it lag; v_ref, a_ref, j_ref are scaled
by rate, rate^2, rate^3. ``t_scale`` stretches the whole plan in time (< 1 faster).

Translational control: a_des = a_ref(tau + t_ff) + kp (p_ref - p) + kd (v_ref - v_com) + ki int(p_ref - p) (integral
in the hover only). v_com is the velocity of the centre of mass: the controlled origin sits 5.4 cm below it and its
velocity reacts to body rates with the wrong sign (feeding that back destabilises the attitude loop). On the approach
to the last gate a zero-effort-miss term k_zem * ZEM / t_go^2 (t_go < t_zem, capped at zem_cap) is added.
Thrust vector per unit mass: A = a_des + g - d, with d the non-thrust specific force (drag, wind, mass and thrust
coefficient errors): the accelerometer residual after removing the modelled thrust and the IMU lever-arm terms,
low-passed with tau_d (incremental / INDI-like), plus ``k_drag`` times a nominal drag model (0 = pure INDI). At the
thrust limit the vertical component is kept and the horizontal one shortened.

Attitude: reduced-attitude control of the thrust axis onto z_d = A/|A| (axis-angle error in body axes), body-rate
command k_att * e + feedforward of the reference thrust-direction rate; heading held weakly at 0 (k_yaw). Angular
acceleration k_rate * (w_d - w) -> torques with the nominal inertia. Collective = A . z_b, capped so that the vertical
thrust does not exceed A_z. Mixer: roll/pitch first, then collective, yaw with the remaining margin; each rotor
command leads its simulated lag state by k_lead (u = T + k_lead (T_des - T)) to speed up the 30 ms motors.
If a gate is missed, a slow pursuit of the remaining gates still finishes the course.
"""

import math
import numpy as np
from hl_control.envs.quadrotor import nominal_parameters

_NP = nominal_parameters()
M_NOM = float(_NP["mass"])
J_NOM = np.array(_NP["inertia"], dtype=float)
COM = np.array(_NP["com"], dtype=float)
ROT = np.array(_NP["rotor_positions"], dtype=float)
YAWC = np.array(_NP["yaw_coeff"], dtype=float)
R_IMU = np.array(_NP["imu_position"], dtype=float)
DRAG_FRAME = np.array(_NP["inertia_frame"], dtype=float)
DRAG_C = np.array(_NP["drag_linear_coeff"], dtype=float)
G = 9.81
GV = np.array([0.0, 0.0, -G])
DT = 0.005
TMAX = 13.0
ETA_NOM = 1.0 / M_NOM
LAG_SUB = 1.0 - math.exp(-0.0025 / 0.03)
BETA = 1.0 - (1.0 - LAG_SUB) ** 2

MIX = np.vstack([np.ones(4), ROT[:, 1] - COM[1], -(ROT[:, 0] - COM[0]), YAWC])
MIX_INV = np.linalg.inv(MIX)
J_INV = np.linalg.inv(J_NOM)
R_CI = R_IMU - COM          # IMU relative to the nominal centre of mass

# kp, kd [1/s^2, 1/s], ki: position / velocity / hover-integral gains of the translational loop
# k_att, k_yaw [1/s]: thrust-axis (roll/pitch) and heading gains -> desired body rates
# k_rate, k_rate_yaw [1/s]: body-rate gains -> angular acceleration;  k_lead: motor-lag lead factor
# tau_d [s]: low-pass of the accelerometer disturbance estimate;  k_drag: weight of the nominal drag model
# t_launch [s]: end of the hover;  t_scale: time stretch of the plan;  t_ff [s]: look-ahead of the feedforward
# k_s [1/s], rate_min, rate_max: time scaling of the reference clock from the along-track lead
# k_zem, t_zem [s], zem_cap [m/s^2]: terminal zero-effort-miss guidance on the last gate
PARAMS = {
    "kp": 22.0, "kd": 9.0, "ki": 2.0,
    "k_att": 12.0, "k_yaw": 3.0, "k_rate": 45.0, "k_rate_yaw": 12.0,
    "k_lead": 2.0, "tau_d": 0.02, "k_drag": 0.0,
    "t_launch": 1.8, "t_scale": 0.95, "t_ff": 0.03, "k_s": 3.0, "rate_min": 0.4, "rate_max": 1.2,
    "k_zem": 2.0, "t_zem": 0.7, "zem_cap": 15.0,
}
PARAM_BOUNDS = {
    "kp": (10.0, 60.0), "kd": (4.0, 20.0), "ki": (0.0, 10.0), "k_att": (6.0, 25.0), "k_yaw": (1.0, 8.0),
    "k_rate": (15.0, 70.0), "k_rate_yaw": (5.0, 25.0), "k_lead": (1.0, 3.5), "tau_d": (0.01, 0.1),
    "k_drag": (0.0, 1.0), "t_launch": (1.2, 3.0), "t_scale": (0.9, 1.2), "t_ff": (0.0, 0.12), "k_s": (0.0, 8.0),
    "rate_min": (0.2, 0.9), "rate_max": (1.0, 1.5), "k_zem": (0.0, 4.0), "t_zem": (0.1, 1.0), "zem_cap": (5.0, 30.0),
}

# ---------------- reference trajectory ----------------
TRAJ = {
    "T": [0.352357, 0.867048, 0.926514, 1.287031, 0.655521],
    "nseg": [6, 14, 14, 14, 12],
    "p0": [-0.131402, -0.103659, 1.933756],
    "J": [
        [22.620, 18.827, -2.613],
        [32.361, 26.685, 67.308],
        [70.389, 57.541, 150.000],
        [60.532, 48.585, -4.807],
        [47.521, 37.374, -53.933],
        [32.422, 24.040, -70.901],
        [18.383, 11.476, -79.260],
        [5.432, -11.694, -67.897],
        [-13.685, -33.344, -8.002],
        [-13.219, -35.599, -7.801],
        [19.738, -78.579, -1.696],
        [-33.405, -69.877, -9.672],
        [7.749, -71.369, 6.845],
        [-3.314, -71.171, 8.637],
        [-25.476, -64.884, 7.277],
        [-45.442, -51.254, 5.775],
        [-55.421, -38.331, 3.931],
        [-62.604, -24.178, 2.012],
        [-66.949, -9.119, 0.211],
        [-67.431, -3.658, 0.645],
        [-67.902, 15.060, 1.875],
        [-64.140, 26.289, 2.558],
        [-56.920, 39.654, 2.685],
        [0.272, 73.000, 3.201],
        [28.274, 64.338, 4.282],
        [22.512, 45.917, 5.541],
        [4.167, 34.859, 14.245],
        [7.147, 31.128, 20.184],
        [14.580, 29.798, 17.931],
        [20.302, 28.528, 14.279],
        [-104.389, 62.705, 150.000],
        [64.867, 45.607, 16.560],
        [74.661, 31.936, 2.982],
        [79.000, 18.017, -2.187],
        [80.145, 7.392, -10.982],
        [23.342, 41.357, -42.164],
        [16.685, 39.169, -69.248],
        [11.598, 5.689, -57.210],
        [6.657, -29.804, -5.950],
        [14.884, -44.096, -8.121],
        [-7.697, -78.085, -17.568],
        [-70.973, -77.019, -16.974],
        [45.838, -15.659, 20.878],
        [19.599, -36.733, 14.406],
        [-7.695, -37.628, 7.473],
        [-17.991, -33.113, 0.777],
        [-1.055, -49.650, 0.699],
        [0.210, -79.992, 0.664],
        [-66.633, -19.308, 0.891],
        [-68.967, -7.336, 1.119],
        [-69.280, 3.949, 1.221],
        [-67.647, 15.509, 1.196],
        [11.712, 8.427, -1.546],
        [68.609, -9.531, 0.654],
        [68.037, 22.000, -5.618],
        [58.232, 54.904, -13.100],
        [40.595, 78.205, -18.599],
        [4.486, 99.696, -20.566],
        [-24.878, 71.813, -12.575],
        [3.259, -8.781, 0.891]
    ],
}


def _build_traj(traj, scale):
    T = np.array(traj["T"]) * scale
    nseg = traj["nseg"]
    J = np.array(traj["J"]) / scale ** 3
    h = np.concatenate([np.full(n, T[i] / n) for i, n in enumerate(nseg)])
    p, v, a = np.array(traj["p0"], dtype=float), np.zeros(3), np.zeros(3)
    S = []
    t0 = 0.0
    for k in range(len(h)):
        S.append((t0, p.copy(), v.copy(), a.copy(), J[k].copy()))
        hk, j = h[k], J[k]
        p = p + v * hk + a * hk * hk / 2 + j * hk ** 3 / 6
        v = v + a * hk + j * hk * hk / 2
        a = a + j * hk
        t0 += hk
    ends = np.cumsum(nseg)
    gate_t = [float(np.sum(T[:i + 1])) for i in range(1, 5)]
    t_start = np.array([s[0] for s in S])
    return {"t": t_start, "S": S, "total": float(np.sum(T)), "end": (p, v, a), "gate_t": gate_t,
            "p0": np.array(traj["p0"], dtype=float), "t_exit": float(T[0])}


def _ref(tr, tau):
    if tau <= 0.0:
        return tr["p0"].copy(), np.zeros(3), np.zeros(3), np.zeros(3)
    if tau >= tr["total"]:
        p, v, a = tr["end"]
        dt = min(tau - tr["total"], 0.3)
        return p + v * dt + 0.5 * a * dt * dt, v + a * dt, a.copy(), np.zeros(3)
    k = int(np.searchsorted(tr["t"], tau, side="right") - 1)
    t0, p, v, a, j = tr["S"][k]
    s = tau - t0
    return (p + v * s + a * s * s / 2 + j * s ** 3 / 6, v + a * s + j * s * s / 2, a + j * s, j)


# ---------------- helpers ----------------
def _skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _expm(w):
    th = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
    K = _skew(w)
    if th < 1e-9:
        return np.eye(3) + K
    return np.eye(3) + (math.sin(th) / th) * K + ((1.0 - math.cos(th)) / (th * th)) * (K @ K)


def _rz(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _drag_acc(v_air, R):
    """Specific force of the nominal inertia-box quadratic drag (world frame) at air velocity v_air (vehicle - air)."""
    Rb = R @ DRAG_FRAME
    vb = Rb.T @ v_air
    return Rb @ (-DRAG_C * np.abs(vb) * vb) / M_NOM


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class _State:
    pass


_S = _State()
_TR = {}


def _reset(obs):
    P = PARAMS
    s = _S
    key = round(P["t_scale"], 6)
    if key not in _TR:
        _TR[key] = _build_traj(TRAJ, P["t_scale"])
    s.tr = _TR[key]
    yaw = float(obs["tracker_yaw"])
    s.R = _rz(yaw)
    s.p = np.array(obs["tracker_pos"], dtype=float) + s.R @ R_IMU     # IMU point
    s.v = np.zeros(3)
    s.bg = np.zeros(3)
    s.ba = np.zeros(3)
    s.P = np.diag([0.004 ** 2] * 3 + [0.02 ** 2] * 3 + [0.04 ** 2, 0.04 ** 2, 0.01 ** 2] + [0.012 ** 2] * 3 + [0.09 ** 2] * 3)
    s.hist_p = [np.array(obs["tracker_pos"], dtype=float)] * 7       # predicted origin positions, oldest first
    s.hist_yaw = [yaw] * 7
    s.gyro_prev = np.array(obs["gyro"], dtype=float)
    s.acc_prev = np.array(obs["accel"], dtype=float)
    s.Tst = np.full(4, M_NOM * G / 4.0)
    s.eta = ETA_NOM
    s.d_res = np.zeros(3)
    s.ei = np.zeros(3)
    s.p_init = np.array(obs["tracker_pos"], dtype=float)
    s.recover = False
    s.tau = 0.0
    s.last_u = np.full(4, M_NOM * G / 4.0)


def _ekf_predict(s, gyro, acc):
    w = 0.5 * (gyro + s.gyro_prev) - s.bg
    f = 0.5 * (acc + s.acc_prev) - s.ba
    s.gyro_prev, s.acc_prev = gyro, acc
    R0 = s.R
    dR = _expm(w * DT)
    Rm = R0 @ _expm(w * (0.5 * DT))
    aw = Rm @ f + GV
    s.p = s.p + s.v * DT + 0.5 * aw * DT * DT
    s.v = s.v + aw * DT
    s.R = R0 @ dR
    # error-state transition
    F = np.eye(15)
    F[0:3, 3:6] = np.eye(3) * DT
    F[3:6, 6:9] = -Rm @ _skew(f) * DT
    F[3:6, 12:15] = -Rm * DT
    F[6:9, 6:9] = dR.T
    F[6:9, 9:12] = -np.eye(3) * DT
    Pm = F @ s.P @ F.T
    q = np.empty(15)
    q[0:3] = 1e-8
    q[3:6] = (0.5 * DT) ** 2
    q[6:9] = (0.03 * DT) ** 2
    q[9:12] = (2e-4) ** 2 * DT
    q[12:15] = (2e-3) ** 2 * DT
    Pm[np.diag_indices(15)] += q
    s.P = Pm


def _ekf_update(s, z_pos, z_yaw):
    p_old = s.hist_p[0]
    yaw_old = s.hist_yaw[0]
    R = s.R
    H = np.zeros((4, 15))
    H[0:3, 0:3] = np.eye(3)
    H[0:3, 3:6] = -6 * DT * np.eye(3)
    H[0:3, 6:9] = R @ _skew(R_IMU)
    den = R[0, 0] ** 2 + R[1, 0] ** 2
    hphi = np.array([-R[0, 0] * R[2, 0] / den, -R[1, 0] * R[2, 0] / den, 1.0])
    H[3, 6:9] = hphi @ R
    y = np.concatenate([z_pos - p_old, [_wrap(z_yaw - yaw_old)]])
    Rm = np.diag([0.004 ** 2] * 3 + [0.006 ** 2])
    PHt = s.P @ H.T
    S = H @ PHt + Rm
    K = PHt @ np.linalg.inv(S)
    dx = K @ y
    s.P = s.P - K @ (H @ s.P)
    s.P = 0.5 * (s.P + s.P.T)
    s.p = s.p + dx[0:3]
    s.v = s.v + dx[3:6]
    s.R = s.R @ _expm(dx[6:9])
    s.bg = s.bg + dx[9:12]
    s.ba = s.ba + dx[12:15]
    dpsi = float(hphi @ (R @ dx[6:9]))
    n = len(s.hist_p)
    for i in range(n):
        s.hist_p[i] = s.hist_p[i] + dx[0:3] - dx[3:6] * (DT * (n - 1 - i))
        s.hist_yaw[i] = s.hist_yaw[i] + dpsi


def _mix(Tsum, tau):
    """Rotor thrusts for a collective and torques, roll/pitch first, then collective, yaw last."""
    rp = MIX_INV @ np.array([0.0, tau[0], tau[1], 0.0])
    yw = MIX_INV @ np.array([0.0, 0.0, 0.0, tau[2]])
    lo, hi = 0.0, TMAX
    spread = rp.max() - rp.min()
    if spread > hi - lo:
        rp = rp * ((hi - lo) / spread)
    base = Tsum / 4.0
    base = min(max(base, lo - rp.min()), hi - rp.max())
    T = base + rp
    # yaw with the remaining margin
    sc = 1.0
    for i in range(4):
        if yw[i] > 1e-9:
            sc = min(sc, (hi - T[i]) / yw[i])
        elif yw[i] < -1e-9:
            sc = min(sc, (lo - T[i]) / yw[i])
    sc = max(0.0, sc)
    return T + sc * yw


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    s = _S
    if int(obs["step"]) == 0:
        _reset(obs)
    t = float(obs["time"])
    gyro = np.asarray(obs["gyro"], dtype=float)
    acc = np.asarray(obs["accel"], dtype=float)
    # motor lag state (command units) after the previous command
    if int(obs["step"]) > 0:
        s.Tst = s.Tst + (np.asarray(obs["last_action"], dtype=float) - s.Tst) * BETA
        _ekf_predict(s, gyro, acc)
    # history of predicted origin pose, then the delayed tracker correction
    p_org = s.p - s.R @ R_IMU
    s.hist_p.append(p_org)
    s.hist_p.pop(0)
    s.hist_yaw.append(math.atan2(s.R[1, 0], s.R[0, 0]))
    s.hist_yaw.pop(0)
    if obs["tracker_fresh"] and int(obs["step"]) >= 6:
        _ekf_update(s, np.asarray(obs["tracker_pos"], dtype=float), float(obs["tracker_yaw"]))
    R = s.R
    w = gyro - s.bg
    p = s.p - R @ R_IMU
    v = s.v + R @ np.cross(w, COM - R_IMU)       # velocity of the centre of mass (the origin swings below it)
    zb = R[:, 2]
    Tsum_state = float(np.sum(s.Tst))

    # specific force at the centre of mass: remove the IMU lever-arm terms (angular acceleration from the thrust state)
    alpha_est = J_INV @ (s.eta * M_NOM * (MIX[1:4] @ s.Tst) - np.cross(w, J_NOM @ w))
    f_com = acc - s.ba - np.cross(alpha_est, R_CI) - np.cross(w, np.cross(w, R_CI))
    # thrust-per-mass estimate in the hover
    t_launch = P["t_launch"]
    if 0.25 < t < t_launch:
        s.eta += 0.02 * (f_com[2] / Tsum_state - s.eta)

    # non-thrust specific force: drag model + low-passed residual (INDI)
    d_model = P["k_drag"] * _drag_acc(v, R)
    f_world = R @ f_com
    d_raw = f_world - s.eta * Tsum_state * zb - d_model
    a_f = DT / (P["tau_d"] + DT)
    s.d_res = s.d_res + a_f * (d_raw - s.d_res)
    d_hat = d_model + s.d_res

    # reference
    tr = s.tr
    tau = t - t_launch
    if tau < 0.0:
        # hover, moving to the run-up point between 0.3 s and t_launch - 0.4 s (min-jerk)
        t1, t2 = 0.3, max(0.4, t_launch - 0.4)
        x = min(max((t - t1) / (t2 - t1), 0.0), 1.0)
        sx = 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5
        dsx = (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / (t2 - t1) if 0 < x < 1 else 0.0
        ddsx = (60 * x - 180 * x ** 2 + 120 * x ** 3) / (t2 - t1) ** 2 if 0 < x < 1 else 0.0
        dp = tr["p0"] - s.p_init
        p_ref, v_ref, a_ref, j_ref = s.p_init + sx * dp, dsx * dp, ddsx * dp, np.zeros(3)
    else:
        # time scaling: the reference clock slows down when the vehicle lags along the track
        p_ref, v_ref, a_ref, j_ref = _ref(tr, s.tau)
        nv = float(np.linalg.norm(v_ref))
        rate = 1.0
        if nv > 0.5:
            lag = float((p - p_ref) @ v_ref) / (nv * max(nv, 2.0))     # s ahead (+) / behind (-)
            rate = min(max(1.0 + P["k_s"] * lag, P["rate_min"]), P["rate_max"])
        s.tau += DT * rate
        p_ref, v_ref, a_ref, j_ref = _ref(tr, s.tau)
        # feedforward from slightly ahead: compensates the lag of the attitude loop
        _, _, a_ref, j_ref = _ref(tr, s.tau + P["t_ff"])
        v_ref = v_ref * rate
        a_ref = a_ref * rate * rate
        j_ref = j_ref * rate ** 3
        gi = int(obs["gate_index"])
        if gi < 4 and s.tau > tr["gate_t"][gi] + 0.25:
            s.recover = True
    if s.recover:
        tgt = np.asarray(obs["target_pos"], dtype=float)
        e = tgt - p
        dist = np.linalg.norm(e)
        vdes = e / max(dist, 1e-6) * min(6.0, 2.0 * dist)
        p_ref, v_ref, a_ref, j_ref = p, vdes, np.zeros(3), np.zeros(3)

    ep = p_ref - p
    ev = v_ref - v
    if tau < 0.0:
        s.ei = s.ei + ep * DT
        s.ei = np.clip(s.ei, -0.5, 0.5)
    a_des = a_ref + P["kp"] * ep + P["kd"] * ev + P["ki"] * s.ei
    # terminal guidance on the last gate: zero-effort miss over the time to go
    gi = int(obs["gate_index"])
    if tau > 0.0 and not s.recover and P["k_zem"] > 0.0 and gi == 3:
        rel = np.asarray(obs["target_pos"], dtype=float) - p
        vn = float(np.linalg.norm(v))
        if vn > 2.0:
            t_go = float(rel @ v) / (vn * vn)
            if 0.03 < t_go < P["t_zem"]:
                zem = rel - v * t_go
                acc = P["k_zem"] * zem / (t_go * t_go)
                na = float(np.linalg.norm(acc))
                if na > P["zem_cap"]:
                    acc *= P["zem_cap"] / na
                a_des = a_des + acc
    if s.recover:
        a_des = a_ref + 4.0 * ev + 0.0 * ep
        n = np.linalg.norm(a_des)
        if n > 12.0:
            a_des *= 12.0 / n
    A = a_des - GV - d_hat
    if A[2] < 2.0:
        A[2] = 2.0
    # thrust limit: keep the vertical component, shorten the horizontal one
    a_lim = s.eta * 4.0 * 12.2
    nA = float(np.linalg.norm(A))
    if nA > a_lim:
        Az = min(A[2], 0.95 * a_lim)
        h = math.sqrt(max(a_lim * a_lim - Az * Az, 0.0))
        nh = math.hypot(A[0], A[1])
        if nh > 1e-9:
            A = np.array([A[0] * h / nh, A[1] * h / nh, Az])
        nA = float(np.linalg.norm(A))
    zd = A / nA
    # reduced attitude: rotate the thrust axis onto zd (body-frame axis-angle); heading held weakly at 0
    zdb = R.T @ zd
    sn = math.hypot(zdb[0], zdb[1])
    ang = math.atan2(sn, zdb[2])
    e_tilt = np.array([-zdb[1], zdb[0], 0.0]) * (ang / sn if sn > 1e-9 else 0.0)
    # feedforward: rotation rate of the reference thrust direction
    Aref = a_ref - GV
    nr = float(np.linalg.norm(Aref))
    zr = Aref / nr
    zdot = (j_ref - (j_ref @ zr) * zr) / nr
    w_ff = R.T @ np.cross(zr, zdot)
    yaw_err = _wrap(-math.atan2(R[1, 0], R[0, 0])) if R[2, 2] > 0.3 else 0.0
    w_d = np.array([P["k_att"] * e_tilt[0] + w_ff[0], P["k_att"] * e_tilt[1] + w_ff[1], P["k_yaw"] * yaw_err * R[2, 2]])
    k_rate = np.array([P["k_rate"], P["k_rate"], P["k_rate_yaw"]])
    alpha = k_rate * (w_d - w)
    tau_cmd = (J_NOM @ alpha + np.cross(w, J_NOM @ w)) / (s.eta * M_NOM)
    # collective: projection on the body axis, capped so that the vertical thrust does not exceed the desired one
    Tproj = float(A @ zb)
    if zb[2] > 0.2:
        Tproj = min(Tproj, A[2] / zb[2])
    Tsum = max(Tproj, 0.0) / s.eta
    T_des = _mix(Tsum, tau_cmd)
    u = s.Tst + P["k_lead"] * (T_des - s.Tst)
    u = np.clip(u, 0.0, TMAX)
    s.last_u = u
    return u
