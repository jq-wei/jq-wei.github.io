"""Gate-course controller for the rotor-thrust quadrotor (min_time task): hover, then fly a precomputed reference.

Control law, outside in (every gain is an entry of PARAMS):

1. Estimator. Strapdown integration of the gyro (attitude) and of the accelerometer (velocity and position of the
   body origin, IMU lever arm removed). Each tracker frame (50 Hz, 30 ms old) is compared with the estimate stored
   for that instant; the innovation corrects position, velocity and tilt (a horizontal acceleration mismatch is a
   tilt error) with the gains of a triple pole at OBS_W, the tracker yaw corrects the heading (OBS_KYAW), and
   while hovering the attitude corrections are integrated into the gyro bias (OBS_TB).
2. Reference. The vehicle holds the start point for T_HOVER seconds (the lap clock is not running yet; the
   estimator, the thrust-per-mass ratio S and the centre-of-mass trim converge), then follows TRAJ: a C3 piecewise
   degree-7 curve through the four gate centres, computed offline (scipy SLSQP) as the minimum-time curve of a
   point mass whose thrust demand stays below 87 % of the nominal maximum. That demand counts the drag along the
   thin axis with a 5 m/s wind blowing against the tilt of the thrust axis (the worst direction) and the rotor
   headroom taken by the angular acceleration; the thrust direction turns at 6.5 rad/s at most and the thrust
   never falls below 13 m/s^2 in flight, which keeps differential authority. A governor slows the reference
   clock (K_SIG, LAG0, SIG_MIN) when the vehicle falls behind along the path, and asks for catch-up
   acceleration (K_CATCH).
3. Position loop, closed on the centre of mass (the tracked origin hangs 5.4 cm below it, a non-minimum-phase
   output): a_cmd = a_ref + KP (p_ref - p) + KV (v_ref - v). The thrust axis must deliver F = a_cmd + g - d_lat,
   d_lat being the measured specific force normal to the thrust axis (lateral drag, low-pass TAU_LAT). If F
   exceeds what the rotors can still deliver (HEADROOM, TAU_AVAIL), the force across the path is kept and the
   force along the path gives way. The collective is incremental (INDI): the summed thrust moves by the gap
   between F along the thrust axis and the low-passed accelerometer z (TAU_AX), which absorbs the thin-axis drag,
   the mass and the thrust coefficient.
4. Attitude loop (thrust axis only): body-rate command = K_ATT * tilt error + FF_RATE * rotation rate of the
   reference thrust direction; the heading is held weakly (K_YAW).
5. Rate loop: angular acceleration = K_RATE (w_cmd - w) + FF_ACC * reference angular acceleration read T_LEAD
   ahead (motor lag); torque = J * that + w x J w + trim, the trim being the thrust times an adapted
   centre-of-mass offset (TI_COM). Mixer: tilt torque and collective are met exactly with the yaw torque as slack
   whenever the rotor limits allow, otherwise tilt torque first. The 30 ms motor lag is inverted by a lead (LEAD)
   on a model of the rotor thrusts.

If a gate is missed a slow fallback flies to obs["target_pos"] (it only serves the `finish` rung).
"""

import math

import numpy as np

from hl_control.envs.quadrotor import nominal_parameters

PARAMS = {
    "T_HOVER": 2.0,     # s of hover on the start point before the launch
    "KP": 16.0,         # 1/s^2, position gain
    "KV": 7.5,          # 1/s, velocity gain
    "K_ATT": 11.0,      # 1/s, tilt error -> body rate
    "K_RATE": 26.0,     # 1/s, body-rate error -> angular acceleration (roll, pitch)
    "K_RATE_Z": 6.0,    # 1/s, the same for yaw
    "K_YAW": 2.0,       # 1/s, heading error -> yaw rate
    "LEAD": 2.0,        # motor lead: commanded step = LEAD * (wanted thrust - modelled thrust)
    "TAU_LAT": 0.08,    # s, low-pass of the lateral specific force (drag normal to the thrust axis)
    "TAU_AX": 0.03,     # s, low-pass of the axial specific force and of the modelled thrust (INDI collective)
    "TI_COM": 0.35,     # s, integration time of the centre-of-mass trim
    "OBS_W": 6.0,       # rad/s, bandwidth of the tracker correction (position, velocity, tilt)
    "OBS_KYAW": 0.15,   # per frame, heading correction
    "OBS_TB": 0.7,      # s, time constant of the gyro-bias estimate (hover only)
    "FF_RATE": 1.0,     # weight of the feed-forward body rate
    "FF_ACC": 1.0,      # weight of the feed-forward angular acceleration
    "T_LEAD": 0.03,     # s, look-ahead of the feed-forward angular acceleration (motor lag)
    "K_SIG": 2.5,       # 1/m, slow-down of the reference clock per metre of along-path lag
    "LAG0": 0.08,       # m, lag tolerated before the clock slows
    "SIG_MIN": 0.55,    # lowest clock rate
    "K_CATCH": 3.0,     # 1/s, forward acceleration per unit of speed deficit while the clock runs slow
    "HEADROOM": 0.96,   # share of the total thrust that the collective may use (the rest is kept for torque)
    "TAU_AVAIL": 0.4,   # s, low-pass of the deliverable thrust-axis force (it depends on the attitude through the drag)
}

# reference: piece durations [s] and, at the end of each piece, position [m], velocity [m/s], acceleration [m/s^2]
# and jerk [m/s^3]; "gate" marks the knots that are gate centres (the start is the hover point, at rest)
TRAJ = {
    "T": [0.40965, 0.94576, 0.47643, 0.35659, 0.55023, 0.60453, 0.27795, 0.27719],
    "P": [[0.33474, 0.33259, 2.21205],
              [8.00000, 3.00000, 3.00000],
              [12.25114, -0.46963, 1.76187],
              [14.00000, -3.00000, 1.50000],
              [15.74853, -1.73056, 3.10216],
              [20.00000, 2.00000, 3.50000],
              [23.16100, 1.83310, 2.97278],
              [26.00000, 0.00000, 2.00000]],
    "V": [[2.75583, 2.73092, 1.42916],
              [11.29212, -3.09463, -1.39831],
              [5.95869, -9.37505, -2.92365],
              [4.27760, -3.63523, 1.74606],
              [3.23105, 6.72061, 2.35683],
              [10.91387, 2.54453, -1.08948],
              [11.27686, -3.77305, -2.70425],
              [8.76351, -9.20432, -4.31469]],
    "A": [[12.90197, 12.05345, 4.36652],
              [-4.69703, -22.07970, -5.81000],
              [-11.85527, 3.06525, 7.27612],
              [-4.04352, 23.19592, 12.46611],
              [7.32721, 8.11076, -5.13346],
              [7.66678, -21.23670, -5.81000],
              [-4.26325, -22.11156, -5.81000],
              [-13.90044, -16.77455, -5.80985]],
    "J": [[-0.71482, -16.84262, -15.63950],
              [-62.69646, 3.79575, -2.13492],
              [56.98868, 83.56008, 76.77007],
              [-16.38128, 17.35552, -36.85692],
              [43.06546, -47.76041, -4.89511],
              [-49.95716, -30.27296, 0.10948],
              [-35.05791, 14.61562, 0.00912],
              [-35.26795, 21.57284, 0.00980]],
    "gate": [False, True, False, True, False, True, False, True],
}

_N = nominal_parameters()
G = 9.81
M_NOM = float(_N["mass"])
J = np.array(_N["inertia"], dtype=float)
ROTORS = np.array(_N["rotor_positions"], dtype=float)
COM = np.array(_N["com"], dtype=float)
H_COM = float(COM[2])                                  # centre of mass above the body origin
R_IO_Z = float(_N["imu_position"][2])                 # IMU above the body origin
R_IC_Z = float(_N["imu_position"][2] - COM[2])        # IMU relative to the centre of mass (negative: below)
T_MAX = float(_N["thrust_max"])
# thrust commands -> (total thrust, roll torque, pitch torque, yaw torque) about the nominal centre of mass
MIX = np.vstack([np.ones(4), ROTORS[:, 1] - COM[1], -(ROTORS[:, 0] - COM[0]), np.array(_N["yaw_coeff"], dtype=float)])
MIX_INV = np.linalg.inv(MIX)
YAW_PATTERN = np.sign(MIX_INV[:, 3])                   # rotor pattern that changes the yaw torque only
YAW_GAIN = float(abs(MIX_INV[0, 3]))                   # rotor newtons along the pattern per N m of yaw torque
MOTOR_TAU = 0.03
DT = 0.005
LAG = 1.0 - math.exp(-0.5 * DT / MOTOR_TAU)            # per physics substep (two per control step)
START = np.array([0.0, 0.0, 2.0])
I3 = np.eye(3)


_M4 = np.linalg.inv(np.array([[1.0, 1, 1, 1], [4, 5, 6, 7], [12, 20, 30, 42], [24, 60, 120, 210]]))


def _septic(p0, v0, a0, j0, p1, v1, a1, j1, T):
    """Coefficients (8, 3) of the degree-7 piece that matches position, velocity, acceleration and jerk at both ends."""
    c = np.zeros((8, 3))
    c[0], c[1], c[2], c[3] = p0, v0, a0 / 2.0, j0 / 6.0
    rhs = np.array([p1 - (c[0] + c[1] * T + c[2] * T ** 2 + c[3] * T ** 3),
                    (v1 - (c[1] + 2 * c[2] * T + 3 * c[3] * T ** 2)) * T,
                    (a1 - (2 * c[2] + 6 * c[3] * T)) * T ** 2,
                    (j1 - 6 * c[3]) * T ** 3])
    d = _M4 @ rhs
    for k in range(4):
        c[4 + k] = d[k] / T ** (4 + k)
    return c


def _build_reference():
    T = np.asarray(TRAJ["T"], dtype=float)
    P = np.vstack([START, np.asarray(TRAJ["P"], dtype=float)])
    V = np.vstack([np.zeros(3), np.asarray(TRAJ["V"], dtype=float)])
    A = np.vstack([np.zeros(3), np.asarray(TRAJ["A"], dtype=float)])
    Jk = np.vstack([np.zeros(3), np.asarray(TRAJ["J"], dtype=float)])
    coef = [_septic(P[k], V[k], A[k], Jk[k], P[k + 1], V[k + 1], A[k + 1], Jk[k + 1], T[k]) for k in range(len(T))]
    knots = np.concatenate([[0.0], np.cumsum(T)])
    gate_times = [float(knots[k + 1]) for k in range(len(T)) if TRAJ["gate"][k]]
    # derivative coefficient tables: velocity, acceleration, jerk
    tables = []
    for c in coef:
        cv = np.array([c[k] * k for k in range(1, 8)])
        ca = np.array([cv[k] * k for k in range(1, 7)])
        cj = np.array([ca[k] * k for k in range(1, 6)])
        tables.append((c, cv, ca, cj))
    return tables, knots, gate_times


def _horner(c, t):
    out = c[-1]
    for k in range(len(c) - 2, -1, -1):
        out = out * t + c[k]
    return out


def _exp(w):
    """Rotation matrix of the rotation vector w."""
    x, y, z = w[0], w[1], w[2]
    th2 = x * x + y * y + z * z
    if th2 < 1e-10:
        a, b = 1.0, 0.5
    else:
        th = math.sqrt(th2)
        a, b = math.sin(th) / th, (1.0 - math.cos(th)) / th2
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return I3 + a * K + b * (K @ K)


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class _Pilot:
    """Estimator and controller state of one episode; reset() is called on the first step of each episode."""

    def reset(self, obs):
        P = PARAMS
        yaw = float(obs["tracker_yaw"])
        c, s = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.p = np.array(obs["tracker_pos"], dtype=float)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.gyro_prev = np.array(obs["gyro"], dtype=float)
        self.fo_prev = np.array([0.0, 0.0, G])          # world specific force at the previous step
        self.f_w = np.array([0.0, 0.0, G])
        self.p_hist = np.tile(self.p, (8, 1))
        self.yaw_hist = np.full(8, yaw)
        self.yaw_ref = yaw
        self.T_model = np.full(4, M_NOM * G / 4.0)      # modelled rotor thrusts (command units)
        self.S = 1.0                                    # thrust coefficient / mass scale: specific force = S * sum(T) / M_NOM
        self.az_slow = G
        self.ts_slow = M_NOM * G
        self.az_f = G
        self.ts_f = M_NOM * G
        self.d_lat = np.zeros(3)
        self.trim = np.zeros(2)                         # centre-of-mass offset trim [m]: torque per newton of thrust
        self.mode = 0                                   # 0 hover, 1 race, 2 fallback
        self.t_launch = math.inf
        self.tau = 0.0
        self.sig = 1.0
        self.w_r_prev = None
        self.n_avail = 4.0 * T_MAX / M_NOM
        self.coef, self.knots, self.gate_times = _REFERENCE
        w = P["OBS_W"]
        frame_dt = float(obs["tracker_dt"])
        self.k_p, self.k_v, self.k_a = 3.0 * w * frame_dt, 3.0 * w * w * frame_dt, w ** 3 * frame_dt
        self.delay = int(round(float(obs["tracker_delay"]) / DT))

    def reference(self, tau):
        """Position, velocity, acceleration, jerk of the reference at time tau after the launch."""
        knots = self.knots
        if tau >= knots[-1]:                            # past the last gate: keep the final velocity
            c, cv, _, _ = self.coef[-1]
            T = knots[-1] - knots[-2]
            v = _horner(cv, T)
            return _horner(c, T) + v * (tau - knots[-1]), v, np.zeros(3), np.zeros(3)
        k = 0
        while tau >= knots[k + 1]:
            k += 1
        c, cv, ca, cj = self.coef[k]
        t = tau - knots[k]
        return _horner(c, t), _horner(cv, t), _horner(ca, t), _horner(cj, t)

    def reference_aj(self, tau):
        """Acceleration and jerk of the reference only (for the look-ahead feed-forward)."""
        knots = self.knots
        if tau >= knots[-1]:
            return np.zeros(3), np.zeros(3)
        k = 0
        while tau >= knots[k + 1]:
            k += 1
        _, _, ca, cj = self.coef[k]
        t = tau - knots[k]
        return _horner(ca, t), _horner(cj, t)

    def step(self, obs):
        P = PARAMS
        i = int(obs["step"])
        if i == 0:
            self.reset(obs)
        t = i * DT
        gyro = np.array(obs["gyro"], dtype=float)
        acc = np.array(obs["accel"], dtype=float)
        w = gyro - self.bg
        wx, wy, wz = w[0], w[1], w[2]
        alpha = (gyro - self.gyro_prev) / DT
        # specific force at the body origin (tracker point) and at the nominal centre of mass, body axes
        cx, cy, cz = alpha[1] + wx * wz, -alpha[0] + wy * wz, -(wx * wx + wy * wy)   # (alpha x ez + w x (w x ez))
        f_o = np.array([acc[0] - cx * R_IO_Z, acc[1] - cy * R_IO_Z, acc[2] - cz * R_IO_Z])
        f_c = np.array([acc[0] - cx * R_IC_Z, acc[1] - cy * R_IC_Z, acc[2] - cz * R_IC_Z])

        # ----- estimator: strapdown step from the previous call to this one -----
        if i > 0:
            w_avg = 0.5 * (self.gyro_prev + gyro) - self.bg
            self.R = self.R @ _exp(w_avg * DT)
            fo_w = self.R @ f_o
            self.f_w = 0.5 * (self.fo_prev + fo_w)
            a_w = self.f_w.copy()
            a_w[2] -= G
            self.p = self.p + self.v * DT + a_w * (0.5 * DT * DT)
            self.v = self.v + a_w * DT
            self.fo_prev = fo_w
        else:
            self.fo_prev = self.R @ f_o
        self.gyro_prev = gyro
        R = self.R
        yaw = math.atan2(R[1, 0], R[0, 0])
        slot = i % 8
        self.p_hist[slot] = self.p
        self.yaw_hist[slot] = yaw

        if obs["tracker_fresh"] and i > 0:
            old = max(i - self.delay, 0) % 8
            e = np.asarray(obs["tracker_pos"], dtype=float) - self.p_hist[old]
            e_yaw = _wrap(float(obs["tracker_yaw"]) - self.yaw_hist[old])
            dp = self.k_p * e
            self.p = self.p + dp
            self.p_hist += dp
            self.v = self.v + self.k_v * e
            f = self.f_w
            g = self.k_a / max(f[0] * f[0] + f[1] * f[1] + f[2] * f[2], 25.0)
            dth = np.array([g * (f[1] * e[2] - f[2] * e[1]), g * (f[2] * e[0] - f[0] * e[2]),
                            g * (f[0] * e[1] - f[1] * e[0]) + P["OBS_KYAW"] * e_yaw])
            self.yaw_hist += P["OBS_KYAW"] * e_yaw
            R = _exp(dth) @ R
            self.R = R
            if self.mode == 0:
                self.bg = self.bg - (R.T @ dth) / P["OBS_TB"]
            yaw = math.atan2(R[1, 0], R[0, 0])
        p, v = self.p, self.v
        b3 = R[:, 2]

        # ----- disturbance estimates from the accelerometer -----
        t_sum = float(self.T_model.sum())
        lat = R[:, 0] * f_c[0] + R[:, 1] * f_c[1]
        self.d_lat += (DT / P["TAU_LAT"]) * (lat - self.d_lat)
        k_ax = DT / P["TAU_AX"]
        self.az_f += k_ax * (f_c[2] - self.az_f)
        self.ts_f += k_ax * (t_sum - self.ts_f)
        if self.mode == 0:                              # thrust-per-newton ratio, identified in hover
            self.az_slow += (DT / 0.3) * (f_c[2] - self.az_slow)
            self.ts_slow += (DT / 0.3) * (t_sum - self.ts_slow)
            if t > 0.4:
                self.S = min(1.25, max(0.8, M_NOM * self.az_slow / self.ts_slow))
        S = self.S

        # ----- mode logic and reference -----
        target = np.asarray(obs["target_pos"], dtype=float)
        # the position loop is closed on the centre of mass: the tracked body origin hangs H_COM below it, and
        # its position responds to a tilt with a right-half-plane zero at sqrt(g / H_COM) = 13 rad/s
        p_c = p + H_COM * b3
        v_c = v + H_COM * (R[:, 0] * wy - R[:, 1] * wx)
        if self.mode == 0:
            hold = np.asarray(obs["start_pos"], dtype=float)
            err = p - hold
            settled = (err @ err) < 0.03 ** 2 and (v @ v) < 0.12 ** 2
            if (t >= P["T_HOVER"] and settled) or t >= 3.2:
                self.mode = 1
                self.t_launch = t
                self.tau = 0.0
                self.sig = 1.0
        if self.mode == 1:
            tau = self.tau
            passed = int(obs["gate_index"])
            missed = any(tau > tg + 0.12 and passed <= k for k, tg in enumerate(self.gate_times))
            if missed:
                self.mode = 2
        j_r = None
        tangent = None
        al_r = None
        if self.mode == 0:
            a_cmd = P["KP"] * (hold - p_c) - P["KV"] * v_c
            a_cmd[2] += P["KP"] * H_COM
        elif self.mode == 1:
            p_r, v_r, a_r, j_r = self.reference(tau)
            # governor: when the vehicle falls behind along the path (short of thrust in a headwind), the
            # reference clock slows down (rate sig) instead of dragging the vehicle off the path
            speed = math.sqrt(v_r @ v_r)
            sig_target = 1.0
            if speed > 1.0:
                tangent = v_r / speed
                lag = float((p_r - p) @ tangent)
                sig_target = min(1.0, max(P["SIG_MIN"], 1.0 - P["K_SIG"] * (lag - P["LAG0"])))
            self.sig += 0.25 * (sig_target - self.sig)
            sig = self.sig
            self.tau = tau + sig * DT
            if speed > 1.0:
                # at clock rate sig the vehicle is slower than planned: the turning part of the acceleration scales
                # with sig^2, the speeding-up part is kept and a catch-up term is added (the allocation below
                # trims the along-path force to what the rotors can give)
                a_t = float(a_r @ tangent)
                a_n = a_r - a_t * tangent
                a_r = (sig * sig) * a_n + ((a_t if a_t > 0.0 else sig * sig * a_t) + P["K_CATCH"] * (1.0 - sig) * speed) * tangent
            else:
                a_r = (sig * sig) * a_r
            v_r = sig * v_r
            j_r = (sig * sig * sig) * j_r
            f_r = a_r.copy()
            f_r[2] += G
            n_r = math.sqrt(f_r @ f_r)
            b3r = f_r / n_r
            b3r_dot = (j_r - (j_r @ b3r) * b3r) / n_r
            # feed-forward angular acceleration: derivative of the rotation rate of the reference thrust direction,
            # read T_LEAD ahead on the reference to make up for the motor lag
            a_l, j_l = self.reference_aj(tau + P["T_LEAD"])
            f_l = (sig * sig) * a_l
            f_l[2] += G
            n_l = math.sqrt(f_l @ f_l)
            b_l = f_l / n_l
            j_l = (sig * sig * sig) * j_l
            bd_l = (j_l - (j_l @ b_l) * b_l) / n_l
            w_r = np.array([b_l[1] * bd_l[2] - b_l[2] * bd_l[1], b_l[2] * bd_l[0] - b_l[0] * bd_l[2],
                            b_l[0] * bd_l[1] - b_l[1] * bd_l[0]])
            if self.w_r_prev is not None:
                al_r = (w_r - self.w_r_prev) / DT
            self.w_r_prev = w_r
            a_fb = P["KP"] * (p_r + H_COM * b3r - p_c) + P["KV"] * (v_r + H_COM * b3r_dot - v_c)
            n_fb = math.sqrt(a_fb @ a_fb)
            if n_fb > 25.0:
                a_fb *= 25.0 / n_fb
            a_cmd = a_r + a_fb
        else:
            to = target - p
            dist = math.sqrt(to @ to) + 1e-9
            v_des = to * (min(5.0, 2.5 * dist) / dist)
            a_cmd = 4.0 * (v_des - v_c)
            n_fb = math.sqrt(a_cmd @ a_cmd)
            if n_fb > 8.0:
                a_cmd *= 8.0 / n_fb

        # ----- thrust vector -----
        F = a_cmd - self.d_lat
        F[2] += G
        if F[2] < 1.0:
            F[2] = 1.0
        Fn = math.sqrt(F @ F)
        # thrust-axis force that the rotors can still deliver (measured axial force + remaining thrust headroom);
        # when the demand exceeds it, the force across the path (and the climb) is kept and the along-path force gives way
        n_now = max(6.0, self.az_f + (S / M_NOM) * (P["HEADROOM"] * 4.0 * T_MAX - self.ts_f))
        self.n_avail += (DT / P["TAU_AVAIL"]) * (n_now - self.n_avail)
        n_avail = self.n_avail
        if Fn > n_avail:
            if tangent is not None:
                ft = float(F @ tangent)
                Fp = F - ft * tangent
                fp2 = float(Fp @ Fp)
                if fp2 < n_avail * n_avail:
                    F = Fp + math.copysign(math.sqrt(n_avail * n_avail - fp2), ft) * tangent
                else:
                    F = Fp * (n_avail / math.sqrt(fp2))
            else:
                F = F * (n_avail / Fn)
            Fn = math.sqrt(F @ F)
        b3d = F / Fn
        fz_cmd = float(F @ b3)
        t_sum_des = self.ts_f + (fz_cmd - self.az_f) * M_NOM / S
        t_sum_des = min(4.0 * T_MAX, max(1.0, t_sum_des))

        # ----- attitude: tilt error in body axes -> body rates -----
        d = R.T @ b3d
        sin_th = math.sqrt(d[0] * d[0] + d[1] * d[1])
        if sin_th > 1e-6:
            th = math.atan2(sin_th, d[2])
            ex, ey = -d[1] * th / sin_th, d[0] * th / sin_th
        else:
            ex, ey = 0.0, 0.0
        wdx, wdy = P["K_ATT"] * ex, P["K_ATT"] * ey
        if j_r is not None:
            jp = j_r - (j_r @ b3d) * b3d
            b3d_dot = jp / Fn
            w_ff = np.array([b3d[1] * b3d_dot[2] - b3d[2] * b3d_dot[1], b3d[2] * b3d_dot[0] - b3d[0] * b3d_dot[2],
                             b3d[0] * b3d_dot[1] - b3d[1] * b3d_dot[0]])
            w_ffb = R.T @ w_ff
            wdx += P["FF_RATE"] * w_ffb[0]
            wdy += P["FF_RATE"] * w_ffb[1]
        n_w = math.sqrt(wdx * wdx + wdy * wdy)
        if n_w > 14.0:
            wdx, wdy = wdx * 14.0 / n_w, wdy * 14.0 / n_w
        wdz = max(-1.5, min(1.5, P["K_YAW"] * _wrap(self.yaw_ref - yaw)))

        # ----- rates -> torques -> rotor thrusts -----
        e_wx, e_wy, e_wz = wdx - wx, wdy - wy, wdz - wz
        acc_des = np.array([P["K_RATE"] * e_wx, P["K_RATE"] * e_wy, P["K_RATE_Z"] * e_wz])
        if al_r is not None:
            al_b = R.T @ al_r
            n_al = math.sqrt(al_b[0] * al_b[0] + al_b[1] * al_b[1])
            g_al = P["FF_ACC"] * min(1.0, 120.0 / max(n_al, 1e-9))
            acc_des[0] += g_al * al_b[0]
            acc_des[1] += g_al * al_b[1]
        Jw = J @ w
        tq = J @ acc_des + np.array([wy * Jw[2] - wz * Jw[1], wz * Jw[0] - wx * Jw[2], wx * Jw[1] - wy * Jw[0]])
        tq /= S
        k_i = P["K_RATE"] / (M_NOM * G * P["TI_COM"]) * DT
        self.trim[0] = max(-0.012, min(0.012, self.trim[0] + k_i * J[0, 0] * e_wx))
        self.trim[1] = max(-0.012, min(0.012, self.trim[1] + k_i * J[1, 1] * e_wy))
        tq[0] += self.trim[0] * t_sum_des
        tq[1] += self.trim[1] * t_sum_des
        tq[2] = max(-0.12, min(0.12, tq[2]))
        # mixer. Roll/pitch torque and collective fix the rotor thrusts up to the yaw pattern (-, +, -, +): the yaw
        # torque is the slack that keeps all four rotors inside [0, T_MAX]; only if no such value exists is the
        # collective shifted (tilt torque first), then clipped
        b = MIX_INV[:, 0] * t_sum_des + MIX_INV[:, 1] * tq[0] + MIX_INV[:, 2] * tq[1]
        b0, b1, b2, b3_ = float(b[0]), float(b[1]), float(b[2]), float(b[3])
        lam_lo = max(-b1, -b3_, b0 - T_MAX, b2 - T_MAX)
        lam_hi = min(T_MAX - b1, T_MAX - b3_, b0, b2)
        lam_des = YAW_GAIN * tq[2]
        if lam_lo <= lam_hi:
            lam = min(lam_hi, max(lam_lo, lam_des))
            want = b + lam * YAW_PATTERN
        else:
            want = b + (0.5 * (lam_lo + lam_hi)) * YAW_PATTERN
            hi = want.max()
            if hi > T_MAX:
                want -= hi - T_MAX
            lo = want.min()
            if lo < 0.0:
                want += min(-lo, max(0.0, T_MAX - want.max()))
            want = np.clip(want, 0.0, T_MAX)
        u = np.clip(self.T_model + P["LEAD"] * (want - self.T_model), 0.0, T_MAX)
        self.T_model += (u - self.T_model) * LAG
        self.T_model += (u - self.T_model) * LAG
        return u


_REFERENCE = _build_reference()
_PILOT = _Pilot()


def controller(obs: dict) -> np.ndarray:
    return _PILOT.step(obs)
