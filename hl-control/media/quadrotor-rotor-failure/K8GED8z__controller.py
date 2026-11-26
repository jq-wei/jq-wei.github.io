"""Quadrotor controller that survives the unannounced loss of one rotor.

Control law (model-based, no learning):

1. Estimator (strap-down INS aided by the tracker).  The gyro is integrated into a rotation matrix and the
   accelerometer, rotated to the world, into the velocity and position of the IMU point.  Every tracker frame
   (50 Hz, 30 ms late) is compared with the estimate stored six steps earlier; the innovation corrects position,
   velocity, tilt (a tilt error shows up as a horizontal acceleration error) and yaw.
     EST_POLE  triple pole [rad/s] of the position / velocity / tilt observer;  L_YAW  yaw correction per frame.
2. Plant-model observers.  The rotor thrusts are predicted with the motor-lag model.  The momentum residual
   (J dw/dt + w x Jw - predicted rotor torque) is low-pass filtered slowly (K_SLOW: torque of the centre-of-mass
   offset, used as feed-forward) and quickly (K_FAST: failure detection).  The thrust/mass ratio is identified
   from the accelerometer before the failure.  A failure of rotor i is declared when the fast residual matches
   the signature of a thrust loss on rotor i above DET_LOSS newtons on DET_COUNT consecutive steps; detection
   is armed at DET_ARM_TIME (the failure never comes before 3 s).  It typically fires about 30 ms after the loss.
3. Motor-lag identification.  Before the failure a second thrust model with a 60 ms lag runs next to the
   nominal 30 ms one, and the residual is regressed on the difference of their torques (a torque square wave of
   DITHER N m between 0.5 and 1.0 s provides excitation).  At ID_TIME the lag fraction (0 = 30 ms, 1 = 60 ms)
   is adopted if it exceeds ID_DEADZONE: thrust model, LQR gains and primary axis (SIGMA_SLOW) follow it.
   On the nominal plant the fraction stays near zero and nothing changes.
4. Position loop.  P on position (KP_POS, velocity command limited to V_MAX), P on velocity (KV, limited to
   A_XY_MAX), minus the measured external specific force (accelerometer minus predicted thrust, filtered with
   TAU_DIST: wind drag and model error), gives the desired thrust direction n_des (tilt below TILT_MAX) and
   the thrust magnitude.
5. Reduced-attitude LQR (after Mueller & D'Andrea).  A body-fixed primary axis n_B is aligned with n_des and yaw
   is given up after the failure.  The state (axis error, body-rate error, lagged rotor torque) is fed back with a
   discrete LQR gain scheduled on the yaw rate (weights Q_ATT and Q_RATE against a unit torque weight).  With
   four rotors n_B is the body z axis and the yaw rate is damped (K_YAW).  With three rotors n_B is tilted so
   that the spinning equilibrium asks SIGMA * r^2 newtons from the rotor opposite to the dead one (SIGMA_REAR /
   SIGMA_FRONT by position of the dead rotor): that keeps this rotor off its zero-thrust bound and limits the
   spin rate.  The tilt is blended in with (r / R_BLEND)^2 while the vehicle spins up.
6. Allocation.  Thrust and the two tilt torques are inverted on the live rotors; if the rotor opposite to the
   dead one would need negative thrust it is set to zero and the torque is projected on the remaining pair.
"""

import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are

PARAMS = {
    "EST_POLE": 8.0,      # rad/s, triple pole of the position/velocity/tilt observer
    "L_YAW": 0.3,         # yaw correction per tracker frame
    "KP_POS": 1.5,        # 1/s, position error -> velocity command
    "KV": 3.5,            # 1/s, velocity error -> acceleration command
    "V_MAX": 3.0,         # m/s, limit of the velocity command
    "A_XY_MAX": 6.0,      # m/s^2, limit of the horizontal acceleration command
    "TILT_MAX": 0.6,      # rad, limit of the commanded thrust direction
    "TAU_DIST": 0.3,      # s, filter of the external specific force
    "Q_ATT": 13.0,        # LQR weight on the primary-axis error
    "Q_RATE": 0.02,       # LQR weight on the body-rate error
    "K_YAW": 6.0,         # 1/s, yaw-rate damping with four rotors
    "SIGMA_REAR": 0.020,  # N/(rad/s)^2, equilibrium thrust of the opposite rotor per r^2 (rear rotor dead)
    "SIGMA_FRONT": 0.012,  # same when a front rotor is dead
    "R_BLEND": 5.0,       # rad/s, yaw rate at which the tilted primary axis is fully blended in
    "K_FAST": 120.0,      # 1/s, bandwidth of the detection residual
    "K_SLOW": 5.0,        # 1/s, bandwidth of the torque-disturbance estimate
    "DET_LOSS": 1.1,      # N, thrust loss that declares a failure
    "DET_COUNT": 2,       # consecutive steps
    "DET_ARM_TIME": 2.5,  # s, the failure never happens before 3 s
    "SIGMA_SLOW": 0.020,  # SIGMA of a dead front rotor when the motors are found to be slow (lower spin rate)
    "DITHER": 0.2,        # N m, torque square wave between 0.5 and 1.0 s that excites the lag identification
    "ID_TIME": 2.45,      # s, when the identified motor lag is adopted
    "ID_DEADZONE": 0.3,   # identified lag fraction (0 = 30 ms, 1 = 60 ms) below which the nominal lag is kept
}

DT = 0.005
GRAV = 9.81
M_NOM = 1.325
TAU_M = 0.03                      # s, motor lag of the nominal plant
TAU_SLOW = 0.06                   # s, second hypothesis of the lag identification
LAG = 1.0 - math.exp(-DT / TAU_M)
LAG_SLOW = 1.0 - math.exp(-DT / TAU_SLOW)
T_MAX = 13.0
try:
    from hl_control.envs.quadrotor import nominal_parameters

    _np = nominal_parameters()
    J = np.array(_np["inertia"], dtype=float)
    _xy = np.array(_np["rotor_positions"], dtype=float)[:, :2] - np.array(_np["com"], dtype=float)[:2]
    _cy = np.array(_np["yaw_coeff"], dtype=float)
    IMU_Z = float(_np["imu_position"][2])
    COM_Z = float(_np["com"][2])
except Exception:  # the numbers of the task statement
    J = np.array([[0.0367, 0.0, -0.0021], [0.0, 0.0254, 0.0], [-0.0021, 0.0, 0.0605]])
    _xy = np.array([[-0.14, -0.18], [-0.14, 0.18], [0.14, 0.18], [0.14, -0.18]])
    _cy = np.array([-0.0201, 0.0201, -0.0201, 0.0201])
    IMU_Z, COM_Z = 0.02, 0.054
GEO = np.vstack([_xy[:, 1], -_xy[:, 0], _cy])          # torque (x, y, z) per newton of each rotor
ALLOC4_INV = np.linalg.inv(np.vstack([np.ones(4), GEO]))
OPP = (2, 3, 0, 1)
ALLOC3_INV = []
for _i in range(4):
    _live = [k for k in range(4) if k != _i]
    ALLOC3_INV.append((_live, np.linalg.inv(np.vstack([np.ones(3), GEO[:2, _live]]))))
R_IMU = np.array([0.0, 0.0, IMU_Z])
RHO_IMU = np.array([0.0, 0.0, IMU_Z - COM_Z])           # IMU relative to the nominal centre of mass
EZ = np.array([0.0, 0.0, 1.0])
HIST = 16


def _lqr_table(q_att, q_rate, tau_m):
    ixx, iyy, izz = J[0, 0], J[1, 1], J[2, 2]
    rs = np.arange(-48.0, 48.01, 1.5)
    q = np.diag([q_att, q_att, q_rate, q_rate, 0.0, 0.0])
    rw = np.eye(2)
    ks = []
    for r in rs:
        a = np.zeros((6, 6))
        a[0, 1], a[0, 3] = r, -1.0
        a[1, 0], a[1, 2] = -r, 1.0
        a[2, 3], a[2, 4] = -(izz - iyy) * r / ixx, 1.0 / ixx
        a[3, 2], a[3, 5] = -(ixx - izz) * r / iyy, 1.0 / iyy
        a[4, 4] = a[5, 5] = -1.0 / tau_m
        b = np.zeros((6, 2))
        b[4, 0] = b[5, 1] = 1.0 / tau_m
        blk = np.zeros((8, 8))
        blk[:6, :6], blk[:6, 6:] = a, b
        md = expm(blk * DT)
        ad, bd = md[:6, :6], md[:6, 6:]
        p = solve_discrete_are(ad, bd, q, rw)
        ks.append(np.linalg.solve(rw + bd.T @ p @ bd, bd.T @ p @ ad))
    return rs, np.array(ks)


def _rot(w):
    """Rotation matrix of the rotation vector w."""
    x, y, z = w
    th2 = x * x + y * y + z * z
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    if th2 < 1e-12:
        return np.eye(3) + k
    th = math.sqrt(th2)
    return np.eye(3) + (math.sin(th) / th) * k + ((1.0 - math.cos(th)) / th2) * (k @ k)


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class _Controller:
    def __init__(self):
        p = PARAMS
        self.rs, self.ks_nom = _lqr_table(p["Q_ATT"], p["Q_RATE"], TAU_M)
        _, self.ks_slow = _lqr_table(p["Q_ATT"], p["Q_RATE"], TAU_SLOW)
        lam, tu = p["EST_POLE"], 4 * DT
        self.l1, self.l2, self.l3 = 3 * lam * tu, 3 * lam * lam * tu, lam ** 3 * tu
        self.bf = 1.0 - math.exp(-p["K_FAST"] * DT)
        self.bs = 1.0 - math.exp(-p["K_SLOW"] * DT)
        # primary-axis tilt (a, b) such that w_eq = r (a, b, 1) needs sigma r^2 from the rotor opposite to the dead one
        self.ab_nom = [self._axis(i, p["SIGMA_REAR"] if i < 2 else p["SIGMA_FRONT"]) for i in range(4)]
        self.ab_slow = [self._axis(i, p["SIGMA_REAR"] if i < 2 else p["SIGMA_SLOW"]) for i in range(4)]
        self.dbg = {}                                   # last step's internals, for diagnostics only

    @staticmethod
    def _axis(i, sig):
        g = GEO[:2, OPP[i]]
        a = (sig * g[1] - J[0, 2]) / (J[0, 0] - J[2, 2])
        for _ in range(3):
            a = (sig * g[1] - J[0, 2] * (1.0 - a * a)) / (J[0, 0] - J[2, 2])
        b = sig * g[0] / (J[2, 2] - J[1, 1] + J[0, 2] * a)
        return a, b

    def reset(self, obs):
        yaw = float(obs["tracker_yaw"])
        c, s = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.p = np.asarray(obs["tracker_pos"], dtype=float) + self.R @ R_IMU
        self.v = np.zeros(3)
        self.gyro_prev = np.asarray(obs["gyro"], dtype=float).copy()
        self.acc_prev = np.asarray(obs["accel"], dtype=float).copy()
        self.ph = np.tile(self.p - self.R @ R_IMU, (HIST, 1))
        self.yh = np.full(HIST, yaw)
        self.tm = np.full(4, M_NOM * GRAV / 4.0)
        self.u_last = self.tm.copy()
        self.d_fast = np.zeros(3)
        self.d_slow = np.zeros(3)
        self.d_hist = np.zeros((64, 3))
        self.kap_num = GRAV
        self.kap_den = GRAV
        self.kappa = 1.0
        self.d_w = np.zeros(3)
        self.failed = -1
        self.det_rotor = -1
        self.det_count = 0
        self.t_detect = -1.0
        # motor-lag identification: thrust model with the slow lag, regression sums per axis (n, S_phi, S_y, S_phi_y, S_phi_phi)
        self.lag = LAG
        self.ks = self.ks_nom
        self.ab = self.ab_nom
        self.tm_b = self.tm.copy()
        self.id_sum = np.zeros((2, 5))
        self.theta_hat = 0.0
        self.theta = 0.0

    def step(self, obs):
        P = PARAMS
        k = int(obs["step"])
        gyro = np.asarray(obs["gyro"], dtype=float)
        acc = np.asarray(obs["accel"], dtype=float)
        if k == 0:
            self.reset(obs)
        else:
            # ----- strap-down propagation -----
            r_old = self.R
            self.R = r_old @ _rot(0.5 * (gyro + self.gyro_prev) * DT)
            a_w = 0.5 * (r_old @ self.acc_prev + self.R @ acc)
            a_w[2] -= GRAV
            self.p += self.v * DT + 0.5 * a_w * DT * DT
            self.v += a_w * DT
        R = self.R
        f_w = R @ acc
        # ----- tracker correction (the frame shows the pose six steps ago) -----
        if k > 0 and obs["tracker_fresh"]:
            j = max(k - int(round(obs["tracker_delay"] / DT)), 0) % HIST
            e_p = np.asarray(obs["tracker_pos"], dtype=float) - self.ph[j]
            e_y = (float(obs["tracker_yaw"]) - self.yh[j] + math.pi) % (2.0 * math.pi) - math.pi
            dth = np.zeros(3)
            f2 = float(f_w @ f_w)
            if f2 > 9.0:
                dth = (self.l3 / f2) * _cross(f_w, e_p)
            dth[2] += P["L_YAW"] * e_y
            R = self.R = _rot(dth) @ R
            self.p += self.l1 * e_p
            self.v += self.l2 * e_p
            self.ph += self.l1 * e_p
            self.yh += P["L_YAW"] * e_y
            f_w = R @ acc
        p_o = self.p - R @ R_IMU
        self.ph[k % HIST] = p_o
        self.yh[k % HIST] = math.atan2(R[1, 0], R[0, 0])

        # ----- plant-model observers -----
        w = gyro
        w_prev = self.gyro_prev
        tm_prev = self.tm
        if k > 0:
            self.tm = tm_prev + (self.u_last - tm_prev) * self.lag
        tm = self.tm
        tm_mid = 0.5 * (tm + tm_prev)
        w_mid = 0.5 * (w + w_prev)
        wdot = (w - w_prev) / DT
        kap = self.kappa
        if k > 0:
            res = J @ wdot + _cross(w_mid, J @ w_mid) - kap * (GEO @ tm_mid)
            self.d_fast += self.bf * (res - self.d_fast)
            self.d_slow += self.bs * (res - self.d_slow)
            k_id = int(round(P["ID_TIME"] / DT))
            if k <= k_id:
                # which motor lag explains the torque residual?  res ~ theta * (slow-model torque - nominal-model torque) + const
                tb_prev = self.tm_b
                self.tm_b = tb_prev + (self.u_last - tb_prev) * LAG_SLOW
                if k >= 30:
                    phi = kap * (GEO[:2] @ (0.5 * (self.tm_b + tb_prev) - tm_mid))
                    s = self.id_sum
                    for ax in (0, 1):
                        f, y = float(phi[ax]), float(res[ax])
                        s[ax, 0] += 1.0
                        s[ax, 1] += f
                        s[ax, 2] += y
                        s[ax, 3] += f * y
                        s[ax, 4] += f * f
                if k == k_id:
                    s = self.id_sum
                    num = float(np.sum(s[:, 3] - s[:, 1] * s[:, 2] / s[:, 0]))
                    den = float(np.sum(s[:, 4] - s[:, 1] ** 2 / s[:, 0]))
                    self.id_den = den
                    self.theta_hat = num / den if den > 0.02 else 0.0
                    if self.theta_hat > P["ID_DEADZONE"]:
                        th = self.theta = min(1.0, self.theta_hat)
                        self.lag = 1.0 - math.exp(-DT / (TAU_M + th * (TAU_SLOW - TAU_M)))
                        self.tm = tm = tm + th * (self.tm_b - tm)
                        tm_mid = tm
                        self.ks = (1.0 - th) * self.ks_nom + th * self.ks_slow
                        self.ab = self.ab_slow
        self.d_hist[k % 64] = self.d_slow
        t = k * DT
        if self.failed < 0:
            # thrust / mass ratio from the accelerometer
            bk = DT / 0.4
            self.kap_num += bk * (acc[2] - self.kap_num)
            self.kap_den += bk * (float(tm_mid.sum()) / M_NOM - self.kap_den)
            self.kappa = kap = min(1.5, max(0.6, self.kap_num / max(self.kap_den, 1.0)))
            # failure detection: does the fast residual look like a thrust loss of one rotor?
            if t >= P["DET_ARM_TIME"]:
                dres = (self.d_fast - self.d_hist[(k - 40) % 64])[:2]
                best, loss_best = -1, 0.0
                for i in range(4):
                    g = GEO[:2, i]
                    loss = -float(dres @ g) / (kap * float(g @ g))
                    if loss > loss_best:
                        best, loss_best = i, loss
                ok = False
                if best >= 0 and loss_best > P["DET_LOSS"]:
                    fit = dres + kap * loss_best * GEO[:2, best]
                    ok = float(fit @ fit) < 0.25 * float(dres @ dres)
                if ok and best == self.det_rotor:
                    self.det_count += 1
                elif ok:
                    self.det_rotor, self.det_count = best, 1
                else:
                    self.det_rotor, self.det_count = -1, 0
                if self.det_count >= P["DET_COUNT"]:
                    self.failed = best
                    self.t_detect = t
                    self.tm = tm = tm.copy()
                    tm[best] = max(tm[best] - loss_best, 0.0)
                    self.d_slow = self.d_hist[(k - 40) % 64].copy()
        failed = self.failed
        d_slow = self.d_slow

        # ----- position loop -----
        rho = -RHO_IMU                                   # centre of mass relative to the IMU
        v_c = self.v + R @ _cross(w, rho)
        e = np.asarray(obs["target_pos"], dtype=float) - p_o
        v_des = P["KP_POS"] * e
        vn = math.sqrt(float(v_des @ v_des))
        if vn > P["V_MAX"]:
            v_des *= P["V_MAX"] / vn
        a_des = P["KV"] * (v_des - v_c)
        an = math.hypot(a_des[0], a_des[1])
        if an > P["A_XY_MAX"]:
            a_des[:2] *= P["A_XY_MAX"] / an
        a_des[2] = min(6.0, max(-5.0, a_des[2]))
        # external specific force (drag, model error) from the accelerometer, referred to the centre of mass
        f_com = acc - _cross(w, _cross(w, RHO_IMU)) - _cross(wdot, RHO_IMU)
        d_b = f_com.copy()
        d_b[2] -= kap * float(tm.sum()) / M_NOM
        self.d_w += (DT / P["TAU_DIST"]) * (R @ d_b - self.d_w)
        d_w = self.d_w
        a_tot = a_des - np.clip(d_w, -4.0, 4.0)
        a_tot[2] += GRAV
        a_tot[2] = max(a_tot[2], 3.0)
        an = math.hypot(a_tot[0], a_tot[1])
        lim = math.tan(P["TILT_MAX"]) * a_tot[2]
        if an > lim:
            a_tot[:2] *= lim / an
        f_spec = math.sqrt(float(a_tot @ a_tot))
        n_des = a_tot / f_spec
        h = R.T @ n_des

        # ----- reduced-attitude LQR -----
        r = float(w[2])
        if failed >= 0:
            blend = min(1.0, (r / P["R_BLEND"]) ** 2)
            a_eq, b_eq = self.ab[failed][0] * blend, self.ab[failed][1] * blend
        else:
            a_eq = b_eq = 0.0
        nn = 1.0 / math.sqrt(1.0 + a_eq * a_eq + b_eq * b_eq)
        e1, e2 = h[0] - a_eq * nn, h[1] - b_eq * nn
        en = math.hypot(e1, e2)
        if h[2] < 0.0:                                  # more than 90 degrees off: full-scale error
            e1, e2 = e1 / max(en, 1e-6), e2 / max(en, 1e-6)
            en = 1.0
        if en > 0.6:
            e1, e2 = e1 * 0.6 / en, e2 * 0.6 / en
        w_eq = np.array([r * a_eq, r * b_eq, r])
        tau_eq = _cross(w_eq, J @ w_eq)
        tau_act = kap * (GEO @ tm) + d_slow
        x = np.array([e1, e2, w[0] - w_eq[0], w[1] - w_eq[1], tau_act[0] - tau_eq[0], tau_act[1] - tau_eq[1]])
        s = (min(47.9, max(-47.9, r)) - self.rs[0]) / 1.5
        i0 = int(s)
        fr = s - i0
        kk = self.ks[i0] * (1.0 - fr) + self.ks[i0 + 1] * fr
        tau_cmd = tau_eq[:2] - d_slow[:2] - kk @ x
        if 100 <= k < 200:                              # 0.5 .. 1.0 s: excite the motor-lag identification
            tau_cmd[0] += P["DITHER"] * (1.0 if (k // 10) % 2 == 0 else -1.0)
            tau_cmd[1] += P["DITHER"] * (1.0 if ((k + 5) // 10) % 2 == 0 else -1.0)
        thrust = M_NOM * f_spec * min(1.0, max(h[2], 0.0) / nn) / (nn * kap)   # newtons of commanded thrust
        thrust = min(thrust, 2.6 * M_NOM * GRAV)
        tx, ty = tau_cmd[0] / kap, tau_cmd[1] / kap

        # ----- allocation -----
        if failed < 0:
            tz = -P["K_YAW"] * J[2, 2] * r
            tz = min(0.04, max(-0.04, tz)) / kap
            u = ALLOC4_INV @ np.array([thrust, tx, ty, tz])
        else:
            live, inv = ALLOC3_INV[failed]
            sol = inv @ np.array([thrust, tx, ty])
            u = np.zeros(4)
            u[live] = sol
            j = OPP[failed]
            if u[j] < 0.0:
                u[j] = 0.0
                pair = [m for m in live if m != j]
                g = GEO[:2, pair[0]]
                diff = (tx * g[0] + ty * g[1]) / float(g @ g)
                diff = min(T_MAX, max(-T_MAX, diff))
                lo = max(thrust, abs(diff))
                u[pair[0]] = 0.5 * (lo + diff)
                u[pair[1]] = 0.5 * (lo - diff)
        if not np.all(np.isfinite(u)):                  # never hand a non-finite command to the plant
            u = np.full(4, M_NOM * GRAV / 4.0)
        u = np.clip(u, 0.0, T_MAX)
        if failed >= 0:
            u[failed] = 0.0
        self.u_last = u.copy()
        self.gyro_prev = gyro.copy()
        self.acc_prev = acc.copy()
        self.dbg = {"R": R, "p": p_o, "v": v_c, "kappa": kap, "failed": failed, "t_detect": self.t_detect,
                    "d_slow": d_slow, "d_w": d_w, "n_des": n_des, "h": h, "tm": tm, "theta_hat": self.theta_hat, "theta": self.theta}
        return u


_CTL = _Controller()


def controller(obs: dict) -> np.ndarray:
    return _CTL.step(obs)
