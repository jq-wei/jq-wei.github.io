"""Slung-payload quadrotor: thread a low window on a swing burst, then settle the swing at the goal.

All state is rebuilt when obs["step"] == 0. The law, from sensing to rotors:

1. Estimator. Strap-down integration of gyro and accelerometer (world acceleration = R f - g), corrected at
   50 Hz by the delayed tracker frame, which is compared with the estimate of six steps earlier. The position
   innovation corrects position, velocity, tilt and gyro bias (a tilt error is a horizontal acceleration error
   g * tilt), the yaw innovation corrects heading and yaw-gyro bias. `obs_w` is the observer bandwidth [rad/s].
2. Payload predictor and swing observer. A point-mass pendulum on a taut-or-slack cable hangs from the body
   origin and is driven by the measured specific force, so it follows the swing through the manoeuvre. In
   level, slow flight the accelerometer's body x/y components (they contain no thrust, only drag and cable)
   correct it: the innovation updates the payload offset, its rate and the vehicle's drag `bw`; `bw` in turn
   gives the wind load on the payload (same wind, known drag ratio K_WIND). Observer poles: `obs_slow` (drag)
   and `obs_re` +- j `obs_im` (swing); `tau_bs` smooths the drag that is applied to the payload.
3. Outer loop. Away from the burst an LQR law on (position, velocity, payload offset from its wind equilibrium,
   offset rate), weights `q_x`, `q_v`, `q_r`, `q_rd`, lets the vehicle follow the payload and so damps the
   swing; `a_max`, `r_cap`, `rd_cap` bound it while the swing is large (`*_g`: the gentler bounds used when the
   wind load is strong, which keeps the swing prediction accurate). During the burst a PD law
   (`kp_xy`, `kv_xy`) tracks the reference; height is always PD (`kp_z`, `kv_z`). A disturbance observer
   (`tau_d`) cancels the part of the measured specific force that the thrust model does not explain (drag,
   cable, mass error); the thrust effectiveness `kappa` is identified in the first second.
4. Inner loop. Tilt error between body z and the desired thrust direction, PD in angular acceleration
   (`k_att`, `k_rate`); the heading is held softly (`k_yaw`, `k_yawrate`) and re-adopted if it slips. The
   mixer keeps the collective thrust and scales the torques into the rotor range.
5. Plan. (a) Wait `t_wait`, move in `t_stage` to a point `x_stage` in front of the wall at the window's y and
   `dz_cross` above its centre, hold `t_hold`. (b) Burst: accelerate forward at `a_burst` for `t_burst`
   (ramps `t_ramp`); the payload swings up behind to about 85 deg just as the vehicle reaches the wall, so
   vehicle and payload thread the opening almost on one line. (c) Once both are through, the LQR law brings
   the vehicle to a point downwind of the goal and absorbs the swing; the height moves to the goal in `t_z`.
   (d) Final approach from T_PLAN: the reference ends on a parabola through the goal that accelerates downwind
   at a_c = the wind load on the payload, times `wind_comp` for a weak (poorly known) load rising to 1 for a
   strong one (`load_lo`, `load_hi`), capped at `a_c_max` (the parabola stays within 0.25 a_c of the goal). With that acceleration the cable hangs nearly vertical in the wind during the last two
   seconds. a_c follows the mean load, pulled by `rho_wind` toward the recent one (`tau_slow`, `tau_ac`), and
   changes sign over two ramps of one pendulum period each, which excite no swing. Gusts: the vehicle also
   follows the change of the wind load with gain `gust_ff` (limit `gust_max`, washed out with `tau_gust`).
"""

import math

import numpy as np
from scipy.linalg import solve_continuous_are

G = 9.81
M_NOM = 1.325
M_PAY = 0.15
CABLE = 1.0
DT = 0.005
LAG = 1.0 - math.exp(-DT / 0.03)
C_PAY = 0.002352                     # payload drag, N per (m/s)^2 per axis
C_VEH_X, C_VEH_Y = 0.0254, 0.0211    # vehicle drag along body x and y, N per (m/s)^2
K_WIND_X = C_PAY / C_VEH_X * M_NOM / M_PAY   # payload acceleration by the wind per unit of vehicle drag specific force
K_WIND_Y = C_PAY / C_VEH_Y * M_NOM / M_PAY
J = np.array([[0.0367, 0.0, -0.0021], [0.0, 0.0254, 0.0], [-0.0021, 0.0, 0.0605]])
MIX = np.array([[1.0, 1.0, 1.0, 1.0],
                [-0.18, 0.18, 0.18, -0.18],
                [0.14, 0.14, -0.14, -0.14],
                [-0.0201, 0.0201, -0.0201, 0.0201]])
MIX_INV = np.linalg.inv(MIX)
U_MAX = 13.0
IMU_ARM = 0.034                      # m, IMU below the nominal centre of mass
T_FREEZE = 12.0                      # s: the planned downwind acceleration is frozen for the last two seconds
T_PLAN = 8.0                         # s: start of the final approach profile (the episode lasts 14 s)

PARAMS = {
    "kp_xy": 10.0, "kv_xy": 6.0, "kp_z": 14.0, "kv_z": 7.0,
    "k_att": 150.0, "k_rate": 22.0, "k_yaw": 20.0, "k_yawrate": 6.0,
    "tau_d": 0.12, "obs_w": 4.0, "tau_bs": 0.2, "obs_slow": 3.0, "obs_re": 0.3, "obs_im": 3.13,
    "t_wait": 0.6, "t_stage": 2.4, "t_hold": 0.8, "x_stage": -2.5, "dz_cross": 0.08,
    "a_burst": 7.4, "t_burst": 0.8, "t_ramp": 0.12,
    "q_x": 8.0, "q_v": 1.0, "q_r": 20.0, "q_rd": 2.0, "a_max": 6.0, "r_cap": 0.4, "rd_cap": 3.0, "a_max_g": 4.0, "r_cap_g": 0.3, "rd_cap_g": 2.0,
    "t_z": 2.0, "wind_comp": 0.8, "load_lo": 0.25, "load_hi": 0.40, "a_c_max": 0.5, "gust_ff": 0.7, "gust_max": 0.6, "tau_gust": 1.5, "rho_wind": 0.6, "tau_slow": 1.0, "tau_ac": 0.5,
}


def _lqr_gain(p):
    w2 = G / CABLE
    a = np.array([[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, -w2, 0.0]])
    b = np.array([[0.0], [1.0], [0.0], [-1.0]])
    q = np.diag([p["q_x"], p["q_v"], p["q_r"], p["q_rd"]])
    x = solve_continuous_are(a, b, q, np.eye(1))
    return (b.T @ x).ravel()


def _obs_gain(p):
    """Observer gains (offset, offset rate, drag) for the innovation in the lateral specific force [m/s^2]."""
    from scipy.signal import place_poles
    a = np.array([[0.0, 1.0, 0.0], [-G / CABLE, 0.0, 0.5 * (K_WIND_X + K_WIND_Y)], [0.0, 0.0, 0.0]])
    c = np.array([[M_PAY * G / (M_NOM * CABLE), 0.0, 1.0]])
    poles = [-p["obs_slow"], complex(-p["obs_re"], p["obs_im"]), complex(-p["obs_re"], -p["obs_im"])]
    return place_poles(a.T, c.T, poles).gain_matrix.ravel()


def _hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _expm(w):
    th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    k = _hat(w)
    if th2 < 1e-12:
        return np.eye(3) + k
    th = math.sqrt(th2)
    return np.eye(3) + (math.sin(th) / th) * k + ((1.0 - math.cos(th)) / th2) * (k @ k)


def _minjerk(s):
    if s <= 0.0:
        return 0.0, 0.0, 0.0
    if s >= 1.0:
        return 1.0, 0.0, 0.0
    s2 = s * s
    return (s2 * s * (10.0 - 15.0 * s + 6.0 * s2), s2 * (30.0 - 60.0 * s + 30.0 * s2), s * (60.0 - 180.0 * s + 120.0 * s2))


class _Ctl:
    def reset(self, obs):
        p = PARAMS
        self.K = _lqr_gain(p)
        yaw = float(obs["tracker_yaw"])
        c, sn = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        self.p = np.array(obs["tracker_pos"], dtype=float)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.baz = 0.0
        self.gyro_prev = np.array(obs["gyro"], dtype=float)
        self.aw_prev = self.R @ np.asarray(obs["accel"], dtype=float)
        self.hist = [(self.p.copy(), yaw)]
        self.th = np.full(4, (M_NOM + M_PAY) * G / 4.0)
        self.u_prev = self.th.copy()
        self.kappa = M_NOM / (M_NOM + M_PAY)
        self.d = np.zeros(3)
        self.q = np.array([0.0, 0.0, -CABLE])
        self.qd = np.zeros(3)
        self.bw = np.zeros(3)          # drag specific force on the vehicle (world), from the swing observer
        self.bs = np.zeros(3)          # its slow part: the wind load that deflects the payload
        self.Lo = _obs_gain(p)
        self.b_mean = np.zeros(3)      # running mean of bw: the mean wind load, used to plan the final approach
        self.b_n = 0.0
        self.alpha = np.zeros(3)
        self.b_lp = np.zeros(3)
        self.b_slow = np.zeros(3)      # recent wind load (about a second)
        self.ac_init = False
        self.gentle = False
        self.a_c = np.zeros(2)
        self.yaw_d = yaw
        self.p0 = self.p.copy()
        self.win = np.array(obs["window_centre"], dtype=float)
        self.goal = np.array(obs["target_pos"], dtype=float)
        self.stage = np.array([p["x_stage"], self.win[1], self.win[2] + p["dz_cross"]])
        self.mode = 0
        self.tb = 0.0
        self.xr = self.stage[0]
        self.vr = 0.0
        self.t2 = 0.0
        self.z2 = self.stage[2]
        w = p["obs_w"]
        ft = 0.02
        self.L_p, self.L_v = 4.0 * w * ft, 6.0 * w * w * ft
        self.L_th, self.L_b = 4.0 * w ** 3 / G * ft, w ** 4 / G * ft
        self.L_pz, self.L_vz, self.L_bz = 3.0 * w * ft, 3.0 * w * w * ft, w ** 3 * ft
        self.L_yaw, self.L_by = 2.0 * w * ft, w * w * ft

    def step(self, obs):
        p = PARAMS
        k = int(obs["step"])
        t = k * DT
        gyro = np.asarray(obs["gyro"], dtype=float)
        f = np.asarray(obs["accel"], dtype=float)

        # ---- estimator: strap-down propagation, tracker correction ----
        if k > 0:
            self.th += (self.u_prev - self.th) * LAG
            w = 0.5 * (gyro + self.gyro_prev) - self.bg
            self.R = self.R @ _expm(w * DT)
            aw_new = self.R @ f
            a = 0.5 * (aw_new + self.aw_prev)
            a[2] -= G + self.baz
            self.p += self.v * DT + (0.5 * DT * DT) * a
            self.v += a * DT
            self.aw_prev = aw_new
            self.alpha += ((gyro - self.gyro_prev) / DT - self.alpha) * 0.3
        self.gyro_prev = gyro
        R = self.R
        yaw = math.atan2(R[1, 0], R[0, 0])
        self.hist.append((self.p.copy(), yaw))
        if len(self.hist) > 7:
            self.hist.pop(0)
        if obs["tracker_fresh"] and k > 0:
            ph, yh = self.hist[0]
            e = np.asarray(obs["tracker_pos"], dtype=float) - ph
            ey = (float(obs["tracker_yaw"]) - yh + math.pi) % (2.0 * math.pi) - math.pi
            dp = np.array([self.L_p * e[0], self.L_p * e[1], self.L_pz * e[2]])
            self.p += dp
            for h in self.hist:
                h[0][:] += dp
            self.v += np.array([self.L_v * e[0], self.L_v * e[1], self.L_vz * e[2]])
            self.R = _expm(np.array([-self.L_th * e[1], self.L_th * e[0], self.L_yaw * ey])) @ self.R
            R = self.R
            self.bg -= R.T @ np.array([-self.L_b * e[1], self.L_b * e[0], self.L_by * ey])
            self.baz -= self.L_bz * e[2]
        omega = gyro - self.bg
        zb = R[:, 2]

        # ---- disturbance observer and thrust effectiveness ----
        tsum = float(self.th.sum())
        aw = self.aw_prev.copy()
        aw[2] -= self.baz
        if 0.05 < t < 1.0:
            self.kappa += (f[2] - self.kappa * tsum / M_NOM) * (M_NOM / tsum) * (DT / 0.25)
            self.kappa = min(1.2, max(0.65, self.kappa))
        raw = aw - (self.kappa * tsum / M_NOM) * zb
        self.d += (raw - self.d) * (DT / p["tau_d"])

        # ---- payload predictor (relative position q of the payload from the hook) ----
        vp = self.v + self.qd
        acc = -aw - (C_PAY / M_PAY) * np.abs(vp) * vp
        acc[0] += K_WIND_X * self.bs[0]
        acc[1] += K_WIND_Y * self.bs[1]
        self.qd += acc * DT
        self.q += self.qd * DT
        n = math.sqrt(float(self.q @ self.q))
        if n >= CABLE:
            u = self.q / n
            self.q = u * CABLE
            vr = float(self.qd @ u)
            if vr > 0.0:
                self.qd -= vr * u
            tens = max(0.0, M_PAY * (float(self.qd @ self.qd) / CABLE - float(aw @ u)))
        else:
            u = self.q / max(n, 1e-6)
            tens = 0.0

        # ---- swing observer (level, slow flight only): the accelerometer's body x/y see no thrust, only drag and cable ----
        if zb[2] > 0.96 and self.mode != 1 and t > 0.3 and abs(self.v[0]) + abs(self.v[1]) < 2.0:
            bx, by = self.bw[0], self.bw[1]
            relx = math.copysign(math.sqrt(M_NOM * abs(bx) / C_VEH_X), bx) - self.v[0]   # air speed seen by the vehicle
            rely = math.copysign(math.sqrt(M_NOM * abs(by) / C_VEH_Y), by) - self.v[1]
            drag = np.array([C_VEH_X * abs(relx) * relx / M_NOM, C_VEH_Y * abs(rely) * rely / M_NOM, 0.0])
            yp = R.T @ (drag + (tens / M_NOM) * u)
            # the IMU sits IMU_ARM below the centre of mass: remove the angular-acceleration part of its reading
            nbx = f[0] + IMU_ARM * self.alpha[1] - yp[0]
            nby = f[1] - IMU_ARM * self.alpha[0] - yp[1]
            nwx = R[0, 0] * nbx + R[0, 1] * nby
            nwy = R[1, 0] * nbx + R[1, 1] * nby
            Lo = self.Lo
            self.q[0] += Lo[0] * nwx * DT
            self.q[1] += Lo[0] * nwy * DT
            self.qd[0] += Lo[1] * nwx * DT
            self.qd[1] += Lo[1] * nwy * DT
            self.bw[0] += Lo[2] * nwx * DT
            self.bw[1] += Lo[2] * nwy * DT
            self.bs += (drag - self.bs) * (DT / p["tau_bs"])
            self.b_slow += (self.bw - self.b_slow) * (DT / p["tau_slow"])
            if t > 1.5:
                self.b_n += 1.0
                self.b_mean += (self.bw - self.b_mean) / self.b_n
        qe0 = self.bs[0] * (K_WIND_X * CABLE / G)
        qe1 = self.bs[1] * (K_WIND_Y * CABLE / G)
        rx, ry = self.q[0] - qe0, self.q[1] - qe1

        # ---- plan ----
        pos, vel = self.p, self.v
        K = self.K
        if self.mode == 0:
            sp, sv, sa = _minjerk((t - p["t_wait"]) / p["t_stage"])
            delta = self.stage - self.p0
            p_ref = self.p0 + delta * sp
            v_ref = delta * (sv / p["t_stage"])
            a_ref = delta * (sa / p["t_stage"] ** 2)
            a_cmd = np.array([
                a_ref[0] - K[0] * (pos[0] - p_ref[0]) - K[1] * (vel[0] - v_ref[0]) - K[2] * rx - K[3] * self.qd[0],
                a_ref[1] - K[0] * (pos[1] - p_ref[1]) - K[1] * (vel[1] - v_ref[1]) - K[2] * ry - K[3] * self.qd[1],
                a_ref[2] + p["kp_z"] * (p_ref[2] - pos[2]) + p["kv_z"] * (v_ref[2] - vel[2])])
            if t >= p["t_wait"] + p["t_stage"] + p["t_hold"]:
                self.mode = 1
                self.tb = t
                self.xr, self.vr = self.stage[0], 0.0
        elif self.mode == 1:
            tt = t - self.tb
            ab = p["a_burst"] * min(1.0, max(0.0, tt / p["t_ramp"])) * min(1.0, max(0.0, (p["t_burst"] - tt) / p["t_ramp"]))
            self.vr += ab * DT
            self.xr += self.vr * DT
            p_ref = np.array([self.xr, self.stage[1], self.stage[2]])
            v_ref = np.array([self.vr, 0.0, 0.0])
            a_cmd = np.array([ab, 0.0, 0.0]) + np.array([p["kp_xy"], p["kp_xy"], p["kp_z"]]) * (p_ref - pos) \
                + np.array([p["kv_xy"], p["kv_xy"], p["kv_z"]]) * (v_ref - vel)
            if pos[0] > 0.3 and pos[0] + self.q[0] > 0.25:
                self.mode = 2
                self.t2 = t
                self.z2 = float(pos[2])
                # in a strong wind the swing is absorbed more gently (smaller demands keep the predictor accurate)
                self.gentle = math.hypot(K_WIND_X * self.b_mean[0], K_WIND_Y * self.b_mean[1]) > p["load_lo"]
        if self.mode == 2:
            sp, sv, sa = _minjerk((t - self.t2) / p["t_z"])
            dz = self.goal[2] - self.z2
            z_ref = self.z2 + dz * sp
            # final approach: the vehicle ends on a parabola that accelerates downwind at a_c, so that the cable hangs
            # (almost) vertically under the wind during the last two seconds; a_c ramps over one pendulum period each time
            if t < T_FREEZE:
                # planned downwind acceleration: mean wind load, pulled toward the recent load (gusts last a few seconds)
                bex0 = self.b_mean[0] + p["rho_wind"] * (self.b_slow[0] - self.b_mean[0])
                bey0 = self.b_mean[1] + p["rho_wind"] * (self.b_slow[1] - self.b_mean[1])
                ac = np.array([K_WIND_X * bex0, K_WIND_Y * bey0])
                # compensate a strong wind load fully, a weak (poorly known) one only by `wind_comp`
                na = math.hypot(ac[0], ac[1])
                ac *= p["wind_comp"] + (1.0 - p["wind_comp"]) * min(1.0, max(0.0, (na - p["load_lo"]) / (p["load_hi"] - p["load_lo"])))
                na = math.hypot(ac[0], ac[1])
                if na > p["a_c_max"]:
                    ac *= p["a_c_max"] / na
                if self.ac_init:
                    self.a_c += (ac - self.a_c) * (DT / p["tau_ac"])
                else:
                    self.a_c = ac
                    self.ac_init = True
            if t < T_PLAN:
                self.b_lp = self.bs.copy()
                gx = gy = 0.0
                bex, bey = self.bs[0], self.bs[1]
            else:
                # gusts: follow the change of the payload's wind load (relative to the planned mean) with the vehicle
                self.b_lp += (self.bs - self.b_lp) * (DT / p["tau_gust"])
                gg = p["gust_ff"] * min(1.0, t - T_PLAN)
                gx = gg * K_WIND_X * (self.bs[0] - self.b_lp[0])
                gy = gg * K_WIND_Y * (self.bs[1] - self.b_lp[1])
                gn = math.hypot(gx, gy)
                if gn > p["gust_max"]:
                    gx, gy = gx * p["gust_max"] / gn, gy * p["gust_max"] / gn
                bex = self.bs[0] - gx / K_WIND_X
                bey = self.bs[1] - gy / K_WIND_Y
            rx = self.q[0] - bex * (K_WIND_X * CABLE / G)
            ry = self.q[1] - bey * (K_WIND_Y * CABLE / G)
            ac = self.a_c
            if t < T_PLAN:
                s_a, s_v, s_x = 0.0, 0.0, 0.0
            elif t < T_PLAN + 2.0:
                tau = t - T_PLAN
                s_a, s_v, s_x = -0.5 * tau, -0.25 * tau * tau, -tau ** 3 / 12.0
            elif t < T_PLAN + 4.0:
                tau = t - T_PLAN - 2.0
                s_a, s_v, s_x = -1.0 + tau, -1.0 - tau + 0.5 * tau * tau, -2.0 / 3.0 - tau - 0.5 * tau * tau + tau ** 3 / 6.0
            else:
                tau = t - T_PLAN - 4.0
                s_a, s_v, s_x = 1.0, -1.0 + tau, -10.0 / 3.0 - tau + 0.5 * tau * tau
            s_p = 43.0 / 12.0 + s_x
            w2 = G / CABLE
            ex = pos[0] - self.goal[0] - ac[0] * s_p
            ey2 = pos[1] - self.goal[1] - ac[1] * s_p
            evx = vel[0] - ac[0] * s_v
            evy = vel[1] - ac[1] * s_v
            rx2 = rx + ac[0] * s_a / w2
            ry2 = ry + ac[1] * s_a / w2
            # large swings right after the window: bound what the swing terms may ask of the vehicle
            a_max, r_cap, rd_cap = (p["a_max_g"], p["r_cap_g"], p["rd_cap_g"]) if self.gentle else (p["a_max"], p["r_cap"], p["rd_cap"])
            rn = math.hypot(rx2, ry2)
            if rn > r_cap:
                rx2, ry2 = rx2 * r_cap / rn, ry2 * r_cap / rn
            qdx, qdy = self.qd[0], self.qd[1]
            qn = math.hypot(qdx, qdy)
            if qn > rd_cap:
                qdx, qdy = qdx * rd_cap / qn, qdy * rd_cap / qn
            ax = -(K[0] * ex + K[1] * evx + K[2] * rx2 + K[3] * qdx)
            ay = -(K[0] * ey2 + K[1] * evy + K[2] * ry2 + K[3] * qdy)
            nrm = math.hypot(ax, ay)
            if nrm > a_max:
                ax, ay = ax * a_max / nrm, ay * a_max / nrm
            ax += ac[0] * s_a + gx
            ay += ac[1] * s_a + gy
            az = dz * sa / p["t_z"] ** 2 + p["kp_z"] * (z_ref - pos[2]) + p["kv_z"] * (dz * sv / p["t_z"] - vel[2])
            a_cmd = np.array([ax, ay, az])

        # ---- thrust vector, attitude loop, mixer ----
        s_des = a_cmd - self.d
        s_des[2] += G
        if s_des[2] < 3.0:
            s_des[2] = 3.0
        hmax = 1.6 * s_des[2]
        hn = math.hypot(s_des[0], s_des[1])
        if hn > hmax:
            s_des[0] *= hmax / hn
            s_des[1] *= hmax / hn
        thrust = M_NOM * float(s_des @ zb) / self.kappa
        thrust = min(4.0 * U_MAX * 0.95, max(2.0, thrust))
        zd = s_des / math.sqrt(float(s_des @ s_des))
        # tilt error: rotation (body axes) that brings body z onto the desired thrust direction; yaw is handled apart
        zdb = R.T @ zd
        sn = math.hypot(zdb[0], zdb[1])
        ang = math.atan2(sn, zdb[2])
        scale = ang / sn if sn > 1e-6 else 1.0
        e_x, e_y = zdb[1] * scale, -zdb[0] * scale
        # heading: hold the initial one, softly; if it has slipped far (rotor saturation), adopt the current heading
        e_yaw = (yaw - self.yaw_d + math.pi) % (2.0 * math.pi) - math.pi
        if abs(e_yaw) > 0.5:
            self.yaw_d = yaw
            e_yaw = 0.0
        alpha = np.array([-p["k_att"] * e_x - p["k_rate"] * omega[0],
                          -p["k_att"] * e_y - p["k_rate"] * omega[1],
                          -p["k_yaw"] * e_yaw * zb[2] - p["k_yawrate"] * omega[2]])
        tau = J @ alpha + np.cross(omega, J @ omega)
        d_rp = MIX_INV[:, 1] * tau[0] + MIX_INV[:, 2] * tau[1]
        d_y = MIX_INV[:, 3] * min(0.12, max(-0.12, tau[2]))
        # collective first (it holds the altitude), then as much of the torque as the rotor range leaves
        base = min(U_MAX - 0.8, max(1.0, thrust / 4.0))
        dd = d_rp + d_y
        lo, hi = float(dd.min()), float(dd.max())
        sc = 1.0
        if lo < -base:
            sc = min(sc, base / -lo)
        if hi > U_MAX - base:
            sc = min(sc, (U_MAX - base) / hi)
        u = np.clip(base + sc * dd, 0.0, U_MAX)
        if not np.all(np.isfinite(u)):          # never hand the plant a non-finite command
            u = np.full(4, (M_NOM + M_PAY) * G / 4.0)
        self.u_prev = u
        return u


_c = _Ctl()


def controller(obs: dict) -> np.ndarray:
    if int(obs["step"]) == 0:
        _c.reset(obs)
    return _c.step(obs)
