"""Gate-course controller for the rotor-thrust quadrotor (Skydio X2 plant of hl_control).

Control law (hand-designed; the only numerical optimisation is a small SLSQP trajectory plan, no learning):

1. Observer.  A strap-down INS (gyro -> attitude, accelerometer -> velocity/position) corrected by the delayed
   tracker frames with fixed gains: the position innovation (tracker frame minus the estimate 30 ms ago) corrects
   position, velocity and, through the cross product with the measured specific force, the tilt; the tracker yaw
   corrects the heading while hovering.  The three gains come from one bandwidth (w0_hover, w0_race).  The gyro
   bias is the mean gyro reading of the hover minus the rotation the observer saw.
2. Hover.  Until t_launch the vehicle holds a point at the back of the start sphere (PD kp, kd plus the measured
   external acceleration); meanwhile it identifies the thrust-to-mass ratio (accelerometer z over the modelled
   rotor thrust, so mass and thrust coefficient need not be known) and the mean wind (from the horizontal drag).
3. Plan.  At the launch, and again from the measured state when the lap clock starts (replan), a minimum-time
   plan of a point mass through the aim points: thrust eta_plan * maximum along a direction that is piecewise
   linear in time and continuous through the gates (the time-optimal law of a point mass), with gravity, the
   body-z and lateral drag in the estimated wind (drag_plan) and a loss of thrust while the thrust direction turns
   (rot_loss, w_ref: the rotors that make the torque).  Unknowns: four direction nodes and four leg times, solved
   by SLSQP from the stored warm start X0 (plan_iter).  The aim points are the gate centres shifted by the mean
   passing offsets measured in simulation (r_off, z2, y3, lx/ly/lz).
4. Tracking.  Acceleration command = plan acceleration (taken ff_lead ahead) + PD on position and velocity
   (kp_t, kv_t, limited to a_fb).  The plan's clock slows down when the vehicle lags along the track (k_clock,
   lag_free, rate_min), so a weak vehicle or a head wind stretches the plan in time instead of leaving the path.
   The thrust magnitude adds the body-z drag as measured (modelled rotor thrust minus accelerometer, dz_gain).
   After gate 3, or if the plan is lost, the next gate is taken by a constant-acceleration intercept (eta).
5. Attitude.  Thrust-vector pointing: tilt error -> body-rate command (k_att, limited by sqrt(2 alpha_max angle))
   plus the plan's own turn rate (ff_rate) -> angular acceleration (k_rate, k_rate_z) -> torques -> rotor thrusts.
   The tilt torque has priority over the collective thrust, the yaw torque gets a small share (yaw_cap), and the
   commands lead the modelled 30 ms motor lag (lead).
"""

import math

import numpy as np

G = 9.81
M_NOM = 1.325
JN = np.array([[0.0366517, 0.0, -0.0021], [0.0, 0.0254117, 0.0], [-0.0021, 0.0, 0.060528]])
ARM_X, ARM_Y, CQ = 0.14, 0.18, 0.0201
SX = np.array([-1.0, 1.0, 1.0, -1.0])     # roll torque sign of each rotor
SY = np.array([1.0, 1.0, -1.0, -1.0])     # pitch torque sign
SZ = np.array([-1.0, 1.0, -1.0, 1.0])     # yaw torque sign
TAU_M = 0.03
U_MAX = 13.0
CD = np.array([0.0254, 0.0211, 0.1656])   # drag N/(m/s)^2 along body x, y, z
EZ = np.array([0.0, 0.0, 1.0])
I3 = np.eye(3)

# warm start of the plan: thrust-direction nodes (start, gates 1..3) and leg times [s], see _Controller._pm
X0 = [0.6598, 0.6179, 0.4276, -0.1619, -0.5753, -0.1113, 0.0123, 0.6995, 0.3596, 0.0972, -0.3947, -0.0849, 0.9299, 0.7244, 0.7972, 0.4753]

PARAMS = {
    "w0_hover": 6.0,      # observer bandwidth while hovering [rad/s]
    "w0_race": 9.0,       # observer bandwidth while racing [rad/s]
    "kp": 9.0,            # hover position gain [1/s^2]
    "kd": 5.5,            # hover velocity gain [1/s]
    "k_att": 13.0,        # tilt error -> body rate [1/s]
    "k_rate": 30.0,       # body-rate error -> angular acceleration [1/s]
    "k_rate_z": 8.0,      # the same about body z
    "yaw_cap": 0.02,      # share of the rotor range given to the yaw torque while racing
    "ff_rate": 0.8,       # share of the reference's thrust-direction rotation rate fed forward to the body-rate command
    "alpha_max": 50.0,    # angular acceleration assumed by the sqrt slew law [rad/s^2]
    "lead": 3.2,          # lead on the motor lag (1 = none)
    "eta": 0.93,          # share of the maximum thrust used by the fallback intercept guidance
    "eta_plan": 0.98,     # share of the maximum thrust the reference trajectory is planned with
    "drag_plan": 1.0,     # share of the body-z drag included in the plan's model
    "replan": 1.0,        # 1: plan again from the measured state when the lap clock starts
    "plan_iter": 30,      # SLSQP iterations of the plan at the launch (warm start: X0)
    "rot_loss": 0.50,     # share of the thrust lost while the thrust direction turns at w_ref or faster (plan model)
    "w_ref": 9.0,         # turn rate of the thrust direction at which that loss is complete [rad/s]
    "nz_min": -5.0,       # lower bound of the vertical component of the thrust-direction nodes (-5: not active)
    "w_plan": 16.0,       # largest turn rate of the thrust direction allowed in the plan [rad/s]
    "dz_gain": 0.3,       # low-pass factor of the measured body-z drag that the thrust command makes up for
    "a_fb": 30.0,         # largest feedback acceleration added to the plan's [m/s^2]
    "kp_t": 30.0,         # tracking position gain [1/s^2]
    "kv_t": 9.0,          # tracking velocity gain [1/s]
    "ff_lead": 0.04,      # the feed-forward acceleration is taken this far ahead (attitude lag) [s]
    "k_clock": 1.0,       # slow-down of the reference clock per metre of lag along the track [1/m] ...
    "lag_free": 0.35,     # ... beyond this lag [m]
    "t_rot": 0.06,        # time given to the first rotation before the reference clock starts [s]
    "k_cross": 0.0,       # slow-down of the reference clock per metre of error across the track [1/m]
    "rate_max": 1.0,      # fastest reference clock (1 = as planned)
    "rate_min": 0.6,      # slowest reference clock
    "err_max": 2.0,       # tracking error beyond which the reference is abandoned [m]
    "lx": -0.12, "ly": -0.45, "lz": 0.08,   # aim point of the last gate in the plan, relative to its centre [m]
    "y3": -0.07,          # lateral shift of the aim point of gate 3 [m] (measured mean passing offset)
    "z2": 0.15,           # aim point of gate 2 above its centre [m] (makes up for the sag during the thrust reversal)
    "r_off": -0.15,       # aim point of gates 1..3 along the inside of the turn [m] (negative: outside, the tracking cuts the turn)
    "t_launch": 3.2,      # launch time [s]
    "back": 0.2,          # hover point behind the start point [m]
}


def _hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _expm(w):
    th2 = float(w @ w)
    k = _hat(w)
    if th2 < 1e-10:
        return I3 + k + 0.5 * (k @ k)
    th = math.sqrt(th2)
    return I3 + (math.sin(th) / th) * k + ((1.0 - math.cos(th)) / th2) * (k @ k)


class _Controller:
    def __init__(self):
        self.tgrid = np.geomspace(0.02, 5.0, 44)
        self.lin = np.linspace(0.0, 1.0, 12)
        self.rot_loss = PARAMS["rot_loss"]
        self.c_lat = 0.5 * (CD[0] + CD[1]) / M_NOM
        self.w_ref2 = PARAMS["w_ref"] ** 2
        self.sgrid = {}
        for ns in range(2, 400):
            self.sgrid[ns] = np.linspace(0.0, 1.0, ns + 1)[:, None]

    # ----- episode start -----
    def reset(self, obs):
        P = PARAMS
        self.dt = float(obs["dt"])
        yaw = float(obs["tracker_yaw"])
        c, s = math.cos(yaw), math.sin(yaw)
        self.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        self.p = np.array(obs["tracker_pos"], dtype=float)
        self.v = np.zeros(3)
        self.bg = np.zeros(3)
        self.hist = [self.p.copy()]
        self.yaw_hist = [yaw]
        self.gyro_prev = np.array(obs["gyro"], dtype=float)
        self.fw_prev = self.R @ np.asarray(obs["accel"], dtype=float)
        self.kappa = 1.0 / M_NOM                      # thrust-to-mass ratio ks / m [1/kg]
        self.That = np.full(4, G / self.kappa / 4.0)  # modelled rotor thrusts (command units)
        self.u_prev = self.That.copy()
        self.dhat = np.zeros(3)                       # external acceleration (drag), world frame
        self.wind = np.zeros(3)
        self.racing = False
        self.gates = np.asarray(obs["gates"], dtype=float)
        self.start = np.asarray(obs["start_pos"], dtype=float)
        self.k_int = 0
        self.tracking = True
        self.replanned = False
        self.dz_f = 0.0
        self.t_race = 0.0
        self.w_ff = np.zeros(3)
        self.g_sum = np.zeros(3)
        self.g_cnt = 0
        self.sel = 0
        self.sel_gate = -1
        self.sel_cnt = 0
        self.lagf = 1.0 - math.exp(-self.dt / TAU_M)
        # gate plan: aim points and passing velocity directions
        pts = np.vstack([self.start, self.gates])
        n = len(self.gates)
        self.aim = self.gates.copy()
        for k in range(n - 1):
            d_in = pts[k + 1] - pts[k]
            d_out = pts[k + 2] - pts[k + 1]
            inside = d_out / np.linalg.norm(d_out) - d_in / np.linalg.norm(d_in)
            self.aim[k] = self.gates[k] + P["r_off"] * inside / max(np.linalg.norm(inside), 1e-6)
        self.aim[n - 1] = self.gates[n - 1] + np.array([P["lx"], P["ly"], P["lz"]])
        self.aim[1, 2] += P["z2"]                       # the vehicle sags at the low gate: aim above its centre
        self.aim[2, 1] += P["y3"]
        d0 = self.aim[0] - self.start
        d0[2] = 0.0
        d0 /= np.linalg.norm(d0)
        self.hold = self.start - P["back"] * d0
        self.yaw_ref = yaw
        self.cz_end = 0.0
        self.set_observer(P["w0_hover"])

    def set_observer(self, w0):
        T = 0.02
        self.L1 = min(0.9, 3.0 * w0 * T)
        self.L2 = 3.0 * w0 * w0 * T
        self.L3 = w0 ** 3 * T

    # ----- observer -----
    def estimate(self, obs, i):
        P = PARAMS
        dt = self.dt
        gyro = np.asarray(obs["gyro"], dtype=float)
        acc = np.asarray(obs["accel"], dtype=float)
        w = 0.5 * (self.gyro_prev + gyro) - self.bg
        self.R = self.R @ _expm(w * dt)
        fw = self.R @ acc
        a = 0.5 * (self.fw_prev + fw)
        a[2] -= G
        self.p = self.p + self.v * dt + (0.5 * dt * dt) * a
        self.v = self.v + a * dt
        self.gyro_prev = gyro
        self.fw_prev = fw
        self.That += (self.u_prev - self.That) * self.lagf
        R = self.R
        self.hist.append(self.p.copy())
        self.yaw_hist.append(math.atan2(R[1, 0], R[0, 0]))
        if len(self.hist) > 7:
            self.hist.pop(0)
            self.yaw_hist.pop(0)
        if obs["tracker_fresh"]:
            e = np.asarray(obs["tracker_pos"], dtype=float) - self.hist[0]
            dp = self.L1 * e
            self.p += dp
            self.v += self.L2 * e
            for h in self.hist:
                h += dp
            dth = (self.L3 / max(float(fw @ fw), 25.0)) * np.cross(fw, e)
            if not self.racing:
                ey = float(obs["tracker_yaw"]) - self.yaw_hist[0]
                ey = (ey + math.pi) % (2.0 * math.pi) - math.pi
                dth[2] += 0.25 * ey
                self.yaw_hist = [y + 0.25 * ey for y in self.yaw_hist]
            self.R = _expm(dth) @ self.R
        if i % 40 == 0:
            u_, _, vt_ = np.linalg.svd(self.R)
            self.R = u_ @ vt_
        return gyro - self.bg, acc, fw

    # ----- guidance -----
    def _need(self, A, vrel, cz):
        """Thrust acceleration needed for the thrust-acceleration vectors A (n, 3): norm plus the body-z drag."""
        nrm = np.sqrt(np.einsum("ij,ij->i", A, A)) + 1e-9
        vn = (A @ vrel) / nrm
        return nrm + cz * vn * np.abs(vn)

    def _solve(self, dp, v, vf, gvec, alim, alim1, cz):
        """Smallest time-to-go whose cubic (or constant-acceleration, vf None) leg fits the thrust; returns (t, A0)."""
        vrel = v - self.wind

        def feas(t):
            t2 = (t * t)[:, None]
            tt = t[:, None]
            if vf is None:
                A0 = (2.0 * (dp - v * tt)) / t2 + gvec
                return self._need(A0, vrel, cz) <= alim, A0
            A0 = (6.0 * dp - (4.0 * v + 2.0 * vf) * tt) / t2 + gvec
            A1 = (-6.0 * dp + (2.0 * v + 4.0 * vf) * tt) / t2 + gvec
            ok = (self._need(A0, vrel, cz) <= alim) & (self._need(A1, vf - self.wind, self.cz_end * cz) <= alim1)
            return ok, A0

        t = self.tgrid
        ok, A0 = feas(t)
        j = int(np.argmax(ok))
        if not ok[j]:
            return float(t[-1]), A0[-1]
        if j == 0:
            return float(t[0]), A0[0]
        t2 = t[j - 1] + (t[j] - t[j - 1]) * self.lin
        ok2, A02 = feas(t2)
        j2 = int(np.argmax(ok2))
        return float(t2[j2]), A02[j2]

    def _pm(self, x, p0, v0, alim, cz, nsub):
        """Point-mass model of the plan: full thrust alim along a direction lam(t) that is piecewise linear in time and
        continuous through the gates (the time-optimal law of a point mass), gravity, and the body-z drag along the
        thrust.  x = [lam at the start and at gates 1..3 (12), leg times (4)].  Returns per-leg (p, v, a, n) arrays."""
        lam = [x[0:3], x[3:6], x[6:9], x[9:12]]
        legs = []
        p, v = p0, v0
        wind = self.wind
        for k in range(4):
            ns = nsub[k]
            h = x[12 + k] / ns
            sg = self.sgrid[ns]
            L = lam[k] + (lam[k + 1] - lam[k]) * sg if k < 3 else lam[3] + 0.0 * sg
            n = L / np.sqrt(np.einsum("ij,ij->i", L, L))[:, None]
            # turning the thrust direction costs collective thrust (the rotors that make the torque)
            cr = np.cross(n, np.gradient(n, h, axis=0))
            wr2 = np.einsum("ij,ij->i", cr, cr)
            thr = alim * (1.0 - self.rot_loss * np.minimum(1.0, wr2 / self.w_ref2))
            a = thr[:, None] * n
            a[:, 2] -= G
            for _ in range(2):
                vv = v + h * np.vstack([np.zeros(3), np.cumsum(0.5 * (a[1:] + a[:-1]), axis=0)])
                vr = vv - wind
                vn = np.einsum("ij,ij->i", vr, n)
                vp = vr - vn[:, None] * n                 # air velocity across the thrust axis: the small body-x/y drag
                a = (thr - cz * vn * np.abs(vn))[:, None] * n - (self.c_lat * np.sqrt(np.einsum("ij,ij->i", vp, vp)))[:, None] * vp
                a[:, 2] -= G
            vv = v + h * np.vstack([np.zeros(3), np.cumsum(0.5 * (a[1:] + a[:-1]), axis=0)])
            pp = p + h * np.vstack([np.zeros(3), np.cumsum(0.5 * (vv[1:] + vv[:-1]), axis=0)])
            legs.append((pp, vv, a, n))
            p, v = pp[-1], vv[-1]
        return legs

    def plan(self, p0, v0, alim, cz, x0, maxiter):
        """Minimum-time plan through the aim points: SLSQP on the thrust-direction nodes and leg times."""
        from scipy.optimize import minimize
        nsub = [24, 24, 24, 16]
        aims = self.aim

        def con(x):
            legs = self._pm(x, p0, v0, alim, cz, nsub)
            return np.concatenate([legs[k][0][-1] - aims[k] for k in range(4)] + [[float(x[0:3] @ x[0:3]) - 1.0]])

        wmax = PARAMS["w_plan"]

        def rot(x):
            # the thrust direction may not turn faster than w_plan [rad/s] anywhere on a leg
            out = []
            for k in range(3):
                a, b = x[3 * k:3 * k + 3], x[3 * k + 3:3 * k + 6]
                d = b - a
                m = a + min(1.0, max(0.0, -float(a @ d) / (float(d @ d) + 1e-12))) * d
                out.append(wmax * float(m @ m) * x[12 + k] - float(np.linalg.norm(np.cross(a, b))))
            return np.array(out)

        res = minimize(lambda x: float(x[12:16].sum()), x0, method="SLSQP",
                       constraints=[{"type": "eq", "fun": con}, {"type": "ineq", "fun": rot}],
                       bounds=[(-5.0, 5.0), (-5.0, 5.0), (PARAMS["nz_min"], 5.0)] * 4 + [(0.15, 3.0)] * 4,
                       options={"maxiter": maxiter, "ftol": 1e-5})
        return res.x, float(np.abs(con(res.x)[:12]).max())

    def build_reference(self, x0=None):
        """The reference trajectory from the estimated state at the launch through the aim points (sampled at dt)."""
        P = PARAMS
        alim = P["eta_plan"] * 4.0 * U_MAX * self.kappa
        cz = P["drag_plan"] * CD[2] * self.kappa
        self.c_lat = P["drag_plan"] * 0.5 * (CD[0] + CD[1]) * self.kappa
        self.rot_loss, self.w_ref2 = P["rot_loss"], P["w_ref"] ** 2
        dt = self.dt
        p0, v0 = self.p.copy(), self.v.copy()
        x, miss = self.plan(p0, v0, alim, cz, np.array(X0) if x0 is None else x0, P["plan_iter"])
        if miss > 0.15 and x0 is None:                    # not converged: fall back to the stored nominal plan
            x = np.array(X0)
        self.plan_x, self.plan_miss = x, miss
        nsub = [max(2, int(round(x[12 + k] / dt))) for k in range(4)]
        legs = self._pm(x, p0, v0, alim, cz, nsub)
        ps, vs, acs, ns = [], [], [], []
        self.ref_gate = []
        for k in range(4):
            pp, vv, a, n = legs[k]
            ps.append(pp[:-1]); vs.append(vv[:-1]); acs.append(a[:-1]); ns.append(n[:-1])
            self.ref_gate.append(sum(len(q) for q in ps))
        tau = (dt * np.arange(1, 101))[:, None]            # coast on after the last gate with the last acceleration
        ps.append(pp[-1] + vv[-1] * tau + 0.5 * a[-1] * tau * tau)
        vs.append(vv[-1] + a[-1] * tau)
        acs.append(np.repeat(a[-1:], 100, axis=0)); ns.append(np.repeat(n[-1:], 100, axis=0))
        self.ref_p, self.ref_v, self.ref_a = np.vstack(ps), np.vstack(vs), np.vstack(acs)
        nr = np.vstack(ns)
        self.ref_w = np.cross(nr, np.gradient(nr, dt, axis=0))    # angular velocity of the thrust direction (world)
        self.ref_s = 0.0

    def _drag(self, v):
        R = self.R
        vrel = v - self.wind
        vb = R.T @ vrel
        kap = self.kappa                        # drag per mass ~ CD / m, with m ~ 1 / kappa (ks ~ 1)
        d_lat = -(CD[0] * kap * abs(vb[0]) * vb[0]) * R[:, 0] - (CD[1] * kap * abs(vb[1]) * vb[1]) * R[:, 1]
        gvec = G * EZ - d_lat
        return vrel, gvec, CD[2] * kap, float(vb[2])

    def track(self):
        """Reference tracking: feed-forward acceleration (taken `lead` ahead) plus PD on position and velocity.
        The reference clock slows down when the vehicle lags behind along the track."""
        P = PARAMS
        N = len(self.ref_p)
        s = self.ref_s
        i = int(s)
        if i >= N - 1:
            return None
        gi = self.gate_seen
        if gi < len(self.ref_gate) and s > self.ref_gate[gi] + (40 if gi < len(self.ref_gate) - 1 else 6):
            return None                              # the reference is past a gate that the vehicle has missed
        fr = s - i
        pr = self.ref_p[i] + fr * (self.ref_p[i + 1] - self.ref_p[i])
        vr = self.ref_v[i] + fr * (self.ref_v[i + 1] - self.ref_v[i])
        il = min(N - 1, int(s + P["ff_lead"] / self.dt))
        ar = self.ref_a[il]
        e = pr - self.p
        if float(e @ e) > P["err_max"] ** 2:
            return None
        sp = float(np.linalg.norm(vr))
        lag = float(e @ vr) / max(sp, 1.0)
        cross = math.sqrt(max(0.0, float(e @ e) - lag * lag))
        rate = min(P["rate_max"], max(P["rate_min"], P["rate_max"] - P["k_clock"] * max(lag - P["lag_free"], 0.0) - P["k_cross"] * cross))
        self.t_race += self.dt
        if self.t_race < P["t_rot"]:
            return self.ref_a[0].copy()               # first turn the thrust toward the initial direction of the plan
        self.ref_s = s + rate
        self.w_ff = (P["ff_rate"] * rate) * self.ref_w[il]
        fb = P["kp_t"] * e + P["kv_t"] * (rate * vr - self.v)
        nf = float(np.linalg.norm(fb))
        if nf > P["a_fb"]:
            fb *= P["a_fb"] / nf
        return (rate * rate) * ar + fb

    def guidance(self, obs, fw):
        P = PARAMS
        R = self.R
        n = len(self.gates)
        vrel, gvec, cz, vn = self._drag(self.v)
        amax = 4.0 * U_MAX * self.kappa
        self.gate_seen = int(obs["gate_index"])
        if P["replan"] > 0.5 and self.tracking and not self.replanned and bool(obs["clock_running"]) and self.gate_seen == 0:
            # the lap clock has just started: plan again from the state the vehicle really left the start sphere with
            self.replanned = True
            keep = (self.ref_p, self.ref_v, self.ref_a, self.ref_w, self.ref_gate, self.ref_s)
            x0 = np.array(self.plan_x)
            x0[12] = max(0.3, x0[12] - self.ref_s * self.dt)
            self.build_reference(x0)
            if self.plan_miss > 0.05:
                self.ref_p, self.ref_v, self.ref_a, self.ref_w, self.ref_gate, self.ref_s = keep
        if self.gate_seen >= n - 1:
            self.tracking = False                     # last leg: home on the last gate (free final velocity)
        a_cmd = self.track() if self.tracking else None
        if a_cmd is None:
            # fallback (reference finished or lost): constant-acceleration intercept of the next gate
            self.tracking = False
            k = min(int(obs["gate_index"]), n - 1)
            alim = P["eta"] * amax
            _, A0 = self._solve(self.gates[k] - self.p, self.v, None, gvec, alim, alim, cz)
        else:
            A0 = a_cmd + gvec
        nrm = float(np.linalg.norm(A0))
        n_des = A0 / max(nrm, 1e-6)
        # body-z drag as measured: modelled rotor thrust minus the accelerometer (moved to the centre of mass)
        w = self.w_last
        dz = self.kappa * float(self.That.sum()) - (float(self.acc_last[2]) - 0.034 * (w[0] * w[0] + w[1] * w[1]))
        self.dz_f += P["dz_gain"] * (dz - self.dz_f)
        f_cmd = float(A0 @ R[:, 2]) + self.dz_f
        return n_des, max(f_cmd, 0.05 * amax)

    # ----- attitude and mixer -----
    def attitude(self, n_des, f_cmd, w, wz_des, raise_to):
        P = PARAMS
        R = self.R
        nb = R.T @ n_des
        ax, ay = -nb[1], nb[0]
        s = math.hypot(ax, ay)
        if s > 1e-9:
            ang = math.atan2(s, nb[2])
            rate = min(P["k_att"] * ang, math.sqrt(2.0 * P["alpha_max"] * ang))
            wx, wy = rate * ax / s, rate * ay / s
        else:
            wx = wy = 0.0
        if self.racing and self.tracking:
            # feed-forward: the angular velocity of the reference's thrust direction
            wf = R.T @ self.w_ff
            wx += float(wf[0])
            wy += float(wf[1])
        alpha = np.array([P["k_rate"] * (wx - w[0]), P["k_rate"] * (wy - w[1]), P["k_rate_z"] * (wz_des - w[2])])
        tau = JN @ alpha + np.cross(w, JN @ w)
        scale = 1.0 / (M_NOM * self.kappa)          # command per nominal-vehicle newton
        tmax = U_MAX / scale
        d = SX * (tau[0] / (4.0 * ARM_Y)) + SY * (tau[1] / (4.0 * ARM_X))
        c = M_NOM * f_cmd / 4.0
        # the tilt torque may lower the collective thrust as far as needed, and raise it up to c_hi
        c_hi = max(c, raise_to * tmax)
        lim = min(c_hi, tmax - c_hi) if c_hi < 0.5 * tmax else 0.5 * tmax
        peak = float(np.abs(d).max())
        if peak > lim:
            d *= lim / peak
        zc = (P["yaw_cap"] if self.racing else 0.1) * tmax      # yaw gets only a small share of the rotor range
        z = SZ * max(-zc, min(zc, tau[2] / (4.0 * CQ)))
        dz = d + z
        if float(dz.max() - dz.min()) > tmax:
            dz = d
        c = max(-float(dz.min()), min(c, tmax - float(dz.max())))
        t_des = (c + dz) * scale
        u = self.That + P["lead"] * (t_des - self.That)
        return np.clip(u, 0.0, U_MAX)

    # ----- one control step -----
    def step(self, obs):
        P = PARAMS
        i = int(obs["step"])
        if i == 0:
            self.reset(obs)
            w = np.asarray(obs["gyro"], dtype=float)
            acc = np.asarray(obs["accel"], dtype=float)
            fw = self.fw_prev
        else:
            w, acc, fw = self.estimate(obs, i)
        t = float(obs["time"])
        R = self.R
        if not self.racing and t >= P["t_launch"]:
            self.racing = True
            self.set_observer(P["w0_race"])
            db = R.T @ self.dhat
            m = 1.0 / self.kappa
            wb = np.array([math.copysign(math.sqrt(abs(db[0]) * m / CD[0]), db[0]),
                           math.copysign(math.sqrt(abs(db[1]) * m / CD[1]), db[1]), 0.0])
            self.wind = R @ wb
            self.wind[2] = 0.0
            self.build_reference()
        if not self.racing:
            tsum = float(self.That.sum())
            if i >= 30:
                beta = 0.05 if i < 120 else 0.01
                self.kappa += beta * (acc[2] / tsum - self.kappa)
                self.kappa = min(max(self.kappa, 0.6 / M_NOM), 1.5 / M_NOM)
                ext = fw - (self.kappa * tsum) * R[:, 2]
                self.dhat += 0.015 * (ext - self.dhat)
            if i >= 200:
                # gyro bias: mean gyro reading minus the mean rotation rate seen by the observer
                if self.g_cnt == 0:
                    self.R_a = R.copy()
                self.g_sum += self.gyro_prev
                self.g_cnt += 1
                if self.g_cnt >= 100 and self.g_cnt % 20 == 0:
                    D = self.R_a.T @ R
                    rot = 0.5 * np.array([D[2, 1] - D[1, 2], D[0, 2] - D[2, 0], D[1, 0] - D[0, 1]])
                    self.bg = np.clip((self.g_sum - rot / self.dt) / self.g_cnt, -0.025, 0.025)
            a_cmd = P["kp"] * (self.hold - self.p) - P["kd"] * self.v
            an = float(np.linalg.norm(a_cmd))
            if an > 3.0:
                a_cmd *= 3.0 / an
            f_des = a_cmd - self.dhat
            f_des[2] += G
            nrm = float(np.linalg.norm(f_des))
            n_des = f_des / nrm
            f_cmd = float(f_des @ R[:, 2])
            yaw = math.atan2(R[1, 0], R[0, 0])
            ey = (self.yaw_ref - yaw + math.pi) % (2.0 * math.pi) - math.pi
            wz_des = 2.0 * ey
        else:
            self.w_last, self.acc_last = w, acc
            n_des, f_cmd = self.guidance(obs, fw)
            wz_des = 0.0
        u = self.attitude(n_des, f_cmd, w, wz_des, 0.5 if self.racing else 0.33)
        self.u_prev = u
        return u


_C = _Controller()


def controller(obs: dict) -> np.ndarray:
    return _C.step(obs)
