"""Commissioning controller for the compliant servo axis: identify the unit, then plan and track.

Control law
-----------
1. Commissioning, open loop (about 0.43 s): a fixed sequence of small zero-mean torque triplets
   (+A for T, -A for 2T, +A for T, with T = 1..9 steps so that 10..125 Hz are excited) followed by a
   free-ringing pause.  Amplitudes are chosen so that the lightest, softest unit of the family stays
   well inside the travel limit and no shaft sees more than about 0.8 N m.
2. Identification: the recorded encoder increments are compared with batched simulations of the
   plant's own model (Coulomb term included).  A dense grid over load inertia, first resonance,
   structure and tool ratio is scored for every dead time and both polarities; the best candidates
   are refined by Levenberg-Marquardt, and the structure with the smaller residual is kept.
3. With the identified unit: a steady-state Kalman filter (plant states plus a motor-side torque
   disturbance state, which is the integral action against friction and the load torque), a
   dead-time predictor, and an LQR state feedback on the predicted state.
4. Every reference change (and the return to rest after the experiment) is turned into a
   feedforward move by a small linear program: minimise the summed absolute tool error subject to
   a torque bound, a shaft-torque bound and rest at the target at the end of the horizon.  The LQR
   tracks the planned state trajectory, so feedback only has to correct model error and disturbances.

Gains (PARAMS)
--------------
exc_scale   scale of the identification torques (1 = the sequence described above).
fb_bw       LQR bandwidth as a multiple of the first resonance: tool-angle weight = (J_total w^2)^2.
kf_w        process-noise torque std [N m] of the Kalman filter (larger: trusts the encoder more).
kf_d        random-walk std [N m per step] of the disturbance state (speed of the integral action).
u_plan      torque bound of the planner [N m]; the headroom below 2 N m is left to the feedback.
s_plan      shaft-torque bound of the planner [N m]; the margin below the 2 N m rating covers model error.
fb_clip     bound on the feedback part of the command [N m].
hor         planning horizon in periods of the first resonance.
"""
import math

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

PARAMS = {"exc_scale": 1.0, "fb_bw": 0.8, "kf_w": 2e-3, "kf_d": 3e-4, "u_plan": 1.8, "s_plan": 1.5,
          "fb_clip": 0.6, "hor": 2.2}

JM, BM, TC, WS, BETA, MU = 2.0e-4, 2.0e-4, 0.004, 0.5, 2.0e-4, 0.35
DT, SUB = 0.002, 2
QUANT = 2.0 * math.pi / 65536
DIAG = {}


# ----------------------------------------------------------------------------- plant model
def _models(JL, f1, three, eta):
    """Batched 6-state continuous models (identical to hl_control's linear_model, padded)."""
    JL, f1, eta = (np.asarray(v, dtype=float) for v in (JL, f1, eta))
    three = np.asarray(three, dtype=bool)
    N = len(JL)
    w1 = 2.0 * math.pi * f1
    k1 = w1 * w1 * JM * JL / (JM + JL)
    c1 = BETA * k1
    j1 = np.where(three, (1.0 - MU) * JL, JL)
    j2 = MU * JL
    k2 = np.where(three, (eta * w1) ** 2 * (1.0 - MU) * MU * JL, 0.0)
    c2 = BETA * k2
    A = np.zeros((N, 6, 6))
    B = np.zeros((N, 6, 3))
    A[:, 0, 1] = 1.0
    A[:, 2, 3] = 1.0
    A[:, 4, 5] = three
    A[:, 1, 0], A[:, 1, 1], A[:, 1, 2], A[:, 1, 3] = -k1 / JM, (-c1 - BM) / JM, k1 / JM, c1 / JM
    A[:, 3, 0], A[:, 3, 1], A[:, 3, 2], A[:, 3, 3] = k1 / j1, c1 / j1, (-k1 - k2) / j1, (-c1 - c2) / j1
    A[:, 3, 4], A[:, 3, 5] = k2 / j1, c2 / j1
    A[:, 5, 2], A[:, 5, 3], A[:, 5, 4], A[:, 5, 5] = k2 / j2, c2 / j2, -k2 / j2, -c2 / j2
    B[:, 1, 0] = B[:, 1, 1] = 1.0 / JM
    B[:, 5, 2] = np.where(three, 1.0 / j2, 0.0)
    B[:, 3, 2] = np.where(three, 0.0, 1.0 / JL)
    S = np.zeros((N, 2, 6))
    S[:, 0, 0], S[:, 0, 1], S[:, 0, 2], S[:, 0, 3] = k1, c1, -k1, -c1
    S[:, 1, 2], S[:, 1, 3], S[:, 1, 4], S[:, 1, 5] = k2, c2, -k2, -c2
    return A, B, S, np.where(three, 4, 2)


def _zoh(A, B, h):
    """Batched exact zero-order hold (scaling and squaring, as the plant does)."""
    N = A.shape[0]
    blk = np.zeros((N, 9, 9))
    blk[:, :6, :6] = A * h
    blk[:, :6, 6:] = B * h
    nrm = float(np.abs(blk).sum(axis=1).max())
    sq = max(0, int(math.ceil(math.log2(max(nrm, 1e-16)))) + 4)
    s = blk / 2.0 ** sq
    out = np.broadcast_to(np.eye(9), (N, 9, 9)).copy()
    term = out.copy()
    for k in range(1, 14):
        term = term @ s / k
        out = out + term
    for _ in range(sq):
        out = out @ out
    return out[:, :6, :6], out[:, :6, 6:]


def _simulate(phi, gam, tool, u):
    """Tool angle of every candidate for the motor torque sequence u (no dead time), from rest."""
    N, K = phi.shape[0], len(u)
    x = np.zeros((N, 6, 1))
    g0, g1 = gam[:, :, 0:1], gam[:, :, 1:2]
    Y = np.zeros((K + 1, N))
    rows = np.arange(N)
    for k in range(K):
        for _ in range(SUB):
            c = -TC * np.tanh(x[:, 1, 0] / WS)
            x = phi @ x + g0 * u[k] + g1 * c[:, None, None]
        Y[k + 1] = x[rows, tool, 0]
    return Y


# ----------------------------------------------------------------------------- identification
def _excitation(scale):
    seq = [0.0] * 5
    for T, A, s in ((3, 0.25, 1.0), (2, 0.40, -1.0), (6, 0.05, 1.0), (1, 0.60, -1.0), (4, 0.12, -1.0), (9, 0.025, 1.0)):
        a = s * A * scale
        seq += [a] * T + [-a] * (2 * T) + [a] * T
    return seq


def _sim_dy(JL, f1, three, eta, u):
    """Simulated encoder increments (K, N) of the candidates for the commands u, without dead time."""
    A, B, _, tool = _models(JL, f1, np.full(len(JL), three), eta)
    phi, gam = _zoh(A, B, DT / SUB)
    return np.diff(_simulate(phi, gam, tool, u), axis=0)


def _score(JL, f1, three, eta, u, dy):
    """Sum of squared errors of the encoder increments for each candidate, minimised over dead time and polarity."""
    dY = _sim_dy(JL, f1, three, eta, u)
    K = len(dy)
    m = dy[6:]
    mm = float(m @ m)
    best = np.full(len(JL), np.inf)
    bd = np.zeros(len(JL), dtype=int)
    bp = np.ones(len(JL), dtype=int)
    for d in range(1, 7):
        seg = dY[6 - d:K - d]
        cross = m @ seg
        ss = (seg * seg).sum(axis=0)
        for p in (1, -1):
            e = mm - 2.0 * p * cross + ss
            better = e < best
            best = np.where(better, e, best)
            bd = np.where(better, d, bd)
            bp = np.where(better, p, bp)
    return best, bd, bp


_LO = np.log(np.array([0.7e-4, 9.0, 1.3]))
_HI = np.log(np.array([2.2e-3, 50.0, 3.5]))


def _lm(theta, d, p, three, u, dy, iters=14):
    """Batched Levenberg-Marquardt on (log J_L, log f_1[, log eta]) from several starts, dead time and polarity fixed."""
    npar = 3 if three else 2
    K = len(dy)
    m = dy[6:]
    nc = len(d)

    def resid(th, dd, pp):
        dY = _sim_dy(np.exp(th[:, 0]), np.exp(th[:, 1]), three, np.exp(th[:, 2]), u)
        R = np.empty((len(dd), K - 6))
        for v in np.unique(dd):
            sel = np.nonzero(dd == v)[0]
            R[sel] = (m[:, None] - pp[sel][None, :] * dY[6 - v:K - v, sel]).T
        return R

    theta = np.clip(theta, _LO, _HI)
    R0 = resid(theta, d, p)
    e0 = (R0 * R0).sum(axis=1)
    lam = np.full(nc, 1e-2)
    h = 1e-3
    for _ in range(iters):
        pert = np.repeat(theta, npar, axis=0)
        for j in range(npar):
            pert[j::npar, j] += h
        Rj = resid(pert, np.repeat(d, npar), np.repeat(p, npar)).reshape(nc, npar, K - 6)
        Jac = (Rj - R0[:, None, :]) / h                       # (nc, npar, T)
        g = np.einsum("cjt,ct->cj", Jac, R0)
        Hh = np.einsum("cjt,ckt->cjk", Jac, Jac)
        dg = np.einsum("cjj->cj", Hh)
        Hl = Hh + (lam[:, None] * dg)[:, :, None] * np.eye(npar)[None] + 1e-30 * np.eye(npar)[None]
        try:
            step = np.linalg.solve(Hl, -g[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            break
        new = theta.copy()
        new[:, :npar] += np.clip(step, -0.5, 0.5)
        new = np.clip(new, _LO, _HI)
        R1 = resid(new, d, p)
        e1 = (R1 * R1).sum(axis=1)
        acc = e1 < e0
        theta = np.where(acc[:, None], new, theta)
        R0 = np.where(acc[:, None], R1, R0)
        e0 = np.where(acc, e1, e0)
        lam = np.where(acc, lam / 3.0, lam * 5.0)
    return theta, e0


SSE_FLOOR_FACTOR = 15.0   # a fit is trusted when its residual is within this factor of the quantisation floor


def _identify(u, y):
    """Dense grid over the family (all dead times, both polarities), then Levenberg-Marquardt from the best
    candidates (the best of every tool-ratio slice among them); the structure with the smaller residual wins.
    If the residual stays well above the encoder's quantisation floor, the search is repeated from more starts."""
    u = np.asarray(u, dtype=float)
    dy = np.diff(np.asarray(y, dtype=float))
    Jg = np.geomspace(1.0e-4, 1.6e-3, 21)
    fg = np.geomspace(12.0, 40.0, 33)
    eg = np.geomspace(1.5, 3.0, 13)
    grids = {}
    for three in (False, True):
        es = eg if three else np.array([2.0])
        J, F, E = (a.ravel() for a in np.meshgrid(Jg, fg, es, indexing="ij"))
        grids[three] = (J, F, E) + _score(J, F, three, E, u, dy)
    sse_ok = SSE_FLOOR_FACTOR * (len(dy) - 6) * QUANT ** 2 / 6.0
    result = None
    for ntop in (5, 40):
        out = []
        for three in (False, True):
            J, F, E, e, d, p = grids[three]
            top = np.argsort(e)[:ntop]
            if three:
                e3 = e.reshape(len(Jg) * len(fg), len(eg))
                top = np.unique(np.concatenate([top, np.argmin(e3, axis=0) * len(eg) + np.arange(len(eg))]))
            theta = np.tile(np.log(np.column_stack([J[top], F[top], E[top]])), (3, 1))
            ds = np.clip(np.concatenate([d[top], d[top] - 1, d[top] + 1]), 1, 6)   # the grid's dead time may be off by one
            ps = np.tile(p[top], 3)
            theta, ee = _lm(theta, ds, ps, three, u, dy)
            i = int(np.argmin(ee))
            out.append({"JL": float(math.exp(theta[i, 0])), "f1": float(math.exp(theta[i, 1])), "three": three,
                        "eta": float(math.exp(theta[i, 2])), "sse": float(ee[i]), "d": int(ds[i]), "p": int(ps[i])})
        best = min(out, key=lambda o: o["sse"])
        if result is None or best["sse"] < result[0]["sse"]:
            result = (best, out)
        if result[0]["sse"] <= sse_ok:
            break
    return result


# ----------------------------------------------------------------------------- controller
class _Axis:
    def __init__(self):
        self.exc = _excitation(PARAMS["exc_scale"])
        self.k_fit = len(self.exc) + 100
        self.u_hist, self.y_hist = [], []
        self.ready = False        # True once the unit is identified and the loop is closed
        self.failed = False       # set if something raised: the axis is then left unpowered (finite commands)
        self.plan_k0 = None

    # ---- design on the identified unit
    def design(self, est):
        self.est = est
        three, self.d, self.p = est["three"], est["d"], est["p"]
        n = self.n = 6 if three else 4
        A, B, S, tool = _models([est["JL"]], [est["f1"]], [three], [est["eta"]])
        phi, gam = _zoh(A, B, DT / SUB)
        self.phi, self.g0, self.g1 = phi[0, :n, :n], gam[0, :n, 0], gam[0, :n, 1]
        self.S = S[0, :2 if three else 1, :n]
        self.tool = int(tool[0])
        self.Phi = self.phi @ self.phi
        self.Gam = self.phi @ self.g0 + self.g0
        self.ones = np.zeros(n)
        self.ones[0::2] = 1.0
        # LQR
        jt = JM + est["JL"]
        kp = jt * (2.0 * math.pi * PARAMS["fb_bw"] * est["f1"]) ** 2
        C = np.zeros((1, n))
        C[0, self.tool] = 1.0
        Q = kp * kp * (C.T @ C) + 1e-9 * np.eye(n)
        R = np.array([[1.0]])
        G = self.Gam.reshape(n, 1)
        P = solve_discrete_are(self.Phi, G, Q, R)
        self.K = np.linalg.solve(R + G.T @ P @ G, G.T @ P @ self.Phi).ravel()
        # Kalman filter with a motor-side disturbance state
        Fa = np.zeros((n + 1, n + 1))
        Fa[:n, :n], Fa[:n, n], Fa[n, n] = self.Phi, self.Gam, 1.0
        Ca = np.zeros((1, n + 1))
        Ca[0, self.tool] = 1.0
        Qn = np.zeros((n + 1, n + 1))
        Qn[:n, :n] = np.outer(self.Gam, self.Gam) * PARAMS["kf_w"] ** 2
        Qn[n, n] = PARAMS["kf_d"] ** 2
        Qn += 1e-16 * np.eye(n + 1)
        Rn = np.array([[QUANT ** 2 / 12.0]])
        Pf = solve_discrete_are(Fa.T, Ca.T, Qn, Rn)
        self.L = (Pf @ Ca.T / (Ca @ Pf @ Ca.T + Rn)).ravel()
        # planner tables
        self.H = int(np.clip(round(PARAMS["hor"] / (est["f1"] * DT)) + 2 * self.d, 50, 140))
        hmax = 2 * self.H
        W = np.zeros((hmax, n))
        w = self.Gam.copy()
        for j in range(hmax):
            W[j] = w
            w = self.Phi @ w
        self.W = W
        self.xh = np.zeros(n)
        self.dh = 0.0

    def plant_step(self, x, tau):
        for _ in range(SUB):
            x = self.phi @ x + self.g0 * tau + self.g1 * (-TC * math.tanh(x[1] / WS))
        return x

    def kf_update(self, y):
        innov = y - self.xh[self.tool]
        self.xh = self.xh + self.L[:self.n] * innov
        self.dh += self.L[self.n] * innov

    def tau_at(self, j):
        return self.p * self.u_hist[j] if 0 <= j < len(self.u_hist) else 0.0

    def predict_ahead(self, k):
        x = self.xh
        for j in range(self.d):
            x = self.plant_step(x, self.tau_at(k - self.d + j) + self.dh)
        return x

    # ---- feedforward move: linear program on the identified model
    def solve_move(self, e0, H, method):
        """LP in error coordinates (target = origin): torques that minimise the summed |tool error|."""
        n, tool = self.n, self.tool
        zf = np.empty((H + 1, n))
        zf[0] = e0
        for j in range(H):
            zf[j + 1] = self.Phi @ zf[j]
        W = self.W[:H]
        idx = np.arange(H)[:, None] - np.arange(H)[None, :]
        mask = idx >= 0
        idc = np.clip(idx, 0, H - 1)
        sc = 100.0                                            # tool error in units of 0.01 rad
        Ty = sc * np.where(mask, W[idc, tool], 0.0)
        ey = -sc * zf[1:, tool]
        simp = W @ self.S.T
        eye, zero = np.eye(H), np.zeros((H, H))
        rows_a, rows_b = [np.hstack([-Ty, -eye]), np.hstack([Ty, -eye])], [-ey, ey]
        lim = PARAMS["s_plan"]
        for s in range(simp.shape[1]):
            Ts = np.where(mask, simp[idc, s], 0.0)
            sf = zf[1:] @ self.S[s] + self.S[s] @ self.xr_plan    # shaft torque of the absolute state
            rows_a += [np.hstack([Ts, zero]), np.hstack([-Ts, zero])]
            rows_b += [lim - sf, lim + sf]
        MT = W[::-1].T
        rs = 1.0 / np.maximum(np.abs(MT).max(axis=1), 1e-12)
        A_eq = np.hstack([MT * rs[:, None], np.zeros((n, H))])
        b_eq = -zf[H] * rs
        c = np.concatenate([np.zeros(H), np.ones(H)])
        ul = PARAMS["u_plan"]
        res = linprog(c, A_ub=np.vstack(rows_a), b_ub=np.concatenate(rows_b), A_eq=A_eq, b_eq=b_eq,
                      bounds=[(-ul, ul)] * H + [(0.0, None)] * H, method=method)
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        return np.asarray(res.x[:H])

    def plan(self, x0, r, H, k0):
        n, tool = self.n, self.tool
        xr = r * self.ones
        self.xr_plan = xr
        self.plan_k0, self.plan_r = k0, r
        hmax = len(self.W)
        ff = None
        for hh, method in ((H, "highs"), (H + 9, "highs"), (H, "highs-ipm"), (int(1.4 * H), "highs"), (int(1.4 * H) + 7, "highs-ds")):
            hh = min(hh, hmax)
            try:
                ff = self.solve_move(x0 - xr, hh, method)
            except Exception:
                ff = None
            if ff is not None:
                break
        if ff is not None:
            z = np.empty((len(ff) + 1, n))
            z[0] = x0
            for j in range(len(ff)):
                z[j + 1] = self.Phi @ z[j] + self.Gam * ff[j]
            self.ff, self.zref = ff, z
            return True
        # fallback: smooth (min-jerk) rigid transition over the horizon, tracked by the feedback alone
        y0 = float(x0[tool])
        s = np.linspace(0.0, 1.0, H + 1)
        prof = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
        dprof = (30 * s ** 2 - 60 * s ** 3 + 30 * s ** 4) / (H * DT)
        z = np.zeros((H + 1, n))
        z[:, 0::2] = (y0 + (r - y0) * prof)[:, None]
        z[:, 1::2] = ((r - y0) * dprof)[:, None]
        self.ff, self.zref = np.zeros(H), z
        return False

    def control(self, k, r):
        xp = self.predict_ahead(k)
        if self.plan_k0 is None or r != self.plan_r:
            self.plan(xp, r, self.H if self.plan_k0 is not None else min(2 * self.H, len(self.W)), k)
        m = k - self.plan_k0
        if m < len(self.ff):
            ff, zr = self.ff[m], self.zref[m]
        else:
            ff, zr = 0.0, self.zref[-1]
        fb = -float(self.K @ (xp - zr))
        fb = min(max(fb, -PARAMS["fb_clip"]), PARAMS["fb_clip"])
        tau = ff + fb - self.dh
        return min(max(self.p * tau, -2.0), 2.0)

    def step(self, obs):
        k = int(obs["step"])
        y, r = float(obs["y"]), float(obs["r"])
        self.y_hist.append(y)
        if not self.ready:
            u = self.exc[k] if k < len(self.exc) else 0.0
            if k >= self.k_fit:
                est, both = _identify(self.u_hist, self.y_hist)
                DIAG["est"], DIAG["both"] = est, both
                self.design(est)
                for j in range(k + 1):          # replay the record through the filter
                    self.kf_update(self.y_hist[j])
                    if j < k:
                        self.xh = self.plant_step(self.xh, self.tau_at(j - self.d) + self.dh)
                self.ready = True
                u = self.control(k, r)
        else:
            self.kf_update(y)
            u = self.control(k, r)
        if not math.isfinite(u):
            u = 0.0
        self.u_hist.append(u)
        if self.ready:
            self.xh = self.plant_step(self.xh, self.tau_at(k - self.d) + self.dh)
        return u


_axis = None


def controller(obs: dict) -> float:
    """Motor torque command [N m] for one control step; step 0 starts a new unit."""
    global _axis
    if _axis is None or int(obs["step"]) == 0:
        _axis = _Axis()
    if _axis.failed:
        return 0.0
    try:
        u = float(_axis.step(obs))
    except Exception:
        _axis.failed = True
        return 0.0
    return u if math.isfinite(u) else 0.0
