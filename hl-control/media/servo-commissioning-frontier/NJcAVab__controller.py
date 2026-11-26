"""Commissioning controller for the compliant servo axis: identify the unit, then position it.

Control law
-----------
1. Excitation (first 0.39 s of the commissioning window, open loop): two torque doublets of
   different width, small enough to be safe on every unit of the family (travel < 0.2 rad,
   shaft torque < 1 N m), each followed by a pause in which the drive train rings.
2. Identification: prediction-error fit of the plant's own equations to the encoder record. Every
   candidate unit (load inertia, first resonance, two/three-mass structure, tool-stage ratio, dead
   time, polarity) is run as a Kalman predictor that integrates exactly as the plant does (1 ms
   substeps, tanh Coulomb friction held per substep; ripple and encoder quantisation as the noise
   model); the sum of squared innovations is the cost. A coarse grid over all structures and dead
   times picks the starts, a batched Levenberg-Marquardt loop refines them, and the best
   (structure, dead time) wins. The fit is repeated on the whole record after closed-loop rehearsal
   moves (the evaluation's own steps, +-0.05 rad) in the rest of the window; the axis then rests at 0.
3. Positioning: every change of the reference is answered with a feedforward move computed by a
   linear programme on the identified model: minimum sum of |tool error| subject to the torque
   limit, the shaft-torque rating and rest at the target (plus Coulomb compensation along the
   planned motor speed). The plan is tracked by observer-based state feedback: a Kalman filter on
   the encoder with a tool-side load-torque state (this is the integral action that rejects the
   load step) and an LQR acting on the state predicted over the dead time.

Gains (PARAMS)
--------------
rho      LQR control weight (the tool-angle weight is 1): smaller = stiffer feedback around the plan.
qw       Kalman random-walk variance of the load-torque state: larger = faster integral action.
rmeas    Kalman measurement-noise scale relative to the encoder quantisation variance.
u_lim    torque bound of the planned move [N m], below the 2 N m limit to leave room for feedback.
s_lim    shaft-torque bound of the planned move [N m], a margin below the 2 N m rating.
tv       total-variation weight on the planned torque (keeps the bang-bang plan from chattering).
horizon  length of the planned move in control steps (the plan must be at rest by then).
"""
import math
import warnings

import numpy as np
from scipy.linalg import solve_discrete_are
from scipy.optimize import linprog

warnings.simplefilter("ignore")   # candidates that diverge in the search are discarded by their cost, not reported

PARAMS = {"rho": 1e-3, "qw": 1e-7, "rmeas": 1.0, "u_lim": 1.9, "s_lim": 1.7, "tv": 1e-5, "horizon": 110}

JM, BM, TC, WS, BETA, MU = 2.0e-4, 2.0e-4, 0.004, 0.5, 2.0e-4, 0.35
DT, H = 0.002, 0.001
RIPPLE = 5e-4
QUANT = 2.0 * math.pi / 65536


# ----------------------------------------------------------------------------------------------
# batched plant model (same equations and integration as hl_control.envs.servo_commissioning)
# ----------------------------------------------------------------------------------------------
def batch_abc(JL, f1, three, eta):
    JL = np.atleast_1d(np.asarray(JL, float))
    f1 = np.atleast_1d(np.asarray(f1, float))
    three = np.atleast_1d(np.asarray(three)).astype(bool)
    eta = np.atleast_1d(np.asarray(eta, float))
    N = len(JL)
    w1 = 2 * np.pi * f1
    k1 = w1 * w1 * JM * JL / (JM + JL)
    c1 = BETA * k1
    j1 = np.where(three, (1 - MU) * JL, JL)
    j2 = MU * JL
    k2 = np.where(three, (eta * w1) ** 2 * j1 * j2 / (j1 + j2), 0.0)
    c2 = BETA * k2
    A = np.zeros((N, 6, 6))
    A[:, 0, 1] = 1.0
    A[:, 2, 3] = 1.0
    A[:, 4, 5] = three
    A[:, 1, 0] = -k1 / JM
    A[:, 1, 1] = (-c1 - BM) / JM
    A[:, 1, 2] = k1 / JM
    A[:, 1, 3] = c1 / JM
    A[:, 3, 0] = k1 / j1
    A[:, 3, 1] = c1 / j1
    A[:, 3, 2] = (-k1 - k2) / j1
    A[:, 3, 3] = (-c1 - c2) / j1
    A[:, 3, 4] = k2 / j1
    A[:, 3, 5] = c2 / j1
    A[:, 5, 2] = k2 / j2
    A[:, 5, 3] = c2 / j2
    A[:, 5, 4] = -k2 / j2
    A[:, 5, 5] = -c2 / j2
    B = np.zeros((N, 6, 2))
    B[:, 1, 0] = 1.0 / JM
    B[:, 5, 1] = np.where(three, 1.0 / j2, 0.0)
    B[:, 3, 1] = np.where(three, 0.0, 1.0 / JL)
    return A, B, k1, k2


def batch_zoh(A, B, h, squarings=14, order=8):
    N, n, m = B.shape
    M = np.zeros((N, n + m, n + m))
    M[:, :n, :n] = A * h
    M[:, :n, n:] = B * h
    Ms = M / (2.0 ** squarings)
    E = np.eye(n + m)[None] + Ms
    term = Ms
    for k in range(2, order + 1):
        term = np.matmul(term, Ms) / k
        E = E + term
    for _ in range(squarings):
        E = np.matmul(E, E)
    return E[:, :n, :n], E[:, :n, n:]


def models_for(JL, f1, three, eta):
    A, B, _, _ = batch_abc(JL, f1, three, eta)
    Phi, G = batch_zoh(A, B, H)
    tool_idx = np.where(np.atleast_1d(three), 4, 2)
    return Phi, np.ascontiguousarray(G[:, :, 0]), tool_idx


def batch_pem(Phi, G0, tool_idx, U, L, y):
    """Run every candidate as a fixed-gain Kalman predictor on the record. Returns innovations (K, N) and final X."""
    N = Phi.shape[0]
    K = U.shape[0]
    X = np.zeros((N, 6))
    X[:, 0::2] = y[0]
    E = np.zeros((K, N))
    ar = np.arange(N)
    cols = [np.ascontiguousarray(Phi[:, :, j]) for j in range(6)]
    for k in range(K):
        uk = U[k]
        for _ in range(2):
            tq = uk - TC * np.tanh(X[:, 1] / WS)
            Xn = G0 * tq[:, None]
            for j in range(6):
                Xn += cols[j] * X[:, j:j + 1]
            X = Xn
        e = y[k + 1] - X[ar, tool_idx]
        E[k] = e
        X += L * e[:, None]
    return E, X


def batch_kalman(JL, f1, three, eta, iters=200):
    """Steady-state Kalman gains (N, 6) for ripple at the motor and encoder quantisation (batched Riccati recursion)."""
    A, B, _, _ = batch_abc(JL, f1, three, eta)
    Phi, G = batch_zoh(A, B, DT)
    N = Phi.shape[0]
    g = G[:, :, 0]
    R0 = QUANT ** 2 / 12.0                      # covariances are scaled so that R = 1
    Q = g[:, :, None] * g[:, None, :] * (RIPPLE ** 2 / R0)
    idx = np.where(np.atleast_1d(three), 4, 2)
    ar = np.arange(N)
    PhiT = np.ascontiguousarray(Phi.transpose(0, 2, 1))
    P = Q * 10.0
    for _ in range(iters):
        Pm = np.matmul(np.matmul(Phi, P), PhiT) + Q
        col = Pm[ar, :, idx]
        K = col / (col[ar, idx] + 1.0)[:, None]
        P = Pm - K[:, :, None] * col[:, None, :]
    Pm = np.matmul(np.matmul(Phi, P), PhiT) + Q
    col = Pm[ar, :, idx]
    return col / (col[ar, idx] + 1.0)[:, None]


# ----------------------------------------------------------------------------------------------
# identification
# ----------------------------------------------------------------------------------------------
LIM_LO = np.array([math.log(0.8e-4), math.log(10.5), math.log(1.35)])
LIM_HI = np.array([math.log(2.0e-3), math.log(46.0), math.log(3.4)])


def coarse_search(u, y, pol):
    """Prediction-error grid over (JL, f1[, eta]) and dead time. Returns starts (cost, three, d, lnJL, lnf1, lneta), best first."""
    K = len(u)
    jl2 = np.exp(np.linspace(math.log(1e-4), math.log(1.6e-3), 15))
    f2 = np.exp(np.linspace(math.log(12.0), math.log(40.0), 30))
    jl3 = np.exp(np.linspace(math.log(1e-4), math.log(1.6e-3), 11))
    f3 = np.exp(np.linspace(math.log(12.0), math.log(40.0), 22))
    e3 = np.exp(np.linspace(math.log(1.5), math.log(3.0), 7))
    a, b = np.meshgrid(jl2, f2, indexing="ij")
    c, d_, e = np.meshgrid(jl3, f3, e3, indexing="ij")
    JL = np.concatenate([a.ravel(), c.ravel()])
    F1 = np.concatenate([b.ravel(), d_.ravel()])
    ETA = np.concatenate([np.full(a.size, 2.0), e.ravel()])
    TH = np.concatenate([np.zeros(a.size, bool), np.ones(c.size, bool)])
    Phi, G0, tool_idx = models_for(JL, F1, TH, ETA)
    L = batch_kalman(JL, F1, TH, ETA, iters=80)
    up = pol * np.asarray(u, float)
    yy = np.asarray(y, float)
    out = []
    for d in range(1, 7):
        U = np.zeros((K, 1))
        U[d:, 0] = up[:K - d]
        E, _ = batch_pem(Phi, G0, tool_idx, U, L, yy)
        cost = (E * E).sum(axis=0)
        cost = np.where(np.isfinite(cost), cost, np.inf)
        for i in np.argsort(cost)[:6]:
            out.append((float(cost[i]), bool(TH[i]), d, math.log(JL[i]), math.log(F1[i]), math.log(ETA[i])))
        for three in (False, True):                      # always keep the best of each structure
            idx = np.where(TH == three)[0]
            i = idx[np.argmin(cost[idx])]
            out.append((float(cost[i]), bool(three), d, math.log(JL[i]), math.log(F1[i]), math.log(ETA[i])))
    out = sorted(set(out), key=lambda t: t[0])
    return out


def pem_candidates(theta, three, d, pol, u, y, L):
    """Innovations (K, N) and final filtered states of candidates theta (N,3 logs) with gains L (N,6)."""
    K = len(u)
    N = theta.shape[0]
    Phi, G0, tool_idx = models_for(np.exp(theta[:, 0]), np.exp(theta[:, 1]), three, np.exp(theta[:, 2]))
    up = pol * np.asarray(u)
    U = np.zeros((K, N))
    for dd in np.unique(d):
        sel = np.where(d == dd)[0]
        U[dd:, sel] = up[:K - dd, None]
    return batch_pem(Phi, G0, tool_idx, U, L, np.asarray(y))


def lm_refine(starts, pol, u, y, rounds=(6, 6, 6), keep=6):
    """Batched Levenberg-Marquardt on the prediction error over several (structure, dead time) starts."""
    S = len(starts)
    three = np.array([s[1] for s in starts], bool)
    d = np.array([s[2] for s in starts], int)
    theta = np.clip(np.array([[s[3], s[4], s[5]] for s in starts], float), LIM_LO, LIM_HI)
    npar = np.where(three, 3, 2)
    delta = 1e-4
    three4, d4 = np.repeat(three, 4), np.repeat(d, 4)
    cost = np.full(S, np.inf)
    Xend = np.zeros((S, 6))
    for rnd, iters in enumerate(rounds):
        if rnd > 0 and S > keep:                       # prune to the most promising starts
            sel = np.argsort(cost)[:keep]
            three, d, theta, npar, Xend = three[sel], d[sel], theta[sel], npar[sel], Xend[sel]
            S = keep
            three4, d4 = np.repeat(three, 4), np.repeat(d, 4)
        L = batch_kalman(np.exp(theta[:, 0]), np.exp(theta[:, 1]), three, np.exp(theta[:, 2]))
        L4 = np.repeat(L, 4, axis=0)
        lam = np.full(S, 1e-3)
        cost = np.full(S, np.inf)
        trial = theta.copy()
        gs, Hs = [None] * S, [None] * S
        for it in range(iters + 1):
            TH = np.repeat(trial, 4, axis=0)
            for j in range(3):
                TH[1 + j::4, j] += delta
            E, X = pem_candidates(TH, three4, d4, pol, u, y, L4)
            for s in range(S):
                r0 = E[:, 4 * s]
                c0 = float(r0 @ r0)
                m = npar[s]
                J = (E[:, 4 * s + 1:4 * s + 1 + m] - r0[:, None]) / delta
                if np.isfinite(c0) and np.all(np.isfinite(J)) and c0 < cost[s]:
                    cost[s] = c0
                    theta[s] = trial[s]
                    Xend[s] = X[4 * s]
                    lam[s] = max(lam[s] * 0.25, 1e-9)
                    gs[s], Hs[s] = J.T @ r0, J.T @ J
                else:
                    lam[s] = min(lam[s] * 8.0, 1e6)
                trial[s] = theta[s]
                if Hs[s] is None:
                    continue                             # this start never produced a finite prediction
                Hm, g = Hs[s], gs[s]
                try:
                    step = -np.linalg.solve(Hm + lam[s] * np.diag(np.diag(Hm)) + 1e-14 * np.eye(m), g)
                except np.linalg.LinAlgError:
                    continue
                if np.all(np.isfinite(step)):
                    trial[s, :m] = theta[s, :m] + np.clip(step, -0.25, 0.25)
            trial = np.clip(trial, LIM_LO, LIM_HI)
    i = int(np.argmin(cost))
    if not np.isfinite(cost[i]):
        raise RuntimeError("identification produced no finite fit")
    return {"sse": float(cost[i]), "three": bool(three[i]), "d": int(d[i]), "JL": float(math.exp(theta[i, 0])),
            "f1": float(math.exp(theta[i, 1])), "eta": float(math.exp(theta[i, 2])), "X": Xend[i].copy(),
            "all": [(float(cost[s]), bool(three[s]), int(d[s])) for s in np.argsort(cost)]}


# ----------------------------------------------------------------------------------------------
# design for one identified unit
# ----------------------------------------------------------------------------------------------
class Design:
    def __init__(self, JL, f1, three, eta, d, pol, P):
        self.JL, self.f1, self.three, self.eta, self.d, self.pol = JL, f1, bool(three), eta, int(d), int(pol)
        A, B, k1, k2 = batch_abc(JL, f1, three, eta)
        n = 6 if three else 4
        self.n = n
        A, B = A[:, :n, :n], B[:, :n, :]
        Phi, G = batch_zoh(A, B, DT)
        Phih, Gh = batch_zoh(A, B, H)
        self.Phi, self.G, self.Phih, self.Gh = Phi[0], G[0], Phih[0], Gh[0]
        self.Gm, self.Gw = self.G[:, 0].copy(), self.G[:, 1].copy()
        self.Ghm, self.Ghw = self.Gh[:, 0].copy(), self.Gh[:, 1].copy()
        k1, k2 = float(k1[0]), float(k2[0])
        self.k1, self.k2 = k1, k2
        C = np.zeros(n)
        C[n - 2] = 1.0
        self.C = C
        if three:
            self.S = np.array([[k1, BETA * k1, -k1, -BETA * k1, 0, 0], [0, 0, k2, BETA * k2, -k2, -BETA * k2]])
            self.xw = np.array([-1 / k1 - 1 / k2, 0, -1 / k2, 0, 0, 0])
        else:
            self.S = np.array([[k1, BETA * k1, -k1, -BETA * k1]])
            self.xw = np.array([-1 / k1, 0, 0, 0])
        self.ang = np.zeros(n)
        self.ang[0::2] = 1.0
        # LQR on the delay-free model, applied to the dead-time-predicted state
        Q = np.outer(C, C)
        Pm = solve_discrete_are(self.Phi, self.Gm[:, None], Q + 1e-12 * np.eye(n), np.array([[P["rho"]]]))
        self.K = np.linalg.solve(np.array([[P["rho"]]]) + self.Gm[None] @ Pm @ self.Gm[:, None], self.Gm[None] @ Pm @ self.Phi)[0]
        # gentle gain for the fallback move (used only if the linear programme fails)
        rs = np.array([[0.5]])
        Ps = solve_discrete_are(self.Phi, self.Gm[:, None], Q + 1e-12 * np.eye(n), rs)
        self.Ksoft = np.linalg.solve(rs + self.Gm[None] @ Ps @ self.Gm[:, None], self.Gm[None] @ Ps @ self.Phi)[0]
        # predictor rows
        self.Kx = self.K @ np.linalg.matrix_power(self.Phi, self.d)
        self.Ku = np.array([self.K @ np.linalg.matrix_power(self.Phi, i - 1) @ self.Gm for i in range(1, self.d + 1)])
        # Kalman filter with tool-side load-torque state
        ne = n + 1
        Fe = np.eye(ne)
        Fe[:n, :n] = self.Phi
        Fe[:n, n] = self.Gw
        Ce = np.zeros((1, ne))
        Ce[0, :n] = C
        Qe = np.zeros((ne, ne))
        Qe[:n, :n] = np.outer(self.Gm, self.Gm) * RIPPLE ** 2 * 4.0
        Qe[n, n] = P["qw"]
        Qe += 1e-16 * np.eye(ne)
        Rm = np.array([[P["rmeas"] * QUANT ** 2 / 12.0]])
        Pe = solve_discrete_are(Fe.T, Ce.T, Qe, Rm)
        self.L = (Pe @ Ce.T / (Ce @ Pe @ Ce.T + Rm))[:, 0]

    def predict(self, x, w, tq):
        """One control step of the plant model (two substeps, Coulomb held per substep)."""
        for _ in range(2):
            t = tq - TC * math.tanh(x[1] / WS)
            x = self.Phih @ x + self.Ghm * t + self.Ghw * w
        return x

    def soft_plan(self, xi0, pend, N=300):
        """Fallback move: the model under a gentle, torque-limited state feedback (slow but safe)."""
        x = np.array(xi0, float)
        for i in range(self.d):
            x = self.Phi @ x + self.Gm * pend[i]
        tau = np.zeros(N)
        xi = np.zeros((N + 1, self.n))
        xi[0] = x
        for j in range(N):
            tau[j] = min(max(-float(self.Ksoft @ xi[j]), -0.8), 0.8)
            xi[j + 1] = self.Phi @ xi[j] + self.Gm * tau[j]
        return tau, xi

    def plan(self, xi0, pend, P, u_off=0.0, N=None):
        """Minimum-|tool error| feedforward to the origin of the deviation coordinates.

        xi0: state deviation now; pend: pending torque deviations, oldest first (len d).
        Returns (tau (N,), xi (N+1, n)) with xi[j] the planned deviation at time now + d + j.
        """
        n, d = self.n, self.d
        N = int(P["horizon"]) if N is None else N
        x = np.array(xi0, float)
        for i in range(d):
            x = self.Phi @ x + self.Gm * pend[i]
        # impulse responses
        g = np.zeros((N, n))
        v = self.Gm.copy()
        for m in range(N):
            g[m] = v
            v = self.Phi @ v
        free = np.zeros((N + 1, n))
        free[0] = x
        for j in range(N):
            free[j + 1] = self.Phi @ free[j]
        cy = g @ self.C
        ns = self.S.shape[0]
        sh = g @ self.S.T                       # (N, ns) shaft response at step ends
        shh = g @ (self.S @ self.Phih).T        # (N, ns) shaft response half a step later
        sdir = self.S @ self.Ghm                # direct mid-step term
        idx = np.arange(N)
        lag = idx[:, None] - idx[None, :]       # row j (state j+1), col i: g index = j - i
        mask = lag >= 0
        Cy = np.where(mask, cy[np.clip(lag, 0, N - 1)], 0.0)
        yfree = free[1:] @ self.C
        nv = 3 * N
        c = np.concatenate([np.zeros(N), np.ones(N), np.full(N, P["tv"])])
        I = np.eye(N)
        Z = np.zeros((N, N))
        rows = [np.hstack([Cy, -I, Z]), np.hstack([-Cy, -I, Z])]
        rhs = [-yfree, yfree]
        # total variation |tau_j - tau_{j-1}| <= v_j
        Dm = np.eye(N) - np.eye(N, k=-1)
        prev = np.zeros(N)
        prev[0] = pend[-1] if d > 0 else 0.0
        rows += [np.hstack([Dm, Z, -I]), np.hstack([-Dm, Z, -I])]
        rhs += [prev, -prev]
        s_lim = P["s_lim"]
        NS = min(N, 70)
        for q in range(ns):
            Se = np.where(mask, sh[:, q][np.clip(lag, 0, N - 1)], 0.0)[:NS]
            sfree = (free[1:] @ self.S[q])[:NS]
            lag2 = lag - 1
            Sm = np.where(lag2 >= 0, shh[:, q][np.clip(lag2, 0, N - 1)], 0.0) + np.eye(N) * sdir[q]
            Sm = Sm[:NS]
            smfree = (free[:-1] @ (self.S[q] @ self.Phih))[:NS]
            Zs = np.zeros((NS, N))
            rows += [np.hstack([Se, Zs, Zs]), np.hstack([-Se, Zs, Zs]), np.hstack([Sm, Zs, Zs]), np.hstack([-Sm, Zs, Zs])]
            rhs += [s_lim - sfree, s_lim + sfree, s_lim - smfree, s_lim + smfree]
        A_ub = np.vstack(rows)
        b_ub = np.concatenate(rhs)
        # terminal rest: xi_N = 0
        Aeq = np.zeros((n, nv))
        Aeq[:, :N] = g[::-1].T
        beq = -free[N]
        scale = np.maximum(np.abs(Aeq).max(axis=1), 1e-12)
        Aeq, beq = Aeq / scale[:, None], beq / scale
        ul = P["u_lim"]
        bounds = [(-ul - u_off, ul - u_off)] * N + [(0, None)] * (2 * N)
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs", options={"time_limit": 2.0})
        if res.status != 0 or res.x is None or not np.all(np.isfinite(res.x)):
            return None
        tau = res.x[:N]
        xi = np.zeros((N + 1, n))
        xi[0] = x
        for j in range(N):
            xi[j + 1] = self.Phi @ xi[j] + self.Gm * tau[j]
        return tau, xi


# ----------------------------------------------------------------------------------------------
# episode logic
# ----------------------------------------------------------------------------------------------
def excitation():
    seq = [0.0] * 3
    seq += [0.21] * 5 + [-0.21] * 5 + [0.0] * 75
    seq += [-0.0525] * 10 + [0.0525] * 10 + [0.0] * 75
    return np.array(seq)


EXC = excitation()


class Ctl:
    def __init__(self):
        self.u = []          # commands as applied (clipped)
        self.y = []
        self.des = None
        self.pol = 1
        self.tau = []        # torque in positive-torque coordinates, design frame
        self.r = 0.0
        self.plan_k0 = None
        self.fit = None
        self.soft_plans = 0
        self.failed = False

    # --- identification ---
    def identify(self, first):
        """Fit the unit to the record so far. first: full search; otherwise refine around the current fit."""
        u = np.array(self.u)
        y = np.array(self.y)
        floor = 6e-9 * len(u)                  # about four times the innovation energy of the true unit
        if first:
            dy = y - y[0]
            i = int(np.argmax(np.abs(dy[:60])))
            self.pol = 1 if dy[i] >= 0 else -1  # the first doublet starts with a positive command
            fit = lm_refine(coarse_search(u, y, self.pol)[:10], self.pol, u, y, rounds=(6, 6), keep=4)
            if fit["sse"] > floor:              # not at the noise floor: search wider, and the other polarity too
                for pol, top in ((-self.pol, 10), (self.pol, 40), (-self.pol, 40)):
                    alt = lm_refine(coarse_search(u, y, pol)[:top], pol, u, y, rounds=(6, 8), keep=8)
                    if alt["sse"] < fit["sse"]:
                        fit, self.pol = alt, pol
                    if fit["sse"] <= floor:
                        break
        else:
            f = self.fit
            th = (math.log(f["JL"]), math.log(f["f1"]), math.log(f["eta"]))
            starts = [(0.0, f["three"], f["d"]) + th, (0.0, not f["three"], f["d"]) + th]
            starts += [(0.0, f["three"], dd) + th for dd in (f["d"] - 1, f["d"] + 1) if 1 <= dd <= 6]
            fit = lm_refine(starts, self.pol, u, y, rounds=(5, 4))
        self.fit = fit
        return fit

    def make_design(self):
        f = self.fit
        self.des = Design(f["JL"], f["f1"], f["three"], f["eta"], f["d"], self.pol, PARAMS)
        n = self.des.n
        self.x = f["X"][:n].copy()     # filtered state at the current step (after len(u) steps)
        self.w = 0.0

    # --- planning ---
    def new_target(self, r):
        des = self.des
        self.r = r
        k = len(self.u)
        xT = r * des.ang + self.w * des.xw
        uT = -self.w
        d = des.d
        pend = [self.tau_nofc[k - d + i] - uT if k - d + i >= 0 else -uT for i in range(d)]
        out = None
        for N in (int(PARAMS["horizon"]), int(PARAMS["horizon"]) + 70):
            try:
                out = des.plan(self.x - xT, pend, PARAMS, u_off=uT, N=N)
            except Exception:
                out = None
            if out is not None:
                break
        if out is None:
            out = des.soft_plan(self.x - xT, pend)
            self.soft_plans += 1
        self.plan_tau, self.plan_xi = out
        self.plan_k0 = k

    def control(self):
        des = self.des
        k = len(self.u)
        d = des.d
        j = k - self.plan_k0
        N = len(self.plan_tau)
        if j < N:
            tff, xi_ref = self.plan_tau[j], self.plan_xi[j]
            wm = 0.5 * (self.plan_xi[j][1] + self.plan_xi[j + 1][1])
            fc = TC * math.tanh(wm / WS)
        else:
            tff, xi_ref, fc = 0.0, np.zeros(des.n), 0.0
        xT = self.r * des.ang + self.w * des.xw
        uT = -self.w
        e = des.Kx @ (self.x - xT) - des.K @ xi_ref
        for i in range(1, d + 1):
            kk = k - i
            tprev = (self.tau_nofc[kk] - uT) if kk >= 0 else -uT
            e += des.Ku[i - 1] * tprev
        tau_lin = uT + tff - e
        tau = tau_lin + fc
        return tau, tau_lin

    # --- main step ---
    def step(self, obs):
        k = int(obs["step"])
        y = float(obs["y"])
        self.y.append(y)
        comm = bool(obs["commissioning"])
        tte = float(obs["time_to_evaluation"])
        if self.des is None:
            if k < len(EXC) and (tte > 0.5 or not comm):
                u = float(EXC[k])
                self.u.append(u)
                return u
            self.identify(first=True)
            self.make_design()
            self.tau = [self.pol * v for v in self.u]
            self.tau_nofc = list(self.tau)
            self.new_target(0.0)
            # rehearsal schedule (commissioning only): list of (step, target)
            sched = []
            n_left = int(round(tte / DT))
            if comm and n_left >= 1100:
                k1 = k + 110
                sched = [(k1, 0.05), (k1 + 170, -0.05), (k1 + 340, 0.0)]
                self.refit_at = k1 + 500
                if n_left >= 1600:
                    k2 = self.refit_at + 10
                    sched += [(k2, -0.05), (k2 + 170, 0.05), (k2 + 340, 0.0)]
                    self.refit2_at = k2 + 500
                else:
                    self.refit2_at = None
            else:
                self.refit_at = None
                self.refit2_at = None
            self.sched = sched
        else:
            # Kalman measurement update (prediction was made at the end of the previous step)
            des = self.des
            innov = y - self.x[des.n - 2]
            self.x = self.x + des.L[:des.n] * innov
            self.w = self.w + des.L[des.n] * innov
            if comm:
                if self.refit_at is not None and k == self.refit_at or self.refit2_at is not None and k == self.refit2_at:
                    old = self.fit
                    try:
                        self.identify(first=False)
                        self.make_design()
                        self.new_target(0.0)
                    except Exception:
                        self.fit = old
                for (ks, rt) in self.sched:
                    if k == ks:
                        self.new_target(rt)
            else:
                r = float(obs["r"])
                if abs(r - self.r) > 1e-9:
                    self.new_target(r)
        tau, tau_lin = self.control()
        fc = tau - tau_lin
        u = self.pol * tau
        u = min(max(u, -2.0), 2.0)
        tau = self.pol * u
        self.u.append(u)
        self.tau.append(tau)
        self.tau_nofc.append(tau - fc)   # command net of the friction compensation (what the linear plan sees)
        # predict to the next step with the torque that acts now
        des = self.des
        ka = len(self.tau) - 1 - des.d
        tq = self.tau[ka] if ka >= 0 else 0.0
        self.x = des.predict(self.x, self.w, tq)
        return float(u)


_C = None


def controller(obs: dict) -> float:
    global _C
    if _C is None or int(obs["step"]) == 0:
        _C = Ctl()
    if _C.failed:
        return 0.0                      # something broke earlier in this episode: stay passive (safe)
    try:
        with np.errstate(all="ignore"):
            u = _C.step(obs)
        if not math.isfinite(u):
            _C.failed = True
            return 0.0
        return u
    except Exception:
        _C.failed = True
        if DEBUG:
            raise
        return 0.0


DEBUG = False
