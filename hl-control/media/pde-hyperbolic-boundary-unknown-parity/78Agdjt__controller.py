"""Certainty-equivalent LQG boundary controller for the 2 x 2 hyperbolic plant with hidden couplings.

Control law
-----------
The plant is linear in its state once the couplings theta = (s1, s2, g) are fixed, and the initial state
lives in a known 6-dimensional space (three sine modes for u, three for v, amplitudes a on the unit
sphere).  So only nine numbers are unknown, and the controller

1. *identifies* (theta, a) from the boundary record y(0..k) and its own past inputs.  For a candidate
   theta a copy of the plant scheme is run from the six basis initial states and for the recorded
   input; the amplitudes then follow from linear least squares, and theta is improved by damped
   Gauss-Newton steps on the residual of the fit constrained to |a| = 1 (finite-difference Jacobian,
   box of the coupling family, restart from a coarse grid of couplings when the fit is poor).
   Two copies of the model are fitted side by side at first, on 100 and on 200 intervals, and the one
   that explains the record better is kept: the plant runs on one of the two grids and the fit is
   sensitive to the difference (residual 1e-3 on the right grid, 2e-3 .. 4e-3 on the wrong one).
   The state estimate x_hat is the state of the fitted model, with the posterior-mean amplitudes
   (unit-sphere prior in its Gaussian approximation), i.e. unobserved directions are shrunk to zero.
2. between two fits, and after the identification window, propagates x_hat with the model and a
   steady-state Kalman output injection  x_hat += L(theta) (y - u_hat(1));
3. applies  U = -K(theta) x_hat, the LQR gain of the benchmark's own cost
   J = int ||(u, v)||^2 + 0.1 U^2 (discrete Riccati equation of the model, doubling algorithm).

K(theta) and L(theta) are tabulated once at import on a 3 x 3 x 3 grid over the coupling family (for
both model grids) and interpolated quadratically (error of the gains below 0.2 %).

Gains (PARAMS)
--------------
sigma       output noise standard deviation: weight of the priors against the residual in the fit
k_first     first step at which a fit is made (amplitudes only, theta at the centre of its box)
k_theta     first step at which theta is adapted
k_commit    earliest step at which the better of the two model grids is kept and the other dropped
k_end       last step with a batch fit; afterwards observer only
upd_div     a fit is made every max(1, k // upd_div) steps
gn_iters    Gauss-Newton iterations per fit
ms_rms      a coarse-grid restart is tried when the residual rms of the fit exceeds ms_rms * sigma
q_proc      process-noise intensity of the Kalman design (observer bandwidth)
r_meas      measurement-noise variance of the Kalman design
u_lim       magnitude limit applied to the command (inside the plant's own limit)
"""

from __future__ import annotations

import itertools
import math
import os
import tempfile

import numpy as np
from scipy.optimize import brentq

PARAMS = {
    "sigma": 1.0e-3,
    "k_first": 3,
    "k_theta": 8,
    "k_commit": 40,
    "k_end": 150,
    "upd_div": 15,
    "gn_iters": 2,
    "ms_rms": 1.6,
    "q_proc": 1.0e-6,
    "r_meas": 1.0e-6,
    "u_lim": 10.0,
}

DT = 0.01                                    # control step
KO = 0.1 / 16.0                              # Kreiss-Oliger coefficient of the plant scheme
R_IN = 0.1                                   # input weight of the cost
BOX_LO = np.array([1.5, 1.5, -0.5])          # the coupling family (s1, s2, g)
BOX_HI = np.array([2.6, 2.6, 0.5])
TH_C = 0.5 * (BOX_LO + BOX_HI)
TH_STD = (BOX_HI - BOX_LO) / math.sqrt(12.0)
AXES = tuple(np.array([lo, 0.5 * (lo + hi), hi]) for lo, hi in zip(BOX_LO, BOX_HI))
MS_GRID = np.array(list(itertools.product([1.68, 2.05, 2.42], [1.68, 2.05, 2.42], [-0.33, 0.0, 0.33])))
FD = np.array([0.02, 0.02, 0.02])            # finite-difference steps in theta


def _dare(A, G, H, iters=40, tol=1e-11):
    """Structure-preserving doubling for X = A' X (I + G X)^-1 A + H."""
    n = A.shape[0]
    eye = np.eye(n)
    Ak, Gk, Hk = A.copy(), G.copy(), H.copy()
    for _ in range(iters):
        sol = np.linalg.solve(eye + Gk @ Hk, np.hstack([Ak, Gk]))
        WA, WG = sol[:, :n], sol[:, n:]
        H1 = Hk + Ak.T @ Hk @ WA
        G1 = Gk + Ak @ WG @ Ak.T
        A1 = Ak @ WA
        d = np.abs(H1 - Hk).max() / max(np.abs(H1).max(), 1e-300)
        Ak, Gk, Hk = A1, G1, H1
        if d < tol:
            break
    return Hk


def _lagrange(axis, value):
    w = np.ones(3)
    for i in range(3):
        for j in range(3):
            if i != j:
                w[i] *= (value - axis[j]) / (axis[i] - axis[j])
    return w


class _Model:
    """Copy of the plant scheme on 100 * m intervals (m substeps per control step), batched over rows."""

    def __init__(self, m: int):
        self.m = m
        self.n = 100 * m
        self.nx = self.n + 1
        self.x = np.linspace(0.0, 1.0, self.nx)
        modes = np.sqrt(2.0) * np.sin(np.pi * np.outer(self.x, [1.0, 2.0, 3.0]))
        self.u0 = np.zeros((7, self.nx))             # rows 0..5: the six basis initial states; row 6: zero (forced response)
        self.v0 = np.zeros((7, self.nx))
        self.u0[0:3] = modes.T
        self.v0[3:6] = modes.T
        w = np.full(self.nx, 1.0 / self.n)
        w[0] = w[-1] = 0.5 / self.n
        self.wq2 = np.concatenate([w, w])
        self._tables = {}

    def coeffs(self, theta):
        """Half-substep reaction matrices [[c, a12], [a21, c]] at every node, one row per row of theta."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        sig1 = theta[:, 0:1] * np.ones((1, self.nx))
        sig2 = theta[:, 1:2] * (1.0 + theta[:, 2:3] * (2.0 * self.x - 1.0)[None, :])
        w = np.sqrt(sig1 * sig2)
        wt = w * (0.5 / self.n)
        ratio = np.sinh(wt) / w
        return np.cosh(wt), sig1 * ratio, sig2 * ratio

    def step(self, u, v, U, coef):
        """One control step for a batch of states (rows); U is a scalar or a column of inputs."""
        c, a12, a21 = coef
        for _ in range(self.m):
            u1 = c * u + a12 * v
            v1 = a21 * u + c * v
            un = np.empty_like(u)
            vn = np.empty_like(v)
            un[:, 1:] = u1[:, :-1]
            vn[:, :-2] = v1[:, 2:]
            vn[:, -2:] = U
            un[:, 0] = vn[:, 0]
            u = c * un + a12 * vn
            v = a21 * un + c * vn
            u[:, 2:-2] -= KO * (u[:, :-4] - 4.0 * u[:, 1:-3] + 6.0 * u[:, 2:-2] - 4.0 * u[:, 3:-1] + u[:, 4:])
            v[:, 2:-2] -= KO * (v[:, :-4] - 4.0 * v[:, 1:-3] + 6.0 * v[:, 2:-2] - 4.0 * v[:, 3:-1] + v[:, 4:])
        return u, v

    def gains(self, theta, q_proc, r_meas):
        """LQR gain K (U = -K x) and Kalman filter gain L of the model at theta."""
        nx = self.nx
        coef = self.coeffs(theta)
        eye = np.eye(2 * nx)
        u, v = self.step(eye[:, :nx].copy(), eye[:, nx:].copy(), 0.0, coef)
        A = np.hstack([u, v]).T.copy()
        u, v = self.step(np.zeros((1, nx)), np.zeros((1, nx)), 1.0, coef)
        B = np.hstack([u, v]).T.copy()
        R = R_IN * DT
        P = _dare(A, (B @ B.T) / R, np.diag(self.wq2) * DT)
        K = (B.T @ P @ A)[0] / (R + float(B[:, 0] @ P @ B[:, 0]))
        Cm = np.zeros((2 * nx, 2 * nx))
        Cm[self.n, self.n] = 1.0 / r_meas
        Pf = _dare(A.T, Cm, q_proc * np.eye(2 * nx))
        L = Pf[:, self.n] / (Pf[self.n, self.n] + r_meas)
        return K, L

    def table(self):
        """(K, L) on the 3 x 3 x 3 grid AXES; computed once (kept in the temporary directory when it is writable)."""
        key = (float(PARAMS["q_proc"]), float(PARAMS["r_meas"]))
        if key in self._tables:
            return self._tables[key]
        path = os.path.join(tempfile.gettempdir(), "hl_pde_hyp_gains_v2_%d_%g_%g.npz" % ((self.n,) + key))
        tab = None
        try:
            with np.load(path) as f:
                tab = (np.array(f["K"]), np.array(f["L"]))
            if tab[0].shape != (3, 3, 3, 2 * self.nx) or not (np.all(np.isfinite(tab[0])) and np.all(np.isfinite(tab[1]))):
                tab = None
        except Exception:
            tab = None
        if tab is None:
            Kt = np.zeros((3, 3, 3, 2 * self.nx))
            Lt = np.zeros((3, 3, 3, 2 * self.nx))
            for i, j, l in itertools.product(range(3), range(3), range(3)):
                Kt[i, j, l], Lt[i, j, l] = self.gains((AXES[0][i], AXES[1][j], AXES[2][l]), *key)
            tab = (Kt, Lt)
            try:
                tmp = path + ".%d.tmp.npz" % os.getpid()
                np.savez(tmp, K=Kt, L=Lt)
                os.replace(tmp, path)
            except Exception:
                pass
        self._tables[key] = tab
        return tab

    def interp_gains(self, theta):
        Kt, Lt = self.table()
        w0, w1, w2 = (_lagrange(AXES[i], float(theta[i])) for i in range(3))
        w = (w0[:, None, None] * w1[None, :, None] * w2[None, None, :]).reshape(-1)
        return w @ Kt.reshape(27, -1), w @ Lt.reshape(27, -1)


MODELS = (_Model(1), _Model(2))


# ----------------------------------------------------------------------------- identification
def _sphere_mean(lam, Q, c, s2):
    """Posterior-mean amplitudes for the prior 'a uniform on the unit sphere', in the Gaussian (spherical-model)
    approximation: a = (G'G + rho I)^-1 G'e with rho such that E|a|^2 = |a|^2 + trace(cov) = 1.  With no data
    this is the ridge 6 sigma^2 of an isotropic prior; once some directions are known it gives the unknown ones
    the remaining radius, and it never returns an estimate outside the unit ball."""
    c2 = c * c
    off = lam - lam[0]

    def h(logd):
        d = off + math.exp(logd)
        return float(np.sum(c2 / (d * d) + s2 / d)) - 1.0

    lo = math.log(0.5 * s2)
    hi = math.log(1.01 * max(2.0 * math.sqrt(float(c2.sum())), 12.0 * s2))
    try:
        d0 = math.exp(brentq(h, lo, hi, xtol=1e-9, maxiter=200))
    except Exception:
        d0 = 6.0 * s2 + max(float(lam[0]), 0.0)
    return Q @ (c / (off + d0))


def _sphere_ls(lam, Q, c):
    """argmin |e - G a|^2 subject to |a| = 1 (secular equation, Newton on 1 / |a(mu)| - 1)."""
    c2 = c * c
    scale = max(float(lam[-1]), 1e-300)
    mu = -float(lam[0]) + max(abs(float(c[0])), 1e-13 * scale)
    for _ in range(25):
        d = lam + mu
        n2 = float(np.sum(c2 / (d * d)))
        if not (n2 > 0.0) or not math.isfinite(n2):
            break
        nrm = math.sqrt(n2)
        dphi = float(np.sum(c2 / (d * d * d))) / (n2 * nrm)
        mu_new = mu - (1.0 / nrm - 1.0) / dphi
        if not math.isfinite(mu_new) or mu_new <= -float(lam[0]):
            break
        if abs(mu_new - mu) <= 1e-12 * (abs(mu) + scale * 1e-12):
            mu = mu_new
            break
        mu = mu_new
    a = Q @ (c / (lam + mu))
    nrm = float(np.linalg.norm(a))
    return a / nrm if nrm > 0.0 and math.isfinite(nrm) else None


def _evaluate(model, thetas, y, us, state_rows=(0,)):
    """For each row of thetas: fit of the amplitudes on y[0..k], k = len(us).  Returns a list of
    (cost, residual of the unit-sphere fit, u_hat, v_hat); the state estimate at step k is formed for the
    rows listed in state_rows only."""
    thetas = np.atleast_2d(thetas)
    P = thetas.shape[0]
    k = len(us)
    coef = model.coeffs(np.repeat(thetas, 7, axis=0))
    u = np.tile(model.u0, (P, 1))
    v = np.tile(model.v0, (P, 1))
    Uin = np.zeros((7 * P, 1))
    forced = np.arange(P) * 7 + 6
    Y = np.empty((k + 1, 7 * P))
    Y[0] = u[:, -1]
    step = model.step
    for j in range(k):
        Uin[forced, 0] = us[j]
        u, v = step(u, v, Uin, coef)
        Y[j + 1] = u[:, -1]
    Y = Y.reshape(k + 1, P, 7)
    s2 = PARAMS["sigma"] ** 2
    out = []
    for p in range(P):
        G = Y[:, p, :6]
        e = y - Y[:, p, 6]
        lam, Q = np.linalg.eigh(G.T @ G)
        c = Q.T @ (G.T @ e)
        a_con = _sphere_ls(lam, Q, c)
        if a_con is None:
            a_con = _sphere_mean(lam, Q, c, s2)
        r = e - G @ a_con
        dth = (thetas[p] - TH_C) / TH_STD
        cost = float(r @ r) + s2 * float(dth @ dth)
        uh = vh = None
        if p in state_rows:
            a = _sphere_mean(lam, Q, c, s2)
            rows = slice(7 * p, 7 * p + 6)
            uh = a @ u[rows] + u[7 * p + 6]
            vh = a @ v[rows] + v[7 * p + 6]
        out.append((cost, r, uh, vh))
    return out


class _Chain:
    """Gauss-Newton chain of one model grid."""

    def __init__(self, model):
        self.model = model
        self.theta = TH_C.copy()
        self.lam = 1.0e-2                    # Levenberg-Marquardt damping
        self.cost = math.inf
        self.uh = None
        self.vh = None
        self.last_ms = -10 ** 9

    def _gn(self, theta, y, us, iters):
        s2 = PARAMS["sigma"] ** 2
        model = self.model
        cur = None
        for _ in range(iters):
            sign = np.where(theta + FD > BOX_HI, -1.0, 1.0)
            ev = _evaluate(model, np.vstack([theta[None, :], theta[None, :] + np.diag(sign * FD)]), y, us)
            if cur is not None and ev[0][0] > cur[0][0]:
                self.lam = min(self.lam * 10.0, 1.0e6)                  # reject: back to the last accepted point
            else:
                J = np.stack([(ev[j + 1][1] - ev[0][1]) / (sign[j] * FD[j]) for j in range(3)], axis=1)
                cur = (ev[0], J, theta.copy())
                self.lam = max(self.lam / 3.0, 1.0e-4)
            base, J, th0 = cur
            H = J.T @ J + s2 * np.diag(1.0 / TH_STD ** 2)
            grad = J.T @ base[1] + s2 * (th0 - TH_C) / TH_STD ** 2
            delta = -np.linalg.solve(H + self.lam * np.diag(np.diag(H)), grad)
            theta = np.clip(th0 + delta, BOX_LO, BOX_HI)
        final = _evaluate(model, theta[None, :], y, us)[0]
        if cur is not None and final[0] > cur[0][0]:
            final, theta = cur[0], cur[2]
            self.lam = min(self.lam * 10.0, 1.0e6)
        return final, theta

    def fit(self, k, y, us, adapt):
        iters = int(PARAMS["gn_iters"]) if adapt else 0
        final, theta = self._gn(self.theta.copy(), y, us, iters)
        rms = math.sqrt(max(final[0], 0.0) / (k + 1))
        if adapt and k >= 12 and rms > PARAMS["ms_rms"] * PARAMS["sigma"] and k - self.last_ms >= max(4, k // 6):
            self.last_ms = k                                            # poor fit: look for a better basin
            ev = _evaluate(self.model, MS_GRID, y, us, state_rows=())
            best = int(np.argmin([e[0] for e in ev]))
            if ev[best][0] < final[0]:
                lam_keep = self.lam
                self.lam = 1.0e-2
                alt, th_alt = self._gn(MS_GRID[best].copy(), y, us, iters + 1)
                if alt[0] < final[0]:
                    final, theta = alt, th_alt
                else:
                    self.lam = lam_keep
        self.theta = theta
        self.cost = final[0]
        self.uh = final[2][None, :].copy()
        self.vh = final[3][None, :].copy()


class _Run:
    pass


_S = _Run()


def _adopt(chain):
    """Take the state estimate, model and gains of a chain."""
    _S.model = chain.model
    _S.theta = chain.theta.copy()
    _S.uh, _S.vh = chain.uh, chain.vh
    _S.K, _S.L = chain.model.interp_gains(chain.theta)
    _S.coef = chain.model.coeffs(chain.theta)


def _reset():
    _S.ys = []
    _S.us = []
    _S.chains = [_Chain(MODELS[1]), _Chain(MODELS[0])]
    _S.model = MODELS[1]
    _S.theta = TH_C.copy()
    _S.uh = None
    _S.vh = None
    _S.K = _S.L = None
    _S.coef = None
    _S.last_fit = -10 ** 9


def controller(obs: dict) -> float:
    k = int(obs["step"])
    if k == 0 or not hasattr(_S, "ys"):
        _reset()
    y = float(obs["y"])
    if k > 0:
        _S.us.append(float(obs["last_action"]))
    _S.ys.append(y)
    if len(_S.ys) != k + 1 or len(_S.us) != k:                     # lost synchronisation: restart the record
        _reset()
        return 0.0

    do_fit = PARAMS["k_first"] <= k <= PARAMS["k_end"] and (k - _S.last_fit) >= max(1, k // int(PARAMS["upd_div"]))
    if do_fit:
        ya = np.asarray(_S.ys)
        adapt = k >= PARAMS["k_theta"]
        for chain in _S.chains:
            chain.fit(k, ya, _S.us, adapt)
        best = min(_S.chains, key=lambda ch: ch.cost)
        if len(_S.chains) > 1 and k >= PARAMS["k_commit"]:
            worst = max(ch.cost for ch in _S.chains)
            if 2.0 * best.cost < worst or k >= 2 * PARAMS["k_commit"]:     # the grids are told apart: keep one
                _S.chains = [best]
        _adopt(best)
        _S.last_fit = k
    elif _S.uh is not None:
        uh, vh = _S.model.step(_S.uh, _S.vh, _S.us[-1], _S.coef)
        innov = y - uh[0, -1]
        nx = _S.model.nx
        uh[0] += _S.L[:nx] * innov
        vh[0] += _S.L[nx:] * innov
        _S.uh, _S.vh = uh, vh
    if _S.uh is None:
        return 0.0
    nx = _S.model.nx
    U = -float(_S.K[:nx] @ _S.uh[0] + _S.K[nx:] @ _S.vh[0])
    if not math.isfinite(U):
        return 0.0
    lim = min(float(PARAMS["u_lim"]), float(obs.get("u_max", 10.0)))
    return float(min(max(U, -lim), lim))


for _m in MODELS:
    _m.table()
