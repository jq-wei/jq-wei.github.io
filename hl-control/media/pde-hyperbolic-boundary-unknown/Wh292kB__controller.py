"""Adaptive output-feedback LQG for the 2 x 2 hyperbolic plant with hidden couplings.

Control law (certainty equivalence):  U = -K(theta_hat) x_hat,  clipped to +-u_max.

* Model.  The plant's own scheme (exact shift + Strang reaction + dissipation) is re-implemented in vectorised
  form for both graded grids (100 and 200 intervals), so that many copies (different couplings
  theta = (s1, s2, g), several state vectors each) advance in one call.  The two grids answer differently to a
  held input, so both are carried until the data reject one.
* Identification, coarse phase.  The initial state is a combination of six known sine modes, so for a given
  theta the output is linear in the six amplitudes:  y_k = phi_k(theta)' a + y_k^U(theta).  A bank of models
  over the coupling family (9 x 9 x 5, on each grid) carries the six mode responses and the forced response;
  per model a ridge least-squares fit of the amplitudes gives the residual J(theta), to which the extra
  residual needed to put the amplitudes on the unit sphere (they are drawn there) is added, and the evidence
  exp(-J / (2 SIGMA2_EFF)) det(I + G / RIDGE)^(-1/2).  The control is the evidence-weighted average of the
  models' certainty-equivalent LQR inputs: the state is not observable during the first second whatever the
  model, so no single model is trusted early.
* Identification, fine phase.  Once the weights have collapsed on one model, theta is refined by
  Levenberg-Marquardt on the same sphere-constrained residual, replayed on the stored data every REFINE_EVERY
  steps; the state estimate is that model's fitted trajectory.  If the residual stays above the noise level
  (FIT_OK, FIT_FLOOR) the search restarts from the best coarse model.
* Tracking (afterwards).  A steady-state Kalman filter for the identified model (process noise Q_PROC per
  node and step, measurement noise R_MEAS) absorbs the residual mismatch.
* Gains.  K is the discrete LQR gain for the benchmark's own cost (state weight = L2 quadrature, input
  weight R_INPUT) on the 100-interval model.  K and the Kalman gain are tabulated at import on a 3 x 3 x 3
  grid of the family (54 Riccati equations) and interpolated at theta_hat, which reproduces the exact gains
  to about 1 %; no Riccati equation is solved during a rollout.
* Compute guard.  The controller times itself; if the mean time per call over the session exceeds SLOW_MS
  (a machine several times slower than the one this was tuned on), later rollouts use a thinner coarse bank.

Knobs: RIDGE (prior on the mode amplitudes, noise^2 / amplitude variance), SIGMA2_EFF (residual variance in
the model weights), K_ID_MIN / K_ID_MAX / NEFF_DONE (end of the coarse phase), GRID_LOGODDS (evidence gap that
rejects a grid), REFINE_EVERY / REFINE_ITERS / FD_STEP (fine identification), FIT_OK / FIT_FLOOR (when a fit is
trusted), K_FINE / K_FINE_MAX (hand-over to the Kalman filter), Q_PROC, R_MEAS (Kalman filter), R_INPUT (LQR),
SLOW_MS (compute guard).  There is no hand-tuned feedback gain: K and the observer gain follow from the cost
weights and the noise levels above.
"""

from __future__ import annotations

import math
import time

import numpy as np
import scipy.linalg as sla
from scipy import ndimage

from hl_control.envs.pde_hyperbolic import reaction_step

# ----- knobs -----
RIDGE = 6.0e-6               # noise variance / prior amplitude variance (1e-6 / (1/6))
SIGMA2_EFF = 4.0e-6          # residual variance used for the model weights (noise + coupling-grid mismatch)
K_ID_MIN, K_ID_MAX = 100, 140   # the coarse bank runs until the posterior over theta has collapsed (within these steps)
NEFF_DONE = 1.5              # ... i.e. until the effective number of models is below this
GRID_LOGODDS = 30.0          # log evidence gap at which the other grid is dropped
REFINE_EVERY = 10            # steps between Gauss-Newton refinements of theta after the coarse phase
REFINE_ITERS = 4             # Gauss-Newton iterations per refinement (each replays the stored data once)
FD_STEP = 2.0e-3             # finite-difference step in theta for the Gauss-Newton Jacobian
FIT_OK = 3.0                 # the fit is trusted when its residual is below FIT_OK * (noise sum) + FIT_FLOOR
FIT_FLOOR = 2.0e-5
K_FINE = 40                  # steps of fine identification before the Kalman filter takes over (if the fit is trusted)
K_FINE_MAX = 250             # ... and the latest step at which it takes over anyway
SLOW_MS = 8.0                # compute guard: if the mean time per call so far exceeds this (ms), later rollouts
                             # use a thinner coarse bank (5 x 5 x 5 instead of 9 x 9 x 5)
Q_PROC = 1.0e-6              # process noise variance per node and step (Kalman filter)
R_MEAS = 1.0e-6              # measurement noise variance
R_INPUT = 0.1                # input weight of the cost
S_LO, S_HI, G_LO, G_HI = 1.5, 2.6, -0.5, 0.5
BOUNDS_LO = np.array([1.4, 1.4, -0.6])
BOUNDS_HI = np.array([2.7, 2.7, 0.6])

KD = 0.1 / 16.0
_DISS = np.array([-KD, 4.0 * KD, 1.0 - 6.0 * KD, 4.0 * KD, -KD])
UIN = np.zeros((7, 1)); UIN[6, 0] = 1.0
EYE6 = np.eye(6)
NC = 101                     # nodes of the 100-interval grid (gain design, Kalman filter)


class _Grid:
    """The plant's scheme on n intervals (n = 100: one substep per control step, n = 200: two)."""

    def __init__(self, n):
        self.n = n
        self.np = n + 1
        self.sub = n // 100
        self.stride = n // 100
        self.x = np.linspace(0.0, 1.0, n + 1)
        self.modes = math.sqrt(2.0) * np.sin(np.pi * np.outer(np.arange(1, 4), self.x))      # (3, n + 1)

    def coeffs(self, thetas):
        thetas = np.atleast_2d(np.asarray(thetas, dtype=float))
        s1 = thetas[:, 0:1] * np.ones((1, self.np))
        s2 = thetas[:, 1:2] * (1.0 + thetas[:, 2:3] * (2.0 * self.x[None, :] - 1.0))
        c, a12, a21 = reaction_step(s1, s2, 0.5 / self.n)
        return c[:, None, :], a12[:, None, :], a21[:, None, :]

    def step(self, u, v, cf, uin):
        """One control step for arrays (..., n + 1); uin broadcasts onto the two inflow nodes of v."""
        c, a12, a21 = cf
        for _ in range(self.sub):
            u1 = c * u + a12 * v
            v1 = a21 * u + c * v
            u2 = np.empty_like(u1)
            v2 = np.empty_like(v1)
            u2[..., 1:] = u1[..., :-1]
            v2[..., :-2] = v1[..., 2:]
            v2[..., -2:] = uin
            u2[..., 0] = v2[..., 0]
            np.multiply(c, u2, out=u1)
            u1 += a12 * v2                       # u1, v1: the fields after the second half reaction
            np.multiply(c, v2, out=v1)
            v1 += a21 * u2
            # fourth-difference dissipation on the interior nodes (the two nodes next to each boundary are kept)
            u = ndimage.correlate1d(u1, _DISS, axis=-1, mode="nearest")
            v = ndimage.correlate1d(v1, _DISS, axis=-1, mode="nearest")
            u[..., :2] = u1[..., :2]
            u[..., -2:] = u1[..., -2:]
            v[..., :2] = v1[..., :2]
            v[..., -2:] = v1[..., -2:]
        return u, v


GRIDS = (_Grid(100), _Grid(200))
G100 = GRIDS[0]


def _ab(theta):
    cf = G100.coeffs(theta)
    u = np.zeros((1, 2 * NC + 1, NC))
    v = np.zeros((1, 2 * NC + 1, NC))
    u[0, :NC, :] = np.eye(NC)
    v[0, NC:2 * NC, :] = np.eye(NC)
    uin = np.zeros((2 * NC + 1, 1)); uin[-1, 0] = 1.0
    u, v = G100.step(u, v, cf, uin)
    m = np.concatenate([u[0], v[0]], axis=1).T
    return m[:, :2 * NC].copy(), m[:, 2 * NC].copy()


_W = np.full(NC, 0.01); _W[0] = _W[-1] = 0.005
_QC = np.diag(np.concatenate([_W, _W])) * 0.01
_RC = np.array([[R_INPUT * 0.01]])
_C = np.zeros((1, 2 * NC)); _C[0, NC - 1] = 1.0


def _lqr(theta):
    a, b = _ab(theta)
    b = b[:, None]
    p = sla.solve_discrete_are(a, b, _QC, _RC)
    return np.linalg.solve(_RC + b.T @ p @ b, b.T @ p @ a)[0]


def _kalman(theta):
    a, _ = _ab(theta)
    p = sla.solve_discrete_are(a.T, _C.T, Q_PROC * np.eye(2 * NC), np.array([[R_MEAS]]))
    return (p @ _C.T / (_C @ p @ _C.T + R_MEAS))[:, 0]


# ----- gain tables over the coupling family (trilinear interpolation; within about 1 % of the exact gains) -----
_KS = np.linspace(S_LO, S_HI, 3)
_KG = np.linspace(G_LO, G_HI, 3)
_KTAB = np.array([[[_lqr((a, b, g)) for g in _KG] for b in _KS] for a in _KS])       # LQR gains
_LTAB = np.array([[[_kalman((a, b, g)) for g in _KG] for b in _KS] for a in _KS])    # Kalman gains


def _interp(table, theta):
    out = 0.0
    idx = []
    for value, axis in ((theta[0], _KS), (theta[1], _KS), (theta[2], _KG)):
        t = (min(max(value, axis[0]), axis[-1]) - axis[0]) / (axis[1] - axis[0])
        i = min(int(t), len(axis) - 2)
        idx.append((i, t - i))
    for di in (0, 1):
        for dj in (0, 1):
            for dk in (0, 1):
                w = ((idx[0][1] if di else 1 - idx[0][1]) * (idx[1][1] if dj else 1 - idx[1][1])
                     * (idx[2][1] if dk else 1 - idx[2][1]))
                if w:
                    out = out + w * table[idx[0][0] + di, idx[1][0] + dj, idx[2][0] + dk]
    return out


def _gain(theta):
    return _interp(_KTAB, theta)


class _Bank:
    """Models theta_m on one grid, each with the six mode responses and the forced response, and their
    least-squares sums."""

    def __init__(self, grid, thetas):
        self.grid = grid
        self.th = np.atleast_2d(np.asarray(thetas, dtype=float))
        self.cf = grid.coeffs(self.th)
        m = len(self.th)
        self.u = np.zeros((m, 7, grid.np))
        self.v = np.zeros((m, 7, grid.np))
        self.u[:, 0:3, :] = grid.modes[None]
        self.v[:, 3:6, :] = grid.modes[None]
        self.G = np.zeros((m, 6, 6))
        self.b = np.zeros((m, 6))
        self.c = np.zeros(m)
        self.hist = None             # (regressors, forced-response residuals) of a recorded replay

    def advance(self, u_applied):
        self.u, self.v = self.grid.step(self.u, self.v, self.cf, UIN * u_applied)

    def update(self, y):
        phi = self.u[:, :6, -1]
        r = y - self.u[:, 6, -1]
        self.G += phi[:, :, None] * phi[:, None, :]
        self.b += phi * r[:, None]
        self.c += r * r

    def replay(self, ys, us, record=False):
        m, n = len(self.th), len(ys)
        phis, rs = np.zeros((m, n, 6)), np.zeros((m, n))
        for k in range(n):
            if k > 0:
                self.advance(us[k - 1])
            self.update(ys[k])
            if record:
                phis[:, k] = self.u[:, :6, -1]
                rs[:, k] = ys[k] - self.u[:, 6, -1]
        if record:
            self.hist = (phis, rs)

    def residuals(self, a, excess):
        """Residual vectors of a recorded replay, with the unit-sphere term appended: (M, n + 1)."""
        phis, rs = self.hist
        res = rs - np.einsum("mki,mi->mk", phis, a)
        extra = np.sqrt(np.maximum(excess, 0.0)) * np.sign(np.linalg.norm(a, axis=1) - 1.0)
        return np.concatenate([res, extra[:, None]], axis=1)

    def solve(self):
        a = np.linalg.solve(self.G + RIDGE * EYE6, self.b[..., None])[..., 0]
        cost = self.c - np.einsum("mi,mi->m", self.b, a)
        return a, cost

    def sphere_excess(self, a):
        """The amplitudes lie on the unit sphere: extra residual min_{|a'| = 1} (a' - a)' (G + RIDGE) (a' - a) needed
        to put each model's fit there (secular equation in the eigenbasis of G, solved by bisection)."""
        g, vec = np.linalg.eigh(self.G + RIDGE * EYE6)
        alpha = np.einsum("mji,mj->mi", vec, a)
        ga = g * alpha
        inside = np.sum(alpha * alpha, axis=1) < 1.0
        lo = np.where(inside, -g[:, 0] * (1.0 - 1e-12), 0.0)
        hi = np.where(inside, 0.0, np.sqrt(np.sum(ga * ga, axis=1)))
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            pos = np.sum((ga / (g + mid[:, None])) ** 2, axis=1) > 1.0
            lo = np.where(pos, mid, lo)
            hi = np.where(pos, hi, mid)
        mu = 0.5 * (lo + hi)[:, None]
        return np.sum(g * (alpha * mu / (g + mu)) ** 2, axis=1)

    def log_evidence(self, a, cost):
        total = cost + self.sphere_excess(a)
        return -total / (2.0 * SIGMA2_EFF) - 0.5 * np.linalg.slogdet(self.G / RIDGE + EYE6)[1]

    def inputs(self, gains, a):
        """Certainty-equivalent LQR input of every model (gains: (M, 2 NC) on the 100-interval nodes)."""
        st = self.grid.stride
        gb = (np.einsum("mx,mix->mi", gains[:, :NC], self.u[:, :, ::st])
              + np.einsum("mx,mix->mi", gains[:, NC:], self.v[:, :, ::st]))
        return -(np.einsum("mi,mi->m", a, gb[:, :6]) + gb[:, 6])

    def state(self, m, a):
        """Fitted state of model m on its own grid, (2, n + 1)."""
        return np.stack([a @ self.u[m, :6] + self.u[m, 6], a @ self.v[m, :6] + self.v[m, 6]])


_COARSE_THETAS = np.array([(a, b, g) for a in np.linspace(S_LO, S_HI, 9) for b in np.linspace(S_LO, S_HI, 9)
                           for g in np.linspace(G_LO, G_HI, 5)])
_COARSE_GAINS = np.array([_gain(th) for th in _COARSE_THETAS])          # (M, 2 NC)
_THIN_THETAS = np.array([(a, b, g) for a in np.linspace(S_LO, S_HI, 5) for b in np.linspace(S_LO, S_HI, 5)
                         for g in np.linspace(G_LO, G_HI, 5)])
_THIN_GAINS = np.array([_gain(th) for th in _THIN_THETAS])
_ORACLE = None               # diagnostics only: (grid index, theta) instead of the coarse banks
_FORCE_THIN = False          # diagnostics only: always use the thin bank
_CLOCK = {"calls": 0, "seconds": 0.0}    # controller time over the whole session (all rollouts)

_S: dict = {}


def _reset():
    _S.clear()
    slow = _FORCE_THIN or (_CLOCK["calls"] >= 800 and 1.0e3 * _CLOCK["seconds"] / _CLOCK["calls"] > SLOW_MS)
    thetas, gains = (_THIN_THETAS, _THIN_GAINS) if slow else (_COARSE_THETAS, _COARSE_GAINS)
    if _ORACLE is None:
        banks = [_Bank(g, thetas) for g in GRIDS]
    else:
        banks, gains = [_Bank(GRIDS[_ORACLE[0]], _ORACLE[1])], np.array([_gain(_ORACLE[1])])
    _S.update(ys=[], us=[], banks=banks, gains=gains, coarse=thetas, phase="coarse", k_id=None,
              theta=np.array([2.05, 2.05, 0.0]), grid=None, bank=None, x=None, K=None, L=None, cf=None, fit=0.0)


_FD = np.vstack([np.zeros(3), np.eye(3)])
_MID = 0.5 * (BOUNDS_LO + BOUNDS_HI)


def _refine(grid, theta, iters):
    """Levenberg-Marquardt on theta for the sphere-constrained least-squares residual over the stored data (the
    amplitudes are solved inside; Jacobian by forward differences on a bank of four models).  Returns the bank
    of the best iterate (model 0 = theta_hat) and its residual sum."""
    ys, us = _S["ys"], _S["us"]
    best = None
    damping = 1.0e-2
    hess = grad = None
    for it in range(iters + 1):
        sign = np.where(theta > _MID, -1.0, 1.0)
        bank = _Bank(grid, theta[None, :] + _FD * (sign * FD_STEP)[None, :])
        bank.replay(ys, us, record=True)
        a, cost = bank.solve()
        res = bank.residuals(a, bank.sphere_excess(a))
        total = float(res[0] @ res[0])
        if best is None or total < best[0]:
            best = (total, theta.copy(), bank)
            jac = (res[1:] - res[0]) / (sign * FD_STEP)[:, None]
            hess, grad = jac @ jac.T, jac @ res[0]
            damping = max(damping / 3.0, 1.0e-6)
        else:
            damping *= 10.0
        if it == iters:
            break
        step = -np.linalg.solve(hess + damping * np.diag(np.diag(hess)) + 1e-12 * np.eye(3), grad)
        theta = np.clip(best[1] + step, BOUNDS_LO, BOUNDS_HI)
    return best[2], best[0]


def _coarse_best(grid):
    """Fallback: the best model of the whole coarse bank on the stored data."""
    bank = _Bank(grid, _S["coarse"])
    bank.replay(_S["ys"], _S["us"])
    a, cost = bank.solve()
    return bank.th[int(np.argmin(cost + bank.sphere_excess(a)))].copy()


def _fit_ok(total, k):
    return total <= FIT_OK * (k + 1) * R_MEAS + FIT_FLOOR


def controller(obs: dict) -> float:
    """U for the next control step."""
    u_max = float(obs.get("u_max", 10.0))
    t0 = time.perf_counter()
    try:
        u_cmd = _control(obs)
    except Exception:                      # a numerical failure must not end the rollout: no actuation this step
        u_cmd = 0.0
    _CLOCK["calls"] += 1
    _CLOCK["seconds"] += time.perf_counter() - t0
    if not math.isfinite(u_cmd):
        return 0.0
    return float(min(max(u_cmd, -u_max), u_max))


def _control(obs: dict) -> float:
    k = int(obs["step"])
    y = float(obs["y"])
    if k == 0 or not _S:
        _reset()
    s = _S
    if k > 0:
        s["us"].append(float(obs["last_action"]))
    s["ys"].append(y)

    if s["phase"] == "coarse":
        # --- all models of the family on the grids still in play, weighted by evidence ---
        banks = s["banks"]
        fits, lws = [], []
        for bank in banks:
            if k > 0:
                bank.advance(s["us"][-1])
            bank.update(y)
        for bank in banks:
            a, cost = bank.solve()
            fits.append(a)
            lw = bank.log_evidence(a, cost)
            lws.append(np.where(np.isfinite(lw), lw, -1.0e300))
        top = max(float(lw.max()) for lw in lws)
        ws = [np.exp(lw - top) for lw in lws]
        total = sum(float(w.sum()) for w in ws)
        u_cmd = sum(float(w @ bank.inputs(s["gains"], a)) for w, bank, a in zip(ws, banks, fits)) / total
        if len(banks) > 1:
            mass = [float(w.sum()) / total for w in ws]
            best = int(np.argmax(mass))
            if mass[1 - best] < math.exp(-GRID_LOGODDS) or k >= K_ID_MAX:
                banks = s["banks"] = [banks[best]]
                ws, fits = [ws[best]], [fits[best]]
        if len(banks) == 1:
            bank, w, a = banks[0], ws[0] / ws[0].sum(), fits[0]
            neff = 1.0 / float(w @ w)
            m = int(np.argmax(w))
            s["theta"] = w @ bank.th
            s["x"] = bank.state(m, a[m])
            s["grid"] = bank.grid
            if _ORACLE is None and ((k >= K_ID_MIN and neff < NEFF_DONE) or k >= K_ID_MAX):
                s["bank"], s["fit"] = _refine(bank.grid, bank.th[m].copy(), REFINE_ITERS + 2)
                s["phase"] = "fine"
                s["k_id"] = k
                s["banks"] = None
    elif s["phase"] == "fine":
        # --- one model, theta refined by Gauss-Newton on all stored data; state = its fitted trajectory ---
        bank = s["bank"]
        bank.advance(s["us"][-1])
        bank.update(y)
        age = k - s["k_id"]
        if age % REFINE_EVERY == 0:
            bank, s["fit"] = _refine(bank.grid, bank.th[0].copy(), REFINE_ITERS)
            if not _fit_ok(s["fit"], k) and age % (2 * REFINE_EVERY) == 0:
                # stuck away from the data: restart from the best coarse model
                alt, fit = _refine(bank.grid, _coarse_best(bank.grid), REFINE_ITERS + 2)
                if fit < s["fit"]:
                    bank, s["fit"] = alt, fit
            s["bank"] = bank
        a, _ = bank.solve()
        theta = bank.th[0].copy()
        s["theta"] = theta
        x = bank.state(0, a[0])
        s["x"] = x
        st = bank.grid.stride
        gain = _gain(theta)
        if age % REFINE_EVERY == 0 and ((age >= K_FINE and _fit_ok(s["fit"], k)) or k >= K_FINE_MAX):
            lc = _interp(_LTAB, theta)
            xc = G100.x
            s["L"] = np.stack([np.interp(bank.grid.x, xc, lc[:NC]), np.interp(bank.grid.x, xc, lc[NC:])])
            s["K"] = gain
            s["cf"] = bank.grid.coeffs(theta)
            s["phase"] = "kf"
            s["bank"] = None
        u_cmd = -float(gain[:NC] @ x[0, ::st] + gain[NC:] @ x[1, ::st])
    else:
        # --- steady-state Kalman filter + LQR for the identified model ---
        grid = s["grid"]
        xu, xv = grid.step(s["x"][None, None, 0], s["x"][None, None, 1], s["cf"], s["us"][-1])
        x = np.stack([xu[0, 0], xv[0, 0]])
        x = x + s["L"] * (y - x[0, -1])
        s["x"] = x
        st = grid.stride
        u_cmd = -float(s["K"][:NC] @ x[0, ::st] + s["K"][NC:] @ x[1, ::st])
    return u_cmd
