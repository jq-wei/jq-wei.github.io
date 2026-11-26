"""Multiple-model adaptive LQG boundary controller for u_t = u_xx + lambda(x) u with hidden lambda(x).

Control law
-----------
The hidden coefficient lambda(x) = a + b cos(pi x) + c cos(2 pi x) is covered by a grid of hypotheses
(a, b, c) (A_GRID x B_GRID x C_GRID, 585 models).  Each hypothesis is a model of the plant in modal form:

  * its N_MODES slowest eigen-modes, computed from the plant's own semi-discrete operator, with the exact
    zero-order-hold dynamics per mode (state z, ||u||^2 = |z|^2, flux y = cz . z);
  * the remaining fast modes.  They are stable, start at zero and are driven by the known input only, so
    they are simulated open loop.  They hardly depend on b and c: one simulation per value of a (constant
    coefficient, analytic discrete sines) is shared, scaled so that the static gain is the hypothesis's
    own.  They cannot be dropped or made static: they are what cancels the instantaneous response of the
    slow modes, so that the flux at x = 0 answers an input at x = 1 only after the diffusion delay.

Each hypothesis carries

  * a Kalman filter on its slow modes, driven by the measured flux and the applied input.  The gain
    sequence starts from the prior of the initial state (unit sphere on eight sine modes) and does not
    depend on the data, so it is computed once at import for T_GAIN steps and held afterwards;
  * an LQR gain K for the cost  int ||u||^2 + r U^2 dt  (Riccati equation on the N_LQR slowest modes).

Every step all filters are updated, the hypotheses are scored by their discounted, normalised squared
innovations S_i, and the input is the score-weighted blend of the hypotheses' own LQR feedbacks:

    U = - sum_i w_i K_i z_i,     w_i ~ exp(-(S_i - S_min) / temp).

Early on, while the flux cannot yet tell the hypotheses (or the slow modes) apart, the weights are spread
and the blend is cautious; once the data single out a coefficient it becomes that model's LQG law.

Gains (PARAMS)
--------------
  r_meas     std of the flux noise assumed by the filters.  Above the sensor's 1e-3 because it also absorbs
             the mismatch between the true coefficient and the nearest hypothesis.
  r_early    extra assumed noise std at t = 0, decaying with time constant tau_early [s]: covers the
  tau_early  difference between the two plant grids (about 1e-2 in the first milliseconds).
  fade       fading-memory rate of the filters [1/s]: the covariance is inflated by exp(2 fade dt) per
             step, which keeps the held gain's observer poles to the left of -fade.
  q_proc     additive process noise (flux-equivalent std per sqrt(s)); 0 = fading memory only.
  forget     per-step forgetting factor of the innovation score (memory 1 / (1 - forget) steps).
  temp       temperature of the blend (0: hard arg-min).  Larger = more averaging over hypotheses.
  r_lqr      input weight of the LQR design (the cost's own weight is 0.01).
  t_start    number of initial steps with zero input (the filters see a few samples first).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh_tridiagonal, solve_banded, solve_continuous_are

from hl_control.envs.pde_reaction_diffusion import reaction_coefficient

PARAMS = {
    "r_meas": 3.0e-3,
    "r_early": 1.0e-2,
    "tau_early": 0.003,
    "fade": 40.0,
    "q_proc": 0.0,
    "forget": 0.998,
    "temp": 2.0,
    "r_lqr": 0.01,
    "t_start": 4,
}

DT = 0.00025
N_GRID = 320          # grid on which the hypotheses' modes are computed
N_MODES = 16          # modes kept per hypothesis
N_LQR = 10            # modes used in the Riccati equation of the feedback gain
T_GAIN = 400          # steps of pre-computed (time-varying) Kalman gain; held afterwards
A_GRID = np.arange(20.0, 32.01, 1.0)
B_GRID = np.arange(-8.0, 8.01, 4.0)
C_GRID = np.arange(-8.0, 8.01, 2.0)


def _phi1(s):
    out = np.ones_like(s)
    big = np.abs(s) > 1e-8
    out[big] = np.expm1(s[big]) / s[big]
    return out


def _modal_model(a, b, c, n=N_GRID, n_modes=N_MODES):
    """Slowest modes of the semi-discrete plant for one coefficient, in L2-normalised modal coordinates
    (||u||^2 = |z|^2), and the static flux gain of all the faster modes."""
    h = 1.0 / n
    x = np.linspace(0.0, 1.0, n + 1)[1:-1]
    d = -2.0 / h ** 2 + reaction_coefficient(x, a, b, c)
    e = np.full(n - 2, 1.0 / h ** 2)
    w, V = eigh_tridiagonal(d, e, select="i", select_range=(n - 1 - n_modes, n - 2))
    w, V = w[::-1], V[:, ::-1]
    sq = np.sqrt(h)
    beta = V[-1, :] / h ** 2 * sq                    # input vector
    cz = (4.0 * V[0, :] - V[1, :]) / (2.0 * h) / sq  # flux row
    ab = np.zeros((3, n - 1))
    ab[0, 1:], ab[1, :], ab[2, :-1] = e, d, e
    rhs = np.zeros(n - 1)
    rhs[-1] = -1.0 / h ** 2
    g = solve_banded((1, 1), ab, rhs)                # static profile for U = 1
    dc = (4.0 * g[0] - g[1]) / (2.0 * h)
    feed = dc - float(np.sum(cz * (-beta / w)))      # static flux gain of the truncated (fast) modes
    S = np.sqrt(2.0) * np.sin(np.pi * np.outer(x, np.arange(1, 9)))
    W = sq * (V.T @ S)                               # modal coordinates of the eight initial sine modes
    return w, beta, cz, feed, W


def _tail_model(a, n=N_GRID, n_modes=N_MODES):
    """Fast modes (index > n_modes) of the plant with the constant coefficient a: discrete sines, analytic."""
    h = 1.0 / n
    k = np.arange(n_modes + 1, n)
    w = a - 4.0 / h ** 2 * np.sin(0.5 * k * np.pi * h) ** 2
    beta = np.sqrt(2.0) * np.sin(k * np.pi * (1.0 - h)) / h
    cz = np.sqrt(2.0) * (4.0 * np.sin(k * np.pi * h) - np.sin(2.0 * k * np.pi * h)) / (2.0 * h)
    return w, beta, cz


class _Bank:
    def __init__(self, params):
        self.p = dict(params)
        abc = [(a, b, c) for a in A_GRID for b in B_GRID for c in C_GRID]
        self.abc = np.array(abc)
        M, N = len(abc), N_MODES
        self.M, self.N = M, N
        lam = np.empty((M, N)); beta = np.empty((M, N)); cz = np.empty((M, N)); feed = np.empty(M)
        P0 = np.empty((M, N, N)); K = np.zeros((M, N))
        Q = np.eye(N_LQR); R = np.array([[self.p["r_lqr"]]])
        for i, (a, b, c) in enumerate(abc):
            w, be, cc, fd, W = _modal_model(a, b, c)
            lam[i], beta[i], cz[i], feed[i] = w, be, cc, fd
            P0[i] = (W @ W.T) / 8.0
            X = solve_continuous_are(np.diag(w[:N_LQR]), be[:N_LQR, None], Q, R)
            K[i, :N_LQR] = (be[:N_LQR] @ X) / R[0, 0]
        self.E = np.exp(lam * DT)
        self.psi = DT * _phi1(lam * DT) * beta
        self.cz, self.K = cz, K
        # fast modes: one shared simulation per value of a (constant coefficient), scaled per hypothesis so
        # that the static gain is the hypothesis's own
        tails = [_tail_model(a) for a in A_GRID]
        wt = np.array([t[0] for t in tails]); bt = np.array([t[1] for t in tails]); ct = np.array([t[2] for t in tails])
        self.Et = np.exp(wt * DT)
        self.psit = DT * _phi1(wt * DT) * bt
        self.czt = ct
        feed_a = np.sum(ct * (-bt / wt), axis=1)
        self.ia = np.array([int(np.argmin(np.abs(A_GRID - a))) for a, _, _ in abc])
        self.kappa = feed / feed_a[self.ia]
        self._gains(P0)
        self.reset()

    def _gains(self, P0):
        """Kalman gain sequence of every hypothesis (data-independent), T_GAIN steps; held afterwards."""
        M, N = self.M, self.N
        p = self.p
        t_axis = np.arange(T_GAIN) * DT
        r_t = p["r_meas"] ** 2 + p["r_early"] ** 2 * np.exp(-2.0 * t_axis / p["tau_early"])
        alpha = np.exp(2.0 * p["fade"] * DT)                       # fading memory: observer poles left of -fade
        EE = alpha * self.E[:, :, None] * self.E[:, None, :]
        q = (p["q_proc"] / self.cz) ** 2 * DT                      # additive process noise, uniform in flux units
        idx = np.arange(N)
        P = P0 + 1e-8 * np.eye(N)[None]
        self.L = np.empty((T_GAIN, M, N))
        self.s = np.empty((T_GAIN, M))
        c = self.cz
        for t in range(T_GAIN):
            if t > 0:
                P *= EE
                P[:, idx, idx] += q
            Pc = np.matmul(P, c[:, :, None])[:, :, 0]
            s = np.einsum("mi,mi->m", c, Pc) + r_t[t]
            L = Pc / s[:, None]
            P -= L[:, :, None] * Pc[:, None, :]
            self.L[t], self.s[t] = L, s

    def reset(self):
        self.z = np.zeros((self.M, self.N))
        self.zt = np.zeros(self.Et.shape)
        self.S = np.zeros(self.M)
        self.t = 0

    def step(self, y, u_prev):
        t = self.t
        if t > 0:
            z = self.E * self.z
            z += self.psi * u_prev
            self.zt *= self.Et
            self.zt += self.psit * u_prev
            f = np.einsum("ai,ai->a", self.czt, self.zt)
            e = y - np.einsum("mi,mi->m", self.cz, z) - self.kappa * f[self.ia]
        else:
            z = self.z
            e = y - np.einsum("mi,mi->m", self.cz, z)
        k = t if t < T_GAIN else T_GAIN - 1
        z += self.L[k] * e[:, None]
        self.z = z
        self.S *= self.p["forget"]
        self.S += e * e / self.s[k]
        self.t = t + 1
        if t < self.p["t_start"]:
            return 0.0
        temp = self.p["temp"]
        if temp <= 0.0:
            i = int(np.argmin(self.S))
            return -float(self.K[i] @ z[i])
        wgt = np.exp(-(self.S - self.S.min()) / temp)
        u_all = -np.einsum("mi,mi->m", self.K, z)
        return float(wgt @ u_all / wgt.sum())


_BANK = _Bank(PARAMS)


def controller(obs: dict) -> float:
    if int(obs["step"]) == 0:
        _BANK.reset()
    u = _BANK.step(float(obs["y"]), float(obs["last_action"]))
    u_max = float(obs.get("u_max", 300.0))
    if not np.isfinite(u):
        u = 0.0
    return float(min(max(u, -u_max), u_max))
