"""Commissioning controller for the ill-conditioned 2 x 2 process: identify the unit, then invert its model.

Coordinates.  The output rotation U is known, so everything is done on zeta = U^T y, where the plant is two
decoupled first-order lags with a common dead time:
    zeta_i[k+1] = a_i zeta_i[k] + c_i . u[k - theta],    i = 1 (high gain), 2 (low gain).
c_1 = b_1 m_1 is a free 2-vector (it carries the wiring and the input angle); the low-gain channel acts along
the orthogonal input direction, c_2 = gam * perp(m_1).  The model is (theta, a_1, c_1, a_2, gam).

Identification.  The model is the output-error least-squares fit to all data recorded so far (maximum
likelihood for white measurement noise): a grid over theta and the two poles, linear in the gains.  It is
refitted every few steps of the commissioning window and every 20 min of the evaluation window until just
before the disturbance can arrive.  The commissioning experiment that feeds it:
  0a  geometric ramp on actuator a (`probe_delta0`, `probe_growth`) until the high-gain output has moved by
      `probe_thresh`; wait out the dead time;
  0b  pulse on actuator b sized for a high-gain response of `pulse_dz`; wait.  First theta, b_1 and m_1;
  1p  short pulse along the estimated low-gain direction, sized from b_1 so that a direction error of 5
      degrees moves the high-gain output by 0.1 at most, while that output is held: m_1 to a fraction of a
      degree before any large move in the low-gain direction;
  1   the high-gain output is held at +-`r1` by the control law below; the low-gain direction gets an
      open-loop step `a2` (first quarter) and then model-inverse steps between +-`r2`;
  ret both outputs are sent back to zero `t_end` steps before the evaluation window opens.
If a measured composition exceeds 0.8 during commissioning, the probing is abandoned for the return to zero.

Control law (stages 1p onwards and the whole evaluation): internal-model dead-beat.  The model is simulated
on the commands actually applied; the command of step k puts the model output of step k + theta + 1 on
    reference - base - predicted disturbance,
with the high-gain move exact and the low-gain move limited to `w2max` per step and to the actuator range.
`base` is an exponentially weighted mean of the residual (measurement - model - disturbance estimate), with
gain `alpha_c` during commissioning (fast, the model is still rough) and `alpha_e` during evaluation (slow
integral action against model error; small, so that measurement noise hardly reaches the actuators).

Disturbance.  Its shape is stated (0.2 along (0.6, 0.8) through a 40 min lag, arriving 395 to 425 min into the
evaluation), so inside that window a likelihood-ratio test on the high-gain residual decides at every step
whether it has arrived (`det_thr`), with which sign and at which step; its whole future course is then
cancelled by feedforward through the model inverse.
"""

import math

import numpy as np
from scipy.signal import lfilter

PARAMS = {
    "probe_delta0": 0.002,   # first command of the geometric probing ramp (stage 0a)
    "probe_growth": 1.2,     # growth per step of that ramp
    "probe_thresh": 0.03,    # high-gain output deviation that ends the ramp
    "pulse_dz": 0.08,        # intended high-gain response to the actuator-b pulse (stage 0b)
    "r1": 0.3,               # hold level of the high-gain output during stage 1
    "a2": 0.25,              # open-loop step along the low-gain input direction (first quarter of stage 1)
    "r2": 0.25,              # low-gain output levels of the model-inverse steps (rest of stage 1)
    "t_end": 25,             # steps before evaluation at which both outputs are sent back to zero
    "alpha_c": 0.2,          # residual-filter gain during commissioning
    "alpha_e": 0.03,         # residual-filter gain during evaluation (slow integral action)
    "w2max": 2.0,            # limit of the low-gain move per step
    "det_thr": 10.0,         # log-likelihood ratio that declares the disturbance
}

_ANG = math.radians(51.3)
_U = np.array([[math.cos(_ANG), -math.sin(_ANG)], [math.sin(_ANG), math.cos(_ANG)]])
_DZ = _U.T @ np.array([0.6, 0.8])      # disturbance direction in zeta coordinates
_AD = math.exp(-1.0 / 40.0)            # disturbance lag per step
_DSTEP = 0.2
_ULIM = 2.0
_TAU1 = np.geomspace(60.0, 260.0, 20)
_TAU2 = np.geomspace(7.0, 40.0, 20)
_EVENT_WINDOW = (394, 426)             # possible onset of the disturbance, steps into the evaluation
_PROBE_TIMEOUT = 60                    # stage 0a ends here even without a response (never on this family)
_LEAK = 0.1                            # high-gain response allowed to the direction-probing pulse (stage 1p)
_GUARD = 0.8                           # measured |composition| that aborts the probing during commissioning


def _sse_channel1(x, z, zz):
    """Least squares z ~ x c (x: (n,2)); returns (sse, c)."""
    g = x.T @ x
    g[0, 0] += 1e-12
    g[1, 1] += 1e-12
    h = x.T @ z
    det = g[0, 0] * g[1, 1] - g[0, 1] * g[1, 0]
    if det < 1e-14 * max(g[0, 0] * g[1, 1], 1e-30):
        # one regressor (or collinear ones): fit along the dominant column only
        j = 0 if g[0, 0] >= g[1, 1] else 1
        c = np.zeros(2)
        c[j] = h[j] / g[j, j]
        return zz - h[j] * c[j], c
    c = np.array([(g[1, 1] * h[0] - g[0, 1] * h[1]) / det, (g[0, 0] * h[1] - g[1, 0] * h[0]) / det])
    return zz - h @ c, c


def _fit(zeta, u, k):
    """Output-error fit of the model to zeta[0..k], u[0..k-1]."""
    n = k + 1
    us0 = np.zeros((n, 2))
    us0[1:] = u[:k]
    z1, z2 = zeta[:n, 0], zeta[:n, 1]
    zz1, zz2 = float(z1 @ z1), float(z2 @ z2)

    def filt(tau):
        return lfilter([1.0], [1.0, -math.exp(-1.0 / tau)], us0, axis=0)

    x1 = [filt(t) for t in _TAU1]
    x2 = [filt(t) for t in _TAU2]
    best = None
    for th in range(1, 11):
        m = n - th
        if m < 4:
            continue
        s1 = math.inf
        for i, x in enumerate(x1):
            s, c = _sse_channel1(x[:m], z1[th:], zz1)
            if s < s1:
                s1, i1, c1 = s, i, c
        nrm = float(np.hypot(c1[0], c1[1]))
        if nrm < 1e-12:
            continue
        p1 = np.array([-c1[1], c1[0]]) / nrm
        s2, i2, gam = math.inf, 0, 0.0
        for i, x in enumerate(x2):
            xp = x[:m] @ p1
            den = float(xp @ xp)
            if den < 1e-12:
                s, g_ = zz2, 0.0
            else:
                num = float(xp @ z2[th:])
                g_ = num / den
                s = zz2 - num * g_
            if s < s2:
                s2, i2, gam = s, i, g_
        if best is None or s1 + s2 < best[0]:
            best = (s1 + s2, th, i1, i2)
    if best is None:
        return None
    _, th, i1, i2 = best
    m = n - th

    # refine the two poles on a finer grid around the best grid point
    def refine(taus, i, cost):
        lo, hi = taus[max(i - 1, 0)], taus[min(i + 1, len(taus) - 1)]
        cand = np.geomspace(lo, hi, 13)
        vals = [cost(t) for t in cand]
        j = int(np.argmin([v[0] for v in vals]))
        return cand[j], vals[j]

    def cost1(t):
        return _sse_channel1(filt(t)[:m], z1[th:], zz1)

    tau1, (_, c1) = refine(_TAU1, i1, cost1)
    nrm = float(np.hypot(c1[0], c1[1]))
    m1 = c1 / nrm
    p1 = np.array([-m1[1], m1[0]])

    def cost2(t):
        xp = filt(t)[:m] @ p1
        den = float(xp @ xp)
        if den < 1e-12:
            return zz2, 0.0
        num = float(xp @ z2[th:])
        return zz2 - num * num / den, num / den

    tau2, (_, gam) = refine(_TAU2, i2, cost2)
    return {"theta": int(th), "a1": math.exp(-1.0 / tau1), "a2": math.exp(-1.0 / tau2), "c1": c1, "b1": nrm, "m1": m1, "p1": p1,
            "gam": float(gam), "c2": gam * p1, "tau1": float(tau1), "tau2": float(tau2)}


def _ewma(resid, alpha, width=300):
    """Exponentially weighted mean of the latest residuals (the newest has weight ~alpha)."""
    n = len(resid)
    m = min(n, width)
    w = (1.0 - alpha) ** np.arange(m)
    return (w @ resid[n - m:][::-1]) / float(w.sum())


class _Ctl:
    def __init__(self, obs):
        dt = float(obs.get("dt", 1.0))
        self.nc = int(round(float(obs["time_to_evaluation"]) / dt))
        self.ne = int(round(float(obs.get("evaluation_time", 600.0)) / dt))
        n = self.nc + self.ne + 16
        self.zeta = np.zeros((n, 2))
        self.u = np.zeros((n, 2))
        self.model = None
        self.stage = "0a"
        self.mark = 0
        self.pulse = 0.0
        self.k1 = 0
        self.kret = 0
        self.last_fit = -10 ** 6
        self.sgn1 = 1.0
        self.sgn2 = 1.0
        self.hold1 = None
        self.q_prev = -1
        self.ev = None

    # ---- model ------------------------------------------------------------------------------------------
    def refit(self, k):
        m = _fit(self.zeta, self.u, k)
        if m is not None:
            self.model = m
        self.last_fit = k

    def simulate(self, k):
        """Model outputs (zeta coordinates) at steps 0..k for the commands applied so far."""
        m = self.model
        th = m["theta"]
        us = np.zeros((k + 1, 2))
        if k - th >= 1:
            us[th + 1:] = self.u[:k - th]
        z1 = lfilter([1.0], [1.0, -m["a1"]], us @ m["c1"])
        z2 = lfilter([1.0], [1.0, -m["a2"]], us @ m["c2"])
        return np.column_stack([z1, z2])

    def deadbeat(self, k, zm_k, target, w2_open=None, w1max=1.5):
        """Command that puts the model output on `target` theta+1 steps ahead."""
        m = self.model
        th = m["theta"]
        zp = np.array(zm_k, dtype=float)
        for j in range(k, k + th):
            uj = self.u[j - th] if j - th >= 0 else np.zeros(2)
            zp[0] = m["a1"] * zp[0] + m["c1"] @ uj
            zp[1] = m["a2"] * zp[1] + m["c2"] @ uj
        w1 = float(np.clip((target[0] - m["a1"] * zp[0]) / m["b1"], -w1max, w1max))
        if w2_open is not None:
            w2 = float(w2_open)
        elif abs(m["gam"]) > 1e-3:
            lim = PARAMS["w2max"]
            w2 = float(np.clip((target[1] - m["a2"] * zp[1]) / m["gam"], -lim, lim))
        else:
            w2 = 0.0
        ua, ub = m["m1"] * w1, m["p1"] * w2
        lam = 1.0
        for j in range(2):  # keep the high-gain move exact, shrink the low-gain move into the limits
            if abs(ub[j]) > 1e-12:
                lam = min(lam, max(0.0, (_ULIM - math.copysign(1.0, ub[j]) * ua[j]) / abs(ub[j])))
        return np.clip(ua + lam * ub, -_ULIM, _ULIM)

    # ---- commissioning ----------------------------------------------------------------------------------
    def commission(self, k):
        P = PARAMS
        zero = np.zeros(2)
        # stage 0a: growing command on actuator a until the high-gain output answers, then let the dead time pass
        if self.stage == "0a":
            if (k >= 1 and abs(self.zeta[k, 0]) > P["probe_thresh"]) or k >= _PROBE_TIMEOUT:
                self.stage, self.mark = "0a_wait", k
                return zero
            return np.array([min(P["probe_delta0"] * P["probe_growth"] ** k, 1.0), 0.0])
        if self.stage == "0a_wait":
            if k < self.mark + 11:
                return zero
            self.refit(k)
            # stage 0b: two-step pulse on actuator b, sized with the gain just found for actuator a
            c1a = abs(self.model["c1"][0]) if self.model is not None else 1.0
            self.pulse = min(0.5 * P["pulse_dz"] / max(c1a, 1e-6), 1.0)
            self.stage, self.mark = "0b", k
        if self.stage == "0b":
            if k < self.mark + 2:
                return np.array([0.0, self.pulse])
            self.stage, self.mark = "0b_wait", k
            return zero
        if self.stage == "0b_wait":
            if k < self.mark + 12:
                return zero
            self.refit(k)
            self.stage, self.mark = "1p", k
            # pulse along the estimated low-gain direction, sized so that a direction error of 5 degrees
            # moves the high-gain output by `_LEAK` at most
            self.pulse = min(P["a2"], _LEAK / (math.sin(math.radians(5.0)) * self.model["b1"] * 3.0))
            self.hold1 = None
        # closed-loop stages: direction probe (1p), main probing (1), return to zero
        if k - self.last_fit >= (3 if (self.stage == "1p" or k - self.k1 < 18) else 6):
            self.refit(k)
        zm = self.simulate(k)
        resid = self.zeta[:k + 1] - zm
        base = _ewma(resid, P["alpha_c"])
        if self.stage == "1p":
            if self.hold1 is None:
                self.hold1 = float(zm[k, 0] + base[0])  # present level of the high-gain output: hold it
            if k < self.mark + min(self.model["theta"], 10) + 7:
                w2 = self.pulse if k < self.mark + 3 else 0.0
                return self.deadbeat(k, zm[k], np.array([self.hold1 - base[0], 0.0]), w2_open=w2)
            self.stage, self.k1 = "1", k
            self.sgn1 = 1.0 if self.hold1 >= 0 else -1.0
            self.kret = max(self.nc - int(P["t_end"]), k + 4)
        if np.max(np.abs(_U @ self.zeta[k])) > _GUARD:
            self.kret = min(self.kret, k)  # an output is near the band edge: stop probing, go back to zero
        if k >= self.kret:
            return self.deadbeat(k, zm[k], -base)
        # stage 1 in four quarters: high-gain level +r1, +r1, -r1, -r1; low-gain open-loop step, then -r2, +r2, -r2
        q = int((k - self.k1) * 4 // max(self.kret - self.k1, 1))
        if q != self.q_prev:
            if q == 1:
                self.sgn2 = 1.0 if self.zeta[k, 1] >= 0 else -1.0
            self.q_prev = q
        r1 = P["r1"] * self.sgn1 * (1.0 if q < 2 else -1.0)
        if q == 0:
            return self.deadbeat(k, zm[k], np.array([r1 - base[0], 0.0]), w2_open=P["a2"])
        r2 = P["r2"] * self.sgn2 * (1.0 if q == 2 else -1.0)
        return self.deadbeat(k, zm[k], np.array([r1, r2]) - base)

    # ---- evaluation -------------------------------------------------------------------------------------
    def event_course(self, n):
        """Estimated disturbance (lagged step) at steps 0..n-1; zero while none has been detected."""
        d = np.zeros(n)
        if self.ev is not None:
            k0 = self.ev["k0"]
            i = np.arange(n) - k0
            pos = i > 0
            d[pos] = self.ev["s"] * _DSTEP * (1.0 - _AD ** i[pos])
        return d

    def detect(self, k, resid1, sig2):
        """Likelihood-ratio test for the known disturbance shape; re-decided from the data at every step."""
        te = k - self.nc
        lo, hi = self.nc + _EVENT_WINDOW[0], self.nc + _EVENT_WINDOW[1]
        if te <= _EVENT_WINDOW[0] or te > _EVENT_WINDOW[1] + 80:
            return  # before the window: nothing to detect; long after it: the estimate is frozen
        amp = _DSTEP * _DZ[0]
        b0 = float(np.mean(resid1[self.nc + 330:lo]))  # residual level before the disturbance can arrive
        best = (-math.inf, 0, 1.0)
        for k0 in range(lo, min(hi, k - 1) + 1):
            r = resid1[k0 + 1:k + 1] - b0
            h = amp * (1.0 - _AD ** np.arange(1, k - k0 + 1))
            sh, hh = float(r @ h) / sig2, 0.5 * float(h @ h) / sig2
            for s in (1.0, -1.0):
                if s * sh - hh > best[0]:
                    best = (s * sh - hh, k0, s)
        self.ev = {"k0": best[1], "s": best[2]} if best[0] > PARAMS["det_thr"] else None

    def evaluate(self, k, obs):
        P = PARAMS
        te = k - self.nc
        if self.model is None:
            return np.zeros(2)
        if te == 0 or (te % 20 == 0 and te <= 380):
            self.refit(k)
        zm = self.simulate(k)
        resid = self.zeta[:k + 1] - zm
        sig2 = float(obs.get("noise_std", 0.004)) ** 2
        self.detect(k, resid[:, 0], sig2)
        th = self.model["theta"]
        d = self.event_course(k + th + 2)
        resid = resid - np.outer(d[:k + 1], _DZ)
        base = _ewma(resid, P["alpha_e"])
        rz = _U.T @ np.asarray(obs["r"], dtype=float)
        return self.deadbeat(k, zm[k], rz - base - _DZ * d[k + th + 1])

    def step(self, obs):
        k = int(obs["step"])
        self.zeta[k] = _U.T @ np.asarray(obs["y"], dtype=float)
        if k > 0:
            self.u[k - 1] = np.asarray(obs["last_action"], dtype=float)
        u = self.commission(k) if bool(obs["commissioning"]) else self.evaluate(k, obs)
        u = np.clip(np.nan_to_num(np.asarray(u, dtype=float)), -_ULIM, _ULIM)
        self.u[k] = u
        return u


_STATE = None


def controller(obs: dict) -> np.ndarray:
    global _STATE
    if _STATE is None or int(obs["step"]) == 0:
        _STATE = _Ctl(obs)
    try:
        return _STATE.step(obs)
    except Exception:
        return np.zeros(2)
