"""Commissioning controller for the ill-conditioned 2 x 2 column.

Coordinates.  U (rotation by 51.3 deg) is known, so everything is done in modal outputs
z = U^T y: z_1 is the high-gain (slow) mode, z_2 the low-gain (faster) mode, and each one is a
first-order lag driven by one direction of the delayed commands:

    z_i[k+1] = a_i z_i[k] + c_i . u[k - theta],      c_i = (1 - a_i) g_i * (row i of V^T W)

Identification (commissioning window), all by output-error least squares on the recorded
commands and outputs, over a grid of dead times and time constants (linear in c_i for each):
  A. an exponentially growing command on actuator a until z_1 moves (scale of the high gain
     without knowing it), then a pulse of comparable size on actuator b: first estimate of c_1.
  B. z_1 is regulated at zero with that model while a square wave is applied along the
     direction orthogonal to c_1 (the low-gain input direction).  The fit is repeated every few
     steps, so any leak of the square wave into z_1 corrects the direction; z_2's response gives
     the low gain, its time constant and the dead time (c_2 is constrained orthogonal to c_1).
  C. the evaluation controller runs on rehearsal set points (same size as the real ones) and the
     model keeps being refitted; the set points return to zero well before the evaluation.

Control law (rehearsal and evaluation): internal model + disturbance observer + dead-time
predictor.  The model is simulated with the commands actually applied; the residual
(measurement - model) of each mode is filtered by a 2-state Kalman filter (p: output
disturbance, s: the level it relaxes to through the known 40 min lag) and extrapolated over
the dead time.  The command makes the predicted error of mode i at k+theta+1 equal to phi_i
times the predicted error at k+theta (phi = 0 is dead-beat).  If the actuators saturate, the
high-gain mode keeps its command and the low-gain move is scaled back.

Gains (PARAMS):
  ramp_u0, ramp_T, detect : start level / e-folding time of the probing ramp, z_1 level ending it
  wait, pulse_len, pulse_frac : rest after a probe, length and relative size of the b pulse
  exc_amp, exc_len        : amplitude and total length of the low-gain square wave
  phi1_id                 : closed-loop pole of the z_1 regulator during identification
  phi1, phi2              : closed-loop poles of the two modes in rehearsal / evaluation
  q1p, q1s, q2p, q2s      : process-noise std of the disturbance observer states (bandwidth)
  rehearse_every          : minutes between rehearsal set-point changes
  settle                  : minutes kept at zero set point before the evaluation opens
"""

import math

import numpy as np
from scipy.signal import lfilter

PARAMS = {
    "ramp_u0": 0.0005, "ramp_T": 8.0, "detect": 0.04,
    "wait": 12, "pulse_len": 10, "pulse_frac": 0.7,
    "exc_amp": 0.25, "exc_len": 100,
    "phi1_id": 0.85,
    "phi1": 0.0, "phi2": 0.0,
    "q1p": 0.0005, "q1s": 0.002, "q2p": 0.0005, "q2s": 0.0005,
    "rehearse_every": 45, "settle": 70,
}

_ANG = math.radians(51.3)
_UT = np.array([[math.cos(_ANG), math.sin(_ANG)], [-math.sin(_ANG), math.cos(_ANG)]])  # z = _UT @ y
_AD = math.exp(-1.0 / 40.0)          # disturbance lag (known)
_UMAX = 2.0
_NOISE = 0.004
_THETAS = tuple(range(2, 11))
_TAU1 = np.geomspace(80.0, 200.0, 17)
_TAU2 = np.geomspace(10.0, 30.0, 23)
_A1 = np.exp(-1.0 / _TAU1)
_A2 = np.exp(-1.0 / _TAU2)
_REHEARSAL = ((1, 0), (1, 1), (0, 1), (0, 0), (-1, 0), (-1, -1), (0, -1), (0, 0))


def _kf_gain(qp, qs):
    """Stationary Kalman gain of the disturbance observer x = (p, s), p+ = ad p + (1 - ad) s, s+ = s."""
    a = np.array([[_AD, 1.0 - _AD], [0.0, 1.0]])
    q = np.diag([qp * qp, qs * qs])
    r = _NOISE * _NOISE
    p = np.eye(2) * 1e-4
    gain = np.zeros(2)
    for _ in range(4000):
        p = a @ p @ a.T + q
        gain = p[:, 0] / (p[0, 0] + r)
        p = p - np.outer(gain, p[0, :])
    return gain


class _Unit:
    def __init__(self, obs):
        self.P = dict(PARAMS)
        self.n_comm = int(round(float(obs["time_to_evaluation"]) / float(obs["dt"])))
        n = self.n_comm + int(round(float(obs["evaluation_time"]) / float(obs["dt"]))) + 8
        self.u = np.zeros((n, 2))
        self.z = np.zeros((n, 2))
        self.stage = "ramp"
        self.t0 = 0
        self.amp_b = 0.0
        self.model = None            # dict(theta, a (2,), c (2, 2), full)
        self.zm = np.zeros(2)        # internal model state at the current step
        self.dist = np.zeros((2, 2))  # rows: mode; columns: (p, s)
        self.gain = np.array([_kf_gain(self.P["q1p"], self.P["q1s"]), _kf_gain(self.P["q2p"], self.P["q2s"])])
        self.t_b = 0
        self.t_c = 0
        self._model_step = -1        # step at which the model state was rebuilt by a refit

    # ---------------------------------------------------------------- identification
    def fit(self, k, full):
        """Output-error least squares on u[0..k-1], z[0..k]."""
        u = self.u[:k]
        z1, z2 = self.z[: k + 1, 0], self.z[: k + 1, 1]
        f1 = np.stack([lfilter([1.0], [1.0, -a], u, axis=0) for a in _A1])       # (n_tau1, k, 2)
        f2 = np.stack([lfilter([1.0], [1.0, -a], u, axis=0) for a in _A2]) if full else None
        best = None
        for th in _THETAS:
            m = k - th
            if m < 6:
                continue
            y1 = z1[th + 1:]
            x = f1[:, :m, :]
            g = np.einsum("tni,tnj->tij", x, x) + 1e-12 * np.eye(2)
            b = np.einsum("tni,n->ti", x, y1)
            c = np.linalg.solve(g, b[:, :, None])[:, :, 0]
            sse1 = float(z1 @ z1) - np.einsum("ti,ti->t", c, b)
            i1 = int(np.argmin(sse1))
            c1 = c[i1]
            cost = float(sse1[i1])
            a2, c2 = _A2[len(_A2) // 2], np.zeros(2)
            if full:
                nrm = float(np.hypot(c1[0], c1[1])) + 1e-12
                m2 = np.array([-c1[1], c1[0]]) / nrm
                y2 = z2[th + 1:]
                xs = f2[:, :m, :] @ m2                                              # (n_tau2, m)
                den = np.einsum("tn,tn->t", xs, xs) + 1e-12
                num = xs @ y2
                sse2 = float(z2 @ z2) - num * num / den
                i2 = int(np.argmin(sse2))
                a2, c2 = _A2[i2], (num[i2] / den[i2]) * m2
                cost += float(sse2[i2])
            if best is None or cost < best[0]:
                best = (cost, th, _A1[i1], c1.copy(), a2, c2.copy())
        if best is None:
            return
        _, th, a1, c1, a2, c2 = best
        self.model = {"theta": th, "a": np.array([a1, a2]), "c": np.array([c1, c2]), "full": full}
        self.resim(k)

    def resim(self, k):
        """Internal model state at step k from the whole command history."""
        mdl = self.model
        j = k - 1 - mdl["theta"]
        zm = np.zeros(2)
        if j >= 0:
            for i in range(2):
                f = lfilter([1.0], [1.0, -mdl["a"][i]], self.u[: j + 1] @ mdl["c"][i])
                zm[i] = f[-1]
        self.zm = zm

    # ---------------------------------------------------------------- control
    def advance(self, k):
        """Move the internal model and the disturbance observer to step k."""
        mdl = self.model
        if mdl is None:
            return
        if self._model_step != k:
            j = k - 1 - mdl["theta"]
            drive = mdl["c"] @ self.u[j] if j >= 0 else np.zeros(2)
            self.zm = mdl["a"] * self.zm + drive
        e = self.z[k] - self.zm
        for i in range(2):
            p, s = self.dist[i]
            p_pred = _AD * p + (1.0 - _AD) * s
            nu = e[i] - p_pred
            self.dist[i] = (p_pred + self.gain[i, 0] * nu, s + self.gain[i, 1] * nu)

    def targets(self, k, zr, phi):
        """Required c_i . u[k] for each mode (dead-time predictor + disturbance extrapolation)."""
        mdl = self.model
        th = mdl["theta"]
        zp = self.zm.copy()
        for j in range(k - th, k):
            zp = mdl["a"] * zp + (mdl["c"] @ self.u[j] if j >= 0 else 0.0)
        w0, w1 = _AD ** th, _AD ** (th + 1)
        p, s = self.dist[:, 0], self.dist[:, 1]
        d0 = w0 * p + (1.0 - w0) * s
        d1 = w1 * p + (1.0 - w1) * s
        return zr - d1 - mdl["a"] * zp - phi * (zr - zp - d0)

    def control(self, k, zr):
        mdl = self.model
        rhs = self.targets(k, zr, np.array([self.P["phi1"], self.P["phi2"]]))
        if abs(float(np.linalg.det(mdl["c"]))) < 1e-14:      # degenerate fit: no usable model, apply nothing
            return np.zeros(2)
        h = np.linalg.inv(mdl["c"])
        u1 = h[:, 0] * rhs[0]
        d = h[:, 1] * rhs[1]
        kappa = 1.0
        for i in range(2):
            if abs(d[i]) > 1e-12:
                lim = (_UMAX * np.sign(d[i]) - u1[i]) / d[i]
                kappa = min(kappa, max(lim, 0.0))
        return u1 + kappa * d

    def regulate_z1(self, k, w2):
        """Stage B: hold z_1 at zero, apply w2 along the estimated low-gain input direction."""
        mdl = self.model
        rhs = self.targets(k, np.zeros(2), np.array([self.P["phi1_id"], 0.0]))
        c1 = mdl["c"][0]
        nrm = float(np.hypot(c1[0], c1[1])) + 1e-12
        m1 = c1 / nrm
        m2 = np.array([-m1[1], m1[0]])
        return m1 * (rhs[0] / nrm) + m2 * w2

    def excitation(self, tb):
        amp, total = self.P["exc_amp"], float(self.P["exc_len"])
        x = tb / total
        if x < 0.10:
            return 0.2 * amp
        if x < 0.40:
            return amp
        if x < 0.70:
            return -amp
        if x < 0.85:
            return amp
        return -amp

    def rehearsal(self, k):
        every = int(self.P["rehearse_every"])
        last = self.n_comm - int(self.P["settle"])
        t = k - self.t_c - 10
        if t < 0 or k >= last:
            return np.zeros(2)
        idx = t // every
        # a set point that could not be held for a full interval is not started
        if self.t_c + 10 + (idx + 1) * every > last:
            return np.zeros(2)
        return 0.15 * np.array(_REHEARSAL[idx % len(_REHEARSAL)], dtype=float)

    # ---------------------------------------------------------------- one step
    def step(self, obs):
        P = self.P
        k = int(obs["step"])
        y = np.asarray(obs["y"], dtype=float)
        self.z[k] = _UT @ y
        if k > 0:
            self.u[k - 1] = np.asarray(obs["last_action"], dtype=float)
        commissioning = bool(obs["commissioning"])
        self._model_step = -1
        u = np.zeros(2)

        if self.stage == "ramp":
            hit = k >= 3 and abs(float(np.mean(self.z[k - 2: k + 1, 0]))) > P["detect"]
            if hit or float(np.max(np.abs(y))) > 0.12 or k >= 90:
                self.stage, self.t0 = "wait1", k
            else:
                u = np.array([min(P["ramp_u0"] * math.exp(k / P["ramp_T"]), 0.5), 0.0])
        if self.stage == "wait1" and k - self.t0 >= P["wait"]:
            self.stage, self.t0 = "pulse", k
            self.amp_b = P["pulse_frac"] * float(np.sum(self.u[:k, 0])) / P["pulse_len"]
        if self.stage == "pulse":
            if k - self.t0 >= P["pulse_len"]:
                self.stage, self.t0 = "wait2", k
            else:
                u = np.array([0.0, self.amp_b])
        if self.stage == "wait2" and k - self.t0 >= P["wait"]:
            self.stage, self.t_b = "B", k
            self.fit(k, full=False)
            self._model_step = k
        if self.stage == "B":
            tb = k - self.t_b
            if tb >= P["exc_len"] or k >= self.n_comm - 60:
                self.stage, self.t_c = "C", k
                self.fit(k, full=True)
                self._model_step = k
            else:
                if tb > 0 and tb % 5 == 0:
                    self.fit(k, full=False)
                    self._model_step = k
                self.advance(k)
                w2 = self.excitation(tb)
                if float(np.max(np.abs(y))) > 0.7:
                    w2 = 0.0
                u = self.regulate_z1(k, w2)
        if self.stage == "C":
            if commissioning and k > self.t_c and (k - self.t_c) % 5 == 0:
                self.fit(k, full=True)
                self._model_step = k
            self.advance(k)
            r = self.rehearsal(k) if commissioning else np.asarray(obs["r"], dtype=float)
            u = self.control(k, _UT @ r)

        u = np.clip(np.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0), -_UMAX, _UMAX)
        self.u[k] = u
        return u


_unit = None


def controller(obs: dict) -> np.ndarray:
    """Two actuator commands for this step; a new unit starts whenever obs["step"] is 0."""
    global _unit
    if _unit is None or int(obs["step"]) == 0:
        _unit = _Unit(obs)
    try:
        return _unit.step(obs)
    except Exception:   # a failed call would fail the unit; with a zero command the plant relaxes to zero
        return np.zeros(2)
