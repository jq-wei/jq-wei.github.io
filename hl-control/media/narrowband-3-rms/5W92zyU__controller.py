"""Indirect adaptive regulator for the narrow-band vibration benchmark (no learning, classical adaptive control).

Control law (internal-model / disturbance-observer structure, the plant being stable):

1. Probe and identify (first second, before any tone can be switched on). A small binary probing sequence
   (amplitude PROBE_AMP) is applied for N_PROBE samples; at sample K_FIT the 16 unknown structure parameters (one
   frequency scale and one damping scale per mode) are fitted to the recorded response by a coarse per-mode search
   followed by Levenberg-Marquardt on the output error of the plant's own model.
2. Disturbance observer. A copy of the identified secondary path runs on the applied commands; the measured residual
   minus its output is the reconstructed open-loop disturbance d_hat, which does not depend on the command.
3. Tone estimation. After a detected change (onset or frequency step: the innovation d_hat - fitted tones exceeds
   DETECT_SIGMA noise deviations / DETECT_REL of the tone level), the tones' frequencies and complex amplitudes are
   re-estimated on a window of d_hat that starts after the change (skipping up to SKIP_MAX samples of the structure's
   ring-down) and is at most WIN_MAX samples long: spectral peak picking on the fit residual (RELAX) for the
   initial guess, Gauss-Newton nonlinear least squares for the refinement.
4. Cancellation. Every tone D e^{jwk} is cancelled by the command -D / G(e^{jw}) e^{jwk}, with G the identified
   secondary path (delay included). Because d_hat is rebuilt with the same model, the fixed point is exact
   cancellation even with a model error (internal model principle).

Gains: PROBE_AMP/N_PROBE set the identification signal-to-noise ratio (the probe moves the sensor by about 0.35 V
peak, range 2 V); FIT_RMS_OK is the identification residual above which the modal frequencies are searched again on
a wider grid; DETECT_* set the change detector; HOLD is the time the detector stays disarmed after a change (changes
are 8 s apart at least); WAIT the samples collected before the first estimate after a change (until then the previous
command continues); SKIP_MAX/WIN_MAX the estimation window; A_ADD/A_PRUNE the amplitudes at which a tone is
accepted/dropped (a tone reaches the residual with 24 mV at least).

Safety: the command is clipped to +-u_max, the observer uses the command the plant reports as applied, and a numerical
failure in the identification or in an estimate falls back on the nominal model / the previous tones (never NaN).
"""

from __future__ import annotations

import math

import numpy as np
from scipy import linalg, signal

from hl_control.envs.narrowband import NarrowbandConfig, structure_matrices

PARAMS = {
    "PROBE_AMP": 0.03,      # V, amplitude of the binary probing sequence
    "DETECT_SIGMA": 8.0,    # change detector: innovation RMS above this many noise deviations ...
    "DETECT_REL": 0.12,     # ... and above this fraction of the fitted tones' RMS
    "A_ADD": 0.010,         # V, a spectral peak of the fit residual above this is a new tone
    "A_PRUNE": 0.006,       # V, a fitted tone below this is dropped
}

_CFG = NarrowbandConfig()
_FS = float(_CFG.fs)
_DT = 1.0 / _FS
_NM = len(_CFG.mode_freqs)
_LAG = 2.0 * math.pi * _CFG.actuator_lag_hz
_DELAY = int(_CFG.delay_steps)

N_PROBE = 720            # samples of probing
K_FIT = 792              # sample at which the structure is identified (the tones start at sample 800 at the earliest)
HOLD = 2400              # samples after a change during which no new change is declared
WAIT = 200               # samples after a change before the first estimate
SKIP_MAX = 480           # samples after a change left out of the estimation window (ring-down of the structure)
WIN_MAX = 1600           # longest estimation window
NFFT = 4096
FIT_RMS_OK = 4.0e-4      # V, identification residual above this (floor: 0.21 mV) triggers a wider search
BAND = (49.0, 96.0)      # Hz, where tones are searched
_RNG = np.random.default_rng(20240531)
_PROBE_SIGN = np.sign(_RNG.standard_normal(N_PROBE))
_FBINS = np.fft.rfftfreq(NFFT, _DT)
_BAND_IDX = np.nonzero((_FBINS >= BAND[0]) & (_FBINS <= BAND[1]))[0]


# ----- the plant's model, mode by mode (each mode with the amplifier lag is a third-order filter) -----

def _mode_tf(i: int, fsc: float, zsc: float) -> tuple[np.ndarray, np.ndarray]:
    """Discrete transfer function (zero-order hold) of mode ``i`` driven through the amplifier lag."""
    w = 2.0 * math.pi * _CFG.mode_freqs[i] * fsc
    z = _CFG.mode_zetas[i] * zsc
    b = _CFG.actuator_gains[i]
    m = np.zeros((4, 4))
    m[0, 1] = 1.0
    m[1, 0] = -w * w
    m[1, 1] = -2.0 * z * w
    m[1, 2] = b
    m[2, 2] = -_LAG
    m[2, 3] = _LAG
    e = linalg.expm(m * _DT)
    ad, bd = e[:3, :3], e[:3, 3]
    c = np.array([-w * w, -2.0 * z * w, b])
    den = np.real(np.poly(ad))
    num = np.real(np.poly(ad - np.outer(bd, c))) - den
    num[0] = 0.0
    return num, den


def _mode_resp(i: int, fsc: float, zsc: float, v: np.ndarray) -> np.ndarray:
    num, den = _mode_tf(i, fsc, zsc)
    return signal.lfilter(num, den, v)


def identify(u: np.ndarray, y: np.ndarray, span: float = 0.025, points: int = 21) -> tuple[np.ndarray, float]:
    """Fit the frequency and damping scales of the eight modes to a recorded response. Returns (p, residual RMS).
    ``span`` is the half-width of the coarse search on every modal frequency scale, ``points`` its grid size."""
    n = len(y)
    v = np.zeros(n)
    v[_DELAY:] = u[: n - _DELAY]
    p = np.ones(2 * _NM)
    resp = [_mode_resp(i, 1.0, 1.0, v) for i in range(_NM)]
    total = np.sum(resp, axis=0)
    # coarse search, one modal frequency at a time (the lightly damped modes first)
    grid = np.linspace(1.0 - span, 1.0 + span, points)
    for _ in range(2):
        for i in (3, 4, 1, 5, 6, 2, 7, 0):
            base = y - (total - resp[i])
            best = (float(np.dot(base - resp[i], base - resp[i])), p[i], resp[i])
            for s in grid:
                r = _mode_resp(i, s, p[_NM + i], v)
                cost = float(np.dot(base - r, base - r))
                if cost < best[0]:
                    best = (cost, s, r)
            total = total - resp[i] + best[2]
            p[i], resp[i] = best[1], best[2]
    # Levenberg-Marquardt on all sixteen scales
    res = y - total
    cost = float(np.dot(res, res))
    lam = 1e-3
    jac = np.zeros((n, 2 * _NM))
    hf, hz = 1e-6, 1e-4
    for _ in range(40):
        for i in range(_NM):
            jac[:, i] = (_mode_resp(i, p[i] + hf, p[_NM + i], v) - resp[i]) / hf
            jac[:, _NM + i] = (_mode_resp(i, p[i], p[_NM + i] + hz, v) - resp[i]) / hz
        jtj = jac.T @ jac
        g = jac.T @ res
        improved = False
        for _ in range(8):
            try:
                dp = np.linalg.solve(jtj + lam * np.diag(np.diag(jtj)), g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            q = p + dp
            q[:_NM] = np.clip(q[:_NM], 0.94, 1.06)
            q[_NM:] = np.clip(q[_NM:], 0.7, 1.4)
            new_resp = [_mode_resp(i, q[i], q[_NM + i], v) for i in range(_NM)]
            new_res = y - np.sum(new_resp, axis=0)
            new_cost = float(np.dot(new_res, new_res))
            if new_cost < cost:
                rel = (cost - new_cost) / max(cost, 1e-300)
                p, resp, res, cost = q, new_resp, new_res, new_cost
                lam = max(lam / 5.0, 1e-10)
                improved = True
                break
            lam *= 5.0
        if not improved or rel < 1e-10:
            break
    return p, math.sqrt(cost / n)


# ----- tone estimation -----

def _nls(s: np.ndarray, w0: np.ndarray, iters: int) -> dict:
    """Nonlinear least squares fit of sinusoids (frequencies ``w0`` in rad/sample as the initial guess) to ``s``.
    The complex amplitudes ``D`` refer to the centre of the window: s[n] ~ Re sum_j D_j exp(j w_j (n - (L-1)/2))."""
    ln = len(s)
    m = len(w0)
    n = np.arange(ln) - 0.5 * (ln - 1)
    w = np.array(w0, dtype=float)
    scale = 2.0 / ln
    step_max = 2.0 * math.pi * 1.0 / _FS
    for it in range(iters + 1):
        ph = np.outer(n, w)
        cs, sn = np.cos(ph), np.sin(ph)
        phi = np.concatenate((cs, sn), axis=1)
        try:
            c = np.linalg.solve(phi.T @ phi, phi.T @ s)
        except np.linalg.LinAlgError:
            return {"ok": False}
        r = s - phi @ c
        if it == iters:
            break
        a, b = c[:m], c[m:]
        jw = (n[:, None] * scale) * (-a * sn + b * cs)
        jac = np.concatenate((phi, jw), axis=1)
        jtj = jac.T @ jac
        jtj[np.diag_indices_from(jtj)] *= 1.0 + 1e-9
        try:
            d = np.linalg.solve(jtj, jac.T @ r)
        except np.linalg.LinAlgError:
            return {"ok": False}
        w = w + np.clip(d[2 * m:] * scale, -step_max, step_max)
    amp = c[:m] - 1j * c[m:]
    ok = bool(np.all(np.isfinite(w)) and np.all(np.isfinite(c)))
    return {"ok": ok, "w": w, "D": amp, "r": r, "r_rms": float(np.sqrt(np.mean(r * r))),
            "s_rms": float(np.sqrt(np.mean(s * s)))}


def _valid(fit: dict) -> bool:
    if not fit.get("ok", False):
        return False
    f = np.sort(fit["w"]) * _FS / (2.0 * math.pi)
    if len(f) == 0:
        return True
    if f[0] < BAND[0] - 0.5 or f[-1] > BAND[1] + 0.5:
        return False
    if len(f) > 1 and float(np.min(np.diff(f))) < 7.0:
        return False
    return bool(np.max(np.abs(fit["D"])) < 1.5)


def _empty_fit(s: np.ndarray) -> dict:
    rms = float(np.sqrt(np.mean(s * s)))
    return {"ok": True, "w": np.zeros(0), "D": np.zeros(0, dtype=complex), "r": s, "r_rms": rms, "s_rms": rms}


def _prune(s: np.ndarray, fit: dict, a_prune: float) -> dict:
    while _valid(fit) and len(fit["w"]) > 0 and float(np.min(np.abs(fit["D"]))) < a_prune:
        keep = np.abs(fit["D"]) > float(np.min(np.abs(fit["D"])))
        w = fit["w"][keep]
        fit = _nls(s, w, 2) if len(w) else _empty_fit(s)
    return fit


def _relax(s: np.ndarray, a_add: float, a_prune: float, max_tones: int = 3) -> dict:
    """Initial estimate: add the largest in-band spectral peak of the fit residual as a new tone, refit, repeat."""
    ln = len(s)
    win = np.hanning(ln)
    gain = 2.0 / float(np.sum(win))
    fit = _empty_fit(s)
    for _ in range(max_tones):
        mag = np.abs(np.fft.rfft(fit["r"] * win, NFFT)) * gain
        seg = mag[_BAND_IDX[0] - 1: _BAND_IDX[-1] + 2]
        peak = (seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])
        idx = _BAND_IDX[peak]
        if len(fit["w"]):
            f_old = fit["w"] * _FS / (2.0 * math.pi)
            far = np.min(np.abs(_FBINS[idx][:, None] - f_old[None, :]), axis=1) >= 8.0
            idx = idx[far]
        if len(idx) == 0:
            break
        best = idx[int(np.argmax(mag[idx]))]
        if mag[best] < a_add:
            break
        w_new = np.append(fit["w"], 2.0 * math.pi * _FBINS[best] / _FS)
        new_fit = _nls(s, w_new, 4)
        if not _valid(new_fit):
            break
        fit = new_fit
    return _prune(s, fit, a_prune)


# ----- the controller -----

class _Regulator:
    def __init__(self, u_max: float) -> None:
        self.u_max = u_max
        n = 4096
        self.y = np.zeros(n)
        self.u = np.zeros(n)
        self.dhat = np.zeros(n)
        self.sigma = 2.1e-4
        self.kc = None                      # sample of the last detected change (None: no tone yet)
        self.have_init = False
        self.w = np.zeros(0)                # tone frequencies [rad/sample]
        self.D = np.zeros(0, dtype=complex)  # tone amplitudes in d_hat at sample kref
        self.U = np.zeros(0, dtype=complex)  # command amplitudes at sample kref
        self.kref = 0.0
        self.r_rms = 0.0
        self.s_rms = 0.0
        self.e2 = np.zeros(16)
        self.e2sum = 0.0

    def _grow(self, k: int) -> None:
        if k >= len(self.y):
            n = 2 * len(self.y)
            for name in ("y", "u", "dhat"):
                old = getattr(self, name)
                new = np.zeros(n)
                new[: len(old)] = old
                setattr(self, name, new)

    # identification of the structure and start of the disturbance observer
    def _fit_model(self, k: int) -> None:
        try:
            p, rms = identify(self.u[:k + 1], self.y[:k + 1])
            if rms > FIT_RMS_OK:            # not at the noise floor: search the modal frequencies on a wider grid
                p2, rms2 = identify(self.u[:k + 1], self.y[:k + 1], span=0.045, points=37)
                if rms2 < rms:
                    p, rms = p2, rms2
        except Exception:                   # never leave the loop without a model: fall back on the nominal structure
            p = np.ones(2 * _NM)
        self.p = p
        model = structure_matrices(_CFG, p[:_NM], p[_NM:])
        self.A, self.B, self.C, _ = model["secondary"]
        tfs = [_mode_tf(i, p[i], p[_NM + i]) for i in range(_NM)]
        self.NUM = np.array([t[0] for t in tfs])
        self.DEN = np.array([t[1] for t in tfs])
        x = np.zeros(self.A.shape[0])
        for i in range(k):                  # x[i + 1] = A x[i] + B u[i - delay]
            v = self.u[i - _DELAY] if i >= _DELAY else 0.0
            self.dhat[i] = self.y[i] - float(self.C @ x)
            x = self.A @ x + self.B * v
        self.dhat[k] = self.y[k] - float(self.C @ x)
        self.x = x
        self.sigma = max(float(np.sqrt(np.mean(self.dhat[8:k + 1] ** 2))), 1.5e-4)

    def _gain(self, w: np.ndarray) -> np.ndarray:
        """Frequency response of the identified secondary path (delay included) at ``w`` [rad/sample]."""
        z = np.exp(1j * w)
        zp = np.array([z ** 3, z ** 2, z, np.ones_like(z)])
        return np.sum((self.NUM @ zp) / (self.DEN @ zp), axis=0) * z ** (-_DELAY)

    def _adopt(self, fit: dict, k0: int, ln: int) -> None:
        self.w = fit["w"]
        self.D = fit["D"]
        self.kref = k0 + 0.5 * (ln - 1)
        self.U = -self.D / self._gain(self.w) if len(self.w) else np.zeros(0, dtype=complex)
        self.r_rms, self.s_rms = fit["r_rms"], fit["s_rms"]
        self.have_init = True

    def _estimate(self, k: int) -> None:
        elapsed = k - self.kc
        skip = min(max(40, elapsed - 400), SKIP_MAX)
        k0 = max(self.kc + skip, k - WIN_MAX + 1)
        s = self.dhat[k0:k + 1]
        a_add, a_prune = PARAMS["A_ADD"], PARAMS["A_PRUNE"]
        fit = None
        if self.have_init and len(self.w):
            fit = _prune(s, _nls(s, self.w, 2), a_prune)
            if not _valid(fit):
                fit = None
        if fit is None or fit["r_rms"] > max(3.0 * self.sigma, 0.03 * fit["s_rms"]):
            alt = _relax(s, a_add, a_prune)
            if _valid(alt) and (fit is None or alt["r_rms"] < fit["r_rms"]):
                fit = alt
        if fit is not None:
            self._adopt(fit, k0, len(s))

    def step(self, k: int, y: float, last_action: float) -> float:
        self._grow(k)
        self.y[k] = y
        if k > 0:
            self.u[k - 1] = last_action     # the command the plant actually applied
        if k < K_FIT:
            u = PARAMS["PROBE_AMP"] * _PROBE_SIGN[k] if k < N_PROBE else 0.0
        elif k == K_FIT:
            self._fit_model(k)
            u = 0.0
        else:
            self.x = self.A @ self.x + self.B * self.u[k - 1 - _DELAY]
            dh = y - float(self.C @ self.x)
            self.dhat[k] = dh
            m = len(self.w)
            if m:
                rot = np.exp(1j * self.w * (k - self.kref))
                e = dh - float(np.sum((self.D * rot).real))
            else:
                rot = None
                e = dh
            j = k & 15
            self.e2sum += e * e - self.e2[j]
            self.e2[j] = e * e
            level = math.sqrt(max(self.e2sum, 0.0) / 16.0)
            if self.kc is None:
                if k >= K_FIT + 16 and level > PARAMS["DETECT_SIGMA"] * self.sigma:
                    self.kc = k - 16
                    self.have_init = False
            else:
                elapsed = k - self.kc
                if elapsed >= HOLD:
                    tone_rms = math.sqrt(0.5 * float(np.sum(np.abs(self.D) ** 2))) if m else 0.0
                    thr = max(PARAMS["DETECT_SIGMA"] * self.sigma, PARAMS["DETECT_REL"] * tone_rms, 4.0 * self.r_rms)
                    if level > thr:
                        self.kc = k - 12
                        self.have_init = False
                        elapsed = 12
                if elapsed >= WAIT and (elapsed - WAIT) % (16 if elapsed < HOLD else 64) == 0:
                    try:
                        self._estimate(k)
                    except Exception:       # keep the previous tones if the estimator fails numerically
                        pass
                    m = len(self.w)
                    rot = np.exp(1j * self.w * (k - self.kref)) if m else None
            u = float(np.sum((self.U * rot).real)) if m else 0.0
        u = min(max(u, -self.u_max), self.u_max)
        self.u[k] = u
        return u


_REG: _Regulator | None = None


def controller(obs: dict) -> float:
    """Residual in, actuator command [V] out; all state is rebuilt at the first sample of an episode."""
    global _REG
    k = int(obs["step"])
    if k == 0 or _REG is None:
        _REG = _Regulator(float(obs.get("u_max", 0.25)))
    try:
        u = _REG.step(k, float(obs["y"]), float(obs.get("last_action", 0.0)))
    except Exception:                       # a numerical failure must not end the run: no command this sample
        u = 0.0
    return u if math.isfinite(u) else 0.0
