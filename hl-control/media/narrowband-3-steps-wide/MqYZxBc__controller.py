"""Self-tuning feedforward cancellation of up to three stepping tones (active vibration control).

Indirect adaptive regulation with an internal-model structure; everything is least squares on a
handful of parameters, adapted online while the controller runs (no learning, no neural network).

1. Identification (samples 0..N_ID, before the tones can be switched on at 1 s).  A random binary
   probe of +-PROBE_AMP volts is applied for PROBE_LEN samples (the residual stays near 0.6 V,
   far inside the +-2 V sensor range).  The secondary path is fitted by linear least squares
   (ARX, refined by Steiglitz-McBride prefiltering so that the sensor noise does not bias it);
   the eight modal frequencies and dampings are read from the fitted poles and put into the
   plant's own structured model (`structure_matrices`).  This removes the +-4 % uncertainty: the
   in-band frequency response is then known to about 0.2 %.  If the fit is worse than the
   nominal model on the probe data, the nominal model is kept.
2. Disturbance reconstruction.  The identified model is run as a predictor on the commands that
   were actually applied; w[k] = y[k] - prediction is the open-loop residual (tones + noise),
   whatever the controller is doing, so estimation and control do not interact.
3. Tone estimation.  A change (the onset or a frequency step) is detected when the RMS of the
   tone model's prediction error over DET_LEN samples exceeds max(THR_ABS, THR_REL * model RMS).
   N_FIRST samples later the tones are located as peaks of a Hann-windowed FFT of w, and from
   then on, every REFIT samples, frequencies, amplitudes and phases of the <= 3 sinusoids are
   refitted by least squares to the last <= N_MAX samples of w (Gauss-Newton steps on the
   frequencies).  Tones that fade are dropped, lines left in the fit residual are added.
4. Cancellation.  u[k] = -Re sum_i  W_i / G(w_i) * exp(j w_i k): every fitted tone (phasor W_i)
   is fed forward through the inverse of the identified secondary-path response G at its
   frequency.  Because w is rebuilt from the measured residual, a model error only changes the
   convergence, not the steady state (the fixed point is exact cancellation).

Gains / constants:
  PROBE_AMP, PROBE_LEN : probe amplitude [V] and length [samples]; N_ID: sample of the model fit
  DET_LEN, THR_ABS, THR_REL : change detector: window [samples], absolute threshold [V], relative
                         threshold (fraction of the tone model's RMS)
  N_FIRST, SKIP        : samples waited after a detected change before the first FFT / fit, and
                         samples right after the change left out of the fit (largest transients)
  HOLDOFF              : samples after a change during which no further change is accepted
  REFIT, N_MAX         : refit period and longest fit window [samples] (2 s: noise-limited fits)
  AMP_MIN, AMP_REL     : smallest line accepted as a tone [V] / as a fraction of the strongest
  F_LO, F_HI, NFFT     : band searched for tones [Hz] and FFT length of the peak search
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import lfilter

from hl_control.envs.narrowband import NarrowbandConfig, frequency_response, structure_matrices

FS = 800.0
PROBE_AMP = 0.05
PROBE_LEN = 600
N_ID = 792
DET_LEN = 40
THR_ABS = 3.0e-3
THR_REL = 0.10
N_FIRST = 280
SKIP = 40
HOLDOFF = 1200
REFIT = 40
N_MAX = 1600
AMP_MIN = 6.0e-3
AMP_REL = 0.05
F_LO, F_HI = 47.0, 98.0
NFFT = 4096

_CFG = NarrowbandConfig()
_FNOM = np.asarray(_CFG.mode_freqs)
_ZNOM = np.asarray(_CFG.mode_zetas)
_PROBE = PROBE_AMP * np.sign(np.random.default_rng(12345).standard_normal(PROBE_LEN))
_NBUF = 40000


def _arx_fit(u, y, na=17, nb=17, d=3):
    n = len(y)
    k0 = d + nb
    phi = np.empty((n - k0, na + nb))
    for i in range(1, na + 1):
        phi[:, i - 1] = -y[k0 - i:n - i]
    for j in range(nb):
        phi[:, na + j] = u[k0 - d - j:n - d - j]
    th = np.linalg.lstsq(phi, y[k0:], rcond=None)[0]
    return th[:na], th[na:]


def _identify_modes(u, y, iters=6):
    """ARX + Steiglitz-McBride; returns frequency / damping scale factors of the eight modes."""
    a, _ = _arx_fit(u, y)
    for _ in range(iters):
        den = np.concatenate(([1.0], a))
        if not np.all(np.isfinite(den)) or np.abs(np.roots(den)).max() >= 0.9995:
            break
        a, _ = _arx_fit(lfilter([1.0], den, u), lfilter([1.0], den, y))
    fsc, zsc = np.ones(8), np.ones(8)
    den = np.concatenate(([1.0], a))
    if not np.all(np.isfinite(den)):
        return fsc, zsc
    poles = np.roots(den)
    poles = poles[np.imag(poles) > 1e-6]
    if len(poles) == 0:
        return fsc, zsc
    s = np.log(poles) * FS
    wn = np.abs(s)
    f = wn / (2.0 * math.pi)
    z = -np.real(s) / wn
    for i in range(8):
        j = int(np.argmin(np.abs(f - _FNOM[i])))
        rf, rz = f[j] / _FNOM[i], z[j] / _ZNOM[i]
        if abs(rf - 1.0) < 0.05 and 0.8 < rz < 1.25:
            fsc[i], zsc[i] = rf, rz
    return fsc, zsc


def _simulate(model, u, n):
    """x[k+1] = A x[k] + B u[k-2]; returns C x[k] for k = 0..n and the state x[n]."""
    a, b, c, _ = model
    x = np.zeros(a.shape[0])
    out = np.zeros(n + 1)
    for k in range(n):
        v = u[k - 2] if k >= 2 else 0.0
        x = a @ x + b * v
        out[k + 1] = c @ x
    return out, x


def _fft_peaks(seg, exclude=(), max_n=3, amp_min=AMP_MIN):
    """Frequencies [Hz] of the dominant spectral lines of seg inside the band (Hann, zero-padded)."""
    n = len(seg)
    win = np.hanning(n)
    spec = np.abs(np.fft.rfft(seg * win, NFFT)) * (2.0 / win.sum())
    df = FS / NFFT
    lo, hi = int(F_LO / df), int(F_HI / df)
    band = spec[lo:hi + 1]
    idx = np.nonzero((band[1:-1] >= band[:-2]) & (band[1:-1] > band[2:]))[0] + 1
    if len(idx) == 0:
        return []
    idx = idx[np.argsort(-band[idx])]
    top = band[idx[0]]
    found = []
    for i in idx:
        amp = band[i]
        if amp < max(amp_min, AMP_REL * top) or len(found) >= max_n:
            break
        y0, y1, y2 = band[i - 1], band[i], band[i + 1]
        den = y0 - 2.0 * y1 + y2
        frac = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        fpk = (lo + i + frac) * df
        if all(abs(fpk - g) >= 9.0 for g in found) and all(abs(fpk - g) >= 9.0 for g in exclude):
            found.append(fpk)
    return found


def _fit_tones(seg, tau, om, iters):
    """Least-squares fit of sum_i a_i cos(om_i tau) + b_i sin(om_i tau); Gauss-Newton on om."""
    n = len(om)
    half = 0.5 * len(seg)
    ts = tau / half
    for it in range(iters + 1):
        ph = np.outer(tau, om)
        cm, sm = np.cos(ph), np.sin(ph)
        h = np.hstack((cm, sm))
        g = h.T @ h
        g[np.diag_indices_from(g)] += 1e-9 * len(seg)
        ab = np.linalg.solve(g, h.T @ seg)
        a, b = ab[:n], ab[n:]
        r = seg - h @ ab
        if it == iters:
            break
        jw = ts[:, None] * (cm * b - sm * a)
        j = np.hstack((jw, h))
        jj = j.T @ j
        jj[np.diag_indices_from(jj)] += 1e-9 * np.trace(jj) / len(jj) + 1e-18
        step = np.linalg.solve(jj, j.T @ r)[:n] / half
        lim = 0.5 * math.pi / len(seg)
        om = om + np.clip(step, -lim, lim)
    return om, a, b, r


class _Controller:
    def __init__(self):
        self.y = np.zeros(_NBUF)
        self.u = np.zeros(_NBUF)
        self.w = np.zeros(_NBUF)
        self.reset()

    def reset(self):
        self.y[:] = 0.0
        self.u[:] = 0.0
        self.w[:] = 0.0
        self.model = None
        self.xm = None
        self.mode = "idle"
        self.kc = 0
        self.eb = np.zeros(DET_LEN)
        self.esum = 0.0
        self.thr_abs = THR_ABS
        self._clear_tones()

    def _clear_tones(self):
        self.om = np.zeros(0)
        self.a = np.zeros(0)
        self.b = np.zeros(0)
        self.ua = np.zeros(0)
        self.ub = np.zeros(0)
        self.kref = 0
        self.pmodel = 0.0

    # ----- identification -----
    def _identify(self, k):
        u, y = self.u[:k + 1], self.y[:k + 1]
        nominal = structure_matrices(_CFG)["secondary"]
        best, best_x, best_out, best_err = nominal, None, None, np.inf
        try:
            fsc, zsc = _identify_modes(u, y)
            cands = [structure_matrices(_CFG, fsc, zsc)["secondary"], nominal]
        except Exception:
            cands = [nominal]
        for m in cands:
            out, x = _simulate(m, u, k)
            err = float(np.sum((y - out) ** 2))
            if err < best_err:
                best, best_x, best_out, best_err = m, x, out, err
        self.model = best
        self.A, self.B, self.C = best[0], best[1], best[2]
        self.xm = best_x
        self.w[:k + 1] = y - best_out
        floor = float(np.sqrt(np.mean(self.w[k - 150:k + 1] ** 2)))
        self.thr_abs = max(THR_ABS, 6.0 * floor)

    # ----- tone model -----
    def _set_tones(self, om, a, b, kref):
        self.om, self.a, self.b, self.kref = om, a, b, kref
        g = frequency_response(self.model, om * FS / (2.0 * math.pi), FS, 2)
        uc = -(a - 1j * b) / g
        self.ua, self.ub = np.real(uc).copy(), -np.imag(uc)
        self.pmodel = 0.5 * float(np.sum(a * a + b * b))

    def _window(self, k):
        since = k - self.kc
        k0 = self.kc + SKIP + min(360, max(0, (since - N_FIRST) // 2))
        return max(k0, k - N_MAX + 1)

    def _acquire(self, k):
        k0 = self._window(k)
        seg = self.w[k0:k + 1]
        freqs = _fft_peaks(seg)
        if not freqs:
            self.mode = "idle"
            self._clear_tones()
            return
        kref = (k0 + k) // 2
        tau = np.arange(k0 - kref, k + 1 - kref, dtype=float)
        om, a, b, _ = _fit_tones(seg, tau, 2.0 * math.pi * np.sort(np.asarray(freqs)) / FS, 4)
        self._set_tones(om, a, b, kref)
        self.mode = "track"

    def _refit(self, k):
        k0 = self._window(k)
        seg = self.w[k0:k + 1]
        n = len(seg)
        kref = (k0 + k) // 2
        tau = np.arange(k0 - kref, k + 1 - kref, dtype=float)
        om, a, b, r = _fit_tones(seg, tau, self.om, 2)
        amp = np.hypot(a, b)
        changed = False
        if n >= 800:
            keep = amp >= max(0.6 * AMP_MIN, 0.02 * amp.max())
            fr = om * FS / (2.0 * math.pi)
            keep &= (fr > F_LO - 2.0) & (fr < F_HI + 2.0)
            for i in range(len(om)):          # two models on one tone: keep the stronger
                for j in range(i + 1, len(om)):
                    if keep[i] and keep[j] and abs(fr[i] - fr[j]) < 4.0:
                        keep[i if amp[i] < amp[j] else j] = False
            if not keep.all():
                om = om[keep]
                changed = True
            if len(om) < 3 and float(np.sqrt(np.mean(r * r))) > 0.5 * self.thr_abs:
                extra = _fft_peaks(r, exclude=list(om * FS / (2.0 * math.pi)), max_n=3 - len(om), amp_min=1.3 * AMP_MIN)
                if extra:
                    om = np.sort(np.concatenate((om, 2.0 * math.pi * np.asarray(extra) / FS)))
                    changed = True
        if changed:
            if len(om) == 0:
                self.mode = "idle"
                self._clear_tones()
                return
            om, a, b, r = _fit_tones(seg, tau, om, 3)
        self._set_tones(om, a, b, kref)

    # ----- one sample -----
    def step(self, obs):
        k = int(obs["step"])
        if k == 0:
            self.reset()
        if k >= _NBUF:
            return 0.0
        y = float(obs["y"])
        self.y[k] = y
        if k > 0:
            self.u[k - 1] = float(obs["last_action"])
        if k < N_ID:
            return float(_PROBE[k]) if k < PROBE_LEN else 0.0
        if k == N_ID:
            self._identify(k)
        else:
            self.xm = self.A @ self.xm + self.B * self.u[k - 3]
            self.w[k] = y - float(self.C @ self.xm)
        w = self.w[k]

        if self.mode == "acquire":
            if k - self.kc >= N_FIRST:
                try:
                    self._acquire(k)
                except Exception:
                    self.mode = "idle"
                    self._clear_tones()
            if self.mode == "acquire":
                return 0.0

        # prediction error of the tone model and the change detector
        if len(self.om):
            ph = self.om * (k - self.kref)
            c, s = np.cos(ph), np.sin(ph)
            e = w - float(self.a @ c + self.b @ s)
        else:
            e = w
        i = k % DET_LEN
        e2 = e * e
        self.esum += e2 - self.eb[i]
        self.eb[i] = e2
        if (self.mode == "idle" or k - self.kc > HOLDOFF) and \
                self.esum > DET_LEN * max(self.thr_abs ** 2, THR_REL ** 2 * self.pmodel):
            self.mode = "acquire"
            self.kc = k
            self.eb[:] = 0.0
            self.esum = 0.0
            self._clear_tones()
            return 0.0

        if self.mode == "track" and (k - self.kc) % REFIT == 0:
            try:
                self._refit(k)
            except Exception:
                pass
            if not len(self.om):
                return 0.0
            ph = self.om * (k - self.kref)
            c, s = np.cos(ph), np.sin(ph)
        if not len(self.om):
            return 0.0
        u = float(self.ua @ c + self.ub @ s)
        um = float(obs["u_max"])
        return min(um, max(-um, u))


_CTL = _Controller()


def controller(obs: dict) -> float:
    """Command [V] for the next sample; any internal failure degrades to zero command, never to a NaN."""
    try:
        u = _CTL.step(obs)
    except Exception:
        u = 0.0
    return u if math.isfinite(u) else 0.0
