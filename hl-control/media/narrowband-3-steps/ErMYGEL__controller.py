"""Adaptive narrow-band disturbance rejection: identify the structure, then cancel the fitted tones.

Control law (classical indirect adaptive regulation in internal-model-control form, no learning offline):

1. Identification (first 0.78 s, before any tone can be switched on).  A small pseudo-random binary
   probe (+-``probe_amp`` volts, ``probe_len`` samples) is applied and the eight modal frequencies and
   dampings of this seed's structure are fitted to the measured response (output-error least squares on
   the known modal model).  This gives the secondary path G and, because both paths share the structure's
   poles, the primary path's poles too.
2. Disturbance observer.  A copy of the identified secondary path runs in parallel with the plant;
   ``w = y - G u`` is the disturbance as it reaches the sensor, independent of the command.
3. Transient equaliser.  ``v = F w`` with ``F = A_p(q) / A_p(q / rho)`` (A_p: the structure's poles): after
   a frequency jump the structure's own ring-down, which is not a tone, dies in ~``guard`` samples
   instead of ~1 s, so that ``v`` is again a clean sum of sinusoids.
4. Tone estimator.  Up to three sinusoids (frequency, amplitude, phase) are fitted to a window of ``v``
   by Gauss-Newton least squares.  A jump of the one-step prediction error (``det_k`` noise sigmas over
   ``det_len`` samples) marks the onset / a frequency change: the window restarts ``guard`` samples later,
   is first fitted when it holds ``n_min`` samples (warm start from the old frequencies or periodogram
   peaks), then grows to ``n_max`` samples and slides.  ``holdoff`` blocks new detections while it grows.
5. Command.  Each fitted tone (complex amplitude V at frequency w) is cancelled through the identified
   plant: ``u = Re sum -V / (F(e^jw) G(e^jw)) e^{jwk}``; the secondary path's delay and phase are thus
   compensated exactly at every tone, and the sensor noise reaches the command only through the
   window average (no broadband feedback, hence no amplification between the tones).
6. Structure checks.  When the growing window reaches 200, 400 and 800 samples the fit is audited: a line
   weaker than ``_A_MIN`` (or 5 % of the strongest) is dropped, a remaining periodogram peak above that
   level is added as a tone, and a fit whose residual exceeds three noise sigmas is restarted from the
   periodogram.  The number of tones (1 to 3) is found this way and is never assumed.

Gains (``_P``): ``probe_amp`` / ``probe_len`` / ``id_step`` set the identification experiment; ``rho`` trades
the equaliser's settling time against the tone level left near the structure's modes; ``guard`` and
``n_min`` set how soon after a change the first estimate is used (about 0.19 s); ``n_max`` sets the
averaging of the steady estimate (noise in the command falls as 1/n_max); ``det_len`` / ``det_k`` set the
detector's delay against its false-alarm rate; ``holdoff`` must stay below the 8 s between changes.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import optimize, signal

from hl_control.envs.narrowband import NarrowbandConfig, structure_matrices

_P = {
    "probe_amp": 0.04,     # V, amplitude of the identification probe
    "probe_len": 600,      # samples of probe (0.75 s; the tones start at 1 s at the earliest)
    "id_step": 620,        # sample at which the structure is fitted
    "rho": 0.85,           # pole radius factor of the transient equaliser F
    "guard": 40,           # samples skipped after a detected change (equaliser transient)
    "n_min": 96,           # window length of the first fit after a change
    "n_max": 800,          # steady window length of the tone fit
    "det_len": 12,         # samples averaged by the change detector
    "det_k": 5.0,          # detection threshold in noise sigmas
    "holdoff": 1200,       # samples without new detections after a change
}

_CFG = NarrowbandConfig()
_FS = _CFG.fs
_W2F = _FS / (2.0 * math.pi)
_LAG = 2.0 * math.pi * _CFG.actuator_lag_hz
_NBUF = 48000
_NW = 2048
_T = np.arange(-_NW + 1, 1.0)
_NFFT = 4096
_K0, _K1 = int(46.0 * _NFFT / _FS), int(99.0 * _NFFT / _FS) + 1
_FB = np.arange(_K0, _K1) * _FS / _NFFT
_OM_LO, _OM_HI = 44.0 / _W2F, 101.0 / _W2F
_A_MIN = 2.0e-3                              # V: a fitted line below this (after the equaliser) is not a tone
DEBUG: dict = {}


def _mode_tf(f_hz: float, zeta: float):
    """Exact ZOH transfer function of one mode driven through the amplifier lag (unit modal gain)."""
    w = 2.0 * math.pi * f_hz
    a = np.array([[0.0, 1.0, 0.0], [-w * w, -2.0 * zeta * w, 1.0], [0.0, 0.0, -_LAG]])
    b = np.array([[0.0], [0.0], [_LAG]])
    c = np.array([[-w * w, -2.0 * zeta * w, 1.0]])
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b, c, np.zeros((1, 1))), 1.0 / _FS, method="zoh")
    num, den = signal.ss2tf(ad, bd, cd, dd)
    return num[0], den


def _identify(vin: np.ndarray, y: np.ndarray):
    """Fit the scale factors of the 8 modal frequencies and dampings to the probe response."""
    f0 = np.asarray(_CFG.mode_freqs)
    z0 = np.asarray(_CFG.mode_zetas)
    g = np.asarray(_CFG.actuator_gains)

    def resp(i, fs_, zs_):
        num, den = _mode_tf(f0[i] * fs_, z0[i] * zs_)
        return g[i] * signal.lfilter(num, den, vin)

    def res(p):
        out = -y
        for i in range(8):
            out = out + resp(i, p[i], p[8 + i])
        return out

    def jac(p):
        jm = np.empty((len(y), 16))
        hf, hz = 1e-6, 1e-5
        for i in range(8):
            base = resp(i, p[i], p[8 + i])
            jm[:, i] = (resp(i, p[i] + hf, p[8 + i]) - base) / hf
            jm[:, 8 + i] = (resp(i, p[i], p[8 + i] + hz) - base) / hz
        return jm

    best = None
    for start in (0.0, 0.012, -0.012):
        x0 = np.ones(16)
        x0[:8] += start
        sol = optimize.least_squares(res, x0, jac=jac, method="lm", x_scale=np.r_[np.full(8, 0.01), np.full(8, 0.05)], max_nfev=60)
        rms = math.sqrt(float(np.mean(sol.fun ** 2)))
        if best is None or rms < best[1]:
            best = (sol.x, rms)
        if rms < 4.0e-4:
            break
    return best


class _Controller:
    def __init__(self) -> None:
        self.reset()

    # ----- episode start -----
    def reset(self) -> None:
        p = _P
        self.k_id = int(p["id_step"])
        self.n_probe = int(p["probe_len"])
        rng = np.random.default_rng(12345)
        self.probe = float(p["probe_amp"]) * np.sign(rng.standard_normal(self.n_probe))
        self.yb = np.zeros(_NBUF)
        self.ub = np.zeros(_NBUF + 4)          # ub[j] = u(j - 3)
        self.vb = np.zeros(_NBUF)
        self.ident = False
        self.n = 0
        self.om = np.zeros(0)
        self.zc: list = []
        self.uc: list = []
        self.rot: list = []
        self.pending = False
        self.ws = 0
        self.hold = 0
        self.next_fit = 0
        self.checks: list = []
        self.det_len = int(p["det_len"])
        self.ering = [0.0] * self.det_len
        self.esum = 0.0
        self.sig = 3.0e-4
        self.thr2 = 0.0
        self.last_rms = 0.0
        DEBUG.clear()
        DEBUG["events"] = []

    # ----- identification and model set-up -----
    def _setup(self, k: int) -> None:
        vin = self.ub[1:k + 2]                  # u(j - 2), j = 0..k
        try:
            scales, rms = _identify(vin, self.yb[:k + 1])
            if not np.all(np.isfinite(scales)):
                raise FloatingPointError("identification diverged")
        except Exception:                       # fall back on the nominal structure
            scales, rms = np.ones(16), math.inf
        DEBUG["id_scales"], DEBUG["id_rms"] = scales.copy(), rms
        model = structure_matrices(_CFG, scales[:8], scales[8:])
        self.A, self.B, self.C, _ = model["secondary"]
        lam, vec = np.linalg.eig(self.A)
        self.g_lam = lam
        self.g_res = (self.C @ vec) * np.linalg.solve(vec, self.B.astype(complex))
        lp = np.linalg.eigvals(model["primary"][0])
        rho = float(_P["rho"])
        self.f_num, self.f_den = lp, rho * lp
        sos = []
        for lm in lp[lp.imag > 0]:
            sos.append([1.0, -2.0 * lm.real, abs(lm) ** 2, 1.0, -2.0 * rho * lm.real, (rho * abs(lm)) ** 2])
        sos = np.asarray(sos)
        # replay the model on the recorded commands: x(j) and w(j) for j = 0..k
        x = np.zeros(self.A.shape[0])
        w = np.empty(k + 1)
        w[0] = self.yb[0]
        for j in range(1, k + 1):
            x = self.A @ x + self.B * self.ub[j]
            w[j] = self.yb[j] - self.C @ x
        self.x = x
        v, zi = signal.sosfilt(sos, w, zi=np.zeros((len(sos), 2)))
        self.vb[:k + 1] = v
        self.sos = [tuple(float(c) for c in (s[0], s[1], s[2], s[4], s[5])) for s in sos]
        self.sz = [[float(z[0]), float(z[1])] for z in zi]
        sig = float(np.sqrt(np.mean(v[k - 200:k + 1] ** 2)))
        self.sig = min(max(sig, 2.5e-4), 6.0e-4)
        DEBUG["sig_v"] = sig
        self._set_thr(self.sig)
        self.hold = k + 2 * self.det_len
        self.ident = True

    def _set_thr(self, sig: float) -> None:
        self.thr2 = (float(_P["det_k"]) * sig) ** 2 * self.det_len

    def _gains(self, om: np.ndarray) -> np.ndarray:
        """-1 / (F G) at the tone frequencies: v-domain phasor -> command phasor."""
        z = np.exp(1j * om)
        g = (self.g_res[None, :] / (z[:, None] - self.g_lam[None, :])).sum(axis=1) / (z * z)
        f = np.prod((z[:, None] - self.f_num[None, :]) / (z[:, None] - self.f_den[None, :]), axis=1)
        return -1.0 / (f * g)

    # ----- tone fitting -----
    def _gn(self, s: int, k: int, om: np.ndarray, ab, iters: int, final: bool):
        """Gauss-Newton fit of sum a_i cos(om_i t) + b_i sin(om_i t) on v[s..k], t = 0 at sample k."""
        nwin = k - s + 1
        vv = self.vb[s:k + 1]
        t = _T[_NW - nwin:]
        tn = t * (1.0 / nwin)
        n = len(om)
        lim = 0.3 * 2.0 * math.pi / nwin
        try:
            for _ in range(iters):
                ph = np.outer(t, om)
                cm, sm = np.cos(ph), np.sin(ph)
                if ab is None:
                    phi = np.concatenate((cm, sm), axis=1)
                    ab = np.linalg.solve(phi.T @ phi, phi.T @ vv)
                jm = np.concatenate((cm, sm, tn[:, None] * (cm * ab[n:] - sm * ab[:n])), axis=1)
                jtj = jm.T @ jm
                jtj[np.diag_indices_from(jtj)] += 1e-10 * float(np.trace(jtj))
                x = np.linalg.solve(jtj, jm.T @ vv)
                ab = x[:2 * n]
                om = om + np.clip(x[2 * n:] / nwin, -lim, lim)
            if final:
                ph = np.outer(t, om)
                phi = np.concatenate((np.cos(ph), np.sin(ph)), axis=1)
                ab = np.linalg.solve(phi.T @ phi, phi.T @ vv)
                r = vv - phi @ ab
            else:
                r = vv - jm @ x
        except np.linalg.LinAlgError:
            return om, ab, math.inf, None
        rms = math.sqrt(float(r @ r) / nwin)
        srt = np.sort(om)
        if not math.isfinite(rms) or srt[0] < _OM_LO or srt[-1] > _OM_HI or (n > 1 and float(np.min(np.diff(srt))) < 4.0 / _W2F):
            return om, ab, math.inf, r
        return om, ab, rms, r

    def _peak(self, r: np.ndarray, om: np.ndarray):
        """Largest periodogram line of r in the band, away from the tones already fitted."""
        spec = np.abs(np.fft.rfft(r, _NFFT))[_K0:_K1]
        for o in om:
            spec = np.where(np.abs(_FB - o * _W2F) < 5.0, 0.0, spec)
        j = int(np.argmax(spec))
        return _FB[j] / _W2F, 2.0 * float(spec[j]) / len(r)

    def _greedy(self, s: int, k: int, n_known):
        """Add tones one by one at the residual's periodogram peak, refitting all jointly."""
        nwin = k - s + 1
        r = self.vb[s:k + 1]
        om, ab, rms = np.zeros(0), None, math.inf
        for i in range(n_known or 3):
            o_new, amp = self._peak(r, om)
            if n_known is None and i > 0:
                amax = float(np.max(np.hypot(ab[:i], ab[i:])))
                if amp < max(12.0 * self.sig / math.sqrt(nwin), 0.05 * amax, _A_MIN):
                    break
            om2, ab2, rms2, r2 = self._gn(s, k, np.append(om, o_new), None, 6, True)
            if not math.isfinite(rms2):
                break
            om, ab, rms, r = om2, ab2, rms2, r2
        return om, ab, rms

    def _set_model(self, k: int, om: np.ndarray, ab: np.ndarray) -> None:
        n = len(om)
        self.n, self.om = n, om
        zc = ab[:n] - 1j * ab[n:]
        uc = self._gains(om) * zc
        self.zc = [complex(z) for z in zc]
        self.uc = [complex(z) for z in uc]
        self.rot = [complex(math.cos(o), math.sin(o)) for o in om]

    def _acquire(self, k: int) -> None:
        s = self.ws
        om, ab, rms = None, None, math.inf
        if self.n > 0:
            om, ab, rms, _ = self._gn(s, k, self.om.copy(), None, 8, True)
        if rms > 3.0 * self.sig:
            om_g, ab_g, rms_g = self._greedy(s, k, self.n if self.n > 0 else None)
            if rms_g < rms:
                om, ab, rms = om_g, ab_g, rms_g
        DEBUG["events"].append(("acq", k, None if om is None else (om * _W2F).round(3).tolist(), rms))
        if om is not None and math.isfinite(rms) and len(om) > 0:
            self._set_model(k, om, ab)
        else:
            self.n, self.om, self.zc, self.uc, self.rot = 0, np.zeros(0), [], [], []
        self.pending = False
        self.next_fit = k + 4
        self.checks = [200, 400, 800]

    def _track(self, k: int) -> None:
        s = max(self.ws, k - int(_P["n_max"]) + 1)
        nwin = k - s + 1
        n = self.n
        ab = np.array([z.real for z in self.zc] + [-z.imag for z in self.zc])
        check = bool(self.checks) and nwin >= self.checks[0]
        om, ab, rms, r = self._gn(s, k, self.om, ab, 2 if nwin < 400 else 1, check)
        if check:
            lvl = self.checks.pop(0)
            if math.isfinite(rms):
                amps = np.hypot(ab[:n], ab[n:])
                keep = amps >= 0.5 * max(0.05 * float(amps.max()), _A_MIN)
                if n > 1 and not bool(np.all(keep)):
                    om2, ab2, rms2, r2 = self._gn(s, k, om[keep], None, 3, True)
                    if math.isfinite(rms2):
                        om, ab, rms, r, n = om2, ab2, rms2, r2, len(om2)
                        amps = np.hypot(ab[:n], ab[n:])
                        DEBUG["events"].append(("prune", k, (om * _W2F).round(3).tolist(), rms))
                o_new, amp = self._peak(r, om)
                if n < 3 and amp > max(12.0 * self.sig / math.sqrt(nwin), 0.05 * float(amps.max()), _A_MIN):
                    om2, ab2, rms2, _ = self._gn(s, k, np.append(om, o_new), None, 6, True)
                    if rms2 < 0.9 * rms:
                        om, ab, rms, n = om2, ab2, rms2, n + 1
                        DEBUG["events"].append(("add", k, (om * _W2F).round(3).tolist(), rms))
            if (not math.isfinite(rms)) or rms > 3.0 * self.sig:
                om_g, ab_g, rms_g = self._greedy(s, k, None)
                DEBUG["events"].append(("reinit", k, (om_g * _W2F).round(3).tolist(), rms_g, rms))
                if rms_g < rms:
                    om, ab, rms = om_g, ab_g, rms_g
            if lvl >= 800 and math.isfinite(rms):
                self._set_thr(max(self.sig, rms))
                DEBUG["events"].append(("steady", k, (om * _W2F).round(4).tolist(), rms))
        if math.isfinite(rms):
            self._set_model(k, om, ab)
            self.next_fit = k + (4 if nwin < 240 else 8)
        else:                                   # degenerate fit: keep the last model, re-initialise soon
            self.checks.insert(0, 0)
            self.next_fit = k + 40
        self.last_rms = rms

    # ----- one sample -----
    def step(self, obs: dict) -> float:
        k = int(obs["step"])
        if k >= _NBUF:
            return 0.0
        y = float(obs["y"])
        if k > 0:
            self.ub[k + 2] = float(obs["last_action"])
        self.yb[k] = y
        if not self.ident:
            if k == self.k_id:
                self._setup(k)
                return 0.0
            return float(self.probe[k]) if k < self.n_probe else 0.0

        # disturbance observer and equaliser
        x = self.A @ self.x + self.B * self.ub[k]
        self.x = x
        v = y - float(self.C @ x)
        for (b0, b1, b2, a1, a2), z in zip(self.sos, self.sz):
            out = b0 * v + z[0]
            z[0] = b1 * v - a1 * out + z[1]
            z[1] = b2 * v - a2 * out
            v = out
        self.vb[k] = v

        # advance the fitted tones by one sample; prediction error of v
        vhat = 0.0
        u = 0.0
        zc, uc, rot = self.zc, self.uc, self.rot
        for i in range(self.n):
            z = zc[i] * rot[i]
            zc[i] = z
            vhat += z.real
            q = uc[i] * rot[i]
            uc[i] = q
            u += q.real
        e2 = (v - vhat) ** 2
        j = k % self.det_len
        self.esum += e2 - self.ering[j]
        self.ering[j] = e2

        refit = False
        try:
            if self.pending:
                if k - self.ws + 1 >= int(_P["n_min"]):
                    refit = True
                    self._acquire(k)
            elif k >= self.hold and self.esum > self.thr2:
                self.pending = True
                self.ws = k + int(_P["guard"])
                self.hold = k + int(_P["holdoff"])
                DEBUG["events"].append(("detect", k))
            elif self.n > 0 and k >= self.next_fit:
                refit = True
                self._track(k)
        except Exception as exc:                # a numerical accident must not stop the run: keep the current tones
            DEBUG["events"].append(("error", k, repr(exc)))
            self.pending = False
            self.next_fit = k + 40
        if refit:
            u = 0.0
            for q in self.uc:
                u += q.real
        if not math.isfinite(u):
            u = 0.0
        u_max = float(obs["u_max"])
        return min(max(u, -u_max), u_max)


_CTL = _Controller()


def controller(obs: dict) -> float:
    if int(obs["step"]) == 0:
        _CTL.reset()
    return _CTL.step(obs)
