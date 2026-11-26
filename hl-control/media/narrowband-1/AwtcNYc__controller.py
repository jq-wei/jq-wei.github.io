"""Adaptive rejection of one tone of unknown frequency (active vibration control, level 1).

Control law (internal-model / Youla-Kucera structure with a phase-locked tone model)
-----------------------------------------------------------------------------------
1. Disturbance reconstruction.  A copy of the nominal secondary path (order 17 + 2 samples of delay) is
   driven by the commands actually applied; its output is subtracted from the measured residual:
   ``d[k] = y[k] - G_nom(q) u[k]``.  With a perfect model ``d`` is the open-loop residual, so estimating
   the tone in ``d`` is an open-loop problem; the real structure (modal frequencies +-2 %) only closes a
   weak loop through the model error.
2. Acquisition.  The tone is detected when ``|d|`` leaves the noise floor (``detect_v``).  The command stays
   at zero for ``n_acq`` samples; the peak of a zero-padded, Hann-weighted periodogram of that record gives
   the first frequency estimate and a least-squares fit over its last ``n_fit`` samples the first complex
   amplitude.
3. Tracking.  A second-order phase-locked loop follows ``d[k] ~ Re(c exp(j phi[k]))``: the innovation,
   demodulated by the oscillator, corrects the complex amplitude ``c`` (gain ``2 a``) and the frequency
   ``w`` (gain ``a^2``), i.e. a double pole of bandwidth ``a`` [Hz]; the integral action on the frequency
   makes the steady-state error at the tone zero.
4. Cancellation.  ``u[k] = -Re(c_u exp(j phi[k]) / G_nom(e^{j w}))`` where ``c_u`` is ``c`` passed through a
   second-order low-pass of bandwidth ``b`` [Hz] in the rotating frame.  The low-pass makes the complementary
   sensitivity roll off steeply away from the tone, which keeps the loop through the model error small at
   the uncertain, lightly damped modes next to the band (45 Hz, 109 Hz).
5. Gain schedule.  ``a`` and ``b`` start at fast values for a quick transient, are held for ``t_hold``
   seconds and then decay (time constant ``t_decay``) to (``a_slow_hz``, ``b_slow_hz``): a narrow notch in
   steady state, hence almost no amplification elsewhere (waterbed) and little noise.  The fast values depend
   on the tracked frequency:
   - near the band edges (``a_fast_hz``, ``b_fast_hz``): a slow low-pass, because the uncertain 45 Hz and
     109 Hz modes are only a few Hz away and the roll-off has to protect the loop there;
   - in the middle of the band (``a_mid_hz``, ``b_mid_hz``, blended in between ``mid_lo_hz`` and
     ``mid_hi_hz`` over +-``mid_blend_hz``): the 72 Hz mode sits inside the band, a +-2 % shift of it changes
     the secondary path at the tone by up to 47 % / 26 deg, and that model error slows the loop down; a
     low-pass much faster than the PLL (``b ~ 7 a``) keeps this loop well damped, and the edge modes are far.

Why these numbers: the transient test compares the residual energy 2..5 s after the onset with the last 3 s,
so the tone must be below the 0.21 mV noise floor (60 dB down for the strongest tones) 2 s after it appears,
which needs effective time constants of about 0.1-0.15 s until then; the amplification limit is met by the
narrow steady-state notch.
"""

from __future__ import annotations

import cmath
import math

import numpy as np

from hl_control.envs.narrowband import frequency_response, nominal_model

FS = 800.0

PARAMS = {
    "detect_v": 2.0e-3,    # |d| above this marks the onset of the tone [V] (noise floor: 0.21 mV RMS)
    "n_acq": 160,          # samples observed with zero command before the first estimate
    "n_fit": 96,           # last samples of that record used for the amplitude fit
    "a_fast_hz": 1.6,      # PLL bandwidth (double pole) right after acquisition, tones near the band edges [Hz]
    "b_fast_hz": 3.0,      # bandwidth of the low-pass on the control amplitude right after acquisition, edges [Hz]
    "a_mid_hz": 2.0,       # the same two bandwidths for tones in the middle of the band (around the 72 Hz mode,
    "b_mid_hz": 14.0,      # where the model error slows the loop and the uncertain 45/109 Hz modes are far away)
    "mid_lo_hz": 62.0,     # centre of the blend from edge to mid-band gains at the low end [Hz]
    "mid_hi_hz": 84.0,     # ... and at the high end [Hz]
    "mid_blend_hz": 3.0,   # half-width of each blend [Hz]
    "a_slow_hz": 0.35,     # PLL bandwidth in steady state [Hz]
    "b_slow_hz": 1.0,      # control low-pass bandwidth in steady state [Hz]
    "t_hold": 1.3,         # time the fast gains are held after acquisition [s]
    "t_decay": 0.35,       # time constant of the decay from fast to slow gains [s]
}

_MODEL = nominal_model()
_A, _B, _C, _ = _MODEL["secondary"]
_DELAY = int(_MODEL["delay_steps"])
_F0, _DF = 40.0, 0.02
_FGRID = np.arange(_F0, 105.0 + 1e-9, _DF)
_GINV = (1.0 / frequency_response(_MODEL["secondary"], _FGRID, FS, _DELAY)).tolist()
_W_MIN, _W_MAX = 2 * math.pi * 47.0 / FS, 2 * math.pi * 98.0 / FS
_NFFT = 16384


class _State:
    def __init__(self) -> None:
        self.x = np.zeros(_A.shape[0])     # state of the nominal secondary path
        self.uh = [0.0] * (_DELAY + 1)     # last commands: u[k-1], u[k-2], u[k-3]
        self.mode = 0                      # 0 waiting for the tone, 1 acquiring, 2 tracking
        self.buf: list[float] = []
        self.phi = 0.0                     # oscillator phase at the current sample [rad]
        self.w = 0.0                       # frequency estimate [rad/sample]
        self.c = 0j                        # complex amplitude of the tone in d
        self.c1 = 0j                       # low-pass states of the control amplitude
        self.c2 = 0j
        self.n = 0                         # samples since the start of tracking


_S = _State()


def _ginv(w: float) -> complex:
    """1 / G_nom(e^{jw}), linearly interpolated on the precomputed grid."""
    pos = (w * FS / (2 * math.pi) - _F0) / _DF
    i = int(pos)
    if i < 0:
        return _GINV[0]
    if i >= len(_GINV) - 1:
        return _GINV[-1]
    t = pos - i
    return _GINV[i] * (1.0 - t) + _GINV[i + 1] * t


def _acquire(s: _State) -> None:
    """First estimate of the frequency (periodogram peak) and of the complex amplitude (least squares)."""
    buf = np.asarray(s.buf)
    n = len(buf)
    spec = np.abs(np.fft.rfft(buf * np.hanning(n), _NFFT))
    lo, hi = int(47.0 / FS * _NFFT), int(98.0 / FS * _NFFT)
    i = lo + int(np.argmax(spec[lo:hi]))
    y0, y1, y2 = math.log(spec[i - 1] + 1e-30), math.log(spec[i] + 1e-30), math.log(spec[i + 1] + 1e-30)
    den = y0 - 2 * y1 + y2
    shift = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
    w = 2 * math.pi * (i + max(-1.0, min(1.0, shift))) / _NFFT
    m = int(PARAMS["n_fit"])
    idx = np.arange(-m + 1, 1)                     # phase zero at the current sample
    basis = np.column_stack([np.cos(w * idx), np.sin(w * idx)])
    ab, *_ = np.linalg.lstsq(basis, buf[-m:], rcond=None)
    s.w = min(_W_MAX, max(_W_MIN, w))
    s.c = complex(ab[0], -ab[1])                   # d[k] ~ Re(c exp(j phi)), phi = 0 now
    s.phi = 0.0
    s.c1 = s.c2 = 0j
    s.n = 0
    s.mode = 2


def controller(obs: dict) -> float:
    global _S
    if obs["step"] == 0:
        _S = _State()
    s = _S
    u_max = obs["u_max"]

    # 1. disturbance reconstruction with the nominal model: x[k] = A x[k-1] + B u[k-1-delay]
    if obs["step"] > 0:
        s.x = _A @ s.x + _B * s.uh[_DELAY]
    d = obs["y"] - float(_C @ s.x)

    u = 0.0
    if s.mode == 0:
        if abs(d) > PARAMS["detect_v"]:
            s.mode = 1
            s.buf = [d]
    elif s.mode == 1:
        s.buf.append(d)
        if len(s.buf) >= PARAMS["n_acq"]:
            _acquire(s)
    else:
        # gain schedule
        t = s.n / FS
        fade = 1.0 if t <= PARAMS["t_hold"] else math.exp(-(t - PARAMS["t_hold"]) / PARAMS["t_decay"])
        f_hz = s.w * FS / (2 * math.pi)
        hw = PARAMS["mid_blend_hz"]
        mid = (max(0.0, min(1.0, 0.5 + (f_hz - PARAMS["mid_lo_hz"]) / (2 * hw)))
               * max(0.0, min(1.0, 0.5 + (PARAMS["mid_hi_hz"] - f_hz) / (2 * hw))))
        a_fast = PARAMS["a_fast_hz"] + (PARAMS["a_mid_hz"] - PARAMS["a_fast_hz"]) * mid
        b_fast = PARAMS["b_fast_hz"] + (PARAMS["b_mid_hz"] - PARAMS["b_fast_hz"]) * mid
        a = 2 * math.pi / FS * (PARAMS["a_slow_hz"] + (a_fast - PARAMS["a_slow_hz"]) * fade)
        b = 2 * math.pi / FS * (PARAMS["b_slow_hz"] + (b_fast - PARAMS["b_slow_hz"]) * fade)
        # 3. phase-locked loop on d
        rot = cmath.exp(1j * s.phi)
        e = d - (s.c * rot).real
        eps = 2.0 * e * rot.conjugate()
        mag2 = s.c.real * s.c.real + s.c.imag * s.c.imag
        phase_err = (eps * s.c.conjugate()).imag / (mag2 + 1e-12)
        s.c += 2.0 * a * eps
        s.w = min(_W_MAX, max(_W_MIN, s.w + a * a * max(-1.0, min(1.0, phase_err))))
        # 4. cancellation through the inverse of the nominal response at the tracked frequency
        s.c1 += b * (s.c - s.c1)
        s.c2 += b * (s.c1 - s.c2)
        u = -(s.c2 * rot * _ginv(s.w)).real
        s.phi += s.w
        if s.phi > math.pi:
            s.phi -= 2 * math.pi
        s.n += 1

    u = max(-u_max, min(u_max, u))
    s.uh = [u] + s.uh[:-1]
    return u
