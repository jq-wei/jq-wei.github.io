"""Dual-stage HDD track following: one fixed multi-model LQG (H2) controller for all nine plant cases.

Control law (linear, time-invariant, one matrix-vector product per PES sample):

    xi[k+1]              = Ac xi[k] + Bc pes[k]
    [u_vcm0, u_pzt0,
     u_vcm1, u_pzt1][k]  = Cc xi[k] + Dc pes[k]        (the two rows of the action: both halves of the sample)

(Ac, Bc, Cc, Dc) is a Kalman filter (current estimator) followed by a state feedback, both computed when this file is
imported, from the benchmark's own model, exactly discretised on the Ts/20 grid and lifted to the PES rate:

* Design model. A rigid body, the VCM resonances and the PZT resonances of *all three temperatures* side by side
  (driven by the same commands), the rotational-vibration shaping filter at the VCM input, and the 13 fan resonances
  at the head position (the Kalman filter's internal model of them acts as a bank of peak filters at 2.2 ... 9.2 kHz;
  the repeatable run-out is the measurement noise).
* State feedback. Minimises the head-position variance on the Ts/20 grid (so intersample ripple and the excitation of
  resonances beyond the Nyquist frequency by the multi-rate hold count), averaged over the three temperature
  variants: at a frequency where the variants disagree (a resonance of one of them) no command suits all three, and
  the optimum backs off, which is a notch at every resonance of every temperature. This is what makes one controller
  stable on LT, RT and HT.
* Kalman filter. Measures the average of the three variants' head positions.

Gains (GAINS):
* r_meas   variance [nm^2] of the measurement noise the Kalman filter assumes. The run-out itself is 0.26 nm^2; a
           larger value slows the estimator, lowers the loop gain and the bandwidth and buys gain/phase margin and
           a lower sensitivity peak at the price of 3-sigma. This is the main robustness/performance knob.
* rho_pzt  weight on the squared PZT command [nm^2] in the cost: limits the PZT stroke and high-frequency activity.
* rho_vcm  weight on the squared (scaled) VCM command: shifts work from the VCM to the PZT.
* w_lt, w_rt, w_ht  weights of the three temperature variants in the cost and in the measurement model.

With these values the nine cases give a worst-case 3-sigma of about 6.7 % of the track pitch, gain margin >= 9 dB,
phase margin >= 40 deg and a sensitivity peak <= 4.7 dB (as measured by the evaluator's certificate).

Scope. The notches sit at the resonances of the benchmark's three temperatures, the only ones its nine cases contain.
At temperatures in between (not part of the benchmark) the loop is still stable and as accurate in the model, but the
margins above are not kept there (PZT resonances near 40 kHz, aliased to 8-12 kHz, then fall between the notches).

Implementation notes. The design (about 0.5 s) runs once at import; a call is one 173 x 170 matrix-vector product,
done with einsum rather than BLAS, and the BLAS thread pool is limited to one thread: on a many-core machine the
worker threads of a threaded BLAS spin between calls and would be charged to the controller's processor-time budget
(for the same reason the import ends with a short sleep).
"""

import ctypes
import time

import numpy as np

from hl_control.envs import hdd

GAINS = {"r_meas": 3.0, "rho_pzt": 0.1, "rho_vcm": 0.0, "w_lt": 1.0, "w_rt": 1.0, "w_ht": 1.0}

_NM = 1e9            # the design works in nanometres
_SV = 200.0          # VCM command scaling (one scaled unit moves the head about 1.5 nm in a sample)
_TEMPS = ("LT", "RT", "HT")
_N_FINE, _N_HALF = hdd.MR_P, hdd.HALF


def _section_ss(b, a):
    """Observable canonical form of a strictly proper discrete section (b, a); the output is the first state."""
    n = len(a) - 1
    A = np.zeros((n, n))
    A[:, 0] = -np.asarray(a[1:])
    for i in range(n - 1):
        A[i, i + 1] = 1.0
    B = np.zeros(n)
    B[:len(b) - 1] = np.asarray(b[1:])
    return A, B


def _fine_model():
    """Design model on the Ts/20 grid. State: rigid body (2), VCM resonances of LT/RT/HT (30 each), PZT modes of
    LT/RT/HT (16 each), rotational-vibration filter (3), fan resonances (26). Inputs: scaled VCM command, PZT command
    [nm]; noise: the 14 white streams. Cy[t] is the head position [nm] that temperature variant t predicts."""
    rv, fans = hdd._disturbance_filters()
    dist = [_section_ss(b, a) for b, a in rv] + [_section_ss(b, a) for b, a in fans]
    nt = len(_TEMPS)
    n = 2 + 46 * nt + sum(A.shape[0] for A, _ in dist)
    Af, Bf, Gf, Cy = np.zeros((n, n)), np.zeros((n, 2)), np.zeros((n, hdd.N_NOISE)), np.zeros((nt, n))
    b_vcm = np.zeros(n)                              # everything the VCM input (and the rotational vibration) drives
    for t, temp in enumerate(_TEMPS):
        av, bv, cv = hdd._fine_model("vcm", temp)
        ap, bp, cp = hdd._fine_model("pzt", temp)
        Af[:2, :2], b_vcm[:2], Cy[t, :2] = av[:2, :2], bv[:2], cv[:2]          # rigid body: the same at every temperature
        i0 = 2 + 30 * t
        Af[i0:i0 + 30, i0:i0 + 30], b_vcm[i0:i0 + 30], Cy[t, i0:i0 + 30] = av[2:, 2:], bv[2:], cv[2:]
        j0 = 2 + 30 * nt + 16 * t
        Af[j0:j0 + 16, j0:j0 + 16], Bf[j0:j0 + 16, 1], Cy[t, j0:j0 + 16] = ap, bp, cp
    Bf[:, 0] = b_vcm * _SV
    pos = 2 + 46 * nt
    for j, (A, B) in enumerate(dist):
        m = A.shape[0]
        Af[pos:pos + m, pos:pos + m] = A
        Gf[pos:pos + m, 0 if j < len(rv) else j - len(rv) + 1] = B * _NM
        if j < len(rv):
            Af[:, pos] += b_vcm                      # rotational vibration: a disturbance of the VCM input
        else:
            Cy[:, pos] = 1.0                         # fan vibration: added to the head position
        pos += m
    scale = np.ones(n)
    scale[1] = 1.0 / hdd.TS                          # rigid-body velocity in nm per sample (conditioning)
    return Af * scale[None, :] / scale[:, None], Bf / scale[:, None], Gf / scale[:, None], Cy * scale[None, :]


def _lift(Af, Bf, Gf, Cy):
    """Lift to the PES rate: x+ = A x + B [u_vcm0, u_pzt0, u_vcm1, u_pzt1] + w, cov(w) = Q, and for each temperature
    variant the 20 fine-grid head positions of the sample as a matrix acting on [x; u]."""
    n = Af.shape[0]
    P = [np.eye(n)]
    for _ in range(_N_FINE):
        P.append(P[-1] @ Af)
    cum = [np.zeros((n, 2))]
    for i in range(_N_HALF):
        cum.append(cum[-1] + P[i] @ Bf)
    gam = []
    for i in range(_N_FINE + 1):
        g = np.zeros((n, 4))
        if i <= _N_HALF:
            g[:, :2] = cum[i]
        else:
            g[:, :2], g[:, 2:] = P[i - _N_HALF] @ cum[_N_HALF], cum[i - _N_HALF]
        gam.append(g)
    W = Gf @ Gf.T / 12.0                             # uniform noise on [-0.5, 0.5): variance 1/12
    Q = np.zeros((n, n))
    for _ in range(_N_FINE):
        Q = Af @ Q @ Af.T + W
    M = [np.array([np.concatenate([c @ P[i], c @ gam[i]]) for i in range(_N_FINE)]) for c in Cy]
    return P[_N_FINE], gam[_N_FINE], Q, M


def _riccati(A, G, Q, iters=60, tol=1e-13):
    """Stabilising solution of X = A' X (I + G X)^-1 A + Q by the structure-preserving doubling algorithm."""
    eye = np.eye(A.shape[0])
    for _ in range(iters):
        W = eye + G @ Q
        WA, WG = np.linalg.solve(W, A), np.linalg.solve(W, G)
        Qn = Q + A.T @ Q @ WA
        Gn = G + A @ WG @ A.T
        A = A @ WA
        done = np.linalg.norm(Qn - Q) <= tol * np.linalg.norm(Qn)
        G, Q = 0.5 * (Gn + Gn.T), 0.5 * (Qn + Qn.T)
        if done:
            break
    return Q


def design(params=None):
    """The controller (Ac, Bc, Cc, Dc): pes [nm] -> [scaled u_vcm0, u_pzt0 [nm], scaled u_vcm1, u_pzt1 [nm]]."""
    p = dict(GAINS, **(params or {}))
    Af, Bf, Gf, Cy = _fine_model()
    A, B, Q, M = _lift(Af, Bf, Gf, Cy)
    n = A.shape[0]
    w = np.array([p["w_lt"], p["w_rt"], p["w_ht"]], dtype=float)
    w = w / w.sum()
    # state feedback: weighted sum over the temperature variants of the fine-grid head-position variance
    H = sum(wt * Mt.T @ Mt for wt, Mt in zip(w, M))
    Qx, S, R = H[:n, :n], H[:n, n:], H[n:, n:] + np.diag([p["rho_vcm"], p["rho_pzt"], p["rho_vcm"], p["rho_pzt"]])
    Ri = np.linalg.inv(R)
    X = _riccati(A - B @ Ri @ S.T, B @ Ri @ B.T, Qx - S @ Ri @ S.T)
    K = np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A + S.T)
    # Kalman filter on pes = -(weighted mean of the variants' head positions) + noise of variance r_meas
    Cm = -(w @ Cy)[None, :]
    Pk = _riccati(A.T, Cm.T @ Cm / p["r_meas"], Q)
    Lf = (Pk @ Cm.T / (Cm @ Pk @ Cm.T + p["r_meas"])).ravel()
    AK = A - B @ K
    ILC = np.eye(n) - np.outer(Lf, Cm)
    return AK @ ILC, AK @ Lf, -K @ ILC, -K @ Lf


def _single_thread_blas():
    """Best effort: limit the OpenBLAS pools that numpy and scipy have loaded to one thread. True if every pool found
    took the setting (a single-processor machine has no worker threads anyway)."""
    done = True
    try:
        with open("/proc/self/maps") as handle:
            paths = sorted({line.split()[-1] for line in handle if "openblas" in line})
        for path in paths:
            lib = ctypes.CDLL(path)
            names = [pre + "openblas_set_num_threads" + suf for pre in ("scipy_", "") for suf in ("64_", "", "_64", "64")]
            setters = [getattr(lib, name) for name in names if hasattr(lib, name)]
            for setter in setters[:1]:
                setter(ctypes.c_int(1))
            done = done and bool(setters)
    except Exception:
        return False
    return done


def _build():
    quiet = _single_thread_blas()
    Ac, Bc, Cc, Dc = design()
    # BLAS worker threads spin for about 0.1 s after they are created (when numpy and scipy are loaded) and after
    # every threaded call; wait here, where it is free, so that none of it is charged to the first rollout
    time.sleep(0.3 if quiet else 0.6)
    n = Ac.shape[0]
    out_scale = np.array([_SV, 1.0, _SV, 1.0]) / _NM                 # back to the plant's units
    step = np.zeros((n + 4, n + 1))                                 # [xi+; u] = step @ [xi; pes]
    step[:n, :n], step[:n, n] = Ac, Bc * _NM
    step[n:, :n], step[n:, n] = Cc * out_scale[:, None], Dc * out_scale * _NM
    return np.ascontiguousarray(step), n


_STEP, _N = _build()
_z = np.zeros(_N + 1)                                               # [xi; pes]


def controller(obs: dict) -> np.ndarray:
    z = _z
    if obs["step"] == 0:
        z[:] = 0.0
    z[_N] = obs["pes"]
    out = np.einsum("ij,j->i", _STEP, z)                            # no BLAS: single-threaded on any machine
    z[:_N] = out[:_N]
    return out[_N:].reshape(2, 2)
