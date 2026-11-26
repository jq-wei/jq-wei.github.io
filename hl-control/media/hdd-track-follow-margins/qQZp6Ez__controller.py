"""Dual-stage HDD track following: one fixed multi-rate LQG (H2-optimal) controller, detuned for robustness.

Control law (linear, time-invariant, the same for all nine plant cases; 169 states):

    e[k]      = pes[k] + H x[k]            innovation (the model predicts pes = -H x)
    xhat[k]   = x[k] + Lf e[k]             Kalman measurement update (no computation delay: current estimator)
    u[k]      = -K xhat[k]                 the four commands of the sample: [u_vcm, u_pzt] for each half period
    x[k+1]    = A xhat[k] + B u[k]         one-sample prediction with the design model

which is run as one matrix product per sample, [x[k+1]; u[k]] = _STEP [x[k]; pes[k]].

The design model is the benchmark's own, lifted exactly to the PES rate (two command holds per sample, head
position on the Ts/20 grid): the room-temperature VCM (rigid body + 15 resonances) and PZT (8 resonances), plus the
shaping filters of the disturbances (13 fan resonances acting on the head position, the rotational-vibration
filter acting on the VCM input). The repeatable run-out is treated as white measurement noise of its RMS (0.51 nm):
the head should not follow it.

K minimises  mean over the fine grid of  y^2 + BETA * sum_T (dy_T)^2  +  R_VCM u_vcm^2 + R_PZT u_pzt^2,
where dy_T = (P_T - P_RT) u is what the plant at temperature T (LT, HT) would add to the head position: "shadow"
copies of the resonances at LT and HT run inside the controller (driven by the commands only), so that it avoids
putting command energy where a resonance may sit at another temperature (the PZT modes near 40 kHz alias to
8-13 kHz depending on temperature). Lf is the Kalman gain of the room-temperature model.

Gains (signals in nm; they only set trade-offs, the loop is linear):
    R_VCM, R_PZT  command penalties: limit the high-frequency command activity (and the PZT stroke);
    RHO           measurement-noise variance used by the Kalman filter, in units of the run-out variance: the
                  bandwidth knob. Larger = less aggressive loop = larger stability margins / lower |S| peak;
    BETA          weight of the temperature-mismatch outputs: robustness of the margins against resonance shifts
                  (with BETA = 0 the LT cases lose their phase margin unless RHO is raised to ~50, at 9.5 %).

Measured with the task's evaluator (seeds 0-14): worst-case 3 sigma 6.3-6.4 % of the track pitch (incumbent
level 14.79 %), PZT displacement below 18 nm; loop margins over the nine cases as the certificate measures them:
gain margin >= 8.2 dB, phase margin >= 39.5 deg, sensitivity peak <= 4.5 dB.
"""

import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from hl_control.envs import hdd

GAINS = {"R_VCM": 0.01, "R_PZT": 0.01, "RHO": 7.0, "BETA": 3.0}

_TS = hdd.TS                      # PES sample period
_NF = 20                          # fine steps per PES sample (the grid the head position is scored on)
_DT = _TS / _NF
_NM = 1e9                         # disturbances are modelled in nm
_RRO_RMS = 0.05 * math.sqrt(209 / 2.0)   # nm


def _zoh(gain, omega, zeta, dt):
    """Exact zero-order-hold discretisation of gain / (s^2 + 2 zeta omega s + omega^2); output = first state."""
    if omega == 0.0:              # rigid body; second state = velocity * Ts
        return np.array([[1.0, dt / _TS], [0.0, 1.0]]), np.array([0.5 * gain * dt * dt, gain * dt * _TS])
    m = np.zeros((3, 3))
    m[0, 1], m[1, 0], m[1, 1], m[1, 2] = omega, -omega, -2.0 * zeta * omega, gain / omega
    e = expm(m * dt)
    return e[:2, :2], e[:2, 2]


def _blocks(modes):
    """Block-diagonal fine-rate model (A, b, c) of a sum of second-order modes (gain, omega, zeta)."""
    n = 2 * len(modes)
    a, b, c = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    for i, (gain, omega, zeta) in enumerate(modes):
        a[2 * i:2 * i + 2, 2 * i:2 * i + 2], b[2 * i:2 * i + 2] = _zoh(gain, omega, zeta, _DT)
        c[2 * i] = 1.0
    return a, b, c


def _disturbance_model():
    """Fine-rate shaping filters: x+ = A x + B w (14 white streams); d_fan = c_fan x, d_rv = c_rv x."""
    fans = [(k, 2.0 * math.pi * f, z) for k, f, z in hdd.FAN_MODES]
    af, _, cf = _blocks(fans)
    nf = af.shape[0]
    n = nf + 3
    a, b, c_fan, c_rv = np.zeros((n, n)), np.zeros((n, 14)), np.zeros(n), np.zeros(n)
    a[:nf, :nf], c_fan[:nf] = af, cf
    for j, mode in enumerate(fans):
        b[2 * j:2 * j + 2, 1 + j] = _zoh(*mode, _DT)[1]
    # rotational vibration: 3e-10 (s + 2 pi 50)(s^2 + 40 w2 s + w2^2) / ((s + 2 pi 3)(s^2 + 0.2 w1 s + w1^2)) less its
    # direct term, as a real pole plus a resonance in parallel
    gain, p1, zeta1 = 3e-10, 2.0 * math.pi * 3.0, 0.1
    w1, w2 = 2.0 * math.pi * 250.0, 2.0 * math.pi * 2000.0
    quad = np.array([1.0, 2.0 * zeta1 * w1, w1 * w1])
    num = gain * np.polymul([1.0, 2.0 * math.pi * 50.0], [1.0, 2.0 * 20.0 * w2, w2 * w2])
    proper = np.polysub(num, gain * np.polymul([1.0, p1], quad))
    residue = np.polyval(proper, -p1) / np.polyval(quad, -p1)
    c1, c0 = np.polydiv(np.polysub(proper, residue * quad), [1.0, p1])[0][-2:]
    pole = math.exp(-p1 * _DT)
    a[nf, nf], b[nf, 0], c_rv[nf] = pole, residue * (1.0 - pole) / p1, 1.0
    m = np.zeros((3, 3))
    m[0, 1], m[1, 0], m[1, 1] = w1, -w1, -2.0 * zeta1 * w1
    m[0, 2], m[1, 2] = c1, (c0 - 2.0 * zeta1 * w1 * c1) / w1
    e = expm(m * _DT)
    a[nf + 1:, nf + 1:], b[nf + 1:, 0], c_rv[nf + 1] = e[:2, :2], e[:2, 2], 1.0
    return a, _NM * b, c_fan, c_rv


def design(gains=None):
    """Returns the controller (Ak, Bk, Ck, Dk): x+ = Ak x + Bk pes, u = Ck x + Dk pes (u = [u_v0, u_p0, u_v1, u_p1])."""
    g = dict(GAINS, **(gains or {}))
    beta = float(g["BETA"])
    vcm = {t: hdd.actuator_modes("vcm", t) for t in ("RT", "LT", "HT")}
    pzt = {t: hdd.actuator_modes("pzt", t) for t in ("RT", "LT", "HT")}
    av, bv, cv = _blocks(vcm["RT"])
    ap, bp, cp = _blocks(pzt["RT"])
    ad, bd, c_fan, c_rv = _disturbance_model()
    nv, npz, nd = av.shape[0], ap.shape[0], ad.shape[0]
    n0 = nv + npz + nd                                    # states of the nominal model (Kalman filter)
    # fine-rate system: x+ = F x + Gu [u_vcm, u_pzt] + Gw w ; outputs Hm x (row 0: head position, then mismatches)
    parts = [(av, np.column_stack([bv, 0 * bv])), (ap, np.column_stack([0 * bp, bp])), (ad, np.zeros((nd, 2)))]
    rows = [[cv, cp, c_fan]]
    if beta > 0.0:
        res = np.concatenate([[0.0, 0.0], cv[2:]])      # nominal VCM resonances only (the rigid body is common)
        for t in ("LT", "HT"):
            asv, bsv, csv = _blocks(vcm[t][1:])
            asp, bsp, csp = _blocks(pzt[t])
            parts += [(asv, np.column_stack([bsv, 0 * bsv])), (asp, np.column_stack([0 * bsp, bsp]))]
            for row in rows:
                row += [0 * csv, 0 * csp]
            rows.append([-res, -cp, 0 * c_fan] + [np.zeros(p[0].shape[0]) for p in parts[3:-2]] + [csv, csp])
    n = sum(p[0].shape[0] for p in parts)
    F, Gu, Hm = np.zeros((n, n)), np.zeros((n, 2)), np.zeros((len(rows), n))
    i = 0
    for a, b in parts:
        k = a.shape[0]
        F[i:i + k, i:i + k], Gu[i:i + k] = a, b
        i += k
    F[:nv, nv + npz:n0] = np.outer(bv, c_rv)              # the rotational vibration enters at the VCM input
    for j, row in enumerate(rows):
        Hm[j] = np.concatenate(row + [np.zeros(n - sum(len(r) for r in row))])
    # lift to the PES rate: S maps [x[k], u (4), w (14 x 20)] to the state along the sample
    nw = 14 * _NF
    S = np.zeros((n, n + 4 + nw))
    S[:, :n] = np.eye(n)
    Y = np.zeros((_NF, len(rows), n + 4 + nw))
    for i in range(_NF):
        Y[i] = Hm @ S
        S = F @ S
        h = 0 if i < _NF // 2 else 1
        S[:, n + 2 * h:n + 2 * h + 2] += Gu
        S[nv + npz:n0, n + 4 + 14 * i:n + 4 + 14 * (i + 1)] += bd
    A, B, G = S[:, :n], S[:, n:n + 4], S[:, n + 4:]
    # regulator: fine-grid head position + weighted mismatch outputs + command penalties
    wts = np.array([1.0] + [beta] * (len(rows) - 1))
    Yx, Yu = Y[:, :, :n], Y[:, :, n:n + 4]
    Q = np.einsum("ioa,o,iob->ab", Yx, wts, Yx) / _NF
    N = np.einsum("ioa,o,iob->ab", Yx, wts, Yu) / _NF
    R = np.einsum("ioa,o,iob->ab", Yu, wts, Yu) / _NF + np.diag([g["R_VCM"], g["R_PZT"]] * 2)
    X = solve_discrete_are(A, B, Q, R, s=N)
    K = np.linalg.solve(R + B.T @ X @ B, B.T @ X @ A + N.T)
    # Kalman filter on the nominal model: pes = -H x + run-out (white, variance RHO * rms^2)
    A0, G0 = A[:n0, :n0], G[:n0]
    Cm = -Hm[0:1, :n0]
    rv = g["RHO"] * _RRO_RMS ** 2
    P = solve_discrete_are(A0.T, Cm.T, G0 @ G0.T / 12.0, np.array([[rv]]))
    Lf = np.zeros((n, 1))
    Lf[:n0] = P @ Cm.T / (Cm @ P @ Cm.T + rv)
    M = np.eye(n) - Lf @ np.concatenate([Cm, np.zeros((1, n - n0))], axis=1)
    Acl = A - B @ K
    return Acl @ M, Acl @ Lf, -K @ M, -K @ Lf


_AK, _BK, _CK, _DK = design()
_NK = _AK.shape[0]
_STEP = np.block([[_AK, _BK], [_CK, _DK]])      # [x[k+1]; u[k]] = _STEP [x[k]; pes[k]]
# The product is taken in row blocks of at most ~6000 entries: small enough that the linear-algebra library never
# hands it to worker threads (which would cost processor time on a multi-core machine), whatever its configuration.
_ROWS = max(1, 6000 // _STEP.shape[1])
_PARTS = [(i, min(i + _ROWS, _NK + 4), np.ascontiguousarray(_STEP[i:i + _ROWS])) for i in range(0, _NK + 4, _ROWS)]
_x = np.zeros(_NK + 1)                           # controller state x[k], then the current PES
_out = np.zeros(_NK + 4)                         # next state, then the four commands


def controller(obs: dict) -> np.ndarray:
    x, out = _x, _out
    if obs["step"] == 0:                         # new rollout: start from rest
        x[:] = 0.0
    x[_NK] = obs["pes"]
    for i, j, part in _PARTS:
        out[i:j] = part @ x
    x[:_NK] = out[:_NK]
    return out[_NK:].reshape(2, 2).copy()
