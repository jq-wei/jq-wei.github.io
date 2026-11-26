"""Dual-stage HDD track following: output-feedback LQG on the benchmark's own model, plus a run-out learner.

Control law (one matrix-vector product per PES sample)
------------------------------------------------------
A Kalman filter runs a copy of the plant (16 VCM modes + 8 PZT modes, at one fixed design temperature inside the range
of the cases) augmented with the benchmark's disturbance models: the rotational-vibration shaping filter at the VCM
input and the 13 fan resonances at the head position.  Everything is discretised exactly on the Ts/20 grid and lifted
to the PES rate.  The state estimate is fed back through an LQR gain that minimises the mean square of the head
position on the Ts/20 grid, so the motion between samples and the two half-sample commands of each actuator are
optimised together:

    m    = rhat[sector] - pes          measured head position (pes = d_rro - y), learned run-out rhat removed
    nu   = m - H z                     innovation (z = one-step prediction of the 77 states)
    xf   = z + L nu                    measurement update
    u    = -K xf                       [u_vcm0, u_pzt0, u_vcm1, u_pzt1], applied at once (no computation delay)
    z+   = A xf + B u                  time update

The four lines are folded into one constant matrix: [z+; u; nu] = M [z; m].  The loop itself is linear and
time-invariant; the plant case is never identified, the same gains serve all nine.

Run-out learner (once per revolution)
-------------------------------------
The repeatable run-out is measurement noise for the head position, and it repeats every 420 samples.  At the end of
each revolution the FFT of that revolution's innovations, divided by the innovation's frequency response to a
measurement error (W, from the model), is an estimate of the run-out still left in m; a fraction of it is added to
rhat.  Per harmonic the fraction is the recursive-least-squares gain 1 / (n + a_h) after n revolutions, where a_h is
the noise-to-prior variance ratio of one revolution's estimate (computed from the model: large below 1 kHz and in the
fan band, where run-out cannot be told from vibration, so those harmonics are simply followed), capped so that the
update contracts on every one of the nine cases.  rhat only shifts the measurement: it cannot destabilise the loop.

Gains (GAINS)
-------------
Chosen by a Nelder-Mead search that minimises the worst case over the nine plants of the closed-loop 3-sigma, computed
exactly from the steady-state covariance of the lifted closed loop (it matches the simulator), with the run-out left
after learning modelled as a small white measurement noise; the result was then checked in simulation, also on
temperatures between and 40 % beyond the cases (performance degrades smoothly, the loop stays stable).

rho_v, rho_p   : LQR weights on the VCM and PZT commands (nano-units squared).  They set how much each actuator is
                 used: the VCM takes the slow, large rotational vibration, the PZT the fan band.
q_pzt          : LQR weight on the PZT displacement; keeps the stroke far below the 50 nm limit (about 13 nm peak).
r_meas         : measurement-noise variance of the Kalman filter [nm^2].  Small, because the learner removes most of
                 the run-out above the fan band (the raw run-out is 0.26 nm^2).  Smaller = faster, less robust filter.
q_in_v, q_in_p : fictitious white noise at the VCM / PZT inputs in the filter design (loop-transfer-recovery style
                 robustness against the +-4 % / +-6 % resonance shifts and the +-5 % PZT gain of the nine cases); with
                 the small r_meas these, not the measurement noise, set how cautious the filter is.
q_mode_v, q_mode_p : fictitious noise on the resonant-mode states of the VCM / PZT model (the filter then trusts the
                 phase of those modes less).
f_vcm, f_pzt, f_zeta, g_pzt : the design model: factors on the VCM / PZT resonance frequencies and on all damping
                 ratios, and the PZT gain (1, 1, 1, 1 = room temperature, nominal gain; the cases span 0.96-1.04,
                 0.94-1.06, 0.8-1.2 and 0.95-1.05).  A model with somewhat higher resonances and more damping than
                 nominal turns out to be the best compromise for the worst case (the low-temperature plants).
rro_noise_scale : multiplies a_h (1 = the model's value; larger = slower, smoother learning).
rro_skip_revs  : revolutions at the start of a rollout left out of the learning (the filter's start-up transient).
"""

import math

import numpy as np
import scipy.linalg as sla

from hl_control.envs import hdd

GAINS = {
    "rho_v": 0.001499,
    "rho_p": 0.08718,
    "q_pzt": 0.003422,
    "r_meas": 0.0006124,
    "q_in_v": 1.501,
    "q_in_p": 0.01043,
    "q_mode_v": 3.156e-06,
    "q_mode_p": 0.0008022,
    "f_vcm": 1.024,
    "f_pzt": 1.052,
    "f_zeta": 1.289,
    "g_pzt": 1.0,
    "rro_noise_scale": 1.0,
    "rro_skip_revs": 1,
}

_NV, _NP, _NR, _NF = 32, 16, 3, 26
_N = _NV + _NP + _NR + _NF
_SCALE = 1e9  # design in nanometres / nano-units


def _disturbance_ss():
    """State-space models on the fine grid of the rotational-vibration filter (3 states) and the fan modes (26)."""
    tsc = hdd.TSC
    gain, p1, zeta1 = 3e-10, 2.0 * math.pi * 3.0, 0.1
    w1, w2 = 2.0 * math.pi * 250.0, 2.0 * math.pi * 2000.0
    quad = np.array([1.0, 2.0 * zeta1 * w1, w1 * w1])
    num = gain * np.polymul([1.0, 2.0 * math.pi * 50.0], [1.0, 2.0 * 20.0 * w2, w2 * w2])
    proper = np.polysub(num, gain * np.polymul([1.0, p1], quad))
    residue = np.polyval(proper, -p1) / np.polyval(quad, -p1)
    c1, c0 = np.polydiv(np.polysub(proper, residue * quad), [1.0, p1])[0][-2:]
    pole = math.exp(-p1 * tsc)
    m = np.zeros((3, 3))
    m[0, 1], m[1, 0], m[1, 1] = w1, -w1, -2.0 * zeta1 * w1
    m[0, 2], m[1, 2] = c1, (c0 - 2.0 * zeta1 * w1 * c1) / w1
    e = sla.expm(m * tsc)
    a_rv = np.zeros((3, 3))
    a_rv[0, 0] = pole
    a_rv[1:, 1:] = e[:2, :2]
    b_rv = np.array([residue * (1.0 - pole) / p1, e[0, 2], e[1, 2]])
    c_rv = np.array([1.0, 1.0, 0.0])
    a_fan = np.zeros((_NF, _NF))
    b_fan = np.zeros((_NF, len(hdd.FAN_MODES)))
    c_fan = np.zeros(_NF)
    for i, (g, f, z) in enumerate(hdd.FAN_MODES):
        ad, bd = hdd._mode_zoh(g, 2.0 * math.pi * f, z, tsc)
        a_fan[2 * i:2 * i + 2, 2 * i:2 * i + 2] = ad
        b_fan[2 * i:2 * i + 2, i] = bd
        c_fan[2 * i] = 1.0
    return a_rv, b_rv, c_rv, a_fan, b_fan, c_fan


def _fine_actuator(actuator, f_scale, z_scale):
    """Exact fine-grid model of an actuator whose resonance frequencies / dampings are scaled (as hdd._fine_model)."""
    if actuator == "vcm":
        modes = [(k * hdd.KP_VCM, 2.0 * math.pi * f * f_scale, z * z_scale)
                 for k, f, z in zip(hdd.VCM_KAPPA, hdd.VCM_FREQ_HZ, hdd.VCM_ZETA)]
    else:
        modes = [(k, 2.0 * math.pi * f * f_scale, z * z_scale)
                 for k, f, z in zip(hdd.PZT_KAPPA, hdd.PZT_FREQ_HZ, hdd.PZT_ZETA)]
        dc = abs(sum(k / (w * w) for k, w, _ in modes))
        modes = [(k / dc, w, z) for k, w, z in modes]
    n = 2 * len(modes)
    a, b, c = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    for i, (gain, omega, zeta) in enumerate(modes):
        ad, bd = hdd._mode_zoh(gain, omega, zeta, hdd.TSC)
        a[2 * i:2 * i + 2, 2 * i:2 * i + 2] = ad
        b[2 * i:2 * i + 2] = bd
        c[2 * i] = 1.0
    return a, b, c


def build_model(f_vcm=1.0, f_pzt=1.0, f_zeta=1.0, pzt_gain=1.0):
    """Plant + disturbance model lifted to the PES rate (states in nm, rigid-body velocity in nm/sample).

    f_vcm, f_pzt, f_zeta scale the VCM / PZT resonance frequencies and all damping ratios (1, 1, 1 = room temperature).
    Returns x+ = A x + B u + w (cov(w) = Qd), head position y = H x and PZT displacement Hp x at the sample, and on the
    Ts/20 grid inside the sample: y_fine = Yx x + Yu u (+ noise of variance Vy), pzt_fine = Px x + Pu u."""
    av, bv, cv = _fine_actuator("vcm", f_vcm, f_zeta)
    ap, bp, cp = _fine_actuator("pzt", f_pzt, f_zeta)
    a_rv, b_rv, c_rv, a_fan, b_fan, c_fan = _disturbance_ss()
    n, nv, npz = _N, _NV, _NP
    i_rv, i_fan = nv + npz, nv + npz + _NR
    f = np.zeros((n, n))
    f[:nv, :nv], f[nv:i_rv, nv:i_rv], f[i_rv:i_fan, i_rv:i_fan], f[i_fan:, i_fan:] = av, ap, a_rv, a_fan
    f[:nv, i_rv:i_fan] = np.outer(bv, c_rv)
    gu = np.zeros((n, 2))
    gu[:nv, 0], gu[nv:i_rv, 1] = bv, bp
    gw = np.zeros((n, hdd.N_NOISE))
    gw[i_rv:i_fan, 0] = b_rv * _SCALE
    gw[i_fan:, 1:] = b_fan * _SCALE
    h = np.zeros(n)
    h[:nv], h[nv:i_rv], h[i_fan:] = cv, pzt_gain * cp, c_fan
    hp = np.zeros(n)
    hp[nv:i_rv] = pzt_gain * cp
    # scale the rigid-body velocity (state 1) to nm per PES sample
    t = np.ones(n)
    t[1] = hdd.TS
    f = (t[:, None] * f) / t[None, :]
    gu, gw = t[:, None] * gu, t[:, None] * gw
    h, hp = h / t, hp / t
    mr_p, half = hdd.MR_P, hdd.HALF
    fp = [np.eye(n)]
    for _ in range(mr_p):
        fp.append(fp[-1] @ f)
    cum = [np.zeros((n, 2))]
    for i in range(half):
        cum.append(cum[-1] + fp[i] @ gu)
    b0, b1 = fp[half] @ cum[half], cum[half]
    b = np.column_stack([b0[:, 0], b0[:, 1], b1[:, 0], b1[:, 1]])
    qd = np.zeros((n, n))
    for j in range(mr_p):
        fg = fp[j] @ gw
        qd += fg @ fg.T / 12.0
    yx, yu, px, pu = np.zeros((mr_p, n)), np.zeros((mr_p, 4)), np.zeros((mr_p, n)), np.zeros((mr_p, 4))
    vy = np.zeros(mr_p)
    for i in range(mr_p):
        if i <= half:
            e0, e1 = cum[i], np.zeros((n, 2))
        else:
            e0, e1 = fp[i - half] @ cum[half], cum[i - half]
        e = np.column_stack([e0[:, 0], e0[:, 1], e1[:, 0], e1[:, 1]])
        yx[i], yu[i] = h @ fp[i], h @ e
        px[i], pu[i] = hp @ fp[i], hp @ e
        vy[i] = sum(float(np.sum((h @ fp[i - 1 - j] @ gw) ** 2)) for j in range(i)) / 12.0
    return {"A": fp[mr_p], "B": b, "Qd": qd, "H": h, "Hp": hp, "Yx": yx, "Yu": yu, "Px": px, "Pu": pu, "Vy": vy}


def design(params=None, model=None):
    """LQG synthesis; returns the (n + 5) x (n + 1) matrix M with [z+; u; nu] = M [z; m], and the model's H."""
    p = dict(GAINS)
    if params:
        p.update(params)
    m = model or build_model(p["f_vcm"], p["f_pzt"], p["f_zeta"], p["g_pzt"])
    a, b, h = m["A"], m["B"], m["H"]
    n = a.shape[0]
    # regulator: mean square of the head position on the fine grid + PZT displacement + command weights
    q = m["Yx"].T @ m["Yx"] + p["q_pzt"] * m["Px"].T @ m["Px"]
    r = m["Yu"].T @ m["Yu"] + p["q_pzt"] * m["Pu"].T @ m["Pu"]
    r += np.diag([p["rho_v"], p["rho_p"], p["rho_v"], p["rho_p"]])
    s = m["Yx"].T @ m["Yu"] + p["q_pzt"] * m["Px"].T @ m["Pu"]
    x = sla.solve_discrete_are(a, b, q, r, s=s)
    k = np.linalg.solve(r + b.T @ x @ b, b.T @ x @ a + s.T)
    # estimator: exact disturbance noise + fictitious input / modal noise for robustness
    qn = m["Qd"].copy()
    bv, bp = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    qn += p["q_in_v"] * np.outer(bv, bv) + p["q_in_p"] * np.outer(bp, bp)
    dq = np.zeros(n)
    dq[2:_NV] = p["q_mode_v"]
    dq[_NV:_NV + _NP] = p["q_mode_p"]
    qn += np.diag(dq)
    pc = sla.solve_discrete_are(a.T, h[:, None], qn, np.array([[p["r_meas"]]]))
    l = pc @ h / (h @ pc @ h + p["r_meas"])
    acl = a - b @ k
    imlh = np.eye(n) - np.outer(l, h)
    mat = np.zeros((n + 5, n + 1))
    mat[:n, :n] = acl @ imlh
    mat[:n, n] = acl @ l
    mat[n:n + 4, :n] = -k @ imlh
    mat[n:n + 4, n] = -k @ l
    mat[n + 4, :n] = -h                                   # innovation row: nu = m - H z
    mat[n + 4, n] = 1.0
    return mat, h


def _innovation_response(mat, h_design, model, zs):
    """Innovation of the filter in closed loop on one plant, at the points zs of the unit circle: its frequency response
    to an error added to the measurement, and its power spectrum under the plant's disturbances (no run-out)."""
    n = _N
    zz, zm, uz, um = mat[:n, :n], mat[:n, n], mat[n:n + 4, :n], mat[n:n + 4, n]
    a, b, h = model["A"], model["B"], model["H"]
    acl = np.zeros((2 * n, 2 * n))
    acl[:n, :n], acl[:n, n:] = a + b @ np.outer(um, h), b @ uz
    acl[n:, :n], acl[n:, n:] = np.outer(zm, h), zz
    bq, cn = np.concatenate([b @ um, zm]), np.concatenate([h, -h_design])
    eye, qd = np.eye(2 * n), model["Qd"]
    response, spectrum = np.zeros(len(zs), dtype=complex), np.zeros(len(zs))
    for i, z in enumerate(zs):
        g = np.linalg.solve((z * eye - acl).T, cn)                  # the row vector cn (zI - Acl)^-1
        response[i] = 1.0 + g @ bq
        spectrum[i] = (g[:n] @ qd @ g[:n].conj()).real
    return response, spectrum


def learning_gains(mat, h_design, noise_scale):
    """Reference innovation response W (mean over the nine cases), the noise term a_h and the gain cap of the learner.

    One revolution's estimate of the residual run-out at harmonic h is -FFT(innovation)_h / W_h.  Its noise-to-prior
    variance ratio is a_h = N Phi_h / (|W_h|^2 (N A / 2)^2): Phi_h is the innovation spectrum without run-out (mean over
    the cases), N = 420 and A = 0.05 nm the amplitude of every run-out harmonic; the recursive least-squares gain after
    n revolutions is 1 / (n + a_h).  On case c the estimate is rho_c times the true residual (rho_c = W_c / W), so an
    update with gain g leaves the factor (1 - g rho_c): the cap min_c Re(rho_c) / |rho_c|^2 keeps it at or below one."""
    ns = hdd.NUM_SECTOR
    zs = np.exp(2j * math.pi * np.arange(1, ns // 2) / ns)
    out = [_innovation_response(mat, h_design, build_model(*hdd.TEMPERATURES[t], g), zs) for t, g in hdd.CASES.values()]
    ws, phi = np.array([o[0] for o in out]), np.mean([o[1] for o in out], axis=0)
    w = ws.mean(axis=0)
    rho = ws / w
    cap = np.clip(np.min(rho.real / np.abs(rho) ** 2, axis=0), 0.0, 1.0)
    prior = (0.5 * ns * hdd.RRO_SCALE * _SCALE) ** 2
    return w, noise_scale * ns * phi / (np.abs(w) ** 2 * prior), cap


_NS = hdd.NUM_SECTOR
_M, _H = design()
_M = np.ascontiguousarray(_M)
_W, _GAIN_DEN, _GAIN_CAP = learning_gains(_M, _H, GAINS["rro_noise_scale"])
_v = np.zeros(_N + 1)
_o = np.zeros(_N + 5)
_rhat = np.zeros(_NS)                                    # learned run-out per sector [m]
_buf = np.zeros(_NS)                                     # innovations of the current revolution
_spec = np.zeros(_NS // 2 + 1, dtype=complex)
_state = {"rev": 0}


def controller(obs: dict) -> np.ndarray:
    v, o = _v, _o
    k = obs["step"]
    if k == 0:
        v[:] = 0.0
        _rhat[:] = 0.0
        _state["rev"] = 0
    s = k % _NS
    v[_N] = _rhat[s] - obs["pes"]                        # measurement of the head position, learned run-out removed
    np.dot(_M, v, out=o)
    v[:_N] = o[:_N]
    _buf[s] = o[_N + 4]
    if s == _NS - 1 and k >= GAINS["rro_skip_revs"] * _NS:
        _state["rev"] += 1
        gain = np.minimum(1.0 / (_state["rev"] + _GAIN_DEN), _GAIN_CAP)
        _spec[1:-1] = -gain * np.fft.rfft(_buf)[1:-1] / _W
        _rhat[:] += np.fft.irfft(_spec, _NS)
    return o[_N:_N + 4].reshape(2, 2).copy()
