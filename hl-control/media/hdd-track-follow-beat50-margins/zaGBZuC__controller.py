"""Dual-stage HDD track following: a model-based LQG servo, one fixed linear controller per drive temperature.

Control law (all designed once, when this file is imported, from the benchmark's own model in ``hl_control.envs.hdd``):

* Model.  For each temperature (LT / RT / HT, nominal PZT gain) the 48-state plant is augmented with the benchmark's
  disturbance shaping filters (3 states of rotational vibration entering at the VCM input, 13 fan resonances added to
  the head position) on the ``Ts/20`` grid and lifted exactly to the PES rate: 77 states, 4 inputs (VCM and PZT
  command for each half of the sample).
* Kalman filter (the plant copy run as an observer).  It estimates plant and disturbance states from the PES.  The
  repeatable run-out is not followed on purpose: it is treated as measurement noise.  ``R_FICT`` [nm^2] is the
  measurement-noise variance the filter is designed for; it is the knob that sets the loop gain, i.e. trades 3-sigma
  against the stability margins (larger = lower bandwidth, larger margins, smaller sensitivity peak).
* LQR with an intersample cost.  The state feedback minimises the head position at all 20 fine-grid instants of a
  sample (so the multi-rate commands also keep the resonances above the Nyquist frequency quiet) plus ``RHO_PZT`` times
  the squared PZT displacement (keeps the micro-actuator inside its 50 nm stroke and leaves the large, slow motion to
  the VCM) plus a tiny regularisation ``RHO_U`` on the commands.  Without computation delay the command uses the
  filtered estimate ``x[k|k]``.
* Temperature selection.  The resonances move by 4-6 % with temperature, more than a single fixed model-based law
  tolerates, so three observers (LT, RT, HT) run in parallel during the first ``N_ID`` samples of the settling time
  (20 ms of the 50 ms).  A small deterministic PZT dither (``PROBE_AMP`` nm for ``N_PROBE`` samples; the PZT stays
  inside its stroke) makes the three predictions differ; after ``N_HOLD`` samples of dither alone (0.5 ms, no feedback)
  the observer with the smallest accumulated squared innovation supplies the feedback.  At ``N_ID`` the choice is
  frozen: from then on (in particular over the whole measurement window) the controller is one fixed linear
  time-invariant 77-state system.  The +-5 % PZT gain spread is covered by the loop's margins.

Everything is reset at ``obs["step"] == 0``.
"""

import math

import numpy as np
from scipy.linalg import expm, solve_discrete_are

from hl_control.envs import hdd

# ---- gains ------------------------------------------------------------------------------------------------------------
R_FICT = 1.0        # nm^2, design measurement-noise variance of the Kalman filter (loop-gain / margin knob)
RHO_PZT = 3e-3      # weight of the squared PZT displacement relative to the squared head position
RHO_U = 1e-8        # regularisation of the commands (nm / nano-unit scale)
N_ID = 1000         # samples during which the temperature hypothesis may still change (then frozen, LTI)
N_HOLD = 24         # samples of dither without feedback before the first decision
N_PROBE = 150       # samples of PZT dither
PROBE_AMP = 3.0     # nm, amplitude of the PZT dither (random sign on each half sample, fixed sequence)

TS, TSC, MR_P, HALF = hdd.TS, hdd.TSC, hdd.MR_P, hdd.HALF
NV, NPZ, NR, NF = 32, 16, 3, 2 * len(hdd.FAN_MODES)
NX = NV + NPZ + NR + NF
TEMPS = ("LT", "RT", "HT")


def _rv_filter():
    """The rotational-vibration shaping filter on the fine grid (strictly proper part, as the benchmark simulates it)."""
    gain, p1, zeta1 = 3e-10, 2.0 * math.pi * 3.0, 0.1
    w1, w2 = 2.0 * math.pi * 250.0, 2.0 * math.pi * 2000.0
    quad = np.array([1.0, 2.0 * zeta1 * w1, w1 * w1])
    num = gain * np.polymul([1.0, 2.0 * math.pi * 50.0], [1.0, 2.0 * 20.0 * w2, w2 * w2])
    proper = np.polysub(num, gain * np.polymul([1.0, p1], quad))
    residue = np.polyval(proper, -p1) / np.polyval(quad, -p1)
    c1, c0 = np.polydiv(np.polysub(proper, residue * quad), [1.0, p1])[0][-2:]
    pole = math.exp(-p1 * TSC)
    m = np.zeros((3, 3))
    m[0, 1], m[1, 0], m[1, 1] = w1, -w1, -2.0 * zeta1 * w1
    m[0, 2], m[1, 2] = c1, (c0 - 2.0 * zeta1 * w1 * c1) / w1
    e = expm(m * TSC)
    a, b = np.zeros((3, 3)), np.zeros(3)
    a[0, 0], b[0] = pole, residue * (1.0 - pole) / p1
    a[1:, 1:], b[1:] = e[:2, :2], e[:2, 2]
    return a, b * 1e9, np.array([1.0, 1.0, 0.0])


def build_model(temperature, pzt_gain=1.0):
    """Plant + disturbance model lifted to the PES rate.  Units: nm, nano-units of VCM command, white noise of the benchmark.

    ``x+ = A x + B u + w`` (cov ``Qd``), head position at the sample ``cy x``; ``Yz x + Yu u`` and ``Zz x + Zu u`` are the
    head position and the PZT displacement at the 20 fine-grid instants of the sample."""
    av, bv, cv = hdd._fine_model("vcm", temperature)
    ap, bp, cp = hdd._fine_model("pzt", temperature)
    ar, br, cr = _rv_filter()
    af, bf, gf = np.zeros((NX, NX)), np.zeros((NX, 2)), np.zeros((NX, 14))
    cy, cz = np.zeros(NX), np.zeros(NX)
    o = NV + NPZ
    af[:NV, :NV], af[NV:o, NV:o], af[o:o + NR, o:o + NR] = av, ap, ar
    af[:NV, o:o + NR] = np.outer(bv, cr)                 # rotational vibration enters at the VCM input
    bf[:NV, 0], bf[NV:o, 1] = bv, bp
    gf[o:o + NR, 0] = br
    cy[:NV], cy[NV:o], cz[NV:o] = cv, pzt_gain * cp, pzt_gain * cp
    o += NR
    for i, (k, f, z) in enumerate(hdd.FAN_MODES):
        ad, bd = hdd._mode_zoh(k, 2.0 * math.pi * f, z, TSC)
        af[o + 2 * i:o + 2 * i + 2, o + 2 * i:o + 2 * i + 2] = ad
        gf[o + 2 * i:o + 2 * i + 2, i + 1] = bd * 1e9
        cy[o + 2 * i] = 1.0
    d = np.ones(NX)
    d[1] = TS                                            # rigid-body velocity in nm per sample (conditioning)
    af, bf, gf, cy, cz = (d[:, None] * af) / d[None, :], d[:, None] * bf, d[:, None] * gf, cy / d, cz / d
    p = [np.eye(NX)]
    for _ in range(MR_P):
        p.append(p[-1] @ af)
    cum = [np.zeros((NX, 2))]
    for i in range(HALF):
        cum.append(cum[-1] + p[i] @ bf)
    b0, b1 = p[HALF] @ cum[HALF], cum[HALF]
    b = np.column_stack([b0[:, 0], b0[:, 1], b1[:, 0], b1[:, 1]])
    qd = sum(p[m] @ gf @ gf.T @ p[m].T for m in range(MR_P)) / 12.0
    yz, zz = np.array([cy @ p[i] for i in range(MR_P)]), np.array([cz @ p[i] for i in range(MR_P)])
    yu, zu = np.zeros((MR_P, 4)), np.zeros((MR_P, 4))
    for i in range(MR_P):
        r0, r1 = (cum[i], np.zeros((NX, 2))) if i <= HALF else (p[i - HALF] @ cum[HALF], cum[i - HALF])
        yu[i], zu[i] = np.concatenate([cy @ r0, cy @ r1]), np.concatenate([cz @ r0, cz @ r1])
    return {"A": p[MR_P], "B": b, "Qd": qd, "cy": cy, "Yz": yz, "Yu": yu, "Zz": zz, "Zu": zu}


def design_lqg(m, r_fict=R_FICT, rho_pzt=RHO_PZT, rho_u=RHO_U):
    """LQR on the intersample cost and steady-state Kalman filter; returns the gains and the controller ``(Ac, Bc, Cc, Dc)``
    from the PES [nm] to the four commands (state: the one-step prediction of the observer)."""
    a, b, cy = m["A"], m["B"], m["cy"]
    q = m["Yz"].T @ m["Yz"] + rho_pzt * m["Zz"].T @ m["Zz"]
    n = m["Yz"].T @ m["Yu"] + rho_pzt * m["Zz"].T @ m["Zu"]
    r = m["Yu"].T @ m["Yu"] + rho_pzt * m["Zu"].T @ m["Zu"] + rho_u * np.eye(4)
    x = solve_discrete_are(a, b, q, r, s=n)
    f = np.linalg.solve(r + b.T @ x @ b, b.T @ x @ a + n.T)
    pk = solve_discrete_are(a.T, cy[:, None], m["Qd"], np.array([[r_fict]]))
    kf = pk @ cy / float(cy @ pk @ cy + r_fict)
    af = a - b @ f
    ikc = np.eye(NX) - np.outer(kf, cy)
    return {"F": f, "Kf": kf, "Ac": af @ ikc, "Bc": -af @ kf, "Cc": -f @ ikc, "Dc": f @ kf}


MODELS = [build_model(t) for t in TEMPS]
DESIGNS = [design_lqg(m) for m in MODELS]

# stacked observers for the identification phase
_A3 = np.ascontiguousarray(np.stack([m["A"] for m in MODELS]))
_B3 = np.ascontiguousarray(np.stack([m["B"] for m in MODELS]))
_CY3 = np.ascontiguousarray(np.stack([m["cy"] for m in MODELS]))
_KF3 = np.ascontiguousarray(np.stack([d["Kf"] for d in DESIGNS]))
_F3 = np.ascontiguousarray(np.stack([d["F"] for d in DESIGNS]))
# frozen controllers: [x+ ; u] = M [x ; pes]
_M3 = [np.ascontiguousarray(np.block([[d["Ac"], d["Bc"][:, None]], [d["Cc"], d["Dc"][:, None]]])) for d in DESIGNS]
_PROBE = np.zeros((N_PROBE, 4))
_PROBE[:, [1, 3]] = PROBE_AMP * np.sign(np.random.default_rng(20240607).random((N_PROBE, 2)) - 0.5)

_state = {"x3": np.zeros((3, NX)), "j": np.zeros(3), "sel": 1, "xu": np.zeros(NX + 1)}


def controller(obs: dict) -> np.ndarray:
    st = _state
    k = obs["step"]
    pes = obs["pes"] * 1e9
    if k == 0:
        st["x3"] = np.zeros((3, NX))
        st["j"] = np.zeros(3)
        st["sel"] = 1
        st["xu"] = np.zeros(NX + 1)
    if k < N_ID:
        # three observers in parallel; feedback from the one that predicts the PES best so far
        x3 = st["x3"]
        innov = -pes - np.einsum("cj,cj->c", _CY3, x3)
        x3 = x3 + _KF3 * innov[:, None]
        st["j"] += innov * innov
        if k >= N_HOLD:
            st["sel"] = int(np.argmin(st["j"]))
        sel = st["sel"]
        u = -(_F3[sel] @ x3[sel]) if k >= N_HOLD else np.zeros(4)
        if k < N_PROBE:
            u = u + _PROBE[k]
        st["x3"] = np.matmul(_A3, x3[:, :, None])[:, :, 0] + _B3 @ u
        if k == N_ID - 1:
            st["xu"][:NX] = st["x3"][sel]
    else:
        # frozen linear time-invariant controller of the selected temperature
        xu = st["xu"]
        xu[NX] = pes
        out = _M3[st["sel"]] @ xu
        xu[:NX] = out[:NX]
        u = out[NX:]
    return (u * 1e-9).reshape(2, 2)
