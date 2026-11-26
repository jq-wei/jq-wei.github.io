"""Adroit pen spin: a vibratory "hop-and-spin" gait on the finger rails, with closed-loop centring.

Physics the law is built on (measured in the simulator): the pen/hand contacts carry a very large
torsional friction, so a resting pen yaws rigidly with whatever link carries its weight and cannot be
pushed round on the palm. It can only turn (a) together with its support or (b) while it is off the
support. The law makes the pen hop at about 6 Hz and gives its support a yaw velocity at every take-off.

1. Roll-in. The hand is opened flat, the four fingers slightly flexed at the knuckle (`beta`), and the
   wrist is tilted so that the pen rolls from the palm onto the proximal phalanges (the "rails"). A PD
   law on the pen's position along the hand, `s` (distance in front of the knuckle line, computed in
   the palm frame from the two wrist angles), sets the world slope of the rails,
   `tilt = kp*(s - s0) + kd*v_y`, clipped to [-tilt_fwd, tilt_back] and rate-limited (`dtilt`); the pen
   rolls downhill towards `s0`. An integrator (`ki`) removes the gravity sag of the wrist servo while
   the hand is still at rest. The wrist deviation is pre-set from the pen's initial offset and yaw
   (`pre`, `xt`) so that the obliquely rolling pen ends up centred across the rails.
2. Spin. After `t_settle` the four finger-abduction joints oscillate together,
   `delta = A sin(2 pi f t)`: the rails yaw about the knuckles and the pen yaws with them. The wrist
   flexion gets a superposed oscillation `B sin(2 pi f t + phi)` that makes the pen leave the rails
   once per period; `phi` places the take-off where the rails' yaw rate is large, so the pen keeps
   turning in flight while the fingers swing back underneath it. Every period is one ratchet step.
3. Centring. Abduction alone would also carry the pen sideways (it sits `s` in front of the knuckle
   axis). The wrist deviation therefore counter-oscillates, `w = w_c - kc*delta`, with `kc = s/R`
   (R = distance of the pen from the wrist's deviation axis): the two rotations add up to a rotation
   about the pen's own centre. The lateral error of the pen in the palm frame, `e = x - x0`, is fed
   back on that ratio (`gx` proportional, `gi` integral, `gd` rate, clipped to +-`kclip`), which turns
   every hop into a small sideways step back to the middle of the rails; `e` is taken at a point
   `com` towards the pen's heavy end (its centre of mass is 1 cm off its geometric centre). When the
   filtered error grows the abduction amplitude is scaled down (`e_soft`, floor `a_min`): spin less,
   centre more.

Gains (PARAMS): s0 rail set point [m]; beta knuckle flexion [rad]; kp, kd roll PD [rad/m, rad s/m];
tilt_fwd, tilt_back, dtilt slope limits [rad] and rate [rad/step]; ki sag integrator; A, f abduction
amplitude [rad] and gait frequency [Hz]; B, phi hop amplitude [rad] and phase [rad]; gx, gi, gd, kclip
lateral feedback on the deviation ratio; x0, com lateral set point and reference point [m]; e_soft,
a_min amplitude back-off; pre, xt, w_c, wmin, wmax wrist pre-positioning; t_settle start of the gait
[s]; thj1 opens the thumb out of the pen's sweep.
"""

import numpy as np

PARAMS = dict(
    s0=0.021773591313207063, beta=0.12945340824169105, kp=14.604925897888236, kd=3.0,
    tilt_fwd=0.10364156013826528, tilt_back=0.341421511899775, ki=0.15, dtilt=0.03,
    A=0.3780298048741871, f=5.979138891017409, B=0.10012043780409963, phi=0.18334279472205162,
    t_settle=1.220969163764867,
    gx=20.0, gi=20.0, gd=0.0, kclip=0.5, x0=0.0, com=0.005, e_soft=0.02, a_min=0.3,
    pre=1.0, xt=0.003, w_c=0.0, wmin=-0.25, wmax=0.06, thj1=0.5,
)

WRJ1, WRJ0, THJ1 = 0, 1, 22
J3 = [2, 6, 10, 15]          # finger abduction (side-to-side) joints
J2 = [3, 7, 11, 16]          # finger knuckle flexion joints
C_DEV = np.array([0.0, -0.304, 0.2])   # the wrist deviation axis (vertical) passes through this point
C_FLEX = np.array([0.0, -0.27, 0.2])   # the wrist flexion axis (along x) passes through this point
Y_KNUCKLE = -0.175                     # knuckle line in palm coordinates [m]
R_KNUCKLE = 0.129                      # knuckle line to deviation axis [m]
R_RAIL = 0.155                         # rails to deviation axis, for the lateral pre-positioning [m]

_state = {}


def _palm_frame(p, w, q):
    """World point -> palm coordinates (the world coordinates the same material point has at the zero pose)."""
    d = p - C_DEV
    c, s = np.cos(w), np.sin(w)                       # deviation turns the hand about -z by w: undo it
    d = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1], d[2]]) + C_DEV
    e = d - C_FLEX
    c, s = np.cos(q), np.sin(q)                       # flexion turns the palm about +x by q: undo it
    return np.array([e[0], c * e[1] + s * e[2], -s * e[1] + c * e[2]]) + C_FLEX


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    mean, half = np.asarray(obs["ctrl_mean"], dtype=float), np.asarray(obs["ctrl_half_range"], dtype=float)
    t = float(obs["time"])
    w, q = o[0], o[1]
    p, v, axis = o[24:27], o[27:30], o[33:36]

    if int(obs["step"]) == 0 or not _state:
        # where the obliquely lying pen will be, laterally, once it has rolled onto the rails
        a0 = (np.arctan2(axis[1], axis[0]) + np.pi / 2) % np.pi - np.pi / 2
        x_end = p[0] - np.tan(a0) * (P["s0"] - (p[1] - Y_KNUCKLE))
        w_pre = P["w_c"] + P["pre"] * np.arcsin(np.clip((x_end - P["xt"]) / R_RAIL, -0.9, 0.9))
        _state.clear()
        _state.update(I=0.07, tilt=0.0, wc=float(np.clip(w_pre, P["wmin"], P["wmax"])), ef=None, rate=0.0, ie=0.0)

    s = _palm_frame(p, w, q)[1] - Y_KNUCKLE            # pen centre in front of the knuckle line [m]
    x_lat = _palm_frame(p - P["com"] * axis, w, q)[0]  # lateral position of the reference point on the pen [m]

    # 1. slope of the rails: PD on s (the pen rolls downhill), rate-limited
    tilt = float(np.clip(P["kp"] * (s - P["s0"]) + P["kd"] * v[1], -P["tilt_fwd"], P["tilt_back"]))
    if t < 0.25:
        tilt = 0.0                                     # let the pen land on the level palm first
    tilt = float(np.clip(tilt, _state["tilt"] - P["dtilt"], _state["tilt"] + P["dtilt"]))
    _state["tilt"] = tilt

    # 2. gait: abduction oscillation (yaw of the rails) and wrist-flexion oscillation (hop)
    delta = hop = 0.0
    spinning = t > P["t_settle"]
    if spinning:
        th = 2.0 * np.pi * P["f"] * (t - P["t_settle"])
        ramp = min(1.0, (t - P["t_settle"]) / 0.4)
        delta = ramp * P["A"] * np.sin(th)
        hop = ramp * P["B"] * np.sin(th + P["phi"])
    else:
        j2m = float(np.mean(o[J2]))                    # world slope of the rails = wrist flexion + knuckle flexion
        _state["I"] = float(np.clip(_state["I"] + P["ki"] * (tilt - (q + j2m)), -0.3, 0.3))

    # 3. centring: lateral feedback on the deviation counter-rotation ratio, amplitude back-off
    e = x_lat - P["x0"]
    ef_prev = e if _state["ef"] is None else _state["ef"]
    ef = ef_prev + 0.1 * (e - ef_prev)
    _state["ef"] = ef
    _state["rate"] += 0.1 * ((ef - ef_prev) / 0.01 - _state["rate"])
    _state["ie"] = float(np.clip(_state["ie"] + 0.01 * e, -0.02, 0.02)) if spinning else 0.0
    fb = float(np.clip(P["gx"] * e + P["gd"] * _state["rate"] + P["gi"] * _state["ie"], -P["kclip"], P["kclip"]))
    scale = max(P["a_min"], 1.0 - abs(ef) / P["e_soft"])

    tgt = np.zeros(24)                                 # joint set points [rad]; everything else stays at the flat pose
    tgt[THJ1] = P["thj1"]
    tgt[WRJ1] = _state["wc"] - (s / (R_KNUCKLE + s)) * delta * scale - fb * delta
    tgt[WRJ0] = tilt - P["beta"] + _state["I"] + hop
    tgt[J3] = delta * scale
    tgt[J2] = P["beta"]
    a = (tgt - mean) / half
    if not np.all(np.isfinite(a)):
        a = np.zeros(24)
    return np.clip(a, -1.0, 1.0)
