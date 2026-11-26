"""Adroit pen spin: a finger-turntable ratchet with a "drop" reset, at joint level.

Physics the gait relies on
--------------------------
Every pen contact in this model has a very large torsional friction, so the pen turns about the
vertical together with whatever carries most of its weight; it cannot be pushed round on the palm.
The four fingers swept sideways together (the knuckle abduction joints xxJ3, all with the same sign)
are a turntable about the vertical that turns relative to the palm, so a pen lying on the fingers
just in front of the knuckles follows the sweep.

One cycle (Tn control steps)
----------------------------
1. Carry (Tn - nd steps): the fingers sweep from -A to +A (clockwise seen from above when sgn = +1)
   and carry the pen round, while the wrist flexion comes back up from the bottom of the last drop.
2. Drop (nd steps): the wrist flexion set point jumps down (wd), the hand falls away faster than
   gravity and the pen floats for a few tens of milliseconds. During the float the fingers snap back
   (overdriven set point aod) and the wrist yaw resets, so neither drags the pen back. The hand then
   catches the pen about 1.5 cm lower and the next carry starts.

Feedback (all from the observed pen centre, expressed in the hand frame of the wrist yaw)
---------------------------------------------------------------------------------------
- Along the fingers: the finger bed is held at a world slope `thf` (knuckle flexion = thf - measured
  wrist flexion), plus ky * (y - yt): the pen rolls or hops back to the knuckle valley.
- Sideways: sweeping the fingers also walks the pen sideways (it turns about the knuckle line, not
  about its own centre). Each cycle the wrist yaw makes a stroke dw = kx * (x - xt) while the pen is
  carried and takes it back during the float, which shifts the pen by about 0.135 m * dw in the hand.

- Thumb zone: for hand-frame azimuths (mod pi) between zlo and zhi one end of the pen sweeps past the
  base of the thumb, where the turn slows down; there the wrist yaw stroke gets an extra asz in the
  sweep direction, so the whole hand helps the pen through.
- Pace: after step nchk, if the turn made so far projects to less than three revolutions at the end
  of the episode, the cycle is shortened from Tn to Tnf steps.

Gains
-----
Tn, nd        cycle and drop length [control steps]
Tnf, nchk     shorter cycle used when behind schedule, and the step from which the pace is checked
asz, zlo, zhi extra wrist yaw stroke [rad] inside the thumb zone and the zone's limits [rad]
A, aod, lead  finger sweep half-amplitude, overdriven reset set point [rad], lead time of the sweep [s]
w0, wd, wb    wrist flexion: top of the cycle, set point during the drop, set point after it [rad]
rise          fraction of the carry used to bring the wrist back up
thf, ky, thmin  finger-bed slope [rad], its feedback gain [rad/m] and lower limit
b1            curl of the finger middle joints (a far stop for the pen)
kx, xt, yt    sideways centring gain [rad/m], pen target in the hand frame [m] (yt measured from the
              wrist yaw axis; the knuckle line is at 0.129)
kxy           shift of the sideways target per metre of error along the fingers
ff, dwmax     feed-forward wrist yaw stroke (fraction of the sweep) and stroke limit [rad]
yod           overdrive of the wrist yaw reset step during the float (cancels the servo lag)
psic          centre of the wrist yaw [rad], counter-clockwise positive (= -WRJ1)
n0            settling steps before the gait starts (the pen lands and rolls to the knuckles)
th4..th0      thumb pose: tucked down beside the palm, out of the pen's sweep
"""

import numpy as np

PARAMS = {
    "Tn": 20.0, "nd": 3.0, "A": 0.24, "aod": 0.44,
    "lead": 0.07, "w0": -0.03583861643266094, "wd": -0.40057275258634845, "wb": -0.21929185747312094,
    "rise": 0.662510927930003, "thf": 0.20862765748314577, "ky": 33.719, "thmin": -0.15,
    "b1": 0.6523293149173212, "kx": 8.0, "xt": -0.012, "kxy": -0.2412934509401713,
    "yt": 0.143, "ff": 0.0, "dwmax": 0.2817743833566103, "yod": 1.8,
    "psic": 0.0, "asz": 0.06, "zlo": 1.7, "zhi": 2.9,
    "n0": 40.0, "sgn": 1.0, "th4": -1.0, "th3": 0.033343821773062984,
    "th2": 0.0, "th1": 0.3297617178694835, "th0": 0.0,
    "Tnf": 19.0, "nchk": 1500.0,
}

PARAM_BOUNDS = {
    "A": (0.18, 0.3), "w0": (-0.1, 0.0), "wd": (-0.55, -0.25), "wb": (-0.3, -0.12), "rise": (0.4, 1.0),
    "thf": (0.1, 0.3), "ky": (5.0, 50.0), "b1": (0.3, 0.8), "kx": (4.0, 14.0), "xt": (-0.022, 0.0),
    "yt": (0.134, 0.15), "dwmax": (0.15, 0.35), "lead": (0.02, 0.07), "yod": (1.2, 2.2),
}

# joint / actuator order of the pen task
_NAMES = ["WRJ1", "WRJ0", "FFJ3", "FFJ2", "FFJ1", "FFJ0", "MFJ3", "MFJ2", "MFJ1", "MFJ0", "RFJ3", "RFJ2", "RFJ1", "RFJ0",
          "LFJ4", "LFJ3", "LFJ2", "LFJ1", "LFJ0", "THJ4", "THJ3", "THJ2", "THJ1", "THJ0"]
_IDX = {n: i for i, n in enumerate(_NAMES)}
_WRIST = np.array([0.0, -0.304])   # wrist yaw axis in the world x-y plane [m]
_S = {}                            # per-episode state: wrist yaw stroke end points


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    step = int(obs["step"])
    if step == 0 or not _S:
        _S.clear()
        _S.update(ya=P["psic"], yb=P["psic"], yprev=P["psic"], ph=0, Tn=int(P["Tn"]), az=None, turn=0.0)

    # pen centre in the hand frame (x to the thumb side, y along the fingers, origin on the yaw axis)
    psi = -o[0]
    d = o[24:26] - _WRIST
    c, s = np.cos(psi), np.sin(psi)
    xh = c * d[0] + s * d[1]
    yh = -s * d[0] + c * d[1]

    nd, n0 = int(P["nd"]), int(P["n0"])
    sgn, A = P["sgn"], P["A"]

    # thumb zone: hand-frame azimuth (mod pi) at which one end of the pen sweeps past the base of the thumb
    azh = (np.arctan2(o[34], o[33]) - psi) % np.pi
    zw = 1.0 if P["zlo"] <= azh <= P["zhi"] else 0.0

    # turn made so far (unwrapped world azimuth), to compare the pace with the clock
    az = float(np.arctan2(o[34], o[33]))
    if _S["az"] is not None:
        _S["turn"] += (az - _S["az"] + np.pi) % (2 * np.pi) - np.pi
    _S["az"] = az

    # finger-bed slope: uphill away from the knuckles, steeper the further out the pen is
    thf = float(np.clip(P["thf"] + P["ky"] * (yh - P["yt"]), P["thmin"], 0.5))

    if step < n0:                                   # settle: pen lands on the palm and rolls to the knuckles
        w, a, yaw = P["w0"], -sgn * A, P["psic"]
    else:
        n = _S["ph"]
        if n == 0:                                  # new cycle: choose the wrist yaw stroke from the sideways error
            # behind schedule (projected turn at the end of the episode below three revolutions): shorter cycle
            pace = abs(_S["turn"]) / (2 * np.pi) / max(step - n0, 1)
            late = step > P["nchk"] and pace * (int(obs["max_steps"]) - n0 - 60) < 3.0
            _S["Tn"] = int(P["Tnf"]) if late else int(P["Tn"])
            xt = P["xt"] + sgn * P["kxy"] * (yh - P["yt"])
            dw = sgn * P["ff"] * 2 * A + P["kx"] * (xh - xt) - sgn * P["asz"] * zw
            dw = float(np.clip(dw, -P["dwmax"], P["dwmax"]))
            _S["yprev"] = _S["yb"]
            _S["ya"] = P["psic"] - dw / 2
            _S["yb"] = P["psic"] + dw / 2
        Tn = _S["Tn"]
        _S["ph"] = (n + 1) % Tn
        if n < nd:                                  # drop: hand falls away, fingers and wrist yaw reset under the floating pen
            w = P["wd"]
            a = -sgn * P["aod"]
            yaw = _S["yprev"] + P["yod"] * (_S["ya"] - _S["yprev"])
        else:                                       # carry: fingers sweep, wrist comes back up, wrist yaw makes its stroke
            sfr = (n - nd + 1) / (Tn - nd)
            a = sgn * A * (-1 + 2 * sfr) + sgn * P["lead"] * 2 * A / ((Tn - nd) * 0.01)
            a = float(np.clip(a, -0.44, 0.44))
            w = P["wb"] + (P["w0"] - P["wb"]) * min(1.0, sfr / P["rise"])
            yaw = _S["ya"] + (_S["yb"] - _S["ya"]) * sfr

    q = np.zeros(24)                                # joint set points [rad]
    q[_IDX["WRJ1"]] = -yaw
    q[_IDX["WRJ0"]] = w
    b = float(np.clip(thf - o[1], 0.0, 1.5))        # knuckle flexion that gives the bed its world slope
    for f in ("FF", "MF", "RF", "LF"):
        q[_IDX[f + "J3"]] = a
        q[_IDX[f + "J2"]] = b
        q[_IDX[f + "J1"]] = P["b1"]
    q[19:24] = [P["th4"], P["th3"], P["th2"], P["th1"], P["th0"]]

    mean = np.asarray(obs["ctrl_mean"], dtype=float)
    half = np.asarray(obs["ctrl_half_range"], dtype=float)
    return np.clip((q - mean) / half, -1.0, 1.0)
