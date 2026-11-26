"""Soft-arm tip controller: nominal-map feedforward + integral feedback in differential-command space.

Law
  1. The target tip position is inverted through the nominal static tip map (9x9 grid embedded below, bilinear,
     the same interpolation the targets are drawn with) to a feedforward differential command dp_ff (box +-BOX).
  2. dp = dp_ff + I, where I integrates the tip error through a damped inverse of the map Jacobian on every
     camera frame:  I += KI * dt_cam * J^+ (target - tip_filtered).  The integrator absorbs the per-seed material
     (stiffness, chamber gains), hysteresis, creep, payload and drift that the nominal map does not know.
  3. A velocity term KD (on the filtered tip velocity, through J^+) damps the 1.25 Hz bending mode.
  4. dp is converted to the four chamber commands with chamber_commands(dp, BIAS) (co-contraction bias keeps both
     chambers pressurised so corrections of either sign are possible).
Gains: KI integral rate [1/s], KD damping [s], LAM Jacobian regularisation [m/unit], BIAS co-contraction,
       ALPHA tip low-pass factor per frame, ILIM integrator clamp.
"""

import numpy as np

PARAMS = {"KI": 0.8, "KD": 0.05, "LAM": 0.03, "BIAS": 0.1, "ALPHA": 0.6, "ILIM": 0.5}
PARAM_BOUNDS = {"KI": (0.1, 3.0), "KD": (0.0, 0.3), "LAM": (0.005, 0.1), "BIAS": (0.05, 0.4), "ALPHA": (0.2, 1.0),
                "ILIM": (0.2, 0.8)}
BOX = 0.65

_DP = np.linspace(-1.0, 1.0, 9)
_TIP = np.array([[[-0.12384106075980639, 0.043410641794503234], [-0.12357757470057554, 0.04356477012752022], [-0.16301738977362576, 0.022716459257660544], [-0.23581792768948712, -0.08007579153559215], [-0.22604269508312522, -0.16457251487273153], [-0.18545413151677798, -0.19871413029145474], [-0.15228869450880997, -0.19976635694500752], [-0.15030158440786037, -0.19903712790602462], [-0.1493062845677774, -0.19910546482718447]], [[-0.1635511815538848, -0.040074344864967644], [-0.16618850428570658, -0.04392809841426836], [-0.19467268479421848, -0.09061300023569302], [-0.20895557602180007, -0.16639454028278094], [-0.18079148370734033, -0.22503653850888894], [-0.13562212418365832, -0.243908060442817], [-0.10036076824576684, -0.2345141456658486], [-0.09726336588677507, -0.23239039686032648], [-0.09622508279669674, -0.23170556994798627]], [[-0.15935804175257895, -0.1185127479140701], [-0.1605277613878079, -0.1223021250487502], [-0.16989513909649973, -0.16406900679435116], [-0.1628562610502433, -0.2248469546804269], [-0.12505621649650395, -0.2671745615582458], [-0.0783142095073398, -0.27310716857334727], [-0.042464287670311435, -0.25340999562445965], [-0.03690909470233906, -0.24717233223446392], [-0.036364111244465536, -0.24645488011128366]], [[-0.12837924009995938, -0.17826331267254278], [-0.12827451591136363, -0.17929387454726334], [-0.1262656841751351, -0.21374949638266738], [-0.10719839989525329, -0.26280365994037064], [-0.06460997292774551, -0.291628212088276], [-0.017992412779893732, -0.2862164811323505], [0.014585275526315191, -0.257139043886386], [0.02273216546185285, -0.24369952307443432], [0.023148720839499908, -0.24274323643122775]], [[-0.07948627500439147, -0.2205290439523658], [-0.07924471889589957, -0.22162954763329246], [-0.07174132032418842, -0.24490389262814216], [-0.04492564471022784, -0.2833222464526185], [0.0, -0.3], [0.04492564471022784, -0.2833222464526185], [0.07174132032418842, -0.24490389262814216], [0.07924471889589957, -0.22162954763329246], [0.07948627500439147, -0.2205290439523658]], [[-0.023148720839499908, -0.24274323643122775], [-0.02273216546185285, -0.24369952307443432], [-0.014585275526315191, -0.257139043886386], [0.017992412779893732, -0.2862164811323505], [0.06460997292774551, -0.291628212088276], [0.10719839989525329, -0.26280365994037064], [0.1262656841751351, -0.21374949638266738], [0.12827451591136363, -0.17929387454726334], [0.12837924009995938, -0.17826331267254278]], [[0.036364111244465536, -0.24645488011128366], [0.03690909470233906, -0.24717233223446392], [0.042464287670311435, -0.25340999562445965], [0.0783142095073398, -0.27310716857334727], [0.12505621649650395, -0.2671745615582458], [0.1628562610502433, -0.2248469546804269], [0.16989513909649973, -0.16406900679435116], [0.1605277613878079, -0.1223021250487502], [0.15935804175257895, -0.1185127479140701]], [[0.09622508279669674, -0.23170556994798627], [0.09726336588677507, -0.23239039686032648], [0.10036076824576684, -0.2345141456658486], [0.13562212418365832, -0.243908060442817], [0.18079148370734033, -0.22503653850888894], [0.20895557602180007, -0.16639454028278094], [0.19467268479421848, -0.09061300023569302], [0.16618850428570658, -0.04392809841426836], [0.1635511815538848, -0.040074344864967644]], [[0.1493062845677774, -0.19910546482718447], [0.15030158440786037, -0.19903712790602462], [0.15228869450880997, -0.19976635694500752], [0.18545413151677798, -0.19871413029145474], [0.22604269508312522, -0.16457251487273153], [0.23581792768948712, -0.08007579153559215], [0.16301738977362576, 0.022716459257660544], [0.12357757470057554, 0.04356477012752022], [0.12384106075980639, 0.043410641794503234]]])


def _tip_map(dp):
    d = np.clip(np.asarray(dp, dtype=float), -1.0, 1.0)
    x = (d + 1.0) / 0.25
    i = np.clip(np.floor(x).astype(int), 0, 7)
    f = x - i
    return ((1 - f[0]) * (1 - f[1]) * _TIP[i[0], i[1]] + f[0] * (1 - f[1]) * _TIP[i[0] + 1, i[1]]
            + (1 - f[0]) * f[1] * _TIP[i[0], i[1] + 1] + f[0] * f[1] * _TIP[i[0] + 1, i[1] + 1])


def _jac(dp, h=0.05):
    cols = []
    for a in range(2):
        e = np.zeros(2)
        e[a] = h
        cols.append((_tip_map(dp + e) - _tip_map(dp - e)) / (2 * h))
    return np.stack(cols, axis=1)


_g = np.linspace(-BOX, BOX, 66)
_FINE = np.array([[_tip_map(np.array([a, b])) for b in _g] for a in _g])


def _invert(target, prev=None):
    err = np.linalg.norm(_FINE - target, axis=2)
    if prev is not None:  # prefer the branch near the previous solution
        err = err + 0.15 * np.linalg.norm(np.stack(np.meshgrid(_g, _g, indexing="ij"), axis=2) - prev, axis=2)
    i, j = np.unravel_index(np.argmin(err), err.shape)
    dp = np.array([_g[i], _g[j]])
    for _ in range(8):  # Newton polish
        r = target - _tip_map(dp)
        dp = np.clip(dp + np.linalg.pinv(_jac(dp), rcond=0.05) @ r * 0.7, -BOX, BOX)
    return dp


_S = {}


def _reset():
    _S.clear()
    _S.update(I=np.zeros(2), filt=None, prev_filt=None, tkey=None, dpff=None, dp=np.zeros(2))


def controller(obs: dict):
    p = PARAMS
    if obs["step"] == 0 or not _S:
        _reset()
    target = np.asarray(obs["target_xz"], dtype=float)
    key = tuple(np.round(target, 5))
    if key != _S["tkey"]:
        _S["tkey"] = key
        _S["dpff"] = _invert(target, _S["dpff"])
    dpff = _S["dpff"]
    if obs["tip_fresh"]:
        tip = np.asarray(obs["tip_xz"], dtype=float)
        cdt = obs["camera_dt"]
        if _S["filt"] is None:
            _S["filt"] = tip.copy()
            vel = np.zeros(2)
        else:
            new = _S["filt"] + p["ALPHA"] * (tip - _S["filt"])
            vel = (new - _S["filt"]) / cdt
            _S["filt"] = new
        dp = np.clip(dpff + _S["I"], -1.0, 1.0)
        J = _jac(np.clip(dp, -BOX, BOX))
        U, s, Vt = np.linalg.svd(J)
        Jp = Vt.T @ np.diag(s / (s ** 2 + p["LAM"] ** 2)) @ U.T
        err = target - _S["filt"]
        _S["I"] = np.clip(_S["I"] + p["KI"] * cdt * (Jp @ err), -p["ILIM"], p["ILIM"])
        _S["damp"] = -p["KD"] * (Jp @ vel)
    dp = dpff + _S["I"] + _S.get("damp", 0.0)
    out = np.clip(dp, -1.0, 1.0)
    pos = p["BIAS"] + (1 - p["BIAS"]) * np.maximum(out, 0.0)
    neg = p["BIAS"] + (1 - p["BIAS"]) * np.maximum(-out, 0.0)
    return np.array([pos[0], neg[0], pos[1], neg[1]])
