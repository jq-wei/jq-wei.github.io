"""Soft-arm reach-and-hold controller.

Law: invert the nominal static tip map (embedded GRID) to get a feed-forward pair of segment differential
commands dp_ff for the current target (nearest map point inside the +-0.6 box, continuing from the previous
solution). On top, an integrator in command space: on each fresh camera frame
    dp += KI * dt_cam * pinv_damped(J(dp)) @ (target - tip)
with J the map Jacobian. This removes the per-seed material offsets (stiffness, chamber gains, weak chamber,
payload, warm-up) and creep/hysteresis drift. Gains:
  KI    integrator rate [1/s] while the tip is moving faster than V_THR [m/s]; KI_FAST when it is quasi-static; LAMBDA  Tikhonov damping of the Jacobian inverse (near-singular map);
  STEP_MAX  clamp on one integrator step; BIAS  co-contraction bias of the chamber commands;
  DP_LIM  box on dp (|dp| > 1 relieves the opposing chamber to reach full scale); BOOST, KP  unused (0).
Between targets a per-segment, per-sign gain (dp settled / dp feed-forward, clipped to [0.8, 1.5]) is learned from
well-settled targets and scales the next feed-forward (compensates a weak chamber or stiff arm). The integrator
runs at KI while the tip moves faster than V_THR and at KI_FAST when it is quasi-static.
"""
import numpy as np

GRID = np.array([[[-0.1238, 0.0434], [-0.1236, 0.0436], [-0.163, 0.0227], [-0.2358, -0.0801], [-0.226, -0.1646], [-0.1855, -0.1987], [-0.1523, -0.1998], [-0.1503, -0.199], [-0.1493, -0.1991]], [[-0.1636, -0.0401], [-0.1662, -0.0439], [-0.1947, -0.0906], [-0.209, -0.1664], [-0.1808, -0.225], [-0.1356, -0.2439], [-0.1004, -0.2345], [-0.0973, -0.2324], [-0.0962, -0.2317]], [[-0.1594, -0.1185], [-0.1605, -0.1223], [-0.1699, -0.1641], [-0.1629, -0.2248], [-0.1251, -0.2672], [-0.0783, -0.2731], [-0.0425, -0.2534], [-0.0369, -0.2472], [-0.0364, -0.2465]], [[-0.1284, -0.1783], [-0.1283, -0.1793], [-0.1263, -0.2137], [-0.1072, -0.2628], [-0.0646, -0.2916], [-0.018, -0.2862], [0.0146, -0.2571], [0.0227, -0.2437], [0.0231, -0.2427]], [[-0.0795, -0.2205], [-0.0792, -0.2216], [-0.0717, -0.2449], [-0.0449, -0.2833], [0.0, -0.3], [0.0449, -0.2833], [0.0717, -0.2449], [0.0792, -0.2216], [0.0795, -0.2205]], [[-0.0231, -0.2427], [-0.0227, -0.2437], [-0.0146, -0.2571], [0.018, -0.2862], [0.0646, -0.2916], [0.1072, -0.2628], [0.1263, -0.2137], [0.1283, -0.1793], [0.1284, -0.1783]], [[0.0364, -0.2465], [0.0369, -0.2472], [0.0425, -0.2534], [0.0783, -0.2731], [0.1251, -0.2672], [0.1629, -0.2248], [0.1699, -0.1641], [0.1605, -0.1223], [0.1594, -0.1185]], [[0.0962, -0.2317], [0.0973, -0.2324], [0.1004, -0.2345], [0.1356, -0.2439], [0.1808, -0.225], [0.209, -0.1664], [0.1947, -0.0906], [0.1662, -0.0439], [0.1636, -0.0401]], [[0.1493, -0.1991], [0.1503, -0.199], [0.1523, -0.1998], [0.1855, -0.1987], [0.226, -0.1646], [0.2358, -0.0801], [0.163, 0.0227], [0.1236, 0.0436], [0.1238, 0.0434]]])

PARAMS = {"KI": 0.55, "LAMBDA": 0.003, "BIAS": 0.1, "DP_LIM": 1.1, "KI_FAST": 1.1, "V_THR": 0.06, "BOOST": 0.0, "KP": 0.0}
PARAM_BOUNDS = {"KI": (0.2, 3.0), "LAMBDA": (0.002, 0.2), "BIAS": (0.05, 0.3), "DP_LIM": (0.6, 1.0), "KI_FAST": (0.5, 4.0), "V_THR": (0.01, 0.1), "BOOST": (0.0, 4.0), "KP": (0.0, 1.0)}

_DPS = np.linspace(-1.0, 1.0, 9)


def _interp(dp):
    d = np.clip(np.asarray(dp, float), -1, 1)
    x = (d + 1.0) / 0.25
    i = np.clip(np.floor(x).astype(int), 0, 7)
    f = x - i
    G = GRID
    return ((1 - f[0]) * (1 - f[1]) * G[i[0], i[1]] + f[0] * (1 - f[1]) * G[i[0] + 1, i[1]]
            + (1 - f[0]) * f[1] * G[i[0], i[1] + 1] + f[0] * f[1] * G[i[0] + 1, i[1] + 1])


def _jac(dp, h=0.05):
    cols = []
    for a in range(2):
        e = np.zeros(2); e[a] = h
        cols.append((_interp(dp + e) - _interp(dp - e)) / (2 * h))
    return np.stack(cols, axis=1)


def _invert(target, start):
    """Nearest-to-start solution of map(dp) = target inside the +-0.6 box (grid seed + damped Newton)."""
    ax = np.linspace(-0.6, 0.6, 25)
    best, bc = None, 1e9
    for a in ax:
        for b in ax:
            dp = np.array([a, b])
            c = np.linalg.norm(_interp(dp) - target) + 0.02 * np.linalg.norm(dp - start)
            if c < bc:
                best, bc = dp, c
    dp = best
    for _ in range(20):
        r = target - _interp(dp)
        J = _jac(dp)
        dp = np.clip(dp + np.linalg.solve(J.T @ J + 1e-4 * np.eye(2), J.T @ r), -0.6, 0.6)
    return dp


class _State:
    def __init__(self):
        self.reset()

    def reset(self):
        self.idx = -1
        self.dp = np.zeros(2)
        self.dp_ff = np.zeros(2)
        self.tgt = None
        self.last_t = -1.0
        self.prev_tip = None
        self.vel = 0.0
        self.err = 1.0
        self.prop = np.zeros(2)
        self.gain = np.ones((2, 2))  # learned [segment, sign(neg, pos)] ratio of settled dp to feed-forward dp


_S = _State()


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    S = _S
    if obs["step"] == 0 or obs["time"] < S.last_t:
        S.reset()
    S.last_t = obs["time"]
    target = np.asarray(obs["target_xz"], float)
    if S.tgt is None or np.linalg.norm(target - S.tgt) > 1e-6:
        if S.tgt is not None and obs["target_mode"] != "circle":
            for a in range(2):  # learn the per-chamber gain from the settled solution of the previous target
                if S.err < 0.008 and abs(S.dp_ff[a]) > 0.25 and S.dp[a] * S.dp_ff[a] > 0:
                    g = S.gain[a, int(S.dp_ff[a] > 0)]
                    S.gain[a, int(S.dp_ff[a] > 0)] = np.clip(0.5 * g + 0.5 * S.dp[a] / S.dp_ff[a], 0.8, 1.5)
        S.tgt = target.copy()
        S.prop = np.zeros(2)
        S.dp_ff = _invert(target, S.dp)
        if obs["target_mode"] != "circle" or S.idx < 0:
            S.dp = np.array([S.dp_ff[a] * S.gain[a, int(S.dp_ff[a] > 0)] for a in range(2)])
            S.dp = np.clip(S.dp, -P["DP_LIM"], P["DP_LIM"])
        S.idx = obs["target_index"]
    if obs["tip_fresh"]:
        tip = np.asarray(obs["tip_xz"], float)
        err = target - tip
        S.err = np.linalg.norm(err)
        if S.prev_tip is not None:
            v = np.linalg.norm(tip - S.prev_tip) / obs["camera_dt"]
            S.vel = 0.6 * S.vel + 0.4 * v
        S.prev_tip = tip
        ki = P["KI"] if S.vel > P["V_THR"] else P["KI_FAST"]
        ki *= 1.0 + P["BOOST"] * min(np.linalg.norm(err) / 0.05, 1.0)
        J = _jac(np.clip(S.dp, -0.9, 0.9))
        step = np.linalg.solve(J.T @ J + P["LAMBDA"] * np.eye(2), J.T @ err)
        S.dp = np.clip(S.dp + ki * obs["camera_dt"] * step, -P["DP_LIM"], P["DP_LIM"])
        S.prop = P["KP"] * step
    return np.clip(_chamber(np.clip(S.dp + S.prop, -1.11, 1.11), P["BIAS"]), 0, 1)


def _chamber(dp, bias):
    """Chamber commands for segment differential commands dp (pressure difference (1-bias) dp p_max, both chambers
    carry >= bias p_max); beyond |dp| = 1 the opposing chamber is relieved to extend the reach to full scale."""
    out = np.zeros(4)
    for a in range(2):
        d = (1.0 - bias) * dp[a]
        hi = min(1.0, abs(d) + bias)
        lo = bias if hi < 1.0 else max(0.0, 1.0 - abs(d))
        out[2 * a], out[2 * a + 1] = (hi, lo) if d >= 0 else (lo, hi)
    return out
