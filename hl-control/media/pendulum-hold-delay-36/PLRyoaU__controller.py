"""Delay-compensated energy swing-up + LQR catch for the delayed pendulum.

The plant is deterministic and the controller knows the torques it has sent, so the
current state is reconstructed exactly: the measurement from D = obs_delay_steps samples
ago is propagated D steps through the nonlinear model with the stored (clipped) torques
(RK4, zero-order hold).  Control is then designed on that predicted current state:

* swing-up: energy pumping u = clip(K_E * (E0 + E_OFF - E) * sign(theta_dot)) with
  E = theta_dot^2/2 + g cos(theta), E0 = g (upright);
* catch: discrete LQR u = -K x (x = [wrapped theta, theta_dot]) once the predicted
  state is inside the ellipse x' P x < C_IN (the region where LQR is torque-feasible).

PARAMS: K_E energy gain, E_OFF energy target offset above upright (arrive with some speed), C_IN catch-ellipse level, Q_TH / Q_W / R LQR weights.
"""

import math

import numpy as np
from scipy.linalg import solve_discrete_are, expm

G, DAMP, DT = 9.81, 0.05, 0.02
PARAMS = {"K_E": 5.0, "E_OFF": 0.3, "C_IN": 90.0, "Q_TH": 10.0, "Q_W": 1.0, "R": 1.0}
PARAM_BOUNDS = {"K_E": (0.1, 10.0), "E_OFF": (-1.0, 5.0), "C_IN": (20.0, 150.0), "Q_TH": (1.0, 100.0),
                "Q_W": (0.1, 20.0), "R": (0.1, 10.0)}

_cache = {}
_st = {"hist": [], "x0": None}


def _lqr():
    key = tuple(PARAMS.items())
    if _cache.get("key") != key:
        A = np.array([[0, 1], [G, -DAMP]], float)
        B = np.array([[0], [1.0]])
        M = expm(np.block([[A, B], [np.zeros((1, 3))]]) * DT)
        Ad, Bd = M[:2, :2], M[:2, 2:]
        Q = np.diag([PARAMS["Q_TH"], PARAMS["Q_W"]])
        R = np.array([[PARAMS["R"]]])
        P = solve_discrete_are(Ad, Bd, Q, R)
        K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        _cache.update(key=key, K=K, P=P)
    return _cache["K"], _cache["P"]


def _step(th, w, u, n=4):
    h = DT / n
    for _ in range(n):
        def f(t_, w_):
            return w_, G * math.sin(t_) + u - DAMP * w_
        k1 = f(th, w)
        k2 = f(th + 0.5 * h * k1[0], w + 0.5 * h * k1[1])
        k3 = f(th + 0.5 * h * k2[0], w + 0.5 * h * k2[1])
        k4 = f(th + h * k3[0], w + h * k3[1])
        th += h / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w += h / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return th, w


def controller(obs: dict):
    step = int(obs["step"])
    D = int(obs.get("obs_delay_steps", 0))
    lim = float(obs["torque_limit"])
    if step == 0:
        _st["hist"] = []
    hist = _st["hist"]  # hist[k] = torque applied at step k
    # delayed measurement is the state at step max(step - D, 0); propagate with torques since then
    s0 = max(step - D, 0)
    th, w = float(obs["theta"]), float(obs["theta_dot"])
    for k in range(s0, step):
        th, w = _step(th, w, hist[k])
    thw = (th + math.pi) % (2 * math.pi) - math.pi
    K, P = _lqr()
    x = np.array([thw, w])
    if float(x @ P @ x) < PARAMS["C_IN"]:
        u = float(-(K @ x)[0])
    else:
        E = 0.5 * w * w + G * math.cos(th)
        s = 1.0 if w >= 0 else -1.0
        if abs(w) < 1e-3:
            s = 1.0
        u = PARAMS["K_E"] * (G + PARAMS["E_OFF"] - E) * s
    u = max(-lim, min(lim, u))
    hist.append(u)
    return np.array([u])
