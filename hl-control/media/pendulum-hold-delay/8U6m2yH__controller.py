"""Delay-compensating swing-up and hold controller for the torque-limited pendulum.

The measurement is ``obs_delay_steps`` samples old, but the plant model is known
exactly and the controller remembers every torque it has applied.  It therefore
replays its own torque history through the pendulum dynamics (RK4, a few
sub-steps per sample) starting from the delayed measurement, which yields a
prediction of the *current* state (a Smith-predictor / state-predictor scheme).
The control law then acts on the predicted state as if there were no delay:

* Swing-up (energy pumping): with E = thd^2/2 + g cos(theta) (per unit m l^2)
  and E0 = g the energy of the upright rest state, dE/dt = thd * u, so
  u = K_E * (E0 - E) * sign(thd) drives the energy toward E0 (saturating at
  the torque limit).  A tiny kick is used while the pendulum is at rest.
* Hold (linear state feedback around upright): u = -(K_TH * theta + K_THD * thd),
  which places the closed-loop poles of  thdd = g theta + u  at  s^2 + K_THD s +
  (K_TH - g).  Engaged when the predicted state is inside the catch region
  (|theta| < CATCH_ANGLE and energy close to E0), with hysteresis so a brief
  excursion does not drop back to swing-up unless the pendulum has really fallen.

Gains
-----
K_E          energy-pumping gain (torque per unit energy error)
K_TH, K_THD  angle / rate gains of the upright feedback
CATCH_ANGLE  predicted |theta| below which the hold law takes over
CATCH_ENERGY |E - E0| tolerance for engaging the hold law
DROP_ANGLE   predicted |theta| above which we fall back to swing-up
"""

from __future__ import annotations

import math

import numpy as np

G = 9.81
DAMPING = 0.05
INERTIA = 1.0  # m l^2
E0 = G        # energy of the upright rest state (per unit inertia)

PARAMS = {
    "K_E": 5.0,
    "K_TH": 30.0,
    "K_THD": 8.0,
    "CATCH_ANGLE": 0.6,
    "CATCH_ENERGY": 2.5,
    "DROP_ANGLE": 1.0,
}
PARAM_BOUNDS = {
    "K_E": (0.3, 10.0),
    "K_TH": (15.0, 60.0),
    "K_THD": (4.0, 16.0),
    "CATCH_ANGLE": (0.2, 0.8),
    "CATCH_ENERGY": (0.5, 4.0),
    "DROP_ANGLE": (0.6, 1.5),
}

SUBSTEPS = 4  # RK4 sub-steps per sample in the state predictor


def wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _f(theta: float, thd: float, u: float) -> tuple[float, float]:
    return thd, (G * math.sin(theta) + u / INERTIA - DAMPING * thd)


def _rk4_step(theta: float, thd: float, u: float, dt: float) -> tuple[float, float]:
    h = dt / SUBSTEPS
    for _ in range(SUBSTEPS):
        k1 = _f(theta, thd, u)
        k2 = _f(theta + 0.5 * h * k1[0], thd + 0.5 * h * k1[1], u)
        k3 = _f(theta + 0.5 * h * k2[0], thd + 0.5 * h * k2[1], u)
        k4 = _f(theta + h * k3[0], thd + h * k3[1], u)
        theta += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        thd += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    return theta, thd


class _State:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.torques: list[float] = []   # torque applied at step i
        self.holding = False
        self.last_step = -1


_S = _State()


def _predict(obs: dict) -> tuple[float, float]:
    """Advance the delayed measurement through the applied torque history."""
    k = int(obs["step"])
    delay = int(obs.get("obs_delay_steps", 0))
    dt = float(obs["sample_dt"])
    s = max(k - delay, 0)  # step index the measurement belongs to
    theta, thd = float(obs["theta"]), float(obs["theta_dot"])
    for i in range(s, k):
        u = _S.torques[i] if i < len(_S.torques) else 0.0
        theta, thd = _rk4_step(theta, thd, u, dt)
    return wrap(theta), thd


def controller(obs: dict) -> np.ndarray:
    k = int(obs["step"])
    if k == 0 or k <= _S.last_step:
        _S.reset()
    _S.last_step = k
    p = PARAMS
    limit = float(obs.get("torque_limit", 2.0))

    theta, thd = _predict(obs)

    energy = 0.5 * thd * thd + G * math.cos(theta)
    e_err = E0 - energy

    # mode switching with hysteresis
    if _S.holding:
        if abs(theta) > p["DROP_ANGLE"]:
            _S.holding = False
    else:
        if abs(theta) < p["CATCH_ANGLE"] and abs(e_err) < p["CATCH_ENERGY"]:
            _S.holding = True

    if _S.holding:
        u = -(p["K_TH"] * theta + p["K_THD"] * thd)
    else:
        if abs(thd) < 1e-3:
            direction = 1.0 if math.sin(theta) >= 0.0 else -1.0  # push away from hanging
        else:
            direction = 1.0 if thd > 0.0 else -1.0
        u = p["K_E"] * e_err * direction

    u = float(np.clip(u, -limit, limit))
    while len(_S.torques) < k:
        _S.torques.append(0.0)  # should not happen; keeps indices aligned
    _S.torques.append(u)
    return np.array([u])
