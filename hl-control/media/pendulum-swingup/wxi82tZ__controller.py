"""Energy-shaping swing-up followed by local upright stabilization."""

import numpy as np


def controller(obs: dict) -> np.ndarray:
    """Swing up by regulating energy, then catch with a saturated PD law.

    ``energy_gain`` pumps/removes energy on the way to the upright energy.
    Near upright, ``kp`` overcomes gravity and ``kd`` brakes angular motion.
    An initial pulse breaks the otherwise invariant exactly-downward equilibrium.
    """
    theta = (float(obs["theta"]) + np.pi) % (2.0 * np.pi) - np.pi
    theta_dot = float(obs["theta_dot"])
    torque_limit = float(obs["torque_limit"])

    gravity = 9.81
    energy = 0.5 * theta_dot**2 + gravity * np.cos(theta)
    energy_error = gravity - energy

    if int(obs["step"]) < 15:
        torque = torque_limit
    elif abs(theta) < 0.35 and abs(theta_dot) < 2.0:
        kp, kd = 18.0, 5.0
        torque = -kp * theta - kd * theta_dot
    else:
        energy_gain = 1.5
        torque = energy_gain * energy_error * theta_dot

    return np.array([np.clip(torque, -torque_limit, torque_limit)], dtype=float)
