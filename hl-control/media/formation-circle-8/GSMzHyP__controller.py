"""Formation controller with short-range collision avoidance."""

import numpy as np


def controller(obs: dict) -> np.ndarray:
    """Track relative offsets with PD feedback and repel nearby agent pairs.

    ``kp`` and ``kd`` set the relative-position stiffness and damping.  ``kr``
    sets the strength of a safety-buffer repulsion that vanishes outside the
    buffer, so it cannot bias the final formation.
    """
    position = np.asarray(obs["position"], dtype=float)
    velocity = np.asarray(obs["velocity"], dtype=float)
    target = np.asarray(obs["desired_offsets"], dtype=float)

    # Remove unobservable/common translation and velocity components.
    error = position - target
    error -= np.mean(error, axis=0, keepdims=True)
    relative_velocity = velocity - np.mean(velocity, axis=0, keepdims=True)

    kp, kd = 16.0, 10.0
    action = -kp * error - kd * relative_velocity

    # Gradient of a compact quadratic pairwise safety potential.
    delta = position[:, None, :] - position[None, :, :]
    distance = np.linalg.norm(delta, axis=2)
    buffer_radius = 1.25 * float(obs["safety_radius"])
    mask = (distance > 1e-9) & (distance < buffer_radius)
    strength = np.where(mask, (buffer_radius - distance) / np.maximum(distance, 1e-9), 0.0)
    kr = 6.0
    action += kr * np.sum(strength[:, :, None] * delta, axis=1)

    limit = float(obs["action_limit"])
    return np.clip(action, -limit, limit)
