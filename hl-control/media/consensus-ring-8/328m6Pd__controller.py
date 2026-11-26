"""Distributed Laplacian consensus controller for single integrators.

Control law (fully distributed: agent i only uses its own adjacency row and
the states of its neighbours):

    u_i = -K * sum_j a_ij * (x_i - x_j)

i.e. each agent moves toward the average of its neighbours. For the sampled
single-integrator plant (zero-order hold, dt = 0.05) each Laplacian mode
evolves as x+ = (1 - K*dt*lambda) x, so stability requires
K * dt * lambda_max(L) < 2. The ring graph on 8 agents has lambda_max = 4,
giving K < 10. K = 8.0 is chosen inside that bound with margin while keeping
the slowest mode (lambda = 2 - sqrt(2)) contracting fast enough to reach
consensus_error <= 1e-3 well within the 10 s horizon. Actions saturate at
the environment's limit of +/-2, which only slows the initial transient.
"""

import numpy as np

K = 8.0  # consensus gain; must satisfy K * sample_dt * lambda_max(L) < 2


def controller(obs: dict) -> np.ndarray:
    x = np.asarray(obs["x"], dtype=float)              # (n_agents, state_dim)
    adjacency = np.asarray(obs["adjacency"], dtype=float)
    degree = adjacency.sum(axis=1, keepdims=True)      # (n_agents, 1)

    # Laplacian feedback: L x = degree * x - A x  (uses only neighbour states).
    u = -K * (degree * x - adjacency @ x)

    limit = float(obs["action_limit"])
    return np.clip(u, -limit, limit)
