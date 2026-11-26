"""Distributed second-order consensus controller over an undirected graph.

Control Law:
For each agent i with neighbors N_i:
    u_i = - k_p * sum_{j in N_i} A_ij * (p_i - p_j) - k_v * sum_{j in N_i} A_ij * (v_i - v_j)
In matrix form:
    u = - k_p * (L @ p) - k_v * (L @ v)
where L = diag(degree) - adjacency is the graph Laplacian matrix.

Role of each gain:
- Proportional gain (k_p = 6.0): Restores spatial consensus by penalizing position
  disagreements between adjacent agents.
- Velocity/Derivative gain (k_v = 4.0): Provides viscous damping on relative motion,
  preventing overshoot and oscillations while aligning agent velocities.
- The combination ensures rapid exponential convergence within the 15-second horizon
  and robust discrete-time stability under sample_dt = 0.05 s and action_limit = 2.0.
"""

import numpy as np


def controller(obs: dict) -> np.ndarray:
    """Compute distributed second-order consensus control actions.

    Parameters
    ----------
    obs : dict
        Observation dictionary containing:
        - "position": np.ndarray of shape (n_agents, 1)
        - "velocity": np.ndarray of shape (n_agents, 1)
        - "adjacency": np.ndarray of shape (n_agents, n_agents)
        - "degree": np.ndarray of shape (n_agents,)
        - "action_limit": float, maximum absolute action magnitude

    Returns
    -------
    np.ndarray
        Control action of shape (n_agents, 1).
    """
    pos = np.asarray(obs["position"], dtype=float)
    vel = np.asarray(obs["velocity"], dtype=float)
    adj = np.asarray(obs["adjacency"], dtype=float)
    deg = np.asarray(obs["degree"], dtype=float)

    # Graph Laplacian: L = diag(degree) - adjacency
    laplacian = np.diag(deg) - adj

    # Relative neighbor position and velocity differences: (L @ x)_i = sum_j A_ij (x_i - x_j)
    pos_diff = laplacian @ pos
    vel_diff = laplacian @ vel

    # PD consensus control gains
    k_p = 6.0
    k_v = 4.0

    action = -k_p * pos_diff - k_v * vel_diff

    action_limit = float(obs.get("action_limit", 2.0))
    return np.clip(action, -action_limit, action_limit)
