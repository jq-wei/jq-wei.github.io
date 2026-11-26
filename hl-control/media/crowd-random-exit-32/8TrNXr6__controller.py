"""Decentralised crowd-evacuation controller: sampling MPC with a braking fallback.

Control law (run independently by every agent at every step, no shared state):

1. Preferred velocity (funnel).  The agent aims at the nearest "lane" of the
   exit opening: its own y clamped into the opening minus LANE_MARGIN, so the
   crowd uses the whole exit width.  An agent with lateral offset dy from its
   lane aims FUNNEL_SLOPE*dy metres in front of the exit plane (capped at
   FUNNEL_MAX), so the crowd approaches inside a cone instead of sliding along
   the right wall; agents already in their lane aim TARGET_OVERSHOOT beyond
   the plane for a square crossing.  Preferred speed is V_PREF.

2. Candidate accelerations.  A fan of constant accelerations (N_DIRS
   directions x ACC_MAGS magnitudes, plus zero and a "track the preferred
   velocity" candidate) is simulated with the plant's own discrete dynamics:
   the acceleration is applied for TAU_ACT seconds, then the agent brakes
   with A_BRAKE until it stops.  Checking the whole act-then-brake trajectory
   guarantees that whatever is chosen can still be stopped safely.

3. Reciprocal neighbour prediction.  Every neighbour is predicted with the
   same act-then-brake model: constant velocity for TAU_ACT, then braking.
   Because all agents run this law, the assumption is self-consistent; it is
   the key to avoiding the panic cascades that a constant-velocity
   prediction produces (an agent fleeing from a neighbour who would in
   fact have braked, straight into the agent behind it).

4. Costs.  Summed over the whole trajectory (so moving apart is rewarded
   even when the current spacing is already violated): W_HARD per
   (metre * s) of separation below D_HARD (collision is below 0.24 m) and
   W_SOFT per (metre^2 * s) below the comfort distance D_SOFT.  Any predicted
   wall contact, including reaching the right wall outside the opening
   (shrunk by EXIT_MARGIN), costs W_WALL; hugging the right wall closer than
   WALL_SOFT outside the opening costs W_WALL_SOFT per (metre^2 * s).  The goal
   term is the squared error between the velocity at the end of the acting
   phase and the preferred velocity, plus W_ACC * |a|^2 for smoothness.  The
   lowest-cost candidate is returned as the acceleration.
"""

from __future__ import annotations

import numpy as np

# --- gains / parameters -------------------------------------------------------
V_PREF = 1.2          # preferred cruising speed [m/s]
LANE_MARGIN = 0.22    # keep this far from the exit jambs when choosing a lane [m]
TARGET_OVERSHOOT = 0.4  # aim this far beyond the exit plane for a square approach [m]
FUNNEL_SLOPE = 1.2    # standoff in front of the exit per metre of lateral offset
FUNNEL_MAX = 3.0      # cap on the standoff [m]
WALL_SOFT = 0.35      # soft clearance from the right wall outside the opening [m]
W_WALL_SOFT = 300.0   # cost per (metre^2 * s) of soft wall penetration

TAU_ACT = 0.30        # duration the candidate acceleration is applied [s]
A_BRAKE = 2.5         # deceleration of the braking fallback [m/s^2]

D_HARD = 0.34         # hard separation requirement (collision is < 0.24) [m]
D_SOFT = 0.50         # comfort separation [m]
W_HARD = 5000.0       # cost per (metre * s) of hard-separation penetration
W_SOFT = 200.0        # cost per (metre^2 * s) of comfort penetration
WALL_MARGIN = 0.03    # extra clearance kept from the walls [m]
EXIT_MARGIN = 0.06    # y clearance from the exit jambs when crossing the wall plane [m]
W_WALL = 3000.0       # cost of any predicted wall contact
W_ACC = 0.01          # smoothing weight on |a|^2

N_DIRS = 16
ACC_MAGS = (1.0, 2.0, 3.0)

_angles = np.linspace(0.0, 2.0 * np.pi, N_DIRS, endpoint=False)
_fan = np.stack([np.cos(_angles), np.sin(_angles)], axis=1)
CANDIDATES = np.concatenate(
    [np.zeros((1, 2))] + [m * _fan for m in ACC_MAGS], axis=0
)  # (49, 2)


def _simulate(
    pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, dt: float, v_max: float
) -> tuple[np.ndarray, np.ndarray]:
    """Roll out act-then-brake trajectories for every candidate acceleration.

    Returns positions (C, H, 2) and the velocity at the end of the acting phase (C, 2).
    """
    n_act = max(1, int(round(TAU_ACT / dt)))
    n_brake = int(np.ceil(v_max / (A_BRAKE * dt))) + 1
    horizon = n_act + n_brake
    c = acc.shape[0]
    v = np.repeat(vel[None, :], c, axis=0)
    p = np.repeat(pos[None, :], c, axis=0)
    traj = np.empty((c, horizon, 2))
    v_act = None
    for k in range(horizon):
        if k < n_act:
            v = v + dt * acc
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            v = v * np.minimum(1.0, v_max / np.maximum(speed, 1e-12))
        else:
            if v_act is None:
                v_act = v.copy()
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            dv = np.minimum(speed, A_BRAKE * dt)
            v = v - dv * v / np.maximum(speed, 1e-12)
        p = p + dt * v
        traj[:, k] = p
    if v_act is None:
        v_act = v
    return traj, v_act


def _predict_braking(p: np.ndarray, v: np.ndarray, dt: float, horizon: int) -> np.ndarray:
    """Neighbour prediction: keep velocity for TAU_ACT, then brake with A_BRAKE."""
    n_act = max(1, int(round(TAU_ACT / dt)))
    out = np.empty((horizon, p.shape[0], 2))
    p = p.copy()
    v = v.copy()
    for k in range(horizon):
        if k >= n_act:
            speed = np.linalg.norm(v, axis=1, keepdims=True)
            dv = np.minimum(speed, A_BRAKE * dt)
            v = v - dv * v / np.maximum(speed, 1e-12)
        p = p + dt * v
        out[k] = p
    return out


def controller(obs: dict) -> np.ndarray:
    pos = np.asarray(obs["self_pos"], dtype=float)
    vel = np.asarray(obs["self_vel"], dtype=float)
    dt = float(obs["sample_dt"])
    v_max = float(obs["max_speed"])
    a_max = float(obs["max_accel"])
    r = float(obs["agent_radius"])
    hw = 0.5 * float(obs["exit_width"])
    wd = np.asarray(obs["nearest_wall_distances"], dtype=float)  # left, right, bottom, top
    left, right = pos[0] - wd[0], pos[0] + wd[1]
    bottom, top = pos[1] - wd[2], pos[1] + wd[3]
    cy = pos[1] - float(obs["exit_centerline_offset"])

    # 1. preferred velocity: funnel toward the nearest lane of the exit.  An agent
    #    with lateral offset dy from its lane aims at a point FUNNEL_SLOPE*dy in
    #    front of the exit plane so that it approaches inside a cone instead of
    #    sliding along the right wall.
    lane = max(hw - LANE_MARGIN, 0.0)
    y_lane = cy + np.clip(pos[1] - cy, -lane, lane)
    dy = abs(pos[1] - y_lane)
    standoff = np.clip(FUNNEL_SLOPE * dy, 0.0, FUNNEL_MAX)
    target = np.array([right + TARGET_OVERSHOOT - standoff, y_lane])
    if target[0] - pos[0] < 0.3 and dy > 0.05:
        target[0] = min(target[0], pos[0] - 0.3)
    to_target = target - pos
    v_pref = V_PREF * to_target / max(np.linalg.norm(to_target), 1e-9)

    # 2. candidates (fan + tracking candidate), clipped to the actuator limit
    track = (v_pref - vel) / (2.0 * dt)
    track = track * min(1.0, a_max / max(np.linalg.norm(track), 1e-12))
    cands = np.concatenate([CANDIDATES, track[None, :]], axis=0)
    traj, v_act = _simulate(pos, vel, cands, dt, v_max)
    c, horizon, _ = traj.shape

    # alive mask: once the agent has crossed the exit plane inside the opening it is gone
    x, y = traj[..., 0], traj[..., 1]
    in_opening = np.abs(y - cy) <= hw - EXIT_MARGIN
    evacuated = (x >= right - r) & in_opening
    alive = np.cumsum(evacuated, axis=1) == 0  # (C, H)

    cost = np.sum((v_act - v_pref) ** 2, axis=1) + W_ACC * np.sum(cands**2, axis=1)

    # 3a. wall cost
    wall_hit = (
        (x < left + r + WALL_MARGIN)
        | (y < bottom + r + WALL_MARGIN)
        | (y > top - r - WALL_MARGIN)
        | ((x > right - r - WALL_MARGIN) & ~in_opening)
    )
    cost += W_WALL * np.sum(wall_hit & alive, axis=1)
    wall_soft = np.maximum(0.0, x - (right - r - WALL_SOFT)) * (~in_opening) * alive
    cost += W_WALL_SOFT * np.sum(wall_soft**2, axis=1) * dt

    # 3b. neighbour cost (reciprocal act-then-brake prediction)
    mask = np.asarray(obs["nearby_agents_mask"], dtype=bool)
    if np.any(mask):
        rel_p = np.asarray(obs["nearby_agents_rel_pos"], dtype=float)[mask]
        rel_v = np.asarray(obs["nearby_agents_rel_vel"], dtype=float)[mask]
        nbr_p = pos + rel_p  # (N, 2)
        nbr_v = vel + rel_v
        pred = _predict_braking(nbr_p, nbr_v, dt, horizon)  # (H, N, 2)
        d = np.linalg.norm(traj[:, :, None, :] - pred[None, :, :, :], axis=3)  # (C, H, N)
        d = np.where(alive[:, :, None], d, np.inf)
        # penetration summed over the whole horizon (and all neighbours): a
        # trajectory that moves apart is rewarded even if the current
        # separation is already violated
        hard = np.maximum(0.0, D_HARD - d)
        soft = np.maximum(0.0, D_SOFT - d)
        cost += W_HARD * np.sum(hard, axis=(1, 2)) * dt
        cost += W_SOFT * np.sum(soft**2, axis=(1, 2)) * dt

    best = int(np.argmin(cost))
    return cands[best].astype(float)
