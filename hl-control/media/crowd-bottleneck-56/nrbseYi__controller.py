"""Decentralized, predictive social-force controller for the bottleneck."""

import numpy as np


def controller(obs: dict) -> np.ndarray:
    """Track the exit while maintaining predictive clearance.

    The velocity gain tracks a density-limited cruise speed; the centreline
    gain forms a queue through the pillar gap.  Radial, closing-speed, and
    closest-approach gains respectively preserve spacing, damp convergence,
    and anticipate crossing paths.  Pillar/wall gains enforce clearance, with
    a stopping-distance guard reserving acceleration at the exit jamb.
    """
    vel = np.asarray(obs["self_vel"], dtype=float)
    goal = np.asarray(obs["goal_direction"], dtype=float)
    max_accel = float(obs["max_accel"])
    max_speed = float(obs["max_speed"])
    radius = float(obs["agent_radius"])

    # Slow modestly in a crowd and more strongly in the pillar/exit approach.
    x_to_wall = float(obs["exit_wall_distance"])
    density = float(obs["local_density"])
    cruise = min(1.32, 0.88 * max_speed) / (1.0 + 0.055 * density)
    if x_to_wall < 5.0:
        cruise = min(cruise, 1.18)
    desired_velocity = cruise * goal
    offset = float(obs["exit_centerline_offset"])
    exit_half_width = 0.5 * float(obs["exit_width"])
    # Treat the exit like a gate: near the wall, remove forward velocity until
    # the agent's centre is comfortably inside the opening.
    if x_to_wall < 1.35 and abs(offset) > 0.72 * exit_half_width:
        gate_fraction = np.clip(
            (exit_half_width - abs(offset)) / max(0.28 * exit_half_width, 0.08),
            0.0,
            1.0,
        )
        desired_velocity[0] = cruise * gate_fraction
        desired_velocity[1] = np.clip(-2.8 * offset, -1.15, 1.15)
    action = 3.2 * (desired_velocity - vel)  # velocity_gain

    # Pull toward the center of the only usable pillar gap before reaching it.
    if 0.0 < x_to_wall < 5.1:
        centre_gain = (
            0.75
            + 1.25 * np.exp(-((x_to_wall - 3.8) / 1.6) ** 2)
            + 2.2 * np.exp(-(x_to_wall / 0.9) ** 2)
        )
        action[1] -= centre_gain * offset

    # Pairwise barrier.  The closest-point prediction detects crossing paths;
    # the radial damping removes closing speed without damping separation.
    rel_pos = np.asarray(obs["nearby_agents_rel_pos"], dtype=float)
    rel_vel = np.asarray(obs["nearby_agents_rel_vel"], dtype=float)
    distances = np.asarray(obs["nearby_agents_dist"], dtype=float)
    mask = np.asarray(obs["nearby_agents_mask"], dtype=bool)
    safe = 2.85 * radius
    influence = 0.78
    for p, dv, distance in zip(rel_pos[mask], rel_vel[mask], distances[mask]):
        d = max(float(distance), 1.0e-6)
        away = -p / d
        closing = max(0.0, -float(np.dot(p / d, dv)))

        strength = 0.0
        if d < influence:
            strength += 7.5 * (influence - d) ** 2 / max(d - 2.0 * radius + 0.035, 0.04)
            strength += 3.8 * closing * max(0.0, (influence - d) / (influence - safe))

        vv = float(np.dot(dv, dv))
        if vv > 1.0e-8 and float(np.dot(p, dv)) < 0.0:
            t_ca = np.clip(-float(np.dot(p, dv)) / vv, 0.0, 1.15)
            future = p + t_ca * dv
            future_d = float(np.linalg.norm(future))
            if future_d < 0.48:
                future_away = -future / max(future_d, 1.0e-6)
                weight = (0.48 - future_d) / 0.48 * (1.15 - t_ca) / 1.15
                action += 2.4 * weight * future_away
        action += strength * away

    # Circular pillar barriers, using the supplied surface clearance.
    obstacle_pos = np.asarray(obs["nearby_obstacles_rel_pos"], dtype=float)
    obstacle_dist = np.asarray(obs["nearby_obstacles_dist"], dtype=float)
    obstacle_mask = np.asarray(obs["nearby_obstacles_mask"], dtype=bool)
    for p, clearance in zip(obstacle_pos[obstacle_mask], obstacle_dist[obstacle_mask]):
        centre_d = max(float(np.linalg.norm(p)), 1.0e-6)
        away = -p / centre_d
        closing = max(0.0, float(np.dot(vel, p / centre_d)))
        if clearance < 0.75:
            strength = 5.0 * (0.75 - clearance) ** 2 / max(clearance + 0.07, 0.07)
            strength += 4.0 * closing * max(0.0, (0.55 - clearance) / 0.55)
            action += strength * away

    # Keep a radius-sized margin from the solid portions of all room walls.
    walls = np.asarray(obs["nearest_wall_distances"], dtype=float)
    wall_normals = np.array(((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)))
    near_future_offset = offset + float(obs["sample_dt"]) * vel[1]
    for i, distance in enumerate(walls):
        # The right wall is open only inside the exit corridor.
        if i == 1 and abs(near_future_offset) < exit_half_width - 0.015:
            continue
        clearance = float(distance) - radius
        influence = 0.95 if i == 1 else 0.72
        if clearance < influence:
            normal = wall_normals[i]
            inward_speed = max(0.0, -float(np.dot(vel, normal)))
            action += (9.0 * (influence - clearance) + 4.5 * inward_speed) * normal

    norm = float(np.linalg.norm(action))
    if norm > max_accel:
        action *= max_accel / norm

    # Last-resort stopping-distance guard at the exit jamb.  It only activates
    # when lateral motion makes the opening unsafe, and reserves enough of the
    # acceleration circle to stop before the right wall.
    outward_speed = max(0.0, np.sign(offset) * float(vel[1]))
    lateral_risk = abs(offset) + 0.20 * outward_speed
    if x_to_wall < 0.70 and lateral_risk > exit_half_width - 0.025:
        room = max(x_to_wall - radius - 0.025, 0.0)
        safe_forward_speed = np.sqrt(2.0 * 2.75 * room)
        required_ax = (safe_forward_speed - vel[0]) / float(obs["sample_dt"])
        if action[0] > required_ax:
            action[0] = max(-max_accel, required_ax)
            y_capacity = np.sqrt(max(max_accel**2 - action[0] ** 2, 0.0))
            action[1] = np.clip(action[1], -y_capacity, y_capacity)
    return action.astype(float)
