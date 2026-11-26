"""Linear state-feedback hopping controller for the MuJoCo Hopper.

Control law (one affine map, no internal state, no phase clock):

    a = clip(K @ ((s - MU) / SD) + B, -1, 1)

where ``s = obs["gym_obs"]`` is the 11-vector [z, pitch, thigh, leg, foot, vx, vz,
pitch_rate, thigh_rate, leg_rate, foot_rate].  MU/SD centre and scale the state (nominal
standing height, cruise speed 1.5 m/s; angles 0.5 rad, rates 1-3 rad/s) so every gain in K
is in "command per typical excursion" units.

The hopping gait (knee thrust in stance, knee fold and hip swing in flight, torso pitch
held by the hip) is a limit cycle of this closed loop. Row 0 of K is the hip command,
row 1 the knee, row 2 the ankle; B is the constant bias of each motor. The gains were
found by search, so a single entry has no stand-alone meaning.
The 36 numbers (K: 3x11, B: 3) were found with a derivative-free CMA-ES search
(population 24, return over 4-8 scratch seeds with a penalty on |pitch| > 0.13 rad to keep
margin to the 0.2 rad termination; no gradients, no neural network).
Cruise is about 2.45 m/s (19.6 m in 8 s, return ~3450); max |pitch| about 0.15 rad.
"""

import numpy as np

MU = np.array([1.25, 0, 0, 0, 0, 1.5, 0, 0, 0, 0, 0])
SD = np.array([0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0])

#   columns: z, pitch, thigh, leg, foot, vx, vz, pitch_rate, thigh_rate, leg_rate, foot_rate
K = np.array([
    [-0.28222, -0.70252, -0.09121, -0.21235, +0.36799, -0.37822, -0.02755, -0.57470, -0.51780, +0.48150, +0.14779],
    [+1.03796, +0.02907, -0.58476, -1.42117, +0.03334, -1.34576, +0.79557, -0.38465, +0.37123, -0.47623, +0.39781],
    [+1.12237, +0.58845, +0.25605, -0.09381, -0.03539, -0.71861, -1.01458, -0.24081, -0.66338, +0.47790, -0.28075],
])  # rows: hip, knee, ankle
B = np.array([-0.07530, -0.08652, +0.49533])


def controller(obs: dict) -> np.ndarray:
    x = (np.asarray(obs["gym_obs"], dtype=float) - MU) / SD
    return np.clip(K @ x + B, -1.0, 1.0)
