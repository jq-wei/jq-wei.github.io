"""Linear full-state feedback hopping controller for the MuJoCo Hopper (Hopper-v5).

Control law:  u = clip(K @ s, -1, 1),  with the feature vector
    s = [ (z - 1.25)/0.1, pitch/0.1, thigh/0.5, leg/0.5, foot/0.5,
          x_vel/1, z_vel/1, pitch_rate/2, thigh_rate/5, leg_rate/5, foot_rate/5,
          foot_contact, 1 ]
built from the Hopper-v5 observation (obs["gym_obs"]), each entry divided by a typical
magnitude and clipped to +-5. Columns 0-10 of K are state-feedback gains (proportional on
the posture and height, derivative on the rates), so the
row of each motor reads as a PD law around a moving set point. Column 11 switches the
set point between stance and flight (the contact flag makes it a two-phase hybrid law)
and column 12 is the constant bias (the nominal posture torques).  Rows: hip (thigh),
knee (leg), ankle (foot).

How it hops: the limit cycle comes from the feedback loop itself, with no clock.  The
pitch and pitch-rate columns of the hip row hold the torso inside the +-0.2 rad band. The
knee and ankle rows push off when the leg is compressed and the body is falling (height
and z-velocity columns), and the x-velocity column sets the forward lean and foot placement
that hold the speed near 2 m/s.  The gains were found by a derivative-free CMA-ES search
over K (no neural network, no gradients). The search scored each candidate over the
nominal plant and the heavy-torso, low-friction, weak-motor and high-gravity variants.
"""

import numpy as np

SCALE = np.array([0.1, 0.1, 0.5, 0.5, 0.5, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0])
OFFSET = np.array([1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

K = np.array([
    [+1.04099, -6.09504, -0.50216, +1.05164, -1.20483, +0.01554, -4.39510, -5.84875, -3.62380, -1.67841, -0.66755, +0.66618, -4.53512],
    [+0.19426, -0.10538, -2.34019, +1.30874, +1.12002, +2.71670, -1.29900, -0.89993, +2.57253, -1.64854, -0.30899, +1.78496, +4.64486],
    [+4.86211, -1.68769, +4.37824, +0.10937, +2.69526, -5.25865, -1.96909, -0.06780, -2.21100, +3.17018, -3.70656, -0.26122, +4.31644],
])


def controller(obs: dict) -> np.ndarray:
    s = np.clip((np.asarray(obs["gym_obs"], dtype=float) - OFFSET) / SCALE, -5.0, 5.0)
    s = np.concatenate([s, [float(obs["foot_contact"]), 1.0]])
    return np.clip(K @ s, -1.0, 1.0)
