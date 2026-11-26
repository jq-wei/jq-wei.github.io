"""Hopper runner: contact-switched affine state feedback on the normalised Hopper-v5 state.

Control law (one line):  u = clip(K @ s + k_c * foot_contact + b, -1, 1),
with s = (o - O_REF) / O_SCALE the 11-vector [height, pitch, thigh, leg, foot, vx, vz,
pitch_rate, thigh_rate, leg_rate, foot_rate] centred on the standing pose and scaled to
order one. Rows of K are the hip, knee and ankle motors. The gains were found by a
derivative-free CMA-ES search over the 3x13 matrix (no neural network, no gradients):
  - pitch / pitch_rate columns: the attitude loop (the hip holds the torso inside +-0.2 rad),
  - height / vz columns: the hop energy loop (knee and ankle thrust when compressed/rising),
  - vx column: Raibert-like speed feedback on the leg placement,
  - joint angle / rate columns: joint stiffness and damping that shape the gait cycle,
  - k_c (contact column, if present) and b: stance/flight switching and posture offsets.
The column roles are a reading of the signs (e.g. hip/pitch < 0: a nose-down pitch gives a
negative hip torque whose reaction lifts the nose); the gait is a fast bounding hop at
about 2.3-2.5 m/s. Status: it does NOT reliably meet the fast gate (on 48 fresh seeds about
33 % reach 18 m alive, 73 % survive, mean return about 3050); most falls are a torso-height drop
below 0.7 m late in a flight when the thigh over-swings backward.
"""

import numpy as np

O_REF = np.array([1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
O_SCALE = np.array([0.1, 0.1, 0.3, 0.3, 0.3, 1.0, 1.0, 2.0, 5.0, 5.0, 5.0])
# columns: height pitch thigh leg foot vx vz pitch_rate thigh_rate leg_rate foot_rate contact bias
GAINS = np.array([
    [-0.4777, -12.1252, -2.5583, -5.8870, +3.5407, +0.2050, -1.0009, -9.5412, -4.2651, -4.7496, +1.0177, -0.4592, -1.1862],
    [-0.4818, +1.6080, +4.3164, -7.4827, +5.3749, +1.5351, -2.7559, -1.8435, +6.7755, -0.6634, -0.6069, +0.7442, +6.6052],
    [+16.4707, -1.0502, +2.8412, -9.1859, +7.8408, -9.6299, +0.3496, +0.1046, -0.0916, -1.9451, -0.8275, +0.0478, +11.2720],
])


def controller(obs: dict) -> np.ndarray:
    s = (np.asarray(obs["gym_obs"], dtype=float) - O_REF) / O_SCALE
    u = GAINS[:, :11] @ s + GAINS[:, -1]
    u += GAINS[:, 11] * float(obs['foot_contact'])
    return np.clip(u, -1.0, 1.0)
