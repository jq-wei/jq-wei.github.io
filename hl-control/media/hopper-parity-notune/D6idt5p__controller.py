"""Contact-switched linear state feedback for the MuJoCo Hopper.

Control law (one gain matrix per contact phase, plus offsets):

    s   = (gym_obs - S0) / SC            # 11 state deviations from the standing pose, scaled to O(1):
                                          # [z-1.25, pitch, thigh, leg, foot, vx, vz, pitch_rate, 3 joint rates]
    phi = [c*s, (1-c)*s, c, 1]            # c = foot_contact (stance gains / flight gains / stance bias / bias)
    u   = clip(W @ phi, -1, 1)            # [thigh, leg, foot] commands, 1.0 = 200 N m

Roles of the gains (rows of W = hip, knee, ankle):
  * columns 0-10  : stance feedback. Hip column on pitch / pitch_rate is the torso attitude servo;
                    knee columns on z, vz and leg angle set leg stiffness and the thrust that sustains
                    the hop; ankle columns implement the toe push-off; the vx column is the speed loop.
  * columns 11-21 : flight feedback. Hip columns on thigh and vx place the foot for touchdown (Raibert
                    foot placement); knee columns fold and re-extend the leg; ankle columns pre-shape the foot.
  * column 22     : extra constant command while in stance; column 23: constant offset in both phases.
The command saturates often, so the law behaves like a switched bang-bang gait generator;
W was tuned by black-box search (CMA-ES over the 72 entries, no gradients, no network)
on the nominal plant with an objective of forward distance plus a reduced survival bonus.
"""

import numpy as np

S0 = np.array([1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
SC = np.array([0.1, 0.1, 0.3, 0.3, 0.3, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
W = np.array([
    [-2.91171, -3.45539, -3.86776, +2.88554, -2.27760, +1.16296, -1.30836, -3.54920, -1.60347, -2.73098, +1.98442, -2.12572, -1.11025, -1.83867, +0.28731, +1.07123, +4.39072, -2.41404, -5.87872, -0.41496, -1.40048, -0.02408, -1.15720, -0.47638],
    [-1.54607, -4.05912, +4.35376, -2.54141, +2.36781, +0.67953, +1.45924, -3.40806, -3.46578, -2.07052, +2.62727, -2.68596, -5.30672, -0.17909, -1.51734, -1.71384, -1.23131, +1.44997, -0.81650, -2.89580, -3.32201, +0.37960, +1.07588, +2.08087],
    [+1.04678, -4.00758, +1.78666, -0.32815, -0.88637, -5.81635, +1.76306, -2.25748, +1.38513, +4.94540, -2.67937, +3.37446, +0.73227, -0.24558, -4.49181, -0.52221, -4.33843, +1.28609, -2.98016, +3.11654, +1.02848, -3.22883, +3.10483, +5.68064],
])


def controller(obs: dict) -> np.ndarray:
    s = (np.asarray(obs["gym_obs"], dtype=float) - S0) / SC
    c = 1.0 if obs["foot_contact"] else 0.0
    phi = np.concatenate([s * c, s * (1.0 - c), [c, 1.0]])
    return np.clip(W @ phi, -1.0, 1.0)
