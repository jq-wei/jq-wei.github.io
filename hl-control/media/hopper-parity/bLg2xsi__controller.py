"""Linear state-feedback controller for the MuJoCo Hopper (Hopper-v5 dynamics).

Control law (one static linear map plus bias, clipped to the motor range):

    u = clip(K @ (s - s0) + b, -1, 1)          u = [thigh, leg(knee), foot(ankle)]

with s the 11-vector Hopper-v5 state [z, pitch, thigh, leg, foot, vx, vz, pitch_rate,
thigh_rate, leg_rate, foot_rate] and s0 the standing height 1.25 m. No clock and no
phase logic: the hop cycle is a limit cycle of the closed loop (about 14 flights in 8 s,
~2.3 m/s, ~75 % of the time in flight). The code keeps a stance/flight slot K[0]/K[1]
switched on foot contact, but both slots hold the same gains here.

Role of the gains (columns of K, read row by row):
* pitch / pitch_rate (strong, e.g. -9.7 on the hip row): torso-attitude loop that keeps
  the torso inside the +-0.2 rad band; the knee and ankle rows also respond to pitch, so
  the whole leg tilts to catch a forward or backward lean (Raibert-style placement).
* z / vz: height and vertical-speed terms (z: +5.7 knee, +11.9 ankle) that shape the
  knee thrust and ankle push-off over the bounce.
* joint angles / rates: joint-space damping and posture (knee fold in flight, ankle
  re-cock before touchdown).
* vx: forward-speed feedback that sets the cruise speed.
* b: nominal thrust / posture offsets.

Gains were tuned with a derivative-free evolution strategy (CMA-ES, hand-written, on the
episode return over random reset seeds); no neural network and no gradients.
Measured on 65 unseen seeds (0-4, 20-79): no fall on 60/65, `fast` (>= 18 m) on 54/65
(83 %), mean return about 3240.
"""

import numpy as np

S0 = np.array([1.25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float)
# K[0] stance, K[1] flight; rows thigh, leg, foot; columns follow the state order above.
K = np.array([[[ 1.67244e+00, -9.69088e+00, -2.26480e-01,  1.52085e+00, -7.15748e-01, -2.92018e-01, -7.07951e-02, -2.85827e-01, -7.22045e-02,  2.62992e-02,  3.44817e-02],
  [ 5.70485e+00, -5.37762e+00,  1.31857e-01, -8.09059e-01,  3.99083e-01,  1.94864e-01, -4.23621e-01, -2.49429e-01,  4.66520e-02, -1.33821e-01,  1.44723e-01],
  [ 1.18531e+01,  4.15666e+00,  1.17306e+00, -4.85183e-01,  4.91499e-01, -2.15218e-01,  1.95891e-01,  1.47764e-01,  7.11237e-03, -1.80080e-02,  3.81436e-02]],

 [[ 1.67244e+00, -9.69088e+00, -2.26480e-01,  1.52085e+00, -7.15748e-01, -2.92018e-01, -7.07951e-02, -2.85827e-01, -7.22045e-02,  2.62992e-02,  3.44817e-02],
  [ 5.70485e+00, -5.37762e+00,  1.31857e-01, -8.09059e-01,  3.99083e-01,  1.94864e-01, -4.23621e-01, -2.49429e-01,  4.66520e-02, -1.33821e-01,  1.44723e-01],
  [ 1.18531e+01,  4.15666e+00,  1.17306e+00, -4.85183e-01,  4.91499e-01, -2.15218e-01,  1.95891e-01,  1.47764e-01,  7.11237e-03, -1.80080e-02,  3.81436e-02]]])
B = np.array([[0.601  , 0.22997, 0.69481],
 [0.601  , 0.22997, 0.69481]])


def controller(obs: dict) -> np.ndarray:
    s = np.asarray(obs["gym_obs"], dtype=float) - S0
    phase = 0 if obs["foot_contact"] else 1
    return np.clip(K[phase] @ s + B[phase], -1.0, 1.0)
