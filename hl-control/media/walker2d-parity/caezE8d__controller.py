"""Walker2d running controller: a clocked, left/right-symmetric linear feedback law.

Control law (per leg, three motors: hip, knee, ankle)
    u_leg = clip( W @ phi_leg , -1, 1 )
with the feature vector phi_leg (22 entries)
    [ torso: height-1.25, pitch, 0.3*x_vel, 0.3*z_vel, 0.3*pitch_rate,
      own leg: hip, knee, ankle angles, 0.1*their rates,
      other leg: hip, knee, ankle angles, 0.1*their rates,
      clock: sin(p), cos(p), sin(2p), cos(2p),  bias 1 ]
where the gait clock is p = 2*pi*F*t for the right leg and p + pi for the left leg,
so both legs share the same 3x22 gain matrix W (mirror symmetry, anti-phase gait).

Role of the gains: the torso columns of W stabilise pitch/height and regulate speed
(lean forward to accelerate), the own-leg columns are joint stiffness/damping (PD about
a posture), the other-leg columns couple the legs (swing leg placed against the stance
leg), and the clock columns are the rhythm generator (stride frequency F, swing/stance
trajectory shape). The numbers were obtained by a gradient-free CMA-ES search on
simulated episodes (no neural network, no learning at run time).
"""

import numpy as np

F_HZ = 2.049028588427106
W = np.array([[-0.5567755662433922, -1.1072021378325476, -0.20284386005888325, 1.0468434627010261, -1.7313641852775483, -2.233689868238363, -0.3277412230148099, 2.4160128290676406, -1.1887598277924598, 0.7129006443140132, 0.3930807001973745, -0.8164000939994404, 1.0906047820837563, 0.8522715669259396, -0.7797701517198435, -2.1509600298599274, -0.1762849010539736, -0.7667460173555625, -3.0939279443618357, -0.47964600126991597, 0.848524341621823, 2.9990670113387816], [-0.041601347654140045, -0.3731690465825195, 3.287196576393542, -1.2884680452025972, -1.6859400758699443, 2.784842327057083, -2.1926666596342956, 0.13734456916734483, 1.245398350073002, -3.11979677783421, -1.0541337242929123, -1.388087383539878, -0.34547559132108524, -0.8774578081657888, -0.1957734750206488, 0.029924017232765002, -0.13650997869532824, 1.4190775792369275, -3.3520429226066617, 1.3727231174860661, -2.0031588401766167, 1.550771428229245], [4.2730410180325, -4.432765272191049, -2.7424575116527814, 3.3308579919627967, -0.1024978268247811, 0.3038180405872872, -1.3890138686499947, 0.12253222423547809, 2.418046375966745, 2.6500899265620506, -0.8938783807934098, -2.03346236740508, -0.46258545077072427, 0.9543618272077364, -1.2147857063578074, -0.5639121331041164, 0.0034690378566191032, -0.47563942603434406, -3.1772373924365227, -0.7829479232231269, 0.5928182665803302, 5.850492185951185]])


def _features(obs, side):
    q = obs["joint_angles"]
    dq = obs["joint_rates"]
    torso = [obs["height"] - 1.25, obs["pitch"], 0.3 * obs["x_velocity"],
             0.3 * obs["z_velocity"], 0.3 * obs["pitch_rate"]]
    r = np.r_[q[0:3], 0.1 * dq[0:3]]
    l = np.r_[q[3:6], 0.1 * dq[3:6]]
    p = 2.0 * np.pi * F_HZ * obs["time"] + (np.pi if side else 0.0)
    own, other = (l, r) if side else (r, l)
    return np.r_[torso, own, other, np.sin(p), np.cos(p), np.sin(2 * p), np.cos(2 * p), 1.0]


def controller(obs: dict) -> np.ndarray:
    u = np.r_[W @ _features(obs, 0), W @ _features(obs, 1)]
    return np.clip(u, -1.0, 1.0)
