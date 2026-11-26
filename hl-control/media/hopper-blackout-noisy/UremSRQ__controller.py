"""Hopper controller for sensing blackouts under noise.

Control law: a linear feedback policy a = W @ ((f - MU) / SD) on the feature vector
f = [height, pitch, thigh, leg, foot, vx, vz, pitch_rate, thigh_rate, leg_rate, foot_rate, foot_contact],
found by derivative-free random search (ARS, no gradients, no network) on the plant with the
estimator below in the loop; the search reward omitted the alive bonus so the policy hops instead
of standing still.  W holds the feedback gains (3 motors x 12 features), MU/SD normalise the features.

State estimator: because blackouts freeze every measured channel, the features come from a model-based
observer, not from the raw sensors.  A MuJoCo copy of the plant is stepped with the last applied
command (predict).  Each reading is then classified: if it matches the previous raw reading to within
the noise (squared difference / 2 sigma^2 < HOLD_GATE=30) and fits that better than the prediction, it is a
stale held sample and the prediction is used; otherwise it is live and is blended in with gain K, or
adopted outright when it disagrees strongly with the prediction (innovation > GATE, e.g. at an impact).
The search reward capped the forward speed at 1.6 m/s so the gait keeps a margin against falls.
"""
import numpy as np
import mujoco
from hl_control.envs.hopper import build_xml

W = np.array([[ 0.111315, -0.492116,  0.049112,  0.007759, -0.023355, -0.044957,  0.141635, -0.412801,
  -0.180654,  0.019681, -0.108192, -0.145971],
 [-0.263379, -0.063536,  0.270529, -0.436498,  0.099607,  0.081851,  0.079444, -0.042191,
  -0.167838, -0.380206,  0.021482, -0.010404],
 [ 0.577929, -0.007352,  0.014045, -0.288019, -0.245198, -0.526383, -0.045502,  0.011796,
   0.146802, -0.029317,  0.082677, -0.153488]])
MU = np.array([ 1.330125, -0.025687, -0.396046, -0.242336,  0.282779,  2.184401, -0.037323, -0.004807, -0.140862,
 -0.060643, -0.529388,  0.264675])
SD = np.array([0.168639, 0.013135, 0.187528, 0.13436 , 0.603396, 0.641958, 1.501409, 0.416031, 2.142753, 2.344047,
 6.096673, 0.44116 ])
EST_K, EST_GATE, EST_MAX_REJ = 0.5, 40.0, 12

class Estimator:
    def __init__(self, K=EST_K, gate=EST_GATE, sigma=0.01, max_rej=EST_MAX_REJ, hold_gate=30.0):
        self.m = mujoco.MjModel.from_xml_string(build_xml())
        self.d = mujoco.MjData(self.m)
        self.K, self.gate, self.sigma, self.max_rej, self.hold_gate = K, gate, sigma, max_rej, hold_gate
        self.foot=self.m.geom('foot_geom').id; self.floor=self.m.geom('floor').id
        self.contact=False
        self.reset()
    def reset(self):
        self.q = None; self.rej = 0; self.last_chi = 0.0
    @staticmethod
    def meas(o):
        # [z, pitch, thigh, leg, foot, vx, vz, w, thigh_r, leg_r, foot_r]
        return np.array(o["gym_obs"], float)
    def update(self, o):
        z = self.meas(o)
        if self.q is None or o["step"] == 0:
            self.q = z.copy(); self.zprev = z.copy(); self.rej = 0; self.contact = bool(o['foot_contact'])
            # full state: qpos[1:], qvel (clipping at 10 ignored)
            return self.q
        # predict
        d = self.d
        d.qpos[:] = np.concatenate([[0.0], self.q[:5]]); d.qvel[:] = self.q[5:]
        d.ctrl[:] = o["last_action"]
        mujoco.mj_step(self.m, d, nstep=4)
        p = np.concatenate([d.qpos[1:], d.qvel])
        self.contact = any({int(a),int(b)}=={self.foot,self.floor} for a,b in d.contact.geom[:d.ncon])
        s2 = self.sigma**2
        chi_pred = float((z - p) @ (z - p)) / s2
        chi_hold = float((z - self.zprev) @ (z - self.zprev)) / (2 * s2)
        self.zprev = z.copy()
        stale = chi_hold < self.hold_gate and chi_hold < chi_pred
        if stale:
            self.q = p
        elif chi_pred > self.gate:
            self.q = z.copy(); self.contact = bool(o["foot_contact"])
        else:
            self.q = p + self.K * (z - p)
        return self.q


_EST = Estimator()


def controller(obs: dict) -> np.ndarray:
    if obs["step"] == 0:
        _EST.reset()
    q = _EST.update(obs)
    f = np.concatenate([q, [float(_EST.contact)]])
    return np.clip(W @ ((f - MU) / SD), -1.0, 1.0)
