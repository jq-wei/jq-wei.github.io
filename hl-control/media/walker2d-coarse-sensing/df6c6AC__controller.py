"""Model-predictive walker for coarse encoders without velocity sensing.

State estimation (MjObserver): an internal copy of the Walker2d MuJoCo model is stepped with the
commands actually applied, then corrected each step: pitch and joint angles by a Gauss-Newton fit
to the exact foot positions (sf) with a prior on the prediction (sp) and on the 0.05-quantised
encoders (sq), clipped to the quantisation cell (h); the torso height from the contacting foot (kz);
the torso x-velocity from the contacting foot's motion (kvx); joint rates by kv * correction / dt.
This recovers the velocities that are not measured.

Control (predictive sampling): from the estimated state, N candidate command sequences of H steps
(the previous best plan plus knot-interpolated Gaussian noise, K knots, std sigma) are rolled out in
the internal model (substep_dt integration); the score per step is min(vx, vmax) - w_over*overspeed
+ 1 - w_pitch*pitch^2 - w_h*max(0, 1.1 - z), and a fall costs pen per remaining step. The first
command of the best sequence is applied and the plan is shifted.
"""
import numpy as np
import mujoco
from hl_control.envs.walker2d import build_xml

L1, L2, LF, HIP = 0.45, 0.5, 0.1, 0.2
def fk(x):
    """x = [pitch, q1..q6] -> foot centres (4,) relative torso centre."""
    th = x[0]; out = []
    for k in range(2):
        q1, q2, q3 = x[1+3*k:4+3*k]
        a1 = th - q1; a2 = a1 - q2; a3 = a2 - q3
        fx = -HIP*np.sin(th) - L1*np.sin(a1) - L2*np.sin(a2) + LF*np.cos(a3)
        fz = -HIP*np.cos(th) - L1*np.cos(a1) - L2*np.cos(a2) - LF*np.sin(a3)
        out += [fx, fz]
    return np.array(out)
def jac(x):
    J = np.zeros((4, 7)); e = 1e-5
    f0 = fk(x)
    for i in range(7):
        xp = x.copy(); xp[i] += e; J[:, i] = (fk(xp) - f0)/e
    return J
class Observer:
    """Fuses 0.05-quantised angles with exact foot positions (and flat-foot contacts) into angle and rate estimates."""
    def __init__(self, sf=0.002, sq=0.03, sp=0.01, sr=2.0, sflat=0.3, h=0.025, sv=0.5):
        self.x = None; self.sf, self.sq, self.sp, self.sr, self.sflat, self.h, self.sv = sf, sq, sp, sr, sflat, h, sv
    def update(self, meas, feet, contacts, dt):
        if self.x is None:
            self.x = meas.copy(); self.v = np.zeros(7); self.f = feet.copy()
        # predict
        xp = self.x + self.v*dt
        # position Gauss-Newton: foot positions exact, prior on prediction, measured quantisation box
        x = xp.copy()
        for _ in range(2):
            J = jac(x); r = feet - fk(x)
            A = J.T@J/self.sf**2 + np.eye(7)/self.sp**2 + np.eye(7)/self.sq**2
            b = J.T@r/self.sf**2 + (xp - x)/self.sp**2 + (meas - x)/self.sq**2
            x = x + np.linalg.solve(A, b)
        x = np.clip(x, meas - self.h, meas + self.h)
        # rates: foot velocity exact, flat-foot constraint in contact, prior from position change
        J = jac(x); fd = (feet - self.f)/dt
        vprior = (x - self.x)/dt
        A = J.T@J/self.sf**2*dt**2 + np.eye(7)/self.sr**2 + np.eye(7)/self.sv**2
        b = J.T@fd/self.sf**2*dt**2 + vprior/self.sr**2 + self.v/self.sv**2
        for k in range(2):
            if contacts[k]:
                c = np.zeros(7); c[0] = 1; c[1+3*k:4+3*k] = -1
                A += np.outer(c, c)/self.sflat**2
        v = np.linalg.solve(A, b)
        self.x, self.v, self.f = x, v, feet.copy()
        return x, v

from hl_control.envs.walker2d import build_xml
class MjObserver:
    def __init__(self, kv=0.5, sf=0.002, sp=0.02, sq=0.05, h=0.025, kz=0.3, kvx=0.3):
        self.m = mujoco.MjModel.from_xml_string(build_xml()); self.d = mujoco.MjData(self.m)
        self.kv, self.sf, self.sp, self.sq, self.h, self.kz, self.kvx = kv, sf, sp, sq, h, kz, kvx
        self.started = False
        self.fg = [self.m.geom("foot_geom").id, self.m.geom("foot_left_geom").id]
    def update(self, obs):
        d, m = self.d, self.m
        meas = np.r_[obs["pitch"], obs["joint_angles"]]
        feet = np.r_[obs["foot_xz_rel"], obs["foot_left_xz_rel"]]
        cont = [obs["foot_contact"], obs["foot_left_contact"]]
        dt = obs["dt"]
        if not self.started:
            d.qpos[:] = 0; d.qpos[1] = obs["height"]; d.qpos[2:9] = meas; d.qvel[:] = 0
            mujoco.mj_forward(m, d); self.started = True; self.fprev = feet.copy()
        else:
            d.ctrl[:] = obs["last_action"]
            mujoco.mj_step(m, d, nstep=4)
        x = d.qpos[2:9].copy(); xp = x.copy()
        for _ in range(2):
            J = jac(x); r = feet - fk(x)
            A = J.T@J/self.sf**2 + np.eye(7)/self.sp**2 + np.eye(7)/self.sq**2
            b = J.T@r/self.sf**2 + (xp - x)/self.sp**2 + (meas - x)/self.sq**2
            x = x + np.linalg.solve(A, b)
        x = np.clip(x, meas - self.h, meas + self.h)
        dx = x - xp
        # height: a contacting foot's lowest point sits on the floor
        zs = []
        for k in range(2):
            if cont[k]:
                psi = x[0] - x[1+3*k] - x[2+3*k] - x[3+3*k]
                zs.append(-feet[2*k+1] + 0.06 + 0.1*abs(np.sin(psi)))
        z = d.qpos[1]
        zt = max(zs) if zs else z
        zt = np.clip(zt, obs["height"] - self.h, obs["height"] + self.h)
        dz = self.kz*(zt - z) if zs else np.clip(z, obs["height"]-self.h, obs["height"]+self.h) - z
        d.qpos[2:9] = x; d.qpos[1] += dz
        d.qvel[2:9] += self.kv*dx/dt; d.qvel[1] += self.kv*dz/dt
        vxs = [-(feet[2*k] - self.fprev[2*k])/dt for k in range(2) if cont[k]]
        if vxs: d.qvel[0] += self.kvx*(np.mean(vxs) - d.qvel[0])
        self.fprev = feet.copy()
        mujoco.mj_forward(m, d)
        com = d.subtree_com[1][[0, 2]] - d.xpos[1][[0, 2]]
        return dict(x=x, rate=d.qvel[2:9].copy(), vx=float(d.qvel[0]), vz=float(d.qvel[1]), z=float(d.qpos[1]), com=com,
                    vcom=float((d.subtree_linvel[1][0]) if hasattr(d,'subtree_linvel') else d.qvel[0]))

PARAMS = dict(N=24, H=50, K=4, sigma=0.4, vmax=0.5, substep_dt=0.004, w_pitch=3.0, w_h=30.0, pen=50.0, kv=0.3, w_over=2.0, w_term=20.0, zmin=0.95, thmax=0.8)
S = {}
def reset(): S.clear()
def _rollout(m, d, q0, v0, U, P, nsub):
    d.qpos[:] = q0; d.qvel[:] = v0; d.act[:] = 0 if d.act.size else 0
    J = 0.0
    for t in range(U.shape[0]):
        d.ctrl[:] = U[t]
        x0 = d.qpos[0]
        mujoco.mj_step(m, d, nstep=nsub)
        vx = (d.qpos[0] - x0)/0.008
        z = d.qpos[1]; th = d.qpos[2]
        if not (P["zmin"] < z < 2.0 and abs(th) < P["thmax"]):
            J -= P["pen"]*(U.shape[0] - t); break
        J += min(vx, P["vmax"]) - P["w_over"]*max(0.0, vx - P["vmax"]) + 1.0 - P["w_pitch"]*th*th - P["w_h"]*max(0.0, 1.1 - z)
    else:
        J -= P["w_term"]*(d.qpos[2]**2 + 0.05*d.qvel[2]**2 + 10*max(0.0, 1.1 - d.qpos[1]))
    return J
def controller(obs):
    P = PARAMS
    if obs["step"] == 0 or "ob" not in S:
        S.clear()
        S["ob"] = MjObserver(kv=P["kv"])
        m = mujoco.MjModel.from_xml_string(build_xml()); m.opt.timestep = P["substep_dt"]
        S["m"] = m; S["d"] = mujoco.MjData(m); S["nsub"] = int(round(0.008/P["substep_dt"]))
        S["U"] = np.zeros((P["H"], 6)); S["rng"] = np.random.default_rng(0)
    ob = S["ob"]; ob.update(obs)
    q0 = ob.d.qpos.copy(); v0 = ob.d.qvel.copy()
    if P.get("oracle"):
        q0[1] = obs["height"]; q0[2] = obs["pitch"]; q0[3:9] = obs["joint_angles"]
        v0[0] = obs["x_velocity"]; v0[1] = obs["z_velocity"]; v0[2] = obs["pitch_rate"]; v0[3:9] = obs["joint_rates"]
    m, d, U, rng, H, K = S["m"], S["d"], S["U"], S["rng"], P["H"], P["K"]
    best = U; bJ = _rollout(m, d, q0, v0, U, P, S["nsub"])
    tk = np.linspace(0, H-1, K); tt = np.arange(H)
    for i in range(P["N"]-1):
        kn = rng.normal(0, P["sigma"], (K, 6))
        noise = np.stack([np.interp(tt, tk, kn[:, j]) for j in range(6)], 1)
        C = np.clip(U + noise, -1, 1)
        J = _rollout(m, d, q0, v0, C, P, S["nsub"])
        if J > bJ: bJ, best = J, C
    a = best[0].copy()
    S["U"] = np.vstack([best[1:], best[-1:]])
    return a
