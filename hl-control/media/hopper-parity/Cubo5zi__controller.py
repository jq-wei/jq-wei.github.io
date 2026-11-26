"""Hopper running controller: sampling-based model-predictive control (no learning).

Control law
-----------
The observation contains the full Hopper state except the horizontal position, on which
the dynamics do not depend. Every `replan_every` control steps the controller copies that
state into its own MuJoCo model of the same plant (the Gymnasium XML shipped with
`hl_control`) and optimises a plan of joint-angle *targets* over the next `horizon`
control steps (about 1 s, long enough to see past the next landing):

* The plan is a piecewise-linear spline of joint targets with a knot every `knot_every`
  steps. The real motors run the PD servo `u = clip(kp (q_target - q) - kv qdot, -1, 1)`.
  The planning model has the same servo built into its actuators, so a sampled plan
  describes a stiff, well-behaved leg rather than a raw torque sequence; raw torque
  sequences fall within a few tens of steps on this unstable plant.
* Optimiser: predictive sampling with refinement. `n_samples` plans are drawn around
  the current best (the previous plan, shifted in time), with Gaussian noise of scale
  `sigma` spread over a range of magnitudes. All of them are rolled out in parallel
  with `mujoco.rollout`, and the best one becomes the next centre. This is repeated
  `iters` times, with the noise scaled by `shrink` each time.
* Cost of a rollout (lower is better):
  - `w_fall`: fraction of the horizon spent after the Gymnasium termination (height < 0.7 m
    or |pitch| > 0.2 rad). Falling dominates every other term.
  - `w_track`: squared error between the mean forward speed and `v_target`;
    `w_vstep`: the same error for each step. `w_vel` optionally rewards forward
    progress up to `v_target`.
  - `w_pitch`: soft barrier on the predicted pitch `pitch + pitch_lead * pitch_rate`
    beyond `pitch_soft`; `w_height`: soft barrier on the torso height below `z_soft`.
* `plan_dt` is the planning model's physics step (4 ms, coarser than the plant's 2 ms, to
  halve the cost). The servo gains are `kp` (command per rad) and `kv` (command per rad/s).
"""

import os

import mujoco
import numpy as np
from mujoco import rollout

from hl_control.envs.hopper import build_xml

PARAMS = dict(
    horizon=128,
    knot_every=8,
    n_samples=48,
    sigma=0.4,
    iters=3,
    shrink=0.5,
    plan_dt=0.004,
    replan_every=2,
    kp=6.0,
    kv=0.25,
    v_target=2.4,
    w_vel=0.0,
    w_track=5.0,
    w_vstep=0.5,
    w_fall=300.0,
    w_pitch=20.0,
    pitch_soft=0.12,
    pitch_lead=0.05,
    w_height=50.0,
    z_soft=0.85,
)

DT = 0.008


class MPC:
    def __init__(self, p):
        self.p = p
        m = mujoco.MjModel.from_xml_string(build_xml())
        m.opt.timestep = p["plan_dt"]
        gear = m.actuator_gear[:, 0].copy()
        # Position servo with the same clipping as the real command: force = clip(kp (ctrl - q) - kv qdot, -1, 1).
        m.actuator_gainprm[:, 0] = p["kp"]
        m.actuator_biasprm[:, :3] = np.c_[np.zeros(3), -p["kp"] / gear, -p["kv"] / gear]
        m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_ctrllimited[:] = 0
        m.actuator_forcelimited[:] = 1
        m.actuator_forcerange[:] = [-1.0, 1.0]
        self.m = m
        self.sub = int(round(DT / p["plan_dt"]))
        nth = int(os.environ.get("MPC_THREADS", min(os.cpu_count() or 1, 6)))
        self.datas = [mujoco.MjData(m) for _ in range(nth)]
        self.d = mujoco.MjData(m)
        self.pool = rollout.Rollout(nthread=nth)
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.lo = np.radians([-150.0, -150.0, -45.0])
        self.hi = np.radians([0.0, 0.0, 45.0])
        self.nk = p["horizon"] // p["knot_every"] + 1
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.knots = np.zeros((self.nk, 3))
        self.plan = np.zeros((self.p["horizon"], 3))
        self.k = 0
        self.j = 0

    def interp(self, knots):
        """Knots (N, nk, 3) -> joint targets (N, horizon, 3) at each control step."""
        t = np.arange(self.p["horizon"]) / self.p["knot_every"]
        i0 = np.floor(t).astype(int)
        a = (t - i0)[None, :, None]
        return knots[:, i0] * (1 - a) + knots[:, i0 + 1] * a

    def shift(self, steps):
        ke = self.p["knot_every"]
        full = self.interp(self.knots[None])[0]
        full = np.vstack([full, np.repeat(full[-1:], steps + ke + 1, 0)])
        self.knots = full[np.clip(np.arange(self.nk) * ke + steps, 0, len(full) - 1)]

    def __call__(self, obs):
        p = self.p
        if obs["step"] == 0:
            self.reset()
        if self.k % p["replan_every"] == 0:
            if self.k > 0:
                self.shift(p["replan_every"])
            self.optimize(obs)
            self.plan = self.interp(self.knots[None])[0]
            self.j = 0
        target = self.plan[self.j]
        self.j += 1
        self.k += 1
        q = np.asarray(obs["joint_angles"], dtype=float)
        qd = np.asarray(obs["joint_rates"], dtype=float)
        return np.clip(p["kp"] * (target - q) - p["kv"] * qd, -1.0, 1.0)

    def optimize(self, obs):
        p = self.p
        d = self.d
        d.qpos[:] = [0.0, obs["height"], obs["pitch"], obs["thigh"], obs["leg"], obs["foot"]]
        d.qvel[:] = [obs["x_velocity"], obs["z_velocity"], obs["pitch_rate"],
                     obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]]
        s0 = np.zeros(mujoco.mj_stateSize(self.m, self.spec))
        mujoco.mj_getState(self.m, d, s0, self.spec)
        n = p["n_samples"]
        st = np.tile(s0, (n, 1))
        center = self.knots
        spread = np.r_[0.0, np.geomspace(0.1, 1.0, n - 1)][:, None, None]
        for it in range(p["iters"]):
            noise = self.rng.normal(size=(n, self.nk, 3)) * spread * p["sigma"] * p["shrink"] ** it
            cand = np.clip(center[None] + noise, self.lo, self.hi)
            center = cand[int(np.argmin(self.cost(cand, st)))]
        self.knots = center

    def cost(self, cand, st):
        p = self.p
        n, H = len(cand), p["horizon"]
        targets = np.repeat(self.interp(cand), self.sub, axis=1)
        state, _ = self.pool.rollout(self.m, self.datas, st, targets)
        S = state[:, self.sub - 1::self.sub, :]   # after each control step: time, qpos(6), qvel(6)
        x, z, th, vx, thd = S[..., 1], S[..., 2], S[..., 3], S[..., 7], S[..., 9]
        alive = np.cumprod((z > 0.7) & (np.abs(th) < 0.2), axis=1).astype(bool)
        nalive = alive.sum(1)
        xend = np.where(nalive > 0, x[np.arange(n), np.maximum(nalive - 1, 0)], 0.0)
        vavg = xend / (np.maximum(nalive, 1) * DT)
        c = -p["w_vel"] * 125.0 * np.minimum(xend, p["v_target"] * nalive * DT) / H
        c += p["w_track"] * (vavg - p["v_target"]) ** 2
        c += p["w_vstep"] * np.where(alive, (vx - p["v_target"]) ** 2, 0).sum(1) / H
        c += p["w_fall"] * (H - nalive) / H
        pitch_pred = np.abs(th + p["pitch_lead"] * thd)
        c += 100 * p["w_pitch"] * np.where(alive, np.maximum(pitch_pred - p["pitch_soft"], 0) ** 2, 0).sum(1) / H
        c += 100 * p["w_height"] * np.where(alive, np.maximum(p["z_soft"] - z, 0) ** 2, 0).sum(1) / H
        return c


_mpc = None


def controller(obs: dict) -> np.ndarray:
    global _mpc
    if _mpc is None:
        _mpc = MPC(PARAMS)
    return _mpc(obs)
