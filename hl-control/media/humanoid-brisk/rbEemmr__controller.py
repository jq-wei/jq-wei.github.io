"""Sampling-based model-predictive walking controller for the MuJoCo Humanoid.

Control law (predictive sampling, as in MuJoCo MPC, over PD joint targets):
  * The full simulator state is read from ``obs["gym_obs"]`` (qpos[2:], qvel; the
    horizontal position is irrelevant to the dynamics and set to zero).
  * The plan is a set of ``n_knots`` piecewise-linear knots of joint-angle targets q*
    over ``horizon`` seconds.  The applied command is a PD servo
    u = clip(kp (q* - q) - kd qd, +-0.4)  (kp, kd in command units per rad and rad/s;
    ``kp_arm``/``kd_arm`` for the arms).
  * Each control step the plan is shifted by one step, ``n_samples - 1`` Gaussian
    perturbations of it (std ``noise`` rad) are drawn, and all candidates are rolled out
    with ``mujoco.rollout`` on a copy of the model whose motors are replaced by the same
    PD servos (gear folded into gains and force limits) and integrated with Euler at
    7.5 ms (about 10x cheaper than the plant's RK4 at 3 ms; MPC absorbs the mismatch).
  * Each rollout is scored with the running cost below; the cheapest candidate becomes
    the new nominal plan and its first target is applied through the PD servo.

Running cost terms (weights in PARAMS):
  w_height   one-sided quadratic on the root dropping below ``z_target`` (do not crouch / fall)
  w_fall     large penalty once the root is below 1.08 m (termination is at 1.0 m)
  w_up       1 - torso_up_z (keep the torso upright)
  w_balance  |capture point (com + cp_time * com velocity) - mid-feet| horizontally; with a
             forward velocity target this forces the feet to keep stepping under the body
  w_vel      (com_vx - v_t)^2 + com_vy^2, v_t ramps linearly to ``v_target`` over ``v_ramp`` s
             and drops to ``v_slow`` once the mass centre has advanced ``d_safe`` m (the
             8 m goal is banked; walking slower lowers the risk of a late stumble)
  w_face     1 - torso_x_axis_x (face +x)
  w_feet     |mean foot velocity - com velocity| (feet move with the body)
  w_posture  |joint angles|^2 (stay near the nominal pose)
  w_ctrl     |q* - q|^2 (servo effort)
"""

import os

import mujoco
import numpy as np
from mujoco import rollout

from hl_control.envs.humanoid import XML_PATH

PARAMS = dict(
    horizon=0.6, n_samples=96, noise=0.15, n_knots=4, kp=1.5, kd=0.04, kp_arm=3.0, kd_arm=0.06,
    v_target=1.2, v_ramp=1.5, d_safe=9.5, v_slow=0.6, z_target=1.3, cp_time=0.25,
    w_height=40.0, w_fall=1000.0, w_up=5.0, w_balance=8.0, w_vel=3.0, w_face=2.0,
    w_feet=0.5, w_posture=0.05, w_ctrl=0.2,
)

DT = 0.015
SUBSTEPS = 5
PLAN_SUB = 2


def _build_model():
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    S, O = mujoco.mjtSensor, mujoco.mjtObj
    spec.add_sensor(name="com", type=S.mjSENS_SUBTREECOM, objtype=O.mjOBJ_BODY, objname="torso")
    spec.add_sensor(name="comvel", type=S.mjSENS_SUBTREELINVEL, objtype=O.mjOBJ_BODY, objname="torso")
    spec.add_sensor(name="zaxis", type=S.mjSENS_FRAMEZAXIS, objtype=O.mjOBJ_XBODY, objname="torso")
    spec.add_sensor(name="xaxis", type=S.mjSENS_FRAMEXAXIS, objtype=O.mjOBJ_XBODY, objname="torso")
    spec.add_sensor(name="rfoot", type=S.mjSENS_FRAMEPOS, objtype=O.mjOBJ_GEOM, objname="right_foot")
    spec.add_sensor(name="lfoot", type=S.mjSENS_FRAMEPOS, objtype=O.mjOBJ_GEOM, objname="left_foot")
    spec.add_sensor(name="rfootv", type=S.mjSENS_FRAMELINVEL, objtype=O.mjOBJ_GEOM, objname="right_foot")
    spec.add_sensor(name="lfootv", type=S.mjSENS_FRAMELINVEL, objtype=O.mjOBJ_GEOM, objname="left_foot")
    m = spec.compile()
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
    m.opt.timestep = DT / PLAN_SUB
    return m


def _gains(model):
    kp = np.full(17, PARAMS["kp"]); kd = np.full(17, PARAMS["kd"])
    kp[11:] = PARAMS["kp_arm"]; kd[11:] = PARAMS["kd_arm"]
    return kp, kd


def _servo(model):
    """Turn the planning model's motors into PD servos u = clip(kp (q* - q) - kd qd, +-0.4)."""
    kp, kd = _gains(model)
    for i in range(model.nu):
        g = model.actuator_gear[i, 0]
        model.actuator_gear[i, 0] = 1.0                 # actuator length = q, force = joint torque
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[i, :3] = [g * kp[i], 0, 0]
        model.actuator_biasprm[i, :3] = [0, -g * kp[i], -g * kd[i]]
        model.actuator_ctrllimited[i] = 0
        model.actuator_forcelimited[i] = 1
        model.actuator_forcerange[i] = [-0.4 * g, 0.4 * g]


class _MPC:
    def __init__(self):
        self.model = _build_model()
        _servo(self.model)
        self.kp, self.kd = _gains(self.model)
        mujoco.mj_setConst(self.model, mujoco.MjData(self.model))
        self.sub = PLAN_SUB
        self.qlo = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0]][0] for i in range(17)])
        self.qhi = np.array([self.model.jnt_range[self.model.actuator_trnid[i, 0]][1] for i in range(17)])
        self.data = mujoco.MjData(self.model)
        self.nthread = int(os.environ.get("MPC_THREADS", max(1, min(4, os.cpu_count() or 1))))
        self.datas = [mujoco.MjData(self.model) for _ in range(self.nthread)]
        self.roll = rollout.Rollout(nthread=self.nthread)
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self):
        p = PARAMS
        self.H = int(round(p["horizon"] / DT))
        self.knot_t = np.linspace(0.0, p["horizon"], p["n_knots"])
        self.knots = np.zeros((p["n_knots"], 17))
        self.t = 0.0
        self.x0 = None
        self.rng = np.random.default_rng(0)

    def _interp(self, knots):
        """knots (..., K, 17) -> per-control-step commands (..., H, 17)."""
        ts = np.arange(self.H) * DT
        idx = np.clip(np.searchsorted(self.knot_t, ts, side="right") - 1, 0, len(self.knot_t) - 2)
        t0, t1 = self.knot_t[idx], self.knot_t[idx + 1]
        a = ((ts - t0) / (t1 - t0))[:, None]
        return knots[..., idx, :] * (1 - a) + knots[..., idx + 1, :] * a

    def act(self, obs):
        p = PARAMS
        if self.x0 is None:
            self.x0 = float(obs["com_xyz"][0])
        self.v_cmd = p["v_target"] if obs["com_xyz"][0] - self.x0 < p["d_safe"] else p["v_slow"]
        g = obs["gym_obs"]
        m, d = self.model, self.data
        d.qpos[2:] = g[:22]
        d.qpos[:2] = 0.0
        d.qvel[:] = g[22:45]
        mujoco.mj_forward(m, d)
        state = np.zeros(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
        mujoco.mj_getState(m, d, state, mujoco.mjtState.mjSTATE_FULLPHYSICS)

        # shift nominal plan by one control step
        shifted_t = self.knot_t + DT
        nom_cmds = self._interp(self.knots)
        last = nom_cmds[-1]
        shifted = np.array([np.interp(t, np.arange(self.H) * DT, nom_cmds[:, j]) if t <= (self.H - 1) * DT else last[j]
                            for t in shifted_t for j in range(17)]).reshape(len(shifted_t), 17)
        N = p["n_samples"]
        cand = np.repeat(shifted[None], N, axis=0)
        cand[1:] += self.rng.normal(0.0, p["noise"], size=cand[1:].shape)
        cand = np.clip(cand, self.qlo, self.qhi)
        cmds = self._interp(cand)                         # (N, H, 17)
        ctrl = np.repeat(cmds, self.sub, axis=1)          # (N, H*5, 17)
        init = np.tile(state, (N, 1))
        states, sens = self.roll.rollout(m, self.datas, init, ctrl)
        states = states[:, self.sub - 1::self.sub]        # state after each control step
        sens = sens[:, self.sub - 1::self.sub]
        cost = self._cost(states, sens, cmds, obs["time"])
        best = int(np.argmin(cost))
        self.knots = cand[best]
        q, qd = obs["joint_angles"], obs["joint_rates"]
        return np.clip(self.kp * (cmds[best, 0] - q) - self.kd * qd, -0.4, 0.4)

    def _cost(self, states, sens, cmds, t0):
        p = PARAMS
        m = self.model
        qpos = states[..., 1:1 + m.nq]
        z = qpos[..., 2]
        com, comvel = sens[..., 0:3], sens[..., 3:6]
        zax, xax = sens[..., 6:9], sens[..., 9:12]
        rf, lf, rfv, lfv = sens[..., 12:15], sens[..., 15:18], sens[..., 18:21], sens[..., 21:24]
        t = t0 + DT * (1 + np.arange(z.shape[1]))
        vt = self.v_cmd * np.clip(t / p["v_ramp"], 0, 1)
        c = p["w_height"] * np.maximum(0.0, p["z_target"] - z) ** 2
        c += p["w_fall"] * (z < 1.08)
        c += p["w_up"] * (1.0 - zax[..., 2])
        cp = com[..., :2] + p["cp_time"] * comvel[..., :2]
        mid = 0.5 * (rf[..., :2] + lf[..., :2])
        c += p["w_balance"] * np.linalg.norm(cp - mid, axis=-1)
        c += p["w_vel"] * ((comvel[..., 0] - vt) ** 2 + comvel[..., 1] ** 2)
        c += p["w_face"] * (1.0 - xax[..., 0])
        c += p["w_feet"] * np.linalg.norm(0.5 * (rfv[..., :2] + lfv[..., :2]) - comvel[..., :2], axis=-1)
        c += p["w_posture"] * np.sum(qpos[..., 7:] ** 2, axis=-1)
        c += p["w_ctrl"] * np.sum((cmds - qpos[..., 7:]) ** 2, axis=-1)
        return c.sum(axis=1)


_mpc = None


def controller(obs: dict) -> np.ndarray:
    global _mpc
    if _mpc is None:
        _mpc = _MPC()
    if obs["step"] == 0:
        _mpc.reset()
    return np.clip(_mpc.act(obs), -0.4, 0.4)
