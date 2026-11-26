"""Sampling-based model-predictive control (predictive sampling) for the MuJoCo Humanoid.

The controller keeps its own copy of the humanoid model (the same XML the plant uses) and,
every control step, rebuilds the full simulator state from the observation (``gym_obs``
holds qpos[2:] and qvel; the horizontal position does not matter on a flat floor).  It then

1. keeps a nominal open-loop plan: a piecewise-linear spline of the 17 motor commands with a
   knot every ``KNOT_DT`` seconds over a horizon of ``HORIZON`` control steps;
2. samples ``N_SAMPLES`` perturbed plans (Gaussian noise of std ``SIGMA`` on the knots), rolls
   all of them out in parallel in MuJoCo (coarser planning timestep, cheap integrator);
3. scores each rollout with a hand-written running cost and keeps the best one (the nominal is
   always a candidate, so the plan only changes when a sample is better), and applies its first
   command.

Running cost (per planning step):
  W_VEL   * (v_com_x - V_TARGET)^2      forward speed of the mass centre (what the reward pays)
  W_VY    * v_com_y^2                    lateral drift
  W_HEIGHT* max(0, H_TARGET - z_head)^2  keep the head high (root height leaves (1, 2) m = fall)
  W_UP    * (1 - torso_z_axis_z)         stay upright
  W_BAL   * |com_xy + T_CAP v_com_xy - feet_xy|^2  capture-point balance over the feet
  W_CTRL  * |u|^2                        effort (Gym's control cost)
  W_FALL  * [z_root < 1.1]               near-termination barrier
No learning is involved: only the known model, forward simulation and a min over samples.
"""

import os

import mujoco
import numpy as np
from mujoco import rollout

from hl_control.envs.humanoid import XML_PATH

PARAMS = {
    "V_TARGET": 1.7,
    "W_VEL": 2.0,
    "W_VY": 0.3,
    "H_TARGET": 1.55,
    "W_HEIGHT": 30.0,
    "W_UP": 3.0,
    "W_BAL": 2.0,
    "T_CAP": 0.25,
    "W_CTRL": 0.1,
    "W_FALL": 100.0,
    "SIGMA": 0.15,
}
HORIZON = 40          # control steps (15 ms) in the planning horizon
KNOT_DT = 0.06        # s between spline knots
N_SAMPLES = 128
PLAN_SUBSTEPS = 2     # physics steps per control step in the planning model
CTRL_DT = 0.015
LIM = 0.4

_S = {}


def _build():
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    for kind, name, obj, objtype in (
        (mujoco.mjtSensor.mjSENS_SUBTREECOM, "com", "torso", mujoco.mjtObj.mjOBJ_BODY),
        (mujoco.mjtSensor.mjSENS_SUBTREELINVEL, "comvel", "torso", mujoco.mjtObj.mjOBJ_BODY),
        (mujoco.mjtSensor.mjSENS_FRAMEZAXIS, "zaxis", "torso", mujoco.mjtObj.mjOBJ_BODY),
        (mujoco.mjtSensor.mjSENS_FRAMEPOS, "head", "head", mujoco.mjtObj.mjOBJ_GEOM),
        (mujoco.mjtSensor.mjSENS_FRAMEPOS, "rfoot", "right_foot", mujoco.mjtObj.mjOBJ_GEOM),
        (mujoco.mjtSensor.mjSENS_FRAMEPOS, "lfoot", "left_foot", mujoco.mjtObj.mjOBJ_GEOM),
    ):
        s = spec.add_sensor()
        s.name, s.type, s.objname, s.objtype = name, kind, obj, objtype
    m = spec.compile()
    m.opt.timestep = CTRL_DT / PLAN_SUBSTEPS
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    nthread = int(os.environ.get("MPC_THREADS", os.cpu_count() or 1))
    _S.update(model=m, data=mujoco.MjData(m), datas=[mujoco.MjData(m) for _ in range(nthread)],
              pool=rollout.Rollout(nthread=nthread), rng=np.random.default_rng(0))
    nk = int(np.ceil(HORIZON * CTRL_DT / KNOT_DT)) + 2
    _S["knots_t"] = np.arange(nk) * KNOT_DT
    _S["knots_u"] = np.zeros((nk, 17))
    _S["t0"] = 0.0


def _interp(knots_t, knots_u, t):
    """Piecewise-linear interpolation of knot controls at times t (batched over leading dim of knots_u)."""
    idx = np.clip(np.searchsorted(knots_t, t, side="right") - 1, 0, len(knots_t) - 2)
    a = ((t - knots_t[idx]) / (knots_t[idx + 1] - knots_t[idx]))[:, None]
    return knots_u[..., idx, :] * (1 - a) + knots_u[..., idx + 1, :] * a


def controller(obs: dict) -> np.ndarray:
    if not _S or obs["step"] == 0:
        if not _S:
            _build()
        _S["knots_u"][:] = 0.0
        _S["t0"] = 0.0
    p = PARAMS
    m, d = _S["model"], _S["data"]
    g = obs["gym_obs"]
    d.qpos[0:2] = 0.0
    d.qpos[2:] = g[0:22]
    d.qvel[:] = g[22:45]
    d.time = 0.0
    state0 = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    # shift the nominal plan to the current time (knots relative to now)
    t_now = obs["time"]
    kt = _S["knots_t"]
    old_t = _S["t0"] + kt
    nominal = _interp(old_t, _S["knots_u"], t_now + kt)
    _S["t0"] = t_now

    K = N_SAMPLES
    rng = _S["rng"]
    cand = nominal[None] + p["SIGMA"] * rng.standard_normal((K, len(kt), 17))
    cand[0] = nominal
    cand = np.clip(cand, -LIM, LIM)

    tc = np.arange(HORIZON) * CTRL_DT
    u = _interp(kt, cand, tc)                       # K x H x 17
    ctrl = np.repeat(u, PLAN_SUBSTEPS, axis=1)      # K x H*sub x 17
    _, sens = _S["pool"].rollout(m, _S["datas"], state0[None], ctrl)
    sens = sens[:, PLAN_SUBSTEPS - 1::PLAN_SUBSTEPS]   # K x H x ns
    st, _ = None, None
    com, vel, zax, head, rf, lf = (sens[..., 0:3], sens[..., 3:6], sens[..., 6:9], sens[..., 9:12],
                                   sens[..., 12:15], sens[..., 15:18])
    feet = 0.5 * (rf[..., :2] + lf[..., :2])
    cap = com[..., :2] + p["T_CAP"] * vel[..., :2] - feet
    zroot = head[..., 2] - 0.19 * zax[..., 2]
    cost = (p["W_VEL"] * (vel[..., 0] - p["V_TARGET"]) ** 2
            + p["W_VY"] * vel[..., 1] ** 2
            + p["W_HEIGHT"] * np.maximum(0.0, p["H_TARGET"] - head[..., 2]) ** 2
            + p["W_UP"] * (1.0 - zax[..., 2])
            + p["W_BAL"] * np.sum(cap ** 2, axis=-1)
            + p["W_CTRL"] * np.sum(u ** 2, axis=-1)
            + p["W_FALL"] * (zroot < 1.1))
    total = cost.sum(axis=1)
    best = int(np.argmin(total))
    _S["knots_u"] = cand[best].copy()
    return u[best, 0].copy()
