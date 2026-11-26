"""Predictive-sampling MPC for the MuJoCo Humanoid (no learning).

Each control step the controller copies the true simulator state (root height,
orientation, joint angles and all velocities, from ``obs["gym_obs"]``; the
horizontal position does not matter on a flat floor) into an internal copy of
the same MuJoCo model (planning step ``SUBSTEP`` = 5 ms instead of 3 ms). It then
rolls out ``N_SAMPLES`` sequences over ``HORIZON`` seconds with ``mujoco.rollout``.

The decision variables are joint-angle *targets*, not torques (``PD = True``):
in the planning model every motor becomes a saturated PD servo
``u = clip(kp (target - q) - kv qdot, -0.4, 0.4)`` (in command units), and the
controller applies the same law to the measured joints. This feedback inside the
rollouts makes sampling far more robust than open-loop torques. ``kv`` must stay
small (0.02): the plant holds each command for 15 ms, and larger damping makes the
real loop diverge from the planned one.

Each target sequence is a piecewise-linear spline with ``N_KNOTS`` knots: the
current nominal sequence plus Gaussian knot perturbations of std ``sigma`` [rad],
clipped to the joint ranges. The cheapest sample becomes the new nominal (warm
start, shifted in time each step), and its first knot is applied.

Cost per sampled instant (weights in PARAMS):
  w_vel    (v_target - com_vx)^2      track forward mass-centre speed
  w_vy     com_vy^2                    no sideways drift
  w_height relu(z_target - z_torso)^2  stay tall (fall = z < 1.0)
  w_up     1 - up_z                    torso upright
  w_yaw    1 - cos(heading)            keep facing +x
  w_bal    |capture point - feet mid|  keep feet under the capture point (forces stepping)
  w_ctrl   |target|^2                  stay near the straight standing pose
  w_post, w_jvel                       optional posture / joint-speed costs (off)
  w_fall   z_torso < 1.1               hard fall penalty
"""

import mujoco
import numpy as np
from mujoco import rollout

from hl_control.envs.humanoid import XML_PATH

PARAMS = {
    "v_target": 1.4,
    "w_vel": 10.0,
    "w_vy": 2.0,
    "w_height": 300.0,
    "z_target": 1.3,
    "w_up": 20.0,
    "w_yaw": 5.0,
    "w_bal": 10.0,
    "w_ctrl": 0.1,
    "w_fall": 1000.0,
    "sigma": 0.15,
    "w_post": 0.0,
    "kp": 2.0,
    "kv": 0.02,
    "w_jvel": 0.0,
}
HORIZON = 0.55      # s
N_KNOTS = 4
N_SAMPLES = 64
SUBSTEP = 0.005     # planning physics step (the plant uses 3 ms)
CTRL_DT = 0.015
NTHREAD = 2
PD = True

_S = {}


def _build():
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    s = spec.add_sensor()
    s.type = mujoco.mjtSensor.mjSENS_SUBTREECOM; s.objtype = mujoco.mjtObj.mjOBJ_BODY; s.objname = "torso"
    s = spec.add_sensor()
    s.type = mujoco.mjtSensor.mjSENS_SUBTREELINVEL; s.objtype = mujoco.mjtObj.mjOBJ_BODY; s.objname = "torso"
    for g in ("right_foot", "left_foot"):
        s = spec.add_sensor()
        s.type = mujoco.mjtSensor.mjSENS_FRAMEPOS; s.objtype = mujoco.mjtObj.mjOBJ_GEOM; s.objname = g
    spec.option.timestep = SUBSTEP
    if PD:
        for a in spec.actuators:
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = PARAMS["kp"]
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.biasprm[0] = 0.0
            a.biasprm[1] = -PARAMS["kp"] / a.gear[0]   # actuator length is gear * q
            a.biasprm[2] = -PARAMS["kv"] / a.gear[0]
            a.ctrllimited = 0
            a.forcelimited = 1
            a.forcerange = [-0.4, 0.4]
    m = spec.compile()
    d = mujoco.MjData(m)
    nth = NTHREAD
    _S.update(m=m, d=d, datas=[mujoco.MjData(m) for _ in range(nth)],
              rng=np.random.default_rng(0), nominal=np.zeros((N_KNOTS, m.nu)),
              nsteps=int(round(HORIZON / SUBSTEP)))
    if PD:
        jr = m.jnt_range[m.actuator_trnid[:, 0]]
        _S["lo"], _S["hi"] = jr[:, 0], jr[:, 1]
    else:
        _S["lo"], _S["hi"] = -0.4 * np.ones(m.nu), 0.4 * np.ones(m.nu)
    _S["knot_t"] = np.linspace(0.0, HORIZON, N_KNOTS)
    _S["step_t"] = np.arange(_S["nsteps"]) * SUBSTEP


def _interp(knots, t_knots, t):
    # knots (..., K, nu) -> (..., T, nu)
    idx = np.clip(np.searchsorted(t_knots, t, side="right") - 1, 0, len(t_knots) - 2)
    w = ((t - t_knots[idx]) / (t_knots[idx + 1] - t_knots[idx]))[:, None]
    return knots[..., idx, :] * (1 - w) + knots[..., idx + 1, :] * w


def _cost(state, sens, ctrl, P, nq):
    # state (N, T, ns), sens (N, T, 12); subsample at control rate
    k = int(round(CTRL_DT / SUBSTEP))
    sub = slice(k - 1, None, k)
    qpos = state[:, sub, 1:1 + nq]
    sens = sens[:, sub]
    z = qpos[..., 2]
    qw, qx, qy, qz = qpos[..., 3], qpos[..., 4], qpos[..., 5], qpos[..., 6]
    up_z = 1 - 2 * (qx * qx + qy * qy)
    head_x = 1 - 2 * (qy * qy + qz * qz)  # body x axis, world x component
    head_y = 2 * (qx * qy + qw * qz)     # body x axis, world y component
    hn = np.sqrt(head_x ** 2 + head_y ** 2) + 1e-6
    com = sens[..., 0:3]
    cv = sens[..., 3:6]
    feet_mid = 0.5 * (sens[..., 6:8] + sens[..., 9:11])
    cp = com[..., :2] + cv[..., :2] * np.sqrt(np.maximum(com[..., 2:3], 0.3) / 9.81)
    bal = np.sqrt(np.sum((cp - feet_mid) ** 2, axis=-1) + 0.01)
    c = (P["w_vel"] * (P["v_target"] - cv[..., 0]) ** 2
         + P["w_vy"] * cv[..., 1] ** 2
         + P["w_height"] * np.maximum(P["z_target"] - z, 0) ** 2
         + P["w_up"] * (1 - up_z)
         + P["w_yaw"] * (1 - head_x / hn)
         + P["w_bal"] * bal
         + P["w_fall"] * (z < 1.1)
         + P["w_post"] * np.sum(qpos[..., 7:] ** 2, axis=-1)
         + P["w_jvel"] * np.sum(state[:, sub, 1 + nq + 6:1 + nq + 6 + 17] ** 2, axis=-1))
    c = c.sum(axis=1) + P["w_ctrl"] * np.sum(ctrl[:, sub] ** 2, axis=(1, 2))
    return c


def controller(obs: dict) -> np.ndarray:
    if not _S:
        _build()
    P = PARAMS
    m, d = _S["m"], _S["d"]
    if obs["step"] == 0:
        _S["nominal"][:] = 0.0
        _S["rng"] = np.random.default_rng(0)
    else:
        # shift the nominal by one control step
        t_new = _S["knot_t"] + CTRL_DT
        _S["nominal"] = _interp(_S["nominal"], _S["knot_t"], np.minimum(t_new, HORIZON - 1e-9))
    g = obs["gym_obs"]
    d.qpos[:2] = 0.0
    d.qpos[2:] = g[0:m.nq - 2]
    d.qvel[:] = g[m.nq - 2:m.nq - 2 + m.nv]
    d.time = 0.0
    mujoco.mj_forward(m, d)
    state0 = np.empty(mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS))
    mujoco.mj_getState(m, d, state0, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    rng = _S["rng"]
    knots = _S["nominal"][None] + P["sigma"] * rng.standard_normal((N_SAMPLES, N_KNOTS, m.nu))
    knots[0] = _S["nominal"]
    knots = np.clip(knots, _S["lo"], _S["hi"])
    ctrl = _interp(knots, _S["knot_t"], _S["step_t"])
    state, sens = rollout.rollout(m, _S["datas"], np.tile(state0, (N_SAMPLES, 1)), ctrl,
                                  nstep=_S["nsteps"])
    c = _cost(state, sens, ctrl, P, m.nq)
    c = np.nan_to_num(c, nan=1e12, posinf=1e12)
    best = int(np.argmin(c))
    _S["nominal"] = knots[best].copy()
    if not PD:
        return knots[best, 0].copy()
    q, qd = obs["joint_angles"], obs["joint_rates"]
    return np.clip(P["kp"] * (knots[best, 0] - q) - P["kv"] * qd, -0.4, 0.4)
