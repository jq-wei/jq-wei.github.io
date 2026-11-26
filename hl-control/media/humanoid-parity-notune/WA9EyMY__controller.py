"""Sampling-based model-predictive controller (predictive sampling) for the Humanoid.

No learning: every control step, the full state is rebuilt from ``gym_obs`` (qpos[2:],
qvel) into a private copy of the MuJoCo model, which serves as a cheaper planning model
(Euler, 3 x 5 ms substeps per 15 ms control step). N open-loop control sequences are
rolled out (``mujoco.rollout``, multithreaded) over a horizon of H control steps. Each
sequence is a piecewise-linear spline with one knot every KNOT steps: the incumbent plan
plus Gaussian noise of std SIGMA. The cheapest sequence becomes the new plan; its first
command is applied, and the plan is shifted by one step for warm start.

Running cost per predicted control step (weights in P):
  W_H   (z_torso - H_DES)^2            keep the torso high (falls end at 1.0 m)
  W_UP  (1 - torso_z . e_z)            torso upright
  W_VX  (v_com,x - V_DES)^2            forward speed (what the reward pays)
  W_VY  v_com,y^2                      no sideways drift
  W_YAW (1 - torso_x . e_x)            face +x
  W_BAL |c + T_CAP (v - CAP_V V_DES e_x) - mid_feet|^2   capture point over the feet
  W_U   |u|^2                          effort
  1000 per step with z_torso < 1.05    predicted fall
"""

import json
import os

import mujoco
import numpy as np
from mujoco import rollout

from hl_control.envs.humanoid import XML_PATH

P = dict(N=48, H=50, KNOT=5, SIGMA=0.15, V_DES=2.0, W_H=10.0, W_UP=5.0, W_VX=3.0, W_VY=0.5, W_YAW=1.0,
         W_BAL=3.0, W_U=0.1, H_DES=1.30, T_CAP=0.3, CAP_V=1.0, NTHREAD=2, SUB=3)
P.update(json.loads(os.environ.get("MPC_P", "{}")))   # tuning hook (unset when graded)

_spec = mujoco.MjSpec.from_file(str(XML_PATH))


def _sensor(kind, objtype, objname):
    s = _spec.add_sensor()
    s.type, s.objtype, s.objname = kind, objtype, objname


_S, _O = mujoco.mjtSensor, mujoco.mjtObj
_sensor(_S.mjSENS_SUBTREECOM, _O.mjOBJ_BODY, "torso")       # 0:3   mass centre
_sensor(_S.mjSENS_SUBTREELINVEL, _O.mjOBJ_BODY, "torso")    # 3:6   its velocity
_sensor(_S.mjSENS_FRAMEPOS, _O.mjOBJ_GEOM, "right_foot")    # 6:9
_sensor(_S.mjSENS_FRAMEPOS, _O.mjOBJ_GEOM, "left_foot")     # 9:12
_sensor(_S.mjSENS_FRAMEZAXIS, _O.mjOBJ_BODY, "torso")       # 12:15
_sensor(_S.mjSENS_FRAMEXAXIS, _O.mjOBJ_BODY, "torso")       # 15:18
_m = _spec.compile()
_m.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
_m.opt.timestep = 0.015 / P["SUB"]
_datas = [mujoco.MjData(_m) for _ in range(P["NTHREAD"])]
_d = mujoco.MjData(_m)
_FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS
_state = np.zeros(mujoco.mj_stateSize(_m, _FULL))
_mem = {"U": None, "rng": np.random.default_rng(0)}


def _interp(knots, H, K):
    """(N, nk, 17) knots -> (N, H, 17) piecewise-linear controls."""
    nk = knots.shape[1]
    t = np.arange(H) / K
    i0 = np.minimum(np.floor(t).astype(int), nk - 2)
    a = (t - i0)[None, :, None]
    return knots[:, i0] * (1 - a) + knots[:, i0 + 1] * a


def controller(obs: dict) -> np.ndarray:
    p = P
    H, K, N, sub = p["H"], p["KNOT"], p["N"], p["SUB"]
    nk = H // K + 1
    U = _mem["U"]
    if obs["step"] == 0 or U is None:
        U = np.zeros((nk, 17))
        _mem["rng"] = np.random.default_rng(0)
    else:  # shift the plan by one control step
        src = np.arange(nk) * K
        U = np.stack([np.interp(src + 1.0, src, U[:, j]) for j in range(17)], axis=1)
    g = obs["gym_obs"]
    _d.qpos[:2] = 0.0
    _d.qpos[2:] = g[:22]
    _d.qvel[:] = g[22:45]
    _d.time = 0.0
    mujoco.mj_getState(_m, _d, _state, _FULL)

    knots = U[None] + p["SIGMA"] * _mem["rng"].standard_normal((N, nk, 17))
    knots[0] = U
    knots = np.clip(knots, -0.4, 0.4)
    u = _interp(knots, H, K)
    state, sens = rollout.rollout(_m, _datas, _state, np.repeat(u, sub, axis=1), persistent_pool=True)
    sens = sens[:, sub - 1::sub]
    z = state[:, sub - 1::sub, 3]          # state = (time, qpos, ...): qpos[2]
    com, comv = sens[..., 0:3], sens[..., 3:6]
    mid = 0.5 * (sens[..., 6:8] + sens[..., 9:11])
    tz, tx = sens[..., 12:15], sens[..., 15:18]
    cap = com[..., :2] + p["T_CAP"] * (comv[..., :2] - np.array([p["CAP_V"] * p["V_DES"], 0.0])) - mid
    c = (p["W_H"] * (z - p["H_DES"]) ** 2 + p["W_UP"] * (1 - tz[..., 2])
         + p["W_VX"] * (comv[..., 0] - p["V_DES"]) ** 2 + p["W_VY"] * comv[..., 1] ** 2
         + p["W_YAW"] * (1 - tx[..., 0]) + p["W_BAL"] * np.sum(cap ** 2, -1)
         + p["W_U"] * np.sum(u ** 2, -1) + 1000.0 * (z < 1.05))
    best = int(np.argmin(c.sum(1)))
    _mem["U"] = knots[best]
    return u[best, 0].copy()
