"""Sampling-based MPC hopper controller (predictive sampling on the MuJoCo model).

Law: every `replan` control steps, `N` perturbed copies of the current 50-step torque plan
(piecewise-constant over 5-step knots, noise `sig`) are simulated with the vendored Hopper model
(4 ms RK4 for speed); the plan with the best cumulative cost is kept and its first command applied.
Cost per step (zero once the rollout has fallen, plus -50 on falling):
  -vw (x_velocity - vt)^2     speed tracking; vt = vmax until ~xstop metres have been travelled
                               (x is dead-reckoned from x_velocity), then 0 so the hopper settles
  +1 + hw min(z, 1.2)          alive bonus and a preference for a tall torso
  -pw (|pitch|-pth)^2+ ...     soft pitch limit, soft height floor (zw), knee-fold limit (wl)
Sensing blackouts: a repeated measurement is detected (identical gym_obs); the exact model
(2 ms RK4) is then stepped with the previously applied command, so the state estimate stays exact
because the plant is nominal. A fresh measurement resets the estimate.
"""
import os
import numpy as np
import mujoco
from mujoco import rollout
from hl_control.envs.hopper import build_xml

_NT = max(1, min(8, os.cpu_count() or 1))
_XML = build_xml()
_EXACT = mujoco.MjModel.from_xml_string(_XML)        # same physics as the plant: state estimation
_PLAN = mujoco.MjModel.from_xml_string(_XML)         # coarser timestep: planning rollouts
_PLAN.opt.timestep = 0.004
_THREADS = [mujoco.MjData(_PLAN) for _ in range(_NT)]

PARAMS = dict(N=128, H=50, K=5, sig=0.5, replan=3, vmax=0.6, vw=5.0, hw=3.0, pw=20.0, pth=0.1,
              zw=100.0, wl=20.0, xstop=3.3, vstop=0.0)
_S = {}


def _reset():
    _S.clear()
    _S.update(plan=None, rng=np.random.default_rng(0), prev=None, sim=mujoco.MjData(_EXACT),
              last=np.zeros(3), x=0.0, vt=PARAMS["vmax"])


def _estimate(obs):
    key = obs["gym_obs"]
    d = _S["sim"]
    if _S["prev"] is not None and np.array_equal(key, _S["prev"]):    # blackout: propagate the model
        d.ctrl[:] = _S["last"]
        mujoco.mj_step(_EXACT, d, nstep=4)
    else:
        mujoco.mj_resetData(_EXACT, d)
        d.qpos[:] = [0, obs["height"], obs["pitch"], obs["thigh"], obs["leg"], obs["foot"]]
        d.qvel[:] = [obs["x_velocity"], obs["z_velocity"], obs["pitch_rate"],
                     obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]]
        mujoco.mj_forward(_EXACT, d)
    _S["prev"] = key.copy()
    _S["x"] += d.qvel[0] * obs["dt"]
    _S["vt"] = PARAMS["vmax"] if _S["x"] < PARAMS["xstop"] else PARAMS["vstop"]
    return np.concatenate([[d.time], d.qpos, d.qvel])


def _plan(state):
    p = PARAMS
    H, K, N = p["H"], p["K"], p["N"]
    if _S["plan"] is None:
        _S["plan"] = np.zeros((H, 3))
    noise = _S["rng"].normal(0, p["sig"], size=(N, H // K, 3))
    noise[0] = 0
    cand = np.clip(_S["plan"][None] + np.repeat(noise, K, axis=1), -1, 1)
    s, _ = rollout.rollout(_PLAN, _THREADS, np.tile(state, (N, 1)), np.repeat(cand, 2, axis=1))
    s = s[:, 1::2]
    z, pitch, leg, vx = s[:, :, 2], s[:, :, 3], s[:, :, 5], s[:, :, 7]
    alive = np.cumsum((z < 0.7) | (np.abs(pitch) > 0.2), axis=1) == 0
    r = (-p["vw"] * (vx - _S["vt"]) ** 2 + 1.0 + p["hw"] * np.minimum(z, 1.2)) * alive
    r -= 1e-3 * (cand ** 2).sum(-1)
    r -= p["pw"] * np.maximum(np.abs(pitch) - p["pth"], 0) ** 2 * alive
    r -= p["zw"] * np.maximum(1.0 - z, 0) ** 2 * alive
    r -= p["wl"] * np.maximum(-0.9 - leg, 0) ** 2 * alive
    r -= 50 * (~alive)
    _S["plan"] = cand[int(np.argmax(r.sum(1)))]


def controller(obs: dict):
    if obs["step"] == 0:
        _reset()
    state = _estimate(obs)
    if obs["step"] % PARAMS["replan"] == 0:
        _plan(state)
    a = _S["plan"][0].copy()
    _S["plan"] = np.vstack([_S["plan"][1:], _S["plan"][-1:]])
    _S["last"] = a
    return a
