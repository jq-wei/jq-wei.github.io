"""MPPI (sampling MPC) hopper controller with a model-based observer for sensing blackouts.

Observer: physics are nominal, so a private copy of the Hopper MuJoCo model is kept. A fresh
measurement resets it to the measured state (x = 0); a bit-identical repeated measurement
marks a blackout, and the copy is advanced one control step with ``last_action``.

Planner: each step, N noisy control sequences (H steps, noise held over K-step knots,
std sigma) around the shifted previous plan are rolled out on the copy; per-step cost
  w_vt (vx - v_cap)^2 - w_v min(vx, v_cap)          speed tracking (v_cap = 0.5 m/s target)
  + w_pitch pitch^2 + w_h (z - h_ref)^2 + w_low min(z - z_low, 0)^2 + w_zv vz^2
  + w_knee knee^2 + w_post ((thigh - th_ref)^2 + 0.3 foot^2) + w_u |u|^2   posture/effort
summed until the first unhealthy state (|pitch| > pitch_lim or z < z_lim), plus `fail`.
The plan is the softmax(-cost / (lam std(cost))) average; its first command is applied.
"""

import os
import numpy as np
import mujoco
from mujoco import rollout

from hl_control.envs.hopper import build_xml
_XML = build_xml()

PARAMS = {
    "N": 48, "H": 60, "sigma": 0.3, "lam": 0.1,
    "v_cap": 0.5, "w_v": 0.0, "w_pitch": 20.0, "w_pr": 0.5, "w_h": 20.0, "w_zv": 0.1, "h_ref": 1.25,
    "fail": 200.0, "pitch_lim": 0.17, "z_lim": 0.78, "w_u": 0.01, "seed": 0, "K": 6, "w_knee": 3.0, "threads": 8, "w_post": 3.0, "w_vt": 5.0, "w_back": 0.0, "th_ref": -0.15, "w_low": 200.0, "z_low": 1.15,
}

_FULL = mujoco.mjtState.mjSTATE_FULLPHYSICS


class _Observer:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        self.prev_key = None

    def update(self, obs):
        key = np.concatenate([obs["gym_obs"], [obs["x_velocity"], float(obs["foot_contact"])]])
        stale = self.prev_key is not None and np.array_equal(key, self.prev_key) and obs["step"] > 0
        self.prev_key = key
        d = self.data
        if not stale:
            d.qpos[:] = [0.0, obs["height"], obs["pitch"], obs["thigh"], obs["leg"], obs["foot"]]
            d.qvel[:] = [obs["x_velocity"], obs["z_velocity"], obs["pitch_rate"],
                         obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]]
            mujoco.mj_forward(self.model, d)
        else:
            d.ctrl[:] = obs["last_action"]
            mujoco.mj_step(self.model, d, nstep=4)
            d.qpos[0] = 0.0
        st = np.zeros(mujoco.mj_stateSize(self.model, _FULL))
        mujoco.mj_getState(self.model, d, st, _FULL)
        return st, stale


class _MPPI:
    def __init__(self):
        P = PARAMS
        self.model = mujoco.MjModel.from_xml_string(_XML)
        self.obs = _Observer(self.model)
        nth = max(1, min(P["threads"], os.cpu_count() or 1))
        self.datas = [mujoco.MjData(self.model) for _ in range(nth)]
        self.U = np.zeros((P["H"], 3))
        self.rng = np.random.default_rng(P["seed"])

    def act(self, obs):
        P = PARAMS
        st, _ = self.obs.update(obs)
        N, H, K = P["N"], P["H"], P["K"]
        eps = self.rng.normal(0.0, P["sigma"], size=(N, H // K, 3))
        eps = np.repeat(eps, K, axis=1)
        eps[0] = 0.0
        U = np.clip(self.U[None] + eps, -1, 1)
        ctrl = np.repeat(U, 4, axis=1)
        states, _ = rollout.rollout(self.model, self.datas, st[None], ctrl, persistent_pool=True)
        s = states[:, 3::4, :]                    # (N, H, 13) after each control step
        z, pitch = s[..., 2], s[..., 3]
        xv, pr = s[..., 7], s[..., 9]
        bad = (np.abs(pitch) > P["pitch_lim"]) | (z < P["z_lim"])
        alive = np.cumprod(~bad, axis=1).astype(float)
        c = (-P["w_v"] * np.minimum(xv, P["v_cap"]) + P["w_vt"] * (xv - P["v_cap"]) ** 2 + P["w_back"] * np.minimum(xv, 0.0) ** 2 + P["w_pitch"] * pitch ** 2 + P["w_pr"] * 0.01 * pr ** 2
             + P["w_h"] * (z - P["h_ref"]) ** 2 + P["w_zv"] * s[..., 8] ** 2 + P["w_u"] * np.sum(U ** 2, axis=2)
             + P["w_knee"] * s[..., 5] ** 2
             + P["w_post"] * ((s[..., 4] - P["th_ref"]) ** 2 + 0.3 * s[..., 6] ** 2)
             + P["w_low"] * np.minimum(z - P["z_low"], 0.0) ** 2)
        cost = np.sum(c * alive, axis=1) + P["fail"] * (1.0 - alive[:, -1]) * (1.0 + np.sum(1 - alive, axis=1) / H)
        cmin = cost.min()
        lam = P["lam"] * max(np.std(cost), 1e-6)
        w = np.exp(-(cost - cmin) / lam)
        w /= w.sum()
        Unew = np.einsum("n,nhk->hk", w, U)
        a = Unew[0].copy()
        self.U = np.vstack([Unew[1:], Unew[-1:]])
        return np.clip(a, -1, 1)


_state = {"ctl": None, "step": -1}


def controller(obs: dict) -> np.ndarray:
    if _state["ctl"] is None or obs["step"] <= _state["step"]:
        _state["ctl"] = _MPPI()
    _state["step"] = obs["step"]
    return _state["ctl"].act(obs)
