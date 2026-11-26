"""Proprioceptive model-predictive walking controller for the MuJoCo Humanoid (predictive sampling).

No learning: each control step the controller
1. reconstructs the full MuJoCo state from onboard sensing
   - joint angles / rates: encoders;
   - torso attitude: IMU roll, pitch; yaw integrated from the body gyro (starts at 0);
   - root angular velocity: the body gyro (MuJoCo's free-joint angular velocity is in the body frame);
   - root linear velocity and height: a complementary filter. The accelerometer (R a_body + g is the root
     acceleration) is integrated every step; while a foot is in contact the estimate is pulled with gain
     kin_gain toward the stance-kinematics velocity (the contact point does not move: v_root = -J_rest
     qdot_rest, both feet averaged in double support) and the height with gain z_gain toward the height
     at which the lowest contacting foot rests on the floor;
2. runs predictive sampling on its own MuJoCo model (implicit-fast, 7.5 ms steps; multi-threaded ``mujoco.rollout``):
   N_SAMPLES perturbed piecewise-linear control plans (knots every KNOT_DT control steps over HORIZON
   steps) are simulated from the estimate, scored with the walking cost below, and the best plan
   becomes the new nominal; its first command is applied. Half the samples use noise sigma, half 2 sigma.
   The target speed is ramped from 0 to v_target over `ramp` seconds.

Cost per predicted control step (gains in PARAMS):
  w_vel   (vcom_x - v_target)^2 + w_vy vcom_y^2   track the forward speed (ramped over RAMP s)
  w_head  max(0, head_ref - head_z)^2              keep the head up (stay tall)
  w_up    (1 - torso_up_z)                         keep the torso upright
  w_bal   |capture_point - mean(feet)|^2           keep the capture point over the support (step!)
  w_face  torso_x_y^2                              face +x
  w_feet  max(0, feet_gap - |dy_feet|)^2           keep the feet apart laterally (no leg crossing)
  w_ctrl  |u|^2                                    effort
  w_fall  [root_z < z_fall]                        hard fall barrier
"""

import os

import mujoco
import numpy as np
from mujoco import rollout

from hl_control.envs.humanoid import ACTUATED_JOINTS, XML_PATH

PARAMS = dict(
    v_target=1.0, ramp=1.5,
    w_vel=2.0, w_vy=3.0, w_head=50.0, head_ref=1.45, w_up=5.0, w_bal=1.0, w_face=3.0,
    w_feet=50.0, feet_gap=0.12, w_ctrl=0.05, w_fall=1000.0, z_fall=1.1,
    sigma=0.15,
    kin_gain=0.03, z_gain=0.3,   # complementary-filter gains of the stance kinematics vs the integrated accelerometer
)
N_SAMPLES = 128
HORIZON = 48         # control steps (0.72 s)
KNOT_DT = 6          # control steps between knots
FRAME_SKIP = 2         # internal model substeps per 15 ms control step
SIM_DT = 0.0075
FOOT_R = 0.075


def _build_model():
    spec = mujoco.MjSpec.from_file(str(XML_PATH))
    def add(name, stype, objtype, objname):
        s = spec.add_sensor()
        s.name, s.type, s.objtype, s.objname = name, stype, objtype, objname
    add("com", mujoco.mjtSensor.mjSENS_SUBTREECOM, mujoco.mjtObj.mjOBJ_BODY, "torso")
    add("comvel", mujoco.mjtSensor.mjSENS_SUBTREELINVEL, mujoco.mjtObj.mjOBJ_BODY, "torso")
    add("rfoot", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_GEOM, "right_foot")
    add("lfoot", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_GEOM, "left_foot")
    add("up", mujoco.mjtSensor.mjSENS_FRAMEZAXIS, mujoco.mjtObj.mjOBJ_BODY, "torso")
    add("fwd", mujoco.mjtSensor.mjSENS_FRAMEXAXIS, mujoco.mjtObj.mjOBJ_BODY, "torso")
    add("head", mujoco.mjtSensor.mjSENS_FRAMEPOS, mujoco.mjtObj.mjOBJ_GEOM, "head")
    m = spec.compile()
    m.opt.timestep = SIM_DT
    m.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    m.opt.iterations = 20
    return m


class _MPC:
    def __init__(self):
        self.m = _build_model()
        self.d = mujoco.MjData(self.m)
        self.nthread = int(os.environ.get("MPC_THREADS", max(1, min(12, os.cpu_count() or 1))))
        self.datas = [mujoco.MjData(self.m) for _ in range(self.nthread)]
        self.pool = rollout.Rollout(nthread=self.nthread)
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in ACTUATED_JOINTS])
        self.dadr = np.array([self.m.jnt_dofadr[self.m.joint(n).id] for n in ACTUATED_JOINTS])
        self.feet = [self.m.geom(n).id for n in ("right_foot", "left_foot")]
        self.foot_body = [int(self.m.geom_bodyid[g]) for g in self.feet]
        self.spec = mujoco.mjtState.mjSTATE_FULLPHYSICS
        self.nstate = mujoco.mj_stateSize(self.m, self.spec)
        self.rng = np.random.default_rng(0)
        self.reset()

    def reset(self):
        self.yaw = 0.0
        self.prev_gyro = None
        self.z = 1.4
        self.v = np.zeros(3)
        self.nk = HORIZON // KNOT_DT + 2
        self.knots = np.zeros((self.nk, self.m.nu))
        self.t = 0

    # ---------------- state estimation ----------------
    def estimate(self, obs, dt):
        m, d = self.m, self.d
        gyro = np.asarray(obs["gyro_body"], float)
        r, p = float(obs["roll"]), float(obs["pitch"])
        if self.prev_gyro is not None:
            w = 0.5 * (gyro + self.prev_gyro)
            self.yaw += dt * (np.sin(r) * w[1] + np.cos(r) * w[2]) / max(np.cos(p), 0.2)
        self.prev_gyro = gyro
        y = self.yaw
        cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(y / 2), np.sin(y / 2)
        quat = np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                         cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])
        qpos = np.zeros(m.nq)
        qvel = np.zeros(m.nv)
        qpos[2] = 1.4
        qpos[3:7] = quat
        qpos[self.qadr] = obs["joint_angles"]
        qvel[3:6] = gyro
        qvel[self.dadr] = obs["joint_rates"]
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        contacts = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]
        # inertial prediction: the accelerometer is the root's specific force, R a + g = root acceleration
        R = d.xmat[1].reshape(3, 3)
        if int(obs["step"]) > 0:
            acc = R @ np.asarray(obs["accel_body"], float) + np.array([0.0, 0.0, -9.81])
            self.v = self.v + acc * dt
            self.z = self.z + self.v[2] * dt
        if any(contacts):
            bottoms = [d.geom_xpos[g][2] - FOOT_R for g in self.feet]
            zmin = min(b for b, c in zip(bottoms, contacts) if c)
            z_kin = 1.4 - zmin - 0.002
            vs = []
            jacp = np.zeros((3, m.nv))
            for k in range(2):
                if contacts[k]:
                    pt = d.geom_xpos[self.feet[k]].copy()
                    pt[2] -= FOOT_R
                    mujoco.mj_jac(m, d, jacp, None, pt, self.foot_body[k])
                    vs.append(-(jacp[:, 3:] @ qvel[3:]))
            v_kin = np.mean(vs, axis=0)
            a = 1.0 if int(obs["step"]) == 0 else PARAMS["kin_gain"]
            self.v = self.v + a * (v_kin - self.v)
            self.z = self.z + PARAMS["z_gain"] * (z_kin - self.z) if int(obs["step"]) else z_kin
        qpos[2] = self.z
        qvel[0:3] = self.v
        return qpos, qvel

    # ---------------- planning ----------------
    def plan(self, qpos, qvel, P):
        m, d = self.m, self.d
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        d.time = 0.0
        s0 = np.zeros(self.nstate)
        mujoco.mj_getState(m, d, s0, self.spec)
        # shift nominal by one control step
        tk = np.arange(self.nk) * KNOT_DT
        shifted = np.array([np.interp(tk + 1, tk, self.knots[:, j]) for j in range(m.nu)]).T
        self.knots = shifted
        N = N_SAMPLES
        sig = np.full((N, 1, 1), P["sigma"])
        sig[N // 2:] *= 2.0
        noise = self.rng.normal(size=(N, self.nk, m.nu)) * sig
        noise[0] = 0.0
        K = np.clip(self.knots[None] + noise, -0.4, 0.4)
        steps = np.arange(HORIZON)
        idx = steps // KNOT_DT
        frac = (steps % KNOT_DT) / KNOT_DT
        U = K[:, idx] * (1 - frac)[None, :, None] + K[:, idx + 1] * frac[None, :, None]  # N,H,nu
        Uphys = np.repeat(U, FRAME_SKIP, axis=1)
        state, sens = self.pool.rollout(m, self.datas, s0, Uphys)
        sens = sens[:, FRAME_SKIP - 1::FRAME_SKIP]
        state = state[:, FRAME_SKIP - 1::FRAME_SKIP]
        com, comvel = sens[..., 0:3], sens[..., 3:6]
        rf, lf = sens[..., 6:9], sens[..., 9:12]
        up, fwd, head = sens[..., 12:15], sens[..., 15:18], sens[..., 18:21]
        rootz = state[..., 3]
        tnow = self.t * 0.015
        vt = P["v_target"] * min(1.0, tnow / max(P["ramp"], 1e-3))
        cp = com[..., :2] + comvel[..., :2] * np.sqrt(np.maximum(com[..., 2:3], 0.3) / 9.81)
        fm = 0.5 * (rf[..., :2] + lf[..., :2])
        cost = (P["w_vel"] * (comvel[..., 0] - vt) ** 2 + P["w_vy"] * comvel[..., 1] ** 2
                + P["w_head"] * np.maximum(0.0, P["head_ref"] - head[..., 2]) ** 2
                + P["w_up"] * (1 - up[..., 2])
                + P["w_bal"] * np.sum((cp - fm) ** 2, axis=-1)
                + P["w_face"] * fwd[..., 1] ** 2
                + P["w_feet"] * np.maximum(0.0, P["feet_gap"] - np.abs(rf[..., 1] - lf[..., 1])) ** 2
                + P["w_ctrl"] * np.sum(U ** 2, axis=-1)
                + P["w_fall"] * (rootz < P["z_fall"]))
        total = cost.sum(axis=1)
        best = int(np.argmin(total))
        self.knots = K[best].copy()
        self.last_cost = total[best]
        return U[best, 0]


_mpc = None


def controller(obs: dict) -> np.ndarray:
    global _mpc
    if _mpc is None:
        _mpc = _MPC()
    if int(obs["step"]) == 0:
        _mpc.reset()
    qpos, qvel = _mpc.estimate(obs, float(obs["dt"]))
    u = _mpc.plan(qpos, qvel, PARAMS)
    _mpc.t += 1
    return np.clip(u, -0.4, 0.4)
