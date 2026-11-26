"""Proprioceptive state estimator + sampling-based MPC (predictive sampling) on the robot's MuJoCo model.

State estimation (each step, on-board sensing only):
  * joints / rates from the encoders; roll and pitch from the IMU; yaw by trapezoidal integration of
    the gyro yaw rate (starts at 0).
  * root velocity: the accelerometer integrated in the world frame (it is the exact finite difference
    of the root velocity up to its bias), blended with gain K_V towards the stance-foot kinematic
    velocity (the contact point of a foot on the floor does not move); K_B integrates an accel bias.
  * root height: the lowest contacting foot sphere sits on the floor (blend gain K_Z with the inertial
    prediction).
Control: joint position targets tracked by a PD law u = (KP (q* - q) - KD q') / gear, clipped to
+-0.4 (KP_ARM/KD_ARM for the light arms). The internal model has matching force-limited servo
actuators, so a sampled plan is a target trajectory: N_SAMPLES Gaussian (SIGMA) perturbations of
the nominal piecewise-linear target plan (N_KNOTS over HORIZON control steps; arms and abdomen
twist held at the pose) are rolled out in parallel (Euler, 20 solver iterations); the cheapest (or,
TEMP > 0, an MPPI average) becomes the new nominal and its first target is applied.
Cost weights: W_V (forward velocity tracking of V_TARGET), W_VY / W_YAW (sideways drift, turning),
W_Z / Z_REF (torso below the reference height), Z_MIN (fall barrier), W_UP / W_ANG (torso
uprightness and tumbling rates), W_CP / W_CPT / CP_LEAD (capture point over the feet midpoint,
running and terminal, optional lead), W_VZ (vertical COM speed), W_FLIGHT (both feet off the
floor), W_QD (joint speeds), W_CTRL (target deviation from the standing pose).
Status: NOT passing the forward gate: visible-seed episodes fall after ~100-600 steps (best 2.6 m).
"""
import os
import numpy as np
import mujoco
from mujoco import rollout as _rollout
from hl_control.envs.humanoid import XML_PATH

PARAMS = dict(V_TARGET=0.4, W_V=10.0, W_Z=200.0, Z_REF=1.22, Z_MIN=1.15, W_UP=50.0, W_ANG=0.5, W_CTRL=0.01,
              W_VY=0.5, W_YAW=0.5, W_CP=100.0, W_CPT=50.0, W_FLIGHT=1.0, W_QD=0.02, CP_LEAD=0.05, W_VZ=1.0, SIGMA=0.15, TEMP=0.0,
              KP=500.0, KD=20.0, KP_ARM=40.0, KD_ARM=1.5, K_V=0.3, K_B=0.0, K_Z=0.3)
N_SAMPLES = int(os.environ.get("MPC_N", 64))
HORIZON = int(os.environ.get("MPC_H", 50))          # control steps (15 ms each)
N_KNOTS = 5
FRAME_SKIP = 5
R_FOOT = 0.075

import json
PARAMS.update(json.loads(os.environ.get("MPC_PARAMS", "{}")))
_XML = open(XML_PATH).read().replace("</mujoco>", """<sensor><subtreecom body="torso"/><subtreelinvel body="torso"/>
<framepos objtype="geom" objname="right_foot"/><framepos objtype="geom" objname="left_foot"/></sensor></mujoco>""")
_M = mujoco.MjModel.from_xml_string(_XML)
if not os.environ.get("MPC_RK4"):
    _M.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER; _M.opt.iterations = 20  # internal model: Euler is 3x cheaper than RK4
_GEAR = _M.actuator_gear[:, 0].copy()
_D = mujoco.MjData(_M)
_NTH = int(os.environ.get("MPC_THREADS", os.cpu_count() or 4))
_DATAS = [mujoco.MjData(_M) for _ in range(_NTH)]
_ROLL = None
_FOOT = [_M.geom("right_foot").id, _M.geom("left_foot").id]
_SPEC = mujoco.mjtState.mjSTATE_FULLPHYSICS
_NS = mujoco.mj_stateSize(_M, _SPEC)
_JQ = np.array([_M.jnt_qposadr[_M.joint(n).id] for n in
                ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
                 "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
                 "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")])
_JV = np.array([_M.jnt_dofadr[_M.joint(n).id] for n in
                ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
                 "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
                 "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")])
_S = {}
_FIXED = [1, 11, 12, 13, 14, 15, 16]
_POSE = np.zeros(_M.nu); _POSE[[6, 10]] = -0.1      # nominal standing targets (slightly bent knees)


def _kp_kd():
    kp = np.full(_M.nu, PARAMS["KP"]); kd = np.full(_M.nu, PARAMS["KD"])
    kp[11:] = PARAMS["KP_ARM"]; kd[11:] = PARAMS["KD_ARM"]          # light arms: low gains keep the 15 ms loop stable
    return kp / _GEAR, kd / _GEAR      # ctrl units per rad, per rad/s


def _setup_servo_model():
    kp, kd = _kp_kd()
    _M.actuator_gaintype[:] = mujoco.mjtGain.mjGAIN_FIXED
    _M.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_AFFINE
    _M.actuator_gainprm[:, 0] = kp
    _M.actuator_biasprm[:, 0] = 0.0
    _M.actuator_biasprm[:, 1] = -kp / _GEAR          # actuator length = gear * q
    _M.actuator_biasprm[:, 2] = -kd / _GEAR
    _M.actuator_ctrlrange[:] = [-3.0, 3.0]
    _M.actuator_forcelimited[:] = 1
    _M.actuator_forcerange[:] = [-0.4, 0.4]


_setup_servo_model()
_RNG = np.random.default_rng(12345)


def _quat(roll, pitch, yaw):
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy])


def estimate(obs):
    """Reconstruct (qpos, qvel) of the MuJoCo model from proprioceptive measurements."""
    dt = obs["dt"]
    roll, pitch = obs["roll"], obs["pitch"]
    w = np.asarray(obs["gyro_body"], float)
    yd = (np.sin(roll) * w[1] + np.cos(roll) * w[2]) / max(np.cos(pitch), 0.2)
    if obs["step"] == 0 or "yaw" not in _S:
        _S.update(yaw=0.0, z=1.4, v=np.zeros(3), yd=yd)
    else:   # trapezoidal integration (the gyro carries step-to-step chatter)
        _S["yaw"] += dt * 0.5 * (yd + _S["yd"])
        _S["yd"] = yd
    qpos = np.zeros(_M.nq)
    qvel = np.zeros(_M.nv)
    qpos[3:7] = _quat(roll, pitch, _S["yaw"])
    qpos[_JQ] = obs["joint_angles"]
    qvel[3:6] = w
    qvel[_JV] = obs["joint_rates"]
    qpos[2] = 0.0
    _D.qpos[:] = qpos
    _D.qvel[:] = qvel
    mujoco.mj_kinematics(_M, _D)
    mujoco.mj_comPos(_M, _D)
    contacts = [bool(obs["foot_contact"]), bool(obs["foot_left_contact"])]
    fz = [_D.geom_xpos[g][2] - R_FOOT for g in _FOOT]
    # inertial prediction: the accelerometer integrates exactly to the root velocity (up to its bias)
    if obs["step"] == 0:
        v_pred, z_pred = np.zeros(3), 1.4
        _S["ab"] = np.zeros(3)
    else:
        rot = _D.xmat[1].reshape(3, 3)
        a = rot @ np.asarray(obs["accel_body"], float) + np.array([0.0, 0.0, -9.81]) + _S["ab"]
        v_pred = _S["v"] + dt * a
        z_pred = _S["z"] + dt * v_pred[2]
    if any(contacts):
        z_kin = -min(fz[k] for k in range(2) if contacts[k])
        vs = []
        jacp = np.zeros((3, _M.nv))
        for k in range(2):
            if contacts[k]:
                p = _D.geom_xpos[_FOOT[k]].copy(); p[2] -= R_FOOT
                mujoco.mj_jac(_M, _D, jacp, None, p, _M.geom_bodyid[_FOOT[k]])
                vs.append(jacp @ qvel)
        v_kin = -np.mean(vs, axis=0)
        kv, kz = PARAMS["K_V"], PARAMS["K_Z"]
        v = v_pred + kv * (v_kin - v_pred)
        _S["ab"] += PARAMS["K_B"] * (v_kin - v_pred)       # slow accelerometer-bias integrator
        z = z_pred + kz * (z_kin - z_pred)
    else:
        v, z = v_pred, z_pred
    _S["z"], _S["v"] = z, v
    qpos[2] = z
    qvel[0:3] = v
    return qpos, qvel


def _plan_controls(knots):
    """knots (N, K, nu) -> per-physics-step controls (N, HORIZON*FRAME_SKIP, nu), piecewise linear."""
    tk = np.linspace(0, HORIZON - 1, N_KNOTS)
    t = np.arange(HORIZON)
    idx = np.clip(np.searchsorted(tk, t, side="right") - 1, 0, N_KNOTS - 2)
    a = ((t - tk[idx]) / (tk[idx + 1] - tk[idx]))[None, :, None]
    u = knots[:, idx] * (1 - a) + knots[:, idx + 1] * a
    return np.repeat(u, FRAME_SKIP, axis=1)


def _cost(states, sens, ctrl, P):
    com, cvel = sens[..., 0:3], sens[..., 3:6]
    fr, fl = sens[..., 6:9], sens[..., 9:12]
    mid = 0.5 * (fr + fl)
    h = np.maximum(com[..., 2] - mid[..., 2], 0.3)
    cp = com[..., :2] + cvel[..., :2] * np.sqrt(h / 9.81)[..., None]
    off = cp - mid[..., :2]
    off[..., 0] -= P["CP_LEAD"]
    bal = (off ** 2).sum(-1)
    lowfoot = np.minimum(fr[..., 2], fl[..., 2]) - R_FOOT          # both feet up = flight (hopping)
    flight = np.maximum(lowfoot - 0.01, 0.0)
    qd = states[:, :, 1 + _M.nq + 6:1 + _M.nq + _M.nv]
    return (_cost0(states, ctrl, P) + P["W_CP"] * bal.mean(1) + P["W_CPT"] * bal[:, -1]
            + P["W_FLIGHT"] * (flight ** 2).mean(1) * 1e4 + P["W_QD"] * (qd ** 2).sum(-1).mean(1)) + P["W_VZ"] * (cvel[..., 2] ** 2).mean(1)


def _cost0(states, ctrl, P):
    # states: (N, T, ns) FULLPHYSICS: [time, qpos(24), qvel(23), ...]
    qpos = states[:, :, 1:1 + _M.nq]
    qvel = states[:, :, 1 + _M.nq:1 + _M.nq + _M.nv]
    z = qpos[..., 2]
    q = qpos[..., 3:7]
    upz = 1 - 2 * (q[..., 1] ** 2 + q[..., 2] ** 2)        # torso z axis . world z
    vx, vy = qvel[..., 0], qvel[..., 1]
    c = (P["W_V"] * (vx - P["V_TARGET"]) ** 2 + P["W_VY"] * vy ** 2
         + P["W_Z"] * np.minimum(z - P["Z_REF"], 0) ** 2 + P["W_UP"] * (1 - upz) ** 2
         + P["W_YAW"] * qvel[..., 5] ** 2 + P["W_ANG"] * (qvel[..., 3] ** 2 + qvel[..., 4] ** 2)
         + 1000.0 * (z < P["Z_MIN"]))
    return c.mean(axis=1) + P["W_CTRL"] * (ctrl ** 2).sum(-1).mean(1)


def controller(obs: dict) -> np.ndarray:
    global _ROLL
    P = PARAMS
    qpos, qvel = estimate(obs)
    if obs["step"] == 0 or "nom" not in _S:
        _S["nom"] = np.repeat(_POSE[None], N_KNOTS, 0)
        _S["t"] = 0
    if _ROLL is None:
        _ROLL = _rollout.Rollout(nthread=_NTH)
    _D.qpos[:] = qpos; _D.qvel[:] = qvel; _D.time = 0.0
    _D.act[:] = 0
    s0 = np.zeros(_NS)
    mujoco.mj_getState(_M, _D, s0, _SPEC)
    # shift nominal by one control step
    nom = _S["nom"]
    tk = np.linspace(0, HORIZON - 1, N_KNOTS)
    full = _plan_controls(nom[None])[0, ::FRAME_SKIP]
    tq = np.clip(tk + 1, 0, HORIZON - 1).astype(int)
    nom = full[tq]
    knots = np.repeat(nom[None], N_SAMPLES, axis=0)
    noise = P["SIGMA"] * _RNG.standard_normal((N_SAMPLES - 1, N_KNOTS, _M.nu))
    noise[..., _FIXED] = 0.0                     # arms and abdomen twist stay at the nominal pose
    knots[1:] += noise
    knots[1:] *= 1.0
    ctrl = _plan_controls(knots)
    states, sens = _ROLL.rollout([_M] * N_SAMPLES, _DATAS, np.repeat(s0[None], N_SAMPLES, 0), ctrl,
                              initial_warmstart=None)
    cost = _cost(states, sens, ctrl - _POSE, P)
    best = int(np.argmin(cost))
    if P["TEMP"] > 0:   # MPPI: cost-weighted average of the samples
        wts = np.exp(-(cost - cost[best]) / (P["TEMP"] * (np.std(cost) + 1e-9)))
        _S["nom"] = np.tensordot(wts / wts.sum(), knots, axes=1)
    else:
        _S["nom"] = knots[best]
    knots = _S["nom"][None]
    best = 0
    kp, kd = _kp_kd()
    u = kp * (knots[best][0] - obs["joint_angles"]) - kd * obs["joint_rates"]
    return np.clip(u, -0.4, 0.4)
