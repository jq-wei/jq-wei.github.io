"""Humanoid get-up: scripted PD posture phases from the back, an LQR catch in a deep squat, gain-scheduled LQR rise.

Control law (state machine, reset when obs["step"] == 0)
--------------------------------------------------------
1. situp   - PD on joint targets, ctrl = clip((KP (q* - q) - KD qdot) / 10, +-0.4):
             raise the straight legs (hip flexion, T_RAISE s), drop them (hip extension, T_DROP s), then flex hips and
             abdomen (AB_S) hard for T_SWING s so the trunk swings up off the floor, then tuck (hips HIP_T, knees KNEE_T,
             abdomen AB_T, stiffness KP_T) so the forward momentum carries the body over the feet.
2. catch   - during the tuck the quadratic cost-to-go V = x'Px of the deep-squat LQR is monitored (x = state error
             about the squat equilibrium; heading and horizontal position removed, both are symmetries of the plant).
             The LQR takes over at the closest approach (V starts to grow again while below exp(MIN_THR)) or as soon as
             V < exp(LOG_THR), and holds the squat for HOLD_T s.
3. rise    - the reference slides along 11 equilibria from the deep squat (hip -1.9, knee -2.7, abdomen -1.0 rad) to
             the upright stance (hip -0.1, knee -0.2) in RISE_T s: gains and feed-forward torques interpolated, the
             reference velocity fed forward; the upright equilibrium is held to the end.
   If the tuck never reaches the catch region, or the robot falls early enough to try again, it relaxes for 1.5 s
   (all joint targets 0) and repeats the sit-up.

Equilibria: COM over the midpoint of the ball feet (root pitch solved), root height where the contact carries the weight,
feed-forward torque from mj_inverse. LQR: discrete Riccati on a centred finite-difference linearisation of one 15 ms
control step (5 RK4 substeps, eps 1e-4, solver warm start held) of the plant's own MuJoCo model, computed at import;
Q = 1000 |J_com - J_feet|^2 (horizontal) + I on the joint angles, R = 10 I.

Status: stands on 3 of seeds 0-59 (seeds 6, 25, 47; all face-up starts), far below the 0.9 gate. The squat catch is
fragile (its basin is small compared with the spread of arrival states after the sit-up) and there is no roll-over for
the face-down and side starts (about half the seeds), which therefore never get past lying.
"""

import numpy as np
import scipy.linalg
from hl_control.envs import humanoid as H

mujoco = H.mujoco

PARAMS = dict(T_RAISE=0.481, T_DROP=0.294, T_SWING=0.288, KNEE_T=-2.505, AB_T=-0.048, HIP_T=-1.73, AB_S=-0.343,
              KP_T=77.854, LOG_THR=7.339, RISE_T=6.0, HOLD_T=1.0)
MIN_THR = 9.0
PARAM_BOUNDS = dict(T_RAISE=(0.4, 0.9), T_DROP=(0.15, 0.35), T_SWING=(0.15, 0.5), KNEE_T=(-2.79, -1.8), AB_T=(-1.3, 0.0),
                    HIP_T=(-1.92, -1.2), AB_S=(-1.2, 0.0), KP_T=(20.0, 80.0), LOG_THR=(4.0, 9.0), RISE_T=(4.0, 9.0),
                    HOLD_T=(0.3, 2.0))

MODEL = mujoco.MjModel.from_xml_path(str(H.XML_PATH))
DATA = mujoco.MjData(MODEL)
NV, NU = MODEL.nv, MODEL.nu
MOTOR_DOF = np.array([MODEL.jnt_dofadr[MODEL.actuator_trnid[i, 0]] for i in range(NU)])
MOTOR_QPOS = np.array([MODEL.jnt_qposadr[MODEL.actuator_trnid[i, 0]] for i in range(NU)])
GEAR = MODEL.actuator_gear[:, 0].copy()
FOOT_R, FOOT_L = MODEL.body('right_foot').id, MODEL.body('left_foot').id
TORSO = MODEL.body('torso').id
GFR, GFL = MODEL.geom('right_foot').id, MODEL.geom('left_foot').id
J = {n: i for i, n in enumerate(H.ACTUATED_JOINTS)}


def pose(**kw):
    """Motor-order joint targets; hip_y/knee/hip_x/hip_z set both sides."""
    q = np.zeros(NU)
    for k, v in kw.items():
        if k in ('hip_y', 'knee', 'hip_x', 'hip_z'):
            q[J['right_' + k]] = q[J['left_' + k]] = v
        else:
            q[J[k]] = v
    return q


def pd(obs, q_t, kp=40.0, kd=4.0):
    return np.clip((kp * (q_t - obs["joint_angles"]) - kd * obs["joint_rates"]) / 10.0, -0.4, 0.4)


# ----------------------------------------------------------------------------------------------- equilibria and LQR
def equilibrium(qj_motor, iters=40):
    """Root pose putting the COM over the feet midpoint with the contact carrying the weight; feed-forward ctrl."""
    m, d = MODEL, DATA
    qpos = m.qpos0.copy(); qpos[MOTOR_QPOS] = qj_motor
    th = 0.0
    for _ in range(iters):
        qpos[3:7] = [np.cos(th / 2), 0, np.sin(th / 2), 0]; qpos[0:3] = 0
        d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
        com = H.mass_center(m, d); feet = 0.5 * (d.geom_xpos[GFR] + d.geom_xpos[GFL])
        err = com[0] - feet[0]
        th -= np.arcsin(np.clip(err / max(com[2] - feet[2], 0.1), -0.5, 0.5))
        if abs(err) < 1e-7:
            break
    qpos[3:7] = [np.cos(th / 2), 0, np.sin(th / 2), 0]
    d.qpos[:] = qpos; mujoco.mj_kinematics(m, d)
    qpos[2] -= min(d.geom_xpos[GFR][2], d.geom_xpos[GFL][2]) - 0.075
    feet = 0.5 * (d.geom_xpos[GFR] + d.geom_xpos[GFL]); qpos[0] -= feet[0]; qpos[1] -= feet[1]
    best = None
    for off in np.linspace(-0.01, 0.01, 401):
        d.qpos[:] = qpos; d.qpos[2] += off; d.qvel[:] = 0; d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        if best is None or abs(d.qfrc_inverse[2]) < best[0]:
            best = (abs(d.qfrc_inverse[2]), off)
    qpos[2] += best[1]
    d.qpos[:] = qpos; d.qvel[:] = 0; d.qacc[:] = 0
    mujoco.mj_inverse(m, d)
    return qpos, d.qfrc_inverse[MOTOR_DOF] / GEAR


def transition(qpos0, ctrl0, eps=1e-4, nsub=5):
    """Centred finite-difference (A, B) of one control step of the true plant (RK4), warm start held fixed."""
    m = MODEL; d = mujoco.MjData(m)
    d.qpos[:] = qpos0; d.ctrl[:] = ctrl0; mujoco.mj_forward(m, d)
    for _ in range(3):
        d.qpos[:] = qpos0; d.qvel[:] = 0; mujoco.mj_step(m, d)
    ws = d.qacc_warmstart.copy()

    def step(dx, du):
        mujoco.mj_resetData(m, d); d.qacc_warmstart[:] = ws
        q = qpos0.copy(); mujoco.mj_integratePos(m, q, dx[:NV], 1.0)
        d.qpos[:] = q; d.qvel[:] = dx[NV:]; d.ctrl[:] = ctrl0 + du
        mujoco.mj_step(m, d, nstep=nsub)
        out = np.zeros(2 * NV); mujoco.mj_differentiatePos(m, out[:NV], 1.0, qpos0, d.qpos); out[NV:] = d.qvel
        return out
    A = np.zeros((2 * NV, 2 * NV)); B = np.zeros((2 * NV, NU)); z = np.zeros(2 * NV); zu = np.zeros(NU)
    for k in range(2 * NV):
        e = np.zeros(2 * NV); e[k] = eps
        A[:, k] = (step(e, zu) - step(-e, zu)) / (2 * eps)
    for k in range(NU):
        e = np.zeros(NU); e[k] = eps
        B[:, k] = (step(z, e) - step(z, -e)) / (2 * eps)
    return A, B


def lqr(qpos0, ctrl0, balance_cost=1000.0, r=10.0):
    A, B = transition(qpos0, ctrl0)
    DATA.qpos[:] = qpos0; mujoco.mj_forward(MODEL, DATA)
    jc = np.zeros((3, NV)); mujoco.mj_jacSubtreeCom(MODEL, DATA, jc, TORSO)
    jr = np.zeros((3, NV)); jl = np.zeros((3, NV))
    mujoco.mj_jacBodyCom(MODEL, DATA, jr, None, FOOT_R); mujoco.mj_jacBodyCom(MODEL, DATA, jl, None, FOOT_L)
    jd = (jc - 0.5 * (jr + jl))[:2]
    Qj = np.eye(NV); Qj[:6, :6] = 0
    Q = scipy.linalg.block_diag(balance_cost * jd.T @ jd + Qj, np.zeros((NV, NV)))
    Q[np.arange(6), np.arange(6)] += 1e-6
    R = r * np.eye(NU)
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A), P


SQ = np.array([-1.9, -2.7, -1.0]); ST = np.array([-0.1, -0.2, 0.0])     # (hip_y, knee, abdomen_y): deep squat, stance
PATH = []
for _s in np.linspace(0, 1, 11):
    _h, _k, _a = SQ + _s * (ST - SQ)
    _q, _c = equilibrium(pose(hip_y=_h, knee=_k, abdomen_y=_a))
    _K, _P = lqr(_q, _c)
    PATH.append((_s, _q, _c, _K, _P))
SS = np.array([p[0] for p in PATH])


def ref_at(s):
    s = float(np.clip(s, 0, 1))
    i = min(int(np.searchsorted(SS, s, side='right')) - 1, len(PATH) - 2)
    a = (s - SS[i]) / (SS[i + 1] - SS[i])
    q = PATH[i][1] * (1 - a) + PATH[i + 1][1] * a; q[3:7] /= np.linalg.norm(q[3:7])
    return q, PATH[i][2] * (1 - a) + PATH[i + 1][2] * a, PATH[i][3] * (1 - a) + PATH[i + 1][3] * a


def state_err(obs, qref, vref=None):
    """State error about qref from gym_obs (qpos[2:], qvel); heading and horizontal position removed."""
    g = obs["gym_obs"]
    qpos = np.zeros(MODEL.nq); qpos[2:] = g[0:22]; qvel = g[22:45].copy()
    R = np.zeros(9); mujoco.mju_quat2Mat(R, qpos[3:7])
    psi = np.arctan2(R[3], R[0])                                  # heading of the torso's x axis
    qn = np.zeros(4); mujoco.mju_mulQuat(qn, np.array([np.cos(-psi / 2), 0, 0, np.sin(-psi / 2)]), qpos[3:7])
    qpos[3:7] = qn
    c, s = np.cos(-psi), np.sin(-psi); vx, vy = qvel[0], qvel[1]
    qvel[0] = c * vx - s * vy; qvel[1] = s * vx + c * vy           # world-frame linear velocity, de-yawed
    qpos[0:2] = qref[0:2]
    dq = np.zeros(NV); mujoco.mj_differentiatePos(MODEL, dq, 1, qref, qpos)
    if vref is not None:
        qvel = qvel - vref
    return np.concatenate([dq, qvel])


def lqr_ctrl(obs, s, sdot=0.0):
    qref, cref, K = ref_at(s)
    vref = None
    if sdot > 0.0 and s < 1.0:
        q2, _, _ = ref_at(s + 0.01)
        vref = np.zeros(NV); mujoco.mj_differentiatePos(MODEL, vref, 0.01 / sdot, qref, q2)
    return np.clip(cref - K @ state_err(obs, qref, vref), -0.4, 0.4)


# ----------------------------------------------------------------------------------------------------- state machine
class _State:
    mode = "situp"
    t0 = 0.0
    nfall = 0


S = _State()


def _situp_phases(p):
    return [(p['T_RAISE'], pose(hip_y=-1.6, knee=-0.1), 40, 4),
            (p['T_DROP'], pose(hip_y=0.35, knee=-0.1), 80, 2),
            (p['T_SWING'], pose(hip_y=-1.9, knee=-0.1, abdomen_y=p['AB_S']), 80, 2),
            (1.5, pose(hip_y=p['HIP_T'], knee=p['KNEE_T'], abdomen_y=p['AB_T']), p['KP_T'], 2)]


def _control(obs: dict) -> np.ndarray:
    p = PARAMS
    t = float(obs["time"])
    if int(obs["step"]) == 0:
        S.mode, S.t0, S.nfall = "situp", t, 0
        S.vprev = 1e18
    if S.mode == "situp":
        ph = _situp_phases(p); tt = t - S.t0
        starts = np.cumsum([0] + [x[0] for x in ph])
        k = min(int(np.searchsorted(starts, tt, side='right')) - 1, len(ph) - 1)
        if k >= 3:
            dx = state_err(obs, PATH[0][1]); V = float(dx @ PATH[0][4] @ dx)
            vprev = getattr(S, 'vprev', 1e18); S.vprev = V
            if (V > vprev and vprev < np.exp(MIN_THR)) or V < np.exp(p['LOG_THR']):
                S.mode, S.t0, S.nfall = "catch", t, 0
                return lqr_ctrl(obs, 0.0)
            if tt > starts[-1]:                       # the tuck missed the catch region: lie back and retry
                S.mode, S.t0 = "relax", t
        _, qt, kp, kd = ph[k]
        return pd(obs, qt, kp, kd)
    if S.mode == "relax":
        if t - S.t0 > 1.5:
            S.mode, S.t0 = "situp", t
        return pd(obs, np.zeros(NU), 40, 4)
    # catch / rise: retry from the floor if fallen and there is still time for a full attempt
    S.nfall = S.nfall + 1 if (obs["height"] < 0.33 or obs["up_vector"][2] < 0.2) else 0
    if S.nfall > 20 and t < 15.0 - 2.0 - p['RISE_T'] - p['HOLD_T'] - 3.0:
        S.mode, S.t0, S.nfall = "relax", t, 0
        return pd(obs, np.zeros(NU), 40, 4)
    if S.mode == "catch":
        if t - S.t0 > p['HOLD_T']:
            S.mode, S.t0 = "rise", t
        return lqr_ctrl(obs, 0.0)
    s = (t - S.t0) / p['RISE_T']
    return lqr_ctrl(obs, s, 1.0 / p['RISE_T'] if s < 1.0 else 0.0)


def controller(obs: dict) -> np.ndarray:
    """Motor commands in [-0.4, 0.4] (motor order); non-finite values are replaced by zero."""
    u = np.nan_to_num(np.asarray(_control(obs), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(u, -0.4, 0.4)
