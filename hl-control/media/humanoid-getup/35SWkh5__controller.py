"""Get-up controller for the lying MuJoCo Humanoid: roll onto the back, sit-up throw into a deep squat, LQR climb.

Honest status: this stands up on most but not enough starts and does NOT meet the 0.9 stand gate. Measured with
the evaluator on getup_v0, seeds 0-39: stand 28/40 (0.70), crouch 31/40, sit 37/40. On the starts where the throw
misses, it falls back to sitting upright (first rung).

Control law, a state machine (`GetUp`):

  roll    any settled start -> lying face up. Segmental roll with joint PD targets (`SupineRoller`): hips fold with
          bent knees, the waist twists, hip rotations tip the pelvis, the waist counter-twist drags the torso over.
  lie     short wind-up on the back (hips at `lie_hip`, waist at `lie_abd`, arms forward) for `t_first` seconds.
  throw   explosive straight-leg sit-up (PD stiffness x `throw_kp`, which saturates the hips). The knees fold as the
          trunk comes up (between up-vector z = `knee_u0` and `knee_u1`, latched), so the feet arrive next to the
          pelvis while the trunk still moves forward: the body pivots onto its feet and hovers in a deep squat.
  climb   when the mass centre is within `catch_x2` of the feet, a gain-scheduled LQR takes over. At the catch a path
          of `n_knots` statically balanced postures (mass centre over the two point feet, equilibrium torques from
          inverse dynamics) is built from the measured squat posture to standing, each with its own LQR gain from a
          finite-difference linearisation of the plant model with the feet in contact (five physics steps per command).
          The path parameter advances at `climb_rate` knots/s only while the tracking error is below `gate`
          (stop-and-go); the last knot is the standing posture, which the same LQR then holds.
          Along the path the waist straightens early (`abd_lead`) and the arms come down early (`arm_lead`): held
          forward they saturate their 10 N m motors once the trunk is upright and the balance is lost.
  retry   a fall during the climb or a missed throw restarts from the roll while `t < t_last_try`.
  sit     fallback: if no climb is under way at `t_fallback`, roll onto the back, sit up (hips `sit_hip`, legs spread
          by `sit_spread`) and hold the trunk upright.

Gains: PD_KP / PD_KD are joint PD gains in command units per rad and per rad/s (low on purpose: stiffer D terms
chatter at the 15 ms command hold). LQR weights (`_lqr_gain`): `balance_cost` on the horizontal offset between
mass centre and feet midpoint, `joint_w` / `root_w` on posture, `vel_w` on velocities, `r_w` on commands.
`clip_pos`, `clip_ang`, `clip_vel` bound the state error passed to the LQR; `t_fade` blends its command in.

Known weak point: the throw is sensitive to the posture it starts from, so the squat is caught on only part of the
starts (its settings come from a random search on ten visible seeds); once caught, the climb usually completes.
The perturbed sibling scenarios were not evaluated with this controller.
"""
from __future__ import annotations

import numpy as np
import mujoco
import scipy.linalg

from hl_control.envs.humanoid import XML_PATH, ACTUATED_JOINTS

PARAMS = dict(
    t_lie=0.9,         # s, blend time scale of the lying wind-up targets (the throw starts before the blend ends)
    lie_hip=-0.401,      # rad, hip angle lying
    lie_abd=-0.178,       # rad, waist angle lying
    arm_swing=0,       # 1: arms go overhead while lying and swing forward in the throw
    t_first=0.4,      # s, wind-up before the first throw
    knee_u0=0.403,       # trunk elevation (up-vector z) at which the knees start to fold during the sit-up
    knee_u1=0.923,       # trunk elevation at which the knees are fully folded
    throw_abd=-1.094,    # rad, waist target of the sit-up throw
    throw_kp=4.0,      # PD stiffness multiplier of the throw (saturates the hips)
    omega=6.0,         # 1/s, pendulum rate used for the capture point
    coast_kp=3.838,      # PD stiffness multiplier while coasting over the feet
    catch_x2=-0.037,    # m, mass-centre offset at which the LQR takes over after coasting
    t_throw_max=1.2,   # s, give up the throw after this long and lie back for another attempt
    clip_pos=0.1,     # m, largest root position error passed to the LQR
    clip_ang=0.5,      # rad, largest angle error passed to the LQR
    clip_vel=2.0,      # rad/s (m/s), largest velocity passed to the LQR
    t_fade=0.1,        # s, the LQR command is blended in over this time after the catch
    abd_lead=1.0,      # the waist is straight after 1/abd_lead of the path
    arm_lead=1.5,      # the arms hang down after 1/arm_lead of the path
    n_knots=12,        # postures on the path from the squat to standing
    w_abd=10.0,        # LQR posture weight of the waist joints
    squat_abd=-0.9,    # rad, waist angle of the squat posture where the LQR catches the throw
    catch_h=0.5,       # m, root height required for the catch
    s_catch=0.0,       # path index of the deep squat, where the LQR catches the throw
    t_fallback=9.0,    # s, if no climb is under way by then, roll over and sit up instead (first rung)
    sit_hip=-1.5,      # rad, hip flexion sitting with the trunk vertical
    sit_spread=-0.3,   # rad, hip abduction of each leg while sitting
    fall_h=0.33,       # m, root height below which a climb counts as fallen
    t_last_try=6.5,    # s, no new attempt is started after this time
    climb_rate=2.0,    # knots/s, speed of the reference posture along the path
    gate=0.15,         # rad, tracking error at which the reference slows down
)

# ----------------------------------------------------------------------------------------------- model
MODEL = mujoco.MjModel.from_xml_string(XML_PATH.read_text())
DATA = mujoco.MjData(MODEL)
MODEL_LIN = mujoco.MjModel.from_xml_string(XML_PATH.read_text())
MODEL_LIN.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER   # transitionFD does not support RK4
MODEL_LIN.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON           # converged contact forces: clean finite differences
MODEL_LIN.opt.iterations = 200
MODEL_LIN.opt.tolerance = 1e-12
NV, NU = MODEL.nv, MODEL.nu
QADR = np.array([MODEL.jnt_qposadr[MODEL.joint(n).id] for n in ACTUATED_JOINTS])
DADR = np.array([MODEL.jnt_dofadr[MODEL.joint(n).id] for n in ACTUATED_JOINTS])
GEAR = MODEL.actuator_gear[:, 0].copy()
RF, LF = MODEL.geom("right_foot").id, MODEL.geom("left_foot").id
TORSO = MODEL.body("torso").id
JI = {n: i for i, n in enumerate(ACTUATED_JOINTS)}

PD_KP = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
PD_KD = np.array([0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])
ARM_FWD = np.array([-0.05, 0.65, -1.5, 0.05, -0.65, -1.5])    # arms reaching forward (right, left)
ARM_UP = np.array([-1.45, 0.3, -1.5, 1.45, -0.3, -1.5])       # arms overhead (wind-up of the throw)
ARM_ZERO = np.zeros(6)
ARM_DOWN = np.array([0.65, -0.75, -1.3, -0.65, 0.75, -1.3])       # arms hanging beside the hips (almost no holding torque)


def _pose(knee, hip, abd, arms=ARM_ZERO):
    j = np.zeros(NU)
    j[JI["right_knee"]] = j[JI["left_knee"]] = knee
    j[JI["right_hip_y"]] = j[JI["left_hip_y"]] = hip
    j[JI["abdomen_y"]] = abd
    j[11:] = arms
    return j


def _balanced_pose(joints, pitch_guess):
    """Root pose such that the mass centre is over the feet midpoint and the feet carry the weight; equilibrium commands."""
    m, d = MODEL, DATA
    q = np.zeros(m.nq)
    q[QADR] = joints

    def place(pitch):
        q[0:3] = (0.0, 0.0, 2.0)
        q[3:7] = [np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0]
        d.qpos[:] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        feet = 0.5 * (d.geom_xpos[RF] + d.geom_xpos[LF])
        com = m.body_mass @ d.xipos / m.body_mass.sum()
        return com[0] - feet[0], feet

    a, b = pitch_guess - 1.2, pitch_guess + 1.2
    fa, _ = place(a)
    for _ in range(50):
        c = 0.5 * (a + b)
        fc, _ = place(c)
        if (fc > 0) == (fa > 0):
            a, fa = c, fc
        else:
            b = c
    pitch = 0.5 * (a + b)
    _, feet = place(pitch)
    q[0], q[1] = -feet[0], -feet[1]
    q[2] = 2.0 - (min(d.geom_xpos[RF][2], d.geom_xpos[LF][2]) - 0.075)
    best = None
    for dz in np.linspace(-0.01, 0.002, 131):       # height at which the floor carries the weight
        d.qpos[:] = q
        d.qpos[2] += dz
        d.qvel[:] = 0
        d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        r = abs(d.qfrc_inverse[2])
        if best is None or r < best[0]:
            best = (r, dz, d.qfrc_inverse.copy())
    q[2] += best[1]
    return q, best[2][DADR] / GEAR, pitch


def _lqr_gain(qpos0, ctrl0, balance_cost=1000.0, r_w=3.0, vel_w=0.01, root_w=(1.0, 1.0, 30.0, 100.0, 100.0, 10.0),
              joint_w=np.r_[np.full(11, 10.0), np.full(6, 3.0)], nsub=5):
    m, d = MODEL_LIN, DATA
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    d.ctrl[:] = ctrl0
    A = np.zeros((2 * NV, 2 * NV))
    B = np.zeros((2 * NV, NU))
    mujoco.mjd_transitionFD(m, d, 1e-5, True, A, B, None, None)
    An, Bn = np.eye(2 * NV), np.zeros((2 * NV, NU))
    for _ in range(nsub):                            # one control step = five physics steps with the command held
        Bn = A @ Bn + B
        An = A @ An
    mujoco.mj_resetData(m, d)
    d.qpos[:] = qpos0
    mujoco.mj_forward(m, d)
    jc, jr, jl = np.zeros((3, NV)), np.zeros((3, NV)), np.zeros((3, NV))
    mujoco.mj_jacSubtreeCom(m, d, jc, TORSO)
    mujoco.mj_jacGeom(m, d, jr, None, RF)
    mujoco.mj_jacGeom(m, d, jl, None, LF)
    jd = (jc - 0.5 * (jr + jl))[:2]
    w = np.zeros(NV)
    w[:6] = root_w
    w[DADR] = joint_w
    Q = np.zeros((2 * NV, 2 * NV))
    Q[:NV, :NV] = balance_cost * jd.T @ jd + np.diag(w)
    Q[NV:, NV:] = vel_w * np.eye(NV)
    R = r_w * np.eye(NU)
    P = scipy.linalg.solve_discrete_are(An, Bn, Q, R)
    return np.linalg.solve(R + Bn.T @ P @ Bn, Bn.T @ P @ An)


def _build_path(j0=None, n=None):
    """Balanced postures from the deep squat in which the throw arrives to standing. `j0` is the measured joint posture at
    the catch (made left/right symmetric), so the LQR starts with almost no posture error."""
    if j0 is None:
        j0 = _pose(-2.7, -1.85, PARAMS["squat_abd"], ARM_FWD)
    j1 = _pose(-0.15, -0.1, 0.0, j0[11:])           # arms stay where they are: moving them across the thighs collides
    n = int(PARAMS["n_knots"]) if n is None else n
    jw = np.r_[np.full(3, PARAMS["w_abd"]), np.full(8, 10.0), np.full(6, 3.0)]
    out, pg = [], 0.5
    for f in np.linspace(0, 1, n + 1):
        j = (1 - f) * j0 + f * j1
        j[0] = j0[0] * max(0.0, 1.0 - PARAMS["abd_lead"] * f)      # the waist straightens early: less load on its weak motor
        g = min(1.0, PARAMS["arm_lead"] * f)                         # the arms come down early: held forward they saturate
        j[11:] = (1 - g) * j0[11:] + g * ARM_DOWN                    # their 10 N m motors once the trunk is upright
        q0, c0, pg = _balanced_pose(j, pg)
        out.append((q0, c0, _lqr_gain(q0, c0, joint_w=jw)))
    return out


def _symmetric(j):
    """Left/right average of a joint posture in motor order (waist twist/bend and hip rotations set to zero)."""
    j = np.array(j, dtype=float)
    out = np.zeros(NU)
    out[0] = j[0]
    out[5] = out[9] = 0.5 * (j[5] + j[9])
    out[6] = out[10] = min(0.5 * (j[6] + j[10]), -0.1)
    out[11], out[14] = 0.5 * (j[11] - j[14]), -0.5 * (j[11] - j[14])
    out[12], out[15] = 0.5 * (j[12] - j[15]), -0.5 * (j[12] - j[15])
    out[13] = out[16] = 0.5 * (j[13] + j[16])
    return out


PATH = _build_path()
_PATH = {"p": PATH}
ERR_CLIP = np.r_[np.full(3, PARAMS["clip_pos"]), np.full(NV - 3, PARAMS["clip_ang"])]
VEL_CLIP = PARAMS["clip_vel"]
_D2 = mujoco.MjData(MODEL)


def _reconstruct(obs):
    """Plant state in the linearisation's coordinates: heading removed, horizontal position relative to the feet midpoint."""
    g = obs["gym_obs"]
    qpos = np.zeros(MODEL.nq)
    qpos[2:] = g[:22]
    qvel = np.array(g[22:45], dtype=float)
    w, x, y, z = qpos[3:7]
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    c, s = np.cos(-yaw), np.sin(-yaw)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.array([np.cos(-yaw / 2), 0, 0, np.sin(-yaw / 2)]), qpos[3:7])
    qpos[3:7] = quat
    vx, vy = qvel[0], qvel[1]
    qvel[0], qvel[1] = c * vx - s * vy, s * vx + c * vy
    _D2.qpos[:] = qpos
    mujoco.mj_kinematics(MODEL, _D2)
    feet = 0.5 * (_D2.geom_xpos[RF] + _D2.geom_xpos[LF])
    qpos[0], qpos[1] = -feet[0], -feet[1]
    return qpos, qvel


def _path_ref(s):
    PATH = _PATH["p"]
    n = len(PATH)
    s = float(np.clip(s, 0, n - 1))
    k = min(int(np.floor(s)), n - 2)
    f = s - k
    qa, ca, Ka = PATH[k]
    qb, cb, Kb = PATH[k + 1]
    q = (1 - f) * qa + f * qb
    q[3:7] /= np.linalg.norm(q[3:7])
    return q, (1 - f) * ca + f * cb, (1 - f) * Ka + f * Kb


def _lqr(obs, s):
    q_ref, c_ref, K = _path_ref(s)
    qpos, qvel = _reconstruct(obs)
    dq = np.zeros(NV)
    mujoco.mj_differentiatePos(MODEL, dq, 1.0, q_ref, qpos)
    lim = np.r_[np.full(3, PARAMS["clip_pos"]), np.full(NV - 3, PARAMS["clip_ang"])]
    dqc = np.clip(dq, -lim, lim)                     # bounded error: a large initial offset must not bang the motors
    return c_ref - K @ np.concatenate([dqc, np.clip(qvel, -PARAMS["clip_vel"], PARAMS["clip_vel"])]), dq


def _axes(obs):
    w, x, y, z = obs["gym_obs"][1:5]
    front = np.array([1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y)])
    up = np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    return front, up


# ------------------------------------------------------------------------------- roll onto the back
# Segmental roll with joint PD targets (any start -> face up); written for a roll over the left side, mirrored for the right.

ROLL_KP = np.array([1.5, 1.5, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5])
ROLL_KD = np.array([0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03])

ROLL_P = dict(
    t_twist=0.35, t_fold=0.6, t_torso=0.5, t_tilt=0.5, t_straight=0.7,
    hip_flex=-1.45, knee_flex=-1.2, twist=0.75, hz_in=0.55, hz_out=-1.0,
    side_fz=-0.8,      # starts with front_z above this are already off their front: skip the twist phase
    t_min=0.2, rho_fold=92.0, rho_torso=60.0, rho_tilt=25.0, rho_again=95.0,   # early phase exits / repeat threshold (deg)
)

ARM_SAFE = np.array([0.8, -1.4, 0.87, -0.8, 1.4, 0.87])   # folded back: cannot prop the chest
ARM_SIDE = np.array([0.3, -0.8, 0.87, -0.3, 0.8, 0.87])   # beside the body when it is on its back
ARM_WEDGE = np.array([-1.05, -0.4, -0.5])                 # left-arm values (roll-left frame): shoulder1 at its limit, forearm under the chest


def _mirror(v):
    m = np.empty(17)
    m[0] = v[0]; m[1] = -v[1]; m[2] = -v[2]
    m[3:7] = v[7:11]; m[7:11] = v[3:7]
    m[11] = -v[14]; m[12] = -v[15]; m[13] = v[16]
    m[14] = -v[11]; m[15] = -v[12]; m[16] = v[13]
    return m


def _body_z(obs):
    w, x, y, z = obs["gym_obs"][1:5]
    return 2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)   # z-components of front, left, long axes


class SupineRoller:
    def __init__(self):
        self.phase = "init"
        self.t0 = 0.0
        self.side = 1.0
        self.calm = 0
        self.done = False
        self.attempt = 0
        self.cycles = 0
        self.flips = 0
        self.rho_t = 180.0

    def _targets(self, phase):
        tg = np.zeros(17)
        tg[6] = tg[10] = -0.05
        tg[11:17] = ARM_SIDE
        if phase in ("twist", "fold"):
            tg[11:14] = 0.0                # top-side arm in front of the chest: it props that side up
        elif phase in ("torso", "tilt"):
            tg[11:14] = ARM_SAFE[0:3]      # top-side arm folded behind the back: its weight pulls the torso over
        if phase in ("twist", "fold", "torso") or (phase == "tilt" and not -90.0 < self.rho_t < 75.0):
            tg[14:17] = ARM_WEDGE          # bottom-side arm wedged under the chest (as in the propped starts): the torso stays
                                           # off its front, so the waist twist turns the pelvis and the counter-twist the torso
        if phase == "twist":
            tg[1] = -ROLL_P["twist"]
        elif phase == "fold":
            tg[5] = tg[9] = ROLL_P["hip_flex"]; tg[6] = tg[10] = ROLL_P["knee_flex"]
            tg[1] = -ROLL_P["twist"]
        elif phase == "torso":
            tg[5] = tg[9] = ROLL_P["hip_flex"]; tg[6] = tg[10] = ROLL_P["knee_flex"]
            tg[1] = ROLL_P["twist"]
        elif phase == "tilt":
            tg[5] = tg[9] = ROLL_P["hip_flex"]; tg[6] = tg[10] = ROLL_P["knee_flex"]
            tg[1] = ROLL_P["twist"]; tg[4] = ROLL_P["hz_in"]; tg[8] = ROLL_P["hz_out"]
        return tg

    def step(self, obs):
        t = float(obs["time"])
        fz, lz, uz = _body_z(obs)
        q = np.asarray(obs["joint_angles"], dtype=float)
        qd = np.asarray(obs["joint_rates"], dtype=float)
        wnorm = float(np.linalg.norm(obs["root_angular_velocity"]))
        supine = fz >= 0.85 and obs["height"] < 0.25
        if self.phase == "init":
            if supine:
                self.phase = "hold"
            else:
                self.side = 1.0 if lz <= 0.0 else -1.0
                if self.attempt % 2 == 1:
                    self.side = -self.side
                self.phase = "twist" if fz <= ROLL_P["side_fz"] else "fold"
                self.cycles = 0
                self.flips = 0
            self.t0 = t
        el = t - self.t0
        nxt = {"twist": "fold", "fold": "torso", "torso": "tilt", "tilt": "straight", "straight": "hold"}
        tmax = {"twist": ROLL_P["t_twist"], "fold": ROLL_P["t_fold"], "torso": ROLL_P["t_torso"], "tilt": ROLL_P["t_tilt"], "straight": ROLL_P["t_straight"]}
        # roll angles about the long axis in the roll frame: 0 face up, 90 on the roll side, 180 face down
        rho_t = np.degrees(np.arctan2(-lz * self.side, fz))
        rho_p = rho_t + np.degrees(q[1]) * self.side
        if self.phase in ("fold", "torso", "tilt") and -150.0 < rho_t < -30.0 and self.flips < 2:
            # the body went over its other side: adopt that side (mirror the frame) and carry on from the fold
            self.side = -self.side
            self.flips += 1
            rho_t = np.degrees(np.arctan2(-lz * self.side, fz))
            rho_p = rho_t + np.degrees(q[1]) * self.side
            self.phase, self.t0, el = "fold", t, 0.0
        early = ((self.phase == "fold" and el > ROLL_P["t_min"] and 0.0 < rho_p < ROLL_P["rho_fold"])
                 or (self.phase == "torso" and el > ROLL_P["t_min"] and -90.0 < rho_t < ROLL_P["rho_torso"])
                 or (self.phase == "tilt" and el > ROLL_P["t_min"] and -90.0 < rho_t < ROLL_P["rho_tilt"]))
        if self.phase in nxt and (el > tmax[self.phase] or early):
            far = rho_t > ROLL_P["rho_again"] or rho_t < -90.0
            if self.phase == "torso" and not early and far and self.cycles < 2:
                self.cycles += 1
                self.phase, self.t0 = "fold", t      # torso still on the far side of its edge: one more fold / counter-twist cycle
            elif self.phase == "tilt" and not early and (rho_t > 80.0 or rho_t < -90.0) and self.cycles < 4:
                self.cycles += 1
                self.phase, self.t0 = "fold", t      # straightening now would drop the body back on its front: cycle again
            else:
                self.phase, self.t0 = nxt[self.phase], t
        elif self.phase == "hold" and el > 0.5 and not supine and not self.done:
            self.attempt += 1
            self.phase = "init"
        if supine and wnorm < 1.0 and self.phase in ("hold", "straight"):
            self.calm += 1
        else:
            self.calm = 0
        if self.calm >= 14:
            self.done = True
        if self.done:
            self.phase = "hold"
        self.rho_t = float(np.degrees(np.arctan2(-lz * self.side, fz)))
        tg = self._targets(self.phase if self.phase != "init" else "hold")
        if self.side < 0:
            q, qd = _mirror(q), _mirror(qd)
        u = np.clip(ROLL_KP * (tg - q) - ROLL_KD * qd, -0.4, 0.4)
        if self.side < 0:
            u = _mirror(u)
        return u, self.done


# ------------------------------------------------------------------------------------- get-up sequence
class GetUp:
    def __init__(self):
        self.phase = "start"
        self.t0 = 0.0
        self.s = 0.0
        self.prev = None
        self.hold = 0
        self.tries = 0
        self.pushed = False
        self.xi_max = -9.0
        self.log = []

    def _go(self, phase, obs):
        self.phase = phase
        self.t0 = obs["time"]
        self.prev = np.array(obs["joint_angles"], dtype=float)
        self.log.append((round(float(obs["time"]), 2), phase))

    def _pd(self, obs, goal, dur, kp=2.0):
        a = min(1.0, (obs["time"] - self.t0 + obs["dt"]) / max(dur, 1e-3))
        tg = (1 - a) * self.prev + a * goal
        return kp * PD_KP * (tg - obs["joint_angles"]) - PD_KD * obs["joint_rates"]

    def com_fwd(self, obs):
        """Mass centre ahead (+) or behind (-) of the feet midpoint along the torso's heading [m], and its rate."""
        front, up = _axes(obs)
        h = front[:2] - 0.5 * up[:2]          # well defined both lying on the back and sitting up
        n = np.linalg.norm(h)
        h = h / n if n > 1e-6 else np.array([1.0, 0.0])
        off = -0.5 * (obs["foot_xyz_rel"] + obs["foot_left_xyz_rel"])[:2]
        return float(off @ h), float(np.asarray(obs["com_velocity_xy"]) @ h)

    def step(self, obs):
        P = PARAMS
        t = obs["time"]
        if self.phase not in ("climb", "sitroll", "sit") and (t >= P["t_fallback"] or (self.phase == "start" and t >= P["t_last_try"])):
            self.roller = SupineRoller()         # too late for another attempt: settle for sitting up
            self._go("sitroll", obs)
        if self.phase == "sitroll":
            u, done = self.roller.step(obs)
            if done:
                self._go("sit", obs)
            return u
        if self.phase == "sit":
            el = t - self.t0
            goal = _pose(-0.05, P["sit_hip"], -0.5 if el < 0.75 else 0.0, ARM_FWD)
            goal[3] = goal[7] = P["sit_spread"]
            return self._pd(obs, goal, 0.45)
        if self.phase == "start":
            self.roller = SupineRoller()
            self._go("roll", obs)
        if self.phase == "roll":
            u, done = self.roller.step(obs)
            if done:
                self.tries = 0
                self.pushed = False
                self._go("lie", obs)
            return u
        if self.phase == "lie":
            # flat on the back, legs straight, arms forward: the starting posture of the sit-up throw
            u = self._pd(obs, _pose(-0.05, P["lie_hip"], P["lie_abd"], ARM_UP if P["arm_swing"] else ARM_FWD), 0.7 * P["t_lie"])
            if t - self.t0 >= P["t_lie"] or (self.tries == 0 and t - self.t0 >= P["t_first"]):
                self.fold = 0.0
                self.pushed = False
                self._go("throw", obs)
                self.tries += 1
            return u
        if self.phase == "throw":
            # explosive sit-up; the knees fold as the trunk comes up so that the feet arrive next to the pelvis while the
            # trunk still moves forward. Hand over to the LQR once the mass centre is within catch_x2 of the feet
            # (xi, the capture point x + v / omega, is only tracked as a diagnostic).
            upz = float(obs["up_vector"][2])
            self.fold = max(self.fold, float(np.clip((upz - P["knee_u0"]) / (P["knee_u1"] - P["knee_u0"]), 0.0, 1.0)))
            a = self.fold                                # latched: the knees never unfold again during a throw
            goal = _pose(-0.05 - 2.67 * a, -1.88, P["throw_abd"], ARM_FWD)
            u = P["throw_kp"] * PD_KP * (goal - obs["joint_angles"]) - PD_KD * obs["joint_rates"]
            x, v = self.com_fwd(obs)
            xi = x + v / P["omega"]
            self.xi_max = max(self.xi_max, xi) if upz > 0.6 else self.xi_max
            if x >= P["catch_x2"] and upz > 0.6 and obs["height"] > P["catch_h"]:
                self._go("climb", obs)                   # hovering over the feet in the tuck: the LQR takes over
                try:
                    _PATH["p"] = _build_path(_symmetric(obs["joint_angles"]))
                except Exception:
                    _PATH["p"] = PATH
                self.s = P["s_catch"]
            elif t - self.t0 >= P["t_throw_max"]:
                self._go("start", obs)                   # missed: roll flat and throw again
            return u
        # climb / stand: scheduled LQR along the path of balanced postures
        if obs["height"] < P["fall_h"] and t - self.t0 > 0.5:
            self._go("start", obs)           # fell: roll onto the back and try again (or sit up if it is too late)
            return np.zeros(NU)
        u, dq = _lqr(obs, self.s)
        a = min(1.0, (t - self.t0 + obs["dt"]) / max(P["t_fade"], 1e-3))
        if a < 1.0:
            u_pd = P["coast_kp"] * PD_KP * (_pose(-2.72, -1.88, P["throw_abd"], ARM_FWD) - obs["joint_angles"]) - PD_KD * obs["joint_rates"]
            u = (1 - a) * np.clip(u_pd, -0.4, 0.4) + a * np.clip(u, -0.4, 0.4)
        err = np.linalg.norm(dq[6:17]) + 0.5 * np.linalg.norm(dq[3:6])
        if t - self.t0 > 0.15:
            self.s = min(self.s + P["climb_rate"] * (P["n_knots"] / 12.0) * obs["dt"] * float(np.clip(1.5 - err / P["gate"], 0.0, 1.0)), len(_PATH["p"]) - 1.0)
        return u


_STATE = {"ctl": None}


def controller(obs: dict) -> np.ndarray:
    if _STATE["ctl"] is None or obs["step"] == 0:
        _STATE["ctl"] = GetUp()
    u = _STATE["ctl"].step(obs)
    return np.clip(np.nan_to_num(np.asarray(u, dtype=float)), -0.4, 0.4)
