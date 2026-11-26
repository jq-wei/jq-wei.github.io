"""Getting the MuJoCo Humanoid up from the floor: roll -> kneel -> tip onto the feet -> LQR-balanced squat rise.

Control law (a state machine; every phase is joint torque = feed-forward + feedback, clipped to +-0.4):

1. ROLL    Joint-space PD on a few keyframes turns any lying pose to prone; the roll angle of the torso about its long
           axis (from the root quaternion) picks the side and gates the steps (ROLL_P: timings and keyframe angles,
           ROLL_KP / ROLL_KD: PD gains in N m/rad and N m s/rad).
2. KNEEL   Time-interpolated PD keyframes (KEYSPEC): heels to the seat, fold the trunk, put the feet down, sink the
           pelvis onto the heels, raise the trunk into a heel-sit (KNEEL_KP / KNEEL_KD; PARAMS["t_sink"] is the earliest
           time the heel-sit is accepted).
3. TIP     Still kneeling, PD + feed-forward (not on the knees) ramps the joints over PARAMS["T_t"] to those of the first
           squat equilibrium E0, which moves the mass centre back to just ahead of the feet (PARAMS["settle"]: wait).
4. RISE    Discrete LQR about a path of static equilibria on the two sphere feet (SquatPath), from the deep squat E0
           (knees just clear of the floor, PARAMS["th_off"]) to a stance with slightly bent knees. Each equilibrium
           (pose, holding torques) comes from the plant's own model at import (inverse dynamics with the unactuated root
           residual zero); each gain from a finite-difference linearisation of that model over one control step with
           Q = qbal * |mass centre - feet|^2 + joint weights (qj_sag hips, qj_knee knees, qj_ab abdomen), R = r * I and a
           discount disc for the neutral heading mode. The reference is indexed by the measured knee angle plus a lead
           (PARAMS["lead"], grown by "creep" up to "lead_max" when the rise stalls, speed-limited by "rate"), so the
           balance loop always works next to an equilibrium and the lead, through the stiff knee weight, is what
           extends the legs. The LQR is engaged softly: its initial state error is faded in over PARAMS["Tw"] seconds.
           Near the end of the path the gain blends into an undiscounted standing gain (q_top) that holds the posture.

Known limits (measured on getup_v0): the engagement of the LQR from the kneeling state is the weak step. It only works
reliably when E0 is the pose whose knees just clear the floor (PARAMS["th_off"] = 0: stand rate 0.05 / 0.75 / 0.85 /
0.90 / 0.95 / 1.00 on seeds 0-39 for th_off = 0.08 / 0.05 / 0.03 / 0.02 / 0.01 / 0, at T_t = 2.0, rate = 0.3). With the
final gains (th_off = 0, T_t = 2.5, rate = 0.4) the stand rate is 96/100 on seeds 0-99; the rest fall at the engagement
or arrive a few tenths of a second late. Nothing recovers a fall (no time for a second attempt in 15 s).
"""
import numpy as np
import mujoco
import scipy.linalg
from hl_control.envs.humanoid import XML_PATH, ACTUATED_JOINTS

# ---------------------------------------------------------------- balance model: equilibria and LQR gains

M = mujoco.MjModel.from_xml_path(str(XML_PATH))
D = mujoco.MjData(M)
ML = mujoco.MjModel.from_xml_path(str(XML_PATH))      # linearisation copy: Euler + Newton (mjd_transitionFD has no RK4)
ML.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER
ML.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
ML.opt.tolerance = 1e-10
DL = mujoco.MjData(ML)
NV, NU = M.nv, M.nu
QADR = {n: int(M.jnt_qposadr[M.joint(n).id]) for n in ACTUATED_JOINTS}
DADR = {n: int(M.jnt_dofadr[M.joint(n).id]) for n in ACTUATED_JOINTS}
GEAR = M.actuator_gear[:, 0].copy()
FOOT_G = [M.geom("right_foot").id, M.geom("left_foot").id]
FOOT_B = [M.body("right_foot").id, M.body("left_foot").id]
TORSO = M.body("torso").id


def pose_q(z, theta, H, K, A, W=-0.1, arms=(0.0, 0.0, -0.6), Zr=0.0):
    """qpos for a left-right symmetric pose: root height z, root pitch theta, hip flexion H (>0), knee flexion K (>0),
    abdomen flexion A (>0), hip abduction W (hip_x value, negative = legs apart)."""
    q = M.qpos0.copy()
    q[0:2] = 0.0
    q[2] = z
    q[3:7] = [np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0]
    q[QADR["abdomen_y"]] = -A
    for s in ("right", "left"):
        q[QADR[f"{s}_hip_x"]] = W
        q[QADR[f"{s}_hip_z"]] = Zr
        q[QADR[f"{s}_hip_y"]] = -H
        q[QADR[f"{s}_knee"]] = -K
        sg = 1.0 if s == "right" else -1.0          # mirrored shoulder axes: a symmetric pose has opposite shoulder angles
        q[QADR[f"{s}_shoulder1"]] = sg * arms[0]
        q[QADR[f"{s}_shoulder2"]] = sg * arms[1]
        q[QADR[f"{s}_elbow"]] = arms[2]
    return q


def _resid(q):
    D.qpos[:] = q; D.qvel[:] = 0.0; D.qacc[:] = 0.0; D.ctrl[:] = 0.0
    mujoco.mj_inverse(M, D)
    return D.qfrc_inverse.copy()


def _solve_z(q):
    """Root height at which the feet carry the weight (vertical force residual zero)."""
    D.qpos[:] = q; mujoco.mj_kinematics(M, D)
    low = min(D.geom_xpos[g][2] for g in FOOT_G) - 0.075
    z0 = q[2] - low                       # feet just touching
    lo, hi = z0 - 0.03, z0 + 0.002        # more penetration -> more upward contact force -> residual changes sign
    q = q.copy()
    for _ in range(40):
        mid = 0.5 * (lo + hi); q[2] = mid
        if _resid(q)[2] > 0.0:            # still needs an upward external force: not enough contact force -> go lower
            hi = mid
        else:
            lo = mid
    q[2] = 0.5 * (lo + hi)
    return q


def lqr_gain(q, ctrl0, qbal=1000.0, qj_sag=3.0, qj_other=30.0, qvel=0.0, r=1.0, nsub=5, disc=0.99, r_knee=1.0, r_ab=1.0, q_pitch=0.0, qj_ab=None, qj_knee=None, drop_yaw=False):
    mujoco.mj_resetData(ML, DL)
    DL.qpos[:] = q; DL.qvel[:] = 0.0; DL.ctrl[:] = ctrl0
    A = np.zeros((2 * NV, 2 * NV)); B = np.zeros((2 * NV, NU))
    mujoco.mjd_transitionFD(ML, DL, 1e-6, True, A, B, None, None)
    Ac = np.linalg.matrix_power(A, nsub)
    Bc = sum(np.linalg.matrix_power(A, i) for i in range(nsub)) @ B
    D.qpos[:] = q; D.qvel[:] = 0.0; mujoco.mj_forward(M, D)
    jc = np.zeros((3, NV)); mujoco.mj_jacSubtreeCom(M, D, jc, TORSO)
    jf = np.zeros((3, NV)); tmp = np.zeros((3, NV))
    for g in FOOT_G:
        mujoco.mj_jacGeom(M, D, tmp, None, g); jf += 0.5 * tmp
    jd = (jc - jf)[:2]
    Qp = qbal * jd.T @ jd
    w = np.zeros(NV)
    for n in ACTUATED_JOINTS:
        sag = n in ("abdomen_y", "right_hip_y", "left_hip_y", "right_knee", "left_knee")
        w[DADR[n]] = qj_sag if sag else qj_other
    if qj_ab is not None: w[DADR['abdomen_y']] = qj_ab
    if qj_knee is not None: w[DADR['right_knee']] = w[DADR['left_knee']] = qj_knee
    w[3] = w[4] = q_pitch                  # root roll / pitch (torso frame): hold the trunk attitude
    Qp += np.diag(w)
    Q = np.zeros((2 * NV, 2 * NV)); Q[:NV, :NV] = Qp; Q[NV:, NV:] = qvel * np.eye(NV)
    rr = r * np.ones(NU); rr[[6, 10]] *= r_knee; rr[0] *= r_ab
    R = np.diag(rr)
    keep = np.r_[2:5, 6:2 * NV] if drop_yaw else np.r_[2:2 * NV]     # drop root x, y (and heading): unobserved / neutral, dynamics invariant
    g = np.sqrt(disc)
    Ar, Br, Qr = g * Ac[np.ix_(keep, keep)], g * Bc[keep], Q[np.ix_(keep, keep)]
    P = scipy.linalg.solve_discrete_are(Ar, Br, Qr, R)
    Kr = np.linalg.solve(R + Br.T @ P @ Br, Br.T @ P @ Ar)
    Kg = np.zeros((NU, 2 * NV)); Kg[:, keep] = Kr
    return Kg


def canon(gym_obs):
    """Observed state -> (qpos, qvel) in the heading-free frame (x = y = 0, zero twist about the vertical)."""
    q = np.zeros(M.nq); v = gym_obs[22:45].copy()
    q[2:] = gym_obs[0:22]
    w, x, y, z = q[3:7]
    psi = 2.0 * np.arctan2(z, w)
    c, s = np.cos(-psi / 2), np.sin(-psi / 2)
    q[3:7] = [c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w]     # Rz(-psi) * q
    cp, sp = np.cos(-psi), np.sin(-psi)
    v[0], v[1] = cp * v[0] - sp * v[1], sp * v[0] + cp * v[1]
    return q, v


def lqr_ctrl(gym_obs, qref, ctrl0, Kg):
    q, v = canon(gym_obs)
    q[0:2] = qref[0:2]
    dq = np.zeros(NV)
    mujoco.mj_differentiatePos(M, dq, 1.0, qref, q)
    return ctrl0 - Kg @ np.concatenate([dq, v])


def _others(q):
    """Contacts other than foot-floor in pose q (names), and knee height."""
    D.qpos[:] = q; D.qvel[:] = 0.0; mujoco.mj_forward(M, D)
    out = []
    for i in range(D.ncon):
        g1, g2 = int(D.contact.geom[i][0]), int(D.contact.geom[i][1])
        if not ({g1, g2} & set(FOOT_G) and 0 in (g1, g2)):
            out.append(M.geom(g1).name + "-" + M.geom(g2).name)
    return out, float(D.xpos[M.body("right_shin").id][2])


def equilibrium2(theta, K, A, W=-0.1, arms=(0.0, 0.0, -0.6), H_near=None, Zr=0.0):
    """Like equilibrium() but scans H for a clean sign change (no contacts other than the feet); picks the root nearest H_near."""
    Hs = np.arange(-0.34, 1.91, 0.075)
    vals = []
    for H in Hs:
        q = _solve_z(pose_q(1.3, theta, H, K, A, W, arms, Zr))
        ok = not _others(q)[0]
        vals.append((_resid(q)[4], ok))
    roots = []
    for i in range(len(Hs) - 1):
        (fa, oka), (fb, okb) = vals[i], vals[i + 1]
        if oka and okb and fa * fb <= 0:
            a, b = Hs[i], Hs[i + 1]
            for _ in range(30):
                c = 0.5 * (a + b); q = _solve_z(pose_q(1.3, theta, c, K, A, W, arms, Zr)); fc = _resid(q)[4]
                if fa * fc <= 0: b = c
                else: a, fa = c, fc
            roots.append(c)
    if not roots:
        return None
    H = min(roots, key=lambda h: abs(h - (H_near if H_near is not None else h)))
    q = _solve_z(pose_q(1.3, theta, H, K, A, W, arms, Zr))
    r = _resid(q)
    ctrl0 = np.array([r[DADR[n]] for n in ACTUATED_JOINTS]) / GEAR
    return q, ctrl0, H, _others(q)[1]

# ---------------------------------------------------------------- squat path (gain schedule)

TH_END, A_END, K_END = 0.15, 0.15, 0.5      # final stance: knees kept bent (clear of the joint stop)


def build_knots(A, K0=2.7, K1=2.4, n1=6, th_off=0.0, W_end=-0.2, Zr_end=-0.43, **kw):
    th0 = None
    for th in np.arange(-1.0, 0.6, 0.05):
        e = equilibrium2(th, K0, A, **kw)
        if e is not None:
            th0 = th; break
    for th in np.arange(th0 - 0.05, th0 + 0.001, 0.01):     # refine: the first pose whose knees clear the floor
        e = equilibrium2(th, K0, A, **kw)
        if e is not None:
            th0 = th + th_off; break
    assert th0 is not None
    W0, Z0 = kw['W'], kw['Zr']
    knots = [(K0 + (K1 - K0) * i / n1, th0 + (0.0 - th0) * i / n1, A, W0, Z0) for i in range(n1 + 1)]
    for K in list(np.arange(K1 - 0.2, K_END + 0.19, -0.2)) + [K_END]:  # extend the knees; end with the trunk slightly forward and
        f = (K1 - K) / (K1 - K_END)                                    # the abdomen straighter, so the hips keep room to extend
        knots.append((K, TH_END * f * f, A + (A_END - A) * f, W0 + (W_end - W0) * f, Z0 + (Zr_end - Z0) * f))   # legs un-splay as they extend
    return knots

class SquatPath:
    def __init__(self, A=0.3, W=-0.1, arms=(0.0, 0.0, -0.6), verbose=False, qj_ab=None, th_off=0.0, Zr=0.0, W_end=-0.2, Zr_end=-0.43, q_top=30.0, **lq):
        self.knots = build_knots(A, th_off=th_off, W=W, arms=arms, Zr=Zr, W_end=W_end, Zr_end=Zr_end)
        self.Q, self.U, self.K, self.Kn = [], [], [], []
        H = None
        for (K, th, A, W, Zr) in self.knots:
            e = equilibrium2(th, K, A, W=W, arms=arms, H_near=H, Zr=Zr)
            assert e is not None, (K, th)
            q, c0, H, kz = e
            D.qpos[:] = q; mujoco.mj_kinematics(M, D)
            q[0] = -0.5 * (D.geom_xpos[FOOT_G[0]][0] + D.geom_xpos[FOOT_G[1]][0])
            self.Q.append(q); self.U.append(c0); self.K.append(lqr_gain(q, c0, qj_ab=qj_ab, **lq)); self.Kn.append(K)
            if verbose: print(f"K={K:.2f} W={W:.2f} Zr={Zr:.2f} th={th:+.2f} H={H:+.2f} z={q[2]:.2f} kneeZ={kz:.2f} umax={np.abs(c0).max():.2f} gain={np.abs(self.K[-1]).max():.0f}")
        self.n = len(self.knots)
        self.Kn = np.array(self.Kn)
        # standing gain for the last pose: undiscounted, heading left out, stiffer posture weights (no slow posture drift)
        top = dict(lq); top.update(disc=1.0, drop_yaw=True, qj_sag=q_top, qj_knee=q_top, qj_other=max(q_top, 30.0))
        try:
            self.K_top = lqr_gain(self.Q[-1], self.U[-1], qj_ab=q_top, **top)
        except Exception:
            self.K_top = self.K[-1]

    def s_of_knee(self, K):
        return float(np.interp(-K, -self.Kn, np.linspace(0.0, 1.0, self.n)))    # knee flexion decreases monotonically along the path

    def ref(self, s):
        x = np.clip(s, 0.0, 1.0) * (self.n - 1)
        i = min(int(x), self.n - 2); f = x - i
        q = (1 - f) * self.Q[i] + f * self.Q[i + 1]
        q[3:7] /= np.linalg.norm(q[3:7])
        K = (1 - f) * self.K[i] + f * self.K[i + 1]
        b = float(np.clip((s - 0.9) / 0.1, 0.0, 1.0))                 # hand over to the standing gain near the top
        return q, (1 - f) * self.U[i] + f * self.U[i + 1], (1 - b) * K + b * self.K_top


# ---------------------------------------------------------------- phase 1: roll to prone

ROLL_NAMES = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
         "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
         "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
ROLL_J = {n: i for i, n in enumerate(ROLL_NAMES)}
ROLL_KP = np.array([80, 80, 80, 60, 60, 200, 120, 60, 60, 200, 120, 15, 15, 15, 15, 15, 15], float)
ROLL_KD = np.array([2, 2, 2, 1, 1, 4, 1.5, 1, 1, 4, 1.5, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3], float)
ROLL_P = dict(t_plant=0.35, t_push=0.6, hip_plant=-1.0, knee_plant=-1.9, abd_plant=-0.35, hip_push=0.3, knee_push=-0.6,
         twist=0.7, arm_mode=0)


def rotmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def roll_angle(obs):
    """rho: 0 face down, +-pi face up, +pi/2 left side up (lying on the right side)."""
    R = rotmat(obs["gym_obs"][1:5])
    return float(np.arctan2(R[2, 1], -R[2, 0])), R


def arm_pose(tgt, side, pose):
    s = 1.0 if side == "right" else -1.0   # shoulder joints are mirrored between the sides
    tgt[ROLL_J[f"{side}_shoulder1"]] = s * pose[0]
    tgt[ROLL_J[f"{side}_shoulder2"]] = s * pose[1]
    tgt[ROLL_J[f"{side}_elbow"]] = pose[2]


ARM_BACK = (-1.48, -1.48, -1.57)     # arm behind the back / out to the side, nothing in front of the chest
ARM_ZERO = (0.0, 0.0, 0.0)
ARM_SIDE = (0.63, -0.64, -1.57)     # arm straight down along the flank, in the frontal plane


def pd(obs, tgt, scale=None):
    q, qd = obs["joint_angles"], obs["joint_rates"]
    kp = ROLL_KP if scale is None else ROLL_KP * scale
    tau = kp * (tgt - q) - ROLL_KD * qd
    return np.clip(tau / obs["gear"], -0.4, 0.4)


def straight_target():
    tgt = np.zeros(17)
    tgt[ROLL_J["right_knee"]] = tgt[ROLL_J["left_knee"]] = -0.05
    arm_pose(tgt, "right", ARMS[ROLL_P["arm_end"]]); arm_pose(tgt, "left", ARMS[ROLL_P["arm_end"]])
    return tgt


ARM_TUCK = (1.05, -0.22, 0.87)     # arm folded in front of the chest, inside the body width
ARMS = {"side": ARM_SIDE, "back": ARM_BACK, "zero": ARM_ZERO, "tuck": ARM_TUCK}
ROLL_P.update(t_s1=0.6, t_s2=0.8, s1_hip=-1.5, s1_knee=-0.4, s1_hz=0.6, s2_hip=-1.0, s2_knee=-1.2, arm_dn="tuck", arm_up="tuck", arm_end="tuck",
         push_abd=-0.35, dn_hip=0.0, t_back=0.35, back_dn_hip=-0.9, back_dn_knee=-0.9, s2_push_max=140.0)


def roll_act(obs, mem):
    t = float(obs["time"])
    rho, R = roll_angle(obs)
    if not mem:
        mem["phase"] = "init"; mem["t0"] = t; mem["tries"] = 0
    if mem["phase"] == "init":
        if abs(rho) < np.radians(25):
            mem["phase"] = "settle"
        else:
            mem["d"] = 1.0 if rho > 0 else -1.0
            mem["phase"] = "s1" if abs(rho) > np.radians(110) else "push"
        mem["t0"] = t
    ph, tp = mem["phase"], t - mem["t0"]
    tgt = straight_target()
    if ph in ("s1", "s2", "push"):  # roll phases
        d = mem["d"]; tw = -d
        up, dn = ("left", "right") if d > 0 else ("right", "left")   # `up` side goes over the top; body rolls over `dn`
        arm_pose(tgt, dn, ARMS[ROLL_P["arm_dn"]]); arm_pose(tgt, up, ARMS[ROLL_P["arm_up"]])
        tgt[ROLL_J[f"{dn}_hip_y"]] = ROLL_P["dn_hip"]
        if ph == "s1":      # raise the up-side leg, cross it over and twist the pelvis towards the roll side
            tgt[ROLL_J[f"{up}_hip_y"]] = ROLL_P["s1_hip"]; tgt[ROLL_J[f"{up}_knee"]] = ROLL_P["s1_knee"]; tgt[ROLL_J[f"{up}_hip_z"]] = ROLL_P["s1_hz"]
            tgt[ROLL_J[f"{up}_hip_x"]] = 0.08
            tgt[ROLL_J["abdomen_z"]] = ROLL_P["twist"] * tw
            if tp > ROLL_P["t_s1"]:
                mem["phase"] = "s2"; mem["t0"] = t
        elif ph == "s2":    # leg down in front as an anchor, torso follows the pelvis and then leads
            tgt[ROLL_J[f"{up}_hip_y"]] = ROLL_P["s2_hip"]; tgt[ROLL_J[f"{up}_knee"]] = ROLL_P["s2_knee"]; tgt[ROLL_J[f"{up}_hip_z"]] = ROLL_P["s1_hz"]
            tgt[ROLL_J["abdomen_z"]] = -ROLL_P["twist"] * tw
            if abs(rho) < np.radians(70) or tp > ROLL_P["t_s2"]:
                mem["phase"] = "push" if abs(rho) < np.radians(ROLL_P["s2_push_max"]) else "init"; mem["t0"] = t; mem["tries"] += 1
        else:               # half-turned: top leg extends behind, pelvis leads to prone
            tgt[ROLL_J[f"{up}_hip_y"]] = ROLL_P["hip_push"]; tgt[ROLL_J[f"{up}_knee"]] = ROLL_P["knee_push"]; tgt[ROLL_J[f"{up}_hip_x"]] = ROLL_P["push_abd"]
            tgt[ROLL_J["abdomen_z"]] = ROLL_P["twist"] * tw
            if (abs(rho) < np.radians(35) and np.sign(rho) == d) or abs(rho) < np.radians(15):
                mem["phase"] = "settle"; mem["t0"] = t
            elif tp > ROLL_P["t_push"]:
                mem["tries"] += 1
                mem["phase"] = "back" if abs(rho) < np.radians(110) else "init"; mem["t0"] = t
    elif ph == "back":      # stuck propped on the lower arm: roll back onto the side to free it, tuck it, then push again
        d = mem["d"]; tw = -d
        up, dn = ("left", "right") if d > 0 else ("right", "left")
        arm_pose(tgt, dn, ARMS[ROLL_P["arm_dn"]]); arm_pose(tgt, up, ARMS[ROLL_P["arm_up"]])
        tgt[ROLL_J[f"{up}_hip_y"]] = 0.3; tgt[ROLL_J[f"{up}_hip_x"]] = -0.4; tgt[ROLL_J[f"{up}_knee"]] = -0.1
        tgt[ROLL_J[f"{dn}_hip_y"]] = ROLL_P["back_dn_hip"]; tgt[ROLL_J[f"{dn}_knee"]] = ROLL_P["back_dn_knee"]
        tgt[ROLL_J["abdomen_z"]] = -ROLL_P["twist"] * tw
        if tp > ROLL_P["t_back"] or abs(rho) > np.radians(100):
            mem["phase"] = "push"; mem["t0"] = t
    elif ph == "settle":
        if abs(rho) > np.radians(40) and tp > 0.6:      # fell back: go again
            mem["phase"] = "init"; mem["t0"] = t
    mem["rho"] = rho
    return pd(obs, tgt)


def roll_done(obs, mem):
    if not mem or mem.get("phase") != "settle":
        return False
    rho, R = roll_angle(obs)
    return bool(R[2, 0] < -0.85 and np.max(np.abs(obs["joint_rates"])) < 1.0 and np.linalg.norm(obs["root_angular_velocity"]) < 0.6
                and float(obs["time"]) - mem["t0"] > 0.2)

# ---------------------------------------------------------------- phase 2: prone -> heel-sit

KNEEL_NAMES = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
         "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
         "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")
KNEEL_IDX = {n: i for i, n in enumerate(KNEEL_NAMES)}
KNEEL_KP = np.array([120, 60, 60, 60, 40, 200, 150, 60, 40, 200, 150, 10, 10, 8, 10, 10, 8], float)
KNEEL_KD = np.array([8, 5, 5, 5, 4, 10, 6, 5, 4, 10, 6, 0.5, 0.5, 0.4, 0.5, 0.5, 0.4], float)


def pose(ab_y=0.0, hip_y=0.0, knee=-0.05, hip_x=0.0, hip_z=0.0, sh1=-0.7, sh2=-1.4, elb=-1.5):
    """Left/right symmetric pose (arms default: out to the side and back, clear of the thighs)."""
    q = np.zeros(17)
    q[KNEEL_IDX["abdomen_y"]] = ab_y
    for s, sg in (("right", 1.0), ("left", -1.0)):
        q[KNEEL_IDX[f"{s}_hip_y"]] = hip_y
        q[KNEEL_IDX[f"{s}_knee"]] = knee
        q[KNEEL_IDX[f"{s}_hip_x"]] = hip_x
        q[KNEEL_IDX[f"{s}_hip_z"]] = hip_z
        q[KNEEL_IDX[f"{s}_shoulder1"]] = sg * sh1
        q[KNEEL_IDX[f"{s}_shoulder2"]] = sg * sh2
        q[KNEEL_IDX[f"{s}_elbow"]] = elb
    return q


KNEEL_P = dict(hz=-0.5)
# (time reached, pose kwargs, knee gain scale while moving toward it)
KEYSPEC = [
    (0.4, dict(ab_y=-0.3, hip_y=-0.3, knee=-2.6), 1.0),       # heels to the butt (shins in the air)
    (1.2, dict(ab_y=-1.25, hip_y=-2.6, knee=-2.6), 1.0),      # fold: hip target beyond the limit = saturated flexion
    (2.0, dict(ab_y=-1.25, hip_y=-2.6, knee=-1.7), 1.0),      # extend the knees: feet down, tripod head + feet
    (2.4, dict(ab_y=-1.25, hip_y=-2.6, knee=-1.7), 1.0),
    (3.4, dict(ab_y=-1.25, hip_y=-2.6, knee=-2.6), 1.0),      # sink the pelvis onto the heels, still folded
    (4.4, dict(ab_y=-0.4, hip_y=-1.5, knee=-2.6), 1.0),       # raise the trunk
]


def keys():
    return [(t, pose(hip_z=KNEEL_P["hz"], **kw), sc) for t, kw, sc in KEYSPEC]


def target_at(t, q0):
    tp, qp = 0.0, q0
    ks = keys()
    for tk, qk, sc in ks:
        if t < tk:
            a = (t - tp) / (tk - tp)
            a = a * a * (3 - 2 * a)
            return (1 - a) * qp + a * qk, sc
        tp, qp = tk, qk
    return ks[-1][1], 1.0


def kneel_act(obs, mem):
    if "t0" not in mem:
        mem["t0"] = float(obs["time"])
        mem["q0"] = np.array(obs["joint_angles"], float)
    t = float(obs["time"]) - mem["t0"]
    q = np.asarray(obs["joint_angles"], float)
    qd = np.asarray(obs["joint_rates"], float)
    tgt, sc = target_at(t, mem["q0"])
    kp = KNEEL_KP.copy()
    kp[[KNEEL_IDX["right_knee"], KNEEL_IDX["left_knee"]]] *= sc
    tau = kp * (tgt - q) - KNEEL_KD * qd
    return np.clip(tau / np.asarray(obs["gear"], float), -0.4, 0.4)


def kneel_done(obs, mem):
    """True only in a real heel-sit: both feet on the floor, knees folded, trunk raised, nearly at rest."""
    j = obs["joints"]
    return bool("t0" in mem and obs["foot_contact"] and obs["foot_left_contact"] and j["right_knee"] < -2.3 and j["left_knee"] < -2.3
                and obs["up_vector"][2] > 0.3 and float(np.max(np.abs(obs["joint_rates"]))) < 1.0)

# ---------------------------------------------------------------- phases 3-4 and the state machine
PARAMS = dict(lead=0.06,     # path lead of the reference over the measured knee phase (drives the knee extension)
              rate=0.4,      # largest path speed [1/s] (the whole path is 1; faster = fewer late arrivals)
              T_t=2.5,       # s, ramp of the kneeling joints to the first squat equilibrium (slower = fewer falls at the engagement)
              settle=0.3,    # s, longest wait after the ramp for the trunk to converge
              ki_tip=0.0,    # 1/s, integral gain from trunk pitch error to the hip target while kneeling
              pitch_margin=0.05,  # rad, trunk kept this far forward of E0 while kneeling (knees stay loaded)
              Tw=1.5,        # s, fade-in of the LQR's initial state error
              t_sink=4.2,    # s, earliest kneel time at which the heel-sit is accepted (trunk raised)
              th_off=0.0,    # rad, trunk pitch of the first equilibrium above the knee-touching one (knee clearance)
              creep=0.08,    # 1/s, growth of the lead while the knee phase stalls (integral action on the rise)
              lead_max=0.12, # largest lead
              s_fade=2.0,    # path coordinate by which the extra lead has faded out
              A=0.3,         # rad, abdomen flexion of the reference poses
              qj_ab=3.0,     # LQR weight on the abdomen angle error
              ff=0.0,        # 1: feed the reference's joint velocities forward
              W0=-0.2,       # rad, hip abduction (hip_x) of the squat poses (as in the heel-sit)
              Z0=-0.43,      # rad, hip rotation (hip_z) of the squat poses (knees apart, as in the heel-sit)
              qj_sag=3.0,    # LQR weight on hip-pitch angle errors (posture stiffness)
              qj_knee=100.0, # LQR weight on knee angle errors (how hard the lead pulls the knees straight)
              qvel=0.0,      # LQR weight on all velocities (damping)
              qbal=1000.0,   # LQR weight on the horizontal offset of the mass centre from the feet
              drop_yaw=0.0,  # 1: leave the heading out of the LQR state along the path (it is a neutral mode)
              q_top=30.0,    # joint-angle weight of the standing LQR used at the end of the path
              disc=0.99,     # LQR discount per step (needed for the neutral heading mode; close to 1 or slow posture modes drift)
              r=10.0)        # LQR control weight (larger = softer gains)


def build_path():
    return SquatPath(A=PARAMS["A"], th_off=PARAMS["th_off"], W=PARAMS["W0"], Zr=PARAMS["Z0"], arms=(-0.3, -0.7, -1.33), r=PARAMS["r"], qj_ab=PARAMS["qj_ab"],
                     qj_sag=PARAMS["qj_sag"], qvel=PARAMS["qvel"], qbal=PARAMS["qbal"], qj_knee=PARAMS["qj_knee"], disc=PARAMS["disc"], drop_yaw=PARAMS["drop_yaw"] > 0, q_top=PARAMS["q_top"])


PATH = build_path()
QJ0 = np.array([PATH.Q[0][QADR[n]] for n in ACTUATED_JOINTS])     # joint angles of the first equilibrium, motor order
MEM = {}


def controller(obs: dict) -> np.ndarray:
    m = MEM
    if obs["step"] == 0 or not m:
        m.clear(); m.update(phase="roll", roll={}, kneel={}, s=0.0, extra=0.0, hip_i=0.0)
    t = float(obs["time"])
    if m["phase"] == "roll":
        if roll_done(obs, m["roll"]):
            m["phase"] = "kneel"; m["t_kneel"] = t
        else:
            return roll_act(obs, m["roll"])
    if m["phase"] == "kneel":
        a = kneel_act(obs, m["kneel"])
        if (kneel_done(obs, m["kneel"]) and t - m["t_kneel"] >= PARAMS["t_sink"]) or t - m["t_kneel"] >= PARAMS["t_sink"] + 1.5:
            m["phase"] = "tip"; m["t_tip"] = t; m["q_tip"] = np.array(obs["joint_angles"], float)
        else:
            return a
    if m["phase"] == "tip":
        x = float(np.clip((t - m["t_tip"]) / PARAMS["T_t"], 0.0, 1.0)); x = x * x * (3 - 2 * x)
        tgt = (1 - x) * m["q_tip"] + x * QJ0
        qg, v = canon(np.asarray(obs["gym_obs"], float)); qg[0:2] = PATH.Q[0][0:2]
        dq = np.zeros(NV); mujoco.mj_differentiatePos(M, dq, 1.0, PATH.Q[0], qg)
        perr = dq[4] - PARAMS["pitch_margin"]                 # trunk pitch relative to E0 (positive = too far forward)
        if x > 0.5:                                            # integral action: the hips bring the trunk to E0's attitude
            m["hip_i"] = float(np.clip(m["hip_i"] + PARAMS["ki_tip"] * perr * obs["dt"], -0.8, 0.8))
        tgt[[5, 9]] += m["hip_i"]
        rate = float(np.abs(obs["joint_rates"]).max())
        if t - m["t_tip"] >= PARAMS["T_t"] + PARAMS["settle"] or (x >= 1.0 and abs(perr) < 0.04 and rate < 0.4):
            m["phase"] = "lqr"; m["t_lqr"] = t; m["e0"] = None
        else:
            ff = x * PATH.U[0]; ff[[6, 10]] = 0.0          # no knee feed-forward while the knees carry the load
            kp = KNEEL_KP.copy(); kp[0] *= 2.0
            return np.clip(ff + (kp * (tgt - obs["joint_angles"]) - KNEEL_KD * obs["joint_rates"]) / obs["gear"], -0.4, 0.4)
    # RISE: gain-scheduled LQR about the squat path
    w = float(np.clip((t - m["t_lqr"]) / PARAMS["Tw"], 0.0, 1.0))
    knee = -0.5 * (obs["joints"]["right_knee"] + obs["joints"]["left_knee"])
    if w >= 1.0:
        fade = float(np.clip((PARAMS["s_fade"] - m["s"]) / 0.2, 0.0, 1.0))      # the extra lead is for the deep squat only
        s_new = min(PATH.s_of_knee(knee) + PARAMS["lead"] + fade * m["extra"], m["s"] + PARAMS["rate"] * obs["dt"])
        if s_new <= m["s"] + 1e-9:      # stalled: let the lead grow
            m["extra"] = min(m["extra"] + PARAMS["creep"] * obs["dt"], PARAMS["lead_max"] - PARAMS["lead"])
        m["s"] = min(max(m["s"], s_new), 1.0)
    s_prev = m.get("s_prev", m["s"]); m["s_prev"] = m["s"]
    q, u, K = PATH.ref(m["s"])
    qg, v = canon(np.asarray(obs["gym_obs"], float)); qg[0:2] = q[0:2]
    dq = np.zeros(NV); mujoco.mj_differentiatePos(M, dq, 1.0, q, qg)
    if PARAMS["ff"] > 0 and m["s"] > s_prev:                      # reference velocity along the path
        vref = np.zeros(NV); mujoco.mj_differentiatePos(M, vref, obs["dt"], PATH.ref(s_prev)[0], q)
        v = v - PARAMS["ff"] * vref
    e = np.concatenate([dq, v])
    if m["e0"] is None:
        m["e0"] = e.copy(); m["e0"][NV:] = 0.0
    return np.clip(u - K @ (e - (1 - w) * m["e0"]), -0.4, 0.4)
