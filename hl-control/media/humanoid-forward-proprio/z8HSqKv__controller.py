"""Raibert-style walking controller for the MuJoCo Humanoid on proprioceptive sensing.

The feet are spheres (no ankles), so the robot cannot stand statically: it balances by
stepping. Control law, per 15 ms step:
  * Kinematic reconstruction: the MuJoCo model (hl_control's humanoid.xml) is loaded once;
    joint angles + IMU attitude (roll/pitch, yaw := 0) fill qpos, gyro + joint rates fill
    qvel. With the stance foot's contact point assumed fixed on the floor, the root velocity
    is minus that point's velocity relative to the root, and the mass-centre (COM) position
    and velocity follow from mj_comPos and the subtree-COM Jacobian. The velocity estimate is
    low-pass filtered (V_FILT) and only updated while the stance foot is in settled contact.
  * Start: after the drop at reset, a short double-support stand (T_STAND, left early once
    the COM moves faster than V_START) with both legs in stance mode; the first stance foot
    is the one the COM is moving towards laterally.
  * Stance leg: hip_y / hip_x torques hold the upper body: PD on the chest pitch/roll from
    the IMU (KP_PITCH, KD_PITCH, KP_ROLL, KD_ROLL; PITCH_DES is a forward lean that pulls the
    gait forward) plus PD on the pelvis attitude from the kinematic model (KP_PPITCH, ...),
    because the hips act on the light pelvis while the chest hangs on the weak abdomen.
    hip_z is PD-held at zero; the knee is PD-held (KP/KD_KNEE_ST) at the bend that gives the
    leg length L_STANCE, plus a feed-forward extension KNEE_FF carrying the body weight.
  * Swing leg: Raibert foot placement relative to the COM in the heading frame:
    x = vx*T/2 + KX*(vx - V_DES) + KXP*(COM lead over the stance foot),
    y = side*W_NOM + KY*vy. The leg is shortened by H_SWING*sin(pi*phase) mid-swing.
    Analytic hip_x/hip_y/knee inverse kinematics gives joint targets tracked by soft joint
    PD (KP_LEG, KD_LEG, KP/KD_KNEE_SW; soft so the touchdown is compliant).
  * Steps alternate every T seconds, or earlier at swing-foot touchdown after half the period.
  * Abdomen (KP/KD_ABD) and arms (KP/KD_ARM): joint PD to zero.
All rate gains are small on purpose: at 66.7 Hz, derivative gains above ~0.1 ctrl/(rad/s)
on the 100-300 N m hips make the joint loops chatter. PARAMS was tuned by random search on
seeds 0-7 (survival + forward progress).
"""
import os
import numpy as np
import mujoco

try:
    from hl_control.envs.humanoid import XML_PATH as _XML
except Exception:  # pragma: no cover
    import hl_control
    _XML = os.path.join(os.path.dirname(hl_control.__file__), "envs", "assets", "humanoid.xml")

PARAMS = dict(
    T=0.22,          # step period [s]
    T_STAND=0.2786984687595533,     # initial double-support stand [s]
    V_START=0.25,    # leave the stand early once the COM moves faster than this [m/s]
    KXP=0.0,         # foot x offset per m of COM lead over the stance foot (capture-point style)
    V_DES=0.16222717525644423,      # desired forward speed [m/s]
    KX=0.05,         # Raibert speed gain (foot x offset per m/s of speed error) [s]
    KY=0.5824069451069139,         # lateral Raibert gain [s]
    W_NOM=0.13194234803217214,      # nominal lateral foot offset from the COM [m]
    H_SWING=0.06948650495201154,    # swing leg shortening at mid-swing [m]
    L_STANCE=0.6660685702745782,   # nominal stance leg length (hip to foot centre) [m]
    PITCH_DES=0.25,  # desired torso pitch (forward lean) [rad]
    KP_PITCH=0.7568550520919063,    # stance hip_y ctrl per rad of pitch error
    KD_PITCH=0.0593,    # stance hip_y ctrl per rad/s pitch rate
    KP_PPITCH=0.885882489822401,   # stance hip_y ctrl per rad of pelvis pitch error
    KD_PPITCH=0.055768177789924066,  # stance hip_y ctrl per rad/s pelvis pitch rate
    KP_PROLL=0.0,    # stance hip_x ctrl per rad of pelvis roll
    KD_PROLL=0.07563113759136823,   # stance hip_x ctrl per rad/s pelvis roll rate
    KP_ROLL=1.1264124784675444,     # stance hip_x ctrl per rad of roll
    KD_ROLL=0.15,     # stance hip_x ctrl per rad/s roll rate
    KP_LEG=1.0,      # swing-leg joint PD position gain [ctrl/rad]
    KD_LEG=0.0467,     # swing-leg joint PD rate gain [ctrl/(rad/s)]
    KP_KNEE_ST=1.1486,
    KD_KNEE_ST=0.06298782394964753,  # stance knee rate gain  # stance knee PD position gain
    KP_KNEE_SW=0.5,  # swing knee PD position gain
    KD_KNEE_SW=0.019306942451111374, # swing knee rate gain (light shin: keep small)
    KNEE_FF=0.16521639780890846,    # stance knee feed-forward extension
    KP_ABD=3.6743840951043025,      # abdomen PD gain
    KD_ABD=0.035727892242962876,
    KP_ARM=1.0,      # arm PD gain
    KD_ARM=0.05,
    V_FILT=0.26026507222314604,     # low-pass factor on the COM velocity estimate (per step)
)

L1 = 0.383   # hip to knee [m]
L2 = 0.37    # knee to foot centre [m]
IDX = dict(abd_y=0, abd_z=1, abd_x=2, rhx=3, rhz=4, rhy=5, rk=6, lhx=7, lhz=8, lhy=9, lk=10)
LEG = [dict(hx=3, hz=4, hy=5, k=6, side=-1.0, foot="right_foot", thigh="right_thigh"),
       dict(hx=7, hz=8, hy=9, k=10, side=+1.0, foot="left_foot", thigh="left_thigh")]


class _Kin:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(_XML))
        self.d = mujoco.MjData(self.m)
        self.qadr = [int(self.m.jnt_qposadr[self.m.joint(n).id]) for n in self._names()]
        self.vadr = [int(self.m.jnt_dofadr[self.m.joint(n).id]) for n in self._names()]
        self.foot_g = [self.m.geom("right_foot").id, self.m.geom("left_foot").id]
        self.thigh_b = [self.m.body("right_thigh").id, self.m.body("left_thigh").id]
        self.foot_b = [self.m.body("right_foot").id, self.m.body("left_foot").id]
        self.pelvis_b = self.m.body("pelvis").id
        self.jacr = np.zeros((3, self.m.nv))
        self.jac = np.zeros((3, self.m.nv))

    @staticmethod
    def _names():
        return ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
                "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
                "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")

    def update(self, obs, stance):
        m, d = self.m, self.d
        r, p = float(obs["roll"]), float(obs["pitch"])
        cr, sr, cp, sp = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2)
        # yaw = 0, x-y-z intrinsic: q = qx(roll) * qy(pitch)
        d.qpos[:3] = 0.0
        d.qpos[3:7] = [cr * cp, sr * cp, cr * sp, -sr * sp]
        d.qpos[self.qadr] = obs["joint_angles"]
        d.qvel[:] = 0.0
        d.qvel[3:6] = obs["gyro_body"]
        d.qvel[self.vadr] = obs["joint_rates"]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        R = d.xmat[1].reshape(3, 3)
        com = d.subtree_com[1].copy()
        feet = [d.geom_xpos[g].copy() for g in self.foot_g]
        hips = [d.xpos[b].copy() for b in self.thigh_b]
        cpoint = feet[stance] - np.array([0.0, 0.0, 0.075])     # sphere/floor contact point
        mujoco.mj_jac(m, d, self.jac, None, cpoint, self.foot_b[stance])
        v_root = -(self.jac @ d.qvel)
        mujoco.mj_jacSubtreeCom(m, d, self.jac, 1)
        v_com = self.jac @ d.qvel + v_root
        # pelvis attitude (what the hips act on directly; the chest wobbles on the abdomen)
        Rp = d.xmat[self.pelvis_b].reshape(3, 3)
        pelvis_pitch = float(np.arctan2(Rp[0, 2], Rp[2, 2]))
        pelvis_roll = float(np.arctan2(-Rp[1, 2], Rp[2, 2]))
        mujoco.mj_jac(m, d, None, self.jacr, d.xpos[self.pelvis_b], self.pelvis_b)
        w_pelvis = self.jacr @ d.qvel                     # yaw-less world frame
        return R, com, feet, hips, v_com, (pelvis_roll, pelvis_pitch, w_pelvis)


_K = None
_S = {}


def _reset():
    _S.update(stance=1, t_ph=0.0, v=np.zeros(3), walking=False, prev_contact=(True, True))


def _leg_ik(p_body, side):
    """hip_x, hip_y, knee angles putting the foot centre at p_body (pelvis frame, relative to the hip)."""
    px, py, pz = p_body
    pz = min(pz, -0.1)
    hx = np.arctan2(side * py, -pz) * (-1.0) if side < 0 else np.arctan2(-py, -pz)
    # right leg: axis +x, positive angle moves the foot to +y  -> hx = atan2(py, -pz)
    if side < 0:
        hx = np.arctan2(py, -pz)
    L = float(np.clip(np.sqrt(px * px + py * py + pz * pz), 0.3, L1 + L2 - 1e-3))
    alpha = np.arctan2(px, np.sqrt(py * py + pz * pz))
    beta = np.arccos(np.clip((L1 * L1 + L * L - L2 * L2) / (2 * L1 * L), -1, 1))
    hy = -(alpha + beta)
    phi = np.pi - np.arccos(np.clip((L1 * L1 + L2 * L2 - L * L) / (2 * L1 * L2), -1, 1))
    return hx, hy, -phi


def controller(obs: dict) -> np.ndarray:
    global _K
    P = PARAMS
    if _K is None:
        _K = _Kin()
    if obs["step"] == 0 or not _S:
        _reset()
    q = np.asarray(obs["joint_angles"], float)
    qd = np.asarray(obs["joint_rates"], float)
    gyro = np.asarray(obs["gyro_body"], float)
    roll, pitch = float(obs["roll"]), float(obs["pitch"])
    dt = float(obs["dt"])
    t = float(obs["time"])
    contacts = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))
    st = _S["stance"]
    sw = 1 - st

    if not _S["walking"]:
        st = 0 if contacts[0] else 1
    R, com, feet, hips, v_com, (p_roll, p_pitch, p_w) = _K.update(obs, st)
    if contacts[st] and _S["prev_contact"][st]:        # settled stance contact only
        _S["v"] = (1 - P["V_FILT"]) * _S["v"] + P["V_FILT"] * v_com
    v = _S["v"]

    u = np.zeros(17)
    # abdomen and arms: PD to zero
    for i in (0, 1, 2):
        u[i] = -P["KP_ABD"] * q[i] - P["KD_ABD"] * qd[i]
    for i in range(11, 17):
        u[i] = -P["KP_ARM"] * q[i] - P["KD_ARM"] * qd[i]

    # torso attitude terms applied through stance hips
    # attitude terms applied through the stance hips: chest (IMU) terms carry the upper-body mass,
    # pelvis terms (kinematic model) damp the light pelvis the hips act on directly
    u_pitch = (P["KP_PITCH"] * (pitch - P["PITCH_DES"]) + P["KD_PITCH"] * gyro[1]
               + P["KP_PPITCH"] * (p_pitch - P["PITCH_DES"]) + P["KD_PPITCH"] * p_w[1])
    u_roll = P["KP_ROLL"] * roll + P["KD_ROLL"] * gyro[0] + P["KP_PROLL"] * p_roll + P["KD_PROLL"] * p_w[0]

    def stance_leg(leg):
        u[leg["hy"]] = u_pitch
        u[leg["hx"]] = -leg["side"] * u_roll     # right (side -1): +u_roll ; left: -u_roll
        u[leg["hz"]] = -P["KP_LEG"] * q[leg["hz"]] - P["KD_LEG"] * qd[leg["hz"]]
        qk_des = -(np.pi - np.arccos(np.clip((L1 * L1 + L2 * L2 - P["L_STANCE"] ** 2) / (2 * L1 * L2), -1, 1)))
        u[leg["k"]] = P["KP_KNEE_ST"] * (qk_des - q[leg["k"]]) - P["KD_KNEE_ST"] * qd[leg["k"]] + P["KNEE_FF"]

    if not _S["walking"]:
        if contacts[0] or contacts[1]:
            _S["t_ph"] += dt                      # settle timer after touchdown
        moving = float(np.hypot(v[0], v[1])) > P["V_START"] and _S["t_ph"] > 0.05
        if _S["t_ph"] < P["T_STAND"] and t < 1.5 and not moving:
            stance_leg(LEG[0])
            stance_leg(LEG[1])
            _S["prev_contact"] = contacts
            return np.clip(u, -0.4, 0.4)
        _S["walking"] = True
        # first stance: the foot the mass centre is moving towards (left foot has +y)
        _S["stance"] = st = 1 if v[1] > 0 else 0
        sw = 1 - st
        _S["t_ph"] = 0.0

    # step switching
    _S["t_ph"] += dt
    s = _S["t_ph"] / P["T"]
    if s >= 1.0 or (s > 0.5 and contacts[sw]):
        _S["stance"], st, sw = sw, sw, st
        _S["t_ph"] = 0.0
        s = 0.0
    stance_leg(LEG[st])

    # swing leg: Raibert foot placement relative to the COM, in the yaw-less world frame
    leg = LEG[sw]
    x_t = v[0] * P["T"] / 2 + P["KX"] * (v[0] - P["V_DES"]) + P["KXP"] * (com[0] - feet[st][0])
    y_t = leg["side"] * P["W_NOM"] + v[1] * P["T"] / 2 + P["KY"] * v[1]
    L = P["L_STANCE"] - P["H_SWING"] * np.sin(np.pi * np.clip(s, 0, 1))
    target_w = np.array([com[0] + x_t, com[1] + y_t, 0.0])
    hip = hips[sw]
    dxy = target_w[:2] - hip[:2]
    dz = -np.sqrt(max(L * L - float(dxy @ dxy), 0.01))
    p_w = np.array([dxy[0], dxy[1], dz])
    p_b = R.T @ p_w
    hx, hy, kn = _leg_ik(p_b, leg["side"])
    u[leg["hx"]] = P["KP_LEG"] * (hx - q[leg["hx"]]) - P["KD_LEG"] * qd[leg["hx"]]
    u[leg["hy"]] = P["KP_LEG"] * (hy - q[leg["hy"]]) - P["KD_LEG"] * qd[leg["hy"]]
    u[leg["k"]] = P["KP_KNEE_SW"] * (kn - q[leg["k"]]) - P["KD_KNEE_SW"] * qd[leg["k"]]
    u[leg["hz"]] = -P["KP_LEG"] * q[leg["hz"]] - P["KD_LEG"] * qd[leg["hz"]]
    _S["prev_contact"] = contacts
    return np.clip(u, -0.4, 0.4)
