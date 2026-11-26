"""Raibert-style foot-placement walker for the proprioceptive Humanoid.

State estimation: a private MuJoCo kinematic model + IMU attitude (yaw integrated from the gyro) gives COM and
foot positions; COM velocity comes from the stance foot not moving.  Stance leg: hip torque holds the torso
attitude, knee holds leg length.  Swing leg: IK to a foot-placement target (Raibert rule), joint PD.
"""
import numpy as np, mujoco
from hl_control.envs.humanoid import XML_PATH, ACTUATED_JOINTS

GEAR = np.array([100,100,100,100,100,300,200,100,100,300,200,25,25,25,25,25,25],float)
LEG = [(3,4,5,6),(7,8,9,10)]   # hip_x, hip_z, hip_y, knee motor indices (right, left)
R_FOOT = 0.075

SPEC = dict(
    Tmin=(0.25,0.1,0.6), Tmax=(0.5,0.25,1.0), Ts=(0.3,0.15,0.6), Hc=(0.12,0.03,0.25),
    c0=(0.0,-0.2,0.3), cv=(0.25,0.0,0.6), cd=(0.1,-0.3,0.6), vdes=(0.7,0.2,1.2),
    w0=(0.12,0.05,0.3), cvy=(0.2,-0.2,0.8),
    Ld=(0.74,0.6,0.75), kh=(0.0,-1.0,1.0), pdes=(0.05,-0.2,0.3), Kpp=(150,0,400), Kdp=(15,0,60),
    Krr=(100,-300,300), Kdr=(10,-40,40), kk=(500,50,1000), kdk=(25,1,60),
    ksw=(250,50,600), kdsw=(12,1,40), ab0=(0.1,-0.4,0.4), kab=(100,0,300),
    vfilt=(0.5,0.05,1.0), T0=(0.4,0.1,1.0), kyaw=(0.0,-1.0,1.0),
)
PARAMS = {k: v[0] for k, v in SPEC.items()}
PARAMS.update({"Tmin": 0.10297, "Tmax": 1.0, "Ts": 0.15, "Hc": 0.04956, "c0": 0.00753, "cv": 0.02201, "cd": 0.0979, "vdes": 0.32624, "w0": 0.14602, "cvy": 0.16237, "Ld": 0.64555, "kh": 0.42471, "pdes": 0.21723, "Kpp": 344.20371, "Kdp": 16.19561, "Krr": 184.03917, "Kdr": 1.29099, "kk": 207.7553, "kdk": 18.90146, "ksw": 465.61704, "kdsw": 1.0, "ab0": 0.04925, "kab": 24.7375, "vfilt": 0.56734, "T0": 0.48786, "kyaw": 0.38489})

class _K:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(XML_PATH)); self.d = mujoco.MjData(self.m)
        m = self.m
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in ACTUATED_JOINTS])
        self.vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in ACTUATED_JOINTS])
        self.feet = [m.body("right_foot").id, m.body("left_foot").id]
        self.fg = [m.geom("right_foot").id, m.geom("left_foot").id]
        self.thigh = [m.body("right_thigh").id, m.body("left_thigh").id]
        self.mass = m.body_mass; self.M = self.mass.sum()
        self.quat = np.zeros(4)
    def set(self, q, roll, pitch, yaw):
        d = self.d; d.qpos[:] = 0
        mujoco.mju_euler2Quat(self.quat, np.array([yaw, pitch, roll]), "zyx")
        d.qpos[3:7] = self.quat; d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, d); mujoco.mj_comPos(self.m, d)
    def com(self): return (self.mass[:, None]*self.d.xipos).sum(0)/self.M
    def foot(self, i): return self.d.geom_xpos[self.fg[i]].copy()
    def hip(self, i): return self.d.xpos[self.thigh[i]].copy()
    def jac(self, i):
        jp = np.zeros((3, self.m.nv)); mujoco.mj_jac(self.m, self.d, jp, None, self.foot(i), self.feet[i]); return jp

_K0 = _K()
def _leglen_table():
    K = _K0; ks = np.linspace(-2.8, -0.03, 200); L = []
    for k in ks:
        q = np.zeros(17); q[6] = k; K.set(q, 0, 0, 0); L.append(np.linalg.norm(K.foot(0)-K.hip(0)))
    return ks, np.array(L)
_KS, _LS = _leglen_table()
def knee_for_len(L):
    return float(np.interp(L, _LS, _KS))

_S = {}
def _rz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def smooth(s): s = min(max(s,0.0),1.0); return s*s*(3-2*s)

def _ik(K, leg, q, roll, pitch, yaw, target_world, iters=4):
    idx = LEG[leg]; q = q.copy()
    cols = [K.vadr[idx[0]], K.vadr[idx[2]], K.vadr[idx[3]]]
    for _ in range(iters):
        K.set(q, roll, pitch, yaw)
        err = target_world - K.foot(leg)
        J = K.jac(leg)[:, cols]
        dq = np.linalg.solve(J.T@J + 1e-3*np.eye(3), J.T@err)
        q[idx[0]] += dq[0]; q[idx[2]] += dq[1]; q[idx[3]] += dq[2]
        q[idx[3]] = min(q[idx[3]], -0.035)
    return q

def controller(obs):
    P = PARAMS; K = _K0
    dt = obs["dt"]
    if obs["step"] == 0 or not _S:
        _S.clear(); _S.update(yaw=0.0, stance=0, tstate=0.0, v=np.zeros(3), prev=None, lift=None, ref=np.zeros(17))
    S = _S
    q = obs["joint_angles"]; qd = obs["joint_rates"]; roll = obs["roll"]; pitch = obs["pitch"]; gy = obs["gyro_body"]
    S["yaw"] += (np.sin(roll)/np.cos(pitch)*gy[1] + np.cos(roll)/np.cos(pitch)*gy[2])*dt
    yaw = S["yaw"]
    K.set(q, roll, pitch, yaw)
    com = K.com(); feet = [K.foot(0), K.foot(1)]
    cont = [obs["foot_contact"], obs["foot_left_contact"]]
    R = _rz(yaw); Rt = R.T
    pr = [feet[i]-com for i in range(2)]
    # COM velocity from stationary contact feet
    vs = []
    if S["prev"] is not None:
        for i in range(2):
            if cont[i] and S["prev"][1][i]:
                vs.append(-(pr[i]-S["prev"][0][i])/dt)
    if vs:
        vm = np.mean(vs, axis=0); S["v"] = (1-P["vfilt"])*S["v"] + P["vfilt"]*vm
    else:
        S["v"] = S["v"] + np.array([0,0,-9.81*dt])
    S["prev"] = (pr, list(cont))
    vh = Rt@S["v"]   # heading-frame velocity
    if obs["time"] < P["T0"]:
        tau = np.zeros(17)
        for leg in (0, 1):
            idx = LEG[leg]; rs = 1.0 if leg == 0 else -1.0
            tau[idx[3]] = P["kk"]*(knee_for_len(P["Ld"])-q[idx[3]]) - P["kdk"]*qd[idx[3]]
            tau[idx[2]] = P["Kpp"]*(pitch-P["pdes"]) + P["Kdp"]*gy[1]
            tau[idx[0]] = rs*(P["Krr"]*roll + P["Kdr"]*gy[0])
            tau[idx[1]] = 30*(0-q[idx[1]]) - 3*qd[idx[1]]
        tau[0] = P["kab"]*(P["ab0"]-q[0]) - 8*qd[0]
        tau[1] = 50*(0-q[1]) - 5*qd[1]
        tau[2] = 50*(0-q[2]) - 5*qd[2]
        tau[11:] = 5*(0-q[11:]) - 0.5*qd[11:]
        S["tstate"] = 0.0
        return np.clip(tau/GEAR, -0.4, 0.4)
    # state machine
    S["tstate"] += dt
    st = S["stance"]; sw = 1-st
    if S["tstate"] >= P["Tmax"] or (S["tstate"] >= P["Tmin"] and cont[sw]):
        S["stance"] = sw; S["tstate"] = 0.0; st, sw = sw, st; S["lift"] = None
    ts = S["tstate"]
    if S["lift"] is None:
        S["lift"] = Rt@pr[sw]   # swing foot rel. COM (heading frame) at lift-off
    # ground level (K frame z) from the stance foot
    zg = feet[st][2] - R_FOOT
    tau = np.zeros(17)
    # ---- swing leg
    sgn = 1.0 if sw == 1 else -1.0
    xd = P["c0"] + P["cv"]*vh[0] + P["cd"]*(vh[0]-P["vdes"])
    yd = sgn*P["w0"] + P["cvy"]*vh[1]
    s = smooth(ts/P["Ts"])
    lift = S["lift"]
    xr = lift[0] + (xd-lift[0])*s
    yr = lift[1] + (yd-lift[1])*s
    sp = min(ts/P["Ts"], 1.0)
    zc = P["Hc"]*np.sin(np.pi*sp) if sp < 1.0 else 0.0
    zt = (zg + R_FOOT + zc - 0.01*(sp >= 1.0)) - com[2]
    target = com + R@np.array([xr, yr, zt])
    qik = _ik(K, sw, q, roll, pitch, yaw, target)
    idx = LEG[sw]
    for j in (idx[0], idx[2], idx[3]):
        tau[j] = P["ksw"]*(qik[j]-q[j]) - P["kdsw"]*qd[j]
    tau[idx[1]] = 30*(0-q[idx[1]]) - 3*qd[idx[1]] - P["kyaw"]*0
    # ---- stance leg
    idx = LEG[st]
    h = com[2] - zg
    Ld = P["Ld"] + P["kh"]*(0.95-h)*0
    kt = knee_for_len(Ld)
    tau[idx[3]] = P["kk"]*(kt-q[idx[3]]) - P["kdk"]*qd[idx[3]]
    tau[idx[2]] = P["Kpp"]*(pitch-P["pdes"]) + P["Kdp"]*gy[1]
    rs = 1.0 if st == 0 else -1.0
    tau[idx[0]] = rs*(P["Krr"]*roll + P["Kdr"]*gy[0])
    tau[idx[1]] = 30*(0-q[idx[1]]) - 3*qd[idx[1]]
    # ---- trunk and arms
    tau[0] = P["kab"]*(P["ab0"]-q[0]) - 8*qd[0]
    tau[1] = 50*(0-q[1]) - 5*qd[1]
    tau[2] = 50*(0-q[2]) - 5*qd[2]
    tau[11:] = 5*(0-q[11:]) - 0.5*qd[11:]
    # joint-range soft limits are handled by the plant
    return np.clip(tau/GEAR, -0.4, 0.4)
