"""Adroit door at joint level: model-based scripted controller (servo set points every 10 ms).

Model and observers
-------------------
A private kinematic copy of the plant (the package's own MJCF, loaded once) serves as observer and IK engine:
* the door frame position is inferred from the observed handle point at the start (averaged over the first steps
  while the door is untouched, which also filters observation noise);
* the unobserved arm slide ARTz follows from the observed palm point S_grasp by forward kinematics (the slide axis
  is world +y);
* the hinge axis position follows from the frame (frame + 0.6 m along x).

Control law
-----------
Every step a desired palm pose (S_grasp position, palm orientation) is chosen by the phase logic below, mapped to
targets q_des of (ARTz, ARRx, ARRy, ARRz, WRJ1, WRJ0) by damped-least-squares IK (warm-started, 8 iterations), and
turned into set points
    arm   : ctrl = (k q_des + tau_g(q_des) + kp_arm (q_des - q)) / g      (k = 200, g = 500: q_eq = q_des)
    wrist : ctrl = q_des + tau_g / 10 + kp_wrist (q_des - q)
    finger: ctrl = flexion command (all four fingers J2, J1, J0 together; thumb and abduction at 0)
tau_g is the gravity load from the model copy, so the arm sits at q_des without integral action; kp_arm and kp_wrist
add stiffness against contact loads (they also keep the weak_v1 servos on track).

Phases
------
0 approach : palm moved above the lever (raised first, then forward), then lowered onto it, fingers open.
1 latch    : fingers closed around the lever (hook grasp). The palm target is a rigid grip attached to the latch body,
             evaluated at latch angle (measured + lead_latch), capped at latch_goal: the arm drags the lever down
             along its own arc and the forearm roll turns with it. Ends when latch > latch_switch (bolt clear).
2 pull     : the same grip evaluated at hinge angle (measured + lead_door): the hand pulls the door open along the
             handle's arc (position IK plus a weak roll term). Ends at hinge > phi_release with the door swinging fast.
3 release  : fingers open; the hand backs off the lever (door-local point back_uv), retreats radially to r_out about
             the hinge (beyond the leaf's 0.56 m reach, so the coasting door cannot hit it), then swings behind the
             leaf at that radius. Each step advances when reached (or on its timeout).
4 push     : fist; the back of the hand is driven onto the leaf's back face near the free edge (door-local point
             push_uv) evaluated at hinge angle (measured + lead_push, capped at phi_goal beyond the 1.57 stop), so the
             door is pushed open if it stalled and is pinned against its stop for the rest of the episode.
"""

import os

import numpy as np
import mujoco

PARAMS = {
    # servo feedback
    "kp_arm": 450.0,        # extra P feedback on the arm joints [N m/rad, N/m] on top of the servo's own k = 200
    "kp_wrist": 1.0,        # extra wrist P feedback, relative to the wrist servo stiffness
    # approach and grasp
    "grip_dx": 0.125,       # grip: S_grasp offset from the latch axis along the lever [m]
    "grip_dy": -0.045,      # grip: S_grasp offset behind the lever (toward the hand) [m]
    "grip_dz": -0.025,      # grip: S_grasp offset below the lever axis [m]
    "pre_dz": 0.07,         # pre-grasp height above the grasp pose [m]
    "t_pre": 0.28,          # time at which the pre-grasp pose is reached [s]
    "t_down": 0.38,         # time at which the grasp pose is reached [s]
    "t_close": 0.45,        # latch follower starts (fingers closing since t_down) [s]
    "finger_close": 1.6,    # finger flexion command for the hook grasp [rad]
    # latch and pull
    "lead_latch": 0.5,      # latch follower lead [rad]
    "latch_goal": 1.75,     # cap on the commanded latch angle [rad] (joint range 1.8)
    "latch_switch": 1.35,   # latch angle above which the door is pulled [rad] (bolt clears near 1.2)
    "lead_door": 0.35,      # hinge follower lead while pulling [rad]
    "w_rot2": [0.0, 0.1, 0.0],  # orientation weights (world x, y, z) while pulling: weak roll term only
    "phi_release": 0.5,     # hinge angle at which the lever is released [rad]
    # release and reposition (door-local u: along the leaf from the hinge, v: front normal; polar about the hinge)
    "back_uv": [0.46, 0.27],  # back-off point in front of the lever [m]
    "wp_tol": 0.04,         # back-off point reached tolerance [m]
    "t_back": 0.08,         # back-off timeout [s]
    "r_out": 0.74,          # radius used to pass the free edge (the leaf reaches 0.56 m) [m]
    "r_clear": 0.67,        # hand radius regarded as clear of the leaf [m]
    "a_front": 0.3,         # angle ahead of the leaf while retreating [rad]
    "a_max": 1.5,           # cap on that angle (reachability) [rad]
    "t_out": 0.25,          # retreat timeout [s]
    "a_behind": 0.25,       # angle behind the leaf for the swing-around [rad]
    "a_behind_tol": 0.12,   # swing-around done when the hand is this far behind the leaf [rad]
    "t_behind": 0.25,       # swing-around timeout [s]
    "z_off": -0.12,         # hand height relative to the latch axis in phases 3-4 [m]
    "w_roll3": 0.1,         # roll weight in phases 3-4 (keeps the forearm roll, back of the hand toward +x)
    "finger_open": 0.0,
    "finger_fist": 1.6,
    # push
    "push_uv": [0.50, -0.09],  # S_grasp target behind the leaf (back of the hand on the back face) [m]
    "lead_push": 0.3,       # hinge lead of the push target [rad]
    "phi_goal": 1.65,       # cap on the push target angle (beyond the 1.57 stop: keeps the door pinned) [rad]
}

IK_IDX = np.array([0, 1, 2, 3, 4, 5])          # qpos indices: ARTz, ARRx, ARRy, ARRz, WRJ1, WRJ0
W_PREF = np.array([0.0, 0.0, 0.0, 0.0, 0.02, 0.02])  # weak IK preference of the wrist joints for 0
G_ARM, K_ARM, G_WRIST = 500.0, 200.0, 10.0     # nominal servo gain / stiffness (arm), wrist gain


def _xml_path():
    import gymnasium_robotics
    return os.path.join(os.path.dirname(gymnasium_robotics.__file__), "envs", "assets", "adroit_hand", "adroit_door.xml")


class _Kin:
    """Kinematic / gravity model copy of the plant."""

    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(_xml_path())
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.palm = m.body("palm").id
        self.grasp = m.site("S_grasp").id
        self.handle = m.site("S_handle").id
        self.latch = m.body("latch").id
        self.frame = m.body("frame").id
        self.jac_p = np.zeros((3, m.nv))
        self.jac_r = np.zeros((3, m.nv))
        self.lo = m.jnt_range[:, 0].copy()
        self.hi = m.jnt_range[:, 1].copy()

    def set_q(self, q27, artz, hinge, latch):
        d = self.d
        d.qpos[0] = artz
        d.qpos[1:28] = q27
        d.qpos[28] = hinge
        d.qpos[29] = latch
        mujoco.mj_kinematics(self.m, d)

    def latch_pose(self, hinge, latch):
        self.d.qpos[28] = hinge
        self.d.qpos[29] = latch
        mujoco.mj_kinematics(self.m, self.d)
        return self.d.xpos[self.latch].copy(), self.d.xmat[self.latch].reshape(3, 3).copy()

    def ik(self, q, p_des, R_des, w_rot, idx, iters=8, lam=1e-3, q_pref=None, w_pref=None):
        """Adjust q[idx] (q: full qpos) so that S_grasp -> p_des and the palm orientation -> R_des (weights w_rot)."""
        m, d = self.m, self.d
        q = q.copy()
        for _ in range(iters):
            d.qpos[:] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            p = d.site_xpos[self.grasp]
            R = d.xmat[self.palm].reshape(3, 3)
            Re = R_des @ R.T
            er = 0.5 * np.array([Re[2, 1] - Re[1, 2], Re[0, 2] - Re[2, 0], Re[1, 0] - Re[0, 1]])
            mujoco.mj_jacSite(m, d, self.jac_p, self.jac_r, self.grasp)
            J = np.vstack([self.jac_p[:, idx], w_rot[:, None] * self.jac_r[:, idx]])
            e = np.concatenate([p_des - p, w_rot * er])
            A = J.T @ J + lam * np.eye(len(idx))
            b = J.T @ e
            if q_pref is not None:
                A += np.diag(w_pref)
                b += w_pref * (q_pref - q[idx])
            q[idx] = np.clip(q[idx] + np.linalg.solve(A, b), self.lo[idx], self.hi[idx])
        return q

    def grav(self, q):
        """Generalised gravity (bias) forces at configuration q, zero velocity."""
        d = self.d
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_fwdPosition(self.m, d)
        mujoco.mj_fwdVelocity(self.m, d)
        return d.qfrc_bias.copy()


_K = None   # model copy (built once per process)
_S = {}     # per-episode state


def _update_frame():
    f = _K.m.body_pos[_K.frame]
    _S["hinge"] = (f[0] + 0.6, f[1])
    _S["z_push"] = f[2] - 0.025 + PARAMS["z_off"]


def _reset(obs):
    global _K
    if _K is None:
        _K = _Kin()
    K = _K
    o = np.asarray(obs["observation"], dtype=float)
    K.m.body_pos[K.frame] = np.array([-0.25, 0.3, 0.3])
    K.set_q(o[:27], 0.0, 0.0, 0.0)
    K.m.body_pos[K.frame] = K.m.body_pos[K.frame] + (o[32:35] - K.d.site_xpos[K.handle])
    _S.clear()
    _S["frame_sum"] = K.m.body_pos[K.frame].copy()
    _S["frame_n"] = 1
    _S["q_cmd"] = None
    _S["phase"] = 0
    K.set_q(o[:27], 0.0, 0.0, 0.0)
    _S["R0"] = K.d.xmat[K.palm].reshape(3, 3).copy()  # zero-pose palm orientation
    _update_frame()


def _door_point(hx, hy, u, v, phi, z):
    """World point at door-local (u, v) for hinge angle phi."""
    return np.array([hx - u * np.cos(phi) + v * np.sin(phi), hy - u * np.sin(phi) - v * np.cos(phi), z])


def controller(obs: dict):
    P = PARAMS
    if obs["step"] == 0 or _K is None:
        _reset(obs)
    K = _K
    o = np.asarray(obs["observation"], dtype=float)
    t = float(obs["time"])
    half = np.asarray(obs["ctrl_half_range"], dtype=float)
    mean = np.asarray(obs["ctrl_mean"], dtype=float)
    latch, hinge = float(o[27]), float(o[28])

    # refine the door frame estimate while the door is untouched (noise averaging)
    if latch < 0.05 and hinge < 0.02 and t < 0.3:
        K.set_q(o[:27], 0.0, hinge, latch)
        _S["frame_sum"] = _S["frame_sum"] + K.m.body_pos[K.frame] + (o[32:35] - K.d.site_xpos[K.handle])
        _S["frame_n"] += 1
        K.m.body_pos[K.frame] = _S["frame_sum"] / _S["frame_n"]
        _update_frame()

    # ARTz observer (slide along world +y)
    K.set_q(o[:27], 0.0, hinge, latch)
    q_obs = np.zeros(30)
    q_obs[0] = float(o[30] - K.d.site_xpos[K.grasp][1])
    q_obs[1:28] = o[:27]
    q_obs[28], q_obs[29] = hinge, latch

    # grip: S_grasp offset and palm orientation in the latch frame (latch frame == world at hinge = latch = 0)
    g_off = np.array([P["grip_dx"], -0.15 + P["grip_dy"], P["grip_dz"]])
    R_grip = _S["R0"]

    ph = _S["phase"]
    w_rot = np.full(3, 0.3)
    if t < P["t_down"] and ph == 0:
        # phase 0: approach (raise first, then forward), then lower onto the lever
        lp, lR = K.latch_pose(0.0, 0.0)
        p_grasp = lp + lR @ g_off
        pre = np.array([0.0, -0.03, P["pre_dz"]])
        if t < P["t_pre"]:
            p0 = np.array([-0.007, -0.2, 0.15])
            s = 0.5 - 0.5 * np.cos(np.pi * t / P["t_pre"])
            p_des = p0 + s * (p_grasp + pre - p0)
            p_des[2] = max(p_des[2], p0[2] + min(1.0, 2 * t / P["t_pre"]) * (p_grasp[2] + P["pre_dz"] - p0[2]))
        else:
            s = (t - P["t_pre"]) / (P["t_down"] - P["t_pre"])
            p_des = p_grasp + (1 - s) * pre
        R_des = R_grip
        fing = 0.0
    else:
        if ph == 0:
            _S["phase"] = ph = 1
        fing = P["finger_close"]
        if ph == 1 and latch > P["latch_switch"]:
            _S["phase"] = ph = 2
        if ph == 2 and hinge > P["phi_release"]:
            _S["phase"] = ph = 3
            _S["wp"] = 0
            _S["t_wp"] = t
        if ph <= 2:
            if ph == 1:   # phase 1: turn the latch (follower on the latch angle)
                th = min(latch + P["lead_latch"] if t > P["t_close"] else latch, P["latch_goal"])
                phi = 0.0
            else:         # phase 2: pull the door (follower on the hinge angle)
                th = max(latch, P["latch_goal"])
                phi = min(hinge + P["lead_door"], 1.57)
                w_rot = np.array(P["w_rot2"], dtype=float)
            lp, lR = K.latch_pose(phi, th)
            p_des = lp + lR @ g_off
            R_des = lR @ R_grip
        else:
            hx, hy = _S["hinge"]
            zh = _S["z_push"]
            R_des = K.latch_pose(0.0, P["latch_goal"])[1] @ R_grip
            w_rot = np.array([0.0, P["w_roll3"], 0.0])
            if ph == 3:
                # phase 3: release and go around the free edge; hand polar coordinates about the hinge axis
                rel = o[29:31] - np.array([hx, hy])
                r_h = float(np.hypot(rel[0], rel[1]))
                a_h = float(np.arctan2(-rel[1], -rel[0]))
                i = _S["wp"]
                dt_wp = t - _S["t_wp"]
                if i == 0:    # back off the lever, fingers open
                    p_des = _door_point(hx, hy, P["back_uv"][0], P["back_uv"][1], hinge, zh)
                    done = np.linalg.norm(p_des[:2] - o[29:31]) < P["wp_tol"] or dt_wp > P["t_back"]
                    fing = P["finger_open"]
                elif i == 1:  # radially out beyond the free edge, in front of the leaf
                    al = min(hinge + P["a_front"], P["a_max"])
                    p_des = np.array([hx - P["r_out"] * np.cos(al), hy - P["r_out"] * np.sin(al), zh])
                    done = r_h > P["r_clear"] or dt_wp > P["t_out"]
                    fing = P["finger_open"]
                else:         # swing behind the leaf at the outer radius
                    al = hinge - P["a_behind"]
                    p_des = np.array([hx - P["r_out"] * np.cos(al), hy - P["r_out"] * np.sin(al), zh])
                    done = a_h < hinge - P["a_behind_tol"] or dt_wp > P["t_behind"]
                    fing = P["finger_fist"]
                if done:
                    _S["wp"] = i + 1
                    _S["t_wp"] = t
                    if _S["wp"] >= 3:
                        _S["phase"] = ph = 4
            if ph == 4:
                # phase 4: push the back face toward the stop and keep it pinned there
                phi = min(hinge + P["lead_push"], P["phi_goal"])
                p_des = _door_point(hx, hy, P["push_uv"][0], P["push_uv"][1], phi, zh)
                fing = P["finger_fist"]

    # IK (warm start from the previous command) and servo set points
    q_ik0 = q_obs.copy()
    if _S["q_cmd"] is not None:
        q_ik0[:6] = _S["q_cmd"][:6]
    q_ik = K.ik(q_ik0, p_des, R_des, w_rot, IK_IDX, q_pref=np.zeros(6), w_pref=W_PREF)
    _S["q_cmd"] = q_ik.copy()

    tau = K.grav(np.concatenate([q_ik[:28], [hinge, latch]]))
    ctrl = np.zeros(28)
    ctrl[:4] = (K_ARM * q_ik[:4] + tau[:4] + P["kp_arm"] * (q_ik[:4] - q_obs[:4])) / G_ARM
    ctrl[4:6] = q_ik[4:6] + tau[4:6] / G_WRIST + P["kp_wrist"] * (q_ik[4:6] - o[3:5])
    for base in (6, 10, 14, 19):  # FFJ3, MFJ3, RFJ3, LFJ3: abduction 0, then J2, J1, J0 flexed together
        ctrl[base + 1:base + 4] = fing
    # LFJ4 and the thumb (23-27) stay at 0
    a = (ctrl - mean) / half
    a = np.nan_to_num(a, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.clip(a, -1.0, 1.0)
