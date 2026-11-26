"""Adroit relocate: scripted reach / wrap-grasp / carry controller at joint level (no learning).

Control law, every 10 ms:

1. Model copy (kinematics and inertia only, never stepped).  A private copy of the plant's MuJoCo model
   gives, for the measured joint angles, the absolute palm point (site S_grasp), the arm's 6x6 mass matrix
   and the gravity load of every joint.  With the observed differences this yields the absolute ball and
   target positions; both are averaged over time (the target never moves, the ball only when it is handled),
   which removes measurement noise.

2. Arm observer.  The six arm joints are tracked by a model-based alpha-beta filter: the prediction is a
   semi-implicit Euler step of  M qdd = g u - k q - c qd - gravity  with the set points actually sent, the
   correction blends in the measured angles (`obs_alpha` on position, `obs_beta` on velocity).  It supplies
   the joint velocity the plant does not report, without differentiating noisy angles.

3. State machine producing a Cartesian set point for the palm point:
   REACH  hand open; fly over the ball and come down on it along a cone (palm kept `cone` x horizontal error
          above the grasp height, never above the start height) to the offset `off_*` from the ball centre.
          The grasp is triggered when the measured palm-ball offset is within `tol` of that, or when the set
          point has rested for `n_stall` steps.
   CLOSE  hold the palm and close fingers and thumb around the ball for `n_close` steps.  If the ball lies
          beyond the arm's reach by more than `rake_min`, the closing only lasts `n_rake` steps: the fingers
          rake the ball toward the palm, the hand reopens and REACH starts again.
   CARRY  palm set point = target + (palm - ball), i.e. the ball is servoed onto the target, fingers closed.
          If the ball drifts more than `lost` from where it was grasped, the hand reopens and grasps again.
   The set point moves at a limited speed (`v_max`, `v_carry`) and brakes with `a_max` ahead of its goal; a
   slow integrator (`ki`) on the palm position error, active once the set point rests, removes servo droop.

4. Inverse kinematics, closed form on the model copy: the two horizontal slides take x and y, the vertical
   slide takes z, and the pitch joint ARRx takes over where the vertical slide runs out of range (the hand
   must pitch down to reach the table); the wrist counter-rotates by `wrist_level` x ARRx.

5. Joint servos.  Arm, computed torque:
       qdd_des = w^2 (q_des - q) + 2 zeta w (qd_des - qd),      ctrl = (M qdd_des + c qd + k q + gravity) / g
   `w_arm` is the closed-loop natural frequency and `zeta` the damping ratio; the plant's own spring (k = 200),
   damping (c = 20) and gravity are cancelled (g = 500).  The full mass matrix matters: slides and rotations
   are strongly coupled (a per-joint PD chatters), and when a set point hits its range (the vertical slide
   cannot push down) the remaining joints are re-solved for their wanted accelerations.
   Wrist: position set point plus gravity compensation.  Fingers and thumb: open-loop set points (zero when
   open; `fj*`, `th*`, `lf4`, `spread` when closed); their servos are plain springs, so the flexion commanded
   beyond the ball's surface is the grip force.

The grasp numbers (`off_*`, `fj*`, `th*`, `lf4`, `spread`, `wrist_level`) come from a random search in
simulation over nominal, slippery and heavy-ball plants; the rest was set by hand.
"""

import numpy as np

from hl_control.envs import adroit as _adroit

mujoco = _adroit.mujoco

GAINS = {
    "w_arm": 20.0, "zeta": 1.0,                  # arm servo: closed-loop natural frequency [rad/s], damping ratio
    "obs_alpha": 0.5, "obs_beta": 0.15,          # arm observer: position / velocity correction gains
    "v_max": 1.5,                                # palm set-point speed limit while reaching [m/s]
    "v_carry": 1.0,                              # ... while carrying [m/s]
    "a_max": 8.0,                                # set-point deceleration when closing in on its goal [m/s^2]
    "ki": 6.0,                                   # integral gain on the palm position error [1/s]
    "n_stall": 25,                               # grasp anyway after the set point has rested this many steps
    "off_x": 0.0132, "off_y": -0.0289, "off_z": 0.0123,   # palm point minus ball centre at grasp [m]
    "cone": 0.7,                                 # approach cone: extra height per metre of horizontal error
    "tol": 0.012,                                # palm error that triggers the grasp [m]
    "n_close": 15,                               # steps spent closing before the carry starts
    "wrist_level": 0.8082,                       # WRJ0 = -wrist_level * ARRx
    "fj2": 1.4609, "fj1": 1.5078, "fj0": 0.572,  # closed finger set points (knuckle, middle, tip) [rad]
    "th4": 0.762, "th3": 0.3667, "th2": -0.1254, "th1": -0.1619, "th0": -0.8944,  # closed thumb set points
    "spread": -0.2357,                           # finger side-to-side fan when closed [rad]
    "lf4": 0.5252,                               # little-finger metacarpal (palm arch) [rad]
    "lost": 0.06,                                # ball-to-palm drift that counts as a lost grasp [m]
    "rake_min": 0.012,                           # reach shortfall above which the fingers first rake the ball in [m]
    "n_rake": 6,                                 # steps of a raking close
}

_G_ARM, _K_ARM, _C_ARM = 500.0, 200.0, 20.0      # arm servo gain, stiffness, joint damping
_G_WRIST = 10.0                                  # wrist servo gain (= its stiffness)
_MARGIN = 0.009                                  # joint limits push back inside this margin
_REACH, _CLOSE, _CARRY = 0, 1, 2


def _load_model():
    """The plant's MuJoCo model (kinematics, inertia, gravity), from the package's own environment."""
    try:
        return _adroit.AdroitEnv(_adroit.AdroitConfig(task="relocate")).model
    except Exception:   # same model straight from its XML file
        import os
        from gymnasium_robotics.envs.adroit_hand import adroit_relocate as rel
        xml = os.path.join(os.path.dirname(os.path.realpath(rel.__file__)), "..", "assets", "adroit_hand", "adroit_relocate.xml")
        return mujoco.MjModel.from_xml_path(xml)


class _Controller:
    def __init__(self):
        self.m = _load_model()
        self.d = mujoco.MjData(self.m)
        self.sid = self.m.site("S_grasp").id
        self.lo = self.m.jnt_range[:30, 0].copy()
        self.hi = self.m.jnt_range[:30, 1].copy()
        self.ulo = self.m.actuator_ctrlrange[:, 0].copy()
        self.uhi = self.m.actuator_ctrlrange[:, 1].copy()
        self.reset()

    def reset(self):
        self.phase, self.t_phase = _REACH, 0
        self.x = None                  # observer: arm joint positions
        self.v = np.zeros(6)           # observer: arm joint velocities
        self.wr = None                 # filtered wrist angles
        self.u_prev = np.zeros(30)
        self.M = None
        self.grav = np.zeros(30)
        self.p_cmd = None              # rate-limited palm set point (world frame)
        self.z_top = 0.0               # palm height at the start: ceiling of the approach
        self.bias = np.zeros(3)        # integral correction of the palm set point
        self.n_rest = 0                # steps the set point has been at rest
        self.qa_prev = None
        self.p_hold = None
        self.ball = None
        self.n_ball = 0
        self.target = None
        self.n_target = 0
        self.rel = None                # filtered palm - ball
        self.rel_grasp = None
        self.still = 0
        self.rake = False              # the current closing only pulls an out-of-reach ball closer
        self.n_rake = 0
        self.t_retreat = 0             # steps left of the open-hand back-off before a new approach

    def reopen(self):
        """Back to REACH with a fresh ball estimate; the hand first backs off upward while it opens."""
        self.phase, self.t_phase, self.n_rest = _REACH, 0, 0
        self.ball, self.n_ball, self.still = None, 0, 0
        self.t_retreat = 8

    # ---- model copy ----
    def fk(self, q, dynamics=False):
        d = self.d
        d.qpos[:30] = q
        d.qpos[30:33] = (0.0, 0.0, 3.0)    # park the ball out of the way: it plays no role in the copy
        d.qvel[:] = 0.0
        if dynamics:
            mujoco.mj_forward(self.m, d)
        else:
            mujoco.mj_kinematics(self.m, d)
        return d.site_xpos[self.sid].copy()

    def arm_dynamics(self):
        """Arm block of the mass matrix and the gravity load, at the state of the last fk(dynamics=True)."""
        nv = self.m.nv
        M = np.zeros((6, 6))
        e = np.zeros(nv)
        res = np.zeros(nv)
        for i in range(6):
            e[:] = 0.0
            e[i] = 1.0
            mujoco.mj_mulM(self.m, self.d, res, e)
            M[:, i] = res[:6]
        return M, self.d.qfrc_bias[:30].copy()

    def ik(self, p, q_ref, P):
        """Arm + wrist joint set points putting S_grasp at world point p (ARRy = ARRz = WRJ1 = 0), and the point reached."""
        q = np.array(q_ref, dtype=float)
        q[:8] = 0.0

        def palm(theta):
            q[3] = theta
            q[7] = float(np.clip(-P["wrist_level"] * theta, self.lo[7], self.hi[7]))
            return self.fk(q)

        theta = 0.0
        p0 = palm(theta)
        q1 = p[2] - p0[2]
        if q1 < _MARGIN or q1 > 0.2 - _MARGIN:
            q1 = float(np.clip(q1, _MARGIN, 0.2 - _MARGIN))
            zt = p[2] - q1
            for _ in range(6):
                f0 = palm(theta)[2] - zt
                f1 = palm(theta + 1e-3)[2] - zt
                g = (f1 - f0) / 1e-3
                if abs(g) < 1e-6:
                    break
                theta = float(np.clip(theta - f0 / g, -0.75 + _MARGIN, 0.75 - _MARGIN))
            p0 = palm(theta)
        q0 = float(np.clip(-(p[0] - p0[0]), self.lo[0] + _MARGIN, self.hi[0] - _MARGIN))
        q2 = float(np.clip(p[1] - p0[1], self.lo[2] + _MARGIN, self.hi[2] - _MARGIN))
        q[0], q[1], q[2] = q0, q1, q2
        reached = np.array([p0[0] - q0, p0[1] + q2, p0[2] + q1])
        return q[:8].copy(), reached

    # ---- one control step ----
    def __call__(self, obs):
        P = GAINS
        o = np.asarray(obs["observation"], dtype=float)
        if int(obs["step"]) == 0:
            self.reset()
        dt = float(obs["dt"])
        y = o[:30]

        # arm observer (model-based alpha-beta filter)
        if self.x is None:
            self.x = y[:6].copy()
            self.v[:] = 0.0
            self.wr = y[6:8].copy()
        else:
            # semi-implicit Euler step of  M qdd = g u - k q - c qd - gravity  (stable for the stiff roll joint)
            A = self.M + (dt * _C_ARM + dt * dt * _K_ARM) * np.eye(6)
            rhs = self.M @ self.v + dt * (_G_ARM * self.u_prev[:6] - _K_ARM * self.x - self.grav[:6])
            vp = np.linalg.solve(A, rhs)
            xp = self.x + dt * vp
            low, high = xp < self.lo[:6] + _MARGIN, xp > self.hi[:6] - _MARGIN
            xp = np.clip(xp, self.lo[:6] + _MARGIN, self.hi[:6] - _MARGIN)
            vp[low] = np.maximum(vp[low], 0.0)
            vp[high] = np.minimum(vp[high], 0.0)
            r = y[:6] - xp
            self.x = xp + P["obs_alpha"] * r
            self.v = vp + P["obs_beta"] / dt * r
            self.wr += 0.5 * (y[6:8] - self.wr)
        q = y.copy()
        q[:6] = self.x
        q[6:8] = self.wr

        palm = self.fk(q, dynamics=True)
        self.M, self.grav = self.arm_dynamics()

        # target: static, running mean.  ball: running mean while it lies on the table.
        self.n_target += 1
        t_meas = palm - o[33:36]
        self.target = t_meas if self.target is None else self.target + (t_meas - self.target) / min(self.n_target, 60)
        rel_meas = o[30:33]
        self.rel = rel_meas.copy() if self.rel is None else self.rel + 0.3 * (rel_meas - self.rel)
        b_meas = palm - rel_meas
        if self.phase == _REACH:
            self.n_ball += 1
            if self.ball is None:
                self.ball = b_meas.copy()
            else:
                jump = np.linalg.norm(b_meas - self.ball)
                self.still = self.still + 1 if jump < 0.03 else 0
                gain = max(1.0 / self.n_ball, 0.15)
                if jump > 0.04:           # the ball was moved: restart the average
                    gain, self.n_ball = 1.0, 1
                self.ball = self.ball + gain * (b_meas - self.ball)

        if self.p_cmd is None:
            self.p_cmd = palm.copy()
            self.z_top = float(palm[2])
        off = np.array([P["off_x"], P["off_y"], P["off_z"]])

        closed = False
        if self.phase == _REACH:
            goal = self.ball + off
            _, reach = self.ik(goal, q, P)
            eh = np.linalg.norm(palm[:2] - reach[:2])
            p_goal = reach.copy()
            p_goal[2] += P["cone"] * max(0.0, eh - 0.01)
            p_goal[2] = max(reach[2], min(p_goal[2], self.z_top))   # stay below the start height: no vertical-slide excursion
            v = P["v_max"]
            if self.t_retreat > 0:      # after a failed or raking grasp: lift the open hand off the ball first
                self.t_retreat -= 1
                p_goal = np.array([self.p_cmd[0], self.p_cmd[1], max(self.p_cmd[2], reach[2] + 0.04)])
            else:
                arrived = np.linalg.norm(self.rel - (reach - self.ball)) < P["tol"]   # measured palm-ball offset as wanted
                if (arrived and self.still >= 3) or self.n_rest > P["n_stall"] or self.t_phase > 110:
                    self.phase, self.t_phase = _CLOSE, 0
                    self.p_hold = reach.copy()
                    short = np.linalg.norm(goal[:2] - reach[:2])   # how far the arm falls short of the ball
                    self.rake = short > P["rake_min"] and self.n_rake < 2 and int(obs["step"]) < 120
        if self.phase == _CLOSE:
            closed = True
            p_goal = self.p_hold
            v = P["v_max"]
            if self.rake and self.t_phase >= P["n_rake"]:
                # the fingers have pulled the ball toward the palm: open and grasp it properly
                self.n_rake += 1
                self.reopen()
                closed = False
                p_goal = self.p_cmd
            elif self.t_phase >= P["n_close"]:
                self.phase, self.t_phase = _CARRY, 0
                self.rel_grasp = self.rel.copy()
        if self.phase == _CARRY:
            closed = True
            p_goal = self.target + self.rel
            v = P["v_carry"]
            if np.linalg.norm(self.rel - self.rel_grasp) > P["lost"] and int(obs["step"]) < 150:
                # the ball left the hand: open, back off upward and grasp again
                self.reopen()
                closed = False
                p_goal = self.p_cmd
        self.t_phase += 1

        # rate-limited Cartesian set point, then joint set points
        step = p_goal - self.p_cmd
        n = np.linalg.norm(step)
        v = min(v, np.sqrt(2.0 * P["a_max"] * n))    # brake ahead of the goal: no set-point jerk, no overshoot
        if n > v * dt:
            step *= v * dt / n
        self.p_cmd = self.p_cmd + step
        # slow integral action on the palm position once the set point rests (servo droop, payload)
        self.n_rest = self.n_rest + 1 if n < 2e-3 else 0
        if n < 5e-3:
            self.bias = np.clip(self.bias + P["ki"] * dt * (self.p_cmd - palm), -0.03, 0.03)
        qa, _ = self.ik(self.p_cmd + self.bias, q, P)

        u = np.zeros(30)
        # arm: computed torque.  Desired joint acceleration from a critically damped PD on the set point,
        # mapped through the arm mass matrix; the servo spring, joint damping and gravity are cancelled.
        qa_dot = np.zeros(6) if self.qa_prev is None else (qa[:6] - self.qa_prev) / dt
        self.qa_prev = qa[:6].copy()
        w = P["w_arm"]
        acc = w * w * (qa[:6] - self.x) + 2.0 * P["zeta"] * w * (qa_dot - self.v)
        h = _C_ARM * self.v + _K_ARM * self.x + self.grav[:6]
        ua = (self.M @ acc + h) / _G_ARM
        # Saturation-aware: a joint whose set point hits its range (the vertical slide cannot push down)
        # accelerates as its clipped force dictates; the others are re-solved for their wanted accelerations.
        lo6, hi6 = self.ulo[:6], self.uhi[:6]
        sat = (ua < lo6) | (ua > hi6)
        for _ in range(3):
            if not sat.any() or sat.all():
                break
            free = ~sat
            us = np.clip(ua[sat], lo6[sat], hi6[sat])
            acc_s = np.linalg.solve(self.M[np.ix_(sat, sat)], _G_ARM * us - h[sat] - self.M[np.ix_(sat, free)] @ acc[free])
            ua[free] = (self.M[np.ix_(free, sat)] @ acc_s + self.M[np.ix_(free, free)] @ acc[free] + h[free]) / _G_ARM
            ua[sat] = us
            new = ((ua < lo6) | (ua > hi6)) & free
            if not new.any():
                break
            sat = sat | new
        u[:6] = ua
        u[6:8] = qa[6:8] + self.grav[6:8] / _G_WRIST
        if closed:
            for b, sgn in ((8, -1.0), (12, -0.3), (16, 0.3), (21, 1.0)):
                u[b] = sgn * P["spread"]
                u[b + 1], u[b + 2], u[b + 3] = P["fj2"], P["fj1"], P["fj0"]
            u[20] = P["lf4"]
            u[25], u[26], u[27], u[28], u[29] = P["th4"], P["th3"], P["th2"], P["th1"], P["th0"]
        else:
            u[25] = -0.3   # thumb swung out of the way while reaching
        u = np.clip(u, self.ulo, self.uhi)
        self.u_prev = u
        mean = np.asarray(obs["ctrl_mean"], dtype=float)
        half = np.asarray(obs["ctrl_half_range"], dtype=float)
        return np.clip((u - mean) / half, -1.0, 1.0)


try:   # build the model copy at import, so that every call costs the same
    _ctl = _Controller()
except Exception:   # retried at the first call
    _ctl = None
_last = None


def controller(obs: dict) -> np.ndarray:
    """Position-servo set points in [-1, 1] for the 30 actuators of the relocate task."""
    global _ctl, _last
    n = int(obs["action_dim"])
    try:
        if _ctl is None:
            _ctl = _Controller()
        a = np.asarray(_ctl(obs), dtype=float)
        if a.shape == (n,) and np.all(np.isfinite(a)):
            _last = a
            return a
    except Exception:
        pass
    # never hand back an invalid action: repeat the last good one
    return _last.copy() if _last is not None and _last.shape == (n,) else np.zeros(n)
