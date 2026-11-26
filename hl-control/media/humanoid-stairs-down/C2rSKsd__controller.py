"""Stepping controller for the MuJoCo Humanoid on a staircase going down (hand-designed, no learning).

Control law (SIMBICON-style: a state machine, joint PD torques and foot-placement feedback)
-----------------------------------------------------------------------------------------
The robot is dropped from 10 cm, so for the first `t_start` seconds both legs just hold a stance.
After that a two-phase state machine runs per step, the stance side alternating:
  phase 0 (the first `t_swing` s): the swing leg is lifted (knee bent) and swung forward;
  phase 1 (until the swing foot's contact switch closes, at most `t_max` s): the swing leg
          straightens and is put down.
Every joint gets a PD torque in N m (divided by the motor gear to give the command in [-0.4, 0.4]):

  * swing hip, pitch and roll: PD on the thigh angle measured in the WORLD frame (torso attitude +
    abdomen + hip).  The target is the gait's nominal angle plus the balance feedback
        pitch:  phi_des = swh - cd_x * d_f - cv_x * (v_f - v_ref) + side * yaw_term
        roll:   psi_des =       cd_y * d_l + cv_y * v_l           - side * psi0
    where d is the horizontal offset of the mass centre from the stance foot and v the mass-centre
    velocity, both in the heading frame (f = forward, l = left).  This is foot placement: the further
    and the faster the body falls, the further out the swing foot is put.  The feet are points, so it
    is the only balance mechanism there is.
  * heading: a point foot cannot transmit a yaw torque, so the heading is steered by step-length
    asymmetry: yaw_term = k_yaw * yaw + k_yawd * yaw_rate lengthens the step of one leg and shortens
    the other's, which (measured on the plant) turns the body back towards world +x.
  * stance hip, pitch and roll: holds the torso upright, tau_stance = -tau_torso - tau_swing, where
    tau_torso is a PD torque on the torso attitude, limited to `t_lim` (the abdomen passes at most
    40 N m on to the torso; asking for more only whips the light pelvis around and makes the gait chatter).
  * knees: local PD.  Swing knee bent in phase 0, straight in phase 1; stance knee straight.
    In phase 1 the stance knee target yields at `k_drop` rad/s: if the swing foot has not found the
    ground when expected (the next tread is 3 cm lower) the body is lowered onto it instead of
    toppling forward over the stance foot, which is what made the gait run away on the stairs.
  * abdomen, hip yaw, arms: local PD to the zero posture.

The terrain is never observed and nothing here knows where the risers are: touch-down comes from the
contact switches, the yielding stance knee reaches for a lower tread, and the foot-placement feedback
absorbs the extra speed each step down adds.

Gains (PARAMS)
--------------
t_start           s, landing time before the first step
t_swing, t_max    s, duration of phase 0 / longest step
swh0, swh1        rad, swing thigh pitch targets in phases 0 / 1 (world frame, negative = forward)
swk0, swk1        rad, swing knee targets in phases 0 / 1 (negative = flexed)
stk, k_drop       rad, rad/s: stance knee target and its yield rate in phase 1
cd_x, cv_x        rad/m, rad s/m: sagittal foot-placement feedback on mass-centre offset / velocity
v_ref             m/s, forward speed at which the velocity feedback is zero (the forward drive: raising it
                  shortens the steps, so the body keeps falling forward; sets the walking speed, ~0.35 m/s)
cd_y, cv_y        rad/m, rad s/m: lateral foot-placement feedback
psi0              rad, constant inward offset of the swing leg (negative = slightly wider stance)
k_yaw, k_yawd     rad/rad, rad s/rad: heading feedback through step-length asymmetry (limited to 0.2 rad)
pitch_des         rad, torso pitch target
t_lim             N m, limit of the torso attitude torques
kp_t, kd_t        torso pitch PD (through the stance hip)
kp_tx, kd_tx      torso roll PD (through the stance hip)
kp_h, kd_h        swing hip pitch PD
kp_hx, kd_hx      swing hip roll PD
kp_k, kd_k        knee PD
kp_a, kd_a        abdomen PD
kp_z              hip yaw stiffness (halved on the stance leg)
kp_arm, kd_arm    arm PD

The structure was designed by hand; the numbers were then set by a derivative-free random search
(perturb a few gains, keep the change if more seeds stay up and get further) on a staircase model,
and checked on the heightfield evaluator.
"""

import math

import numpy as np

PARAMS = {
    "t_start": 0.2346,
    "t_swing": 0.2692,
    "t_max": 0.7,
    "swh0": -0.5444,
    "swh1": 0.1,
    "swk0": -1.1,
    "swk1": -0.05,
    "stk": -0.02,
    "k_drop": 4.0,
    "cd_x": 0.5302,
    "cv_x": 0.2091,
    "v_ref": 0.11,
    "cd_y": 0.5632,
    "cv_y": 0.261,
    "psi0": -0.0067,
    "k_yaw": 0.4,
    "k_yawd": 0.1,
    "pitch_des": 0.0,
    "t_lim": 25.0,
    "kp_t": 492.8245,
    "kd_t": 30.0,
    "kp_tx": 255.8213,
    "kd_tx": 30.0,
    "kp_h": 300.3441,
    "kd_h": 19.4959,
    "kp_hx": 198.6179,
    "kd_hx": 13.5199,
    "kp_k": 109.2124,
    "kd_k": 5.4893,
    "kp_a": 135.9778,
    "kd_a": 22.0917,
    "kp_z": 20.0,
    "kp_arm": 10.0,
    "kd_arm": 1.0,
}

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], dtype=float)
TAU_MAX = 0.4 * GEAR
# motor indices
ABD_Y, ABD_Z, ABD_X = 0, 1, 2
LEG = {-1: (3, 4, 5, 6), +1: (7, 8, 9, 10)}  # side -> (hip_x, hip_z, hip_y, knee); -1 = right, +1 = left
HIPX_SIGN = {-1: 1.0, +1: -1.0}              # world-x rotation per unit hip_x angle (the left hip_x axis is -x)
YAW_TERM_MAX = 0.2                           # rad, limit of the heading correction of the swing thigh
STANCE_KNEE_MIN = -0.8                       # rad, how far the stance knee may yield


def _clip(x, lim):
    return lim if x > lim else (-lim if x < -lim else x)


class Walker:
    """The controller with its step state (stance side, start time of the current step)."""

    def __init__(self, params=None):
        self.p = dict(PARAMS)
        if params:
            self.p.update(params)
        self.reset()

    def reset(self):
        self.stance = +1      # +1: the left leg is the stance leg, -1: the right
        self.t0 = None        # start time of the current step (None while landing)

    def __call__(self, obs):
        p = self.p
        if obs["step"] == 0:
            self.reset()
        t = float(obs["time"])
        q = obs["joint_angles"]
        qd = obs["joint_rates"]
        yaw = obs["yaw"]
        cy, sy = math.cos(yaw), math.sin(yaw)
        w = obs["root_angular_velocity"]
        w_roll = cy * w[0] + sy * w[1]            # torso rates in the heading frame
        w_pitch = -sy * w[0] + cy * w[1]
        pitch, roll = obs["pitch"], obs["roll"]
        v = obs["com_velocity_xy"]
        v_f = cy * v[0] + sy * v[1]               # mass-centre velocity, forward / left
        v_l = -sy * v[0] + cy * v[1]
        foot = {-1: obs["foot_xyz_rel"], +1: obs["foot_left_xyz_rel"]}
        contact = {-1: bool(obs["foot_contact"]), +1: bool(obs["foot_left_contact"])}

        tau = np.zeros(17)
        # posture joints: abdomen, arms, hip yaw
        tau[ABD_Y] = p["kp_a"] * (0.0 - q[ABD_Y]) - p["kd_a"] * qd[ABD_Y]
        tau[ABD_Z] = p["kp_a"] * (0.0 - q[ABD_Z]) - p["kd_a"] * qd[ABD_Z]
        tau[ABD_X] = p["kp_a"] * (0.0 - q[ABD_X]) - p["kd_a"] * qd[ABD_X]
        for i in range(11, 17):
            tau[i] = p["kp_arm"] * (0.0 - q[i]) - p["kd_arm"] * qd[i]
        for side in (-1, 1):
            hz = LEG[side][1]
            tau[hz] = p["kp_z"] * (0.0 - q[hz])

        if self.t0 is None and t >= p["t_start"]:
            self.t0 = t
            # first stance leg: the foot the mass centre is closer to
            self.stance = +1 if foot[+1][1] ** 2 + foot[+1][0] ** 2 <= foot[-1][1] ** 2 + foot[-1][0] ** 2 else -1

        if self.t0 is None:
            # landing: both legs hold a straight stance, both hips keep the torso upright
            t_pitch = p["kp_t"] * (p["pitch_des"] - pitch) - p["kd_t"] * w_pitch
            t_roll = p["kp_tx"] * (0.0 - roll) - p["kd_tx"] * w_roll
            for side in (-1, 1):
                hx, hz, hy, kn = LEG[side]
                tau[hy] = -0.5 * t_pitch
                tau[hx] = HIPX_SIGN[side] * (-0.5 * t_roll)
                tau[kn] = p["kp_k"] * (p["stk"] - q[kn]) - p["kd_k"] * qd[kn]
            return np.clip(tau / GEAR, -0.4, 0.4)

        # ---- state machine: swap the legs when the swing foot touches down after phase 0 ----
        el = t - self.t0
        sw = -self.stance
        if (el > p["t_swing"] and contact[sw]) or el > p["t_max"]:
            self.stance = sw
            sw = -self.stance
            self.t0 = t
            el = 0.0
        st = self.stance
        phase = 0 if el < p["t_swing"] else 1
        s_hx, s_hz, s_hy, s_kn = LEG[sw]
        t_hx, t_hz, t_hy, t_kn = LEG[st]

        # ---- balance feedback: mass centre relative to the stance foot, heading frame ----
        d_x, d_y = -foot[st][0], -foot[st][1]
        d_f = cy * d_x + sy * d_y
        d_l = -sy * d_x + cy * d_y

        # ---- swing leg: world-frame thigh angles track the foot-placement targets ----
        pel_pitch = pitch + q[ABD_Y]
        pel_pitch_rate = w_pitch + qd[ABD_Y]
        pel_roll = roll + q[ABD_X]
        pel_roll_rate = w_roll + qd[ABD_X]
        phi = pel_pitch + q[s_hy]                          # thigh pitch, negative = forward
        phi_rate = pel_pitch_rate + qd[s_hy]
        psi = pel_roll + HIPX_SIGN[sw] * q[s_hx]           # thigh roll, positive = foot to the left
        psi_rate = pel_roll_rate + HIPX_SIGN[sw] * qd[s_hx]
        yaw_term = _clip(p["k_yaw"] * yaw + p["k_yawd"] * w[2], YAW_TERM_MAX)
        phi_des = (p["swh0"] if phase == 0 else p["swh1"]) - p["cd_x"] * d_f - p["cv_x"] * (v_f - p["v_ref"])
        phi_des += sw * yaw_term                           # heading: step-length asymmetry
        psi_des = p["cd_y"] * d_l + p["cv_y"] * v_l - sw * p["psi0"]
        tau_sw_y = _clip(p["kp_h"] * (phi_des - phi) - p["kd_h"] * phi_rate, TAU_MAX[s_hy])
        tau_sw_x = _clip(p["kp_hx"] * (psi_des - psi) - p["kd_hx"] * psi_rate, TAU_MAX[s_hx])
        tau[s_hy] = tau_sw_y
        tau[s_hx] = HIPX_SIGN[sw] * tau_sw_x
        tau[s_kn] = p["kp_k"] * ((p["swk0"] if phase == 0 else p["swk1"]) - q[s_kn]) - p["kd_k"] * qd[s_kn]

        # ---- stance leg: torso attitude through the hip, knee straight but yielding in phase 1 ----
        t_pitch = _clip(p["kp_t"] * (p["pitch_des"] - pitch) - p["kd_t"] * w_pitch, p["t_lim"])
        t_roll = _clip(p["kp_tx"] * (0.0 - roll) - p["kd_tx"] * w_roll, p["t_lim"])
        tau[t_hy] = -t_pitch - tau_sw_y
        tau[t_hx] = HIPX_SIGN[st] * (-t_roll - tau_sw_x)
        stk = p["stk"]
        if phase == 1:
            stk = max(stk - p["k_drop"] * (el - p["t_swing"]), STANCE_KNEE_MIN)   # reach down for a lower tread
        tau[t_kn] = p["kp_k"] * (stk - q[t_kn]) - p["kd_k"] * qd[t_kn]
        tau[t_hz] = -p["kp_z"] * q[t_hz] * 0.5
        return np.clip(tau / GEAR, -0.4, 0.4)


_WALKER = Walker()


def controller(obs: dict) -> np.ndarray:
    """Motor commands (17,) in [-0.4, 0.4] in the XML's actuator order."""
    return _WALKER(obs)
