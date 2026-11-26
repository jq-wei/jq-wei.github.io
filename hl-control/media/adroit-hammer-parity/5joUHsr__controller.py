"""Adroit hammer: phased joint-space controller (reach -> grasp -> lift and sweep -> press).

No learning and no task-space interface: every 10 ms the 46-vector observation is mapped to 26 joint set points.
The numbers below were found by derivative-free random / local search over rollouts of the simulator
(scratch script, not needed at run time) on boards between 0.08 m and 0.27 m high.

Control law
-----------
1. Reach (steps 0-2).  The arm pitches down toward ``q_low`` and the wrist flexes (``W0_GRASP``), which brings the
   open hand down over the handle; the fingers stay straight, the thumb is pre-shaped (``THUMB_OPEN``).
2. Grasp (from step ``T_CLOSE``, while the hand is still coming down).  The four fingers curl around the handle
   (``FINGER_GRASP``, the same three flexion set points for every finger) and scoop it up against the
   little-finger side of the palm, which arches toward it (``LF4_GRASP``); the thumb tip closes on the handle /
   neck from the other side (``THUMB_GRASP``).  The finger servos are plain position servos, so a posture is
   commanded and the handle is clamped by the servo springs; these set points are held for the rest of the episode.
3. Lift and sweep (from step ``T_LIFT``).
   * Height loop (arm pitch).  The centre of the hammer head is reconstructed from the measured hammer pose
     (body position + Euler angles; the head sits 0.24 m along the hammer's -x axis).  Its height error to the
     nail, ``e = z_nail + dz - z_head``, is turned into a pitch target ``q_des = q - clip(e / len, +-dq_max)``,
     i.e. an integrating outer loop on the measured head height: it removes the droop of the hammer in the
     hand and adapts to the board height, which is the only thing that changes from seed to seed.
   * Sweep (arm yaw + wrist deviation).  Both targets are slewed at ``yaw_rate`` / ``w1_rate`` to their end stops
     (``YAW_END``, ``W1_END``): this turns the hammer about the vertical by about 45 deg and carries its head
     along +x into the nail head.  The wrist goes to the carrying flexion ``w0_carry`` (extended).
   * Gate.  While the head is still more than ``z_tol`` off the nail height, the sweep is frozen as soon as the
     head comes within ``x_gate`` of the nail along x, so the head never arrives under or over the nail.
4. Press.  Nothing changes after the stroke: the slewed targets stay at their end stops, so the wrist-deviation
   servo keeps the head resting on the driven nail while the height loop keeps it level with the nail until the
   episode ends.  The nail is held about 2 mm deeper than where its soft end stop would leave it on its own,
   about 2.5 mm inside the 10 mm success radius, and the hammer stays in the hand.

Arm servo.  The arm servos apply ``g*ctrl - k*q`` (g = 500, k = 200), which on its own is slow and sags under
gravity.  The set point is therefore computed by inverting that law around a PD loop on the joint target,
``ctrl = (k*q + kp*(q_des - q) - kd*qdot + tau_ff) / g`` (qdot by finite differences of the measured joint angle),
which cancels the built-in spring and replaces it with a stiffer one.

Role of each gain (``PARAMS``)
------------------------------
kp, kd      PD gains of the two arm joints [N m/rad, N m s/rad]: stiffness / damping of pitch and yaw tracking.
tau_g       gravity feed-forward torque on the pitch joint [N m] (negative = holds the arm up).
len         lever arm that converts a head-height error into a pitch step [m]; larger = softer height loop.
dq_max      largest pitch lead over the measured angle [rad]; caps the lift speed at about kp*dq_max/(kd+20) rad/s.
dz          height of the head centre relative to the nail axis that the height loop aims for [m].
yaw_rate    slew rate of the arm-yaw target during the sweep [rad/s].
w1_rate     slew rate of the wrist-deviation target during the sweep [rad/s].
x_gate      distance along x between head centre and nail head at which the sweep waits for the lift [m].
z_tol       head-height error below which the sweep may pass the gate [m].
q_low       arm pitch target of the reach / grasp [rad] (how low the hand comes over the handle).
w0_carry    wrist flexion while carrying and striking [rad] (negative = extended).
"""

import numpy as np

PARAMS = {
    "kp": 1000.0,
    "kd": 25.0,
    "tau_g": -16.0,
    "len": 0.8823,
    "dq_max": 0.2068,
    "dz": -0.0107,
    "yaw_rate": 6.3509,
    "w1_rate": 12.0,
    "x_gate": 0.1027,
    "z_tol": 0.0298,
    "q_low": 0.2141,
    "w0_carry": -0.3436,
}
PARAM_BOUNDS = {
    "kp": (500.0, 2000.0), "kd": (10.0, 40.0), "tau_g": (-30.0, 0.0),
    "len": (0.3, 1.2), "dq_max": (0.1, 0.45), "dz": (-0.03, 0.03),
    "yaw_rate": (1.0, 8.0), "w1_rate": (1.0, 12.0),
    "x_gate": (0.05, 0.16), "z_tol": (0.012, 0.06),
    "q_low": (0.12, 0.25), "w0_carry": (-0.4, 0.4),
}

# ---- schedule [control steps] ----
T_CLOSE = 3            # the fingers start closing
T_LIFT = 7             # the height loop and the sweep start

# ---- postures [rad] ----
W0_GRASP = 0.4031                                  # wrist flexion while reaching / grasping
FINGER_GRASP = (1.2932, 1.4715, 1.5944)            # knuckle, middle and tip flexion of every finger
LF4_GRASP = 0.3384                                 # little-finger metacarpal (palm arch)
THUMB_OPEN = (-0.7114, 0.7914, 0.0, 0.0, 0.0)      # THJ4..THJ0 while reaching
THUMB_GRASP = (0.8625, 0.5797, -0.1584, 0.3064, -1.2034)   # THJ4..THJ0 of the grasp
YAW_END, W1_END = -0.3, -0.524                     # end stops of the sweep (arm yaw, wrist deviation)

# ---- plant constants ----
_ARM_G, _ARM_K = 500.0, 200.0                      # arm servo: force = g*ctrl - k*q
_HEAD_IN_HAMMER = np.array([-0.24, 0.0, 0.0])      # centre of the head in the hammer's body frame [m]
_PITCH_RANGE = (-0.4, 0.25)

# observation / action indices (joint order = actuator order)
_PITCH, _YAW, _WRJ1, _WRJ0 = 0, 1, 2, 3
_FINGER_KNUCKLES = (5, 9, 13, 18)                  # FFJ2, MFJ2, RFJ2, LFJ2 (then J1, J0 follow)
_LFJ4 = 16
_THUMB = slice(21, 26)                             # THJ4..THJ0

_S = {}                                            # per-episode memory (previous joint angles, slewed targets)


def _euler2mat(e):
    """Rotation matrix of the package's Euler angles (R = Rx(e0) Ry(e1) Rz(e2))."""
    cx, sx = np.cos(e[0]), np.sin(e[0])
    cy, sy = np.cos(e[1]), np.sin(e[1])
    cz, sz = np.cos(e[2]), np.sin(e[2])
    rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rx @ ry @ rz


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    o = np.asarray(obs["observation"], dtype=float)
    k = int(obs["step"])
    dt = float(obs["dt"])
    mean = np.asarray(obs["ctrl_mean"], dtype=float)
    half = np.asarray(obs["ctrl_half_range"], dtype=float)

    q = o[:26]
    if k == 0 or "q_prev" not in _S:               # new episode
        _S.clear()
        _S["q_prev"] = q.copy()
        _S["yaw_des"] = 0.0
        _S["w1_des"] = 0.0
    qd = (q - _S["q_prev"]) / dt                   # joint velocities by finite differences
    _S["q_prev"] = q.copy()

    nail = o[42:45]                                # nail head (its height is the board height of this seed)
    head = o[36:39] + _euler2mat(o[39:42]) @ _HEAD_IN_HAMMER   # centre of the hammer head

    ctrl = np.zeros(26)                            # set points [rad]; zero = straight fingers

    # ---- hand: open while reaching, then the grasp posture for the rest of the episode ----
    if k < T_CLOSE:
        ctrl[_THUMB] = THUMB_OPEN
    else:
        for j in _FINGER_KNUCKLES:
            ctrl[j:j + 3] = FINGER_GRASP
        ctrl[_LFJ4] = LF4_GRASP
        ctrl[_THUMB] = THUMB_GRASP

    # ---- arm and wrist targets ----
    if k < T_LIFT:                                 # reach / grasp: palm on the handle
        pitch_des = P["q_low"]
        yaw_des, w1_des = 0.0, 0.0
        w0_des = W0_GRASP
    else:                                          # lift and sweep
        err = (nail[2] + P["dz"]) - head[2]        # head-height error [m]
        pitch_des = q[_PITCH] - np.clip(err / P["len"], -P["dq_max"], P["dq_max"])
        w0_des = P["w0_carry"]
        if abs(err) > P["z_tol"] and nail[0] - head[0] < P["x_gate"]:
            yaw_des, w1_des = _S["yaw_des"], _S["w1_des"]      # wait at the gate until the head is level with the nail
        else:
            yaw_des = max(_S["yaw_des"] - P["yaw_rate"] * dt, YAW_END)
            w1_des = max(_S["w1_des"] - P["w1_rate"] * dt, W1_END)
    _S["yaw_des"], _S["w1_des"] = yaw_des, w1_des
    pitch_des = float(np.clip(pitch_des, *_PITCH_RANGE))

    # ---- arm: PD on the joint targets, realised through the servo set points ----
    ctrl[_PITCH] = (_ARM_K * q[_PITCH] + P["kp"] * (pitch_des - q[_PITCH]) - P["kd"] * qd[_PITCH] + P["tau_g"]) / _ARM_G
    ctrl[_YAW] = (_ARM_K * q[_YAW] + P["kp"] * (yaw_des - q[_YAW]) - P["kd"] * qd[_YAW]) / _ARM_G
    # ---- wrist: its own position servo ----
    ctrl[_WRJ1] = w1_des
    ctrl[_WRJ0] = w0_des

    return np.clip((ctrl - mean) / half, -1.0, 1.0)
