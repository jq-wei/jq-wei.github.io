"""Humanoid floor recovery: roll onto the back, then sit up and hold the seated posture.

STATUS (honest): this controller reaches the first rung of the ladder (`sit`: root >= 0.45 m, stable
seated posture at about 0.51 m) from face-up, face-down and side-lying starts.  It does NOT reach
`crouch` or `stand`: balancing this ankle-less, sphere-footed body on its feet needs an active
sagittal balance controller that I did not get working within the session (see the notes at the end).

Control law
-----------
Every joint is driven by a joint-space PD law on a rate-limited reference,

    tau = KP * (ref - q) - KD * qdot,      command = clip(tau / gear, -0.4, 0.4)

where `ref` slides towards the goal posture of the current phase at most `rate` rad/s and never
leads the measured joint by more than `WIND` (anti-windup, so a blocked joint does not saturate).
Dividing by the observed gear makes the torques identical under the weak-motor scenario.

Phases (a small state machine on the torso orientation; `frontz`/`leftz` are the vertical components
of the torso's front and left axes, taken from the root quaternion):

  roll, stage 1 (face down, roll angle phi < 50 deg): tuck both knees under the body with one hip
        flexed less than the other and the trunk bent sideways, so the body topples onto a side.
  roll, stage 2 (on a side): top leg extended back and abducted, bottom leg flexed forward; the
        body tips over onto its back.  If it is still not supine after T_STUCK the tuck is repeated
        (every second retry also twists the waist).
  sit   (supine for 25 steps): legs straight and apart, hips flexed to SIT_HIP, waist to SIT_ABY.
        The trunk is heavier about the hips than the legs, so the legs lift first and the body
        rocks forward over the pelvis into a long-sit; the posture leans ~15 deg forward of
        vertical, which is statically stable (mass centre in front of the pelvis).  If the trunk
        falls back onto a side the machine returns to `roll`.

Gains
-----
KP, KD   joint stiffness / damping (N m/rad, N m s/rad) in motor order.  KD is deliberately small on
         the hips, waist and hip yaw: the pelvis is a light body between strong motors, and larger
         derivative gains make the 15 ms loop chatter at the Nyquist rate.
WIND     largest lead of the reference over the joint (rad): caps the PD torque per joint.
RATE_*   reference slew rates (rad/s) for the roll, tuck and sit-up motions.
TUCK_*   stage-1 posture (hip/knee/waist flexion, hip asymmetry, side bend).
TOP_HIP, BOT_HIP, BOT_KNEE, TOP_HX   stage-2 posture (top leg back and abducted, bottom leg forward).
SIT_HIP, SIT_ABY, SIT_HX             seated posture (hip flexion, waist flexion, leg spread).
T_STUCK  time on a side before the roll is restarted; T_WINDUP is how long a retry tucks before kicking out again.

Notes on the upper rungs (what was tried)
-----------------------------------------
A squat-rise under a gain-scheduled LQR built from the plant model (central finite differences of the
plant's 15 ms control-step map around 33 squat postures with the mass centre over the feet, a
converged contact solver in the design model, half of the path velocity fed forward) does rise from
an exactly balanced deep squat to a 3 s stand at 1.28 m on the nominal plant.  It is not in this file
because (a) it loses balance after sagittal pushes of only ~0.05-0.1 m/s, and (b) I found no reliable
way to get from the floor into a balanced squat: the tucked sit has the mass centre ~0.23 m behind
the feet and tips backwards, and the prone -> kneel -> sit-on-heels route stalls on hands and knees
(the weak waist and arm motors cannot slide or lift the trunk).
"""

import numpy as np

JOINTS = ("abdomen_y", "abdomen_z", "abdomen_x", "right_hip_x", "right_hip_z", "right_hip_y", "right_knee",
          "left_hip_x", "left_hip_z", "left_hip_y", "left_knee", "right_shoulder1", "right_shoulder2",
          "right_elbow", "left_shoulder1", "left_shoulder2", "left_elbow")          # motor order
IDX = {n: i for i, n in enumerate(JOINTS)}

KP = np.array([100, 80, 100, 120, 40, 300, 200, 120, 40, 300, 200, 20, 20, 15, 20, 20, 15], float)
KD = np.array([5, 4, 5, 5, 1.5, 6, 6, 5, 1.5, 6, 6, 1.5, 1.5, 0.8, 1.5, 1.5, 0.8], float)
WIND = np.array([0.3, 0.4, 0.4, 0.4, 0.5, 0.6, 0.4, 0.4, 0.5, 0.6, 0.4, 1, 1, 1, 1, 1, 1], float)

PARAMS = {
    "RATE_ROLL": 3.0, "RATE_TUCK": 1.5, "RATE_SIT": 0.8,            # reference slew rates [rad/s]
    "TUCK_HIP": -1.7, "TUCK_ASYM": 0.5, "TUCK_KNEE": -2.2, "TUCK_ABY": -0.8, "TUCK_ABX": 0.5,
    "TOP_HIP": 0.3, "TOP_HX": -0.43, "BOT_HIP": -1.0, "BOT_KNEE": -1.0,
    "SIT_HIP": -1.5, "SIT_ABY": -0.3, "SIT_HX": -0.3,
    "T_STUCK": 2.5, "T_WINDUP": 0.7,
}

_S = {}   # per-episode state, re-initialised at step 0


def _pose(**angles):
    q = np.zeros(17)
    for name, value in angles.items():
        q[IDX[name]] = value
    return q


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    w, x, y, z = obs["gym_obs"][1:5]                       # root quaternion
    frontz = 2.0 * (x * z - w * y)                         # +1 face up, -1 face down
    leftz = 2.0 * (y * z + w * x)                          # +1 lying on the right side (left side up)
    upz = float(obs["up_vector"][2])
    q = np.asarray(obs["joint_angles"], float)
    qd = np.asarray(obs["joint_rates"], float)
    t = float(obs["time"])
    dt = float(obs["dt"])

    if obs["step"] == 0 or "ref" not in _S:
        _S.clear()
        _S.update(ref=q.copy(), dir=1.0 if leftz >= 0 else -1.0, sup=0, phase="roll", stage=1, t_stage=0.0, t_sit=0.0, tries=0)
    S = _S

    # roll angle about the body's long axis in the chosen direction: 0 prone, 90 on a side, 180 supine
    def roll_angle(d):
        phi = np.degrees(np.arctan2(leftz * d, -frontz))
        return phi + 360.0 if phi < -60.0 else phi

    S["sup"] = S["sup"] + 1 if (frontz > 0.9 and abs(leftz) < 0.35) else 0
    if S["phase"] == "roll" and S["sup"] > 25:
        S["phase"], S["t_sit"] = "sit", t
    if S["phase"] == "sit" and t - S["t_sit"] > 1.0 and upz < 0.5 and frontz < 0.6:      # toppled while sitting up
        S.update(phase="roll", stage=1, t_stage=t, sup=0, tries=S["tries"] + 1)

    if S["phase"] == "roll":
        if S["stage"] == 1 and abs(leftz) > 0.3:           # roll towards the side that is already going down
            S["dir"] = 1.0 if leftz >= 0 else -1.0
        d = S["dir"]
        phi = roll_angle(d)
        top, bot = ("left", "right") if d > 0 else ("right", "left")
        since = t - S["t_stage"]
        hold = P["T_WINDUP"] if S["tries"] > 0 else 0.0      # on a retry, tuck for a moment first so the un-tuck has momentum
        if S["stage"] == 1 and since >= hold and (phi > 50 or (phi > 25 and since > 1.8) or since > 3.0):
            S["stage"], S["t_stage"] = 2, t
        elif S["stage"] == 2 and since > P["T_STUCK"]:
            S["stage"], S["t_stage"], S["tries"] = 1, t, S["tries"] + 1
        rate = P["RATE_ROLL"]
        if phi >= 150:                                      # almost supine: straighten out
            goal = _pose()
        elif S["stage"] == 1:                               # tuck and topple
            goal = _pose(**{f"{top}_hip_y": P["TUCK_HIP"], f"{bot}_hip_y": P["TUCK_HIP"] + P["TUCK_ASYM"],
                            "left_knee": P["TUCK_KNEE"], "right_knee": P["TUCK_KNEE"],
                            "abdomen_y": P["TUCK_ABY"], "abdomen_x": P["TUCK_ABX"] * d})
            rate = P["RATE_TUCK"]
        else:                                               # on a side: top leg back, bottom leg forward
            goal = _pose(**{f"{top}_hip_y": P["TOP_HIP"], f"{top}_knee": -0.3, f"{top}_hip_x": P["TOP_HX"],
                            f"{bot}_hip_y": P["BOT_HIP"], f"{bot}_knee": P["BOT_KNEE"],
                            "abdomen_z": -0.6 * d * (S["tries"] % 2)})
    else:                                                   # sit up and hold
        rate = P["RATE_SIT"]
        goal = _pose(right_hip_y=P["SIT_HIP"], left_hip_y=P["SIT_HIP"], right_knee=-0.05, left_knee=-0.05,
                     right_hip_x=P["SIT_HX"], left_hip_x=P["SIT_HX"], abdomen_y=P["SIT_ABY"])

    S["ref"] += np.clip(goal - S["ref"], -rate * dt, rate * dt)
    S["ref"] = np.clip(S["ref"], q - WIND, q + WIND)
    tau = KP * (S["ref"] - q) - KD * qd
    return np.clip(tau / np.asarray(obs["gear"], float), -0.4, 0.4)
