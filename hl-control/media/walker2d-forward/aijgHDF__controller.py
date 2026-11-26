"""In-place bouncing controller for MuJoCo Walker2d (alive on every seed,
slight forward drift; it does NOT reach the 3 m `forward` rung).

Both legs act in phase: a sinusoidal clock (period) pumps the knees around a
statically stable crouch (knee kmid, ankle amid dorsiflexed, hip -0.05) while
the ankles follow with phase lead phi_a and amplitude aamp. Servo layout:
  - PD on all six joints (stiffness KP, damping KD; 1.0 command = 100 N m);
  - a torso-pitch PD (kp_pitch, setpoint pitch_t) shared by both hips: a
    positive hip command reacts nose-down on the torso;
  - knee height feedback kz extends the knees when the torso drops below the
    1.18 m reference, preventing bounce collapse;
  - a speed governor (kv, v_des) can throttle the ankle drive with forward
    velocity; kv = 0 here, i.e. inactive in the tuned configuration.
For t < T_SETTLE only the crouch is held so the reset noise settles, then the
oscillation amplitude (master gain amp) ramps in over RAMP seconds.

Design notes: the plant's hips cannot swing forward of the torso (range
[-150 deg, 0]) and the ankle sits at the heel, so a backward (nose-up) fall is
uncorrectable and the swing foot cannot be placed ahead of the centre of mass.
Every stepping/bounding gait tried moved up to ~1-2.3 m but collapsed (torso
below 0.8 m) once the planted/trailing feet fell ~1 m behind the body ("the
splits"). The tuned bounce keeps |pitch| < 0.22 and height > 1.15 for the full
1000 steps on all tested seeds (0-12), scoring the `alive` rung with x ~ +0.2 m.
"""

import numpy as np

PARAMS = {
    "amp": 0.5285651195764919,
    "period": 0.608066254285056,
    "kmid": -0.5085999438549169,
    "kamp": 0.45717598758105205,
    "amid": 0.3309389221084223,
    "aamp": 0.1427491637545345,
    "phi_a": 0.9253469620646624,
    "pitch_t": 0.24861139832018953,
    "kp_pitch": 9.48082027444276,
    "kz": 0.9909649662232385,
    "v_des": 0.55,
    "kv": 0.0,
}

PARAM_BOUNDS = {
    "amp": (0.5, 1.5),
    "period": (0.25, 1.0),
    "kmid": (-0.8, -0.2),
    "kamp": (0.1, 0.6),
    "amid": (0.05, 0.4),
    "aamp": (0.0, 0.4),
    "phi_a": (-3.14, 3.14),
    "pitch_t": (-0.35, 0.25),
    "kp_pitch": (2.0, 12.0),
    "kz": (0.0, 5.0),
    "v_des": (0.2, 1.5),
    "kv": (0.0, 4.0),
}

KP = np.array([4.0, 5.0, 4.0, 4.0, 5.0, 4.0])   # joint PD stiffness (1.0 = 100 N m / rad)
KD = np.array([0.3, 0.3, 0.15, 0.3, 0.3, 0.15])
KD_PITCH = 1.0
HIP0 = -0.05       # hip crouch angle (near the forward limit)
HAMP = 0.05        # residual hip sweep amplitude
PHI_H = 0.0        # hip phase
T_SETTLE = 0.4     # crouch hold before bouncing [s]
RAMP = 0.4         # amplitude ramp time [s]


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = obs["time"]
    q = obs["joint_angles"]   # [thigh, leg, foot, thigh_left, leg_left, foot_left]
    qd = obs["joint_rates"]

    pitch_u = -P["kp_pitch"] * (obs["pitch"] - P["pitch_t"]) - KD_PITCH * obs["pitch_rate"]

    amp = 0.0
    th = 0.0
    if t >= T_SETTLE:
        amp = P["amp"] * min(1.0, (t - T_SETTLE) / RAMP)
        th = 2.0 * np.pi * (t - T_SETTLE) / P["period"]

    hip_t = HIP0 - amp * HAMP * 0.5 * (1.0 - np.sin(th + PHI_H))
    # height feedback: extend the knees when the torso drops (z_ref 1.18 m)
    knee_t = P["kmid"] + amp * P["kamp"] * np.sin(th) + P["kz"] * max(0.0, 1.18 - obs["height"])
    knee_t = min(knee_t, -0.02)
    drive = float(np.clip(1.0 - P["kv"] * (obs["x_velocity"] - P["v_des"]), 0.0, 1.5))
    ankle_t = P["amid"] + amp * P["aamp"] * drive * np.sin(th + P["phi_a"])
    tgt = np.array([hip_t, knee_t, ankle_t, hip_t, knee_t, ankle_t])

    u = KP * (tgt - q) - KD * qd
    u[0] += pitch_u / 2
    u[3] += pitch_u / 2

    return np.clip(u, -1.0, 1.0)
