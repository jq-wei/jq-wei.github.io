"""Phase-based jogging controller for the MuJoCo Humanoid (CMA-ES-tuned gains).

A clock at `freq` Hz drives alternating leg phases (right swings while sin(ph) > 0).
Joint references are tracked with PD (tau = KP (ref - q) - KD qdot, divided by the
motor gear and clipped to [-0.4, 0.4]):

- Stance hip_y: `hip0 - prop * sin(ph)` propels the body; propulsion `prop` backs
  off by `k_prop` per m/s that the COM speed exceeds `v_tgt`.
- Swing hip_y: capture-point placement `-(step0 + k_cap*vx + k_brake*max(0,vx-v_tgt)
  + k_pcap*(pitch-lean))` -- the foot lands further ahead the faster or more
  pitched-forward the body, which brakes and stabilizes sagittally.
- Knee: near-straight `knee_st` in stance, flexes `knee_amp` during swing
  (phase lead `knee_phase`).
- Hip_x: baseline `lat0` plus swing-leg lateral placement `k_vy * vy` (mirrored
  axes); stance hip_x/hip_y also receive direct attitude torques `kt_roll`,
  `kt_pitch`, `kt_pitchd` that hold the torso against the anchored stance foot.
- Hip_z: the stance leg applies a heading torque `-kt_yaw*yaw - kt_yawd*yaw_rate`
  and the swing leg pre-rotates by `k_steer*yaw` so the next touchdown steers the
  run back onto the world x axis (yaw drift was the dominant late failure mode).
- Abdomen PD keeps the trunk upright and the heading at zero; arms swing in
  antiphase with the legs. `ramp` blends the gait in from the initial stand.

Gains were optimized with CMA-ES/TPE (Optuna) over rollouts on seeds 0-9; no
gradient-based learning was used.
"""

import numpy as np

AB_Y, AB_Z, AB_X = 0, 1, 2
R_HX, R_HZ, R_HY, R_KNEE = 3, 4, 5, 6
L_HX, L_HZ, L_HY, L_KNEE = 7, 8, 9, 10
R_S1, R_S2, R_ELB = 11, 12, 13
L_S1, L_S2, L_ELB = 14, 15, 16

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200,
                 25, 25, 25, 25, 25, 25], float)

PARAMS = {
    'freq': 4.3563,
    'hip_amp': 0.339912,
    'step0': 0.198931,
    'k_cap': 0.674642,
    'knee_amp': 0.979063,
    'knee_phase': -0.526269,
    'hip0': -0.217838,
    'knee_st': -0.290527,
    'lean': 0.154245,
    'kt_pitch': -113.017,
    'kt_pitchd': -5.09787,
    'kt_roll': -97.2898,
    'k_vy': -0.108833,
    'lat0': -0.216828,
    'ramp': 0.355311,
    'v_tgt': 1.2,
    'k_brake': 0.638361,
    'k_pcap': 0.96172,
    'k_prop': 1.07899,
    'kt_yaw': 60.0,   # stance hip_z steering torque [Nm/rad]
    'kt_yawd': -15.0,
    'k_steer': 0.4,     # swing hip_z per rad of yaw (foot steering)
}

KP = np.array([150, 80, 80, 120, 80, 250, 250, 120, 80, 250, 250,
               20, 20, 10, 20, 20, 10], float)
KD = np.array([8, 4, 4, 6, 4, 12, 8, 6, 4, 12, 8, 1, 1, 0.5, 1, 1, 0.5], float)


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    q = obs["joint_angles"]
    dq = obs["joint_rates"]
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    wx, wy = obs["root_angular_velocity"][0], obs["root_angular_velocity"][1]
    vx, vy = obs["com_velocity_xy"]
    t = obs["time"]

    ramp = min(1.0, t / max(p["ramp"], 1e-3))
    ph = 2.0 * np.pi * p["freq"] * t
    s_r = np.sin(ph)
    s_l = -s_r
    # smooth swing weights in [0,1]
    sw_r = min(1.0, max(0.0, 2.0 * s_r))
    sw_l = min(1.0, max(0.0, 2.0 * s_l))

    ref = np.zeros(17)
    tau_extra = np.zeros(17)

    wz = obs["root_angular_velocity"][2]
    over = max(0.0, vx - p["v_tgt"])
    swing_target = -(p["step0"] + p["k_cap"] * vx + p["k_brake"] * over
                     + p["k_pcap"] * (pitch - p["lean"]))   # negative hip_y = leg forward
    prop = p["hip_amp"] * max(0.0, 1.0 - p["k_prop"] * over)
    for HY, HX, HZ, KNEE, s, sw, mir in (
        (R_HY, R_HX, R_HZ, R_KNEE, s_r, sw_r, +1.0),
        (L_HY, L_HX, L_HZ, L_KNEE, s_l, sw_l, -1.0),
    ):
        # sagittal hip: stance propulsion vs swing placement
        st_ref = p["hip0"] - prop * s
        ref[HY] = ramp * ((1 - sw) * st_ref + sw * swing_target)
        # knee: stance nearly straight, swing flexed
        bell = max(0.0, np.sin(ph * (1 if mir > 0 else 1) + (0 if mir > 0 else np.pi) + p["knee_phase"]))
        ref[KNEE] = -0.06 + ramp * ((1 - sw) * p["knee_st"] + (-p["knee_amp"]) * bell * sw)
        # lateral: swing leg places foot against lateral velocity; mirrored axes
        # (negative right_hip_x = right foot outward; negative left_hip_x = left foot outward)
        place_y = p["k_vy"] * vy          # want foot toward velocity: +y means right foot inward(+), left outward(-)
        ref[HX] = p["lat0"] + sw * (mir * place_y)
        # stance hip torques hold the torso attitude
        tau_extra[HY] += (1 - sw) * (-p["kt_pitch"] * (pitch - p["lean"] * ramp) - p["kt_pitchd"] * wy)
        tau_extra[HX] += (1 - sw) * mir * (-p["kt_roll"] * roll - 0.2 * p["kt_roll"] * wx)
        # stance hip_z steers the heading back to zero (axes mirrored between legs);
        # the swing hip_z pre-rotates the leg so the next touchdown corrects heading
        tau_extra[HZ] += (1 - sw) * mir * (-p["kt_yaw"] * yaw - p["kt_yawd"] * wz)
        ref[HZ] = sw * mir * p["k_steer"] * yaw

    # abdomen keeps torso upright, hips_z/abdomen_z keep heading
    ref[AB_Y] = -0.3 * (pitch - p["lean"] * ramp)
    ref[AB_X] = -0.3 * roll
    ref[AB_Z] = -0.5 * yaw

    # arms swing opposite to the legs
    ref[R_S1] = 0.3 * s_l
    ref[L_S1] = 0.3 * s_r
    ref[R_ELB] = ref[L_ELB] = -0.5

    tau = KP * (ref - q) - KD * dq + tau_extra
    return np.clip(tau / GEAR, -0.4, 0.4)
