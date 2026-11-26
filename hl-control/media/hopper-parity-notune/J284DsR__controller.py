"""Hopper controller: two-mode (stance / flight) linear state feedback, gains tuned by CMA-ES.

Control law.  Each 8 ms step the foot-contact flag selects one of two gain sets, and the
motor command is  u = clip(b_mode + K_mode @ z, -1, 1)  with z the scaled feature vector
    z = [pitch, pitch_rate, thigh, leg, foot, thigh_rate, leg_rate, foot_rate,
         x_velocity, z_velocity, height - 1.2, time since the last contact change] / FSCALE.
So every gain is a plain feedback coefficient: e.g. st_hip_pitch is the stance hip torque
per 0.1 rad of torso pitch (the attitude loop), fl_ank_foot the flight ankle stiffness per
0.3 rad of ankle angle (foot pre-positioning), *_x_vel terms set the forward-speed bias,
*_phase_t terms give the within-stance / within-flight timing (compression then push), and
*_bias is the feed-forward torque of that mode.  The tuned gait keeps the knee locked
straight (knee command saturated), pogo-hops on the ankle, and holds the torso pitch
within about +-0.1 rad by hip torque in stance and in flight (the torso is light, so the
hip steers it without moving the leg).

Gains (THETA) were tuned with CMA-ES on the distance travelled (the alive bonus removed
from the objective to avoid the stand-still optimum), 3 random seeds per generation.
"""
import numpy as np
FEAT = ["pitch", "pitch_rate", "thigh", "leg", "foot", "thigh_rate", "leg_rate", "foot_rate", "x_vel", "z_vel", "height", "phase_t"]
FSCALE = np.array([0.1, 1.0, 0.3, 0.3, 0.3, 3.0, 3.0, 3.0, 1.0, 1.0, 0.1, 0.1])
NF = len(FEAT)
NAMES = [f"{m}_{j}_{f}" for m in ("st", "fl") for j in ("hip", "knee", "ank") for f in ["bias"] + FEAT]
X0 = np.zeros(len(NAMES)); SCALE = np.full(len(NAMES), 0.15)
def features(obs, t_phase):
    z = np.array([obs["pitch"], obs["pitch_rate"], obs["thigh"], obs["leg"], obs["foot"], obs["thigh_rate"], obs["leg_rate"],
                  obs["foot_rate"], obs["x_velocity"], obs["z_velocity"], obs["height"] - 1.2, min(t_phase, 0.3)]) / FSCALE
    return np.concatenate([[1.0], z])
def make_controller(theta):
    W = np.asarray(theta).reshape(2, 3, NF + 1); st = {"step": -1, "t0": 0.0, "prev_c": None}
    def controller(obs):
        if obs["step"] <= st["step"]:
            st.update(step=-1, t0=0.0, prev_c=None)
        st["step"] = obs["step"]; c = bool(obs["foot_contact"]); t = obs["time"]
        if c != st["prev_c"]: st["t0"] = t
        st["prev_c"] = c
        a = W[0 if c else 1] @ features(obs, t - st["t0"])
        return np.clip(a, -1, 1)
    return controller

# Tuned gains (CMA-ES over seeds); see PARAMS for the role of each.
THETA = np.array([
    0.165325,  # st_hip_bias
    -0.236396,  # st_hip_pitch
    -0.695715,  # st_hip_pitch_rate
    0.085106,  # st_hip_thigh
    -0.289024,  # st_hip_leg
    0.211193,  # st_hip_foot
    -0.091686,  # st_hip_thigh_rate
    -0.069830,  # st_hip_leg_rate
    -0.118074,  # st_hip_foot_rate
    0.012489,  # st_hip_x_vel
    -0.031872,  # st_hip_z_vel
    -0.107521,  # st_hip_height
    -0.046083,  # st_hip_phase_t
    -0.296095,  # st_knee_bias
    -0.006641,  # st_knee_pitch
    -0.251674,  # st_knee_pitch_rate
    -0.777896,  # st_knee_thigh
    -0.622224,  # st_knee_leg
    -0.228514,  # st_knee_foot
    0.404192,  # st_knee_thigh_rate
    -0.012537,  # st_knee_leg_rate
    0.332617,  # st_knee_foot_rate
    -0.071934,  # st_knee_x_vel
    -0.243425,  # st_knee_z_vel
    0.179388,  # st_knee_height
    0.297036,  # st_knee_phase_t
    -0.153935,  # st_ank_bias
    0.145422,  # st_ank_pitch
    0.140503,  # st_ank_pitch_rate
    -0.071337,  # st_ank_thigh
    -0.732784,  # st_ank_leg
    -0.063628,  # st_ank_foot
    0.097699,  # st_ank_thigh_rate
    -0.158897,  # st_ank_leg_rate
    0.144627,  # st_ank_foot_rate
    -0.698702,  # st_ank_x_vel
    0.084525,  # st_ank_z_vel
    0.449305,  # st_ank_height
    0.634337,  # st_ank_phase_t
    0.100407,  # fl_hip_bias
    -0.375333,  # fl_hip_pitch
    -0.270596,  # fl_hip_pitch_rate
    0.073345,  # fl_hip_thigh
    -0.504609,  # fl_hip_leg
    0.066960,  # fl_hip_foot
    -0.244130,  # fl_hip_thigh_rate
    0.281506,  # fl_hip_leg_rate
    0.168061,  # fl_hip_foot_rate
    -0.047356,  # fl_hip_x_vel
    -0.220219,  # fl_hip_z_vel
    -0.025918,  # fl_hip_height
    -0.012984,  # fl_hip_phase_t
    -0.226415,  # fl_knee_bias
    0.016802,  # fl_knee_pitch
    -0.966149,  # fl_knee_pitch_rate
    -0.539610,  # fl_knee_thigh
    -0.399395,  # fl_knee_leg
    0.171073,  # fl_knee_foot
    1.887295,  # fl_knee_thigh_rate
    1.094218,  # fl_knee_leg_rate
    -0.096423,  # fl_knee_foot_rate
    0.577862,  # fl_knee_x_vel
    -1.345800,  # fl_knee_z_vel
    0.653558,  # fl_knee_height
    -1.029319,  # fl_knee_phase_t
    0.463242,  # fl_ank_bias
    0.071028,  # fl_ank_pitch
    -0.108394,  # fl_ank_pitch_rate
    0.137009,  # fl_ank_thigh
    -0.255658,  # fl_ank_leg
    -0.511826,  # fl_ank_foot
    -0.186733,  # fl_ank_thigh_rate
    -0.110586,  # fl_ank_leg_rate
    -0.172158,  # fl_ank_foot_rate
    0.231075,  # fl_ank_x_vel
    0.191520,  # fl_ank_z_vel
    0.133240,  # fl_ank_height
    0.075653,  # fl_ank_phase_t
])

_ctrl = make_controller(THETA)


def controller(obs: dict) -> np.ndarray:
    return _ctrl(obs)
