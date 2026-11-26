"""Phase-machine hopping controller for the Gymnasium Hopper.

Hop cycle: CROUCH (knee PD to a bent push-off posture) -> THRUST (full knee
extension until takeoff; a toe-up ankle bias injects nose-down momentum) ->
FLIGHT (knee refolds; the hip runs a pitch servo whose reaction also performs
Raibert foot placement: correcting nose-down pitch swings the leg forward,
which brakes) -> ABSORB (soft knee on touchdown) -> THRUST again.

In stance the hip servos torso pitch to a small forward lean (reduced as the
speed exceeds the target) and the ankle balances by shifting the centre of
pressure along the foot (toe-down = negative command moves the CoP forward).

Sign conventions (hopper.xml): leg joints hinge about -y, pitch about +y, so a
positive hip command pitches the torso nose-down, a positive knee command
extends the knee, and a positive ankle command lifts the toe.
"""

import numpy as np

# Tunable (searched by hl_control.harbor.tune):
PARAMS = {
    "pitch_target": 0.0101,
    "kv": 0.073,
    "v_target": 1.5745,
    "kfp": 0.0151,
    "ktv": 0.4091,
    "thrust": 0.9166,
    "ankle_thrust": 0.3814,
    "leg_crouch": -0.2858,
    "leg_fold": -0.2273,
    "kp_pitch_f": 2.446,
    "kp_thigh_f": 0.0895,
    "dx_target": 0.0022,
}

PARAM_BOUNDS = {
    "pitch_target": (-0.02, 0.12),
    "kv": (0.0, 0.25),
    "v_target": (0.4, 2.0),
    "kfp": (0.0, 0.6),
    "ktv": (0.0, 0.6),
    "thrust": (0.5, 1.0),
    "ankle_thrust": (-0.2, 0.5),
    "leg_crouch": (-0.7, -0.25),
    "leg_fold": (-0.75, -0.2),
    "kp_pitch_f": (0.2, 2.5),
    "kp_thigh_f": (0.0, 2.5),
    "dx_target": (-0.15, 0.05),
}

# Fixed gains:
KP_PITCH, KD_PITCH = 1.5, 0.4       # stance hip PD on pitch
KA, KAD, KAP = 1.0, 0.3, 0.3        # ankle CoP: position, vx, pitch-rate gains
KP_KNEE, KD_KNEE = 1.2, 0.1         # knee PD in crouch/absorb
KP_LEG_F, KD_LEG_F = 1.2, 0.1      # flight knee PD
KD_THIGH_F = 0.15                   # flight thigh rate damping
THIGH_TARGET = -0.22                # flight thigh target at zero speed error [rad]
PITCH_FLIGHT = 0.05                 # flight pitch set-point [rad]
KD_PITCH_F = 0.25                   # flight pitch rate gain
KP_FOOT, KD_FOOT = 0.6, 0.04        # flight ankle PD
FOOT_TARGET = 0.0                   # flight foot angle target [rad]
H_PANIC = 0.85                      # grounded height below which the knee extends fully [m]

_state = {"phase": "CROUCH", "step": -1}


def controller(obs: dict) -> np.ndarray:
    p = PARAMS
    if obs["step"] <= _state["step"]:
        _state["phase"] = "CROUCH"  # new episode
    _state["step"] = obs["step"]

    pitch, pitch_rate = obs["pitch"], obs["pitch_rate"]
    thigh, leg, foot = obs["thigh"], obs["leg"], obs["foot"]
    thigh_rate, leg_rate, foot_rate = obs["thigh_rate"], obs["leg_rate"], obs["foot_rate"]
    vx, vz = obs["x_velocity"], obs["z_velocity"]
    foot_z = obs["height"] + obs["foot_xz_rel"][1]
    grounded = obs["foot_contact"] or (foot_z < 0.12 and vz < 0.05)

    # --- phase transitions ---
    ph = _state["phase"]
    if ph == "CROUCH" and leg <= p["leg_crouch"] + 0.08:
        ph = "THRUST"
    elif ph == "THRUST" and not grounded:
        ph = "FLIGHT"
    elif ph == "THRUST" and leg > -0.03 and vz < 0.1:
        ph = "CROUCH"  # extension complete but no takeoff: re-crouch
    elif ph == "FLIGHT" and grounded:
        ph = "ABSORB"
    elif ph == "ABSORB" and vz >= -0.05:
        ph = "THRUST"
    _state["phase"] = ph

    # --- stance attitude (hip) and CoP balance (ankle) ---
    v_err = vx - p["v_target"]
    pt = float(np.clip(p["pitch_target"] - p["kv"] * v_err, -0.1, 0.1))
    u_hip_st = -KP_PITCH * (pitch - pt) - KD_PITCH * pitch_rate
    dx = -obs["foot_xz_rel"][0]  # torso ahead of foot centre
    u_ank_st = -KA * (dx - p["dx_target"]) - KAD * vx - KAP * pitch_rate

    if ph == "CROUCH" or ph == "ABSORB":
        u_hip = u_hip_st
        u_knee = KP_KNEE * (p["leg_crouch"] - leg) - KD_KNEE * leg_rate
        if obs["height"] < H_PANIC:  # sinking toward the 0.7 m health limit
            u_knee = 1.0
        u_ankle = u_ank_st
    elif ph == "THRUST":
        u_hip = u_hip_st
        u_knee = float(np.clip(p["thrust"] - p["ktv"] * max(0.0, v_err), 0.3, 1.0))
        u_ankle = p["ankle_thrust"] - KAP * pitch_rate
    else:  # FLIGHT
        u_knee = KP_LEG_F * (p["leg_fold"] - leg) - KD_LEG_F * leg_rate
        thigh_t = np.clip(THIGH_TARGET + p["kfp"] * v_err, -0.6, -0.02)
        u_hip = (-p["kp_pitch_f"] * (pitch - PITCH_FLIGHT) - KD_PITCH_F * pitch_rate
                 + p["kp_thigh_f"] * (thigh_t - thigh) - KD_THIGH_F * thigh_rate)
        u_ankle = KP_FOOT * (FOOT_TARGET - foot) - KD_FOOT * foot_rate

    return np.clip(np.array([u_hip, u_knee, u_ankle]), -1.0, 1.0)
