"""SIMBICON-style walking controller for the MuJoCo Humanoid (Humanoid-v5), ~1.45 m/s.

Control law (all torques are computed in N m and divided by the motor gear, then clipped
to [-0.4, 0.4]):

* Gait state machine: one leg is the stance leg, the other swings; they swap every
  T_STEP seconds, or earlier (after half a step) when the swing foot touches the floor.
  The result is a fast, short-stepped shuffle (about 13 touchdowns per second).
* Swing hip (sagittal), Raibert/SIMBICON foot placement:
      hip_y* = -(SW0 + CD * d + CV * (v - v_ref)) - WORLD * pitch
  d = mass centre ahead of the stance foot, v = mass-centre speed; v_ref ramps at ACC
  towards V_DES. Placing the foot further ahead when the body is ahead/fast catches it.
* Swing hip (lateral), same idea sideways: hip_x* = -(ABD0 + CDL * d_out + CVL * v_out)
  (+ WORLD * roll), where d_out, v_out are the mass-centre offset/velocity towards the
  swing side; the swing hip_z is held at 0.
* Knees: the swing knee folds to KNEE_SW for the first SW_LIFT of the step (ground
  clearance), then extends to KNEE_ST; the stance knee holds KNEE_ST.
* Torso: a world-frame PD on torso pitch (target LEAN + LEAN_V * v_ref + VEL_FB *
  (v_ref - v); gains KT, DT), roll (KR, DR) and yaw (KY, DY) is applied through the
  stance hip, minus the swing-hip torque (SIMBICON's stance-hip rule), so the torso is
  balanced on the stance leg.
* Abdomen and arms: PD to the zero posture (KP_AB/KD_AB, KP_ARM/KD_ARM).
* KP_*/KD_* are the joint PD gains (N m/rad, N m s/rad) of the hips (y and x/z) and knees.

The structure is hand-written; the numerical gains were tuned by black-box (random
perturbation) search on simulated episodes, with no learning in the controller itself.
Checked on 120 standard_v0 seeds: all alive, 21-23 m travelled, return about 6350.
"""
import numpy as np

P = dict(
    T_STEP=0.07321959841812521,
    LEAN=-0.00523620315151235,
    KT=299.89123526001924,
    DT=12.140987865714763,
    KR=236.7711998445044,
    DR=12.182309623958725,
    KY=8.063132392289129,
    DY=6.205104409204244,
    SW0=0.10328359224505575,
    CD=0.6304652676993937,
    CV=0.20440844181756787,
    ABD0=0.03749440875778535,
    CDL=0.574091344803739,
    CVL=0.5476667854046534,
    KNEE_SW=-1.4422496351143246,
    KNEE_ST=-0.19282763259962954,
    SW_LIFT=0.3775299895478374,
    KP_HIP=243.86235239116746,
    KD_HIP=2.8008112215013616,
    KP_KNEE=268.5412190370316,
    KD_KNEE=2.512551371901512,
    KP_HX=281.6040393836688,
    KD_HX=4.184331514651212,
    KP_AB=203.87351640648615,
    KD_AB=8.676323371205402,
    KP_ARM=44.61376410067575,
    KD_ARM=4.500935389933849,
    VEL_FB=0.06933786331384363,
    WORLD=0.0709860669150648,
    V_DES=1.4,
    ACC=0.03388261291453039,
    LEAN_V=-0.04344031280954469,
)

GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
# motor indices (motor order of the XML)
AY, AZ, AX = 0, 1, 2
R = dict(hx=3, hz=4, hy=5, kn=6)
L = dict(hx=7, hz=8, hy=9, kn=10)

_s = {"stance": 1, "t0": 0.0}   # stance 0 = right, 1 = left


def controller(obs):
    p = P
    t = obs["time"]
    if obs["step"] == 0:
        _s.update(stance=1, t0=0.0)
    q, qd = obs["joint_angles"], obs["joint_rates"]
    phase = (t - _s["t0"]) / p["T_STEP"]
    swing_contact = obs["foot_contact"] if _s["stance"] == 1 else obs["foot_left_contact"]
    if phase >= 1.0 or (phase > 0.5 and swing_contact):
        _s["stance"] = 1 - _s["stance"]
        _s["t0"] = t
        phase = 0.0
    st, sw = (R, L) if _s["stance"] == 0 else (L, R)
    s_out = 1.0 if _s["stance"] == 0 else -1.0    # outward (+y) direction of the swing leg: left swing -> +1
    tau = np.zeros(17)

    def pd(i, target, kp, kd):
        tau[i] = kp * (target - q[i]) - kd * qd[i]

    # posture of abdomen and arms
    pd(AY, 0.0, p["KP_AB"], p["KD_AB"]); pd(AZ, 0.0, p["KP_AB"], p["KD_AB"]); pd(AX, 0.0, p["KP_AB"], p["KD_AB"])
    for i in range(11, 17):
        pd(i, 0.0, p["KP_ARM"], p["KD_ARM"])

    vx, vy = obs["com_velocity_xy"]
    stf = obs["foot_xyz_rel"] if _s["stance"] == 0 else obs["foot_left_xyz_rel"]
    dx, dy = -stf[0], -stf[1]        # COM relative to stance foot
    # swing leg
    v_ref = min(p["V_DES"], p["ACC"] * t)
    hy_t = -(p["SW0"] + p["CD"] * dx + p["CV"] * (vx - v_ref))
    hy_t -= p["WORLD"] * obs["pitch"]          # swing-leg angle held in the world frame
    pd(sw["hy"], hy_t, p["KP_HIP"], p["KD_HIP"])
    hx_t = -(p["ABD0"] + p["CDL"] * s_out * dy + p["CVL"] * s_out * vy)
    hx_t += p["WORLD"] * obs["roll"] * (-1.0 if _s["stance"] == 1 else 1.0)
    pd(sw["hx"], np.clip(hx_t, -0.43, 0.08), p["KP_HX"], p["KD_HX"])
    pd(sw["hz"], 0.0, p["KP_HX"], p["KD_HX"])
    pd(sw["kn"], p["KNEE_SW"] if phase < p["SW_LIFT"] else p["KNEE_ST"], p["KP_KNEE"], p["KD_KNEE"])
    pd(st["kn"], p["KNEE_ST"], p["KP_KNEE"], p["KD_KNEE"])
    # torso via stance hip
    w = obs["root_angular_velocity"]
    lean = p["LEAN"] + p["LEAN_V"] * v_ref + p["VEL_FB"] * (v_ref - vx)
    Ty = -p["KT"] * (obs["pitch"] - lean) - p["DT"] * w[1]
    Tx = -p["KR"] * obs["roll"] - p["DR"] * w[0]
    Tz = -p["KY"] * obs["yaw"] - p["DY"] * w[2]
    tau[st["hy"]] = -Ty - tau[sw["hy"]]
    if _s["stance"] == 0:   # right stance: right hip_x axis +x, left -x
        tau[st["hx"]] = -Tx + tau[sw["hx"]]
        tau[st["hz"]] = -Tz + tau[sw["hz"]]
    else:
        tau[st["hx"]] = Tx + tau[sw["hx"]]
        tau[st["hz"]] = Tz + tau[sw["hz"]]
    return np.clip(tau / GEAR, -0.4, 0.4)
