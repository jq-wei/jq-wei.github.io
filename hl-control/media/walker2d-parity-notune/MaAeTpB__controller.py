"""Open-loop central-pattern-generator gait with joint PD tracking and body feedback.

A phase oscillator (F Hz) drives each leg's thigh, knee and ankle targets as
offset + amplitude*(sin(phase+shift) + a second harmonic); the left leg runs half a cycle
behind the right. Amplitudes ramp up over RAMP s from standing. Motor command =
KP*(target - q) - KD*qdot, clipped to [-1, 1]. Both thigh targets get
PK*pitch + PD*pitch_rate + VK*(x_velocity - 2) so the torso lean and speed are regulated
through the hips. Parameters were tuned by derivative-free search (CMA-ES) in simulation.
"""
import numpy as np

P = [2.022742042347248, 4.702404960538872, 0.13852717000769565, -0.47530382839293894, 1.1245510072820961, -0.6763630975755306, -0.957820932774808, 1.0733331933997223, 2.3989752222630405, 0.5699899323534161, 0.38283479485947775, -1.0649611543848614, -1.485976023953991, -0.05490426667410525, 0.8811914984766057, -0.23772421794294046, 0.17218244812391328, -0.2586144439805451, 0.09929786351097991, -0.49596072163389743, 0.22076825598968514, -0.3236176136520344]
(F, KP, KD, TO, TA, TS, LO, LA, LS, FO, FA, FS, PK, PD, RAMP, VK,
 A2T, S2T, A2L, S2L, A2F, S2F) = P


def controller(obs: dict) -> np.ndarray:
    t = obs["time"]
    ph = 2 * np.pi * F * t
    a = min(1.0, t / RAMP)
    tgt = np.zeros(6)
    for k in range(2):
        p = ph + k * np.pi
        tgt[3 * k] = TO + a * (TA * np.sin(p + TS) + A2T * np.sin(2 * p + S2T))
        tgt[3 * k + 1] = LO + a * (LA * np.sin(p + LS) + A2L * np.sin(2 * p + S2L))
        tgt[3 * k + 2] = FO + a * (FA * np.sin(p + FS) + A2F * np.sin(2 * p + S2F))
    tgt[[0, 3]] += PK * obs["pitch"] + PD * obs["pitch_rate"] + VK * (obs["x_velocity"] - 2.0)
    return np.clip(KP * (tgt - obs["joint_angles"]) - KD * obs["joint_rates"], -1, 1)
