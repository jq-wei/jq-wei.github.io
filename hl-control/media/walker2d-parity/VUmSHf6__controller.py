"""Open-loop central-pattern-generator gait with joint PD tracking and pitch feedback.

Each joint of the right leg follows ref = bias + a1*sin(w t + p1) + a2*sin(2 w t + p2) (hip, knee,
ankle); the left leg uses the same waveforms shifted by phase `loff`. Amplitudes ramp up over
`ramp` seconds from standing. Motor command = kp*(ref - q) - kd*qdot (per-joint kp), where the
hip/knee/ankle references also get torso-pitch feedback (kpit*pitch + kdpit*pitch_rate on the hip,
kpitk/kpita proportional terms on knee/ankle). Parameters were found by CMA-ES on the visible seeds.
"""
import numpy as np

PARAMS = {
    "w": 14.64485397632202,
    "kd": 0.2028524402631859,
    "ramp": 1.7998643299401031,
    "loff": 3.0983937036123566,
    "kpit": -0.040019867951828214,
    "kdpit": 0.006075405229946096,
    "kpitk": -0.12209997628338165,
    "kpita": -0.9974513585970057,
    "hb": -0.040849342748703776,
    "ha1": 0.4525106340903985,
    "hp1": 1.117637184170781,
    "ha2": 0.544426961445339,
    "hp2": 1.3304763219217826,
    "hkp": 10.186099251637186,
    "kb": 0.413747467732857,
    "ka1": 0.5159815251104942,
    "kp1": 0.3440359014798964,
    "ka2": 0.14769394037118416,
    "kp2": 0.42863305251982275,
    "kkp": 15.327491396174736,
    "ab": 0.5764404887337321,
    "aa1": 0.3755004446712088,
    "ap1": 0.5168025979718354,
    "aa2": 0.30861305229266595,
    "ap2": 1.131882231780355,
    "akp": 20.318366037158103
}
J = ("h", "k", "a")


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    t = obs["time"]
    ph = P["w"] * t
    r = min(1.0, t / max(P["ramp"], 1e-3))
    q = obs["joint_angles"]
    qd = obs["joint_rates"]
    ref = np.zeros(6)
    kp = np.zeros(6)
    for side, off in ((0, 0.0), (3, P["loff"])):
        for i, j in enumerate(J):
            ref[side + i] = P[j + "b"] + r * (P[j + "a1"] * np.sin(ph + off + P[j + "p1"])
                                              + P[j + "a2"] * np.sin(2 * (ph + off) + P[j + "p2"]))
            kp[side + i] = P[j + "kp"]
    fb = obs["pitch"] * np.array([P["kpit"], P["kpitk"], P["kpita"]] * 2) \
        + obs["pitch_rate"] * np.array([P["kdpit"], 0, 0] * 2)
    ref += fb
    return np.clip(kp * (ref - q) - P["kd"] * qd, -1, 1)
