"""Event-driven (SIMBICON-style) walking controller for the MuJoCo Humanoid.

Control law
-----------
* Two alternating states: right-stance / left-stance.  A state ends when the swing foot
  touches down after at least `fc` of the nominal duration `Ts`, or when `Ts` elapsed.
* Swing leg: hip pitch targets a forward angle (`hs0`) corrected by the centre-of-mass
  offset from the stance foot (`cdx`) and forward speed error (`cvx`); the knee flexes
  as a half sine (`kA`) over the state, hip abduction follows the lateral COM offset.
* Stance leg: hip pitch regulates torso pitch (PD on pitch / pitch rate); knee stays
  slightly bent; hip abduction regulates roll; hip rotation regulates yaw.
* Abdomen: PD to pitch/roll/yaw feedback; arms swing opposite to legs.
* Every joint uses tau = kp (q* - q) - kd qdot, divided by gear (clip +-0.4).
Parameters found by derivative-free search (scratch/es2.py).
"""

import numpy as np

SPEC = [
    ("Ts", 0.35, 0.03), ("fc", 0.5, 0.1), ("ramp", 0.3, 0.1), ("vdes", 1.2, 0.0),
    # swing leg
    ("hs0", -0.35, 0.1), ("cdx", 0.0, 0.3), ("cvx", 0.0, 0.1), ("ks0", -0.1, 0.1), ("kA", -0.8, 0.2),
    ("hxs0", -0.05, 0.05), ("cdy", 0.0, 0.3), ("cvy", 0.0, 0.1), ("hzs0", 0.0, 0.05),
    # stance leg
    ("ht0", 0.1, 0.1), ("kpt", 0.0, 0.5), ("kwt", 0.0, 0.1), ("pdes", 0.1, 0.05), ("kt0", -0.15, 0.1), ("ktA", 0.0, 0.2),
    ("hxt0", -0.03, 0.05), ("krt", 0.0, 0.5), ("kwxt", 0.0, 0.1), ("hzt0", 0.0, 0.05),
    # yaw
    ("kyaw", 0.0, 0.5), ("kwz", 0.0, 0.1), ("kyaw_sw", 0.0, 0.5), ("kwz_sw", 0.0, 0.1),
    # abdomen
    ("ay0", 0.0, 0.05), ("kap", 0.0, 0.5), ("kaw", 0.0, 0.1), ("kar", 0.0, 0.5), ("ax_off", 0.0, 0.05),
    ("kay", 0.0, 0.5), ("kawz", 0.0, 0.1),
    # arms
    ("sA", 0.0, 0.2), ("s0", 0.0, 0.1), ("e0", -0.3, 0.1),
    # log gains
    ("lkp_abd", np.log(100.0), 0.3), ("lkp_hx", np.log(80.0), 0.3), ("lkp_hz", np.log(60.0), 0.3),
    ("lkp_hy", np.log(200.0), 0.3), ("lkp_kn", np.log(150.0), 0.3), ("lkp_arm", np.log(10.0), 0.3),
    ("lkd_abd", np.log(5.0), 0.3), ("lkd_hx", np.log(4.0), 0.3), ("lkd_hz", np.log(3.0), 0.3),
    ("lkd_hy", np.log(8.0), 0.3), ("lkd_kn", np.log(6.0), 0.3), ("lkd_arm", np.log(0.5), 0.3),
]
NAMES = [s[0] for s in SPEC]
PARAMS = np.array([
    0.25291565504745334,
    0.215351837733621,
    0.07642696552127191,
    1.2,
    -0.3320599248451254,
    -2.156724867248599,
    -0.4772973155339244,
    -0.37858200794822155,
    0.47633253038707385,
    0.0013580083607235207,
    -0.256497392143009,
    -0.5306820409916478,
    0.09173990461854348,
    0.438146003840556,
    4.04254361090638,
    0.2622152521435011,
    0.20508891098285878,
    0.38439620877582953,
    0.5588176367742913,
    -0.17338019374091299,
    0.211170376136384,
    -0.17214722812999805,
    -0.11637084804609144,
    0.31676514748726237,
    0.09979366750243288,
    1.0862849318408772,
    0.1375235837982935,
    -0.06199147693054217,
    0.36658810106611184,
    0.0932225341753722,
    3.260493779900131,
    0.25207373317164783,
    -0.6552141589007423,
    0.009167180518845675,
    0.3165973869405263,
    0.021928090277113312,
    -0.3400496017426664,
    5.18991115060099,
    5.018163879209359,
    3.7850912458801105,
    4.861469822526365,
    4.760532190688534,
    1.2307289424839976,
    1.6737469513713283,
    1.3532593424373622,
    0.2854551002978214,
    1.032368430633436,
    1.8703540306617448,
    -0.9487825769818422,
])  # order = SPEC
GEAR = np.array([100, 100, 100, 100, 100, 300, 200, 100, 100, 300, 200, 25, 25, 25, 25, 25, 25], float)
GROUPS = ((slice(0, 3), "abd"), (slice(3, 4), "hx"), (slice(7, 8), "hx"), (slice(4, 5), "hz"),
          (slice(8, 9), "hz"), (slice(5, 6), "hy"), (slice(9, 10), "hy"), (slice(6, 7), "kn"),
          (slice(10, 11), "kn"), (slice(11, 17), "arm"))
_state = {"stance": 0, "t0": 0.0}


def controller(obs: dict) -> np.ndarray:
    P = dict(zip(NAMES, PARAMS))
    t = obs["time"]
    q, qd = obs["joint_angles"], obs["joint_rates"]
    if obs["step"] == 0:
        _state.update(stance=0, t0=0.0)
    amp = min(1.0, t / max(P["ramp"], 1e-3))
    contact = (bool(obs["foot_contact"]), bool(obs["foot_left_contact"]))
    st = _state["stance"]
    tau_s = t - _state["t0"]
    if tau_s >= P["Ts"] or (tau_s >= P["fc"] * P["Ts"] and contact[1 - st]):
        st = 1 - st
        _state.update(stance=st, t0=t)
        tau_s = 0.0
    sw = 1 - st
    u = min(tau_s / P["Ts"], 1.0)
    sgn = 1.0 if st == 0 else -1.0     # +1 right stance
    pitch, roll, yaw = obs["pitch"], obs["roll"], obs["yaw"]
    w = obs["root_angular_velocity"]
    vx, vy = obs["com_velocity_xy"]
    rel = (obs["foot_xyz_rel"], obs["foot_left_xyz_rel"])[st]
    dx = -rel[0]                       # COM ahead of stance foot
    dm = -sgn * rel[1]                 # COM medial of stance foot
    vm = sgn * vy
    rollm = -sgn * roll                # lean toward the medial side
    wxm = -sgn * w[0]
    vx_err = vx - P["vdes"]
    qt = np.zeros(17)
    bs, bt = 3 + 4 * sw, 3 + 4 * st
    ssw, sst = (1.0 if sw == 0 else -1.0), sgn
    # swing leg
    qt[bs + 2] = P["hs0"] + P["cdx"] * dx + P["cvx"] * vx_err
    qt[bs + 3] = min(P["ks0"] + P["kA"] * np.sin(np.pi * u), -0.03)
    qt[bs + 0] = P["hxs0"] + P["cdy"] * dm + P["cvy"] * vm
    qt[bs + 1] = P["hzs0"] + ssw * (P["kyaw_sw"] * yaw + P["kwz_sw"] * w[2])
    # stance leg
    qt[bt + 2] = P["ht0"] + P["kpt"] * (pitch - P["pdes"]) + P["kwt"] * w[1]
    qt[bt + 3] = min(P["kt0"] + P["ktA"] * u, -0.03)
    qt[bt + 0] = P["hxt0"] + P["krt"] * rollm + P["kwxt"] * wxm
    qt[bt + 1] = P["hzt0"] + sst * (P["kyaw"] * yaw + P["kwz"] * w[2])
    # abdomen
    qt[0] = P["ay0"] + P["kap"] * (pitch - P["pdes"]) + P["kaw"] * w[1]
    qt[2] = sgn * P["ax_off"] + P["kar"] * roll
    qt[1] = P["kay"] * yaw + P["kawz"] * w[2]
    for idx, s_ in ((11, 1.0), (14, -1.0)):
        qt[idx] = P["s0"] + amp * P["sA"] * sgn * s_
        qt[idx + 1] = P["s0"]
        qt[idx + 2] = P["e0"]
    qt[3:11] = q[3:11] + amp * (qt[3:11] - q[3:11]) if amp < 1.0 else qt[3:11]

    kp, kd = np.empty(17), np.empty(17)
    for sl, a in GROUPS:
        kp[sl] = np.exp(P["lkp_" + a])
        kd[sl] = np.exp(P["lkd_" + a])
    tau = kp * (qt - q) - kd * qd
    return np.clip(tau / GEAR, -0.4, 0.4)
