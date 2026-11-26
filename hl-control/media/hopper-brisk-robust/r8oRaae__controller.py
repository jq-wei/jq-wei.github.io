"""Hopper controller: phase machine + attitude servo at the hip + ankle centre-of-pressure leg-angle loop.

Control law
-----------
The leg is ~10x the torso's inertia, so a hip torque in flight mostly rotates the torso,
not the leg. The roles are therefore split:

* hip (both phases): PD servo of the torso pitch to ``pd`` (``a1f``/``a2f`` in flight,
  ``a1s``/``a2s`` in stance); in flight a weak extra term ``kth`` pulls the thigh to ``thf``.
* ankle in stance: regulates the world leg angle ``psi = pitch - thigh`` (leg trailing
  behind the torso is positive). Pressing the toe down moves the centre of pressure
  forward and gives the body nose-up angular momentum, which swings the leg forward during
  the next flight. Without it the thrust passes behind the COM and the leg drifts back
  stride after stride until the robot sinks. Feedback:
  ``e = b1 (psi - psi0 + kvs max(0, v - vs)) + b2 dpsi/dt + ka (v - vt)``;
  the ``kvs`` term is a speed governor (it lowers the leg-angle setpoint above ``vs`` m/s,
  which keeps the weak-motor plant from running away and saturating the ankle).
* knee and ankle produce the hop:
    - FLIGHT (no contact): knee PD to ``kfl``, ankle PD (gain ``kfa``) to ``ff``.
    - COMPRESSION (touchdown until z-velocity >= ``zth``): knee PD to ``kns``,
      ankle PD to ``fs`` plus ``e``.
    - THRUST (until two steps without contact): knee push ``T + bT (psi - psi0)``,
      ankle push ``F + e`` (toe down, the main source of lift).

The gains were found by a CMA-ES parameter search on the visible seeds, first on the
nominal plant, then jointly over the nominal and the four perturbed plants (mass, friction,
gear, gravity), and checked on unseen seeds.
"""

import numpy as np

PARAMS = {
    # torso pitch servo (hip)
    "pd": -0.0821, "a1s": 2.0534, "a2s": 0.247, "a1f": 6.9326, "a2f": 0.0602,
    "thf": 0.1128, "kth": 0.4592,
    # knee: thrust, compression target, flight target
    "T": 0.7805, "bT": 0.1413, "kns": 0.033, "kfl": -0.0834,
    # ankle: push, compression target, flight target and gain
    "F": -0.8946, "fs": 0.04, "ff": 0.1519, "kfa": 0.4546,
    # stance leg-angle / speed loop through the ankle
    "b1": -3.2834, "b2": -0.1851, "psi0": 0.1658, "ka": -0.4905, "vt": 3.315,
    "kvs": 0.1, "vs": 2.2,
    # compression -> thrust switch (z-velocity threshold, m/s)
    "zth": 0.0981,
}

FLIGHT, COMPRESSION, THRUST = 0, 1, 2
_state = {"phase": FLIGHT, "n": 0, "air": 0}


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    S = _state
    if obs["step"] == 0:
        S.update(phase=FLIGHT, n=0, air=0)

    p, pr = obs["pitch"], obs["pitch_rate"]
    th, kn, ft = obs["joint_angles"]
    thd, knd, ftd = obs["joint_rates"]
    v = obs["x_velocity"]
    contact = obs["foot_contact"]
    psi, psid = p - th, pr - thd
    pd = P["pd"]

    # phase transitions
    S["air"] = 0 if contact else S["air"] + 1
    if S["phase"] == FLIGHT and contact:
        S["phase"], S["n"] = COMPRESSION, 0
    elif S["phase"] == COMPRESSION and S["n"] > 2 and obs["z_velocity"] >= P["zth"]:
        S["phase"], S["n"] = THRUST, 0
    elif S["phase"] == THRUST and S["air"] >= 2:
        S["phase"], S["n"] = FLIGHT, 0
    S["n"] += 1

    u = np.zeros(3)
    if S["phase"] == FLIGHT:
        u[0] = -(P["a1f"] * (p - pd) + P["a2f"] * pr) + P["kth"] * (P["thf"] - th)
        u[1] = (P["kfl"] - kn) - 0.05 * knd
        u[2] = P["kfa"] * (P["ff"] - ft) - 0.05 * ftd
    else:
        u[0] = -(P["a1s"] * (p - pd) + P["a2s"] * pr)
        overspeed = max(0.0, v - P["vs"])
        e = P["b1"] * (psi - P["psi0"] + P["kvs"] * overspeed) + P["b2"] * psid + P["ka"] * (v - P["vt"])
        if S["phase"] == COMPRESSION:
            u[1] = (P["kns"] - kn) - 0.03 * knd
            u[2] = (P["fs"] - ft) - 0.05 * ftd + e
        else:
            u[1] = P["T"] + P["bT"] * (psi - P["psi0"])
            u[2] = P["F"] + e
    return np.clip(u, -1.0, 1.0)
