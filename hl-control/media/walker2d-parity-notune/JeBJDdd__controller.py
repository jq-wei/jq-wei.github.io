"""Walker2d gait controller: clock-driven joint targets + PD + torso-pitch feedback.

Control law (125 Hz, one command unit = 100 N m):
  * A clock phase phi = 2*pi*freq*t drives both legs, the left leg half a cycle behind.
    At each touchdown the clock is pulled a fraction k_reset toward the nominal
    touchdown phase phi_td (right foot; phi_td + pi left), synchronising it with contact.
  * Each joint tracks a periodic target c + A*cos(phi + d) (+ B*cos(2*phi + e) for the
    thigh and knee) with a PD law u = kp*(q_des - q) - kd*qdot (kp in command/rad, kd in
    command/(rad/s)); thigh/knee targets are clipped to <= 0 (joint ranges), ankles to
    +-45 deg. The swing-leg (no contact) thigh target is shifted by k_fp*(v - v_des)
    (velocity-dependent foot placement).
  * Torso attitude: while a foot is on the ground its hip adds
    -kpp*(pitch - pitch_des_eff) - kdp*pitch_rate with pitch_des_eff = pitch_des -
    kv*(v - v_des) (lean further forward when slow). ff_a is a constant toe-down
    push-off on the ankle of any foot in contact.
Parameters were tuned by a derivative-free evolution strategy (CMA-ES) on seeds 100-139.
"""
import math
import numpy as np

P_NAMES = ['freq', 'pitch_des', 't_c', 't_A', 't_d', 'k_c', 'k_A', 'k_d', 'a_c', 'a_A', 'a_d', 't_B', 't_e', 'k_B', 'k_e', 'kp_t', 'kp_k', 'kp_a', 'kd_t', 'kd_k', 'kd_a', 'kpp', 'kdp', 'kv', 'v_des', 'ff_a', 'k_fp', 'k_reset', 'phi_td', 'ff_h', 'ff_k']
P_VALUES = [2.232722, 0.635477, -0.127962, 0.870348, -0.324806, -1.028621, 0.273591, 2.432325, 0.765904, 0.152797, -0.878412, -0.283722, 0.757504, -0.018606, -0.723470, 3.137487, 2.585648, 3.791250, 0.052246, 0.098646, 0.029952, 1.398265, 0.385637, 0.111579, 2.940086, -0.370470, -0.096703, -0.059159, 2.067334, -0.248017, 0.307098]

ANK = 0.78
TWO_PI = 2 * math.pi


def make_controller(p):
    p = dict(zip(P_NAMES, p))
    kp = np.array([p["kp_t"], p["kp_k"], p["kp_a"]] * 2)
    kd = np.array([p["kd_t"], p["kd_k"], p["kd_a"]] * 2)
    cos = math.cos
    t_c, t_A, t_d, t_B, t_e = p["t_c"], p["t_A"], p["t_d"], p["t_B"], p["t_e"]
    k_c, k_A, k_d, k_B, k_e = p["k_c"], p["k_A"], p["k_d"], p["k_B"], p["k_e"]
    a_c, a_A, a_d = p["a_c"], p["a_A"], p["a_d"]
    k_reset = min(max(p["k_reset"], 0.0), 1.0)
    state = {"off": 0.0, "cR": True, "cL": True}

    def targets(phi, dth):
        th = t_c + t_A * cos(phi + t_d) + t_B * cos(2 * phi + t_e) + dth
        kn = k_c + k_A * cos(phi + k_d) + k_B * cos(2 * phi + k_e)
        an = a_c + a_A * cos(phi + a_d)
        return (min(th, 0.0), min(kn, 0.0), max(-ANK, min(ANK, an)))

    def controller(obs):
        if obs["step"] == 0:
            state["off"] = 0.0
            state["cR"], state["cL"] = True, True
        cR, cL = obs["foot_contact"], obs["foot_left_contact"]
        phi = TWO_PI * p["freq"] * obs["time"] + state["off"]
        # touchdown phase reset
        for c, prev, target in ((cR, state["cR"], p["phi_td"]), (cL, state["cL"], p["phi_td"] + math.pi)):
            if c and not prev:
                err = (target - phi + math.pi) % TWO_PI - math.pi
                state["off"] += k_reset * err
                phi += k_reset * err
        state["cR"], state["cL"] = cR, cL
        v = obs["x_velocity"]
        verr = v - p["v_des"]
        dfp = p["k_fp"] * verr
        q = obs["joint_angles"]
        qd = obs["joint_rates"]
        qdes = np.array(targets(phi, 0.0 if cR else dfp) + targets(phi + math.pi, 0.0 if cL else dfp))
        u = kp * (qdes - q) - kd * qd
        pitch_des = p["pitch_des"] - p["kv"] * verr
        corr = -p["kpp"] * (obs["pitch"] - pitch_des) - p["kdp"] * obs["pitch_rate"]
        if cR:
            u[0] += corr + p["ff_h"]
            u[1] += p["ff_k"]
            u[2] += p["ff_a"]
        if cL:
            u[3] += corr + p["ff_h"]
            u[4] += p["ff_k"]
            u[5] += p["ff_a"]
        return np.clip(u, -1.0, 1.0)

    return controller


controller = make_controller(P_VALUES)
