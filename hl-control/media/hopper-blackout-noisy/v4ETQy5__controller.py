"""Hopper controller: forward bounce with a toe push-off, robust to sensing blackouts.

Gait (found by hand, gains refined by random search on the visible seeds):
  Stance (foot_contact): the hip is a PD servo on torso pitch (kp_pitch, kd_pitch) about a target
    pitch0 - k_v * (x_velocity - v_des), so the body leans back when it runs faster than v_des; the knee
    is a stiff PD strut to kn_d (kp_kn, kd_kn); the ankle acts as a 'leg spring': it plantarflexes
    (negative command, toe push) in proportion to how far the thigh trails behind th_d (kp_ank, kd_ank),
    clipped to [u_ft_min, u_ft_max]. Because the whole body is ahead of the trailing foot, the toe push
    launches it forward and up. When |pitch| exceeds p_em an extra hip gain k_em pulls it back.
  Flight: PD (kp_fl, kd_fl) of thigh/knee to th_fl/kn_fl and of the ankle to the angle that lands the
    foot flat (clipped to +-ft_lim), plus a weak hip pitch servo (kp_pitch_fl, kd_pitch_fl) that uses the
    heavy leg as a reaction wheel.
  Start: knee and ankle commands ramp in over t_start so the initial drop does not excite the pitch.
Blackouts: joint angles must move as the measured rates say; when the bo_k-step prediction disagrees
  with the reading by more than bo_thresh the sensors are held (the readings are stale). For bo_hold
  further steps the law runs on dead-reckoned angles/pitch (last rates integrated over the elapsed time),
  the height is propagated ballistically and touchdown is assumed below h_land (bo_mode=1; bo_mode=0 just
  repeats the last command). The history is cleared afterwards so the stale window cannot re-trigger.
"""
import numpy as np

P = dict(
    th_d=-0.1219,
    kn_d=-0.1777,
    kp_kn=2.562,
    kd_kn=0.08,
    pitch0=-0.0288,
    k_v=0.0186,
    v_des=0.2405,
    kp_pitch=0.9989,
    kd_pitch=0.0828,
    kp_ank=1.459,
    kd_ank=0.0296,
    u_ft_min=-1.0651,
    u_ft_max=0.2901,
    kn_fl=-0.1865,
    th_fl=-0.0784,
    kp_fl=np.array([0.2, 1.5, 1.0]),
    kd_fl=np.array([0.02, 0.08, 0.05]),
    ft_lim=0.2098,
    bo_thresh=0.05,
    bo_k=3,
    bo_hold=9,
    bo_mode=1,
    h_land=1.18,
    kp_pitch_fl=0.3419,
    kd_pitch_fl=0.0276,
    k_em=0.7351,
    p_em=0.15,
    t_start=0.4266,
)
S = {}

def reset():
    S.clear()

def controller(obs):
    if not S or obs["step"] == 0:
        S.update(last_u=np.zeros(3), hist=[], landed=False)
    t, dt = obs["time"], obs["dt"]
    q, qd = obs["joint_angles"].copy(), obs["joint_rates"].copy()
    S["hist"].append((q, qd))
    if len(S["hist"]) > P["bo_k"] + 1:
        S["hist"].pop(0)
    bo = False
    if len(S["hist"]) == P["bo_k"] + 1:
        q0, _ = S["hist"][0]
        pred = q0 + dt * sum(0.5 * (S["hist"][i][1] + S["hist"][i + 1][1]) for i in range(P["bo_k"]))
        bo = bool(np.max(np.abs(q - pred)) > P["bo_thresh"])
    if S.get("bo_left", 0) > 0:
        S["bo_left"] -= 1
        if S["bo_left"] == 0:
            S["hist"] = []
        bo = True
    elif bo:
        S["bo_left"] = int(P["bo_hold"]); S["bo_tau"] = 1.5 * dt
    th, kn, ft = q; thr, knr, ftr = qd
    p, pr, xv = obs["pitch"], obs["pitch_rate"], obs["x_velocity"]
    contact = bool(obs["foot_contact"])
    if bo:
        if P["bo_mode"] == 0:
            return S["last_u"]
        tau = S["bo_tau"]; S["bo_tau"] += dt
        th, kn, ft = np.clip(q + qd * tau, [-2.6, -2.6, -0.78], [0.0, 0.0, 0.78])
        p = p + pr * tau
        if not contact:
            h_est = obs["height"] + obs["z_velocity"] * tau - 4.905 * tau ** 2
            contact = h_est < P["h_land"]
    if contact:
        S["landed"] = True
    if contact or not S["landed"]:
        pitch_d = float(np.clip(P["pitch0"] - P["k_v"] * (xv - P["v_des"]), -0.15, 0.1))
        u_th = -(P["kp_pitch"] * (p - pitch_d) + P["kd_pitch"] * pr)
        u_kn = P["kp_kn"] * (P["kn_d"] - kn) - P["kd_kn"] * knr
        u_ft = float(np.clip(P["kp_ank"] * (th - P["th_d"]) + P["kd_ank"] * thr, P["u_ft_min"], P["u_ft_max"]))
        if abs(p) > P["p_em"]:
            u_th -= P["k_em"] * (p - np.sign(p) * P["p_em"])
        ramp = min(1.0, t / P["t_start"])
        u = np.array([u_th, ramp * u_kn, ramp * u_ft])
    else:
        ft_fl = float(np.clip(p - th - kn, -P["ft_lim"], P["ft_lim"]))
        q_fl = np.array([P["th_fl"], P["kn_fl"], ft_fl])
        u = P["kp_fl"] * (q_fl - q) - P["kd_fl"] * qd
        pitch_d = float(np.clip(P["pitch0"] - P["k_v"] * (xv - P["v_des"]), -0.15, 0.1))
        u[0] += -(P["kp_pitch_fl"] * (p - pitch_d) + P["kd_pitch_fl"] * pr)
        if abs(p) > P["p_em"]:
            u[0] -= P["k_em"] * (p - np.sign(p) * P["p_em"])
    u = np.clip(u, -1, 1)
    S["last_u"] = u
    return u
