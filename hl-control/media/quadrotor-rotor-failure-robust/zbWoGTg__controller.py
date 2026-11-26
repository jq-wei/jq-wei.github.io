"""Quadrotor that survives the unannounced loss of one rotor (rotor-thrust level control).

One cascade is used before and after the failure; only the allocation of the torque to the rotors changes.

1. Observer. The attitude R is integrated from the gyro, velocity and position from the accelerometer rotated
   by R. Each tracker frame (30 ms late) is compared with the estimate stored for that instant; the residual
   corrects position, velocity, tilt (a tilt error shows up as a horizontal acceleration error) and yaw.
   The three gains follow from one observer pole OBS_POLE; LY is the yaw gain.
2. Position loop. a_cmd = -KP e - KD v - KI int(e) + g - (measured drag acceleration). The drag is what the
   accelerometer reads along body x, y (the rotors push along body z only), rotated to the world and
   low-passed (TAU_DRAG), so the wind is compensated without waiting for the integrator. The thrust per
   newton of command is identified from accel z (TAU_ETA); it absorbs mass, thrust coefficient, vertical drag.
3. Reduced attitude loop. Only the direction of one body-fixed axis c is steered towards a_cmd; yaw is free.
   c is body z while the vehicle does not spin and the principal axis of inertia (5 deg behind z, because
   of Ixz) once it spins (blend between R_BLEND0 and R_BLEND1), so that a steady spin about c needs no torque.
   Tilt error -> tilt-rate command (KN, limit W_MAX) -> body-rate feedback (KW, limit AL_MAX). On a spinning
   vehicle a tilt rate needs a precession torque at right angles: it is fed forward from the *commanded*
   tilt rate (never from the measured one, which would turn motor lag into negative damping), advanced in
   phase by the angle the body turns during the motor lag (TAU_LEAD or TAU_LEAD_SLOW) and half a step, and
   only for the part of the command that is not fixed in the body (low-pass T_DC), which would feed coning.
4. Identification on the healthy vehicle (0.3 s .. T_ARM). A small rotating torque dither (DITH_A, DITH_W) is
   added; two rotor models (motor lag 30 ms and 60 ms) predict the gyro increments, the one with the smaller
   residual energy is kept (it sets the phase advance and the detector's model), and a least-squares fit of
   the same data gives the torque effectiveness (thrust coefficient / mass ratio) used for all torques.
   The mean residual (TAU_B) is the torque of the centre-of-mass offset and is fed forward.
5. Failure detection (armed from T_ARM; the failure comes between 3 and 5 s). The leaky sum (TAU_Z) of the
   gyro-increment residual is projected on the four "rotor i lost" directions. Rotor i is declared dead when
   its score exceeds DET_W with a thrust deficit on the accelerometer (DET_A), or DET_W2 alone, and is at
   least twice any other score. Typical delay 13-20 ms (18-25 ms with slow motors).
6. After the failure the opposite rotor only pushes when the wanted torque asks for it (it cannot pull), the
   two adjacent rotors carry the weight and steer differentially. Their unbalanced yaw reaction spins the
   vehicle up, which makes the tilt controllable in every direction on average.
   Before the failure a yaw-rate loop (KR) keeps the healthy vehicle from turning.
"""

import math

import numpy as np

PARAMS = {
    "OBS_POLE": 7.0,     # rad/s, triple pole of the tracker-correction observer
    "LY": 0.25,          # yaw correction per tracker frame
    "KP": 5.0,           # 1/s^2 position gain
    "KD": 3.6,           # 1/s velocity gain
    "KI": 1.0,           # 1/s^3 integral gain (small: drag is fed forward)
    "KN": 7.0,           # 1/s tilt error -> tilt rate
    "KW": 20.0,          # 1/s body-rate loop
    "KR": 4.0,           # 1/s yaw-rate loop (healthy vehicle only)
    "TAU_LEAD": 0.04,    # s, phase-advance time constant when the motors are identified as fast (30 ms)
    "TAU_LEAD_SLOW": 0.068,  # s, the same when they are identified as slow (60 ms)
    "T_DC": 0.4,         # s, low-pass that extracts the body-fixed part of the tilt-rate command
    "R_BLEND0": 2.0,     # rad/s: below this spin rate the steered axis is body z ...
    "R_BLEND1": 6.0,     # rad/s: ... above this one it is the principal axis
    "W_MAX": 5.0,        # rad/s tilt-rate command limit
    "AL_MAX": 100.0,     # rad/s^2 limit of the rate-feedback angular acceleration
    "A_H_MAX": 5.0,      # m/s^2 horizontal acceleration command limit
    "TAU_DRAG": 0.15,    # s, two-stage low-pass of the measured drag acceleration
    "TAU_ETA": 0.3,      # s, thrust effectiveness identification
    "DITH_A": 6.0,       # rad/s^2, amplitude of the identification dither (healthy vehicle, 0.3 s .. T_ARM - 0.3 s)
    "DITH_W": 30.0,      # rad/s, its frequency
    "TAU_Z": 0.04,       # s, leak of the detection residual sum
    "TAU_B": 0.4,        # s, mean of the angular-acceleration residual (centre-of-mass offset torque)
    "DET_W": 0.06,       # rad/s detection threshold with accelerometer confirmation
    "DET_W2": 0.09,      # rad/s detection threshold without confirmation
    "DET_A": 0.25,       # m/s^2 thrust-deficit confirmation
    "T_ARM": 2.9,        # s, detector armed from here
}

G = 9.81
M_NOM = 1.325
DT = 0.005
FMAX = 13.0
J = np.array([[0.036652, 0.0, -0.0021], [0.0, 0.025412, 0.0], [-0.0021, 0.0, 0.060528]])
JINV = np.linalg.inv(J)
RX = np.array([-0.14, -0.14, 0.14, 0.14])
RY = np.array([-0.18, 0.18, 0.18, -0.18])
KAPPA = np.array([-0.0201, 0.0201, -0.0201, 0.0201])
MIX = np.vstack([np.ones(4), RY, -RX, KAPPA])          # [T, tx, ty, tz] = MIX @ f
MIXINV = np.linalg.inv(MIX)
TQ = MIX[1:4]                                           # torque rows
TAU_HYP = (0.03, 0.06)                                  # motor lag hypotheses [s]: fast (nominal), slow
LAG2 = np.array([[1.0 - math.exp(-DT / TAU_HYP[0])], [1.0 - math.exp(-DT / TAU_HYP[1])]])
JT2 = (JINV @ TQ)[:2].T                                 # rotor thrusts -> (p, q) acceleration of the nominal vehicle
# direction of the body-rate jump (p, q) when rotor i is lost
_U = []
for _i in range(4):
    _d = JINV[:2, :2] @ np.array([-RY[_i], RX[_i]])
    _U.append(_d / np.linalg.norm(_d))
U_FAIL = np.array(_U)
# three-rotor allocation: torque = delta * w_a + f_o * w_o
ALLOC3 = []
for _d in range(4):
    _a, _o, _b = (_d + 1) % 4, (_d + 2) % 4, (_d + 3) % 4
    _W = np.array([[RY[_a], RY[_o]], [-RX[_a], -RX[_o]]])
    ALLOC3.append((_a, _b, _o, np.linalg.inv(_W)))
I3 = np.eye(3)
PHI_P = 0.5 * math.atan2(2.0 * 0.0021, 0.060528 - 0.036652)   # tilt of the principal axis from body z (towards -x)


def _expm(w):
    th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    K = np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])
    if th2 < 1e-12:
        return I3 + K
    th = math.sqrt(th2)
    return I3 + (math.sin(th) / th) * K + ((1.0 - math.cos(th)) / th2) * (K @ K)


def _cross(a, b):
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


class _S:
    pass


S = _S()
S.ready = False


def _reset(obs):
    P = PARAMS
    yaw = float(obs["tracker_yaw"])
    c, s = math.cos(yaw), math.sin(yaw)
    S.R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    S.p = np.asarray(obs["tracker_pos"], dtype=float).copy()
    S.v = np.zeros(3)
    S.w_prev = np.asarray(obs["gyro"], dtype=float).copy()
    S.aw_prev = np.zeros(3)
    S.hp = np.tile(S.p, (8, 1))
    S.hy = np.full(8, yaw)
    S.fm2 = np.full((2, 4), M_NOM * G / 4.0)   # rotor thrusts predicted with each motor-lag hypothesis
    S.hyp = 0                                  # index of the hypothesis in use
    S.E = [0.0, 0.0]                           # residual energy of each hypothesis
    S.z2 = np.zeros((2, 2))
    S.x2 = np.zeros((2, 2))                    # (p, q) acceleration predicted from the rotors, per hypothesis
    S.gyr = np.zeros(2)                        # gyroscopic part of the (p, q) acceleration
    S.num = np.zeros(2)
    S.den = np.zeros(2)
    S.eta_t = 1.0                              # torque effectiveness (thrust coefficient / mass ratio)
    S.eta = 1.0
    S.integ = np.zeros(3)
    S.d1 = np.zeros(3)
    S.d2 = np.zeros(3)
    S.zb = np.zeros(2)
    S.raz = 0.0
    S.failed = -1
    S.wdc = [0.0, 0.0]
    S.t_fail = -1.0
    lam, ts = P["OBS_POLE"], 0.02
    S.LP = min(0.9, 3.0 * lam * ts)
    S.LV = 3.0 * lam * lam * ts
    S.LA = lam ** 3 * ts / G
    S.lamz = math.exp(-DT / P["TAU_Z"])
    S.ready = True


def controller(obs: dict) -> np.ndarray:
    P = PARAMS
    k = int(obs["step"])
    if k == 0 or not S.ready:
        _reset(obs)
    t = k * DT
    w = np.asarray(obs["gyro"], dtype=float)
    acc = np.asarray(obs["accel"], dtype=float)
    target = np.asarray(obs["target_pos"], dtype=float)

    # ---------- observer ----------
    if k > 0:
        S.R = S.R @ _expm(0.5 * (S.w_prev + w) * DT)
    R = S.R
    # specific force at the body origin (IMU 2 cm above it: remove the centripetal part)
    fb = np.array([acc[0] - 0.02 * w[0] * w[2], acc[1] - 0.02 * w[1] * w[2], acc[2] + 0.02 * (w[0] * w[0] + w[1] * w[1])])
    aw = R @ fb
    aw[2] -= G
    if k > 0:
        am = 0.5 * (aw + S.aw_prev)
        S.p = S.p + S.v * DT + am * (0.5 * DT * DT)
        S.v = S.v + am * DT
    S.aw_prev = aw
    yaw = math.atan2(R[1, 0], R[0, 0])
    S.hp[k % 8] = S.p
    S.hy[k % 8] = yaw
    if k > 0 and obs["tracker_fresh"]:
        j = max(k - 6, 0) % 8
        res = np.asarray(obs["tracker_pos"], dtype=float) - S.hp[j]
        rn = math.sqrt(res[0] * res[0] + res[1] * res[1] + res[2] * res[2])
        if rn > 0.5:                       # never let one frame throw the estimate
            res = res * (0.5 / rn)
        yres = (float(obs["tracker_yaw"]) - S.hy[j] + math.pi) % (2.0 * math.pi) - math.pi
        S.p = S.p + S.LP * res
        S.v = S.v + S.LV * res
        S.hp += S.LP * res
        S.hy += P["LY"] * yres
        dth = S.LA * _cross(R[:, 2], res)
        dth[2] += P["LY"] * yres
        S.R = _expm(dth) @ R
        R = S.R

    # ---------- thrust effectiveness, motor-lag identification and failure detection ----------
    fsum = float(S.fm2[S.hyp].sum())
    raz = fb[2] - S.eta * fsum / M_NOM            # thrust deficit seen by the accelerometer
    S.raz += (raz - S.raz) * 0.4
    if k > 20 and fsum > 3.0 and (S.failed < 0 or t > S.t_fail + 0.15):
        S.eta += (fb[2] * M_NOM / fsum - S.eta) * DT / P["TAU_ETA"]
        S.eta = min(1.5, max(0.5, S.eta))
    if k > 0 and S.failed < 0:
        y = (w[:2] - S.w_prev[:2]) + S.gyr * DT             # gyro increment, gyroscopic part removed
        x2 = S.x2 * DT                                      # increment predicted per hypothesis for eta_t = 1
        e2 = y - S.eta_t * x2                               # residual, per hypothesis
        if k > 40:
            S.zb += (e2[S.hyp] / DT - S.zb) * DT / P["TAU_B"]
        e2 = e2 - S.zb * DT
        S.z2 = S.lamz * S.z2 + e2
        if 0.3 < t < P["T_ARM"]:
            S.E[0] += e2[0, 0] * e2[0, 0] + e2[0, 1] * e2[0, 1]
            S.E[1] += e2[1, 0] * e2[1, 0] + e2[1, 1] * e2[1, 1]
            yb = y - S.zb * DT
            S.num += x2 @ yb                                # least squares for the torque effectiveness
            S.den += (x2 * x2).sum(axis=1)
            if t > 1.0:
                S.hyp = 1 if S.E[1] < S.E[0] else 0
        if t > 1.0 and S.den[S.hyp] > 1e-3:
            S.eta_t += (min(1.5, max(0.55, S.num[S.hyp] / S.den[S.hyp])) - S.eta_t) * DT / 0.1
        else:
            S.eta_t = S.eta
        if t >= P["T_ARM"]:
            sc = U_FAIL @ S.z2[S.hyp]
            i = int(np.argmax(sc))
            others = max(sc[(i + 1) % 4], sc[(i + 2) % 4], sc[(i + 3) % 4], 0.0)
            if ((sc[i] > P["DET_W"] and S.raz < -P["DET_A"]) or sc[i] > P["DET_W2"]) and sc[i] > 2.0 * others:
                S.failed = i
                S.t_fail = t
                S.fm2[:, i] = 0.0
    S.w_prev = w.copy()

    # ---------- position loop ----------
    e = S.p - target
    en = math.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
    if en < 0.6:
        S.integ += P["KI"] * e * DT
        np.clip(S.integ, -1.5, 1.5, out=S.integ)
    a_fb = -P["KP"] * e - P["KD"] * S.v
    ah = math.hypot(a_fb[0], a_fb[1])
    if ah > P["A_H_MAX"]:
        a_fb[0] *= P["A_H_MAX"] / ah
        a_fb[1] *= P["A_H_MAX"] / ah
    a_fb[2] = min(6.0, max(-4.0, a_fb[2]))
    dw = R[:, 0] * fb[0] + R[:, 1] * fb[1]         # drag acceleration (world), measured
    S.d1 += (dw - S.d1) * DT / P["TAU_DRAG"]
    S.d2 += (S.d1 - S.d2) * DT / P["TAU_DRAG"]
    a_cmd = a_fb - S.integ - S.d2
    a_cmd[2] += G
    if a_cmd[2] < 3.0:
        a_cmd[2] = 3.0
    an = math.sqrt(a_cmd[0] * a_cmd[0] + a_cmd[1] * a_cmd[1] + a_cmd[2] * a_cmd[2])
    nd = a_cmd / an
    az = min(an, 22.0)
    T = az * M_NOM / S.eta

    # ---------- reduced attitude loop ----------
    # control axis c (body-fixed): body z when not spinning, the principal axis (5 deg behind z) when spinning,
    # so that a steady spin about c needs no torque; control frame C = body frame turned by -phi about y
    p_, q_, r_ = w[0], w[1], w[2]
    phi = PHI_P * min(1.0, max(0.0, (abs(r_) - P["R_BLEND0"]) / (P["R_BLEND1"] - P["R_BLEND0"])))
    cph, sph = math.cos(phi), math.sin(phi)
    nb = R.T @ nd                                   # wanted thrust direction in body axes
    ncx, ncy, ncz = cph * nb[0] + sph * nb[2], nb[1], -sph * nb[0] + cph * nb[2]
    pc, qc, om = cph * p_ + sph * r_, q_, -sph * p_ + cph * r_
    sxy = math.hypot(ncx, ncy)
    ang = math.atan2(sxy, ncz)
    if sxy > 1e-9:
        g = P["KN"] * ang / sxy
        wdx, wdy = -g * ncy, g * ncx
        wn = math.hypot(wdx, wdy)
        if wn > P["W_MAX"]:
            wdx *= P["W_MAX"] / wn
            wdy *= P["W_MAX"] / wn
    else:
        wdx = wdy = 0.0
    # feedback: plain rate damping/tracking (no phase advance: the nutation it has to damp turns the other way)
    afx = P["KW"] * (wdx - pc)
    afy = P["KW"] * (wdy - qc)
    afn = math.hypot(afx, afy)
    if afn > P["AL_MAX"]:
        afx *= P["AL_MAX"] / afn
        afy *= P["AL_MAX"] / afn
    # feed-forward of the precession: the commanded tilt rate is fixed in space, so it turns at -om in the body;
    # its derivative and the gyroscopic torque of the commanded rates are advanced in phase by the angle the
    # body turns during the motor lag and half a control step
    # (only for the part of the command that is not fixed in the body: a body-fixed tilt error means the vehicle
    # turns about another axis than c, and precession torque on it would feed that coning instead of damping it)
    S.wdc[0] += (wdx - S.wdc[0]) * DT / P["T_DC"]
    S.wdc[1] += (wdy - S.wdc[1]) * DT / P["T_DC"]
    wrx, wry = wdx - S.wdc[0], wdy - S.wdc[1]
    tl = P["TAU_LEAD_SLOW"] if S.hyp else P["TAU_LEAD"]
    lead = complex(om * wry, -om * wrx) * complex(1.0, -tl * om) * complex(math.cos(0.5 * om * DT), -math.sin(0.5 * om * DT))
    ax_, ay_ = afx + lead.real, afy + lead.imag
    al3 = np.array([cph * ax_ - S.zb[0], ay_ - S.zb[1], sph * ax_])
    wdes = np.array([cph * wrx - sph * om, wry, sph * wrx + cph * om])
    wdot = np.array([cph * om * wry, -om * wrx, sph * om * wry])
    Jwd = J @ wdes
    hd = _cross(wdes, Jwd)
    hdot = _cross(wdot, Jwd) + _cross(wdes, J @ wdot)
    lt = tl + 0.5 * DT
    tau = (J @ al3 + hd + lt * hdot) / S.eta_t
    h = _cross(w, J @ w)
    T = T / cph
    if S.failed < 0 and 0.3 < t < P["T_ARM"] - 0.3:
        # small rotating torque dither: makes the two motor-lag hypotheses distinguishable in the gyro
        ph = P["DITH_W"] * t
        tau[0] += J[0, 0] * P["DITH_A"] * math.cos(ph) / S.eta_t
        tau[1] += J[1, 1] * P["DITH_A"] * math.sin(ph) / S.eta_t

    if S.failed < 0:
        tz = -J[2, 2] * P["KR"] * r_ / S.eta_t
        tz = min(0.2, max(-0.2, tz))
        f = MIXINV[:, 0] * T + MIXINV[:, 1] * tau[0] + MIXINV[:, 2] * tau[1]
        fy = MIXINV[:, 3] * tz
        sc_ = 1.0
        for i in range(4):
            if fy[i] > 1e-9:
                sc_ = min(sc_, max(0.0, (FMAX - f[i]) / fy[i]))
            elif fy[i] < -1e-9:
                sc_ = min(sc_, max(0.0, (f[i] - 0.3) / -fy[i]))
        u = np.clip(f + sc_ * fy, 0.0, FMAX)
    else:
        a, b, o, Winv = ALLOC3[S.failed]
        delta = Winv[0, 0] * tau[0] + Winv[0, 1] * tau[1]
        fo = Winv[1, 0] * tau[0] + Winv[1, 1] * tau[1]
        fo = min(FMAX, max(0.0, fo))
        Tab = min(2.0 * FMAX, max(T - fo, 0.35 * M_NOM * G))
        dmax = min(Tab, 2.0 * FMAX - Tab)
        delta = min(dmax, max(-dmax, delta))
        u = np.zeros(4)
        u[a] = 0.5 * (Tab + delta)
        u[b] = 0.5 * (Tab - delta)
        u[o] = fo

    # ---------- model of the rotor thrusts, prediction of the next gyro increment ----------
    fm_new = S.fm2 + (u - S.fm2) * LAG2
    if S.failed >= 0:
        fm_new[:, S.failed] = 0.0
    else:
        S.x2 = (0.5 * (S.fm2 + fm_new)) @ JT2
        S.gyr = (JINV @ h)[:2]
    S.fm2 = fm_new
    return u
