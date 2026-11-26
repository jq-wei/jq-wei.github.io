"""Point hold for the rotor-thrust quadrotor in Dryden gusts (IMU + delayed 50 Hz tracker).

All per-step maths is scalar Python (about 0.04 ms per call).  Four parts:

1. Strapdown observer.  The gyro (minus its estimated bias) turns the attitude quaternion; the accelerometer,
   rotated to world axes, is integrated to velocity and position.  The accelerometer measures every
   non-gravitational force, so wind and drag need no model here.  Each tracker frame (a pose 30 ms old) is
   compared with the stored estimate of that instant, and the residual corrects position, velocity, tilt, yaw,
   gyro bias and the vertical accelerometer bias.  The correction gains are the gain sequence of a Kalman filter
   for the linear error model of each axis ([p, v, tilt, tilt-rate bias] horizontally, [p, v, accel bias]
   vertically, [yaw, yaw-rate bias]); that model is time invariant, so the sequence is computed once at import
   and no covariance is propagated at run time.  A correction computed for the frame's instant is carried forward
   over the delay with the model's transition.  (A constant horizontal accelerometer bias is not separable from
   tilt; the observer absorbs it as a tilt offset, and because part 2 uses the same accelerometer the two
   errors cancel in the commanded thrust direction.)

2. Thrust gain and air-load observer (what makes the gusts manageable).
   - `khat`: delivered specific force per commanded one (thrust coefficient over mass ratio), from the low-passed
     body-z acceleration over the lagged collective command.  Every rotor command is divided by it, which also
     normalises the torque loops because the inertia scales with the mass.  It adapts only while the air load is
     small (`d_gate`): in a strong wind the leaning vehicle's thin axis catches the wind, and that body-z load
     must not be mistaken for weak rotors.
   - Air load across the thrust axis: body x and y accelerometer readings are pure air load (the rotors push
     along body z only).  Low-passed in world axes it is `d`, and it is subtracted from the commanded force, so
     the vehicle leans into a gust as soon as the gust pushes, before a position error develops.
   - Air load along the thrust axis: what body z reads beyond `khat` times the modelled thrust, kept as a
     body-axis scalar `dbz` and made up with collective.  It turns with the vehicle, so leaning against it
     would only expose the thin axis further.
   The accelerometer is referred to the nominal centre of mass with the measured angular acceleration.

3. Cascade.  Position PID -> desired specific force F = a_cmd + g - d -> thrust direction and collective.
   Tilt error of the thrust axis (yaw is separate and weak: little yaw authority) -> body-rate command -> rate
   loop with angular-acceleration feedback (a lead against the motor lag) -> torques.

4. Mixer.  Roll and pitch torque are scaled together into the room around the collective and yaw gets what is
   left, so no rotor command is clipped: a clipped rotor changes the collective and the torque direction, which
   with a slow motor turned the first step of the approach into a limit cycle.

Gains (PARAMS).  The values keep a margin beyond the graded scenarios: they still hold with a 90 ms motor, with
the torque loop gain scaled by 0.6 to 1.5, with a 40 ms tracker delay and with twice the accelerometer noise.
  kp, kd, ki        horizontal position PID [1/s^2, 1/s, 1/s^3]: stiffness against what the observer misses;
                    higher values lose the margin for a slow motor
  kpz, kdz, kiz     vertical position PID (faster: the collective answers with the motor lag only)
  ka                tilt error [rad] -> body-rate command [rad/s] (bandwidth of the tilt loop)
  kw                rate error -> angular acceleration command [1/s]
  kal               gain on the measured angular acceleration; shortens the effective motor lag by 1 + kal
  kyaw, kwz         yaw angle -> yaw rate [1/s], yaw rate -> yaw acceleration [1/s]
  tau_d             time constant of the air-load observer [s]: shorter rejects gusts sooner, passes more noise
  tau_k             time constant of the thrust-gain estimate [s]
  tau_m             motor lag assumed by the thrust model [s] (the short end of the range: the residual then
                    acts as a lead on the collective)
  tilt_max          largest commanded tilt [rad]; by a static force balance a 4-sigma gust on an 11 m/s wind
                    needs about 0.56
  d_gate            the thrust gain adapts only while the air-load estimate is below this [m/s^2]
"""

import math

import numpy as np

G = 9.81
MASS = 1.325
IXX, IYY, IZZ = 0.0367, 0.0254, 0.0605
DT = 0.005
ARM_Y, ARM_X, YAW_C = 0.18, 0.14, 0.0201
RZ = 0.02 - 0.054          # IMU height relative to the nominal centre of mass [m]
U_MAX = 13.0
U_MIN = 0.2                # N, lowest thrust the mixer asks of a rotor
W_CMD_MAX = 3.0            # rad/s, largest roll / pitch rate command
AL_CMD_MAX = 45.0          # rad/s^2, largest roll / pitch angular acceleration command
I_BAND = 0.15              # m, the position integrators run only within this distance per axis

PARAMS = {
    "kp": 6.5, "kd": 4.0, "ki": 1.5,
    "kpz": 12.0, "kdz": 6.0, "kiz": 3.0,
    "ka": 12.0, "kw": 60.0, "kal": 1.5,
    "kyaw": 2.0, "kwz": 5.0,
    "tau_d": 0.035, "tau_k": 0.25, "tau_m": 0.03,
    "tilt_max": 0.75, "d_gate": 1.0,
}
PARAM_BOUNDS = {
    "kp": (2.0, 12.0), "kd": (2.0, 8.0), "ki": (0.0, 5.0),
    "kpz": (5.0, 25.0), "kdz": (3.0, 12.0), "kiz": (0.0, 8.0),
    "ka": (5.0, 20.0), "kw": (20.0, 100.0), "kal": (0.0, 3.0),
    "kyaw": (0.5, 5.0), "kwz": (2.0, 10.0),
    "tau_d": (0.03, 0.4), "tau_k": (0.1, 1.0), "tau_m": (0.02, 0.07),
    "tilt_max": (0.4, 0.8), "d_gate": (0.5, 2.0),
}


def _gain_schedule(a_c, q_c, r_meas, p0, period, n_updates):
    """Kalman gain sequence for x' = a_c x + noise(q_c), y = x[0] + noise(r_meas), one measurement per period."""
    n = a_c.shape[0]
    eye = np.eye(n)

    def phi(t):
        out, term = eye.copy(), eye.copy()
        for k in range(1, n):
            term = term @ a_c * (t / k)
            out = out + term
        return out

    sub = 20
    h = period / sub
    qd = np.zeros((n, n))
    for i in range(sub):
        f = phi((i + 0.5) * h)
        qd += f @ q_c @ f.T * h
    f = phi(period)
    p = p0.copy()
    gains = []
    for _ in range(n_updates):
        k = p[:, 0] / (p[0, 0] + r_meas)
        gains.append(tuple(float(v) for v in k))
        p = p - np.outer(k, p[0, :])
        p = f @ p @ f.T + qd
        p = 0.5 * (p + p.T)
    return gains


_Q_ACC = 8e-4      # (m/s^2)^2 s, accelerometer noise density (inflated)
_Q_TILT = 2e-6     # (rad/s)^2 s, gyro noise density (inflated)
_Q_GB = 1e-7       # gyro-bias random walk
_N_UPD = 800
_KH = _gain_schedule(np.array([[0, 1, 0, 0], [0, 0, G, 0], [0, 0, 0, 1], [0, 0, 0, 0.0]]),
                     np.diag([0.0, _Q_ACC, _Q_TILT, _Q_GB]), 0.003 ** 2,
                     np.diag([0.003 ** 2, 0.25 ** 2, 0.04 ** 2, 0.015 ** 2]), 0.02, _N_UPD)
_KV = _gain_schedule(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0.0]]),
                     np.diag([0.0, _Q_ACC, 1e-4]), 0.003 ** 2,
                     np.diag([0.003 ** 2, 0.25 ** 2, 0.15 ** 2]), 0.02, _N_UPD)
_KY = _gain_schedule(np.array([[0, 1], [0, 0.0]]),
                     np.diag([_Q_TILT, _Q_GB]), 0.005 ** 2,
                     np.diag([0.005 ** 2, 0.015 ** 2]), 0.02, _N_UPD)


class _State:
    pass


_S = _State()


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _reset(s, obs):
    tp = obs["tracker_pos"]
    s.px, s.py, s.pz = float(tp[0]), float(tp[1]), float(tp[2])
    s.vx = s.vy = s.vz = 0.0
    yaw = float(obs["tracker_yaw"])
    s.qw, s.qx, s.qy, s.qz = math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)
    s.bgx = s.bgy = s.bgz = 0.0
    s.baz = 0.0
    s.yaw_des = yaw
    s.hist = {0: (s.px, s.py, s.pz, yaw)}
    s.n_upd = 0
    g = obs["gyro"]
    s.gpx, s.gpy, s.gpz = float(g[0]), float(g[1]), float(g[2])
    s.fwx, s.fwy, s.fwz = 0.0, 0.0, G
    s.alx = s.aly = s.alz = 0.0
    s.xlag = G
    s.knum = G
    s.kden = G
    s.khat = 1.0
    s.dx = s.dy = s.dz = 0.0
    s.dbz = 0.0
    s.ix = s.iy = s.iz = 0.0


def controller(obs: dict) -> np.ndarray:
    s = _S
    P = PARAMS
    step = int(obs["step"])
    if step == 0 or not hasattr(s, "px"):
        _reset(s, obs)
        first = True
    else:
        first = False
    g = obs["gyro"]
    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
    a = obs["accel"]
    ax, ay, az = float(a[0]), float(a[1]), float(a[2])

    # ---- angular acceleration (filtered gyro difference; the bias cancels) ----
    if not first:
        s.alx += 0.4 * ((gx - s.gpx) / DT - s.alx)
        s.aly += 0.4 * ((gy - s.gpy) / DT - s.aly)
        s.alz += 0.4 * ((gz - s.gpz) / DT - s.alz)

    # ---- attitude propagation (trapezoidal body rate) ----
    qw, qx, qy, qz = s.qw, s.qx, s.qy, s.qz
    if not first:
        hx = 0.25 * DT * (gx + s.gpx - 2.0 * s.bgx)
        hy = 0.25 * DT * (gy + s.gpy - 2.0 * s.bgy)
        hz = 0.25 * DT * (gz + s.gpz - 2.0 * s.bgz)
        nw = qw - qx * hx - qy * hy - qz * hz
        nx = qx + qw * hx + qy * hz - qz * hy
        ny = qy + qw * hy - qx * hz + qz * hx
        nz = qz + qw * hz + qx * hy - qy * hx
        inv = 1.0 / math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        qw, qx, qy, qz = nw * inv, nx * inv, ny * inv, nz * inv
    s.gpx, s.gpy, s.gpz = gx, gy, gz

    r00 = 1.0 - 2.0 * (qy * qy + qz * qz); r01 = 2.0 * (qx * qy - qw * qz); r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz); r11 = 1.0 - 2.0 * (qx * qx + qz * qz); r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy); r21 = 2.0 * (qy * qz + qw * qx); r22 = 1.0 - 2.0 * (qx * qx + qy * qy)

    # ---- velocity / position propagation with the world-frame specific force ----
    azb = az - s.baz
    fwx = r00 * ax + r01 * ay + r02 * azb
    fwy = r10 * ax + r11 * ay + r12 * azb
    fwz = r20 * ax + r21 * ay + r22 * azb
    if not first:
        awx = 0.5 * (fwx + s.fwx)
        awy = 0.5 * (fwy + s.fwy)
        awz = 0.5 * (fwz + s.fwz) - G
        s.px += s.vx * DT + 0.5 * awx * DT * DT
        s.py += s.vy * DT + 0.5 * awy * DT * DT
        s.pz += s.vz * DT + 0.5 * awz * DT * DT
        s.vx += awx * DT
        s.vy += awy * DT
        s.vz += awz * DT
    s.fwx, s.fwy, s.fwz = fwx, fwy, fwz
    yaw = math.atan2(r10, r00)
    hist = s.hist
    hist[step] = (s.px, s.py, s.pz, yaw)

    # ---- tracker correction ----
    if (not first) and obs["tracker_fresh"]:
        j = step - 6 if step >= 6 else 0
        ref = hist.get(j)
        if ref is not None:
            tp = obs["tracker_pos"]
            rx = float(tp[0]) - ref[0]
            ry = float(tp[1]) - ref[1]
            rzz = float(tp[2]) - ref[2]
            ryaw = _wrap(float(obs["tracker_yaw"]) - ref[3])
            n = s.n_upd if s.n_upd < _N_UPD else _N_UPD - 1
            s.n_upd += 1
            k1, k2, k3, k4 = _KH[n]
            v1, v2, v3 = _KV[n]
            y1, y2 = _KY[n]
            # corrections at the frame's instant: channel x -> (p_x, v_x, theta_y, w_y); channel y -> (p_y, v_y, -theta_x, -w_x)
            dpx, dvx, thy, wy = k1 * rx, k2 * rx, k3 * rx, k4 * rx
            dpy, dvy, thxn, wxn = k1 * ry, k2 * ry, k3 * ry, k4 * ry
            dpz, dvz, cz = v1 * rzz, v2 * rzz, v3 * rzz
            dyaw, wz = y1 * ryaw, y2 * ryaw
            # the stored estimate that the next frame will be compared with (20 ms after this frame's instant)
            nxt = hist.get(j + 4)
            if nxt is not None:
                t = 0.02
                hist[j + 4] = (nxt[0] + dpx + dvx * t + G * (0.5 * thy * t * t + wy * t * t * t / 6.0),
                               nxt[1] + dpy + dvy * t + G * (0.5 * thxn * t * t + wxn * t * t * t / 6.0),
                               nxt[2] + dpz + dvz * t + 0.5 * cz * t * t,
                               nxt[3] + dyaw + wz * t)
            # carried forward to now
            t = (step - j) * DT
            s.px += dpx + dvx * t + G * (0.5 * thy * t * t + wy * t * t * t / 6.0)
            s.py += dpy + dvy * t + G * (0.5 * thxn * t * t + wxn * t * t * t / 6.0)
            s.pz += dpz + dvz * t + 0.5 * cz * t * t
            s.vx += dvx + G * (thy * t + 0.5 * wy * t * t)
            s.vy += dvy + G * (thxn * t + 0.5 * wxn * t * t)
            s.vz += dvz + cz * t
            tx = -(thxn + wxn * t)
            ty = thy + wy * t
            tz = dyaw + wz * t
            # world-frame rotation: q <- dq(theta) * q
            hx, hy, hz = 0.5 * tx, 0.5 * ty, 0.5 * tz
            nw = qw - hx * qx - hy * qy - hz * qz
            nx = qx + hx * qw + hy * qz - hz * qy
            ny = qy - hx * qz + hy * qw + hz * qx
            nz = qz + hx * qy - hy * qx + hz * qw
            inv = 1.0 / math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
            qw, qx, qy, qz = nw * inv, nx * inv, ny * inv, nz * inv
            # gyro bias: b <- b - R^T w  (w = world-frame rate error)
            wxw, wyw, wzw = -wxn, wy, wz
            s.bgx -= r00 * wxw + r10 * wyw + r20 * wzw
            s.bgy -= r01 * wxw + r11 * wyw + r21 * wzw
            s.bgz -= r02 * wxw + r12 * wyw + r22 * wzw
            s.baz -= cz
            r00 = 1.0 - 2.0 * (qy * qy + qz * qz); r01 = 2.0 * (qx * qy - qw * qz); r02 = 2.0 * (qx * qz + qw * qy)
            r10 = 2.0 * (qx * qy + qw * qz); r11 = 1.0 - 2.0 * (qx * qx + qz * qz); r12 = 2.0 * (qy * qz - qw * qx)
            r20 = 2.0 * (qx * qz - qw * qy); r21 = 2.0 * (qy * qz + qw * qx); r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
            yaw = math.atan2(r10, r00)
            hist[step] = (s.px, s.py, s.pz, yaw)
        if step >= 16:
            hist.pop(step - 16, None)
            hist.pop(step - 15, None)
            hist.pop(step - 14, None)
            hist.pop(step - 13, None)
    s.qw, s.qx, s.qy, s.qz = qw, qx, qy, qz

    wx, wy_, wz_ = gx - s.bgx, gy - s.bgy, gz - s.bgz

    # ---- thrust model, thrust-gain estimate, disturbance observer ----
    if not first:
        la = obs["last_action"]
        usum = float(la[0]) + float(la[1]) + float(la[2]) + float(la[3])
        s.xlag += (usum / MASS - s.xlag) * (1.0 - math.exp(-DT / P["tau_m"]))
    ax_c = ax - s.aly * RZ - RZ * wx * wz_
    ay_c = ay + s.alx * RZ - RZ * wy_ * wz_
    az_c = az + RZ * (wx * wx + wy_ * wy_) - s.baz
    # adapt the thrust gain only while the air load is small: in a strong wind the leaning vehicle's thin axis
    # catches the wind and that body-z load would be mistaken for weak rotors (it belongs to the observer below)
    if s.dx * s.dx + s.dy * s.dy < P["d_gate"] * P["d_gate"]:
        bk = DT / P["tau_k"]
        s.knum += bk * (az_c - s.knum)
        s.kden += bk * (s.xlag - s.kden)
        khat = s.knum / max(s.kden, 1.0)
        s.khat = 0.6 if khat < 0.6 else (1.5 if khat > 1.5 else khat)
    khat = s.khat
    # air load across the thrust axis (body x, y readings), kept in world axes: the vehicle leans against it.
    # The load along the thrust axis (what body z reads beyond the modelled thrust) stays a body-axis scalar and
    # is made up with collective: it turns with the vehicle, and leaning against it would only expose the thin
    # axis further to the wind.
    bd = DT / P["tau_d"]
    s.dx += bd * (r00 * ax_c + r01 * ay_c - s.dx)
    s.dy += bd * (r10 * ax_c + r11 * ay_c - s.dy)
    s.dz += bd * (r20 * ax_c + r21 * ay_c - s.dz)
    s.dbz += bd * (az_c - khat * s.xlag - s.dbz)

    # ---- position loop ----
    tgt = obs["target_pos"]
    ex, ey, ez = s.px - float(tgt[0]), s.py - float(tgt[1]), s.pz - float(tgt[2])
    # integrators trim what the observer leaves (sensor bias); they run only near the setpoint, so that the
    # approach from the start point and a large gust excursion do not wind them up
    if step > 200:
        lim = 1.5 / max(P["ki"], 1e-6)
        if -I_BAND < ex < I_BAND:
            s.ix = max(-lim, min(lim, s.ix + ex * DT))
        if -I_BAND < ey < I_BAND:
            s.iy = max(-lim, min(lim, s.iy + ey * DT))
        limz = 2.0 / max(P["kiz"], 1e-6)
        if -I_BAND < ez < I_BAND:
            s.iz = max(-limz, min(limz, s.iz + ez * DT))
    fx = -P["kp"] * ex - P["kd"] * s.vx - P["ki"] * s.ix - s.dx
    fy = -P["kp"] * ey - P["kd"] * s.vy - P["ki"] * s.iy - s.dy
    fz = -P["kpz"] * ez - P["kdz"] * s.vz - P["kiz"] * s.iz - s.dz + G
    if fz < 0.4 * G:
        fz = 0.4 * G
    elif fz > 2.0 * G:
        fz = 2.0 * G
    fh = math.hypot(fx, fy)
    fh_max = math.tan(P["tilt_max"]) * fz
    if fh > fh_max:
        fx *= fh_max / fh
        fy *= fh_max / fh
    fn = math.sqrt(fx * fx + fy * fy + fz * fz)

    # ---- attitude: desired thrust axis in body coordinates, tilt error -> rate command ----
    zdx = (r00 * fx + r10 * fy + r20 * fz) / fn
    zdy = (r01 * fx + r11 * fy + r21 * fz) / fn
    ka = P["ka"]
    wdx = -ka * zdy
    wdy = ka * zdx
    wdx = max(-W_CMD_MAX, min(W_CMD_MAX, wdx))
    wdy = max(-W_CMD_MAX, min(W_CMD_MAX, wdy))
    wdz = -P["kyaw"] * _wrap(yaw - s.yaw_des)
    wdz = max(-1.0, min(1.0, wdz))
    kw, kal = P["kw"], P["kal"]
    alx = kw * (wdx - wx) - kal * s.alx
    aly = kw * (wdy - wy_) - kal * s.aly
    alz = P["kwz"] * (wdz - wz_)
    alx = max(-AL_CMD_MAX, min(AL_CMD_MAX, alx))
    aly = max(-AL_CMD_MAX, min(AL_CMD_MAX, aly))
    tqx = IXX * alx
    tqy = IYY * aly
    tqz = max(-0.08, min(0.08, IZZ * alz))

    # ---- collective and mixer ----
    coll = fx * r02 + fy * r12 + fz * r22          # commanded specific force along the present thrust axis
    coll = max(0.3 * G, min(2.5 * G, coll))
    base = 0.25 * MASS * (coll - s.dbz) / khat     # the rotors also make up the load along the thrust axis
    base = max(U_MIN + 0.5, min(U_MAX - 0.5, base))
    cx = tqx / (4.0 * ARM_Y) / khat
    cy = tqy / (4.0 * ARM_X) / khat
    cz_ = tqz / (4.0 * YAW_C) / khat
    # keep every rotor inside its range without clipping: a clipped rotor would change the collective and the
    # torque direction.  Roll and pitch torque are scaled together into the room around the collective; yaw gets
    # what is left.
    room = min(base - U_MIN, U_MAX - base)
    need = abs(cx) + abs(cy)
    if need > room:
        cx *= room / need
        cy *= room / need
        need = room
    left = room - need
    cz_ = max(-left, min(left, cz_))
    u = np.array([base - cx + cy - cz_, base + cx + cy + cz_, base + cx - cy - cz_, base - cx - cy + cz_])
    return np.clip(u, 0.0, U_MAX)
