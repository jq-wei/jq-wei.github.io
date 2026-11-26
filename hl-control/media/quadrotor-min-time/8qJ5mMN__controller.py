"""Gate-race controller for the rotor-thrust quadrotor (min_time task).

Control law
-----------
1. Estimation: 15-state error-state Kalman filter (position, velocity, attitude error, gyro bias, accelerometer
   bias). The IMU drives a strapdown propagation (mid-interval attitude, IMU lever arm removed), so thrust, drag and
   wind are measured rather than modelled; each 50 Hz tracker frame (position + yaw, 30 ms late) is compared with the
   filter's own estimate from 30 ms ago (ring buffer).                      [r_pos, q_vel, q_att: filter tuning]
2. Launch: the vehicle glides (glide_time) to a point hover_back metres behind the start, toward gate 1, inside
   the start sphere (clock not running), hovers until t_launch so the filter converges, and measures the thrust
   effectiveness keff = k m_nom / m from the accelerometer.
3. Trajectory: an offline near-time-optimal point-mass trajectory (plant drag model, thrust <= a fraction of the
   maximum, limits on the thrust-axis rate / angular acceleration, tilt <= ~107 deg, gates as small spheres whose
   centres are shifted against the systematic miss measured in closed loop) is embedded below as nodes (t, p, v,
   thrust vector u). At launch it is time-stretched by the smallest factor in [scale_min, scale_max] that keeps its
   thrust below u_frac of this vehicle's maximum (the inertial + drag part of u scales exactly as 1/stretch^2).
   The trajectory clock is warped (k_warp, warp_min): it slows down when the vehicle falls behind along the path.
4. Tracking: F = u_ref(t + ff_lead) + m (k e_p + d e_v + ki int e_p) with separate gains along the path
   (kp_al, kd_al) and across it (kp_cr, kd_cr); the plan's drag is swapped for the drag at the current attitude
   and velocity (drag_beta; wind_gain = 0 leaves the air at rest); above sat_frac of the available thrust the
   along-track part of F is given up first. Collective = F . body z. Thrust axis follows F by a reduced-attitude
   law (no singularity beyond 90 deg tilt): rate command min(kr/kw |e|, sqrt(2 a_max |e|)) plus the reference rate
   and angular acceleration (w_ff), rate loop gain kw; yaw held weakly (kr_yaw, kw_yaw). Torques are mixed to the
   rotors with roll/pitch priority over collective and yaw; the 30 ms motor lag is compensated by overdriving the
   command (lag_gain) against a model of the rotor thrusts. wz_ff = 0 and wind_gain = 0 are disabled options.
"""

import math
import numpy as np

PARAMS = {
    "t_launch": 2.0,
    "time_scale": 1.0,
    "kp": 20.0,
    "kd": 9.0,
    "kp_h": 16.0,
    "kd_h": 7.0,
    "ki": 2.0,
    "kr": 250.0,
    "kw": 45.0,
    "kr_yaw": 30.0,
    "kw_yaw": 8.0,
    "w_ff": 1.0,
    "lag_gain": 2.5,
    "yaw": 0.0,
    "r_pos": 0.004,
    "q_vel": 2.0,
    "q_att": 0.03,
    "u_frac": 0.88,
    "a_max": 60.0,
    "ff_lead": 0.03,
    "k_warp": 0.3,
    "warp_min": 0.6,
    "warp_max": 1.0,
    "wz_ff": 0.0,
    "wz_alpha": 0.3,
    "sat_frac": 0.95,
    "glide_time": 1.5,
    "hover_back": 0.18,
    "scale_max": 1.1,
    "scale_min": 1.0,
    "drag_beta": 1.0,
    "wind_avg": 0.8,
    "wind_gain": 0.0,
    "kp_al": 20.0,
    "kp_cr": 25.0,
    "kd_al": 9.0,
    "kd_cr": 9.0,
    "k_gate": 0.0,
    "t_gate": 0.3,
    "r_safe": 0.12}

# ---- embedded reference trajectory (generated offline by plan_traj.py) ----
_T = np.array([0, 0.0744706, 0.148941, 0.223412, 0.297882, 0.372353, 0.446823, 0.521294, 0.595764, 0.670235, 0.744706, 0.819176, 0.893647, 0.968117, 1.04259, 1.09602, 1.14946, 1.2029, 1.25633, 1.30977, 1.3632, 1.41664, 1.47008, 1.52351, 1.57695, 1.63039, 1.68382, 1.73726, 1.79069, 1.85017, 1.90965, 1.96913, 2.0286, 2.08808, 2.14756, 2.20704, 2.26651, 2.32599, 2.38547, 2.44494, 2.50442, 2.5639, 2.62338, 2.6569, 2.69043, 2.72396, 2.75749, 2.79102, 2.82454, 2.85807, 2.8916, 2.92513, 2.95866, 2.99218, 3.02571, 3.05924, 3.09277])
_P = np.array([[-0.232495, -0.0871857, 1.97094],
    [-0.227348, -0.0811625, 2.00441],
    [-0.193912, -0.0428885, 2.12922],
    [-0.0932745, 0.0712684, 2.34401],
    [0.10659, 0.29485, 2.58981],
    [0.419337, 0.633397, 2.81418],
    [0.847503, 1.06815, 2.99779],
    [1.39354, 1.56551, 3.14768],
    [2.0634, 2.08456, 3.27652],
    [2.86172, 2.58129, 3.38765],
    [3.78402, 3.00535, 3.46976],
    [4.80729, 3.30293, 3.50087],
    [5.88907, 3.43497, 3.45713],
    [6.97909, 3.38598, 3.32265],
    [8.03512, 3.15571, 3.09428],
    [8.75677, 2.88181, 2.87536],
    [9.44011, 2.52064, 2.61521],
    [10.08, 2.07518, 2.32596],
    [10.6717, 1.55142, 2.03128],
    [11.21, 0.963467, 1.75775],
    [11.6907, 0.333465, 1.52716],
    [12.1111, -0.310983, 1.35602],
    [12.4724, -0.93761, 1.25389],
    [12.7817, -1.51214, 1.2192],
    [13.052, -2.00442, 1.23705],
    [13.2982, -2.39633, 1.28536],
    [13.5314, -2.68472, 1.34819],
    [13.7579, -2.87458, 1.42077],
    [13.9832, -2.97102, 1.50369],
    [14.2374, -2.97167, 1.60884],
    [14.4944, -2.86129, 1.72645],
    [14.7496, -2.64205, 1.85407],
    [14.999, -2.31924, 1.99103],
    [15.246, -1.90401, 2.14203],
    [15.5054, -1.41446, 2.31412],
    [15.7996, -0.872645, 2.50946],
    [16.1515, -0.301724, 2.72381],
    [16.5806, 0.272545, 2.94827],
    [17.0992, 0.819523, 3.16925],
    [17.7075, 1.30681, 3.36727],
    [18.3911, 1.70766, 3.51866],
    [19.1277, 2.00538, 3.6048],
    [19.8978, 2.1922, 3.619],
    [20.3414, 2.24756, 3.5951],
    [20.7886, 2.26674, 3.54823],
    [21.2376, 2.25017, 3.47858],
    [21.6873, 2.19848, 3.38667],
    [22.136, 2.11209, 3.27408],
    [22.5831, 1.99107, 3.1442],
    [23.0284, 1.83513, 3.001],
    [23.4713, 1.64391, 2.8468],
    [23.9113, 1.41801, 2.68206],
    [24.3481, 1.15874, 2.50738],
    [24.7814, 0.867071, 2.32519],
    [25.2109, 0.545139, 2.13865],
    [25.6362, 0.199546, 1.9491],
    [26.0577, -0.160839, 1.75665]])
_V = np.array([[0, 0, 0],
    [0.138233, 0.161762, 0.899013],
    [0.759735, 0.866135, 2.45289],
    [1.94301, 2.19969, 3.31546],
    [3.42461, 3.80486, 3.28587],
    [4.9746, 5.28726, 2.73997],
    [6.52434, 6.38869, 2.19104],
    [8.14026, 6.96841, 1.83447],
    [9.84951, 6.97133, 1.62568],
    [11.5905, 6.3689, 1.35869],
    [13.179, 5.01982, 0.846617],
    [14.3024, 2.97216, -0.0111964],
    [14.7502, 0.573874, -1.16355],
    [14.5235, -1.88953, -2.44785],
    [13.8376, -4.29483, -3.68534],
    [13.1722, -5.95661, -4.50841],
    [12.4034, -7.56091, -5.22845],
    [11.5478, -9.1118, -5.59746],
    [10.5956, -10.4916, -5.43191],
    [9.55296, -11.514, -4.80579],
    [8.43755, -12.0656, -3.8247],
    [7.29653, -12.0547, -2.58054],
    [6.22688, -11.3986, -1.24206],
    [5.35148, -10.1049, -0.0561357],
    [4.76556, -8.32001, 0.724298],
    [4.44904, -6.3481, 1.08383],
    [4.27683, -4.44584, 1.26754],
    [4.2001, -2.66024, 1.44893],
    [4.23399, -0.94917, 1.65453],
    [4.3135, 0.927311, 1.88156],
    [4.33015, 2.78422, 2.07309],
    [4.25124, 4.58802, 2.21845],
    [4.13508, 6.26705, 2.38676],
    [4.17002, 7.69553, 2.69088],
    [4.55214, 8.76618, 3.09596],
    [5.34121, 9.45296, 3.47257],
    [6.49225, 9.74496, 3.73492],
    [7.93504, 9.56557, 3.81303],
    [9.50377, 8.82726, 3.61754],
    [10.9505, 7.55825, 3.04125],
    [12.0387, 5.92087, 2.04941],
    [12.7305, 4.09043, 0.847157],
    [13.1629, 2.19174, -0.369634],
    [13.3, 1.1102, -1.05615],
    [13.3745, 0.0340163, -1.73981],
    [13.4135, -1.02234, -2.4149],
    [13.4067, -2.06085, -3.06773],
    [13.3622, -3.09292, -3.64815],
    [13.3091, -4.12584, -4.09954],
    [13.251, -5.17644, -4.44247],
    [13.1689, -6.23019, -4.75577],
    [13.0767, -7.24467, -5.07122],
    [12.9804, -8.2215, -5.34867],
    [12.8697, -9.17698, -5.51976],
    [12.7466, -10.0268, -5.60763],
    [12.6272, -10.5884, -5.69919],
    [12.5104, -10.9091, -5.78085]])
_U = np.array([[0, 0, 12.9983],
    [4.92218, 5.7737, 45.1267],
    [17.5602, 19.8229, 37.3185],
    [26.4692, 30.1122, 22.0595],
    [31.372, 32.8058, 5.80341],
    [34.3656, 29.9969, 3.65706],
    [39.2461, 22.1309, 7.99666],
    [42.5747, 8.45068, 14.4899],
    [42.9825, -5.20556, 14.8117],
    [40.7092, -18.3337, 10.0311],
    [31.7812, -32.8759, 1.76489],
    [18.5939, -41.3353, -6.29595],
    [5.6093, -44.502, -9.0605],
    [-4.83266, -43.8935, -12.0003],
    [-11.3296, -43.2954, -9.54648],
    [-15.3349, -42.2846, -8.41714],
    [-16.9244, -42.3518, -3.72403],
    [-18.8662, -40.7891, 8.6199],
    [-21.7085, -34.1377, 21.3856],
    [-24.4659, -23.2882, 30.8719],
    [-24.4499, -10.1124, 37.3368],
    [-22.8754, 4.33298, 39.3946],
    [-18.7292, 18.2838, 37.5353],
    [-13.7304, 30.534, 31.1951],
    [-7.40642, 39.6555, 21.6003],
    [-3.88748, 42.7211, 15.9324],
    [-2.35172, 42.6702, 16.3623],
    [-0.153169, 42.2605, 17.5501],
    [2.72843, 41.7631, 18.5036],
    [1.66868, 42.1109, 17.8288],
    [-0.0698327, 42.3046, 17.4441],
    [-2.89873, 42.4416, 16.8625],
    [-2.36292, 40.7941, 20.6003],
    [4.23639, 36.2061, 27.6616],
    [17.5513, 27.7437, 31.8782],
    [29.8962, 17.2079, 30.068],
    [38.5755, 4.69907, 24.1627],
    [42.1161, -8.7528, 15.6077],
    [39.7007, -21.8911, 6.2146],
    [32.5837, -31.7909, -4.64916],
    [21.7341, -38.0662, -13.1383],
    [14.4951, -41.1756, -13.728],
    [10.5887, -42.3502, -13.728],
    [6.74695, -43.1327, -13.7193],
    [6.20883, -43.2123, -13.728],
    [4.45173, -43.4263, -13.728],
    [3.14862, -43.8596, -12.6662],
    [1.73046, -44.951, -8.39979],
    [1.61951, -45.6503, -2.73485],
    [0.188607, -45.7596, 0.0595126],
    [-0.0105317, -45.76, 0.041417],
    [-0.023223, -45.7599, 0.133137],
    [-0.640697, -45.6464, 3.17515],
    [-1.47793, -44.7944, 9.23891],
    [-1.75915, -37.294, 10.5065],
    [-1.15156, -20.9836, 9.16647],
    [-1.07011, -14.608, 10.2585]])
_DPL = np.array([[0, 0, 0],
    [-0.00323592, -0.0174677, -0.137443],
    [-0.363277, -0.514325, -1.01727],
    [-1.55975, -1.96683, -1.66998],
    [-3.5595, -3.83165, -1.24958],
    [-7.0212, -6.21948, -1.64052],
    [-11.4427, -6.71341, -3.55026],
    [-12.8763, -3.23889, -5.62818],
    [-11.858, 0.0974627, -5.10668],
    [-9.88264, 2.00478, -3.24028],
    [-6.08071, 1.1983, -0.781236],
    [-4.31971, 0.147825, 0.783895],
    [-3.94922, 0.347525, -0.436605],
    [-4.89234, 0.389041, 1.79281],
    [-3.35178, 1.20878, 1.7145],
    [-2.9822, 1.96035, 1.42822],
    [-2.88673, 3.11573, 1.00136],
    [-3.75146, 3.11424, 1.79969],
    [-2.89835, 3.38454, 2.40104],
    [-2.63278, 3.33792, 2.38858],
    [-3.7674, 2.71016, 4.05291],
    [-5.49321, 3.60985, 6.91239],
    [-5.94812, 6.30841, 8.53178],
    [-5.0049, 9.03369, 7.54652],
    [-2.91494, 9.29157, 4.35763],
    [-1.48756, 6.12251, 1.93588],
    [-0.813253, 2.82234, 0.87606],
    [-0.487276, 0.798502, 0.203746],
    [-0.407321, 0.0329067, -0.0649994],
    [-0.447336, -0.300663, -0.155641],
    [-0.409589, -1.3808, -0.587111],
    [-0.137603, -2.99744, -1.24683],
    [0.223502, -5.42884, -2.72056],
    [-0.540552, -7.92511, -5.99467],
    [-4.22168, -8.32155, -9.50052],
    [-8.06936, -6.03067, -9.6692],
    [-9.11772, -2.8665, -6.87602],
    [-7.29087, -1.07227, -3.41732],
    [-4.63129, -1.17943, -1.11844],
    [-3.1939, -1.67867, -0.127273],
    [-2.63873, -1.4175, -0.279781],
    [-2.77025, -0.894938, -0.423648],
    [-3.04795, -0.173753, -0.337257],
    [-3.45746, 0.175994, -0.476891],
    [-3.623, 1.11255, -0.106351],
    [-3.96101, 2.03545, 0.209712],
    [-4.17928, 3.16868, 0.586715],
    [-4.21599, 4.06919, 0.607598],
    [-3.37212, 4.88989, 0.859132],
    [-3.0592, 3.47932, 0.714521],
    [-3.52827, 4.76939, 0.445243],
    [-3.6053, 6.59527, 0.483178],
    [-3.31619, 7.61456, 0.294555],
    [-3.33296, 7.30476, -0.227449],
    [-3.1601, 7.61729, -0.464239],
    [-3.3666, 6.26458, -0.44712],
    [-3.64959, 3.98306, 0.567492]])   # drag force of the plan at its nodes [N, world]

# ---- nominal vehicle ----
M_NOM = 1.325
G = 9.81
J_NOM = np.array([[0.03665, 0.0, -0.0021], [0.0, 0.02541, 0.0], [-0.0021, 0.0, 0.06053]])
ROTOR_XY = np.array([[-0.14, -0.18], [-0.14, 0.18], [0.14, 0.18], [0.14, -0.18]])
YAW_K = np.array([-0.0201, 0.0201, -0.0201, 0.0201])
TAU_M = 0.03
MIX = np.vstack([np.ones(4), ROTOR_XY[:, 1], -ROTOR_XY[:, 0], YAW_K])   # [T, tx, ty, tz] = MIX @ f
MIX_INV = np.linalg.inv(MIX)
FMAX = 13.0
E3 = np.array([0.0, 0.0, 1.0])
DRAG_AX = np.array([[-0.08695, -0.99621, 0.0], [0.0, 0.0, -1.0], [0.99621, -0.08695, 0.0]])  # principal axes (columns), body
DRAG_C = np.array([0.16557, 0.0254, 0.02113])                                                # N per (m/s)^2


def _drag(R, v):
    va = (R.T @ v) @ DRAG_AX
    return R @ (DRAG_AX @ (-DRAG_C * va * np.abs(va)))
IMU_R = np.array([0.0, 0.0, 0.02])

# reference thrust direction and its angular rate / acceleration (world), precomputed at the nodes
_Z = _U / np.linalg.norm(_U, axis=1, keepdims=True)
_ZD = np.gradient(_Z, _T, axis=0)
_W = np.cross(_Z, _ZD)
_WD = np.gradient(_W, _T, axis=0)


def _skew(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_exp(th):
    a = math.sqrt(th[0] * th[0] + th[1] * th[1] + th[2] * th[2])
    if a < 1e-12:
        return np.array([1.0, 0.5 * th[0], 0.5 * th[1], 0.5 * th[2]])
    s = math.sin(0.5 * a) / a
    return np.array([math.cos(0.5 * a), s * th[0], s * th[1], s * th[2]])


def _quat_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


class _State:
    def __init__(self, obs):
        self.dt = float(obs["dt"])
        self.delay = int(round(float(obs["tracker_delay"]) / self.dt))
        self.p = np.array(obs["tracker_pos"], dtype=float)
        self.p_init = self.p.copy()
        self.v = np.zeros(3)
        yaw = float(obs["tracker_yaw"])
        self.q = np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)])
        self.bg = np.zeros(3)
        self.ba = np.zeros(3)
        self.Pc = np.diag([1e-4] * 3 + [1e-4] * 3 + [0.03 ** 2, 0.03 ** 2, 0.01 ** 2] + [0.012 ** 2] * 3 + [0.09 ** 2] * 3)
        self.hist_p = [self.p.copy()] * (self.delay + 1)
        self.hist_yaw = [yaw] * (self.delay + 1)
        self.prev_gyro = np.array(obs["gyro"], dtype=float)
        self.prev_acc = np.array(obs["accel"], dtype=float)
        self.f_hat = np.full(4, M_NOM * G / 4.0)      # model of rotor thrusts (command units)
        self.integ = np.zeros(3)
        self.keff_num = 0.0
        self.keff_den = 0.0
        self.keff = 1.0
        self.t_launch = None
        self.scale = 1.0
        self.tau = 0.0
        self.zd_prev = None
        self.wind = np.zeros(3)
        self.fh_sum = np.zeros(2)
        self.fh_n = 0
        self.wz_f = np.zeros(3)
        self.lag = 1.0 - math.exp(-self.dt / TAU_M)
        dt = self.dt
        qv, qa = PARAMS["q_vel"], PARAMS["q_att"]
        self.Q = np.diag([1e-9] * 3 + [(qv * dt) ** 2] * 3 + [(qa * dt) ** 2] * 3 + [1e-10] * 3 + [1e-8] * 3)
        d0 = _P[1] - _P[0]
        self.launch_dir = d0 / max(np.linalg.norm(d0), 1e-9)

    # ---- filter ----
    def propagate(self, gyro, acc):
        dt = self.dt
        w = 0.5 * (gyro + self.prev_gyro) - self.bg
        f = 0.5 * (acc + self.prev_acc) - self.ba
        # the IMU sits r = 2 cm above the tracked origin: remove the tangential and centripetal terms
        alpha = (gyro - self.prev_gyro) / dt
        f = f - np.cross(alpha, IMU_R) - np.cross(w, np.cross(w, IMU_R))
        self.prev_gyro, self.prev_acc = gyro, acc
        R = _quat_rot(_quat_mul(self.q, _quat_exp(w * (0.5 * dt))))   # attitude at mid-interval
        a = R @ f - G * E3
        self.p = self.p + self.v * dt + 0.5 * a * dt * dt
        self.v = self.v + a * dt
        self.q = _quat_mul(self.q, _quat_exp(w * dt))
        self.q /= np.linalg.norm(self.q)
        F = np.eye(15)
        F[0:3, 3:6] = np.eye(3) * dt
        F[3:6, 6:9] = -R @ _skew(f) * dt
        F[3:6, 12:15] = -R * dt
        F[6:9, 6:9] = np.eye(3) - _skew(w) * dt
        F[6:9, 9:12] = -np.eye(3) * dt
        self.Pc = F @ self.Pc @ F.T + self.Q
        self.hist_p.append(self.p.copy())
        self.hist_p.pop(0)
        Rn = _quat_rot(self.q)
        self.hist_yaw.append(math.atan2(Rn[1, 0], Rn[0, 0]))
        self.hist_yaw.pop(0)

    def update(self, zp, zyaw):
        R = _quat_rot(self.q)
        y = np.zeros(4)
        y[0:3] = zp - self.hist_p[0]
        y[3] = _wrap(zyaw - self.hist_yaw[0])
        H = np.zeros((4, 15))
        H[0:3, 0:3] = np.eye(3)
        H[0:3, 3:6] = -np.eye(3) * (self.delay * self.dt)
        n = R[0, 0] ** 2 + R[1, 0] ** 2
        H[3, 7] = (R[1, 0] * R[0, 2] - R[0, 0] * R[1, 2]) / n
        H[3, 8] = (R[0, 0] * R[1, 1] - R[1, 0] * R[0, 1]) / n
        Rm = np.diag([PARAMS["r_pos"] ** 2] * 3 + [0.006 ** 2])
        PHt = self.Pc @ H.T
        S = H @ PHt + Rm
        K = PHt @ np.linalg.inv(S)
        dx = K @ y
        self.Pc = (np.eye(15) - K @ H) @ self.Pc
        self.Pc = 0.5 * (self.Pc + self.Pc.T)
        self.p = self.p + dx[0:3]
        self.v = self.v + dx[3:6]
        self.q = _quat_mul(self.q, _quat_exp(dx[6:9]))
        self.q /= np.linalg.norm(self.q)
        self.bg = self.bg + dx[9:12]
        self.ba = self.ba + dx[12:15]
        for i in range(len(self.hist_p)):
            self.hist_p[i] = self.hist_p[i] + dx[0:3]
        Rn = _quat_rot(self.q)
        dyaw = _wrap(math.atan2(Rn[1, 0], Rn[0, 0]) - math.atan2(R[1, 0], R[0, 0]))
        for i in range(len(self.hist_yaw)):
            self.hist_yaw[i] = self.hist_yaw[i] + dyaw


_S = None


def _choose_scale(u_lim):
    """Smallest time stretch (0.85..1.4) at which the reference thrust stays below u_lim: the non-gravity part of
    the thrust (inertia and quadratic drag) scales exactly as 1 / scale^2."""
    g = M_NOM * G * E3
    for lam in np.arange(0.85, 1.4, 0.005):
        if np.max(np.linalg.norm(g + (_U - g) / lam ** 2, axis=1)) <= u_lim:
            return float(lam)
    return 1.4


def _reference(tau, scale):
    """Reference (p, v, u, w) at trajectory time tau (scaled by `scale`)."""
    s = tau / scale
    if s <= _T[0]:
        return _P[0].copy(), np.zeros(3), _U[0].copy(), np.zeros(3), np.zeros(3), False
    if s >= _T[-1]:
        u = _U[-1].copy()
        if scale != 1.0:
            u = (u - M_NOM * G * E3) / scale ** 2 + M_NOM * G * E3
        return (_P[-1] + _V[-1] * (s - _T[-1]), _V[-1] / scale, u, np.zeros(3), np.zeros(3),
                bool(s > _T[-1] + 0.5))
    i = int(np.searchsorted(_T, s) - 1)
    h = _T[i + 1] - _T[i]
    x = (s - _T[i]) / h
    # cubic Hermite on position with node velocities
    h00 = 2 * x ** 3 - 3 * x ** 2 + 1
    h10 = x ** 3 - 2 * x ** 2 + x
    h01 = -2 * x ** 3 + 3 * x ** 2
    h11 = x ** 3 - x ** 2
    p = h00 * _P[i] + h10 * h * _V[i] + h01 * _P[i + 1] + h11 * h * _V[i + 1]
    v = (1 - x) * _V[i] + x * _V[i + 1]
    u = (1 - x) * _U[i] + x * _U[i + 1]
    w = (1 - x) * _W[i] + x * _W[i + 1]
    wd = (1 - x) * _WD[i] + x * _WD[i + 1]
    if scale != 1.0:
        u = (u - M_NOM * G * E3) / scale ** 2 + M_NOM * G * E3
    return p, v / scale, u, w / scale, wd / scale ** 2, False


def controller(obs: dict) -> np.ndarray:
    global _S
    if int(obs["step"]) == 0 or _S is None:
        _S = _State(obs)
    st = _S
    prm = PARAMS
    gyro = np.asarray(obs["gyro"], dtype=float)
    acc = np.asarray(obs["accel"], dtype=float)
    t = float(obs["time"])
    if int(obs["step"]) > 0:
        st.propagate(gyro, acc)
    if bool(obs["tracker_fresh"]) and int(obs["step"]) > 0:
        st.update(np.asarray(obs["tracker_pos"], dtype=float), float(obs["tracker_yaw"]))

    R = _quat_rot(st.q)
    w_meas = gyro - st.bg
    t_launch = prm["t_launch"]
    sp0 = np.asarray(obs["start_pos"], dtype=float)
    hover_pt = sp0 + (_P[0] - sp0) * min(1.0, prm["hover_back"] / max(1e-6, float(np.linalg.norm(_P[0] - sp0))))
    if t < t_launch:
        # glide from the initial position to the launch point (inside the start sphere, clock not running)
        sgl = min(1.0, t / prm["glide_time"])
        sgl = sgl * sgl * (3.0 - 2.0 * sgl)
        p_ref = st.p_init + (hover_pt - st.p_init) * sgl
        v_ref, u_ref, w_ref, wd_ref = np.zeros(3), M_NOM * G * E3, np.zeros(3), np.zeros(3)
        if t > 0.6:
            st.keff_num += (acc[2] - st.ba[2])
            st.keff_den += np.sum(st.f_hat) / M_NOM
        racing = False
    else:
        if st.t_launch is None:
            st.t_launch = t
            if st.keff_den > 0:
                st.keff = float(np.clip(st.keff_num / st.keff_den, 0.8, 1.25))
            if st.fh_n > 0 and prm["wind_gain"] > 0.0:
                # hover: the commanded horizontal force balances the wind drag c |w| w  ->  invert for w
                dw = -st.fh_sum / st.fh_n
                dn = float(np.linalg.norm(dw))
                if dn > 1e-6:
                    wv = dw / math.sqrt(0.0233 * dn)
                    wn = float(np.linalg.norm(wv))
                    if wn > 9.0:
                        wv *= 9.0 / wn
                    st.wind[0:2] = prm["wind_gain"] * wv
            st.scale = min(prm["scale_max"], max(prm["scale_min"], _choose_scale(st.keff * prm["u_frac"] * 4.0 * FMAX))) * prm["time_scale"]
        # time-warped reference: the trajectory clock slows when the vehicle is behind along the path
        p_ref, v_ref, _, _, _, done = _reference(st.tau, st.scale)
        vn = float(np.linalg.norm(v_ref))
        if vn > 0.5:
            e_al = float((p_ref - st.p) @ v_ref) / vn
            rate = min(prm["warp_max"], max(prm["warp_min"], 1.0 - prm["k_warp"] * e_al))
        else:
            rate = 1.0
        # feed-forward (thrust vector, rates) taken ff_lead ahead to cover the motor and attitude lag
        _, _, u_ref, w_ref, wd_ref, _ = _reference(st.tau + prm["ff_lead"], st.scale)
        st.tau += st.dt * rate
        racing = True
        if done:
            p_ref, v_ref, u_ref, w_ref, wd_ref = np.asarray(obs["target_pos"], float), np.zeros(3), M_NOM * G * E3, np.zeros(3), np.zeros(3)

    e_p = p_ref - st.p
    e_v = v_ref - st.v
    if racing:
        kp, kd = prm["kp"], prm["kd"]
    else:
        kp, kd = prm["kp_h"], prm["kd_h"]
    st.integ += e_p * st.dt
    st.integ = np.clip(st.integ, -1.0, 1.0)
    a_fb = kp * e_p + kd * e_v + prm["ki"] * st.integ
    vn_r = float(np.linalg.norm(v_ref))
    if racing and vn_r > 1.0:
        # along-track errors are absorbed by the clock warp: weaker gains along the path, stronger across it
        tt = v_ref / vn_r
        ep_a = float(e_p @ tt) * tt
        ev_a = float(e_v @ tt) * tt
        a_fb = (prm["kp_al"] * ep_a + prm["kp_cr"] * (e_p - ep_a) + prm["kd_al"] * ev_a + prm["kd_cr"] * (e_v - ev_a)
                + prm["ki"] * st.integ)
    F = u_ref + M_NOM * a_fb
    if racing and prm["drag_beta"] > 0.0:
        # replace the plan's drag (at the plan's attitude and speed) by the drag at the current attitude and speed
        s_ = min(max(st.tau / st.scale, _T[0]), _T[-1])
        i_ = min(int(np.searchsorted(_T, s_)) - 1, len(_T) - 2)
        i_ = max(i_, 0)
        x_ = (s_ - _T[i_]) / (_T[i_ + 1] - _T[i_])
        d_plan = ((1 - x_) * _DPL[i_] + x_ * _DPL[i_ + 1]) / st.scale ** 2
        F = F + prm["drag_beta"] * (d_plan - _drag(R, st.v - st.wind))
    if not racing and t > t_launch - prm["wind_avg"]:
        st.fh_sum += F[0:2]
        st.fh_n += 1
    # saturation: if the demanded force exceeds what the rotors can give (minus a margin for attitude torques),
    # give up the along-track component first and keep the cross-track (gate-relevant) correction
    f_lim = prm["sat_frac"] * 4.0 * FMAX * st.keff
    if racing and float(F @ F) > f_lim * f_lim:
        vn_ = float(np.linalg.norm(v_ref))
        if vn_ > 1.0:
            tt = v_ref / vn_
            fa = float(F @ tt)
            fperp = F - fa * tt
            pn2 = float(fperp @ fperp)
            if pn2 < f_lim * f_lim:
                fa = math.copysign(math.sqrt(f_lim * f_lim - pn2), fa)
                F = fperp + fa * tt
            else:
                F = fperp * (f_lim / math.sqrt(pn2))
    Fn = float(np.linalg.norm(F))
    zd = F / Fn
    T_cmd = max(float(F @ R[:, 2]), 0.15 * Fn) / st.keff

    # reduced attitude: rotate body z onto zd (no singularity at large tilt); yaw is a weak separate loop
    zb = R[:, 2]
    nax = np.cross(zb, zd)
    sn = float(np.linalg.norm(nax))
    ang = math.atan2(sn, float(zb @ zd))
    e_rp = R.T @ (nax * (ang / sn)) if sn > 1e-9 else np.zeros(3)
    psi = math.atan2(R[1, 0], R[0, 0])
    e_psi = _wrap(prm["yaw"] - psi) if R[2, 2] > 0.5 else 0.0
    # rate feed-forward from the motion of the desired thrust axis itself (plan + feedback), filtered
    if st.zd_prev is None:
        st.zd_prev = zd.copy()
    wz = np.cross(st.zd_prev, zd) / st.dt
    st.zd_prev = zd.copy()
    st.wz_f += prm["wz_alpha"] * (wz - st.wz_f)
    w_world = prm["w_ff"] * w_ref + prm["wz_ff"] * st.wz_f
    w_d = R.T @ w_world
    a_d = R.T @ wd_ref * prm["w_ff"]
    w_d[2] = 0.0
    a_d[2] = 0.0
    alpha = np.empty(3)
    # cascade: rate command from the angle error, capped by a braking profile sqrt(2 a_max |e|) so that a large
    # attitude step under torque saturation does not overshoot; then rate loop with feed-forward
    e_n = float(np.linalg.norm(e_rp[0:2]))
    if e_n > 1e-9:
        r_cmd = min(prm["kr"] / prm["kw"] * e_n, math.sqrt(2.0 * prm["a_max"] * e_n)) / e_n
    else:
        r_cmd = 0.0
    alpha[0:2] = prm["kw"] * (r_cmd * e_rp[0:2] + w_d[0:2] - w_meas[0:2]) + a_d[0:2]
    alpha[2] = prm["kr_yaw"] * e_psi - prm["kw_yaw"] * w_meas[2]
    tau = J_NOM @ alpha + np.cross(w_meas, J_NOM @ w_meas)

    # mixing with priorities: roll/pitch > collective > yaw
    f_rp = MIX_INV[:, 1] * tau[0] + MIX_INV[:, 2] * tau[1]
    f_y = MIX_INV[:, 3] * tau[2]
    f_c = MIX_INV[:, 0] * T_cmd
    span = f_rp.max() - f_rp.min()
    if span > FMAX:
        f_rp *= FMAX / span
    f = f_c + f_rp + f_y
    if f.max() > FMAX or f.min() < 0.0:
        f = f_c + f_rp
        lo, hi = -f_rp.min(), FMAX - f_rp.max()
        c = min(max(T_cmd / 4.0, lo), hi)
        f = c + f_rp
        # add as much yaw as fits
        room_hi = FMAX - f
        room_lo = f
        s = 1.0
        for i in range(4):
            if f_y[i] > 1e-12:
                s = min(s, room_hi[i] / f_y[i])
            elif f_y[i] < -1e-12:
                s = min(s, room_lo[i] / -f_y[i])
        f = f + max(s, 0.0) * f_y
    st.dbg = (F.copy(), zd.copy(), T_cmd, f.copy())
    # motor lag compensation
    u = st.f_hat + prm["lag_gain"] * (f - st.f_hat)
    u = np.clip(u, 0.0, FMAX)
    st.f_hat = st.f_hat + (u - st.f_hat) * st.lag
    return u
