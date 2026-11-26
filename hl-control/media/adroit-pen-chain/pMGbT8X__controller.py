"""Pen reorientation controller for the Adroit hand (joint-level position set points).

Control law
-----------
The target axis is Rx(roll) Ry(pitch) z.  The two wrist joints turn the whole grasp exactly about world x
(flexion, WRJ0) and about world z (deviation, WRJ1), so the law has two layers.

1. Grasp layer (open loop).  A grasp is a set-point vector, or a short sequence of them, found offline on a copy
   of the plant by derivative-free search and replayed here.  Each grasp holds the pen at a fixed pitch in the hand
   frame firmly enough to be carried by the wrist:
     lift     the little finger pins the pen's lower end on the palm edge, the thumb props the upper end; the
              coordinate s in [0, 1] blends POSE_LIFT_LO -> POSE_LIFT_HI (pen pitch about 0.8 -> 1.0 rad).
     upright  three-pose entry (UP_A, UP_B, UP_C): the wrist deviation shifts the palm under the pen while the pen
              is still falling at reset, then the thumb stands the pen up (pitch about 0).  The pen does not follow
              the wrist in this grasp, so its flexion range is kept small and it only serves near-vertical targets.
     left     the same kind of entry carried a little past the vertical: the pen stands on the little-finger side of
              the palm, tilted about -0.2 rad in pitch, and follows the wrist flexion between about -0.75 and +0.2.
     right    the same kind of entry stopped short of the vertical (pitch about +0.4 -> +0.3), acquired with the wrist
              fully extended; raising the wrist rolls the pen from about -0.1 to about +0.4 rad, in that direction
              only (the offset limit 0.60 stops short of the wrist angle at which the pen topples).
   At the first step the grasp whose reachable set (pitch, wrist flexion, wrist deviation ranges in GRASPS)
   contains the best match to the target is chosen; there is no regrasp later in the episode.
2. Wrist layer (feedback).  With a the measured pen axis and t the target axis, e = a x t is the rotation still to
   be made.  Its components drive three clipped, rate-limited integrators:
       wrist flexion   psi   += clip(ki_roll  * e_x)    rotation about world x
       wrist deviation delta += clip(ki_yaw   * e_z)    rotation about world z
       lift coordinate s     += clip(ki_pitch * e_y)    pitch in the hand frame (lift grasp only)

Gains (PARAMS)
--------------
ki_roll, ki_yaw, ki_pitch : integral gains per 10 ms step of the three channels.
rate, rate_s              : largest change per step of psi / delta [rad] and of s (keeps the motion quasi-static).
t_lift                    : steps of the open-loop lift before the wrist layer is switched on.
reach_cos                 : a new target is chased only if the grasp's reachable set comes within this cosine of it;
                            otherwise the wrist stays where it is (chasing an unreachable target can topple the pen).

Known limits (measured, see the evaluator): the lift grasp reaches targets with pitch above about +0.5 rad, the
upright grasp those within about 0.3 rad of the vertical, the left grasp pitch between about -0.5 and +0.1 rad with
roll below about +0.3 rad and the right grasp pitch between about 0 and +0.7 rad with roll between about -0.4 and
+0.7 rad; pitch below -0.5 rad is not reached, and there is no transition between grasps, so a chain only continues
while successive targets stay inside the reachable set of the grasp chosen for the first target.  This is far below
the five-target gate.
"""

import numpy as np

PARAMS = {"ki_roll": 0.12, "ki_yaw": 0.10, "ki_pitch": 0.08, "rate": 0.035, "rate_s": 0.05, "t_lift": 45, "reach_cos": 0.93}
PARAM_BOUNDS = {"ki_roll": (0.03, 0.3), "ki_yaw": (0.0, 0.3), "ki_pitch": (0.0, 0.3), "rate": (0.01, 0.08),
                "rate_s": (0.01, 0.2), "t_lift": (30, 70), "reach_cos": (0.85, 0.97)}

# Set-point vectors in action units (one per actuator, in actuator order).
POSE_LIFT_HI = np.array([0.583536656192362, 0.10851961668912455, 0.30528347526374777, -0.6081062757339781, 0.1723462277037002, 0.1387477845851759, -0.009981825244973305, -0.19303227603043277, -0.44321042728194027, 0.22168274012170358, -0.4105733747170287, 0.21575041821539165, 0.2210805475805171, -0.3181220322938407, -0.8737248752443294, 0.9426149839480656, 0.4722131353152274, 0.700925970068177, 0.6628767057139059, 0.434352679552962, 0.5172682514956427, 0.5890651331941845, -0.19460414624577202, 0.45537945080830866])
POSE_LIFT_LO = np.array([0.4959942035952003, -0.019276072478085007, -0.188382611663999, -0.12094482758807368, -0.01866762651996874, 0.3471872662612777, 0.7260457184634597, -0.13198974906418431, -0.3754109947018449, 0.2158707155181723, 1.0, -0.09898343080007388, -0.6048455783318417, -0.3601189516696192, -0.6465531595034507, 1.0, 0.8596011279910628, -0.0008594484345163245, 0.33792685721043164, 0.5718344816138485, 0.9848510435668726, 1.0, -0.3204025190436802, 0.6264849655071959])
UP_A = np.array([-1.0, -0.7655564930235963, 0.3799163668031988, -0.6726194764474922, -0.34768385385696743, -0.8327197062140869, 0.026012910385136184, -0.9226426573672983, -0.42375152254118936, -0.6917883744755348, 0.06435178310663675, -0.7608944886639812, -0.7139370551959862, -0.15346165412333304, -0.999200769835083, -0.5782661162562339, -0.5324862099312367, -1.0, -0.998350246100344, -0.2791976312753014, -0.8050941658352587, -0.3195230252207044, 0.25064160259192714, 0.988182007889076])
UP_B = np.array([-1.0, -0.16471983951198257, 0.8864268629476391, 0.934648510193167, -0.5657310806391075, -0.8824770824243581, -0.5696248066534415, 0.26667936201030706, -0.3815873162525071, -1.0, -0.5832176524100855, 0.6066652203702424, -0.9478714902411647, -0.25375085396654706, -1.0, -0.44972066687691914, 0.6557538687080231, -0.9904465709173013, -0.7113619262135799, 1.0, 1.0, 0.6765370300252316, 0.47951259897811244, -0.24528469616137294])
UP_C = np.array([0.32117758225564524, -0.16471983951198257, 0.8864268629476391, 0.934648510193167, -0.5657310806391075, -0.8824770824243581, -0.5696248066534415, 0.26667936201030706, -0.3815873162525071, -1.0, -0.5832176524100855, 0.6066652203702424, -0.9478714902411647, -0.25375085396654706, -1.0, -0.44972066687691914, 0.6557538687080231, -0.9904465709173013, -0.7113619262135799, 1.0, 1.0, 0.6765370300252316, 0.47951259897811244, -0.24528469616137294])
LEFT_A = np.array([-0.9522109918730961, -0.6762389083824862, 0.2615772067385433, -0.994360929325043, -0.7342966283074704, -0.829388708040542, -0.3853917259164558, -1.0, -0.00180612318238553, -0.28121662190718677, -0.12262351076493669, -1.0, -0.16220784025161425, -0.19702145206702765, -0.7366514578667305, -0.7127945045994006, -0.3350292937959142, -0.9854860687580888, -0.780758348748628, -0.3935574778041325, -0.8566144520421136, -0.7855208273356741, 0.22897203878520972, 1.0])
LEFT_B = np.array([-0.690934552100485, -0.31842846206685543, 0.6179444802310816, 0.9529118186817678, -0.8384060041301695, -0.5975635420510147, -0.3784187363710634, 1.0, -0.555185275967716, -0.6478454667862715, -0.6340657329613741, 0.6229456940197079, -0.9594939164819579, -0.16810161196769946, -0.9716357320397297, -0.8679477954146694, 1.0, -0.9028525958447973, -0.8485262159232759, 0.8951032564969755, 1.0, 0.8755959192636089, 0.8246141302070441, -0.456565759677668])
LEFT_C = np.array([0.2818501856981834, -0.31842846206685543, 0.6179444802310816, 0.9529118186817678, -0.8384060041301695, -0.5975635420510147, -0.3784187363710634, 1.0, -0.555185275967716, -0.6478454667862715, -0.6340657329613741, 0.6229456940197079, -0.9594939164819579, -0.16810161196769946, -0.9716357320397297, -0.8679477954146694, 1.0, -0.9028525958447973, -0.8485262159232759, 0.8951032564969755, 1.0, 0.8755959192636089, 0.8246141302070441, -0.456565759677668])
RIGHT_A = np.array([-1.0, -0.8347835447672138, 0.4527464120839972, -0.5953954938219724, -0.237812340122071, -1.0, -0.04679340339605215, -0.4317705594468264, -0.586991476021024, -0.67538284610578, -0.18947158680263496, -0.7317539331211078, -0.9033745263625784, -0.08698516609236348, -1.0, -0.681959300352104, -0.726441018174721, -0.886200987064886, -0.9066075146552008, -0.5445391242682062, -0.9584379308696999, -0.6028232161992519, 0.5615736121812175, 0.9218771612176221])
RIGHT_B = np.array([-0.7941291581827539, -1.0, 0.8890455700471976, 0.9719120187188073, -0.2961082691079102, -1.0, -0.6194757958268172, 0.1949128990731809, -0.3600204449326394, -0.648352371368145, -0.5008658761506353, 0.5690027543387618, -0.872978477508986, -0.45917842339633924, -0.8887305878320023, -0.8811202637620252, 0.9552841548173434, -0.6602053553607764, -0.7688237442514945, 0.3929225345088579, 0.908054559642056, 0.9433784379862151, 0.5936191582541551, -0.4457625917845716])
RIGHT_C = np.array([0.5868008118291864, -1.0, 0.8890455700471976, 0.9719120187188073, -0.2961082691079102, -1.0, -0.6194757958268172, 0.1949128990731809, -0.3600204449326394, -0.648352371368145, -0.5008658761506353, 0.5690027543387618, -0.872978477508986, -0.45917842339633924, -0.8887305878320023, -0.8811202637620252, 0.9552841548173434, -0.6602053553607764, -0.7688237442514945, 0.3929225345088579, 0.908054559642056, 0.9433784379862151, 0.5936191582541551, -0.4457625917845716])

# Reachable set of each grasp [rad]: pen pitch in the hand frame, wrist-flexion offset psi from the grasp pose (the pen's
# roll is roll0 + psi) and wrist deviation (yaw).  The limits keep the pen centre inside the 0.075 m ball.
GRASPS = {
    "lift": {"pitch": (0.80, 1.03), "roll0": 0.0, "psi": (-0.72, 0.55), "yaw": (-0.15, 0.30)},
    "upright": {"pitch": (0.0, 0.0), "roll0": 0.0, "psi": (-0.20, 0.20), "yaw": (0.0, 0.0), "entry": (UP_A, UP_B, UP_C)},
    "left": {"pitch": (-0.30, -0.15), "roll0": -0.32, "psi": (-0.33, 0.30), "yaw": (0.0, 0.0), "rate": 0.02, "entry": (LEFT_A, LEFT_B, LEFT_C)},
    "right": {"pitch": (0.36, 0.45), "roll0": -0.13, "psi": (0.0, 0.60), "yaw": (0.0, 0.0), "rate": 0.02, "entry": (RIGHT_A, RIGHT_B, RIGHT_C)},
}
ENTRY_STEPS = (10, 70, 30)   # steps spent on the entry poses A, B, C
ENTRY_RAMPS = (3, 20, 15)    # steps over which each entry pose is blended in

_S = {}   # per-episode memory, re-initialised when obs["step"] == 0


def _ramp(a_from, a_to, k, n):
    """Linear blend from a_from to a_to, complete after n steps (k = steps since the blend started)."""
    return a_from + min(1.0, (k + 1) / n) * (a_to - a_from)


def _best_match(grasp, target):
    """Largest cosine between the target and Rz(yaw) Rx(psi) (sin p, 0, cos p) over the grasp's ranges."""
    p = np.linspace(*grasp["pitch"], 4)[:, None, None]
    psi = grasp["roll0"] + np.linspace(*grasp["psi"], 25)[None, :, None]   # pen roll = roll0 + wrist offset
    yaw = np.linspace(*grasp["yaw"], 7)[None, None, :]
    x, y, z = np.sin(p), -np.sin(psi) * np.cos(p), np.cos(psi) * np.cos(p)
    xw, yw = np.cos(yaw) * x - np.sin(yaw) * y, np.sin(yaw) * x + np.cos(yaw) * y
    return float(np.max(xw * target[0] + yw * target[1] + z * target[2]))


def controller(obs: dict) -> np.ndarray:
    o = np.asarray(obs["observation"], dtype=float)
    step = int(obs["step"])
    mean, half = np.asarray(obs["ctrl_mean"], dtype=float), np.asarray(obs["ctrl_half_range"], dtype=float)
    axis, target = o[33:36], o[36:39]
    p = PARAMS
    if step == 0 or not _S:
        _S.clear()
        name = max(GRASPS, key=lambda g: _best_match(GRASPS[g], target))
        _S.update(grasp=name, psi=0.0, delta=0.0, s=1.0, open=-mean / half, index=-1, chase=True)
    st = _S
    grasp = GRASPS[st["grasp"]]
    if int(obs.get("target_index", 0)) != st["index"]:   # new target: only chase it if this grasp can reach it
        st["index"] = int(obs.get("target_index", 0))
        st["chase"] = _best_match(grasp, target) > p["reach_cos"]
    a_open = st["open"]                       # set points of the flat open hand (all joints at zero)

    # ---- grasp layer: open-loop acquisition from the reset state ----
    if "entry" in grasp:
        pa, pb, pc = grasp["entry"]
        n1, n2, n3 = ENTRY_STEPS
        if step < n1:
            return np.clip(_ramp(a_open, pa, step, ENTRY_RAMPS[0]), -1.0, 1.0)
        if step < n1 + n2:
            return np.clip(_ramp(pa, pb, step - n1, ENTRY_RAMPS[1]), -1.0, 1.0)
        if step < n1 + n2 + n3:
            return np.clip(_ramp(pb, pc, step - n1 - n2, ENTRY_RAMPS[2]), -1.0, 1.0)
        base = pc
    else:
        if step < int(p["t_lift"]):
            return np.clip(_ramp(a_open, POSE_LIFT_HI, step, 15), -1.0, 1.0)
        base = POSE_LIFT_LO + st["s"] * (POSE_LIFT_HI - POSE_LIFT_LO)

    # ---- wrist layer: e = axis x target is the rotation still to be made ----
    e = np.cross(axis, target) if st["chase"] else np.zeros(3)   # an unreachable target leaves the wrist where it is
    r = grasp.get("rate", p["rate"])          # a grasp may ask for a slower wrist
    st["psi"] = float(np.clip(st["psi"] + np.clip(p["ki_roll"] * e[0], -r, r), *grasp["psi"]))
    st["delta"] = float(np.clip(st["delta"] + np.clip(p["ki_yaw"] * e[2], -r, r), *grasp["yaw"]))
    if "entry" not in grasp:
        st["s"] = float(np.clip(st["s"] + np.clip(p["ki_pitch"] * e[1], -p["rate_s"], p["rate_s"]), 0.0, 1.0))
    a = base.copy()
    a[1] = base[1] + st["psi"] / half[1]      # wrist flexion: rotation about world +x
    a[0] = base[0] - st["delta"] / half[0]    # wrist deviation: the joint axis is world -z
    return np.clip(a, -1.0, 1.0)
