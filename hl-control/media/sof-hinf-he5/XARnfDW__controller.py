"""Static output-feedback H-infinity controller for the COMPleib helicopter HE5.

Control law:  u = K y,  with K a constant 4 x 2 matrix (nu = 4 inputs, ny = 2 measured outputs).

The plant (nx = 8, open loop unstable, spectral abscissa 0.234) is

    dx/dt = A x + B1 w + B u,   z = C1 x + D12 u,   y = C x + D21 w.

K was obtained offline by nonsmooth optimisation of the closed-loop H-infinity norm
||T_wz||_inf over the 8 entries of K (scratch/hinfopt.py): many random starts are first
stabilised by BFGS on the spectral abscissa of A + B K C, then BFGS with a weak-Wolfe line
search on the exact H-infinity norm (Bruinsma-Steinbuch), whose gradient at the peak frequency
w* is Re(conj(Tzu(jw*)^H u1) (Tyw(jw*) v1)^T), with u1, v1 the top singular vectors of T(jw*).
The best basin (reached from ~110 of ~370 starts) was then polished by a minimax SQP
(scratch/polish.py): min t s.t. t >= each of the four equal active peaks of sigma_max(T(jw)),
at w = 0, 0.352, 0.608 and 5.78 rad/s, the nonsmooth local minimum of the norm.

Role of the gains: column 1 multiplies y1 = x3 (first measured rate), column 2 multiplies
y2 = x4 (second measured rate); row i is the feedback into actuator u_i.

Closed loop with this K: stable (spectral abscissa -0.126), H-infinity norm 8.8951609
(best published static value 8.8952, HIFOO; the gate is gamma <= 8.89525).
"""

import numpy as np

_GAINS = {
    "HE5": np.array([
        [0.5916306128176272, -0.12071632797449229],
        [11.964004173790753, 3.9294853044514757],
        [-2.414663465271149, 1.257681256521337],
        [17.530715957734213, 16.777412030574613],
    ]),
}


def controller(obs: dict) -> np.ndarray:
    """u = K y for the instance named in obs (a constant, memoryless linear map)."""
    y = np.asarray(obs["y"], dtype=float).reshape(-1)
    K = _GAINS.get(str(obs.get("instance", "HE5")))
    if K is None or K.shape != (int(obs["nu"]), y.size):
        return np.zeros(int(obs["nu"]))
    return K @ y
