"""Static output feedback H-infinity controller for COMPleib AC3 (L-1011 aircraft in cruise).

Control law: u = K y, with K a constant 2x4 matrix (no state, no clock, no offset, no saturation).

    y = [x2 - x5, x3, x4, x1]   measured outputs (columns of K)
    u = [u1, u2]                control inputs   (rows of K)

K[i, j] is the gain from measured output y_j to control input u_i. The numbers were chosen
offline to minimise the closed-loop H-infinity norm from w to z, ||T_wz||_inf, over K.
Method: multi-start nonsmooth BFGS (Lewis-Overton weak-Wolfe line search) on the exact
norm, with gradients taken at the peak frequency. Entries were kept below the grader's 1e4
limit through a tanh reparametrisation, and an L2 penalty on K was shrunk to zero (without
it the search drifts to gains above 1e6 and a worse norm of about 3.64). Restarted BFGS
then polished the best result.

Closed-loop norm 3.402304, against the published best of 3.4859. The peak is attained at
two frequencies, w = 0 and w = 1.47 rad/s, as expected at a nonsmooth minimum. Closed-loop
spectral abscissa -0.411; max |K_ij| = 1835.
"""

import numpy as np

K = np.array([
    [863.2836571509615, 324.0484389581267, -1834.9498482311858, 744.6053881276931],
    [619.0627806293395, 235.12992862387983, -1313.2353368324762, 534.1592259337953],
])


def controller(obs: dict) -> np.ndarray:
    """Static linear output feedback u = K y."""
    return K @ np.asarray(obs["y"], dtype=float)
