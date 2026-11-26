"""Decentralized longitudinal controller for a 20-vehicle platoon.

This module implements a decentralized longitudinal controller for a platoon of
heterogeneous vehicles under variable observation delays, measurement noise,
actuator lag, and predecessor acceleration dropouts.

Control Architecture:
1. Exact Online Parameter Identification:
   Identifies each follower's true actuator lag constant alpha_i = dt / tau_i online
   from noise-free historical acceleration and command records.

2. Delay-Compensated Predictor & State Observer:
   Maintains a rolling historical buffer and forward-propagates the vehicle's own
   exact state (velocity and acceleration) through its known actuator dynamics to the
   current time step k. Uses an observer to estimate predecessor velocity, gap,
   and predecessor acceleration with jerk extrapolation and dropout handling.

3. Decentralized Feedback Linearization (CACC Law):
   Imposes desired second-order spacing error dynamics:
     ddot{e} + kv * dot{e} + kp * e = 0
   where e = gap - (d0 + h * v_own).
   The resulting command:
     u = (tau / h) * (a_pred + kv * dot{e} + kp * e) + (1 - tau / h) * a_own
   cancels the vehicle's actuator lag, guarantees strong high-frequency disturbance
   attenuation (since tau / h < 1), and prevents collision and error amplification.

Gains and Parameters:
- kp = 1.00: Proportional gain on spacing error.
- kv = 1.50: Damping gain on relative velocity / error derivative.
- ka_ff = 1.00: Feedforward gain on estimated predecessor acceleration.
- drop_gain = 1.00: Predecessor acceleration innovation gain during dropout steps.
"""

from __future__ import annotations
import numpy as np


class PlatoonPredictor:
    """Sliding-window state observer and delay compensator."""

    def __init__(
        self,
        dt: float = 0.1,
        l_gap: float = 0.35,
        l_dv: float = 0.25,
        l_bv: float = 0.012,
        j_decay: float = 0.85,
        drop_gain: float = 1.0,
    ) -> None:
        self.dt = dt
        self.dt2 = 0.5 * dt * dt
        self.alpha = 0.1 / 0.45
        self.commands: dict[int, float] = {}
        self.exact_a: dict[int, float] = {}
        self.exact_v: dict[int, float] = {}
        self.obs_buf: dict[int, tuple[float, float, float, float, float]] = {}

        self.gap: float | None = None
        self.dv = 0.0
        self.apred = 0.0
        self.jpred = 0.0
        self.bv = 0.0
        self.anchor_step = 0

        self.l_gap = l_gap
        self.l_dv = l_dv
        self.l_bv = l_bv
        self.j_decay = j_decay
        self.drop_gain = drop_gain

    def update_and_predict(self, obs: dict) -> tuple[float, float, float, float, float, float]:
        k = int(obs["step"])
        d = int(obs["obs_delay_steps"])
        m = k - d

        if self.gap is None:
            self.gap = float(obs["gap_front"])
            self.dv = float(obs["relative_velocity_front"])
            self.anchor_step = m

        self.obs_buf[m] = (
            float(obs["gap_front"]),
            float(obs["relative_velocity_front"]),
            float(obs["predecessor_acceleration"]),
            float(obs["own_acceleration"]),
            float(obs["own_velocity"]),
        )
        self.exact_a[m] = float(obs["own_acceleration"])
        self.exact_v[m] = float(obs["own_velocity"])

        # Exact actuator lag identification
        if m - 1 in self.exact_a and m - 1 in self.commands:
            denom = float(np.clip(self.commands[m - 1], -obs["brake_limit"], obs["accel_limit"])) - self.exact_a[m - 1]
            if abs(denom) > 0.02:
                meas_alpha = (self.exact_a[m] - self.exact_a[m - 1]) / denom
                if 0.05 <= meas_alpha <= 0.8:
                    self.alpha = float(meas_alpha)

        # Propagate own acceleration and velocity exactly to step k
        cur_a = self.exact_a[m]
        cur_v = self.exact_v[m]
        for s in range(m, k):
            u_c = float(np.clip(self.commands.get(s, 0.0), -obs["brake_limit"], obs["accel_limit"]))
            cur_a += self.alpha * (u_c - cur_a)
            cur_v = float(np.clip(cur_v + self.dt * cur_a, 0.0, obs["speed_limit"]))
            self.exact_a[s + 1] = cur_a
            self.exact_v[s + 1] = cur_v

        # Advance anchor step to prevent memory growth
        if k - self.anchor_step > 18:
            target_anchor = k - 12
            for s in range(self.anchor_step, target_anchor):
                self._step_anchor(s)
            self.anchor_step = target_anchor

        # Forward simulate state from anchor to current step k
        gap = self.gap
        dv = self.dv
        apred = self.apred
        jpred = self.jpred
        bv = self.bv
        for s in range(self.anchor_step, k):
            if s in self.obs_buf:
                z_gap, z_v, z_a, _, _ = self.obs_buf[s]
                err_gap = z_gap - gap
                err_v = z_v - (dv + bv)

                gap += self.l_gap * err_gap
                dv += self.l_dv * err_v + 0.15 * err_gap
                bv += self.l_bv * err_v

                if z_a != 0.0:
                    meas_j = (z_a - apred) / self.dt
                    jpred = 0.5 * jpred + 0.5 * float(np.clip(meas_j, -15.0, 15.0))
                    apred = z_a
                else:
                    apred += self.drop_gain * err_v

            a_own_next = self.exact_a.get(s + 1, self.exact_a.get(s, 0.0))
            apred_next = float(np.clip(apred + self.dt * jpred, -4.5, 2.0))
            jpred = self.j_decay * jpred

            gap += self.dt * dv + self.dt2 * (apred_next - a_own_next)
            dv += self.dt * (apred_next - a_own_next)
            apred = apred_next

        return gap, dv, apred, bv, self.exact_a[k], self.exact_v[k]

    def _step_anchor(self, s: int) -> None:
        if s in self.obs_buf:
            z_gap, z_v, z_a, _, _ = self.obs_buf[s]
            err_gap = z_gap - self.gap
            err_v = z_v - (self.dv + self.bv)

            self.gap += self.l_gap * err_gap
            self.dv += self.l_dv * err_v + 0.15 * err_gap
            self.bv += self.l_bv * err_v

            if z_a != 0.0:
                meas_j = (z_a - self.apred) / self.dt
                self.jpred = 0.5 * self.jpred + 0.5 * float(np.clip(meas_j, -15.0, 15.0))
                self.apred = z_a
            else:
                self.apred += self.drop_gain * err_v

        a_own_next = self.exact_a.get(s + 1, self.exact_a.get(s, 0.0))
        apred_next = float(np.clip(self.apred + self.dt * self.jpred, -4.5, 2.0))
        self.jpred = self.j_decay * self.jpred

        self.gap += self.dt * self.dv + self.dt2 * (apred_next - a_own_next)
        self.dv += self.dt * (apred_next - a_own_next)
        self.apred = apred_next


def controller(obs: dict, memory: dict | None = None) -> tuple[float, dict]:
    """Decentralized longitudinal controller function for one vehicle follower.

    Args:
        obs: Local observation dictionary for one follower vehicle.
        memory: Per-vehicle memory dictionary retained across simulation steps.

    Returns:
        tuple[float, dict]: Commanded acceleration [m/s^2] and updated memory.
    """
    if memory is None or "est" not in memory:
        mem = {"est": PlatoonPredictor(dt=obs["dt"], drop_gain=1.0)}
    else:
        mem = memory

    est: PlatoonPredictor = mem["est"]
    k = int(obs["step"])
    gap_hat, dv_hat, apred_hat, bv_hat, own_a, own_v = est.update_and_predict(obs)

    d0 = float(obs["d0"])
    h = float(obs["time_headway"])
    dt = float(obs["dt"])

    # State errors
    desired_gap = d0 + h * own_v
    spacing_err = gap_hat - desired_gap
    e_dot = dv_hat - h * own_a

    # Gains
    kp = 1.00
    kv = 1.50
    ka_ff = 1.00

    # Actuator lag compensation via feedback linearization
    tau_i = dt / max(est.alpha, 0.05)
    ratio = tau_i / h
    raw_cmd = ratio * (ka_ff * apred_hat + kv * e_dot + kp * spacing_err) + (1.0 - ratio) * own_a

    cmd = float(np.clip(raw_cmd, -obs["brake_limit"], obs["accel_limit"]))
    est.commands[k] = cmd
    return cmd, mem
