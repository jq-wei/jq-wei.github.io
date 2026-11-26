"""Two-link arm controller under unknown payload and actuator lag/delay.

Control Architecture:
1. State Estimation & Noise Filtering:
   - First-order exponential filtering with predictive correction (alpha_q=0.8, alpha_qd=0.6)
     to reject sensor noise on joint angles (sigma=1e-3 rad) and velocities (sigma=1e-2 rad/s).
2. Anticipatory Feedforward:
   - Small lead time (t_lead=15 ms) compensating for the actuator first-order lag (tau=20 ms)
     and 1-step discrete delay (10 ms).
   - Minimum-jerk acceleration feedforward with exact Jacobian derivative compensation
     mapped through the configuration-dependent mass matrix M(q_lead, payload_hat).
3. Continuous Adaptive Payload Estimation:
   - Slotine-Li regressor-based payload mass estimation:
     Y_pay(q, qd, qdd) = M_pay(q) * qdd_lead + C_pay(q, qd) + G_pay(q)
   - Online adaptation law driven by composite filtered tracking error:
     d(payload_hat)/dt = gamma * (Y_pay^T M [Kp*e + Kd*de]) / (||Y_pay||^2 + eps)
     bounded to [0, 1.0] kg.
4. Decoupled Computed-Torque Feedback:
   - Decoupled feedback using configuration mass matrix M(q_lead) with critically damped
     joint bandwidth (Kp=[175.0, 135.0], Kd=[25.5, 17.5]) ensuring fast disturbance recovery
     from pulses and payload transitions without torque saturation or limit cycles.
"""

from __future__ import annotations

import math
import numpy as np
from hl_control.envs.two_link_arm import forward_kinematics, jacobian, inverse_kinematics, min_jerk


class ArmController:
    """Adaptive computed-torque controller for the two-link planar arm."""

    def __init__(
        self,
        kp: tuple[float, float] = (175.0, 135.0),
        kd: tuple[float, float] = (25.5, 17.5),
        t_lead: float = 0.015,
        gamma_payload: float = 4.5,
        alpha_q: float = 0.8,
        alpha_qd: float = 0.6,
    ):
        self.kp = np.array(kp, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.t_lead = float(t_lead)
        self.gamma_payload = float(gamma_payload)
        self.alpha_q = float(alpha_q)
        self.alpha_qd = float(alpha_qd)
        self.reset()

    def reset(self) -> None:
        self.step = -1
        self.q_est = np.zeros(2)
        self.qd_est = np.zeros(2)
        self.payload_hat = 0.0
        self.p0 = np.zeros(2)
        self.last_segment = -1
        self.last_target = np.zeros(2)

    def _eval_ref(
        self, tau: float, move_time: float, p0: np.ndarray, p1: np.ndarray, link_len: tuple[float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        l1, l2 = float(link_len[0]), float(link_len[1])
        if tau < move_time and move_time > 0:
            u = min(1.0, max(0.0, tau / move_time))
            s, s_dot = min_jerk(u)
            s_ddot = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
            ref_xy = p0 + (p1 - p0) * s
            ref_xy_dot = (p1 - p0) * (s_dot / move_time)
            ref_xy_ddot = (p1 - p0) * (s_ddot / (move_time**2))
        else:
            ref_xy = p1.copy()
            ref_xy_dot = np.zeros(2)
            ref_xy_ddot = np.zeros(2)

        q_des = inverse_kinematics(ref_xy, link_len, elbow_up=True)
        J_d = jacobian(q_des, link_len)
        qd_des = np.linalg.solve(J_d, ref_xy_dot)

        q1_d, q2_d = q_des[0], q_des[1]
        qd1_d, qd2_d = qd_des[0], qd_des[1]
        c1_d, s1_d = math.cos(q1_d), math.sin(q1_d)
        c12_d, s12_d = math.cos(q1_d + q2_d), math.sin(q1_d + q2_d)
        w1_d = qd1_d
        w12_d = qd1_d + qd2_d
        J_dot_qd_des = np.array([
            -l1 * c1_d * w1_d**2 - l2 * c12_d * w12_d**2,
            -l1 * s1_d * w1_d**2 - l2 * s12_d * w12_d**2,
        ])
        qdd_des = np.linalg.solve(J_d, ref_xy_ddot - J_dot_qd_des)
        return ref_xy, ref_xy_dot, q_des, qd_des, qdd_des

    def __call__(self, obs: dict) -> np.ndarray:
        step = obs["step"]
        if step == 0 or step <= self.step:
            self.reset()
        self.step = step
        dt = float(obs["sample_dt"])

        q_meas = np.asarray(obs["q"], dtype=float)
        qd_meas = np.asarray(obs["q_dot"], dtype=float)
        target_xy = np.asarray(obs["target_xy"], dtype=float)
        segment = int(obs["segment"])
        in_move = bool(obs["in_move"])
        time_in_seg = float(obs["time_in_segment"])
        move_time = float(obs["move_time"])

        link_len = obs["link_length"]
        l1, l2 = float(link_len[0]), float(link_len[1])
        g = float(obs["gravity"])
        viscous = np.asarray(obs["viscous_friction"], dtype=float)
        m_link = np.asarray(obs["link_mass"], dtype=float)
        c_link = np.asarray(obs["link_com"], dtype=float)
        i_link = np.asarray(obs["link_inertia"], dtype=float)

        # 1. State estimation (filtering)
        if step == 0:
            self.q_est = q_meas.copy()
            self.qd_est = qd_meas.copy()
            self.p0 = obs["ref_xy"].copy()
            self.last_segment = segment
        else:
            if segment != self.last_segment:
                self.p0 = self.last_target.copy()
                self.last_segment = segment
            q_pred = self.q_est + self.qd_est * dt
            self.q_est = (1.0 - self.alpha_q) * q_pred + self.alpha_q * q_meas
            self.qd_est = (1.0 - self.alpha_qd) * self.qd_est + self.alpha_qd * qd_meas

        self.last_target = target_xy.copy()
        q = self.q_est
        qd = self.qd_est

        # 2. Reference trajectory evaluation (current and lead time)
        lead = self.t_lead if in_move else 0.0
        _, _, q_now, qd_now, _ = self._eval_ref(time_in_seg, move_time, self.p0, target_xy, link_len)
        _, _, q_lead, qd_lead, qdd_lead = self._eval_ref(time_in_seg + lead, move_time, self.p0, target_xy, link_len)

        # 3. Tracking error
        e = q_now - q
        ed = qd_now - qd

        c2_ang = math.cos(q[1])
        s2_ang = math.sin(q[1])
        c1_ang = math.cos(q[0])
        c12_ang = math.cos(q[0] + q[1])

        # 4. Payload regressor vector: Y_pay(q, qd, qdd)
        M_pay = np.array([
            [l1**2 + l2**2 + 2.0 * l1 * l2 * c2_ang, l2**2 + l1 * l2 * c2_ang],
            [l2**2 + l1 * l2 * c2_ang, l2**2],
        ])
        C_pay = l1 * l2 * s2_ang * np.array([-(2.0 * qd[0] * qd[1] + qd[1]**2), qd[0]**2])
        G_pay = g * np.array([l1 * c1_ang + l2 * c12_ang, l2 * c12_ang])
        Y_pay = M_pay @ qdd_lead + C_pay + G_pay

        # 5. Payload adaptation
        if obs["payload_known"]:
            self.payload_hat = float(obs["payload_mass"])
        else:
            M22 = i_link[1] + m_link[1] * c_link[1]**2 + self.payload_hat * l2**2
            M12 = M22 + m_link[1] * l1 * c_link[1] * c2_ang + self.payload_hat * l1 * l2 * c2_ang
            M11 = (
                i_link[0]
                + m_link[0] * c_link[0]**2
                + i_link[1]
                + m_link[1] * (l1**2 + c_link[1]**2 + 2.0 * l1 * c_link[1] * c2_ang)
                + self.payload_hat * (l1**2 + l2**2 + 2.0 * l1 * l2 * c2_ang)
            )
            M_curr = np.array([[M11, M12], [M12, M22]])

            norm_Y = np.dot(Y_pay, Y_pay)
            if norm_Y > 1e-2:
                delta_m = float(np.dot(Y_pay, M_curr @ (self.kp * e + self.kd * ed))) / norm_Y
                self.payload_hat = np.clip(self.payload_hat + self.gamma_payload * delta_m * dt, 0.0, 1.0)

        m_p = self.payload_hat

        # 6. Dynamics evaluation along lead trajectory
        a1 = (
            i_link[0]
            + m_link[0] * c_link[0]**2
            + i_link[1]
            + m_link[1] * (l1**2 + c_link[1]**2)
            + m_p * (l1**2 + l2**2)
        )
        a2 = i_link[1] + m_link[1] * c_link[1]**2 + m_p * l2**2
        a3 = m_link[1] * l1 * c_link[1] + m_p * l1 * l2
        g1 = (m_link[0] * c_link[0] + m_link[1] * l1 + m_p * l1) * g
        g2 = (m_link[1] * c_link[1] + m_p * l2) * g

        c2_ang_lead = math.cos(q_lead[1])
        s2_ang_lead = math.sin(q_lead[1])
        M_lead = np.array([
            [a1 + 2.0 * a3 * c2_ang_lead, a2 + a3 * c2_ang_lead],
            [a2 + a3 * c2_ang_lead, a2],
        ])
        C_lead = a3 * s2_ang_lead * np.array([-(2.0 * qd_lead[0] * qd_lead[1] + qd_lead[1]**2), qd_lead[0]**2])
        c1_ang_lead = math.cos(q_lead[0])
        c12_ang_lead = math.cos(q_lead[0] + q_lead[1])
        G_lead = np.array([g1 * c1_ang_lead + g2 * c12_ang_lead, g2 * c12_ang_lead])
        F_lead = viscous * qd_lead

        # 7. Control law: computed torque + feedforward
        u_cmd = qdd_lead + self.kp * e + self.kd * ed
        tau = M_lead @ u_cmd + C_lead + G_lead + F_lead

        lim = np.asarray(obs["torque_limit"], dtype=float)
        return np.clip(tau, -lim, lim)


_controller_instance = ArmController()


def controller(obs: dict) -> np.ndarray:
    """Main entrypoint for the two-link arm controller."""
    return _controller_instance(obs)
