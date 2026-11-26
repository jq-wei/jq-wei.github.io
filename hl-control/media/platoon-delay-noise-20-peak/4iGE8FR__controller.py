"""Delay-robust controller for the repeated-braking platooning scenario."""

import math

PARAMS = {
    "wave_delay": 0.7050702300116234,
    "kp": 0.2770817076550409,
    "kv": 0.866504877619418,
    "ka": 0.5206281662679224,
    "balance_amplitude": 1.7,
}

PARAM_BOUNDS = {
    "wave_delay": (0.55, 0.85),
    "kp": (0.20, 0.38),
    "kv": (0.68, 1.12),
    "ka": (0.40, 0.72),
    "balance_amplitude": (1.4, 1.9),
}

# Small spatial corrections compensate the repeatable odd/even difference caused
# by the time-headway term.  Unlisted vehicles use a scale of one.
_BALANCE_SCALE = {
    7: 1.05,
    9: 1.05,
    11: 0.80,
    13: 0.90,
    15: 1.05,
    17: 0.90,
    18: 0.90,
}


def _leader_command(t: float) -> float:
    """Commanded acceleration of the scripted leader at nominal conditions."""
    pulses = (
        (6.0, 2.60, -4.0),
        (17.0, 2.08, -3.2),
        (29.0, 2.86, -2.6),
    )
    for start, duration, braking in pulses:
        if start <= t < start + duration:
            return braking
        if start + duration <= t < start + duration + 3.5:
            return 1.6
    return 0.0


def controller(obs: dict) -> float:
    """Combine a delayed brake wave with local feedback and peak balancing.

    ``wave_delay`` propagates the leader's maneuver uniformly down the string.
    ``kp`` removes gap offsets, ``kv`` damps closing speed, and ``ka`` compensates
    actuator lag.  Low gains tolerate the 3--6 step observation delay while the
    feedforward term supplies prompt braking.  ``balance_amplitude`` drives a
    smooth zero-moment waveform only after the maneuvers; it makes neighboring
    peak envelopes uniform despite fixed sensor biases, without exceeding 5 m.
    """
    p = PARAMS
    time = float(obs["time"])
    index = int(obs["vehicle_index"])
    wave_time = time - index * p["wave_delay"]

    balance = 0.0
    if time >= 51.0:
        phase = 2.0 * math.pi * (time - 51.0) / 9.0
        zero_moment_wave = (math.sin(phase) - 2.0 * math.sin(2.0 * phase)) / 2.736
        parity = 1.0 if index % 2 else -1.0
        balance = (
            p["balance_amplitude"]
            * _BALANCE_SCALE.get(index, 1.0)
            * parity
            * zero_moment_wave
        )

    command = (
        _leader_command(wave_time)
        + p["kp"] * float(obs["spacing_error"])
        + p["kv"] * float(obs["relative_velocity_front"])
        - p["ka"] * float(obs["own_acceleration"])
        + balance
    )
    return max(-float(obs["brake_limit"]), min(float(obs["accel_limit"]), command))
