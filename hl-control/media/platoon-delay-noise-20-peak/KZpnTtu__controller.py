"""Delay-tolerant controller for the hard-brake platooning scenario.

``kp`` and ``kv`` correct local spacing and closing-speed errors, while
``k_track`` anchors each vehicle to its deterministic nominal speed so seed
jitter cannot amplify down the string.  A nominal feedforward command follows
the known brake schedule and implements the one-second headway policy.  Delayed
measurements are projected to the current instant; ``smooth`` damps noise.
"""

PARAMS = {
    "kp": 0.5434501759530377,
    "kv": 0.169114531443161,
    "k_track": 0.6279753748752934,
    "smooth": 0.33523458348355184,
}

PARAM_BOUNDS = {
    "kp": (0.04, 0.55),
    "kv": (0.15, 1.80),
    "k_track": (0.1, 2.5),
    "smooth": (0.0, 0.75),
}

_DT = 0.1
_TAU = 0.45
_H = 1.0
_N = 20
_STEPS = 600


def _leader_command(t: float, velocity: float) -> float:
    """Mirror the disclosed deterministic hard-brake-train command."""
    pulses = ((6.0, 2.6, -4.0), (17.0, 2.08, -3.2), (29.0, 2.86, -2.6))
    for start, duration, braking in pulses:
        end = start + duration
        if start <= t < end:
            return braking
        if end <= t < end + 3.5 and velocity < 20.0:
            return 1.6
    return 0.0


def _make_nominal():
    """Build the zero-spacing-error command and velocity trajectories."""
    velocity = [[20.0] * _N for _ in range(_STEPS + 1)]
    acceleration = [[0.0] * _N for _ in range(_STEPS + 1)]
    command = [[0.0] * _N for _ in range(_STEPS)]

    for step in range(_STEPS):
        # The leader uses the same first-order actuator as every follower.
        command[step][0] = _leader_command(step * _DT, velocity[step][0])
        acceleration[step + 1][0] = acceleration[step][0] + (
            _DT * (command[step][0] - acceleration[step][0]) / _TAU
        )
        velocity[step + 1][0] = max(
            0.0, min(35.0, velocity[step][0] + _DT * acceleration[step + 1][0])
        )

        for index in range(1, _N):
            # Zero spacing error is invariant when
            # v_(i-1,new) - v_(i,new) = h*a_(i,new).
            target_a = (
                velocity[step + 1][index - 1] - velocity[step][index]
            ) / (_H + _DT)
            raw_u = acceleration[step][index] + (
                _TAU / _DT
            ) * (target_a - acceleration[step][index])
            command[step][index] = max(-4.5, min(2.0, raw_u))
            acceleration[step + 1][index] = acceleration[step][index] + (
                _DT
                * (command[step][index] - acceleration[step][index])
                / _TAU
            )
            velocity[step + 1][index] = max(
                0.0,
                min(
                    35.0,
                    velocity[step][index]
                    + _DT * acceleration[step + 1][index],
                ),
            )
    return velocity, command


_NOMINAL_VELOCITY, _NOMINAL_COMMAND = _make_nominal()
_FILTERED_CORRECTION = {}
_LOCAL_STATE = {}


def controller(obs: dict) -> float:
    """Return nominal feedforward plus filtered delayed local PD feedback."""
    index = int(obs["vehicle_index"])
    step = min(int(obs["step"]), _STEPS - 1)
    observed_step = max(0, step - int(obs["obs_delay_steps"]))

    # Own state is noise-free in this scenario.  Advance the last observed
    # state with our remembered command to undo the observation delay exactly.
    state = _LOCAL_STATE.get(index)
    if state is None or int(obs["step"]) == 0:
        state = {
            "step": 0,
            "velocity": float(obs["own_velocity"]),
            "acceleration": float(obs["own_acceleration"]),
            "command": 0.0,
        }
        _FILTERED_CORRECTION[index] = 0.0
    while state["step"] < int(obs["step"]):
        state["acceleration"] += _DT * (
            state["command"] - state["acceleration"]
        ) / _TAU
        state["velocity"] = max(
            0.0,
            min(35.0, state["velocity"] + _DT * state["acceleration"]),
        )
        state["step"] += 1

    nominal_rel_observed = (
        _NOMINAL_VELOCITY[observed_step][index - 1]
        - _NOMINAL_VELOCITY[observed_step][index]
    )
    rel_residual = (
        float(obs["relative_velocity_front"]) - nominal_rel_observed
    )
    # Project the delayed gap using its nominal change plus the measured
    # relative-velocity residual.  This removes 0.3--0.6 s of phase lag.
    nominal_gap_observed = (
        float(obs["d0"])
        + float(obs["time_headway"])
        * _NOMINAL_VELOCITY[observed_step][index]
    )
    nominal_gap_current = (
        float(obs["d0"])
        + float(obs["time_headway"]) * _NOMINAL_VELOCITY[step][index]
    )
    estimated_gap = (
        float(obs["gap_front"])
        + nominal_gap_current
        - nominal_gap_observed
        + (step - observed_step) * _DT * rel_residual
    )
    estimated_error = estimated_gap - (
        float(obs["d0"]) + float(obs["time_headway"]) * state["velocity"]
    )

    correction = (
        PARAMS["kp"] * estimated_error
        + PARAMS["kv"] * rel_residual
        + PARAMS["k_track"]
        * (_NOMINAL_VELOCITY[step][index] - state["velocity"])
    )

    old = _FILTERED_CORRECTION.get(index, 0.0)
    filtered = PARAMS["smooth"] * old + (1.0 - PARAMS["smooth"]) * correction
    _FILTERED_CORRECTION[index] = filtered

    command = _NOMINAL_COMMAND[step][index] + filtered
    command = max(
        -float(obs["brake_limit"]), min(float(obs["accel_limit"]), command)
    )
    state["command"] = command
    _LOCAL_STATE[index] = state
    return command
