---
layout: post
title: "From LunarLander to String-Stable Platooning"
date: 2026-06-11 22:20:00
description: "A stage-one decentralized platooning result: an LLM-guided handwritten controller suppresses string amplification under delay, noise, and heterogeneous actuation."
tags: llm-agents control heuristic-learning reinforcement-learning robustness platooning multi-agent
categories: llm-agents
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

In the last two posts, I used LunarLander as a small testbed for an LLM-guided heuristic-learning control harness. That was useful because the feedback loop was fast: the agent could read metrics, edit a handwritten controller, tune exposed parameters, and carry a compact whiteboard between attempts.

But LunarLander also has a weakness as a research benchmark. A good language model may already know many landing-controller patterns. So the next question is sharper:

> Can the same harness synthesize a decentralized multi-agent controller where the target is not just reward, but string stability under delay and uncertainty?

This post records the first stage-one result on a vehicle platooning environment. It is not yet a final benchmark, but it is a useful step: the harness produced a local, non-neural controller that keeps a 20-vehicle platoon safe and suppresses downstream error amplification in a hard delay/noise scenario.

## Why platooning?

Vehicle platooning is a good next target because the important property is not only tracking error. It is whether disturbances amplify as they travel down the string of vehicles.

In a simple reward-only setup, a controller can look good while quietly passing too much disturbance to downstream vehicles. For platooning, this is exactly the wrong failure mode. A follower near the tail should not experience a larger spacing transient than the vehicle in front of it.

That makes string stability a natural target for this harness. It also moves the experiment closer to networked control: local observations, delayed information, heterogeneous vehicles, and robustness to leader disturbances.

## Environment

The current target scenario is `hard_delay_noise_v1`.

The simulator is lightweight: a scripted leader and 19 followers, where the same decentralized controller is applied to each follower. Each follower only sees local predecessor information.

The stressors are:

| Ingredient | Setting |
|---|---:|
| Vehicles | 20 |
| Time step | 0.1 |
| Horizon | 600 |
| Leader profile | repeated hard braking |
| Observation delay | 3 steps |
| Delay jitter | up to 3 steps |
| Predecessor acceleration dropout | 0.35 |
| Observation noise | enabled |
| Gap / relative velocity bias | enabled |

The key metric is `string_peak_gain`. It is computed from peak spacing errors along the platoon, using the maximum of:

- neighbor peak amplification: downstream peak error divided by upstream peak error
- tail peak amplification: tail peak error divided by the first follower peak error

For this round, I added a dedicated `string_performance` success mode. The main gate was:

```text
string_peak_gain_max <= 1.35
collision_rate = 0
spacing_error_rms_mean <= 3.0
success_rate high enough on train and holdout
```

This matters because earlier performance-style runs could pass too easily by finding a safe-enough controller with a loose negative reward target. The new mode makes the platooning claim explicit: suppress amplification.

## Accepted controller

The current best controller is `candidate_0010`, resumed from the `hard_delay_noise_v1` platooning run and accepted under `string_performance`.

On the target scenario it passed both train and holdout gates:

| Split | Seeds | Success | Collision | Spacing RMS | Mean string peak gain | Max string peak gain | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 0, 1, 2 | 1.000 | 0.000 | 1.348 | 0.978 | 1.111 | -1131.14 |
| Holdout | 799010, 111609, 813424, 667812, 893411 | 1.000 | 0.000 | 1.400 | 1.040 | 1.170 | -1223.14 |

Here is a rollout from the accepted candidate:

<img src="{{ '/assets/img/heuristic-learning-control/platooning_candidate0010_hard_delay_noise.gif' | relative_url }}" alt="A 20-vehicle platooning rollout from candidate 0010 under hard delay and noisy local observations." width="90%">

## What did the controller learn?

The controller is still a small handwritten Python function. It is not a neural network. Its structure is close to a robust CACC-like law:

- proportional spacing feedback
- relative-velocity damping
- small predecessor acceleration feedforward
- strong own-acceleration damping
- dropout mitigation by remembering the last nonzero predecessor acceleration
- command smoothing
- jerk limiting
- emergency braking based on gap and time-to-collision
- capped catch-up acceleration for large spacing errors

The tuned parameters were:

```python
CANDIDATE_0010_PARAMS = {
    "catchup_accel_fraction": 0.7615343988101403,
    "catchup_error_limit": 21.999018501858096,
    "command_smoothing": 0.3439886173049022,
    "emergency_brake_fraction": 0.9257460030328349,
    "gap_margin_emergency": 7.047212781184004,
    "jerk_limit": 5.550824552116185,
    "ka_ff": 0.29310224838384874,
    "ka_self": 1.46641879653996,
    "kp": 0.6079893788853924,
    "kv": 1.6797829618437539,
    "ttc_margin_emergency": 4.975383082915987,
}
```

The interesting part is the design pressure. The accepted controller is not just trying to minimize reward penalties. It is shaped by a string-gain gate, so it becomes more conservative about amplification: stronger damping, smoother commands, safety overrides, and reduced reliance on a noisy feedforward channel.

## Frozen robustness comparison

After acceptance, I froze `candidate_0010` and compared it against:

- `zero`
- `acc_pd`
- `cacc`
- `damped_cacc`

The full suite used 10 fixed seeds and 8 scenarios:

- `nominal_v0`
- `lead_brake_v0`
- `lead_sine_v0`
- `stop_go_v0`
- `delay_noise_v0`
- `heterogeneous_v0`
- `weak_brake_mid_v0`
- `delayed_heterogeneous_brake_v0`

The aggregate result:

| Controller | Avg success | Worst success | Avg collision | Avg spacing RMS | Avg string gain | Avg reward |
|---|---:|---:|---:|---:|---:|---:|
| `candidate_0010` | 0.963 | 0.700 | 0.000 | 0.554 | 0.748 | -247.06 |
| `cacc` | 0.787 | 0.000 | 0.000 | 0.344 | 1.094 | -100.31 |
| `acc_pd` | 0.875 | 0.000 | 0.000 | 0.553 | 1.188 | -231.83 |
| `damped_cacc` | 0.750 | 0.000 | 0.025 | 2.919 | 4.363 | -19454.17 |
| `zero` | 0.000 | 0.000 | 0.875 | 6.082 | 6.456 | -45741.43 |

<img src="{{ '/assets/img/heuristic-learning-control/platooning_candidate0010_robustness_comparison.png' | relative_url }}" alt="Frozen platooning robustness comparison for candidate 0010 against zero, ACC, CACC, and damped CACC baselines." width="100%">

The important reading is not "candidate has the best reward." It does not. `cacc` still has better average reward. The important reading is:

- `candidate_0010` has the best average string gain.
- `candidate_0010` has zero collisions across this full comparison.
- `candidate_0010` survives delay/noise where `cacc` and `damped_cacc` degrade.
- The remaining weakness is the combined delayed heterogeneous case, where success is 0.700.

The scenario-level candidate result:

| Scenario | Success | Collision | Spacing RMS | String gain | Reward |
|---|---:|---:|---:|---:|---:|
| `nominal_v0` | 1.000 | 0.000 | 0.064 | 0.665 | -2.49 |
| `lead_brake_v0` | 1.000 | 0.000 | 0.494 | 0.667 | -154.04 |
| `lead_sine_v0` | 1.000 | 0.000 | 0.896 | 0.654 | -483.12 |
| `stop_go_v0` | 1.000 | 0.000 | 1.127 | 0.654 | -764.51 |
| `delay_noise_v0` | 1.000 | 0.000 | 0.375 | 0.857 | -96.56 |
| `heterogeneous_v0` | 1.000 | 0.000 | 0.547 | 0.695 | -190.64 |
| `weak_brake_mid_v0` | 1.000 | 0.000 | 0.543 | 0.668 | -184.78 |
| `delayed_heterogeneous_brake_v0` | 0.700 | 0.000 | 0.386 | 1.126 | -100.34 |

## Error diagnostics

The spacing-error heatmap gives a more direct look at the transient behavior over vehicle index and time:

<img src="{{ '/assets/img/heuristic-learning-control/platooning_candidate0010_spacing_error_heatmap.png' | relative_url }}" alt="Spacing error heatmap for candidate 0010 platooning rollout." width="100%">

The peak-error propagation plot is closer to the string-stability question:

<img src="{{ '/assets/img/heuristic-learning-control/platooning_candidate0010_peak_error_propagation.png' | relative_url }}" alt="Peak spacing error propagation for candidate 0010 platooning rollout." width="90%">

These plots are useful because the raw reward can hide the mechanism. A controller can have a worse reward because it keeps larger safety margins, but still be better for the networked-control objective if it prevents amplification and collisions.

## Interpretation

This result is the first stage where the harness feels like it is aiming at a real networked-control property rather than just solving another benchmark.

The claim I would make is modest:

1. The harness can synthesize an interpretable decentralized controller that is competitive with simple hand-designed baselines.
2. Adding `string_performance` changes the search pressure in the right direction.
3. The discovered controller is not magic; it looks like a plausible robust local CACC variant.
4. The current scenario is useful, but not yet hard enough to be a final paper benchmark.

That last point matters. This is not a proof of formal string stability. It is also not a full industrial platooning stack. It is a controlled experiment showing that an LLM-guided code-search loop can move toward a network-level control objective when the objective is made explicit.

## What this does not yet prove

This stage does not yet prove:

- formal string stability
- scalability to 50 or 100 vehicles
- robustness to burst communication loss
- robustness to cut-in or cut-out events
- superiority over a tuned MPC or formally designed robust CACC controller
- that the whiteboard mechanism is causally necessary

The last point is especially important. The right ablation is not another nice GIF. It is a code-search curve: full whiteboard versus no whiteboard, no reflection, no previous-candidate summaries, and random/no-whiteboard baselines.

## Next

The next environment ladder should make the task harder in dimensions that matter for networked control:

1. Scale from 20 vehicles to 50 vehicles.
2. Use bursty packet dropout, not only independent acceleration dropout.
3. Add stochastic communication delay with a longer tail, for example 0 to 8 steps.
4. Add heterogeneous mass, actuator lag, acceleration limit, and braking limit together.
5. Add one or two weak-braking vehicles at unknown positions.
6. Add sensor bias that drifts slowly during the rollout.
7. Add cut-in, cut-out, or platoon split/merge events.
8. Report a Pareto frontier: nominal reward versus worst-case string gain.

For the next acceptance target, I would keep:

```text
collision_rate = 0
success_rate >= 0.9 on full suite
string_peak_gain_max <= 1.35 on holdout
```

but add a full-suite robustness metric so a controller cannot overfit only the training scenario.

## Reproduction

The accepted run was:

```text
harness_runs/platooning/hard_delay_noise_v1/zero/run_20260611-202910-614914_round001_seeds0-1-2
candidate_0010
```

The frozen full comparison command was:

```bash
PYTHONPATH=src .venv311/bin/python -m hl_control.eval.run_platooning_robustness \
  --heuristic-workspace harness_runs/platooning/hard_delay_noise_v1/zero/run_20260611-202910-614914_round001_seeds0-1-2/candidates/candidate_0010/candidate_workspace \
  --heuristic-controller candidate_0010 \
  --include-baseline zero acc_pd cacc damped_cacc \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --suite standard lead delay heterogeneity combo \
  --output-dir experiments/platooning_robustness
```

The output folder was:

```text
experiments/platooning_robustness/20260611-215844-179716
```

