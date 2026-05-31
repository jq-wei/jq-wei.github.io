---
layout: post
title: "When Reward Chasing Meets Robustness"
date: 2026-05-29 22:30:00
description: "A follow-up LunarLander lab note: an LLM-guided handwritten controller reaches PPO-like reward, then exposes the tradeoff between nominal performance and robustness."
tags: llm-agents control heuristic-learning reinforcement-learning robustness lunar-lander
categories: llm-agents
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

In the [previous post](https://jq-wei.github.io/blog/2026/heuristic-learning-control-harness/), I described a small heuristic-learning control harness: let an LLM inspect experiments, read trajectory diagnostics, edit controller code, tune exposed parameters, and keep a compact "whiteboard" between attempts.

However, the LunarLander result there was incomplete. The harness had found a small handwritten controller that could land reliably, but it was still behind a pretrained SB3 PPO policy on Gym reward. A follow-up question would be:

> Can the same harness push a non-neural LunarLander controller toward PPO-level reward, and if it can, how about robustness of the controller?

Short answer: yes, it can chase nominal reward surprisingly well. But when I moved robustness into the loop, the story became more interesting: the harness recovered much of the lost robustness, then exposed a very specific remaining failure mode.

## Starting point

The parent controller was `candidate_0009` from the [previous post](https://jq-wei.github.io/blog/2026/heuristic-learning-control-harness/) . It was already a real controller, not a toy. On the original run it reached terminal landing success with zero crashes:

| Candidate | Success mode | Terminal success | Crash rate | Reward mean | Steps mean |
|---|---:|---:|---:|---:|---:|
| `candidate_0009` | `terminal` | 1.00 | 0.00 | 245.41 | 371.0 |
| `candidate_0009` holdout | `terminal` | 1.00 | 0.00 | 234.87 | 362.6 |

That was a good control result, but not yet PPO-level behavior. The PPO reference I used, from the [Stable-Baselines3 LunarLanderContinuous checkpoint](https://huggingface.co/sb3/ppo-LunarLanderContinuous-v2), scored about `271.30` mean reward on the fixed standard-v0 seed set.

Visually, `candidate_0009` landed, but it was conservative and slow:

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0009_terminal.gif' | relative_url }}" alt="LunarLander candidate 0009 landing successfully but below PPO reward." width="75%">

The PPO reference was cleaner and faster:

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_sb3_ppo_reference.gif' | relative_url }}" alt="Stable-Baselines3 PPO reference policy on LunarLanderContinuous." width="75%">

## Adding a performance mode

The harness previously had two success modes:

| Mode | Meaning |
|---|---|
| `achievement` | Enter the success region at some point. |
| `terminal` | Finish in the success region with no crash. |

For this stage, I added a third mode:

| Mode | Meaning |
|---|---|
| `performance` | Keep terminal success and safety, then require a reference score target. |

For the LunarLander refinement, the frontend configuration was:

```text
Env: lunar_lander
Scenario: standard_v0
Resume candidate: candidate_0009, then later candidate_0007
Success mode: performance
Reference score metric: total_reward_mean
Reference score target: 271.3
Reference score tolerance: 5.0
Reference label: sb3_ppo_lunar_lander_continuous_v2
Tuning trials: 120
Tuning workers: 8
```

The important part is that PPO was not used as a teacher policy. The harness only saw a scalar reference target: "get close to this reward while preserving terminal success and no crashes." The controller remained a small Python function.

The whiteboard guide for the next attempt was roughly:

```text
Preserve:
- terminal_success_rate = 1.0
- crash_rate = 0.0
- clean final landing geometry

Improve:
- total_reward_mean toward the PPO reference
- reduce hover time
- reduce fuel use and side-engine chatter
- reduce final lateral drift and touchdown speed

Avoid:
- redesigning the controller family from scratch
- accepting higher reward at the cost of crashes
```

This is the useful part of the harness design: the next attempt is not just "maximize reward." It is "hold the behavioral contract, then improve reward."

## The controller that passed

The accepted controller was `candidate_0018`.

On the synthesis seeds it passed the performance gate:

| Metric | `candidate_0018` |
|---|---:|
| Terminal success | 1.00 |
| Achievement success | 1.00 |
| Landing rate | 1.00 |
| Crash rate | 0.00 |
| Reward mean | 277.80 |
| Episode steps mean | 284.33 |
| Fuel proxy total mean | 240.39 |
| Mean absolute side engine | 0.400 |
| Final landing error mean | 0.061 |

On the holdout seeds it still passed:

| Metric | `candidate_0018` holdout |
|---|---:|
| Terminal success | 1.00 |
| Achievement success | 1.00 |
| Landing rate | 1.00 |
| Crash rate | 0.00 |
| Reward mean | 267.87 |
| Episode steps mean | 308.40 |
| Fuel proxy total mean | 210.75 |
| Mean absolute side engine | 0.271 |
| Final landing error mean | 0.077 |

Because we have the gate as `271.3 - 5.0`, this controller got accepted. The controller cleared it, barely but legitimately.

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0018_performance.gif' | relative_url }}" alt="LunarLander candidate 0018, the accepted performance-mode handwritten controller." width="75%">

The behavior looks a little shaking. The controller uses high-gain lateral and attitude correction, plus a relatively hard side-engine deadzone:

```python
CANDIDATE_0018_PARAMS = {
    "k_ang_vel": 24.0,
    "k_angle": 22.0,
    "k_vy": 45.0,
    "k_x": 2.0,
    "k_x_dot": 50.0,
    "main_power_scale": 0.4,
    "side_deadzone": 0.7,
    "side_min_mag": 0.75,
    "target_vy_descend": -2.5,
    "target_vy_soft": -0.08,
}
```

In other words, it is:

1. Descend quickly when high above the pad.
2. Switch to a soft vertical target near the ground.
3. Use a PD-like main engine throttle for vertical speed.
4. Use lateral position, lateral velocity, angle, and angular velocity to fire side thrusters.
5. Reduce side thrust after one leg touches, and stop after stable two-leg contact.

That is an inspectable controller. It is also a clue about the later failure: the policy is thresholded, high-gain, and tuned to a narrow landing envelope.

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0018_metrics.png' | relative_url }}" alt="LunarLander candidate 0018 trajectory and action diagnostics." width="85%">

## The robustness test

After that, I froze both controllers:

- `candidate_0018` fixed, no more LLM edits.
- SB3 PPO fixed, no fine-tuning.
- zero controller included as a sanity baseline.

Then I evaluated all controllers on the same ten seeds, across a small perturbation suite:

```text
standard_v0
gravity_low_10
gravity_high_10
gravity_low_20
gravity_high_20
main_weak_10
main_weak_20
main_strong_10
main_strong_20
side_weak_20
side_strong_20
heavy_weak_main_v0
light_strong_main_v0
weak_attitude_control_v0
```

These perturbations are simple: change gravity, main engine power, side engine power, or combine heavier effective dynamics with a weaker main engine. The controller does not observe these perturbation values. It only sees the standard LunarLander observation vector.

The result was not close:

The robustness score below is:

```text
mean terminal success - 0.25 * crash rate - 0.10 * nominal regression penalty
```

| Controller | Robustness score | Avg terminal success | Worst terminal success | Avg reward |
|---|---:|---:|---:|---:|
| SB3 PPO | 0.964 | 0.964 | 0.800 | 265.97 |
| `candidate_0018` | 0.656 | 0.714 | 0.100 | 149.55 |
| zero | -0.340 | 0.000 | 0.000 | -138.75 |

Some representative rows:

| Scenario | `candidate_0018` terminal | `candidate_0018` reward | PPO terminal | PPO reward |
|---|---:|---:|---:|---:|
| `standard_v0` | 0.80 | 146.25 | 1.00 | 271.30 |
| `gravity_high_20` | 0.60 | 144.09 | 0.80 | 223.66 |
| `main_weak_20` | 0.90 | 218.04 | 1.00 | 268.03 |
| `side_strong_20` | 0.70 | 155.94 | 1.00 | 272.93 |
| `heavy_weak_main_v0` | 0.10 | 67.77 | 0.80 | 221.47 |
| `weak_attitude_control_v0` | 0.80 | 159.59 | 1.00 | 271.60 |

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_robustness_candidate0018_vs_ppo_heatmap.png' | relative_url }}" alt="Robustness heatmap comparing zero, candidate 0018, and SB3 PPO terminal success across LunarLander perturbations." width="90%">

There is an uncomfortable detail here: on the larger frozen seed set, `candidate_0018` even regressed on `standard_v0`, with terminal success `0.8` and mean reward `146.25`. That does not contradict the performance gate above. But rather says the gate was too small. The synthesis holdout was enough to catch many bad controllers, but not enough to certify broad seed robustness.

This is the kind of thing a harness should make visible.

## What changed

The performance pass was real. The robustness failure was also real.

My reading after `candidate_0018` was:

1. `candidate_0009` from previous blog was safer and slower.
2. `candidate_0018` learned to spend reward budget more aggressively.
3. The reward improvement came from faster descent and stronger attitude/lateral correction.
4. That made the controller narrower.

The wobbly GIF is the visual symptom of the same thing. The controller is actively trading angle, lateral velocity, and main-engine usage around a tuned descent schedule. It can look efficient on the sampled seeds and still be fragile when the effective physics change.

For a high-gain switching logic often gives crisp nominal behavior before it gives robust behavior.

## Putting robustness into the loop

So I added another success mode:

| Mode | Meaning |
|---|---|
| `robust_performance` | Keep nominal performance near a reference score, then require a small robustness probe over hidden physics perturbations. |

This was intentionally not a full robust-control certificate. It was a cheap smoke test inside the harness loop:

```text
Nominal:
- terminal_success_rate must stay high
- crash_rate must stay low
- total_reward_mean must stay within 5 points of the PPO reference

Robust smoke suite:
- standard_v0
- gravity_high_20
- main_weak_20
- heavy_weak_main_v0

Robust gates:
- average terminal success >= 0.8
- worst-scenario terminal success >= 0.5
- average crash rate <= 0.2
```

I also made the whiteboard more explicit. Instead of saying "be more robust", it now summarized the parent candidates by role:

```text
Best nominal parent:
- high reward on standard LunarLander

Best robust parent:
- keeps success across the smoke perturbation suite

Failure mode to fix:
- heavy_weak_main_v0 collapse

Do not:
- hard-code scenario names
- branch on unobserved perturbation values
- trade away the nominal reward just to hover forever
```

The best result in this branch was not the highest-reward controller. It was `candidate_0008`, which was more conservative than `candidate_0018` but kept the robustness gates alive.

On the main run:

| Metric | `candidate_0008` |
|---|---:|
| Terminal success | 1.00 |
| Achievement success | 1.00 |
| Crash rate | 0.00 |
| Reward mean | 262.43 |
| Episode steps mean | 320.00 |
| Robust smoke avg terminal | 0.90 |
| Robust smoke worst terminal | 0.80 |
| Robust smoke crash rate | 0.00 |
| Robust smoke reward mean | 252.16 |

This did not beat PPO's nominal reward. But the gap was small enough to matter differently: it was now a question of tradeoffs, not failure. The controller landed reliably, remained interpretable, and behaved well under the smoke perturbations.

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0008_robust_performance.gif' | relative_url }}" alt="LunarLander candidate 0008, a conservative robust-performance handwritten controller." width="75%">

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0008_metrics.png' | relative_url }}" alt="LunarLander candidate 0008 trajectory and action diagnostics." width="85%">

## Frozen comparison, again

Then I repeated the frozen full robustness comparison, now with `candidate_0008`:

```text
candidate_0008 fixed, no more LLM edits
SB3 PPO fixed, no fine-tuning
zero controller as sanity baseline
10 seeds: 0 1 2 3 4 5 6 7 8 9
suite: standard + gravity + engine + combo
```

The headline is more nuanced than the `candidate_0018` failure. PPO still wins on average, but not by a lot.

| Metric | `candidate_0008` | SB3 PPO |
|---|---:|---:|
| `standard_v0` reward | 259.53 | 271.30 |
| Full-suite avg terminal | 0.907 | 0.964 |
| Full-suite worst terminal | 0.800 | 0.800 |
| Full-suite avg reward | 259.11 | 265.97 |
| Reward gap | -6.86 | 0.00 |
| Crash rate | 0.00 | 0.00 |

Some representative rows:

| Scenario | `candidate_0008` terminal | `candidate_0008` reward | PPO terminal | PPO reward |
|---|---:|---:|---:|---:|
| `standard_v0` | 0.90 | 259.53 | 1.00 | 271.30 |
| `gravity_high_20` | 1.00 | 253.97 | 0.80 | 223.66 |
| `main_weak_20` | 0.80 | 249.76 | 1.00 | 268.03 |
| `main_strong_20` | 0.90 | 265.35 | 0.90 | 275.07 |
| `side_weak_20` | 0.80 | 258.63 | 1.00 | 271.60 |
| `heavy_weak_main_v0` | 0.90 | 245.38 | 0.80 | 221.47 |
| `light_strong_main_v0` | 1.00 | 272.79 | 1.00 | 282.17 |
| `weak_attitude_control_v0` | 0.80 | 258.63 | 1.00 | 271.60 |

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_robustness_candidate0008_vs_ppo_heatmap.png' | relative_url }}" alt="Robustness heatmap comparing zero, candidate 0008, and SB3 PPO terminal success across LunarLander perturbations." width="90%">

The full-suite mean reward gap is about `6.86` reward points, or roughly `2.6%` of PPO's mean. That is smaller than I expected for a compact handwritten controller. PPO is still more efficient on most rows, especially the nominal and actuator-weak cases, but `candidate_0008` has a different robustness profile.

The most interesting row is the stress case:

```text
candidate_0008 / heavy_weak_main_v0: terminal=0.90, reward=245.38
SB3 PPO        / heavy_weak_main_v0: terminal=0.80, reward=221.47
```

That is an important scenario: heavier effective dynamics plus weaker main-engine authority. Under that perturbation, the small controller is better than the PPO checkpoint on both terminal success and reward.

The price is also clear: `candidate_0008` burns more fuel and lands more conservatively. In the matrix, its fuel-per-step is usually higher than PPO. It is less elegant, less efficient, and more cautious. But it is not a failed controller. It is a small inspectable controller with a slightly lower average reward and one very interesting robust-control-shaped win.

## Why this matters

This experiment clarified a distinction I had been blurring:

| Objective | What it asks for |
|---|---|
| Terminal success | Can it land safely at the end? |
| Gym reward | Can it land efficiently according to the environment's reward shaping? |
| Smoke robustness | Does it survive a small uncertainty set during synthesis? |
| Full robustness | Does it survive a broader seed and physics perturbation matrix? |

Those are not the same target.

The progression is the useful part:

| Stage | Parent | What happened |
|---|---|---|
| Terminal controller | `candidate_0009` | Landed reliably but lagged PPO reward. |
| Performance controller | `candidate_0018` | Reached PPO-level sampled reward, then failed broad robustness. |
| Robust-performance controller | `candidate_0008` | Accepted a small reward gap in exchange for clean smoke robustness and strong `heavy_weak_main_v0`. |

That last row is the actual research direction. The harness is no longer just asking the LLM to write a clever controller. It is asking the LLM to move through a sequence of specifications:

```text
land -> land efficiently -> land efficiently under uncertainty
```

This starts to look like a lightweight, program-search version of robust control. It is not H-infinity synthesis, tube MPC, or a formal min-max controller. But the shape is familiar: define an uncertainty set, preserve nominal performance, and improve worst-case behavior without using the hidden disturbance as an observation.


