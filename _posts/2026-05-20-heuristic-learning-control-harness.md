---
layout: post
title: "Heuristic Learning for Control: Building a Harness Beyond Gradients"
date: 2026-05-20 19:00:00
description: Notes from building a small heuristic-learning control harness where an LLM reads experiment logs, edits non-neural controllers, tunes parameters, and refines failure modes without gradient training.
tags: llm-agents control heuristic-learning reinforcement-learning agents
categories: llm-agents
giscus_comments: true
related_posts: true
toc:
  sidebar: left
---

This post is a short lab note from building a small **heuristic-learning control harness**. The core idea is close in spirit to n+e's post [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/): instead of updating neural-network weights by backpropagation, we let a language model inspect experiments, read rewards and logs, and directly rewrite a candidate solution. While the original experiments in that post were mostly RL-style tasks, control problems seem like a particularly natural fit: the goal is often not just high reward, but an interpretable analytic controller.

Here the candidate solution is not a neural-network based policy. It is controller code in python: PID-like rules, energy shaping, LQR-style switching logic, barriers, geometries, and a small dictionary of tunable parameters. The harness runs the controller in an environment, records metrics and trajectories, gives the agent a curated context, asks for a new controller proposal, applies it in a sandbox, reruns evaluation, and repeats.


## Why not just RL?

For many control problems, a neural RL policy is not the most natural first object to search over. A hand-designed controller can be:

- interpretable,
- cheap to run,
- easy to verify against constraints,
- easy to transplant into a different simulator,
- and possible optimal in certain sense.

The question is whether an LLM can help discover such controllers from experiment feedback. Not by training a neural net, but by iterating over controller code.

The first version of my harness was intentionally simple:

1. choose an environment and a baseline controller, say zero
2. run baseline experiments over several seeds,
3. ask an analyst agent to diagnose failure,
4. ask an editor agent to propose a new controller function,
5. validate the proposal schema and registry patch,
6. apply it in an isolated candidate workspace,
7. run tests and evaluations,
8. tune exposed parameters,
9. compare against the baseline,
10. summarize failure modes into a "whiteboard" for the next attempt.

The important part turned out not to be the orchestration itself. It was **what context to give the model between iterations**.

## The harness loop

At a high level, the loop is:

```text
baseline eval
  -> environment profile
  -> analyst diagnosis
  -> controller editor proposal
  -> proposal validation
  -> candidate apply
  -> unit tests
  -> parameter tuning
  -> candidate rerun
  -> comparison and holdout
  -> reflection / whiteboard update
  -> next candidate
```

The editor does not directly patch the main repo. It returns a structured proposal:

```json
{
  "target_file": "src/hl_control/controllers/cart_link_baselines.py",
  "new_function_name": "candidate_0006_controller",
  "controller_name": "candidate_0006",
  "code_block": "...",
  "controller_registry_file": "src/hl_control/eval/run_cart_link.py",
  "controller_registry_patch": "...",
  "difference_vs_baseline": "...",
  "difference_vs_discovered": "...",
  "test_commands": ["..."],
  "tunable_parameters": {
    "k_energy": {"low": 50.0, "high": 500.0, "default": 120.0}
  }
}
```

The harness checks the schema, applies the code in a copied candidate workspace, updates the controller registry, runs tests, optionally tunes parameters, and then evaluates the candidate. This is a useful separation of concerns: the LLM proposes, but middleware decides whether the proposal is syntactically valid, executable, and better.

## Whiteboard context

The biggest practical lesson was that raw logs are not enough.

Early on I gave the remote model file paths like:

```markdown
- /Users/.../baseline/aggregate/summary.md
- /Users/.../baseline/aggregate/metrics.json
```

This looked reasonable in a local IDE, but it was wrong for a remote API model. DeepSeek sees text, not my filesystem. If the prompt only contains file paths, the model cannot inspect the files. So the harness now embeds the important artifacts directly into the prompt:

- environment source snippets,
- controller source snippets,
- baseline metrics,
- candidate metrics,
- selected trajectory diagnostics,
- previous candidate summaries,
- current best attempt,
- dominant failure mode,
- concrete next-attempt guidance.

I started thinking of this as a research **whiteboard**. It is not a transcript. It is not every log line. It is a compact, curated working memory.

A typical environment profile looks like this:

```json
{
  "observation_keys": ["x", "x_dot", "theta", "theta_dot", "cos_theta", "sin_theta"],
  "action_shape": "(1,) scalar horizontal force",
  "action_limits": "[-10.0, 10.0] N",
  "dynamics_summary": "Underactuated cart-link. theta=0 is upright, theta=pi is hanging down.",
  "reward_terms": [
    "cos(theta)",
    "- 0.05 * x^2",
    "- 0.01 * x_dot^2",
    "- 0.01 * theta_dot^2",
    "- 1e-4 * force^2",
    "- 2.0 * track_violation^2"
  ],
  "success_metrics": [
    "achievement_success",
    "terminal_success",
    "final_upright_error",
    "final_abs_theta_dot",
    "final_abs_x"
  ]
}
```

Then the editor prompt includes the active objective:

```markdown
Benchmark: `cart_link`
Controller: `zero`
Candidate controller name: `candidate_0006`
Candidate function name: `candidate_0006_controller`
Candidate tunable parameter dict name: `CANDIDATE_0006_PARAMS`
Active success mode: `terminal`
```

That `Active success mode` line was added after a surprisingly important failure.

## Pitfall 1: a reward is not a spec

For the cart-link task, the reward encouraged upright posture and penalized cart motion, angular velocity, force, and track violation:

$$
r = \cos(\theta)
  - c_x x^2
  - c_{\dot{x}} \dot{x}^2
  - c_{\dot{\theta}} \dot{\theta}^2
  - c_u u^2
  - c_{\mathrm{track}} \max(|x| - x_{\max}, 0)^2.
$$

This reward is useful, but it did not fully express what I cared about: the link should be upright **at the end** and stay there.

The first success metric was "achievement success": did the controller enter the success region for a consecutive window of steps at any point in the rollout? This produced a candidate that looked pretty good:

<img src="{{ '/assets/img/heuristic-learning-control/cart_link_achievement_candidate_0015.gif' | relative_url }}" alt="Cart-link candidate 0015 reaching the upright region but not used as the final terminal controller." width="80%">

However, this was subtly not the same as terminal stabilization. Some candidates could swing up, briefly satisfy the success predicate, and later drift away or violate the track. So I split the metric:

- `achievement_success_rate`: held the success region at least once.
- `terminal_success_rate`: held the success region during the final hold window.

This changed the task from "touch the goal" to "finish in the goal." For control, that distinction matters.

## Pitfall 2: history can mislead the model

At first, I wanted to give the model all previous attempts. But after a few rounds, the history becomes contradictory. One failed candidate says "increase gains"; another says "decrease gains"; a third says "switch earlier"; another says "switch later." If all of that gets dumped into the next prompt, the model has too many stale local explanations.

The reflection step now tries to summarize:

- the strongest current parent candidate,
- the dominant failure mode,
- which constraints were violated,
- which idea should be preserved,
- which direction should not be repeated.

For example, after a candidate achieved swing-up but failed terminal hold, the reflection was:

```json
{
  "failure_summary": "Candidate achieved swing-up and capture in all seeds but failed to hold the terminal success condition.",
  "likely_causes": [
    "stabiliser gains are too high and cause force saturation",
    "fixed gains do not robustly stabilize the underactuated dynamics",
    "permanent lock prevents re-swing or recovery after drift"
  ],
  "next_attempt_guidance": [
    "keep energy-shaping swing-up",
    "replace fixed-gain stabiliser with LQR around upright",
    "use [theta, theta_dot, x, x_dot] near capture",
    "avoid normal balancing forces near saturation"
  ]
}
```

This is the whiteboard doing its job: preserve the useful discovery, discard stale noise.

## Pitfall 3: tuning is part of the agent

The LLM is good at proposing a controller structure, but exact gains are often a separate search problem. So the proposal schema lets the editor expose tunable parameters:

```json
{
  "Q_theta": {"low": 20.0, "high": 200.0, "default": 100.0},
  "Q_theta_dot": {"low": 5.0, "high": 80.0, "default": 30.0},
  "R_scalar": {"low": 0.1, "high": 5.0, "default": 1.0},
  "theta_capture": {"low": 0.1, "high": 0.5, "default": 0.35}
}
```

The harness runs parameter trials in parallel candidate workspaces, scores them, applies the best parameters back into the candidate, and then reruns the official evaluation. In the successful cart-link run, I used:

```bash
--tuning-trials 90
--tuning-max-workers 6
```

This changed the feel of the loop. The LLM no longer needed to hit exact constants in one shot. It needed to propose a plausible controller family and parameter ranges. The tuner then searched within that local family.

## Experiments

### Inverted pendulum

The simpler inverted pendulum case was a warm-up. Starting from a zero controller, the harness found a controller that stabilizes the pendulum after a few iterations.

<img src="{{ '/assets/img/heuristic-learning-control/inverted_pendulum_candidate_0004.gif' | relative_url }}" alt="Inverted pendulum candidate controller found by the heuristic-learning harness." width="65%">

This task is not the interesting one by itself. Its value was in debugging the harness: proposal schema, registry patching, candidate workspaces, reruns, and dashboard visualization.

### Cart-link: baseline

The cart-link starts near the downward equilibrium. The zero controller simply lets it stay there.

<img src="{{ '/assets/img/heuristic-learning-control/cart_link_zero_baseline.gif' | relative_url }}" alt="Cart-link zero baseline staying near the downward equilibrium." width="80%">

Baseline metrics over three seeds:

```json
{
  "final_upright_error_mean": 3.1316,
  "total_reward_mean": -699.91,
  "achievement_success_rate": 0.0,
  "terminal_success_rate": 0.0,
  "max_track_violation_max": 0.0
}
```

The agent first discovered energy-shaping swing-up controllers. These could often reach the upright region but did not always stay there. That was useful, but not yet the right controller.

### Cart-link: terminal refinement

The important run resumed from an earlier near-solution, `candidate_0025`, but changed the objective to terminal success:

```bash
--success-mode terminal
--resume-run .../run_20260519-202208-021732_round001_seedsrandom3
--resume-candidate candidate_0025
```

The sequence looked like this:

| candidate | result | achievement | terminal | track violation |
|---|---:|---:|---:|---:|
| `candidate_0025` | parent context | 1.0 | old objective | 0.0 |
| `candidate_0001` | rejected | 1.0 | 0.0 | 0.0 |
| `candidate_0003` | rejected | 1.0 | 0.0 | 0.0 |
| `candidate_0004` | rejected | 0.0 | 0.0 | 0.0 |
| `candidate_0005` | rejected | 1.0 | 0.0 | 0.819 |
| `candidate_0006` | accepted | 1.0 | 1.0 | 0.0 |

The final candidate used the previous swing-up insight but switched the terminal stabilizer to a more principled local controller family. After tuning, it passed both train seeds and holdout seeds.

<img src="{{ '/assets/img/heuristic-learning-control/cart_link_terminal_candidate_0006.gif' | relative_url }}" alt="Cart-link terminal-stable candidate 0006 found by resume refinement." width="80%">

Train metrics:

```json
{
  "achievement_success_rate": 1.0,
  "terminal_success_rate": 1.0,
  "final_upright_error_mean": 1.82e-08,
  "final_abs_theta_dot_mean": 2.58e-08,
  "final_abs_x_mean": 7.72e-08,
  "max_track_violation_max": 0.0,
  "total_reward_mean": 514.15
}
```

Holdout metrics over five new seeds:

```json
{
  "seeds": [981373, 960845, 531023, 780242, 150136],
  "achievement_success_rate": 1.0,
  "terminal_success_rate": 1.0,
  "final_upright_error_mean": 4.94e-06,
  "max_track_violation_max": 0.0,
  "total_reward_mean": 474.64
}
```

The trajectory metrics also look clean:

<img src="{{ '/assets/img/heuristic-learning-control/cart_link_terminal_candidate_0006_metrics.png' | relative_url }}" alt="Cart-link terminal candidate metrics." width="80%">

### Crowd evacuation: local policy synthesis

After the cart-link result, I wanted a second experiment that was less likely to be solved by simply recalling a textbook controller. So I added a decentralized crowd evacuation benchmark. This is still a toy environment, but it changes the flavor of the problem:

- there are many agents, not one plant,
- every agent runs the same local controller,
- the controller sees nearby agents, walls, obstacles, and an exit direction,
- it does not get the full global state of all agents,
- success requires evacuation and safety, not just high reward.

The controller signature stayed the same style:

```python
def crowd_evacuation_controller(obs: dict) -> np.ndarray:
    ...
```

but now `obs` is a local observation for one agent:

```json
{
  "self_pos": "...",
  "self_vel": "...",
  "goal_direction": "...",
  "distance_to_exit": "...",
  "inside_exit_corridor": "...",
  "local_density": "...",
  "nearby_agents_rel_pos": "...",
  "nearby_agents_rel_vel": "...",
  "nearby_agents_mask": "...",
  "nearby_obstacles_rel_pos": "...",
  "nearest_wall_distances": "...",
  "max_speed": "...",
  "max_accel": "..."
}
```

The `goal_direction` and exit features are task-level cues, not a centralized planner. The controller still cannot inspect every other agent or solve a global routing problem.

I then made the environment harder in stages:

| scenario | agents | change | accepted candidate | success | collisions | wall/obstacle violations |
|---|---:|---|---|---:|---:|---:|
| `default` | 32 | fixed exit | `candidate_0003` | 1.0 | 0.0 | 0.0 |
| `random_exit_v1` | 32 | randomized exit | `candidate_0005` | 1.0 | 0.0 | 0.0 |
| `dense_random_exit_v2` | 48 | denser crowd | `candidate_0007` | 1.0 | 0.0 | 0.0 |
| `obstacle_bottleneck_v3` | 56 | bottleneck obstacles | initially failed | - | - | - |

The first three were solved quickly. The learned controllers were not fancy. They looked like local social-force policies: move toward the exit, repel nearby agents, avoid walls, use density-aware speed limits, and add small tie-breaking terms to prevent symmetric jams.

Here is the first fixed-exit version:

<img src="{{ '/assets/img/heuristic-learning-control/crowd_evacuation_v0_candidate_0003.gif' | relative_url }}" alt="Crowd evacuation v0 fixed-exit candidate 0003." width="80%">

Here is the dense v2 run:

<img src="{{ '/assets/img/heuristic-learning-control/crowd_evacuation_v2_dense_candidate_0007.gif' | relative_url }}" alt="Dense crowd evacuation v2 candidate 0007." width="80%">

The interesting case was `obstacle_bottleneck_v3`. In one failed run, the best "naively successful" candidate evacuated everyone, but did it unsafely:

```json
{
  "candidate": "candidate_0003",
  "evacuation_fraction_mean": 1.0,
  "remaining_agent_count_mean": 0.0,
  "collision_count_mean": 1579.33,
  "wall_violation_count_mean": 28.0,
  "obstacle_violation_count_mean": 28.67
}
```

Another candidate was less visually exciting, but much more useful as a parent:

```json
{
  "candidate": "candidate_0019",
  "evacuation_fraction_mean": 0.9821,
  "remaining_agent_count_mean": 1.0,
  "collision_count_mean": 0.0,
  "wall_violation_count_mean": 0.0,
  "obstacle_violation_count_mean": 0.0,
  "stuck_agent_count_mean": 0.0
}
```

This is exactly the kind of judgment I had been doing manually: pick the candidate that is closest in the right way, then ask the next run to preserve its useful behavior and close the acceptance gap.

So I added a run-level synthesis agent. It runs only at the end of a run. It does not interfere with the within-run whiteboard. If there is no accepted candidate, it reads the whole candidate history and writes a compact resume recommendation:

```markdown
# Run Synthesis Whiteboard

- Mode: `refine_parent`
- Parent candidate: `candidate_0019`
- Confidence: `medium`

## Next Objective
Resume from `candidate_0019`: preserve its best measured behavior and close
these gaps: success_rate, achievement_success_rate, remaining_agent_count_mean.

## Preserve
- Preserve the behavior that produced high evacuation fraction with zero
  collision, wall, and obstacle violations.

## Fix
- Close success_rate from 0.6667 toward 0.9.
- Close remaining_agent_count_mean from 1.0 toward 0.0.
- Reduce near_collision_count_mean without reintroducing obstacle violations.
```

The next run used that file as context:

```bash
--resume-run .../run_20260523-214226-373112_round001_seedsrandom3
--resume-candidate candidate_0019
```

The resumed run accepted `candidate_0010`:

```json
{
  "scenario": "obstacle_bottleneck_v3",
  "n_agents": 56,
  "success_rate": 1.0,
  "achievement_success_rate": 1.0,
  "terminal_success_rate": 1.0,
  "evacuation_fraction_mean": 1.0,
  "remaining_agent_count_mean": 0.0,
  "collision_count_mean": 0.0,
  "wall_violation_count_mean": 0.0,
  "obstacle_violation_count_mean": 0.0,
  "stuck_agent_count_mean": 0.0
}
```

<img src="{{ '/assets/img/heuristic-learning-control/crowd_evacuation_v3_bottleneck_candidate_0010.gif' | relative_url }}" alt="Crowd evacuation obstacle bottleneck v3 accepted candidate 0010." width="80%">

This felt like the most important harness lesson so far. The model did not just need reward. It needed an experimental notebook: which attempt almost worked, what to preserve, what precise gap remained, and what not to regress on.

## What worked

The pieces that mattered most were:

- **structured proposals** instead of free-form patches,
- **inline artifacts** instead of local file paths,
- **separate achievement and terminal success metrics**,
- **candidate workspaces** so bad proposals do not contaminate the repo,
- **reflection summaries** that compress history into a whiteboard,
- **run-level synthesis** that recommends a resume parent after a failed run,
- **parameter tuning** as a first-class step, not an afterthought,
- **holdout evaluation** so the candidate does not only pass the seeds it saw.

The agent was not doing generic "autonomous science" in one magic leap. It was closer to an experimental control loop: propose, run, observe, compress, refine.

## What I would change next

The next step is to make the harness more sandboxed and more benchmark-friendly:

1. run each external RL/control repo in an isolated sandbox profile,
2. support native Python first, then Docker or process-level sandboxes,
3. make trajectory diagnostics richer,
4. add formal verification backends for simple invariants,
5. support resume-from-candidate as a normal workflow,
6. make the whiteboard explicitly versioned.

There is also a deeper research question: how much context should the model see? Too little and it misses the key failure. Too much and it follows stale advice into a local optimum. My current answer is to give it a living whiteboard, not an archive.

## Takeaway

This experiment made the "learning beyond gradients" idea feel concrete to me. The learning signal was not a gradient. It was a bundle of logs, rewards, trajectories, failed controllers, and short reflections. The update was not a weight step. It was a new piece of controller code.

That is slower and messier than gradient descent, but it has a different kind of leverage: the candidate can jump from energy shaping to LQR, from a loose success definition to terminal hold, from a brittle gain schedule to a structured controller family.

For control problems where the answer should be a simple, inspectable controller rather than a neural policy, this is a surprisingly promising search loop.
