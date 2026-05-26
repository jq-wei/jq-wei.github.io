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

This harness sits in a family of recent work where the LLM acts as a code-mutating search operator. The closest works are FunSearch, which searches programs for mathematical problems through LLM mutation and a program database; AlphaEvolve, a larger evolutionary coding-agent version of the same basic instinct; and Voyager, where an LLM writes reusable code skills in open-ended Minecraft. These methods lean on population diversity and many samples. By contrast, Eureka from Nvidia keeps gradient-based RL but uses an LLM to write the reward function, sitting one level above where this harness operates. The **whiteboard mechanism** here feels closer in spirit to GEPA's reflective prompt evolution: natural-language reflection becomes an update signal that compresses experience instead of summing gradients. Compared to a classical optimizer like CMA-ES for parameter search, the LLM contributes structured proposals -- controller families, switching logic, geometric reframings -- that a Gaussian mutation cannot produce.

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
6. apply it in an isolated candidate workspace (sandboxes),
7. run tests and evaluations,
8. tune exposed parameters,
9. compare against the baseline,
10. summarize failure modes into a "whiteboard" (overview of the experiment so far) for the next attempt.

The important part turned out not to be the orchestration itself. It was **LLM agents harness: what context to give the model between iterations**. For the backend LLM, we use Deepseek V4 Pro. 

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

We start the harness which takes the important artifacts like
```markdown
- /Users/.../baseline/aggregate/summary.md
- /Users/.../baseline/aggregate/metrics.json
```
directly into the prompt:

- environment source snippets,
- controller source snippets,
- baseline metrics,
- candidate metrics,
- selected trajectory diagnostics,
- previous candidate summaries,
- current best attempt,
- dominant failure mode,
- concrete next-attempt guidance.

I started thinking of this as a research **whiteboard**. It is not a transcript, nor every log lines. It is a compact, curated working memory.

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

This is the whiteboard doing its job: keep the overview context about the experiments so far, preserve the useful discovery, discard stale noise.

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

Under the initial achievement-style objective, the useful parent that emerged was `candidate_0025`. It used an energy-shaping swing-up strategy and reached the upright success region on all five holdout seeds, without leaving the track:

```json
{
  "controller": "candidate_0025",
  "success_rate": 1.0,
  "first_success_step_mean": 327.2,
  "final_upright_error_mean": 0.1833,
  "final_abs_theta_dot_mean": 0.8065,
  "max_track_violation_max": 0.0
}
```

<img src="{{ '/assets/img/heuristic-learning-control/cart_link_candidate_0025_achievement_parent.gif' | relative_url }}" alt="Cart-link candidate 0025 as the achievement-success parent before terminal refinement." width="80%">

This was useful, but not yet the right controller: it could swing up and enter the success region, while still carrying enough angular velocity that terminal stabilization was not guaranteed. That is why the next run resumed from `candidate_0025` instead of starting over.

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
- the controller uses local information: sees nearby agents, walls, obstacles, and an exit direction,
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

Another candidate was less visually exciting (one agent always failed to escape), but much more useful as a parent:

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

This is exactly the kind of judgment I had been doing manually: pick the candidate that is closest in the right way, then ask the next run to preserve its useful behavior and close the acceptance gap. This is the last piece of my harness: a run-level synthesis agent. It runs only at the end of a run. It does not interfere with the within-run whiteboard. If there is no accepted candidate, it reads the whole candidate history and writes a compact resume recommendation:

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

### LunarLander: an RL-style benchmark

The third experiment was `LunarLanderContinuous`. This was useful because it put the harness much closer to a standard RL benchmark. There is a known neural-policy baseline from Stable-Baselines3, and the task is familiar enough that a handcrafted controller is plausible but not trivial.

This also forced a question that did not matter as much in the first two experiments: **what environment information is fair to give the model?**

In n+e's [Learning Beyond Gradients](https://trinkle23897.github.io/learning-beyond-gradients/) setup, this boundary depends on the task. Their Atari prompt, as released in the [repo](https://github.com/Trinkle23897/learning-beyond-gradients), is much closer to black-box interaction: use public observations, actions, rewards, renderings, and API-visible fields, but do not read simulator internals. The MuJoCo Ant part is different. The blog describes first reading public environment semantics such as observation layout, reward, action order, root velocity, torso orientation, joint positions, and joint velocities (which is already part of the environment dynamics). Later, the stronger Ant result uses residual MPC (model predictive control) with a local MuJoCo model: at each step, roll out short candidate action sequences in a local dynamics model, execute the first action from the best short-horizon plan, and replan at the next step. That is no longer pure black-box reward chasing. It is model-based, or at least gray-box.

For LunarLander I chose a gray-box version of the benchmark. The harness writes a compact environment source digest from the installed Gymnasium implementation and includes it in the prompt. It does not dump the entire simulator, but it tells the model the important semantics:

```markdown
Observation:
- x, y
- x_dot, y_dot
- angle, angular_velocity
- left_leg_contact, right_leg_contact

Action:
- action[0] controls the main engine
- action[1] controls the side engines
- action[0] <= 0 means main engine off
- action[0] > 0 gives at least half-power main thrust
- abs(action[1]) <= 0.5 means side engines off

Reward and termination:
- reward is a change in shaping minus fuel costs
- main-engine use and side-engine use are penalized
- body contact or flying out of bounds terminates badly
- sleeping after a stable landing terminates positively
```

This matters. Without the action-threshold details, a controller can emit tiny positive main-engine commands and accidentally fire at half power. Without the observation semantics, the model has to rediscover the task from reward traces alone. RL can treat the environment as a black box because gradient-based training can do millions of interactions. This harness is trying to synthesize code with tens of experiments, so it benefits from the same kind of task specification a human controls engineer would ask for.

The zero controller falls immediately:

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_zero_baseline.gif' | relative_url }}" alt="LunarLander zero controller baseline." width="80%">

The first good heuristic parent was `candidate_0015`. It solved the reward-threshold style objective on all five reference seeds, but only achieved terminal quiet landing on four of five:

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0015_achievement.gif' | relative_url }}" alt="LunarLander candidate 0015 reaching reward success but not terminal success on every seed." width="80%">

That is the same lesson as cart-link, but in an RL environment: reward success and the state I actually wanted were not identical. The next run resumed from `candidate_0015` with a terminal-refinement whiteboard:

```markdown
Active success mode: `terminal`

Preserve:
- stable descent and reward-solving behavior from `candidate_0015`
- crash_rate = 0.0
- achievement_success_rate = 1.0

Fix:
- convert solved-but-not-terminal cases into true quiet landings
- reduce final lateral error and final angle
- avoid hovering forever or burning unnecessary fuel
```

The accepted result was `candidate_0009`, a compact rule controller rather than a neural network. It uses a vertical descent target, lateral correction, attitude stabilization, contact-aware shutdown, and a tunable set of thresholds and gains.

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_candidate_0009_terminal.gif' | relative_url }}" alt="LunarLander terminal-refined heuristic candidate 0009." width="80%">

On new holdout seeds, `candidate_0009` passed both success modes:

```json
{
  "achievement_success_rate": 1.0,
  "terminal_success_rate": 1.0,
  "landing_rate": 1.0,
  "crash_rate": 0.0,
  "timeout_rate": 0.0,
  "total_reward_mean": 234.88,
  "episode_steps_mean": 362.6,
  "final_landing_error_mean": 0.058
}
```

I then compared it against the Stable-Baselines3 PPO policy from Huggingface `sb3/ppo-LunarLanderContinuous-v2` on the same five seeds. The PPO policy lands faster and scores higher:

<img src="{{ '/assets/img/heuristic-learning-control/lunar_lander_sb3_ppo_reference.gif' | relative_url }}" alt="Stable-Baselines3 PPO LunarLander reference policy." width="80%">

| policy | terminal success | mean reward | mean steps | landing error | main engine | side engine |
|---|---:|---:|---:|---:|---:|---:|
| heuristic `candidate_0015` | 0.80 | 225.19 | 408.2 | 0.119 | 0.467 | 0.108 |
| heuristic `candidate_0009` | 1.00 | 236.36 | 381.8 | 0.068 | 0.210 | 0.287 |
| SB3 PPO reference | 1.00 | 257.35 | 278.6 | 0.033 | 0.123 | 0.352 |

This comparison is exactly the nuance I wanted. The heuristic controller closed the main specification gap: it landed terminally on all five seeds. PPO was still better at the native RL objective: higher reward, fewer steps, and more precise final placement.

The reason is partly objective design. The Gym reward is dense and includes shaping, terminal bonuses/penalties, and fuel costs. PPO optimizes that reward directly, so shorter, cleaner, lower-fuel landings are naturally preferred. My harness acceptance, on the other hand, first asked: did it land safely and quietly? Reward was still recorded and used as a secondary signal, but it was not the only gate. If I wanted the heuristic controller to chase PPO's score more aggressively, I would add another refinement phase that explicitly optimizes fuel and step count after terminal success is solved.

This is also where the control-vs-RL distinction becomes concrete. PPO gives a strong policy, but it is a trained neural network artifact plus its inference stack. The heuristic candidate is a small piece of controller code with a few scalar operations and a parameter dictionary. I have not yet done a careful embedded inference benchmark, so I will not claim a measured speedup. But as an object, the heuristic controller is cheaper to inspect, easier to port, and easier to reason about than a neural policy. The tradeoff is that it depends more on task semantics and on the harness giving the model the right experimental whiteboard.

So I do not read this as "heuristic control replaces RL." For me this is: for some dynamic or geometric control tasks, there may be a much smaller controller hiding behind a neural-policy benchmark, and an LLM-guided harness can sometimes find it. 

## What worked

The pieces that mattered most were:

- **structured proposals** instead of free-form patches,
- **inline artifacts** instead of local file paths,
- **environment source digests** for third-party benchmark semantics,
- **separate achievement and terminal success metrics**,
- **candidate workspaces** so bad proposals do not contaminate the repo,
- **reflection summaries** that compress history into a whiteboard,
- **run-level synthesis** that recommends a resume parent after a failed run,
- **parameter tuning** as a first-class step, not an afterthought,
- **holdout evaluation** so the candidate does not only pass the seeds it saw.

The agent was not doing generic "autonomous science" in one magic leap. It was closer to an experimental control loop: propose, run, observe, compress, refine.

## What is coming

The next step is to make the harness more sandboxed and more benchmark-friendly:

1. run each external RL/control repo in an isolated sandbox profile,
2. make trajectory diagnostics richer (multimodal)
3. add formal verification backends for safty,

There are also deeper research questions: how much context should the model see? Too little and it misses the key failure. Too much and it follows stale advice into a local optimum. Where is the boundary of this method?

## Backend note

All of these runs used `deepseek-v4-pro` as the backend LLM. One practical surprise was cost: the dashboard for this month showed 796 API requests and about 15.4M tokens, with total spend around 45 CNY. That makes this style of experiment feel much more plausible as an iterative research loop. 

## Takeaway

This experiment made the "learning beyond gradients" idea feel concrete to me. The learning signal was not a gradient. It was a bundle of logs, rewards, trajectories, failed controllers, and short reflections. The update was not a weight step. It was a new piece of controller code.

That is slower and messier than gradient descent, but it has a different kind of leverage: the candidate can jump from energy shaping to LQR, from a loose success definition to terminal hold, from a brittle gain schedule to a structured controller family.

For control problems where the answer should be a simple, inspectable controller rather than a neural policy, this is a surprisingly promising search loop.

## References: 
1. [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) 
2. [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
3. [Voyager](https://arxiv.org/abs/2305.16291)
4. [Eureka](https://arxiv.org/abs/2310.12931)
5. [GEPA](https://arxiv.org/abs/2507.19457)
6. [CMA-ES](https://www.cmap.polytechnique.fr/~nikolaus.hansen/cmaesintro.html).

## Controller appendix

For completeness, here are the three final controller bodies from the accepted runs. I keep them here because the details are part of the point: the final artifacts are small, inspectable programs rather than opaque policy checkpoints.

<details markdown="1">
<summary>Cart-link final controller: swing-up plus Riccati/LQR stabilizer</summary>

```python
# Harness-tunable parameters for CANDIDATE_0006_PARAMS.
CANDIDATE_0006_PARAMS = {
    "Q_theta": 104.54373361725591,
    "Q_theta_dot": 27.120087769452226,
    "Q_x": 93.5888398527996,
    "Q_x_dot": 19.261746401250043,
    "R_scalar": 1.319804917991193,
    "k_break": 17.01040850167213,
    "k_energy": 116.63426700076063,
    "k_x_barrier": 25.20462043723865,
    "k_x_dot_swing": 5.881524624109873,
    "k_x_swing": 21.82250892589433,
    "swing_modulation_steepness": 48.40347644757355,
    "theta_capture": 0.3667194533654743,
    "theta_dot_break_threshold": 0.9835076873750547,
    "theta_dot_capture": 1.0855961579629034,
    "x_capture_absmax": 0.4174467380969573,
    "x_dot_capture_absmax": 0.9894427502531395,
    "x_soft_limit": 0.972709654002867,
}


def candidate_0006_controller(obs: dict) -> np.ndarray:
    params = CANDIDATE_0006_PARAMS
    x = obs["x"]
    x_dot = obs["x_dot"]
    theta = obs["theta"]
    theta_dot = obs["theta_dot"]
    cos_theta = obs["cos_theta"]
    sin_theta = obs["sin_theta"]
    force_limit = obs["force_limit"]

    cart_mass = 1.0
    pole_mass = 0.1
    pole_half_length = 0.5
    gravity = 9.81
    cart_damping = 0.02
    pole_damping = 0.01

    m_pole = pole_mass
    l = pole_half_length
    m_total = cart_mass + m_pole
    m_l = m_pole * l
    g = gravity
    b_c = cart_damping
    b_p = pole_damping

    if not hasattr(candidate_0006_controller, "_lqr_K"):
        from scipy.linalg import solve_continuous_are

        D0 = l * (4.0 / 3.0 - m_pole / m_total)
        factor = m_l / (m_total * D0)
        A = np.array([
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -b_c / m_total * (1 - factor),
             -m_l * g / (m_total * D0),
             m_l * b_p / (m_total * D0)],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, b_c / (m_total * D0),
             g / D0,
             -b_p / D0],
        ])
        B = np.array([
            [0.0],
            [(1 - factor) / m_total],
            [0.0],
            [-1.0 / (m_total * D0)],
        ])

        Q = np.diag([
            params["Q_x"],
            params["Q_x_dot"],
            params["Q_theta"],
            params["Q_theta_dot"],
        ])
        R = np.array([[params["R_scalar"]]])
        P = solve_continuous_are(A, B, Q, R)  # ricatti equation
        candidate_0006_controller._lqr_K = np.linalg.inv(R) @ B.T @ P

    K = candidate_0006_controller._lqr_K

    I_pend = (4.0 / 3.0) * m_pole * l ** 2
    mgl = m_pole * g * l
    E = 0.5 * I_pend * theta_dot ** 2 + mgl * (cos_theta - 1.0)

    captured = (
        abs(theta) < params["theta_capture"]
        and abs(theta_dot) < params["theta_dot_capture"]
        and abs(x) < params["x_capture_absmax"]
        and abs(x_dot) < params["x_dot_capture_absmax"]
    )

    if captured:
        u_stab = -np.dot(K, np.array([x, x_dot, theta, theta_dot]))
        return np.clip(u_stab, -force_limit, force_limit)

    steepness = params["swing_modulation_steepness"]
    u_swing_raw = params["k_energy"] * E * np.tanh(theta_dot * cos_theta * steepness)
    u_swing = force_limit * np.tanh(u_swing_raw / force_limit) if force_limit > 0 else u_swing_raw

    u_break = 0.0
    if abs(theta_dot) < params["theta_dot_break_threshold"]:
        u_break = -params["k_break"] * sin_theta

    u_cart = -params["k_x_swing"] * x - params["k_x_dot_swing"] * x_dot

    u_barrier = 0.0
    if abs(x) > params["x_soft_limit"]:
        u_barrier = -params["k_x_barrier"] * np.sign(x) * (abs(x) - params["x_soft_limit"])

    u_total = u_swing + u_break + u_cart + u_barrier
    return np.clip(u_total, -force_limit, force_limit)
```

</details>

<details markdown="1">
<summary>Crowd evacuation final controller: local goal seeking, repulsion, corridor centering, stuck recovery</summary>

```python
# Harness-tunable parameters for CANDIDATE_0010_PARAMS.
CANDIDATE_0010_PARAMS = {
    "centering_threshold": 1.364311354815481,
    "early_centering_gain": 3.7230993365131972,
    "early_centering_threshold": 2.7241833440893757,
    "exit_centerline_gain": 8.043229854644142,
    "k_damp": 4.243425129408768,
    "k_goal": 3.0829730354299416,
    "k_neighbor": 6.070913415695911,
    "k_neighbor_vel_brake": 6.72098102341149,
    "k_obstacle": 6.737337630968138,
    "k_wall": 10.542310105078164,
    "neighbor_margin": 1.0889233832423975,
    "obstacle_margin": 0.6400570467682112,
    "right_wall_suppression_distance": 1.2398409830303327,
    "stuck_goal_boost": 1.31063808814712,
    "stuck_speed_threshold": 0.17391402277625154,
    "wall_margin": 0.6400041506464316,
}


def candidate_0010_controller(obs: dict) -> np.ndarray:
    params = CANDIDATE_0010_PARAMS

    goal_dir = np.asarray(obs["goal_direction"], dtype=float)
    self_vel = np.asarray(obs["self_vel"], dtype=float)
    max_accel = float(obs["max_accel"])
    inside_corridor = bool(obs["inside_exit_corridor"])

    action = params["k_goal"] * goal_dir - params["k_damp"] * self_vel

    obstacle_rel_pos = np.asarray(obs["nearby_obstacles_rel_pos"], dtype=float)
    obstacle_dist = np.asarray(obs["nearby_obstacles_dist"], dtype=float)
    obstacle_mask = np.asarray(obs["nearby_obstacles_mask"], dtype=bool)
    for i in np.flatnonzero(obstacle_mask):
        d = obstacle_dist[i]
        if d < params["obstacle_margin"]:
            direction = -obstacle_rel_pos[i] / (d + 1e-8)
            mag = params["k_obstacle"] * (
                1.0 / (d + 1e-4) - 1.0 / params["obstacle_margin"]
            )
            action += mag * direction

    wall_dists = np.asarray(obs["nearest_wall_distances"], dtype=float)
    wall_normals = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]], dtype=float)
    exit_wall_dist = float(obs["exit_wall_distance"])
    for i in range(4):
        d = wall_dists[i]
        if d < params["wall_margin"]:
            if i == 1:
                if inside_corridor and exit_wall_dist < params["right_wall_suppression_distance"]:
                    continue
            mag = params["k_wall"] * (1.0 / (d + 1e-4) - 1.0 / params["wall_margin"])
            action += mag * wall_normals[i]

    nb_rel_pos = np.asarray(obs["nearby_agents_rel_pos"], dtype=float)
    nb_rel_vel = np.asarray(obs["nearby_agents_rel_vel"], dtype=float)
    nb_dist = np.asarray(obs["nearby_agents_dist"], dtype=float)
    nb_mask = np.asarray(obs["nearby_agents_mask"], dtype=bool)
    for i in np.flatnonzero(nb_mask):
        d = nb_dist[i]
        if d < params["neighbor_margin"]:
            unit_dir = -nb_rel_pos[i] / (d + 1e-8)
            rep_mag = params["k_neighbor"] * (
                1.0 / (d + 1e-4) - 1.0 / params["neighbor_margin"]
            )
            action += rep_mag * unit_dir
            closing = np.dot(nb_rel_vel[i], unit_dir)
            if closing > 0:
                action += params["k_neighbor_vel_brake"] * closing * unit_dir

    if inside_corridor and exit_wall_dist < params["centering_threshold"]:
        offset = float(obs["exit_centerline_offset"])
        action[1] -= params["exit_centerline_gain"] * offset

    if (not inside_corridor) and exit_wall_dist < params["early_centering_threshold"]:
        offset = float(obs["exit_centerline_offset"])
        weight = 1.0 - exit_wall_dist / params["early_centering_threshold"]
        action[1] -= params["early_centering_gain"] * weight * offset

    self_speed = float(np.linalg.norm(self_vel))
    if self_speed < params["stuck_speed_threshold"] and not inside_corridor:
        action += params["stuck_goal_boost"] * goal_dir

    norm = float(np.linalg.norm(action))
    if norm > max_accel:
        action = (action / norm) * max_accel
    return action.astype(float)
```

</details>

<details markdown="1">
<summary>LunarLander final controller: descent schedule, attitude control, landing latch</summary>

```python
# Harness-tunable parameters for CANDIDATE_0009_PARAMS.
CANDIDATE_0009_PARAMS = {
    "k_ang_vel": 25.93714348415618,
    "k_angle": 19.098638950419804,
    "k_vy": 40.38281687444112,
    "k_x": 0.938192800387466,
    "k_x_dot": 50.75314908218012,
    "land_ang_vel_thresh": 0.33302085121301467,
    "land_angle_thresh": 0.02464608827566077,
    "land_vy_thresh": 0.2928250962106883,
    "main_power_scale": 0.25548053709167745,
    "side_deadzone": 0.5732596023537759,
    "side_min_mag": 0.8946047782667153,
    "target_vy_descend": -0.7114750601886578,
    "target_vy_soft": -0.14833753699931732,
    "vy_deadband": 0.039846249153573335,
    "y_high": 8.129027797271942,
    "y_threshold_low": 1.8610752599674172,
}


def candidate_0009_controller(obs: dict) -> np.ndarray:
    params = CANDIDATE_0009_PARAMS

    k_x = params["k_x"]
    k_x_dot = params["k_x_dot"]
    k_angle = params["k_angle"]
    k_ang_vel = params["k_ang_vel"]
    k_vy = params["k_vy"]
    vy_deadband = params["vy_deadband"]
    target_vy_descend = params["target_vy_descend"]
    target_vy_soft = params["target_vy_soft"]
    y_high = params["y_high"]
    y_threshold_low = params["y_threshold_low"]
    side_deadzone = params["side_deadzone"]
    side_min_mag = params["side_min_mag"]
    main_power_scale = params["main_power_scale"]
    land_vy_thresh = params["land_vy_thresh"]
    land_angle_thresh = params["land_angle_thresh"]
    land_ang_vel_thresh = params["land_ang_vel_thresh"]

    global _candidate_0009_state
    if obs["step"] == 0:
        _candidate_0009_state = {"landed": False, "touchdown_phase": False}

    x = obs["x"]
    y = obs["y"]
    x_dot = obs["x_dot"]
    vy = obs["y_dot"]
    angle = obs["angle"]
    ang_vel = obs["angular_velocity"]
    left_leg = obs["left_leg_contact"]
    right_leg = obs["right_leg_contact"]
    both_legs = left_leg and right_leg

    if not _candidate_0009_state["landed"]:
        if (
            both_legs
            and abs(vy) < land_vy_thresh
            and abs(angle) < land_angle_thresh
            and abs(ang_vel) < land_ang_vel_thresh
        ):
            _candidate_0009_state["landed"] = True

    if _candidate_0009_state["landed"]:
        return np.zeros(2, dtype=float)

    if not _candidate_0009_state["touchdown_phase"] and (left_leg or right_leg):
        _candidate_0009_state["touchdown_phase"] = True

    if _candidate_0009_state["touchdown_phase"]:
        target_vy = target_vy_soft
    elif y >= y_high:
        target_vy = target_vy_descend
    elif y <= y_threshold_low:
        target_vy = target_vy_soft
    else:
        frac = (y - y_threshold_low) / (y_high - y_threshold_low)
        target_vy = target_vy_soft + frac * (target_vy_descend - target_vy_soft)

    err_vy = target_vy - vy
    if err_vy > vy_deadband:
        des_thrust = k_vy * err_vy
        thrust_frac = min(max(des_thrust * main_power_scale, 0.0), 1.0)
        main_action = max(2.0 * thrust_frac - 1.0, 0.0)
    else:
        main_action = 0.0

    lateral_correction = k_x * x - k_x_dot * x_dot
    attitude_correction = k_angle * angle + k_ang_vel * ang_vel
    raw_side = lateral_correction + attitude_correction
    raw_side = max(-1.0, min(1.0, raw_side))

    if abs(raw_side) < side_deadzone:
        side_action = 0.0
    else:
        mag_frac = (abs(raw_side) - side_deadzone) / (1.0 - side_deadzone)
        mag = side_min_mag + (1.0 - side_min_mag) * mag_frac
        side_action = np.sign(raw_side) * mag
        side_action = max(-1.0, min(1.0, side_action))

    return np.array([main_action, side_action], dtype=float)
```

</details>
