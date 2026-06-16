---
layout: post
title: "From LunarLander to String-Stable Platooning"
date: 2026-06-11 22:20:00 +0200
description: "An LLM-guided handwritten controller suppresses string amplification under delay, noise, and heterogeneity; a tuned robust CACC baseline then clarifies the real boundary."
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

This post records the platooning experiment end to end. It started with a clean stage-one result on delay and noise, then moved through hard heterogeneity, and finally hit a useful limit on a combined stress test. The final result is not a proof of worst-case string stability, but it is more interesting than a single nice rollout: the harness produced local, non-neural controllers that are clearly better than simple classical baselines under the hard combined setting, while a later Optuna-tuned robust CACC baseline showed that the strongest result is better read as competitive controller synthesis, not as a claim that LLM search dominates all tuned classical designs.

## Why platooning?

Vehicle platooning is a good next target because the important property is not only tracking error. It is whether disturbances amplify as they travel down the string of vehicles.

In a simple reward-only setup, a controller can look good while quietly passing too much disturbance to downstream vehicles. For platooning, this is exactly the wrong failure mode. A follower near the tail should not experience a larger spacing transient than the vehicle in front of it.

That makes string stability a natural target for this harness. It also moves the experiment closer to networked control: local observations, delayed information, heterogeneous vehicles, and robustness to leader disturbances.

## Environment

The experiment used a small environment ladder rather than a single fixed task.

The simulator is lightweight: a scripted leader and 19 followers, where the same decentralized controller is applied to each follower. Each follower only sees local predecessor information.

The first hard target was `hard_delay_noise_v1`. After that passed, I moved to `hard_heterogeneous_v1`, and then to `hard_combo_v1`.

| Scenario | Added stressors | Purpose |
|---|---|---|
| `hard_delay_noise_v1` | repeated hard braking, 3-step observation delay, up to 3 extra jitter steps, 0.35 predecessor-acceleration dropout, observation noise, fixed sensor bias | Can the controller suppress string amplification with delayed and unreliable local observations? |
| `hard_heterogeneous_v1` | repeated hard braking, broad actuator lag/limit spread, one weak-braking mid-platoon vehicle | Can the controller handle vehicle-to-vehicle dynamics mismatch? |
| `hard_combo_v1` | repeated hard braking, variable delay/noise, 0.40 acceleration dropout, sensor bias, heterogeneous dynamics, weak braking | Does the design survive when the stressors are combined? |

The key metric is `string_peak_gain`. It is computed from peak spacing errors along the platoon, using the maximum of:

- neighbor peak amplification: downstream peak error divided by upstream peak error
- tail peak amplification: tail peak error divided by the first follower peak error

For these rounds, I added a dedicated `string_performance` success mode. The main gate was:

```text
string_peak_gain_max <= 1.35
collision_rate = 0
spacing_error_rms_mean <= 3.0
success_rate high enough on train and holdout
```

This matters because earlier performance-style runs could pass too easily by finding a safe-enough controller with a loose negative reward target. The new mode makes the platooning claim explicit: suppress amplification.

One detail is worth spelling out because it changes how I read the later failures. The random seeds are not mainly making the cars start unrealistically close together. With the default `d0 = 8`, `time_headway = 1`, and `nominal_speed = 20`, the desired initial gap is about 28 meters, with only +/- 0.5 meters of initial gap jitter. In `hard_combo_v1`, the hard part is the interaction of delay, dropout, noise, bias, heterogeneous actuation, and repeated braking.

## First accepted controller

The first accepted hard-scenario controller was `candidate_0010`, resumed from the `hard_delay_noise_v1` platooning run and accepted under `string_performance`.

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

## Harder ladder

After the first acceptance, I pushed the same harness up the hard-scenario ladder. This is where the experiment became more informative.

One small naming caveat: candidate names are local to a run. The `candidate_0010` in the hard-heterogeneous run is a new candidate in that run, not literally the same file as the `hard_delay_noise_v1` candidate above.

The ladder result was:

| Stage | Scenario | Best candidate | Train success | Holdout / fresh success | Collision | Max string gain | Status |
|---|---|---|---:|---:|---:|---:|---|
| Delay/noise | `hard_delay_noise_v1` | `candidate_0010` | 1.000 | 1.000 holdout | 0.000 | 1.170 holdout | accepted |
| Heterogeneous | `hard_heterogeneous_v1` | `candidate_0010` | 1.000 | 1.000 holdout | 0.000 | 0.753 holdout | accepted |
| Combined | `hard_combo_v1` | `candidate_0002` | 1.000 | 0.800 holdout / 0.900 fresh-50 | 0.000 | 1.544 holdout / 1.801 fresh-50 | not accepted |

The hard-heterogeneous result was cleaner than I expected. With broad actuator lag/limit spread and one weak-braking vehicle, the accepted candidate passed both train and holdout with low spacing error:

| Split | Seeds | Success | Collision | Min gap | Spacing RMS | Mean string peak gain | Max string peak gain | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0, 1, 2 | 1.000 | 0.000 | 15.818 | 0.062 | 0.616 | 0.706 | -5.89 |
| Holdout | 799010, 111609, 813424, 667812, 893411 | 1.000 | 0.000 | 15.972 | 0.060 | 0.574 | 0.753 | -5.62 |

`hard_combo_v1` changed the story. It combines the previous stressors: repeated leader braking, delay jitter, observation noise, acceleration dropout, sensor bias, heterogeneous dynamics, and a weak-braking follower. The best useful candidate there was `candidate_0002`.

On the original train seeds, `candidate_0002` passed. On the fixed holdout seeds, it failed the string-performance gate:

| Split | Seeds | Success | Collision | Min gap | Spacing RMS | Mean string peak gain | Max string peak gain | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Train | 0, 1, 2 | 1.000 | 0.000 | 7.938 | 1.273 | 0.958 | 1.216 | -1047.03 |
| Holdout | 799010, 111609, 813424, 667812, 893411 | 0.800 | 0.000 | 8.560 | 1.212 | 1.039 | 1.544 | -923.40 |

The holdout failure was seed `893411`: no collision, a healthy minimum gap, but `string_peak_gain = 1.544`, above the 1.35 gate. That is a qualitatively different failure from the simple baselines crashing.

I tried one more small repair loop. It was instructive but not a win. A later `candidate_0009` passed the original train seeds but made the holdout worse, with success 0.600, collision 0.200, and max string gain 3.321. A later `candidate_0012` fixed seed `893411` on a five-seed middle gate, but failed seeds `0` and `667812`, ending at success 0.600 and max string gain 1.372. In other words, the repair was trading one tail failure for another.

At that point the right move was not another 20-round LLM run. It was a fresh-seed sweep.

## Fresh-seed sweep

I evaluated `candidate_0002`, the later `candidate_0012`, three simple baselines, and then a tuned robust CACC baseline on the same 50 fresh `hard_combo_v1` seeds.

| Controller | Success | Collision | Spacing RMS | Mean string gain | P95 string gain | Max string gain | Reward | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `optuna_robust_cacc_shared` | 0.920 | 0.000 | 1.156 | 1.054 | 1.472 | 1.669 | -841.15 | 4/50 |
| `candidate_0002` | 0.900 | 0.000 | 1.206 | 1.049 | 1.477 | 1.801 | -938.27 | 5/50 |
| `candidate_0012` | 0.580 | 0.000 | 1.492 | 1.398 | 1.853 | 3.215 | -1383.32 | 21/50 |
| `acc_pd` | 0.000 | 1.000 | 2.791 | 2.904 | 4.302 | 5.074 | -11879.94 | 50/50 |
| `cacc` | 0.000 | 0.920 | 3.010 | 6.772 | 11.875 | 20.858 | -12764.22 | 50/50 |
| `damped_cacc` | 0.000 | 1.000 | 4.706 | 3.758 | 6.446 | 9.238 | -14928.90 | 50/50 |

This table changed my interpretation twice. First, `candidate_0002` is not just a seed-specific patch: on 50 unseen seeds it stayed collision-free and passed 45 out of 50. But it is also not a worst-case string-stable controller. The five failed seeds were all string-gain tail failures, not collisions:

| Seed | Collision | Min gap | Spacing RMS | String peak gain |
|---:|---:|---:|---:|---:|
| 781245 | false | 8.283 | 1.241 | 1.520 |
| 775016 | false | 7.022 | 1.757 | 1.523 |
| 468502 | false | 10.231 | 0.883 | 1.801 |
| 612709 | false | 7.787 | 1.294 | 1.424 |
| 804226 | false | 10.576 | 1.158 | 1.357 |

Second, the tuned baseline matters, but it should be interpreted carefully. This was not an independent robust CACC design dropped in from nowhere. It was a second-stage baseline built from the structure exposed by the LLM-discovered controller: damping, predecessor-acceleration filtering/feedforward, command smoothing, jerk limiting, emergency braking, and capped catch-up. Once I gave that robust CACC-shaped family a fair Optuna tuning loop, it reached 46 out of 50 fresh seeds, stayed collision-free, and slightly improved the worst observed string gain compared with the raw LLM candidate.

That makes the endpoint clearer and less inflated. The harness found a much stronger controller than the simple baselines for this hard combined setting. But the final result is not "LLM beats classical control." It is closer to: the LLM-guided harness discovered a useful robust-CACC-like structure, and Optuna then refined that structure slightly better than the raw LLM candidate.

There is also a prior-knowledge caveat. The structure is not novel in the control-theoretic sense. Most of its ingredients are familiar CACC or robust-control motifs: damping, feedforward, filtering, smoothing, jerk limits, safety guards, and string-stability-oriented objectives. A language model may well have seen these patterns in papers, code, or technical explanations. So the result should not be read as "the LLM invented CACC." It is better read as an engineering result: the harness could assemble known motifs, adapt them to this custom simulator and hard metric, and turn them into a runnable controller through failure-driven iteration.

## Tuned robust CACC baseline

The Optuna baseline was deliberately not a neural controller. It was also not meant to be a fully independent baseline. I fixed a robust CACC-like controller family by distilling the mechanisms that kept appearing in the successful LLM candidates, especially `candidate_0002`, and then tuned only its gains and thresholds:

- spacing feedback
- relative-velocity damping
- predecessor-acceleration feedforward
- own-acceleration damping
- predecessor-acceleration filtering
- delay-aware feedforward decay
- command smoothing
- jerk limiting
- gap/time-to-collision emergency braking
- capped catch-up acceleration

There is one important fairness caveat. During inspection, I found that the strong LLM candidate effectively uses rollout-local shared smoothing/filter state. A strictly per-follower-memory Optuna controller could not reproduce that behavior. So I used `optuna_robust_cacc_shared`, which allows the same kind of rollout-local shared state but resets it at the start of each rollout and seed. This is a fair comparison to the discovered candidate, but it is not the same as proving a strictly decentralized per-follower memory design.

This makes the comparison three-layered:

| Layer | What it tests | Fresh-50 result |
|---|---|---:|
| Simple hand baselines: `acc_pd`, `cacc`, `damped_cacc` | Default classical controllers under the hard combined stressor | 0/50 success |
| LLM-discovered controller: `candidate_0002` | Can the harness discover a useful controller structure? | 45/50 success |
| LLM-structured + Optuna-tuned: `optuna_robust_cacc_shared` | Can numerical tuning refine the discovered structure? | 46/50 success |

The tuned baseline's training and holdout metrics were:

| Split | Seeds | Success | Collision | Spacing RMS | Mean string peak gain | Max string peak gain | Reward |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 20 mixed seeds | 1.000 | 0.000 | 1.103 | 0.994 | 1.325 | -759.51 |
| Holdout | 799010, 111609, 813424, 667812, 893411 | 0.800 | 0.000 | 1.183 | 1.322 | 2.319 | -869.83 |

That holdout max gain looks worse than `candidate_0002`, but the broader Fresh-50 result was slightly better. This is exactly why the fresh sweep was useful: a five-seed gate can overstate or understate tail behavior depending on which rare cases it samples.

## Interpretation

This result is the point where the harness feels like it is aiming at a real networked-control property rather than just solving another benchmark.

The claim I would make is modest:

1. The harness can synthesize an interpretable controller that is much stronger than simple hand-designed baselines on the hard combined platooning setting.
2. Adding `string_performance` changes the search pressure in the right direction.
3. The discovered controller is not magic; it looks like a plausible robust local CACC variant.
4. A strong tuned robust CACC baseline can match or slightly exceed the best LLM-discovered hard-combo candidate, but that baseline reuses the controller family suggested by the LLM search.
5. The current result is statistical and empirical, not a formal guarantee.
6. The hard-combo failures identify a real next problem: rare string-gain amplification without collision.

The guarantee point matters. This is not a proof of formal string stability. It is also not a full industrial platooning stack. It is a controlled experiment showing that an LLM-guided code-search loop can move toward a network-level control objective when the objective is made explicit. The tuned-baseline result makes the story healthier: the harness is useful less as a final numerical optimizer and more as a structure-discovery loop. Once the right controller family is exposed, ordinary black-box tuning can do what it is good at: calibrate gains and thresholds.

## What this does not yet prove

This stage does not yet prove:

- formal string stability
- worst-case robustness over all random seeds or all uncertainty realizations
- scalability to 50 or 100 vehicles
- robustness to burst communication loss
- robustness to cut-in or cut-out events
- superiority over a tuned MPC or formally designed robust CACC controller
- superiority over carefully tuned robust baselines; the Optuna-tuned robust CACC baseline slightly outperformed the best hard-combo LLM candidate on this Fresh-50 suite
- strict decentralization if rollout-local shared smoothing/filter state is disallowed
- that the whiteboard mechanism is causally necessary

The last point is especially important. The right ablation is not another nice GIF. It is a code-search curve: full whiteboard versus no whiteboard, no reflection, no previous-candidate summaries, and random/no-whiteboard baselines.

## Next

For this stage, I would stop here.

The experiment has done enough to be useful: it produced accepted controllers for delay/noise and heterogeneity, found a strong but imperfect hard-combo controller, and then showed on 50 fresh seeds that the remaining issue is tail string-gain amplification rather than collision avoidance. Another round of single-seed patching would be less informative than changing the evaluation protocol.

The Optuna baseline adds one more useful closure: the best hard-combo LLM candidate is genuinely strong, but not uniquely strong. A tuned robust CACC family can reach the same regime. That is a better endpoint than an easy win over weak baselines.

If I continue this line, the next version should make the task harder in dimensions that matter for networked control and report robustness more directly:

1. Scale from 20 vehicles to 50 vehicles.
2. Use bursty packet dropout, not only independent acceleration dropout.
3. Add stochastic communication delay with a longer tail, for example 0 to 8 steps.
4. Add heterogeneous mass, actuator lag, acceleration limit, and braking limit together.
5. Add one or two weak-braking vehicles at unknown positions.
6. Add sensor bias that drifts slowly during the rollout.
7. Add cut-in, cut-out, or platoon split/merge events.
8. Report a Pareto frontier: nominal reward versus worst-case string gain.
9. Separate strictly per-follower memory controllers from rollout-local shared-state controllers.

For the next acceptance target, I would keep:

```text
collision_rate = 0
success_rate >= 0.9 on a fresh-seed suite
string_peak_gain_max <= 1.35 on fixed holdout and fresh seeds
```

but add a full-suite robustness metric so a controller cannot overfit only the training scenario. I would also keep the tuned robust CACC baseline in the loop from the start, and add a stronger model-based baseline such as MPC if the goal is to make a broader control claim.

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

The hard-heterogeneous accepted run was:

```text
harness_runs/platooning/hard_heterogeneous_v1/zero/run_20260611-234904-746176_round001_seeds0-1-2
candidate_0010
```

The best hard-combo parent was:

```text
harness_runs/platooning/hard_combo_v1/zero/run_20260612-080317-420061_round001_seeds0-1-2
candidate_0002
```

The 50-seed hard-combo sweep outputs were:

```text
experiments/platooning_seed_sweep/candidate_0002/20260612-212849-088659
experiments/platooning_seed_sweep/candidate_0012/20260612-213007-156787
experiments/platooning_seed_sweep/acc_pd/20260613-115748-824820
experiments/platooning_seed_sweep/cacc/20260613-115749-139933
experiments/platooning_seed_sweep/damped_cacc/20260613-115748-823188
experiments/platooning_seed_sweep/optuna_robust_cacc_shared/20260613-115449-551506
```

For example, the fresh-seed sweep for `candidate_0002` was:

```bash
SEEDS=(58421 92734 140569 203847 286105 319774 401226 452991 506318 537902 \
  590144 641337 682019 734508 781245 826730 874192 918604 963771 995318 \
  12763 68244 155902 236481 309115 370226 429884 493021 558730 604118 \
  665902 719443 775016 831557 889201 944602 982331 34791 101884 178223 \
  249650 333908 391770 468502 522614 612709 706381 804226 871009 953440)

PYTHONPATH=harness_runs/platooning/hard_combo_v1/zero/run_20260612-080317-420061_round001_seeds0-1-2/candidates/candidate_0002/candidate_workspace/src \
  .venv311/bin/python -m hl_control.eval.run_platooning \
  --controller candidate_0002 \
  --scenario hard_combo_v1 \
  --seeds "${SEEDS[@]}" \
  --output-root experiments/platooning_seed_sweep/candidate_0002
```

The tuned robust CACC run was:

```bash
PYTHONPATH=src .venv311/bin/python -m hl_control.eval.tune_platooning_optuna \
  --controller-family shared \
  --scenario hard_combo_v1 \
  --n-trials 400 \
  --n-jobs 4 \
  --sampler-seed 2 \
  --output-dir experiments/platooning_optuna_robust_cacc_shared
```

It wrote:

```text
experiments/platooning_optuna_robust_cacc_shared/20260613-115427-591661
```

The fresh-seed sweep for the tuned baseline used the frozen parameters from that run:

```bash
PYTHONPATH=src .venv311/bin/python -m hl_control.eval.run_platooning \
  --controller optuna_robust_cacc_shared \
  --scenario hard_combo_v1 \
  --controller-params-json experiments/platooning_optuna_robust_cacc_shared/20260613-115427-591661/best_params.json \
  --seeds "${SEEDS[@]}" \
  --output-root experiments/platooning_seed_sweep/optuna_robust_cacc_shared
```
