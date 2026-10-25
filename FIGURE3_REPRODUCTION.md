# Figure 3 reproduction and default audit

## What we can trace

The closest traceable training configuration is `ma3_obs_starve_maxsteps_t800_5fish_two_net`.
This is an evidence-based reconstruction, not an identified original model/seed.

* Paper Figure 3 describes two predators, at most six prey, 21M nominal steps, and eventual maintenance of roughly two prey.
* `plots/gen_inn_last_fish_pop_chart.py:14` reads `pickles/t800_5fish_last_fish_pop.pickle`. The filename identifies t800 and five initial fish; the two-network interpretation is supported by two logged rollouts per update and the author's blog, which explicitly says two PPO networks.
* The saved NumPy array has 13,981 values, consistent with a 20-point valid moving average of 14,000 rollout summaries (inferred, not recorded in the pickle). The plot then applies a further 100-point valid average and retains an inaccurate 13,980-value axis extent. First 200 values average 6.0; indices 2000–4799 average about 1.575; last 200 average 2.484. Thus the distributed data itself contains a transition consistent with the figure; our earlier run does not invalidate that curve.
* Candidate config introduced in `0a6a9ba` (2020-12-13); it is unchanged between the artifact-addition commit `f80f03b` (2021-01-04) and checked-out `0060131`.
* README example config was introduced later, in `387bbc3` (2021-02-08). It belongs to the later hard-hunting/cooperation family. The README offers it as an example, not as Figure 3's exact recipe.

Author's explanation: https://blog.xa0.de/post/Emergent-Behaviour-in-Multi-Agent-Reinforcement-Learning%20---%20Independent-PPO/
Source: https://github.com/instance01/fish-rl-alife

## Resolved configuration comparison

| Setting | JSON base `1` | README example / previous run | Figure 3 candidate |
|---|---:|---:|---:|
| Predators | 1 | 2 | 2 |
| Initial prey | 10 | 5 | 5 |
| `max_fish` / actual ceiling | 10 / 11 | 5 / 6 | 5 / 6 |
| World | walls | torus | walls |
| World width | 30 | 30 | 30 |
| Episode steps | 1000 | 3000 | 3000 |
| Initial starvation deadline | 5000 | 3000 | 800 |
| Survival extension per caught fish | 75 | 150 | 75 |
| Prey reproduction interval | 100 | 200 | 600 |
| Prey vision distance | 10 | 35 | 10 |
| Shark speed-change parameter | .06 | .05 | .06 |
| Shark nominal max speed (parameter/friction) | .6 | .5 | .6 |
| Shared catch reward | off | on | off |
| Simple split catch reward | off | on | off |
| Observed prey slots | 3 | 5 | 5 |
| PPO nominal steps | 2M | 21M | 21M |
| PPO learning rate | .0003 | .00009 | .00009 |
| PPO clipping | .2 | .1 | .1 |
| Hidden layers | 2 x 64 | 3 x 64 + batch norm | 3 x 64 + batch norm |
| Two-learner branch | off | on | on |

The previous example enables easier/faster replenishment and almost eliminates starvation during a 3000-step episode. Its torus removes wall trapping and its prey vision is increased; the author's blog describes slower predators and increased hunting difficulty as ways to induce group hunting. These mechanisms explain why settings were varied; no one change by itself establishes why our result differed.

## Why there are several defaults

`Aquarium.__init__` is a generic fallback (e.g. size 40, max_steps 500, max_fish 30, max_sharks 5). `Experiment` supplies resolved JSON values explicitly, so those constructor defaults do not describe the actual experiment. `Fish` and `Shark` class constants are also overwritten from JSON before environment construction.

`Config.get_cfg` starts with config `1`, overlays the selected `base_cfg`, then the experiment. This is one-level overlay, not recursive inheritance. New config keys fall back to `1` for backwards compatibility. The Figure 3 candidate uses `base_cfg=1`, so there is no deeper inheritance ambiguity.

The JSON base is an early generic/single-agent setup, not a universal paper setting. Later named families contain their own explicit environments and PPO hyperparameters. The paper's Table 1 describes the final PPO settings, while its text states 3000-step initial survival unless specified otherwise. The Figure 3 artifact name is evidence for the t800 exception; the caption does not provide an exact config ID or original seed. The candidate should therefore remain labelled 'closest traceable', not 'exact'.

## Implementation discrepancies retained and documented

* Reproduction permits `max_fish+1` because its block condition uses `>` rather than `>=`. The observed ceiling of six agrees with Figure 3. Changing it to five would reduce fidelity, so it is preserved.
* `ppo2_ma.learn` passes the same Keras hidden network into both Models. Runtime object-identity inspection finds 12 shared trainable variables out of 17 per model. The output/value heads and optimizers differ, but these are not fully independent neural networks. This qualifies the paper/blog description. We preserve the authors' implementation rather than silently create a different algorithm.
* Each update collects two separate 3000-slot rollouts. The nominal 21M axis means 7000 updates and 21M samples per learner: 42M total sampled transition slots. Actual environment advances may be lower after all sharks die because the wrapper returns terminal observations without advancing. The runner records both rollout index and simulated steps; do not conflate the nominal axis with unique world steps.
* Upstream `Experiment.train` does not forward the JSON PPO seed, so Baselines calls `set_global_seeds(None)`. The local runner forwards a recorded seed explicitly. This is a reproducibility change, not a reward/dynamics/optimizer change; original published seeds are unavailable.
* Set-order-based shark assignment and starvation handling remain upstream during training. Headless evaluation assigns each policy to a stable shark label during an episode and alternates labels across episodes. It advances the raw world to the horizon even after a predator dies; that is documented separately from training rollouts.
* The upstream helper `load_eval` defaults to 10000 steps and forces GUI; `evaluator_herding.py` instead requests 5000 steps in `ma3_obs`, which changes initial prey, population limit, and starvation. Other evaluator scripts use still different overrides. Those helpers are not a single exact paper protocol. The new runner explicitly evaluates the training condition for 3000 steps and records all settings.
* The 1–10-final-prey proxy is uninformative for a six-prey-ceiling world. We retain it for comparison but additionally require below-ceiling prey and both sharks alive, and report late catches and full prey trajectories. These diagnostics still do not prove deliberate waiting; behavioural video and policy interventions would be needed for that interpretation.

## Execution and provenance

`reproduce_figure3.py` runs a serial batch, keeping a separate manifest, full resolved config, training metrics, TensorBoard directory, and 20 evaluation episodes for each seed. No environment or PPO source is patched on disk. The exact source and Baselines commit hashes are recorded. A failed process stops the batch; incomplete runs are not silently resumed from random initialization. Complete training can be evaluated independently. Original models and data are retained.

```bash
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2
python reproduce_figure3.py batch --seeds 42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61
```

This matches the paper's count of 20 training replications x 20 evaluations for this candidate condition, not all scenarios/figures in the paper. Running serially limits simultaneous CPU use but can take days. CPU TensorFlow 2.3.4, Python 3.8.20 and source versions are pinned as in LOCAL_SETUP; original OS, compiler, exact package versions, original seed and hardware are not recoverable. Bitwise reproduction is not claimed.

## Validation and review

The end-to-end smoke run trains both learners, exports and reloads the final pair, evaluates two short episodes, checks provenance, and writes an across-run aggregate. A separate full 3000-step stationary-action test confirmed the raw evaluation path handles both sharks dying and an empty joint action dictionary. `git diff --check` passed.

The independent Codex review agreed on the candidate's uncertain provenance, shared hidden parameters, and two-rollout step accounting. Source/runtime hashes, run-identity checks, a global training lock, and safe early-termination metrics address its implementation findings. Across-run statistics are aggregated after each seed; confidence intervals use training runs as the sampling unit. Partial training is deliberately not automatically resumed: upstream SavedModel continuation does not provide a complete optimizer/RNG restart, so failed runs require an explicit new output directory. No false exact-restart claim is made. Causal cooperation and intentional restraint are not inferred from endpoint counts.
