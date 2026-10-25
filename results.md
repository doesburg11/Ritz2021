# Local reproduction results

Recorded 25 October 2026. All 20 training runs and their 400 evaluation episodes completed. **This reconstruction did not demonstrate sustainable herding.** Both predators were dead at the end of every evaluation; surviving fish alone are not evidence of a sustainable ecosystem.

## Protocol

- Configuration: `ma3_obs_starve_maxsteps_t800_5fish_two_net`.
- Training seeds: 42–61; 20 separate training runs, executed serially.
- Training budget: 21M nominal sampled slots per learner (7000 updates); two sequential rollouts give 42M aggregate sampled slots. Actual simulator advances can be lower after predator death.
- Evaluation: 20 held-out stochastic-policy episodes per trained pair, up to 3000 environment steps each. Both policies remain assigned to stable shark labels during an episode; labels are alternated across episodes.
- Environment: two initial predators, five initial prey, effective prey ceiling six, walls, starvation deadline 800, survival extension 75 per catch, reproduction interval 600, prey vision 10, shark speed-change parameter .06, no shared catch reward.
- No new training or evaluation was launched while recording this report.

## Aggregate results

| Measure | Result |
|---|---:|
| Completed training runs | 20 |
| Completed evaluation episodes | 400 |
| Mean catches per episode | 7.1825 |
| Mean final prey | 1.315 |
| Both predators alive at end | 0 / 400 |
| One predator alive at end | 0 / 400 |
| Below-ceiling prey with both predators alive | 0 / 400 |
| Above condition with catches in the second half | 0 / 400 |
| Any catches in the second half of the 3000-step horizon | 0 / 400 |
| Prey extinct at end | 63 / 400 (15.75%) |
| Full-horizon evaluations | 337 / 400 |
| Early termination with both populations extinct | 63 / 400 |

Mean catches: 95% Student-t interval across the 20 training-run means is **[6.8644, 7.5006]**.

Mean final prey: 95% Student-t interval across the 20 training-run means is **[0.9789, 1.6511]**.

Evaluation episodes from the same trained pair are grouped within their training run for these intervals. The generated aggregate contains degenerate [0, 0] t intervals for all-zero diagnostics; these must not be interpreted as proof that the true success probability is exactly zero.

## Per-training-seed results

| Training seed | Evaluations | Mean catches | Mean final prey | Below-ceiling prey with two living predators |
|---|---:|---:|---:|---:|
| 42 | 20 | 7.75 | 2.20 | 0% |
| 43 | 20 | 7.00 | 0.90 | 0% |
| 44 | 20 | 6.15 | 0.45 | 0% |
| 45 | 20 | 8.35 | 2.25 | 0% |
| 46 | 20 | 6.65 | 0.85 | 0% |
| 47 | 20 | 7.20 | 1.05 | 0% |
| 48 | 20 | 7.00 | 0.90 | 0% |
| 49 | 20 | 7.00 | 1.30 | 0% |
| 50 | 20 | 8.05 | 2.55 | 0% |
| 51 | 20 | 6.95 | 1.00 | 0% |
| 52 | 20 | 7.10 | 1.00 | 0% |
| 53 | 20 | 6.00 | 0.10 | 0% |
| 54 | 20 | 7.30 | 0.80 | 0% |
| 55 | 20 | 6.60 | 2.40 | 0% |
| 56 | 20 | 8.40 | 2.25 | 0% |
| 57 | 20 | 7.15 | 0.95 | 0% |
| 58 | 20 | 8.30 | 2.25 | 0% |
| 59 | 20 | 6.95 | 1.05 | 0% |
| 60 | 20 | 6.65 | 1.00 | 0% |
| 61 | 20 | 7.10 | 1.05 | 0% |

## Interpretation and limits

The policies caught prey, but predator survival and continued hunting were not maintained. All predators were dead at the endpoint, no evaluation recorded second-half catches, and 63 episodes ended with both populations extinct. In the remaining episodes, fish could continue reproducing after the predators died. Consequently, the mean of 1.315 final fish does not reproduce a living pair maintaining a herd of roughly two fish.

This is the closest traceable configuration, not a verified exact Figure 3 reproduction. The plot source reads `t800_5fish_last_fish_pop.pickle`, and the two-learner interpretation is supported by the author's blog and rollout counts. The original figure's exact model, seed, runtime dependencies and full provenance are unavailable. The repository's archived series itself shows the reported three-stage pattern; this batch's different outcome does not invalidate that artifact or refute the paper.

The upstream two-learner branch shares a hidden network: runtime inspection found 12 shared trainable variables out of 17 per learner. We retained this behavior and the off-by-one prey ceiling to reproduce released code rather than silently change the algorithm. Explicit seed forwarding fixes the upstream omission of the configured PPO seed. Set iteration and TensorFlow behavior prevent a promise of bitwise replay.

The paper's 1–10-final-prey proxy is uninformative with a six-prey ceiling, and proximity-based catch counters cannot establish causal cooperation. Our additional living-predator and late-hunting diagnostics are explicitly local diagnostics, not a claim to reproduce every paper metric. No behavioural video or causal intervention was performed.

## Earlier README-example run

A separate earlier run used `ma9_i5_p150_r10_s05_sp200_two_net_vd35_f`, a later cooperation configuration with different boundaries, starvation, vision, reproduction and rewards. Its 20 original-setting evaluations averaged 19.45 catches and six surviving fish. A larger-population diagnostic averaged 32.6 catches versus 0.65 for random actions, but all ended at eleven prey. It demonstrated hunting skill without establishing herding. This earlier run is separate from the 20-run reconstruction reported above.

## Versioned results

Compact outputs are included in [reproduction_results/](reproduction_results/README.md): [aggregate](reproduction_results/aggregate.json), [all 400 evaluation records](reproduction_results/evaluations.json), and [training manifests](reproduction_results/manifests.json). Full trajectories and large training artifacts remain local.

## Evidence and commands

- [Reproduction and default audit](FIGURE3_REPRODUCTION.md)
- [Reproduction runner](reproduce_figure3.py)
- [Batch aggregate](local-artifacts/figure3-reproduction/aggregate.json)
- [Batch process record](local-artifacts/figure3-reproduction/batch-process.json)
- Per-seed manifests, training logs and `evaluation.json`: `local-artifacts/figure3-reproduction/seed-42/` through `seed-61/`.
- [Earlier-run report](local-artifacts/final-evaluation/RESULTS.md)

Generated artifacts are local and ignored by Git; the committed documentation can be read without them, but these links require the local output directory. Source/runtime hashes and resolved configurations are recorded in each manifest. Independent Codex CLI review findings and our assessment are documented in the reproduction audit. This update changes documentation only.
