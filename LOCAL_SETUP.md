# Local installation and reproduction

Separate checkout: `/home/doesburg/Projects/Ritz2021`.
Upstream: https://github.com/instance01/fish-rl-alife
Upstream commit: `0060131e897ead1d823ecc2d8b88a35f382f7345`.
Required Baselines fork: https://github.com/Instance-contrib/baselines/tree/tf2-instance
Baselines commit: `f24e100fd2afb79f046a38b0a1ce633335419921`.

## Environment

Python 3.8.20 is installed locally in `.python/`; `.venv/` is isolated from other projects.
Use the pinned `requirements-local.lock.txt` with Python 3.8. Build requires `g++`.
The upstream requirements are unpinned and use a legacy `git://` dependency URL; use HTTPS for the fork.

```bash
cd /home/doesburg/Projects/Ritz2021
source .venv/bin/activate
python -m pip install -r requirements-local.lock.txt
python -m pip install --no-deps --no-build-isolation -e vendor/baselines
python setup.py build_ext --inplace
```

For a fresh checkout, first create the Python 3.8 environment and clone the fork into
`vendor/baselines`, checking out the exact commit above. The lock records PyPI dependencies;
the fork is installed separately because its setup checks that TensorFlow is already installed.

## Short installation checks

Run from the repository root. CPU mode avoids requiring the CUDA libraries expected by this old TensorFlow version.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 python main.py local_smoke_shared single
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 python main.py local_smoke_independent single
```

Each configuration runs two 64-step PPO updates. These exercise environment stepping,
policy optimization, TensorBoard logging and final SavedModel export. They are installation
checks, with deliberately shortened episodes; they do not demonstrate herding or cooperation.
The existing upstream configurations are unchanged.

## Full upstream experiment

```bash
CUDA_VISIBLE_DEVICES=-1 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 python main.py ma9_i5_p150_r10_s05_sp200_two_net_vd35_f single
python -m tensorboard.main --logdir runs
```

The upstream configuration uses 21 million steps, 3000-step episodes, two independent
networks, and learning rate 9e-5. Models go in `models/`; TensorBoard data goes in `runs/`.
The paper uses multiple scenarios and repeated evaluation: completing one training run
is not sufficient to reproduce all figures. No full paper-length training is started by installation.
Upstream does not pass its configured PPO seed to the learner (see `Experiment.train`),
so the seed in JSON alone does not guarantee reproducible training.

To watch a two-network checkpoint on a graphical desktop, omit the `-m1`/`-m2` suffix:

```bash
python main.py ma9_i5_p150_r10_s05_sp200_two_net_vd35_f load models/REPLACE_WITH_MODEL_PREFIX-F
```

The supplied local PDF was checked: *A Sustainable Ecosystem through Emergent Cooperation
in Multi-Agent Reinforcement Learning*, ALIFE 2021. The repository README links the publication.

## Verification and independent review

Cython build and `pip check` passed. Both smoke commands exited zero and completed
two PPO updates. The resulting shared checkpoint and both independent checkpoints
were exported. Verification logs are retained in ignored `local-artifacts/`.

Codex CLI independently reviewed the configurations using `gpt-5.6-sol`. It confirmed
that all original experiment entries are unchanged. It flagged a missing lock file
while installation was still underway; the lock file is now included. Its claimed
independent-run division-by-zero is contradicted by successful execution and by
`vendor/baselines/baselines/ppo2/runner.py:58`, which logs rewards on every rollout
and populates `tot_rew_queue`. No algorithm change was needed. It recommended
automated smoke regression coverage; installation was checked with the actual
training commands and checkpoint reload/inference instead.

## Evaluate the completed 21M-step run

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 OMP_NUM_THREADS=2 python evaluate_final_local.py
```

The evaluator pins the completed `59543360` run and uses its stochastic SavedModel
actions. It runs 20 episodes in each of three conditions: the trained pair in the
original environment (`max_fish=5`), the trained pair with `max_fish=10`, and random
actions with `max_fish=10`. All start with five fish and last 3000 steps. Upstream
allows one fish above `max_fish`, giving effective limits six and eleven. Policies
are assigned consistently within episodes and swapped between shark labels across
episodes. Completed episodes are resumed from `local-artifacts/final-evaluation/`;
move that directory aside to run a fresh sample. Seeds control initialisation and
random streams but do not imply bitwise reproduction across resumed processes.

Outputs include episode records with full population trajectories and `summary.json`.
The paper's sustainability score is catches plus final prey; its endpoint herding
proxy uses one through ten remaining prey. Neither proves deliberate restraint.
The latter is uninformative in the original six-prey-limit setting. The cooperative
catch counter measures partner proximity at capture, not causal cooperation.
Starvation is largely excluded by the configured 3000-step initial survival time.
These are evaluations of one trained pair, not independent training replications.

The independent Codex review identified the upstream off-by-one prey limit, RNG
coupling in the original random baseline, and checkpoint selection by modification
time. The evaluator accounts for the limit, separates baseline action randomness,
and pins the checkpoint. The review also highlighted inference limits for herding
and cooperation, which are reflected above.
