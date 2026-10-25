"""Evaluate one trained policy pair; does not reproduce across-training-seed results."""
import json
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from pipeline import Experiment

CFG = 'ma9_i5_p150_r10_s05_sp200_two_net_vd35_f'
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'local-artifacts' / 'final-evaluation'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = str(ROOT / 'models' / (CFG + '-HBP-26.10.21-21:18:58-59543360-model-F'))
    models = [tf.saved_model.load(prefix + '-m1'), tf.saved_model.load(prefix + '-m2')]
    exp = Experiment(CFG, show_gui=False, dump_cfg=False)
    env = exp.env.env
    episode_file = OUT / 'episodes.jsonl'
    rows = [json.loads(line) for line in episode_file.read_text().splitlines()] if episode_file.exists() else []
    completed = {(r['mode'], r['cap'], r['seed']) for r in rows}
    for mode, cap in [('trained', 5), ('trained', 10), ('random', 10)]:
        for episode in range(20):
            seed = 10000 + episode
            if (mode, cap, seed) in completed:
                continue
            action_rng = np.random.RandomState(seed + 100000)
            np.random.seed(seed)
            random.seed(seed)
            tf.random.set_seed(seed)
            env.max_fish = cap
            obs = env.reset()
            # Fix each policy to a shark for the entire episode. Alternate the
            # assignment to avoid making results depend on set iteration order.
            names = sorted(obs, key=lambda name: name())
            assignment = {name: (i + episode % 2) % 2 for i, name in enumerate(names)}
            population = [len(env.fishes)]
            total_reward = 0.0
            while not env.is_finished:
                actions = {}
                for name, observation in obs.items():
                    if mode == 'random':
                        action = action_rng.uniform(-1, 1, size=2)
                    else:
                        tensor = tf.convert_to_tensor(observation.reshape(1, -1), tf.float32)
                        action = models[assignment[name]].step(tensor)[0].numpy()[0]
                    assert np.isfinite(action).all()
                    actions[name] = (float(action[0]), float(action[1]), False)
                obs, rewards, _ = env.step(actions)
                total_reward += float(sum(rewards.values()))
                population.append(len(env.fishes))
            catches = int(env.dead_fishes)
            row = dict(mode=mode, cap=cap, seed=seed, policy_assignment={name(): index for name, index in assignment.items()},
                       steps=int(env.current_step), catches=catches,
                       final_prey=len(env.fishes), final_sharks=len(env.sharks),
                       dead_sharks=int(env.dead_sharks), coop_catches=int(env.coop_kills),
                       total_reward=total_reward,
                       sustainability=catches + len(env.fishes),
                       end_population_proxy=1 <= len(env.fishes) <= 10,
                       prey_survive_and_caught=len(env.fishes) > 0 and catches > 0,
                       population=population)
            rows.append(row)
            with (OUT / 'episodes.jsonl').open('a' if len(rows) > 1 else 'w') as f:
                f.write(json.dumps(row) + '\n')
            print(mode, cap, episode + 1, 'catches', catches, 'prey', len(env.fishes), flush=True)
    summary = {'checkpoint_prefix': prefix, 'episodes_per_condition': 20,
               'episode_steps': 3000, 'initial_prey': 5, 'random_action_rng': 'separate NumPy RandomState', 'prey_limit_note': 'Upstream reproduction permits max_fish + 1 prey; diagnostic max_fish=10 gives a limit of 11', 'policy_actions': 'stochastic SavedModel.step',
               'conditions': []}
    for mode, cap in [('trained', 5), ('trained', 10), ('random', 10)]:
        group = [r for r in rows if r['mode'] == mode and r['cap'] == cap]
        n = len(group)
        catches = sum(r['catches'] for r in group)
        summary['conditions'].append(dict(
            mode=mode, cap=cap, episodes=n,
            mean_catches=float(np.mean([r['catches'] for r in group])),
            mean_final_prey=float(np.mean([r['final_prey'] for r in group])),
            prey_extinction_episodes=sum(r['final_prey'] == 0 for r in group),
            capped_prey_episodes=sum(r['final_prey'] >= cap + 1 for r in group),
            end_population_proxy_rate=sum(r['end_population_proxy'] for r in group) / n,
            prey_survive_and_caught_rate=sum(r['prey_survive_and_caught'] for r in group) / n,
            mean_sustainability=float(np.mean([r['sustainability'] for r in group])),
            shark_death_episodes=sum(r['dead_sharks'] > 0 for r in group),
            pooled_cooperative_catch_ratio=sum(r['coop_catches'] for r in group) / catches if catches else None))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
