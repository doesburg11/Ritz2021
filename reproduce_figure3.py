"""Closest traceable Figure 3 configuration; preserves upstream learning semantics."""
import argparse
import hashlib
import platform
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
CFG = 'ma3_obs_starve_maxsteps_t800_5fish_two_net'


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def source_hashes():
    files = ['reproduce_figure3.py', 'pipeline.pyx', 'config.pyx',
             'requirements-local.lock.txt', 'simulations.json',
             'env/aquarium.pyx', 'env/animal_controller.pyx',
             'vendor/baselines/baselines/ppo2/ppo2_ma.py']
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files}


def aggregate(base):
    import numpy as np
    from scipy.stats import t
    results=[]
    for path in sorted(base.glob('seed-*/evaluation.json')):
        data=json.loads(path.read_text())
        rows=data['episodes']
        if not rows:
            continue
        results.append(dict(seed=data['training_seed'], episodes=len(rows),
            below_ceiling_with_two_sharks=float(np.mean([r['below_ceiling_with_two_sharks'] for r in rows])),
            mean_catches=float(np.mean([r['catches'] for r in rows])),
            mean_final_prey=float(np.mean([r['final_prey'] for r in rows])),
            below_ceiling_with_late_hunting=float(np.mean([r['below_ceiling_with_two_sharks'] and r['late_catches']>0 for r in rows]))))
    summary={'completed_training_runs': len(results), 'per_training_run': results, 'metrics': {}}
    for key in ['below_ceiling_with_two_sharks', 'below_ceiling_with_late_hunting', 'mean_catches', 'mean_final_prey']:
        values=np.array([r[key] for r in results])
        if not len(values):
            continue
        mean=float(np.mean(values))
        ci=None
        if len(values)>1:
            half=float(t.ppf(.975,len(values)-1)*np.std(values,ddof=1)/np.sqrt(len(values)))
            ci=[mean-half,mean+half]
        summary['metrics'][key]={'mean_across_training_runs':mean,'t_interval_95':ci}
    write_json(base/'aggregate.json',summary)


def experiment(seed, smoke):
    import numpy as np
    import tensorflow as tf
    from pipeline import Experiment
    random.seed(seed)
    tf.random.set_seed(seed)
    exp = Experiment(CFG, show_gui=False, dump_cfg=False)
    # Aquarium.__init__ resets NumPy to its JSON seed; explicitly set the
    # replication seed afterwards. No optimizer or environment logic is changed.
    np.random.seed(seed)
    exp.cfg['aquarium']['seed'] = seed
    exp.cfg['ppo']['seed'] = seed
    if smoke:
        exp.env.env.max_steps = 64
        exp.cfg['aquarium']['max_steps'] = 64
        exp.cfg['ppo']['total_timesteps'] = 128
    return exp


def train(seed, folder, smoke):
    exp = experiment(seed, smoke)
    manifest = folder / 'manifest.json'
    if manifest.exists():
        raise RuntimeError('Run already exists; use a new output directory or evaluate it.')
    import tensorflow as tf
    metadata = dict(platform=platform.platform(), python=sys.version, tensorflow=tf.__version__,
                    source_hashes=source_hashes(),
                    git_status=subprocess.check_output(['git','status','--short','--untracked-files=no'],text=True),
                    seed=seed, config_id=CFG, resolved_config=exp.cfg,
                    smoke=smoke, algorithm='unmodified upstream two_nets (shared hidden network)',
                    steps_note='21M nominal samples per learner; 42M aggregate transition slots; actual simulator advances can be lower after deaths',
                    source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    baselines_commit=subprocess.check_output(['git','-C','vendor/baselines','rev-parse','HEAD'],text=True).strip(),
                    config_sha256=hashlib.sha256((ROOT/'simulations.json').read_bytes()).hexdigest())
    write_json(manifest, metadata)
    previous = set((ROOT/'models').glob(CFG+'*-F-m1'))
    original_callback = exp.after_epoch_cb
    def callback(epoch):
        env = exp.env.env
        row = dict(update=epoch, prey=len(env.fishes), sharks=len(env.sharks),
                   catches=int(env.dead_fishes), simulated_steps=int(env.current_step))
        with (folder/'training-episodes.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
        original_callback(epoch)
    exp.after_epoch_cb = callback
    exp.runs_folder = str(folder/'tensorboard')
    # Upstream Experiment.train omits the PPO seed, so Baselines otherwise
    # reseeds every RNG from entropy. Forward it explicitly for recorded runs.
    import pipeline
    original_learn = pipeline.ppo2_ma.learn
    def seeded_learn(**kwargs):
        return original_learn(seed=seed, **kwargs)
    pipeline.ppo2_ma.learn = seeded_learn
    try:
        exp.train()
    finally:
        pipeline.ppo2_ma.learn = original_learn
    created = set((ROOT/'models').glob(CFG+'*-F-m1')) - previous
    if len(created) != 1:
        raise RuntimeError('Expected exactly one new checkpoint pair; refusing ambiguous selection.')
    prefix = str(created.pop())[:-3]
    for suffix in ['-m1', '-m2']:
        assert Path(prefix+suffix+'/saved_model.pb').is_file()
    metadata.update(checkpoint_prefix=prefix, training_complete=True)
    write_json(manifest, metadata)


def evaluate(seed, folder, smoke):
    import numpy as np
    import tensorflow as tf
    metadata=json.loads((folder/'manifest.json').read_text())
    if not metadata.get('training_complete') or metadata['smoke'] != smoke:
        raise RuntimeError('Missing completed matching training run')
    if metadata['seed'] != seed or metadata['config_id'] != CFG or metadata['source_hashes'] != source_hashes():
        raise RuntimeError('Evaluation source/seed does not match the training manifest')
    exp=experiment(seed, smoke)
    env=exp.env.env
    models=[tf.saved_model.load(metadata['checkpoint_prefix']+suffix) for suffix in ['-m1','-m2']]
    rows=[]
    for episode in range(2 if smoke else 20):
        eval_seed=100000+seed*100+episode
        np.random.seed(eval_seed)
        tf.random.set_seed(eval_seed)
        obs=env.reset()
        assignment={name:(i+episode%2)%2 for i,name in enumerate(sorted(obs,key=lambda n:n()))}
        population=[len(env.fishes)]
        catches_at_half=None
        while not env.is_finished:
            actions={}
            for name,observation in obs.items():
                action=models[assignment[name]].step(tf.convert_to_tensor(observation.reshape(1,-1),tf.float32))[0].numpy()[0]
                if not np.isfinite(action).all():
                    raise RuntimeError('Nonfinite policy action')
                actions[name]=(float(action[0]),float(action[1]),False)
            obs,_,_=env.step(actions)
            population.append(len(env.fishes))
            if env.current_step==env.max_steps//2:
                catches_at_half=int(env.dead_fishes)
        row=dict(seed=eval_seed, steps=int(env.current_step), final_prey=len(env.fishes),
                 final_sharks=len(env.sharks), catches=int(env.dead_fishes),
                 late_catches=int(env.dead_fishes)-catches_at_half if catches_at_half is not None else 0,
                 # Report the paper proxy separately from a non-ceiling diagnostic.
                 paper_endpoint_proxy=1<=len(env.fishes)<=10,
                 below_ceiling_with_two_sharks=1<=len(env.fishes)<env.max_fish+1 and len(env.sharks)==2,
                 population=population)
        rows.append(row)
        print('evaluation',seed,episode+1,row['catches'],row['final_prey'],row['final_sharks'],flush=True)
    write_json(folder/'evaluation.json',dict(training_seed=seed,config_id=CFG,smoke=smoke,episodes=rows,
        notes='Original training condition; endpoint proxies do not prove deliberate restraint.'))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode',choices=['train','evaluate','batch'])
    p.add_argument('--seeds',default='42')
    p.add_argument('--output',default='local-artifacts/figure3-reproduction')
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    os.chdir(ROOT)
    base=Path(args.output).resolve()
    seeds=[int(s) for s in args.seeds.split(',')]
    if len(seeds)!=len(set(seeds)) or any(s<0 or s>=1000000 for s in seeds):
        p.error('Use unique seeds between zero and 999999')
    for seed in seeds:
        folder=base/('seed-'+str(seed))
        folder.mkdir(parents=True,exist_ok=True)
        if args.mode=='batch':
            manifest=folder/'manifest.json'
            done=json.loads(manifest.read_text()).get('training_complete',False) if manifest.exists() else False
            for mode in (['evaluate'] if done else ['train','evaluate']):
                if mode=='evaluate' and (folder/'evaluation.json').exists():
                    existing=json.loads((folder/'evaluation.json').read_text())
                    if existing['training_seed'] != seed or existing['smoke'] != args.smoke or existing['config_id'] != CFG:
                        raise RuntimeError('Existing evaluation does not match requested run')
                    continue
                cmd=[sys.executable,str(Path(__file__).resolve()),mode,'--seeds',str(seed),'--output',str(base)]
                if args.smoke: cmd.append('--smoke')
                with (folder/(mode+'.log')).open('a') as log:
                    subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
            aggregate(base)
        elif args.mode=='train':
            with (ROOT/'local-artifacts'/'figure3-training.lock').open('w') as lock:
                fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
                train(seed,folder,args.smoke)
        else:
            evaluate(seed,folder,args.smoke)


if __name__=='__main__':
    main()
