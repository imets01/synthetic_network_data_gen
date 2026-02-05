"""Quick tuning script for testing - runs only 2 trials with fast settings"""
import subprocess
import os
import sys
import optuna
from copy import deepcopy
import shutil
import argparse
from pathlib import Path

# Add lib path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib

parser = argparse.ArgumentParser()
parser.add_argument('ds_name', type=str)
parser.add_argument('train_size', type=int)
parser.add_argument('eval_type', type=str)
parser.add_argument('eval_model', type=str)
parser.add_argument('prefix', type=str)
parser.add_argument('--eval_seeds', action='store_true', default=False)
parser.add_argument('--n_trials', type=int, default=2)  # Quick: only 2 trials

args = parser.parse_args()
train_size = args.train_size
ds_name = args.ds_name
eval_type = args.eval_type
assert eval_type in ('merged', 'synthetic')
prefix = str(args.prefix)
n_trials = args.n_trials

pipeline = 'scripts/pipeline.py'
base_config_path = f'exp/{ds_name}/config.toml'
parent_path = Path(f'exp/{ds_name}/')
exps_path = Path(f'exp/{ds_name}/many-exps/')
eval_seeds_script = 'scripts/eval_seeds.py'

os.makedirs(exps_path, exist_ok=True)

def _suggest_mlp_layers(trial):
    # Simplified: just use small fixed layers for quick testing
    return [128, 128]

def objective(trial):
    # Quick settings - much faster than original
    lr = trial.suggest_float('lr', 0.001, 0.002, log=True)
    d_layers = _suggest_mlp_layers(trial)
    weight_decay = 0.0
    batch_size = 256  # Fixed for speed
    steps = 200  # Very few steps for quick test
    gaussian_loss_type = 'mse'
    num_timesteps = 50  # Few timesteps for speed
    num_samples = min(train_size, 100)  # Small sample count

    base_config = lib.load_config(base_config_path)

    base_config['train']['main']['lr'] = lr
    base_config['train']['main']['steps'] = steps
    base_config['train']['main']['batch_size'] = batch_size
    base_config['train']['main']['weight_decay'] = weight_decay
    base_config['model_params']['rtdl_params']['d_layers'] = d_layers
    base_config['eval']['type']['eval_type'] = eval_type
    base_config['sample']['num_samples'] = num_samples
    base_config['diffusion_params']['gaussian_loss_type'] = gaussian_loss_type
    base_config['diffusion_params']['num_timesteps'] = num_timesteps

    base_config['parent_dir'] = str(exps_path / f"{trial.number}")
    base_config['eval']['type']['eval_model'] = args.eval_model
    if args.eval_model == "mlp":
        base_config['eval']['T']['normalization'] = "quantile"
        base_config['eval']['T']['cat_encoding'] = "one-hot"

    trial.set_user_attr("config", base_config)

    lib.dump_config(base_config, exps_path / 'config.toml')

    subprocess.run([sys.executable, f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--train', '--change_val'], check=True)

    # Only 1 dataset for quick eval (instead of 5)
    n_datasets = 1
    score = 0.0

    for sample_seed in range(n_datasets):
        base_config['sample']['seed'] = sample_seed
        lib.dump_config(base_config, exps_path / 'config.toml')
        
        subprocess.run([sys.executable, f'{pipeline}', '--config', f'{exps_path / "config.toml"}', '--sample', '--eval', '--change_val'], check=True)

        report_path = str(Path(base_config['parent_dir']) / f'results_{args.eval_model}.json')
        report = lib.load_json(report_path)

        if 'r2' in report['metrics']['val']:
            score += report['metrics']['val']['r2']
        else:
            score += report['metrics']['val']['macro avg']['f1-score']

    shutil.rmtree(exps_path / f"{trial.number}")

    return score / n_datasets

study = optuna.create_study(
    direction='maximize',
    sampler=optuna.samplers.TPESampler(seed=0),
)

print(f"Running quick tuning with {n_trials} trials...")
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

best_config_path = parent_path / f'{prefix}_best/config.toml'
best_config = study.best_trial.user_attrs['config']
best_config["parent_dir"] = str(parent_path / f'{prefix}_best/')

os.makedirs(parent_path / f'{prefix}_best', exist_ok=True)
lib.dump_config(best_config, best_config_path)
lib.dump_json(optuna.importance.get_param_importances(study), parent_path / f'{prefix}_best/importance.json')

print(f"Best trial: {study.best_trial.number}, Score: {study.best_trial.value:.4f}")

subprocess.run([sys.executable, f'{pipeline}', '--config', f'{best_config_path}', '--train', '--sample'], check=True)

if args.eval_seeds:
    subprocess.run([sys.executable, f'{eval_seeds_script}', '--config', f'{best_config_path}', '3', "ddpm", eval_type, args.eval_model, '2'], check=True)

print(f"Quick tuning complete! Best model at: {parent_path / f'{prefix}_best'}")
