import os
import sys
import argparse
import subprocess
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# filepath: /home/ach18423hu/Evolutionary-Reinforcement-Learning/experiments/compare_humanoid.py
def run_main(cmd_args, cwd, env=None):
    envd = os.environ.copy()
    if env:
        envd.update(env)
    print("Running:", " ".join(cmd_args))
    subprocess.run(cmd_args, check=True, cwd=cwd, env=envd)

def load_score_csv(savetag):
    path = os.path.join('Results', 'Plots', 'score_' + savetag + '.csv')
    if not os.path.exists(path):
        print("Warning: csv not found:", path)
        return None, None
    try:
        data = np.loadtxt(path, delimiter=',')
        if data.ndim == 1 and data.size == 0:
            return None, None
        # If single row -> ensure shape (N,2)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        x = data[:, 0]
        y = data[:, 1]
        return x, y
    except Exception as e:
        print("Failed to read", path, e)
        return None, None

def interp_on_grid(x, y, grid):
    if x is None or y is None or len(x) == 0:
        return np.full_like(grid, np.nan, dtype=np.float64)
    # ensure monotonic x for interp
    idx = np.argsort(x)
    x = x[idx]; y = y[idx]
    # clip domain
    y_interp = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return y_interp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Humanoid-v2')
    parser.add_argument('--total_steps', type=float, default=0.02, help='in millions (same semantics as main.py)')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--python', type=str, default=sys.executable)
    parser.add_argument('--project_root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    parser.add_argument('--quick', action='store_true', help='set smaller default steps for fast debug')
    args = parser.parse_args()

    project_root = args.project_root
    py = args.python
    env_name = args.env
    repeats = args.repeats
    gpu_id = args.gpu_id
    total_steps = args.total_steps  # in millions, main.py multiplies by 1e6

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

    # Configurations to evaluate
    configs = [
        {'name': 'ERL', 'popsize': 10, 'rollsize': 5, 'gradperstep': 1.0, 'learning_start': 5000},
        {'name': 'SAC-only', 'popsize': 1, 'rollsize': 5, 'gradperstep': 1.0, 'learning_start': 5000},
        {'name': 'Evo-only', 'popsize': 10, 'rollsize': 0, 'gradperstep': 0.0, 'learning_start': 999999999},
    ]

    saved_tags = {c['name']: [] for c in configs}

    # Run experiments
    for cfg in configs:
        for seed in range(repeats):
            savetag = f"{cfg['name']}_{env_name}_seed{seed}_{timestamp}"
            saved_tags[cfg['name']].append(savetag)

            cmd = [
                py, 'main.py',
                '--env', env_name,
                '--total_steps', str(total_steps),
                '--popsize', str(cfg['popsize']),
                '--rollsize', str(cfg['rollsize']),
                '--gradperstep', str(cfg['gradperstep']),
                '--learning_start', str(cfg['learning_start']),
                '--savetag', savetag,
                '--seed', str(1000 + seed),
                '--gpu_id', str(gpu_id)
            ]
            # run main.py in project root
            try:
                run_main(cmd, cwd=project_root, env={'CUDA_VISIBLE_DEVICES': str(gpu_id)})
            except subprocess.CalledProcessError as e:
                print("Run failed for", cfg['name'], "seed", seed, "error:", e)
                # continue to next run

    # Plotting: build common grid (frames)
    # main.py uses total_steps * 1e6 as frame_limit
    frame_limit = int(total_steps * 1e6)
    if frame_limit <= 0:
        frame_limit = int(1e4)  # fallback for tiny debug runs
    grid = np.linspace(0, frame_limit, 400)

    plt.figure(figsize=(10,6))
    for cfg in configs:
        ys = []
        for tag in saved_tags[cfg['name']]:
            x, y = load_score_csv(tag)
            if x is None or y is None:
                continue
            yi = interp_on_grid(x, y, grid)
            ys.append(yi)
        if len(ys) == 0:
            print("No valid runs for", cfg['name'])
            continue
        ys = np.vstack(ys)  # shape (runs, len(grid))
        mean = np.nanmean(ys, axis=0)
        std = np.nanstd(ys, axis=0)
        plt.plot(grid, mean, label=cfg['name'])
        plt.fill_between(grid, mean-std, mean+std, alpha=0.2)
    plt.xlabel('Frames')
    plt.ylabel('Test score (mean ± 1std)')
    plt.title(f'ERL vs SAC vs NeuroEvolution on {env_name}')
    plt.legend()
    out_dir = os.path.join(project_root, 'Results', 'Plots')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'compare_humanoid_{env_name}_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(out_path)
    print("Saved figure to", out_path)

if __name__ == "__main__":
    main()