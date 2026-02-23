"""Reproducible run harness for single-node and DDP experiments.

This small helper centralizes config, seeding, log directory setup, and launching the training CLI.
"""
import os
import json
from datetime import datetime

from .train import main as train_main


def make_run_dir(base: str = 'runs', name: str | None = None):
    name = name or datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(base, name)
    os.makedirs(path, exist_ok=True)
    return path


def run_experiment(config: dict, run_dir: str | None = None):
    run_dir = run_dir or make_run_dir()
    # persist config
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    # forward to train.main
    train_main(**config)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=str, default=None)
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    cfg = json.loads(open(args.config, 'r', encoding='utf-8').read())
    run_experiment(cfg, run_dir=args.run_dir)
