"""
Hyperparameter sweep for CRSM training.
"""
import subprocess
import json
from pathlib import Path
import itertools


# Define hyperparameter grid
param_grid = {
    'lr': [0.001, 0.0005, 0.0001],
    'd_model': [128, 256],
    'num_layers': [2, 4],
    'batch_size': [16, 32],
    'dropout': [0.1, 0.2],
}

# Generate all combinations
keys = param_grid.keys()
values = param_grid.values()
experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]

print(f"Running {len(experiments)} experiments...")

results = []

for i, params in enumerate(experiments, 1):
    exp_name = f"sweep_{i:03d}"
    exp_dir = f"experiments/sweep/{exp_name}"
    
    print(f"\n{'='*70}")
    print(f"Experiment {i}/{len(experiments)}: {exp_name}")
    print(f"Parameters: {params}")
    print('='*70)
    
    # Run training
    cmd = [
        'python', '-m', 'crsm.cli', 'train',
        '--epochs', '20',
        '--batch-size', str(params['batch_size']),
        '--vocab-size', '1000',
        '--seq-len', '32',
        '--lr', str(params['lr']),
        '--data-dir', 'data/text_corpus',
        '--checkpoint-dir', exp_dir,
        '--no-value-loss',
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        # Parse final loss from output
        lines = result.stdout.split('\n')
        final_loss = None
        for line in reversed(lines):
            if 'loss=' in line:
                final_loss = float(line.split('loss=')[1].split()[0])
                break
        
        experiment_result = {
            'exp_name': exp_name,
            'params': params,
            'final_loss': final_loss,
            'success': result.returncode == 0,
        }
        
        results.append(experiment_result)
        
        print(f"✓ Completed: Loss = {final_loss:.4f}")
        
    except subprocess.TimeoutExpired:
        print("✗ Timeout")
        results.append({
            'exp_name': exp_name,
            'params': params,
            'final_loss': None,
            'success': False,
            'error': 'timeout'
        })
    except Exception as e:
        print(f"✗ Error: {e}")
        results.append({
            'exp_name': exp_name,
            'params': params,
            'final_loss': None,
            'success': False,
            'error': str(e)
        })

# Save results
output_file = 'experiments/sweep/results.json'
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("SWEEP COMPLETE")
print('='*70)

# Find best configuration
successful = [r for r in results if r['success'] and r['final_loss'] is not None]
if successful:
    best = min(successful, key=lambda x: x['final_loss'])
    print(f"\nBest configuration:")
    print(f"  Loss: {best['final_loss']:.4f}")
    print(f"  Params: {json.dumps(best['params'], indent=4)}")
    print(f"  Directory: experiments/sweep/{best['exp_name']}")
else:
    print("\n⚠ No successful experiments")

print(f"\nResults saved to: {output_file}")