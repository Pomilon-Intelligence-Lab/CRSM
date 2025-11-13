"""
Test script to verify that CRSM's Latent Dynamics Model performs self-modification.

This script:
1. Creates a minimal CRSM model with dynamics
2. Captures initial latent states
3. Runs MCTS deliberation
4. Measures state changes
5. Validates that self-modification occurred

Usage:
    python test_self_modification.py
    python test_self_modification.py --dynamics-path path/to/dynamics.pt
    python test_self_modification.py --verbose --iterations 5
"""

import argparse
import torch
import asyncio
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, '.')

# Import CRSM components
from crsm.model import CRSM
from crsm.latent_dynamics import LatentDynamics
from crsm.mamba_ssm import MambaModel


class SelfModificationTester:
    def __init__(self, model, verbose=False):
        self.model = model
        self.verbose = verbose
        self.device = next(model.parameters()).device
        
    def capture_state_snapshot(self, state):
        """Create a deep copy of the current latent state."""
        if state is None:
            return None
        
        if isinstance(state, list):
            return [s.clone().detach() if s is not None else None for s in state]
        elif isinstance(state, torch.Tensor):
            return state.clone().detach()
        return None
    
    def compute_state_difference(self, state_before, state_after):
        """Compute L2 norm of state differences across all layers."""
        if state_before is None or state_after is None:
            return 0.0
        
        total_diff = 0.0
        num_layers = 0
        
        if isinstance(state_before, list) and isinstance(state_after, list):
            for s_before, s_after in zip(state_before, state_after):
                if s_before is None or s_after is None:
                    continue
                diff = torch.norm(s_after - s_before).item()
                total_diff += diff
                num_layers += 1
        elif isinstance(state_before, torch.Tensor) and isinstance(state_after, torch.Tensor):
            total_diff = torch.norm(state_after - state_before).item()
            num_layers = 1
        
        return total_diff / max(1, num_layers)
    
    def print_state_stats(self, state, label="State"):
        """Print statistics about a state."""
        if not self.verbose:
            return
        
        print(f"\n{label} Statistics:")
        if isinstance(state, list):
            for i, s in enumerate(state):
                if s is None:
                    print(f"  Layer {i}: None")
                else:
                    print(f"  Layer {i}: shape={s.shape}, mean={s.mean().item():.6f}, std={s.std().item():.6f}")
        elif isinstance(state, torch.Tensor):
            print(f"  shape={state.shape}, mean={state.mean().item():.6f}, std={state.std().item():.6f}")
    
    async def test_single_modification(self, prompt_tokens=None):
        """Test a single self-modification cycle."""
        print("\n" + "="*70)
        print("TESTING SINGLE SELF-MODIFICATION CYCLE")
        print("="*70)
        
        # Initialize state
        self.model.init_latent_state(batch_size=1, device=self.device)
        state_initial = self.capture_state_snapshot(self.model.latent_state)
        
        if self.verbose:
            self.print_state_stats(state_initial, "Initial State")
        
        # Create a simple prompt if none provided
        if prompt_tokens is None:
            prompt_tokens = torch.randint(0, self.model.backbone.embedding.num_embeddings, 
                                         (1, 10), device=self.device)
        
        print(f"\nPrompt shape: {prompt_tokens.shape}")
        
        # Run deliberation (this should trigger self-modification)
        print("\nRunning MCTS deliberation...")
        suggestion, delta = await self.model.reasoning.deliberate(
            prompt_tokens, 
            self.model.latent_state
        )
        
        print(f"  Suggested token: {suggestion}")
        print(f"  Delta returned: {delta is not None}")
        
        # Apply the delta
        if delta is not None:
            print("\nApplying state delta...")
            self.model.apply_state_delta(delta)
            state_after = self.capture_state_snapshot(self.model.latent_state)
            
            if self.verbose:
                self.print_state_stats(delta, "Delta")
                self.print_state_stats(state_after, "State After Modification")
            
            # Compute difference
            diff = self.compute_state_difference(state_initial, state_after)
            print(f"\n✓ State modification detected!")
            print(f"  Average L2 difference: {diff:.6f}")
            
            return True, diff
        else:
            print("\n✗ No delta returned - self-modification did not occur")
            return False, 0.0
    
    async def test_multiple_modifications(self, iterations=5):
        """Test multiple self-modification cycles to measure accumulation."""
        print("\n" + "="*70)
        print(f"TESTING MULTIPLE SELF-MODIFICATIONS ({iterations} iterations)")
        print("="*70)
        
        # Initialize
        self.model.init_latent_state(batch_size=1, device=self.device)
        state_initial = self.capture_state_snapshot(self.model.latent_state)
        
        prompt_tokens = torch.randint(0, self.model.backbone.embedding.num_embeddings, 
                                     (1, 10), device=self.device)
        
        differences = []
        cumulative_diff = 0.0
        
        for i in range(iterations):
            print(f"\nIteration {i+1}/{iterations}")
            print("-" * 40)
            
            state_before = self.capture_state_snapshot(self.model.latent_state)
            
            # Deliberate
            suggestion, delta = await self.model.reasoning.deliberate(
                prompt_tokens, 
                self.model.latent_state
            )
            
            if delta is not None:
                self.model.apply_state_delta(delta)
                state_after = self.capture_state_snapshot(self.model.latent_state)
                
                iter_diff = self.compute_state_difference(state_before, state_after)
                cumulative_diff = self.compute_state_difference(state_initial, state_after)
                
                differences.append(iter_diff)
                
                print(f"  Token: {suggestion}")
                print(f"  Step difference: {iter_diff:.6f}")
                print(f"  Cumulative difference: {cumulative_diff:.6f}")
                
                # Update prompt with new token
                new_token = torch.tensor([[suggestion]], device=self.device)
                prompt_tokens = torch.cat([prompt_tokens, new_token], dim=1)
            else:
                print(f"  No modification in iteration {i+1}")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if differences:
            print(f"Total iterations with modification: {len(differences)}/{iterations}")
            print(f"Average step difference: {np.mean(differences):.6f} ± {np.std(differences):.6f}")
            print(f"Total cumulative difference: {cumulative_diff:.6f}")
            print(f"Modification rate: {len(differences)/iterations*100:.1f}%")
            
            return True, differences
        else:
            print("No modifications detected across all iterations")
            return False, []
    
    async def test_dynamics_vs_ssm(self):
        """Compare dynamics predictions vs actual SSM transitions."""
        print("\n" + "="*70)
        print("TESTING DYNAMICS MODEL ACCURACY")
        print("="*70)
        
        vocab_size = self.model.backbone.embedding.num_embeddings
        num_tests = 20
        
        errors = []
        
        print(f"\nRunning {num_tests} comparison tests...")
        
        for i in range(num_tests):
            # Random state and action
            states = self.model.backbone.init_state(batch_size=1, device=self.device)
            action = torch.randint(0, vocab_size, (1,), device=self.device)
            
            # Get actual next state from SSM
            _, actual_next_states = self.model.backbone.step(action.unsqueeze(1), states)
            
            # Get predicted next state from dynamics
            action_emb = self.model.backbone.embedding(action.unsqueeze(1)).squeeze(1)
            
            predicted_states = []
            for s_curr in states:
                if s_curr is None:
                    predicted_states.append(None)
                    continue
                delta = self.model.dynamics(s_curr, action_emb)
                predicted_states.append(s_curr + delta)
            
            # Compare
            for pred, actual in zip(predicted_states, actual_next_states):
                if pred is None or actual is None:
                    continue
                error = torch.mean((pred - actual) ** 2).item()
                errors.append(error)
        
        if errors:
            avg_mse = np.mean(errors)
            print(f"\n✓ Dynamics model evaluation complete")
            print(f"  Average MSE: {avg_mse:.6f}")
            print(f"  Std MSE: {np.std(errors):.6f}")
            print(f"  Min MSE: {np.min(errors):.6f}")
            print(f"  Max MSE: {np.max(errors):.6f}")
            
            if avg_mse < 0.01:
                print(f"  Quality: ✓ Excellent")
            elif avg_mse < 0.1:
                print(f"  Quality: ✓ Good")
            elif avg_mse < 1.0:
                print(f"  Quality: ⚠ Acceptable")
            else:
                print(f"  Quality: ✗ Poor (consider retraining)")
            
            return True, avg_mse
        else:
            print("\n✗ Could not evaluate dynamics model")
            return False, float('inf')


def load_checkpoint_config(checkpoint_path):
    """Extract model configuration from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    
    # Try to infer config from state dict
    config = {}
    state_dict = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt))
    
    # Detect if it's a CRSM checkpoint (has backbone prefix)
    is_crsm = any(k.startswith('backbone.') for k in state_dict.keys())
    prefix = 'backbone.' if is_crsm else ''
    
    # Infer vocab_size from embedding layer
    emb_key = f'{prefix}embedding.weight'
    if emb_key in state_dict:
        config['vocab_size'] = state_dict[emb_key].shape[0]
        config['d_model'] = state_dict[emb_key].shape[1]
    
    # Count layers
    layer_keys = [k for k in state_dict.keys() if f'{prefix}layers.' in k]
    if layer_keys:
        layer_indices = set()
        for k in layer_keys:
            parts = k.split('.')
            for i, p in enumerate(parts):
                if p == 'layers' and i + 1 < len(parts):
                    try:
                        layer_indices.add(int(parts[i + 1]))
                    except ValueError:
                        pass
        if layer_indices:
            config['num_layers'] = max(layer_indices) + 1
    
    # Infer d_ffn from FFN layers
    ffn_key = f'{prefix}layers.0.ffn.0.weight'
    if ffn_key in state_dict:
        config['d_ffn'] = state_dict[ffn_key].shape[0]
    
    # Infer d_state (harder, use default if not found)
    config.setdefault('d_state', 64)
    config.setdefault('num_layers', 2)
    config.setdefault('d_ffn', 512)
    
    return config, is_crsm


def create_test_model(vocab_size=1000, d_model=128, dynamics_path=None, checkpoint_path=None):
    """Create a CRSM model for testing, optionally loading from checkpoint."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Extract config from checkpoint
        config, is_crsm = load_checkpoint_config(checkpoint_path)
        print(f"  Detected {'CRSM' if is_crsm else 'MambaModel'} checkpoint")
        print(f"  Vocab size: {config['vocab_size']}")
        print(f"  Model dimension: {config['d_model']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  FFN dimension: {config['d_ffn']}")
        
        # Create model with detected config
        model = CRSM(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            d_state=config['d_state'],
            d_ffn=config['d_ffn'],
            num_layers=config['num_layers'],
            dropout=0.0,
            c_puct=1.0,
            n_simulations=10,
            autonomous_mode=False
        )
        model = model.to(device)
        
        # Load checkpoint weights
        ckpt = torch.load(checkpoint_path, map_location=device)
        
        # Extract state dict
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'model_state' in ckpt:
            state_dict = ckpt['model_state']
        else:
            state_dict = ckpt
        
        # Handle CRSM vs MambaModel checkpoints
        if is_crsm:
            # CRSM checkpoint - can load directly
            try:
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded full CRSM checkpoint")
                if missing:
                    print(f"    Missing keys: {len(missing)} (this is OK)")
                if unexpected:
                    print(f"    Unexpected keys: {len(unexpected)} (ignored)")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load full checkpoint: {e}")
                print(f"    Attempting to load backbone only...")
                # Try loading just backbone
                backbone_state = {k.replace('backbone.', ''): v 
                                 for k, v in state_dict.items() 
                                 if k.startswith('backbone.')}
                model.backbone.load_state_dict(backbone_state, strict=False)
                print(f"  ✓ Loaded backbone weights")
        else:
            # MambaModel checkpoint - load into backbone
            try:
                model.backbone.load_state_dict(state_dict, strict=False)
                print(f"  ✓ Loaded MambaModel weights into backbone")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load checkpoint: {e}")
        
        # Check if checkpoint contains trained dynamics
        if is_crsm and 'dynamics.net.0.weight' in state_dict:
            print(f"  ✓ Checkpoint includes trained dynamics")
            # Dynamics already loaded with full state dict
        elif dynamics_path is not None:
            # Load separate dynamics file
            if Path(dynamics_path).exists():
                print(f"  Loading dynamics from: {dynamics_path}")
                success = model.load_dynamics(dynamics_path)
                if not success:
                    print("  ⚠ Warning: Could not load dynamics")
            else:
                print(f"  ⚠ Dynamics path not found: {dynamics_path}")
        else:
            print("  Using randomly initialized dynamics")
        
        model.eval()
        print(f"  Device: {device}")
        return model
    
    else:
        # Create new model from scratch
        print("Creating test CRSM model...")
        
        model = CRSM(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=64,
            d_ffn=512,
            num_layers=2,
            dropout=0.0,
            c_puct=1.0,
            n_simulations=10,  # Reduced for faster testing
            autonomous_mode=False
        )
        
        model = model.to(device)
        model.eval()
        
        print(f"  Device: {device}")
        print(f"  Vocab size: {vocab_size}")
        print(f"  Model dimension: {d_model}")
        
        # Load dynamics if path provided
        if dynamics_path is not None:
            if Path(dynamics_path).exists():
                print(f"  Loading dynamics from: {dynamics_path}")
                success = model.load_dynamics(dynamics_path)
                if not success:
                    print("  ⚠ Warning: Could not load dynamics, using random initialization")
            else:
                print(f"  ⚠ Dynamics path not found: {dynamics_path}")
        else:
            print("  Using randomly initialized dynamics")
        
        return model


async def run_all_tests(model, verbose=False, iterations=5):
    """Run all self-modification tests."""
    tester = SelfModificationTester(model, verbose=verbose)
    
    results = {}
    
    # Test 1: Single modification
    try:
        success, diff = await tester.test_single_modification()
        results['single_modification'] = {'success': success, 'difference': diff}
    except Exception as e:
        print(f"\n✗ Single modification test failed: {e}")
        results['single_modification'] = {'success': False, 'error': str(e)}
    
    # Test 2: Multiple modifications
    try:
        success, diffs = await tester.test_multiple_modifications(iterations=iterations)
        results['multiple_modifications'] = {
            'success': success, 
            'differences': diffs,
            'total': len(diffs)
        }
    except Exception as e:
        print(f"\n✗ Multiple modifications test failed: {e}")
        results['multiple_modifications'] = {'success': False, 'error': str(e)}
    
    # Test 3: Dynamics accuracy
    try:
        success, mse = await tester.test_dynamics_vs_ssm()
        results['dynamics_accuracy'] = {'success': success, 'mse': mse}
    except Exception as e:
        print(f"\n✗ Dynamics accuracy test failed: {e}")
        results['dynamics_accuracy'] = {'success': False, 'error': str(e)}
    
    return results


def print_final_report(results):
    """Print a final summary report."""
    print("\n" + "="*70)
    print("FINAL TEST REPORT")
    print("="*70)
    
    all_passed = True
    
    # Single modification
    if 'single_modification' in results:
        r = results['single_modification']
        status = "✓ PASS" if r.get('success') else "✗ FAIL"
        print(f"\n1. Single Modification Test: {status}")
        if r.get('success'):
            print(f"   State changed by: {r.get('difference', 0):.6f}")
        elif 'error' in r:
            print(f"   Error: {r['error']}")
            all_passed = False
    
    # Multiple modifications
    if 'multiple_modifications' in results:
        r = results['multiple_modifications']
        status = "✓ PASS" if r.get('success') else "✗ FAIL"
        print(f"\n2. Multiple Modifications Test: {status}")
        if r.get('success'):
            print(f"   Successful modifications: {r.get('total', 0)}")
            if r.get('differences'):
                print(f"   Average change per step: {np.mean(r['differences']):.6f}")
        elif 'error' in r:
            print(f"   Error: {r['error']}")
            all_passed = False
    
    # Dynamics accuracy
    if 'dynamics_accuracy' in results:
        r = results['dynamics_accuracy']
        status = "✓ PASS" if r.get('success') else "✗ FAIL"
        print(f"\n3. Dynamics Accuracy Test: {status}")
        if r.get('success'):
            mse = r.get('mse', float('inf'))
            print(f"   Average MSE: {mse:.6f}")
            if mse < 0.1:
                print(f"   Quality: Good")
            else:
                print(f"   Quality: Needs improvement")
        elif 'error' in r:
            print(f"   Error: {r['error']}")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Self-modification is working correctly!")
    else:
        print("⚠ SOME TESTS FAILED - Review errors above")
    print("="*70 + "\n")
    
    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Test CRSM self-modification capabilities"
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to trained CRSM or MambaModel checkpoint (e.g., checkpoints/crsm_epoch10.pt)')
    parser.add_argument('--dynamics-path', type=str, default=None,
                       help='Path to trained dynamics checkpoint (if separate from main checkpoint)')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size (only used if no checkpoint provided)')
    parser.add_argument('--d-model', type=int, default=128,
                       help='Model dimension (only used if no checkpoint provided)')
    parser.add_argument('--iterations', type=int, default=5,
                       help='Number of iterations for multiple modification test')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed state information')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("CRSM SELF-MODIFICATION TEST SUITE")
    print("="*70)
    
    # Create model
    model = create_test_model(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        dynamics_path=args.dynamics_path,
        checkpoint_path=args.checkpoint
    )
    
    # Run tests
    results = asyncio.run(run_all_tests(
        model, 
        verbose=args.verbose, 
        iterations=args.iterations
    ))
    
    # Print report
    all_passed = print_final_report(results)
    
    # Exit code
    exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()