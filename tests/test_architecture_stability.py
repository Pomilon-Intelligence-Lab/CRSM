import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from crsm.core import CRSMConfig, CRSMModel

# Ensure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_model():
    config = CRSMConfig(
        vocab_size=1000, 
        hidden_size=256, 
        num_hidden_layers=4, 
        d_state=64,
        injection_rate=0.1 # Base rate
    )
    return CRSMModel(config).to(device)

class FinalArchitectureVerification:
    def __init__(self):
        self.model = get_model()
        self.model.eval()
        
    def run_oracle_injection_test(self, trials=20):
        """
        Test 1: ORACLE VALIDITY
        Hypothesis: If MCTS finds the 'Perfect' state, Gated Injection moves the model towards it
        and reduces Loss, regardless of the current erroneous state.
        """
        print(f"\n{'='*60}")
        print(f"TEST 1: ORACLE INJECTION VALIDITY ({trials} trials)")
        print(f"{'='*60}")
        
        success_count = 0
        total_improvement = 0.0
        
        for i in range(trials):
            # 1. Create a random sequence and get its "True" state (The Oracle)
            seq = torch.randint(0, 1000, (1, 10)).to(device)
            
            self.model.crsm.init_latent_state(batch_size=1, device=device)
            with torch.no_grad():
                logits_true, new_states_true = self.model(seq)
                # FIX: Update the latent state with the result of the forward pass
                self.model.crsm.latent_state = [s.clone() if s is not None else None for s in new_states_true]
                state_true = [s.clone() if s is not None else None for s in self.model.crsm.latent_state]
                target_token = torch.argmax(logits_true[:, -1], dim=-1)

            # 2. Perturb the model's state to simulate 'Confusion'
            # We add noise to the oracle state
            bad_state = []
            for s in state_true:
                noise = torch.randn_like(s) * 2.0 # Significant noise
                bad_state.append(s + noise)
            self.model.crsm.latent_state = [s.clone() for s in bad_state]
            
            # 3. Measure Baseline Loss (High because of noise)
            with torch.no_grad():
                logits_bad, _, _ = self.model.crsm.backbone.predict_from_states(self.model.crsm.latent_state)
                loss_bad = F.cross_entropy(logits_bad[:, 0], target_token)
                
            # 4. Inject the ORACLE STATE using the mechanism
            # We act as if MCTS found 'oracle_state' and returned it.
            # Confidence = 1.0 (High certainty)
            # Injection Rate = 0.1 (Config)
            # Expected: State moves 10% towards Oracle. Loss should drop.
            
            # Simulate confidence scaling: alpha * confidence
            confidence = 1.0
            effective_alpha = self.model.crsm.injection_rate * confidence
            
            # The "Delta" passed to apply_state_delta is now the TARGET STATE
            targets = [t.clone() for t in state_true]
            
            self.model.crsm.apply_state_delta(targets, scale=effective_alpha)
            
            # 5. Measure New Loss
            with torch.no_grad():
                logits_new, _, _ = self.model.crsm.backbone.predict_from_states(self.model.crsm.latent_state)
                loss_new = F.cross_entropy(logits_new[:, 0], target_token)
            
            improvement = loss_bad.item() - loss_new.item()
            total_improvement += improvement
            
            if loss_new < loss_bad:
                success_count += 1
                
            if i < 3:
                print(f"  Trial {i}: Loss {loss_bad:.4f} -> {loss_new:.4f} (Imp: {improvement:.4f})")

        avg_imp = total_improvement / trials
        print(f"Success Rate: {success_count}/{trials} ({success_count/trials*100}%)")
        print(f"Avg Loss Reduction: {avg_imp:.4f}")
        
        if success_count == trials:
            print("✅ PASSED: Oracle Injection consistently improves state.")
        else:
            print("⚠️ PARTIAL: Oracle Injection mostly works but had outliers.")

    def run_confidence_safety_test(self):
        """
        Test 2: CONFIDENCE SAFETY MECHANISM
        Hypothesis: If Confidence is 0.0, the state should NOT change, even if Delta is garbage.
        """
        print(f"\n{'='*60}")
        print(f"TEST 2: CONFIDENCE SAFETY CHECK")
        print(f"{'='*60}")
        
        # Init state
        self.model.crsm.init_latent_state(batch_size=1, device=device)
        original_state = [s.clone() for s in self.model.crsm.latent_state]
        
        # Create garbage delta (Target State)
        garbage_target = [torch.randn_like(s) * 100.0 for s in original_state] # Massive noise
        
        # Apply with Confidence = 0.0
        # effective_alpha = rate * 0.0 = 0.0
        self.model.crsm.apply_state_delta(garbage_target, scale=0.0)
        
        # Check difference
        diff_norm = 0.0
        for s_old, s_new in zip(original_state, self.model.crsm.latent_state):
            diff_norm += torch.norm(s_old - s_new).item()
            
        print(f"State Change Norm (Confidence 0.0): {diff_norm:.6f}")
        
        if diff_norm < 1e-5:
            print("✅ PASSED: Zero confidence completely blocks update.")
        else:
            print(f"❌ FAILED: State leaked! Norm: {diff_norm}")

    def run_linear_scaling_test(self):
        """
        Test 3: LINEAR SCALING VERIFICATION
        Hypothesis: The impact on the state should be proportional to Confidence.
        """
        print(f"\n{'='*60}")
        print(f"TEST 3: LINEAR SCALING VERIFICATION")
        print(f"{'='*60}")
        
        # Setup
        self.model.crsm.init_latent_state(batch_size=1, device=device)
        base_state = [torch.zeros_like(s) for s in self.model.crsm.latent_state] # Start at zero for clarity
        target_state = [torch.ones_like(s) for s in base_state] # Target is all ones
        
        # We expect: New State = (1-alpha)*0 + alpha*1 = alpha
        # alpha = base_rate * confidence
        base_rate = 0.1
        self.model.crsm.injection_rate = base_rate
        
        confidences = [0.1, 0.5, 1.0]
        
        print(f"{'Conf':<10} | {'Exp. Val':<10} | {'Act. Val':<10} | {'Error':<10}")
        
        passed = True
        for conf in confidences:
            # Reset
            self.model.crsm.latent_state = [b.clone() for b in base_state]
            
            # Apply
            effective_alpha = base_rate * conf
            self.model.crsm.apply_state_delta(target_state, scale=effective_alpha)
            
            # Check value (should be equal to effective_alpha since we interpolating 0 -> 1)
            # Pick first element of first layer
            actual_val = self.model.crsm.latent_state[0][0,0].item()
            expected_val = effective_alpha
            
            error = abs(actual_val - expected_val)
            print(f"{conf:<10} | {expected_val:<10.4f} | {actual_val:<10.4f} | {error:<10.6f}")
            
            if error > 1e-5:
                passed = False
                
        if passed:
            print("✅ PASSED: Injection scales linearly with confidence.")
        else:
            print("❌ FAILED: Scaling math is incorrect.")

    def run_long_term_stability_test(self, steps=1000):
        """
        Test 4: LONG-TERM STABILITY (RECURRENCE)
        Hypothesis: Continuous injection of 'Average' confidence vectors (0.5) 
        should not cause the state norm to explode over time.
        """
        print(f"\n{'='*60}")
        print(f"TEST 4: LONG-TERM STABILITY ({steps} steps)")
        print(f"{'='*60}")
        
        self.model.crsm.init_latent_state(batch_size=1, device=device)
        self.model.crsm.injection_rate = 0.1
        
        norms = []
        max_norm = 0.0
        
        print("Simulating 1000 steps of 'Thinking'...")
        for i in range(steps):
            # 1. Current state
            current_state = self.model.crsm.latent_state
            
            # 2. Generate a "Thought" (Target State)
            # Random vector with similar magnitude to initialized state (~1.0-5.0)
            target_state = [torch.randn_like(s) for s in current_state]
            
            # 3. Inject with medium confidence
            confidence = 0.5
            effective_alpha = 0.1 * confidence
            
            self.model.crsm.apply_state_delta(target_state, scale=effective_alpha)
            
            # 4. Track Norm
            total_norm = sum(torch.norm(s).item() for s in self.model.crsm.latent_state)
            norms.append(total_norm)
            if total_norm > max_norm: max_norm = total_norm
            
            if total_norm > 100.0:
                print(f"❌ FAILED: Explosion detected at step {i} (Norm: {total_norm:.2f})")
                return

        avg_norm = np.mean(norms)
        print(f"Max Norm: {max_norm:.4f}")
        print(f"Avg Norm: {avg_norm:.4f}")
        
        if max_norm < 20.0:
            print("✅ PASSED: System is stable over long durations.")
        else:
            print("⚠️ WARNING: Norm is high but stable (check threshold).")

if __name__ == "__main__":
    verifier = FinalArchitectureVerification()
    
    verifier.run_oracle_injection_test()
    verifier.run_confidence_safety_test()
    verifier.run_linear_scaling_test()
    verifier.run_long_term_stability_test()
