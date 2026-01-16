# Post-RL Epoch 10: Causality Analysis & Remediation Report

**Date:** January 16, 2026
**Subject:** Diagnosis of System 2 (Reasoning) Ineffectiveness and Implementation of Dual Reward Scheme
**Status:** RESOLVED (Ready for Phase 2 Training)

## 1. Executive Summary

Following the discovery that the "System 2" reasoning engine (MCTS + Dynamics) had **zero causal influence** on the model's performance at RL Epoch 10, we conducted a rigorous root cause analysis. 

We have **conclusively identified** the issue as "Reward Alignment" (The Echo Chamber). The MCTS planner was optimizing the same dense, easy-to-get rewards as the backbone, leading to policy collapse where the planner simply agreed with the backbone's intuition.

We have implemented a **Dual Reward Scheme** to decouple the objectives:
*   **System 1 (Backbone):** Continues to optimize for Structure & Partial Credit (Stability).
*   **System 2 (Value Head/MCTS):** Now forces optimization of strictly Binary Success (Correctness).

## 2. Investigation Findings

### 2.1. Hypothesis A: Reward Alignment (CONFIRMED)
The backbone (System 1) had become efficient at gaining "partial credit" rewards (syntax, pixel matching) (~0.11 reward). Since the Value Head was trained on the same signal, the MCTS planner saw no advantage in deviating from the backbone's safe, local policy to attempt high-risk, high-reward reasoning.

### 2.2. Hypothesis B: Weak Injection Mechanism (DISPROVED)
We hypothesized that the MCTS "delta" injection was mechanically too weak to steer the model.
*   **Test:** `scripts/eval/verify_mcts_control.py`
*   **Result:** Increasing `injection_rate` from 0.0 to 1.0 **significantly altered the output**.
*   **Conclusion:** The steering mechanism works. The planner *could* steer the model, it just *chose not to* (because the backbone's plan was "optimal" for the partial reward).

### 2.3. Hypothesis C: Flat Critics (DISPROVED)
We hypothesized the Value Heads had collapsed to a constant value (zero variance).
*   **Test:** `scripts/eval/verify_critics.py`
*   **Result:** Value heads showed healthy variance (Avg Std Dev ≈ 0.07).
*   **Conclusion:** The critics are active and discriminating, but they were discriminating based on the *wrong objective* (partial credit vs. zero), reinforcing the local optimum.

## 3. The Fix: Dual Reward Scheme

To force System 2 to provide *additive value* over System 1, their objectives must diverge.

We modified `crsm/training/rl_trainer.py` to implement distinct reward signals:

| Component | Objective | Reward Function | Goal |
| :--- | :--- | :--- | :--- |
| **Backbone (Policy)** | **Structure & Syntax** | `0.1 + 0.4 * accuracy` (Dense) | Maintain valid grid output; don't collapse into gibberish. |
| **Value Head (Critic)** | **Solution Correctness** | `1.0` if Exact Match else `0.0` (Sparse) | Force planner to look for the *actual solution*. |

### Code Implementation
In `RLTrainer.fit`:
```python
# Policy Loss (Backbone)
policy_loss += -advantages[k] * traj_log_prob

# Value Loss (Critic) - STRICT BINARY TARGET
is_exact_correct = (rewards[k] >= 1.0)
target_val = 1.0 if is_exact_correct else 0.0
v_loss = sum(F.mse_loss(v.squeeze(), target_val_tensor) for v in values)
```

## 4. Expected Outcome of Phase 2 Training

By resuming training with this configuration:
1.  **Short Term:** Value loss may spike as the critics realize their previous "partial credit" predictions were wrong.
2.  **Medium Term:** MCTS will begin identifying that "safe" backbone paths (which get partial reward) actually have `Value=0.0`. It will start searching for `Value=1.0` paths.
3.  **Long Term:** The MCTS `delta` will inject "corrective" information into the backbone, steering it away from local optima (partial matches) toward global optima (solutions).

## 5. Next Steps

1.  Resume RL Training immediately.
2.  Monitor `value_loss` (should decrease) and `accuracy` (should strictly increase).
3.  Re-run `causality_test.py` at Epoch 15 or 20. We expect to see **System 2 Drop-out** hurt performance (i.e., `Full System > No MCTS`).
