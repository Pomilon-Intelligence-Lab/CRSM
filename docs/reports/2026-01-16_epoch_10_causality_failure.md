# Post-RL Epoch 10: Causality Failure Analysis

**Date:** January 16, 2026
**Subject:** Lack of Causal Influence from System 2 (Reasoning) Components in 5M Parameter Model
**Status:** CRITICAL - BLOCKER

## 1. Executive Summary

Evaluation of the CRSM model at RL Epoch 10 reveals a **complete lack of causal influence** from the reasoning components (MCTS, Dynamics, Critics) on model performance. While the Reinforcement Learning (RL) process successfully improved the backbone's ability to structure output and adhere to syntax (System 1), the "System 2" reasoning engine is currently decorative.

Ablation studies confirm that disabling MCTS, the Dynamics Model, or the Critics results in **statistically identical** performance metrics. The model is effectively operating as a standard Transformer with a vestigial reasoning appendage.

## 2. Experimental Setup

- **Model:** 5M Parameter "Small" Config (`d_model=176`, `layers=6`).
- **Training State:**
    -   SFT: 20 Epochs (Loss ~0.005).
    -   RL: 10 Epochs (GRPO, Dense Rewards).
- **Evaluation Protocol:** `scripts/eval/causality_test.py`
- **Configurations Tested:**
    1.  **Full System:** MCTS + Dynamics + Critics active.
    2.  **No MCTS:** MCTS disabled, relying on backbone priors.
    3.  **No Critics:** Value heads disabled/random.
    4.  **No Dynamics:** State transitions disabled.

## 3. Findings

### 3.1. The "Smoking Gun"
Performance metrics across all four configurations were identical.

| Configuration | Accuracy | Avg Reward | Avg Length | Syntax Error Rate |
| :--- | :--- | :--- | :--- | :--- |
| **Full System** | 0.0% | 0.1173 | 2.27 | 0.0% |
| **No MCTS** | 0.0% | 0.1173 | 2.27 | 0.0% |
| **No Critics** | 0.0% | 0.1173 | 2.27 | 0.0% |
| **No Dynamics** | 0.0% | 0.1173 | 2.27 | 0.0% |

### 3.2. System 1 Success (Reward Hacking)
The RL process *did* work, but strictly on the backbone (System 1).
-   **Baseline Reward:** ~0.04 (SFT model)
-   **Current Reward:** ~0.11 (RL Epoch 10)
-   **Behavior:** The model learned to emit short, syntactically valid sequences to maximize the dense "structure" and "pixel-match" rewards. It optimizes for *compliance*, not *solution*.

### 3.3. System 2 Failure
The reasoning engine is running but effective contribution is zero. The "delta" injection from the MCTS planner is either:
1.  Too weak to override the backbone.
2.  Optimizing for the exact same "easy" objective as the backbone, adding no new information.
3.  Guiding based on flat/useless value heads.

## 4. Root Cause Analysis

### Hypothesis A: Reward Alignment (The "Echo Chamber")
**Likelihood: High**
The Backbone and the MCTS Planner are both optimizing the same dense reward signals (syntax, immediate pixel matches). Since the Backbone (System 1) can achieve these easy rewards instantly without search, the MCTS Planner converges to the same policy as the Backbone. MCTS adds no "lookahead" value because the short-term rewards don't require it.

### Hypothesis B: Weak Injection (The "Timid Advisor")
**Likelihood: Medium**
The gating mechanism (`alpha=0.05` or similar) or the additive nature of the delta `h = h + delta` might be insufficient to steer a confident backbone. If the backbone's logits are sharp (high confidence), a small latent perturbation won't change the argmax token selection.

### Hypothesis C: Flat Critics (The "Blind Guide")
**Likelihood: High**
The Value Heads (Critics) may have collapsed to predicting a constant mean reward. If `Variance(V(s)) ≈ 0`, the MCTS search has no gradient to climb and effectively performs a random walk or falls back to the backbone prior.

## 5. Remediation Plan

We will proceed with the following prioritized actions to force System 2 into relevance.

### Priority 1: Verify Critics
**Action:** Inspect value head outputs on a batch of data.
**Success Criteria:** `Variance(Value_Predictions) > Threshold`. If variance is near zero, critics are broken.

### Priority 2: Force Injection (Falsification)
**Action:** Temporarily increase injection rate to `1.0` or use **Override Mode** (`h = target_state`) during inference.
**Goal:** Determine if a *perfect* plan (or a strong plan) *can* even control the model. If strong injection ruins performance, the planner is generating garbage. If it changes nothing, the backbone is ignoring the latent state.

### Priority 3: Break Reward Alignment
**Action:** Shift to **Sparse / Delayed Rewards** for the MCTS planner.
**Goal:** Remove intermediate "structure" rewards for the planner. Force it to only value the terminal state (Solution Correctness). This forces System 2 to look ahead where System 1 cannot see.

### Priority 4: Reduce Action Space (Long Term)
**Action:** Move from token-level planning to macro-actions (e.g., "Fill Region", "Reflect Object").
**Goal:** Make the search horizon reachable for MCTS.

## 6. Conclusion
The architecture is mechanically sound but functionally disconnected. The immediate goal is no longer "improving accuracy" but **establishing causality**. We must break the model's reliance on System 1 before resuming full training.
