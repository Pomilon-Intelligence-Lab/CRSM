# Report: Hierarchical State Sovereignty & Sparse-Gated Injection
**Date:** January 15, 2026
**Commit:** 61
**Status:** Verified & Integrated

## 1. Executive Summary
This report details the transition of the CRSM architecture from a "Global Gating" model to a **Hierarchical State Sovereignty** model. This shift addresses core flaws in latent state integration—specifically "Policy Blurring" and "Averaging Discrepancy"—that hindered the model's ability to coordinate low-level perception with high-level reasoning.

## 2. Identified Flaws (The "Blur" Problem)
Prior to Commit 61, the CRSM architecture treated the multi-layered Mamba backbone as a monolithic state. 
*   **Averaging Discrepancy:** The reasoning engine averaged the latent states of all layers into a single vector before evaluation. This "smeared" high-level semantic logic with low-level syntactic features.
*   **Blind Planning:** The asynchronous planner optimized for the state at step $t$, but by the time the plan was ready, the model was at step $t+k$. The resulting injection was mathematically misaligned.

## 3. The Hierarchical Solution

### 3.1 Sparse-Gated Hierarchical Injection
Instead of a single global gate, we implemented **Sparse-Gating**. Each layer in the Mamba hierarchy is now a sovereign entity.
*   **Mechanism:** Each layer has an independent Sigmoid gate modulated by its own confidence score.
*   **Math:** $h_{i,t} \leftarrow (1 - \alpha_{i}) \cdot h_{i,t} + \alpha_{i} \cdot h_{i,target}$
*   **Result:** The planner can now aggressively update high-level strategy layers (where it is confident) while leaving low-level syntax layers untouched, preserving generation fluency.

### 3.2 Multi-Headed Value Critic (MV-Critic)
We replaced the single scalar value head with a **ModuleList of Value Heads** (one per layer).
*   **Consensus Logic:** The MCTS utility score is now a **Weighted Consensus**.
*   **Uncertainty Penalty:** High variance between layer-wise values now penalizes a reasoning path, forcing the MCTS to favor trajectories that are stable across all levels of abstraction.

### 3.3 Hierarchical Policy Fusion
We moved away from "Last Layer Selection" for the policy head.
*   **Implementation:** A **Learned Weighted Sum** (Softmax-normalized) fuses all layer states into the final token predictor.
*   **Hierarchical Weight Supervision:** Added an **Entropy Loss** term to these weights during training, forcing the model to explore and fuse multiple abstraction levels instead of collapsing into a single-layer policy.

### 3.4 Upgraded Dynamics (Recurrent World Model)
Replaced the static MLP dynamics with a **GRUCell**.
*   **Benefit:** The world model can now capture temporal dependencies and state residuals, providing a much higher-fidelity simulation for the MCTS rollouts.

## 4. Precise Async Alignment

### 4.1 Forward-Projected Planning
The planner now uses its internal dynamics model to "fast-forward" the state ($S_t \to S_{t+lag}$) before search begins. This ensures the plan is optimized for the *exact future context* where it will be applied.

### 4.2 Targeted Delta Buffer
We implemented a buffering system to solve the "Fast/Slow MCTS" problem.
*   **Logic:** Plans are no longer applied immediately. They are stored in a buffer and injected **only** when the generation loop reaches the specific step the plan was optimized for.

## 5. Verification Results

| Test | Result | Insight |
| :--- | :--- | :--- |
| `verify_sparse_gating.py` | **PASSED** | Confirmed confidence-proportional layer updates. |
| `verify_projection.py` | **PASSED** | Confirmed deterministic state advancement. |
| `verify_lag_correction.py` | **PASSED** | Confirmed precise alignment and decay of stale plans. |
| `verify_capabilities.py` | **PASSED** | Confirmed dynamics fidelity (0.99+ CosSim) and stability. |

## 6. Conclusion
The move to **Hierarchical State Sovereignty** resolves the "Cargo Cult" mathematical errors of earlier versions. The system now respects the natural abstraction hierarchy of the Mamba backbone, leading to more stable state steering and sharper reasoning capabilities. This architecture is now positioned as a viable candidate for ARC-AGI task benchmarking.
