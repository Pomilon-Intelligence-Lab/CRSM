# CRSM Project Roadmap

This document outlines the strategic roadmap for the Continuous Reasoning State Model (CRSM) project, from its current state as a functional prototype to a mature, publishable research project.

## Phase 1: Foundation and Core Implementation (✅ COMPLETE)

This phase involved the initial design, implementation, and debugging of the core CRSM architecture. All tasks in this phase have been successfully completed.

*   **[DONE]** Design the hybrid Mamba + MCTS architecture.
*   **[DONE]** Implement the Mamba backbone with an integrated value head (`crsm/mamba_ssm.py`).
*   **[DONE]** Implement the asynchronous MCTS deliberation loop (`crsm/reasoning.py`).
*   **[DONE]** Implement the core state-delta mechanism for self-modification (`model.py`, `reasoning.py`).
*   **[DONE]** Implement the lightweight `LatentDynamics` model for fast MCTS rollouts (`crsm/latent_dynamics.py`).
*   **[DONE]** Create a comprehensive test suite to verify the self-modification engine (`test_self_modification.py`).

## Phase 2: Architecture Stability & Verification (✅ COMPLETE)

*   **[DONE]** **Architecture Verification**: Diagnosed "State Explosion" issue and implemented **Gated State Injection** to solve it.
*   **[DONE]** **Safety Mechanisms**: Implemented Confidence Scaling to prevent model lobotomization by untrained planners.
*   **[DONE]** **Verification Suite**: Created `tests/test_architecture_stability.py` to prove mathematical correctness of the injection.

## Phase 3: The "Centaur" Pipeline Implementation (✅ COMPLETE)

We have successfully refactored the codebase into a modular 4-stage training pipeline (Backbone → Dynamics → Offline Value Training → Assembly) and modernized the configuration system.

*   **[DONE]** **Refactoring & Archival:**
    *   Archived v0 prototype scripts to `scripts/archive/v0_prototype/`.
    *   Centralized configuration in YAML format (`configs/*.yaml`).
*   **[DONE]** **Training Pipeline Implementation:**
    *   **Stage 1 (System 1):** Implemented `scripts/training/stage_1_backbone.py` for Mamba backbone training.
    *   **Stage 2 (Subconscious):** Implemented `scripts/training/stage_2_dynamics.py` for dynamics distillation.
    *   **Stage 3 (Judgment):** Implemented `scripts/training/stage_3_value_head.py` for offline Expert Iteration and value head training.
    *   **Stage 4 (Assembly):** Implemented `scripts/training/stage_4_assembly.py` for final model integration.
*   **[DONE]** **Evaluation & Verification:**
    *   Implemented `benchmark_reasoning.py` to compare System 1 vs. System 2 performance.
    *   Implemented `verify_steering.py` to validate state injection.
    *   Verified pipeline end-to-end on synthetic data.
*   **[DONE]** **Cloud Readiness:**
    *   Created Google Colab notebooks for all three training stages (`notebooks/cloud_training/`).

## Phase 4: Hierarchical State Sovereignty & Aligned Planning (✅ COMPLETE)

This phase addressed the "Policy Blurring" and "Async Drift" issues, moving the architecture from a monolithic state model to a hierarchical committee of sovereign layers.

*   **[DONE]** **Multi-Headed Value Critic (MV-Critic):** Implemented independent value heads for every layer to enable granular confidence scoring.
*   **[DONE]** **Sparse-Gated Hierarchical Injection:** Replaced global gating with per-layer sovereignty, allowing strategy layers to update while syntax layers remain stable.
*   **[DONE]** **Hierarchical Policy Fusion:** Implemented a Learned Weighted Sum of all abstraction levels for the final output policy.
*   **[DONE]** **Forward-Projected Planning:** Enabled the planner to fast-forward states using the dynamics model to align with future generation steps.
*   **[DONE]** **Targeted Delta Buffer:** Implemented precise state-step alignment to ensure MCTS results are injected exactly at their intended positions.
*   **[DONE]** **Upgraded Dynamics (Recurrent World Model):** Upgraded the MLP dynamics to a GRUCell for higher simulation fidelity.

## Phase 5: Modular Refactor & ARC-AGI Ready (✅ COMPLETE)

This phase decoupled the core architecture from task logic, enabling rapid iteration on the ARC-AGI benchmark within tight parameter constraints.

*   **[DONE]** **Domain Decoupling:** Separated the codebase into `core/`, `tasks/`, `training/`, and `data/`.
*   **[DONE]** **Task Abstraction:** Implemented the `BaseTask` interface and migrated LM and Distillation logic.
*   **[DONE]** **Generic Training Engine:** Unified procedural training scripts into a robust `Trainer` class.
*   **[DONE]** **Hierarchical Weight Supervision:** Added Entropy Loss to prevent feature collapse during multi-layer fusion.
*   **[DONE]** **Targeted Delta Buffer:** Resolved asynchronous application drift for precise state alignment.

## Phase 6: Large-Scale Training & Benchmarking (🚀 NEXT STEPS)

With the hierarchical, modular architecture certified, we are ready to target **ARC-AGI**.

*   **Tasks:**
    *   **1. ARC-AGI Integration:**
        *   **[TODO]** Implement `ARCTask` for grid-based spatial reasoning.
        *   **[TODO]** Develop a grid-to-token encoding strategy optimized for Nano-scale models.
    *   **2. Nano-Scale Training (100k - 500k):**
        *   **[TODO]** Benchmark CRSM against standard Mamba/Transformer baselines on ARC tasks.
    *   **3. Advanced Evaluation:**
        *   **[TODO]** Evaluate on standard reasoning benchmarks (GSM8K, ARC).
        *   **[TODO]** Conduct ablation studies on "Injection Rate" and "Deliberation Lag".

## Phase 6: Scaling and Optimization

*   **Tasks:**
    *   **1. Scale to 1B+ Parameters:**
        *   **[TODO]** Train a 1B+ model using distributed training (DDP).
    *   **2. Inference Optimization:**
        *   **[TODO]** Port MCTS logic to Rust/C++ for lower latency.
        *   **[TODO]** Implement true parallel execution (System 2 not blocking System 1).

## Phase 7: Publication and Release

*   **Tasks:**
    *   **[TODO]** Write and submit research paper.
    *   **[TODO]** Release open-source models on Hugging Face.
    *   **[TODO]** Create interactive demos.
