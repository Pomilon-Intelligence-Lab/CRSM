# CRSM Project: Comprehensive Synthesis

This document synthesizes all information gathered from the project's source code, conversation logs, and supplementary documents. It serves as the single source of truth for creating the project roadmap, planning documents, and research paper.

## 1. Core Concept and Architecture

The **Continuous Reasoning State Model (CRSM)** is a novel neural network architecture designed to overcome the limitations of standard autoregressive Transformers, specifically their high latency for planning tasks and their inability to perform genuine lookahead reasoning (the "Clever Hans" problem).

The architecture is a hybrid model composed of two main components operating in parallel:

*   **Backbone (Continuous State Module):**
    *   **Implementation:** A **Mamba** State Space Model (SSM).
    *   **Function:** Maintains a continuous, evolving latent state (`h(t)`) that provides a compressed representation of the history. Its linear-time complexity (`O(N)`) makes it highly efficient for long sequences.
    *   **Files:** `crsm/mamba_ssm.py`

*   **Planner (Asynchronous Deliberation Module):**
    *   **Implementation:** A **Monte Carlo Tree Search (MCTS)** algorithm.
    *   **Function:** Runs asynchronously in a separate thread, exploring future possibilities (reasoning paths) without blocking the main generation process. It evaluates the quality of states using a learned `value_head`.
    *   **Files:** `crsm/reasoning.py`

### Key Innovation: The State Delta Mechanism

The two components are fused via a **state delta (Δ)** mechanism. This is the core innovation of CRSM.

1.  **Deliberation:** The MCTS planner explores possible future states.
2.  **Delta Computation:** After its search, the MCTS computes a `delta` vector, representing the weighted difference between the current latent state and the most promising future state it discovered (`_compute_delta_from_mcts` in `reasoning.py`).
3.  **State Modification:** This `delta` is passed back to the main model, which applies it directly to its canonical `latent_state` (`apply_state_delta` in `model.py`).

This creates a feedback loop where the planner's "thoughts" continuously "nudge" and correct the backbone's internal state, enabling self-correction and more coherent, long-horizon reasoning.

### The "Fast Path" Optimization: Learned Dynamics

To make the MCTS rollouts efficient, the architecture includes a third, smaller neural network:

*   **Latent Dynamics Model:**
    *   **Implementation:** A small Multi-Layer Perceptron (MLP).
    *   **Function:** Acts as a lightweight, learned "world model" that predicts the next state delta (`f_θ(state, action) -> Δh`).
    *   **Purpose:** During MCTS simulations, the planner uses this fast model for internal rollouts instead of the full, slow Mamba backbone, dramatically speeding up the search process.
    *   **Files:** `crsm/latent_dynamics.py`

## 2. The Four-Stage Training Pipeline

The project employs a sophisticated, four-stage training pipeline, orchestrated by `scripts/train_full_crsm.py`.

*   **Stage 1: Backbone Training**
    *   **Goal:** Train the `MambaModel` as a competent language model.
    *   **Process:** Standard next-token prediction using `nn.CrossEntropyLoss`. The value head is not trained (`--no-value-loss`).
    *   **Script:** `crsm/cli.py train`

*   **Stage 2: Dynamics Distillation**
    *   **Goal:** Train the small `LatentDynamics` MLP.
    *   **Process:**
        1.  **Data Collection:** The trained Mamba backbone is used to process a real text corpus (`traces.jsonl`). For each step, it records the tuple `(current_state, action_embedding, next_state)`.
        2.  **Training:** The `LatentDynamics` model is trained on this collected dataset to predict `next_state - current_state` using `nn.MSELoss`.
    *   **Script:** `scripts/distill_dynamics.py` (which uses `crsm/latent_train.py` internally).

*   **Stage 3: CRSM Assembly**
    *   **Goal:** Combine the trained components into a single, functional CRSM.
    *   **Process:** A new `CRSM` instance is created. The weights from the trained backbone (Stage 1) and the trained dynamics model (Stage 2) are loaded into it. This combined model is saved as a new checkpoint.
    *   **Script:** `scripts/train_full_crsm.py` (`create_crsm_with_dynamics` function).

*   **Stage 4: Value Head Fine-tuning**
    *   **Goal:** Train the `value_head` on the Mamba backbone, which is essential for the MCTS planner to evaluate states.
    *   **Process:** The full CRSM model from Stage 3 is loaded. Training is resumed, but this time the loss function includes a component for the value head.
    *   **Script:** `crsm/cli.py train` (resuming from the Stage 3 checkpoint).

## 3. Project History and Debugging

The conversation logs reveal a detailed history of debugging and architectural refinement. Key issues that were identified and fixed include:

*   **Initial "Facade" Implementation:** The first version lacked a true state delta mechanism and had a blocking MCTS loop.
*   **Checkpoint Incompatibility:** A major recurring issue was the incompatibility between checkpoints saved by different pipeline stages and the format expected by the training CLI's `--resume` function. This was the root cause of the "Failed to load checkpoint states" error and was eventually fixed by ensuring all checkpoints were saved in a consistent, CLI-compatible dictionary format.
*   **Performance Bottleneck:** The MCTS loop was initially using the full Mamba backbone for rollouts, leading to extremely slow performance (90+ seconds for 30 tokens). This was fixed by correctly implementing and integrating the fast `LatentDynamics` model in `reasoning.py`.
*   **Vocabulary Mismatches:** Generation produced `<unk>` tokens due to mismatches between the vocabulary size used for training and the one used for generation. This was resolved by implementing a consistent vocabulary building, saving, and loading process.
*   **Numerous API Mismatches:** A long series of `AttributeError`, `TypeError`, and `KeyError` exceptions were fixed, primarily in the generation and testing scripts, to align them with the final model's class structure.

## 4. Current Status and Next Steps

*   **Current Status:** The project is a **functionally complete and validated prototype**. The core architecture is implemented, the training pipeline is operational, and the self-modification engine has been empirically verified to work.
*   **Immediate Next Steps (from conversations):**
    1.  **Improve Data Quality:** The current training corpus is small. The next step is to train on a larger, more diverse dataset (e.g., WikiText).
    2.  **Scale Up:** Train a larger model (e.g., increase `d_model`, `num_layers`) for more epochs to achieve better performance (lower perplexity).
    3.  **Evaluation:** Implement and run a comprehensive evaluation suite to measure perplexity, generation quality (diversity, repetition), and the impact of MCTS on reasoning tasks.
    4.  **Secure Compute:** Apply for compute grants (e.g., `fal.ai`) to facilitate larger-scale training.
*   **Future Roadmap (from `docs/ROADMAP.md`):**
    *   **Phase 2 (Learned Dynamics):** This is now complete.
    *   **Phase 3 (Advanced Training):** Implement RL-based training for the value head, use a learned aggregator for states instead of simple averaging.
    *   **Phase 4 (Scale & Performance):** Multi-GPU training, model quantization.
    *   **Phase 5 (Evaluation):** Benchmark on reasoning tasks like GSM8K and compare against baselines.
