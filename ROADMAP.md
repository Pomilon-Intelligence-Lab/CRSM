# CRSM Project Roadmap

This document outlines the strategic roadmap for the Continuous Reasoning State Model (CRSM) project, from its current state as a functional prototype to a mature, publishable research project.

## Phase 1: Foundation and Core Implementation (✅ COMPLETE)

This phase involved the initial design, implementation, and debugging of the core CRSM architecture. All tasks in this phase have been successfully completed.

*   **[DONE]** Design the hybrid Mamba + MCTS architecture.
*   **[DONE]** Implement the Mamba backbone with an integrated value head (`crsm/mamba_ssm.py`).
*   **[DONE]** Implement the asynchronous MCTS deliberation loop (`crsm/reasoning.py`).
*   **[DONE]** Implement the core state-delta mechanism for self-modification (`model.py`, `reasoning.py`).
*   **[DONE]** Implement the lightweight `LatentDynamics` model for fast MCTS rollouts (`crsm/latent_dynamics.py`).
*   **[DONE]** Build the complete four-stage training pipeline (`scripts/train_full_crsm.py`).
*   **[DONE]** Debug and validate the end-to-end execution of the training and generation scripts.
*   **[DONE]** Create a comprehensive test suite to verify the self-modification engine (`test_self_modification.py`).

## Phase 2: Initial Training and Validation (🚀 IN PROGRESS)

This phase focuses on moving from a structurally complete prototype to a model that demonstrates meaningful capabilities. The primary goal is to train a baseline model on a respectable dataset and establish initial performance metrics.

*   **Tasks:**
    *   **1. Data Curation and Preprocessing:**
        *   **[IN PROGRESS]** Expand the training corpus beyond the initial small text files.
        *   **[TODO]** Download and preprocess a standard dataset (e.g., a subset of WikiText, SlimPajama, or C4).
        *   **[TODO]** Implement a robust vocabulary generation process with a larger vocabulary size (e.g., 8,000-16,000 tokens) to ensure good coverage of the training data.
    *   **2. Baseline Model Training:**
        *   **[TODO]** Execute the full four-stage training pipeline on the newly curated dataset.
        *   **[TODO]** Monitor training using `wandb` or TensorBoard, tracking LM loss, value head loss, and dynamics model loss.
        *   **[TODO]** Save the final, fully trained checkpoint and all intermediate artifacts (backbone, dynamics model).
    *   **3. Initial Evaluation and Benchmarking:**
        *   **[TODO]** Run the `evaluate_metrics.py` script on a held-out test set to establish baseline perplexity.
        *   **[TODO]** Use the `generate_crsm.py` script to generate text samples and qualitatively assess coherence and reasoning.
        *   **[TODO]** Perform an ablation study by running generation with and without MCTS (`--use-mcts` flag) to get a preliminary measure of the impact of the reasoning module.
    *   **4. Secure Compute Resources:**
        *   **[TODO]** Draft and submit a compute grant proposal to `fal.ai`, using the project's GitHub repository and successful prototype validation as evidence.

*   **Exit Criteria:**
    *   A baseline CRSM model is successfully trained on a standard dataset.
    *   Initial perplexity and generation quality metrics are documented.
    *   A compute grant has been applied for.

## Phase 3: Advanced Training and Architectural Refinements

This phase focuses on improving the model's performance and refining the architecture based on the results from Phase 2.

*   **Tasks:**
    *   **1. Advanced Value Head Training:**
        *   **[TODO]** Implement a Reinforcement Learning (RL) fine-tuning loop (`scripts/rl_value_head_finetuning.py`).
        *   **[TODO]** Use rewards based on generation quality (e.g., coherence, task success) or MCTS statistics to train the value head, instead of just proxy losses.
    *   **2. Hierarchical Dynamics and State Aggregation:**
        *   **[TODO]** Replace the simple averaging in `predict_from_states` with a learned aggregator (e.g., an attention mechanism or a small MLP) to better combine information from different layers of the Mamba state.
        *   **[TODO]** (Optional) Explore training per-layer dynamics models for more granular state predictions.
    *   **3. Hyperparameter Tuning:**
        *   **[TODO]** Use the `hyperparameter_sweep.py` script to systematically tune key parameters (`learning_rate`, `c_puct`, `n_simulations`, delta damping factor).

*   **Exit Criteria:**
    *   Demonstrable improvement in perplexity and/or reasoning task performance over the baseline model.
    *   The RL-trained value head shows better MCTS guidance.

## Phase 4: Scaling and Benchmarking

This phase focuses on scaling the model and performing rigorous, comparative benchmarks to prove the CRSM's advantages.

*   **Tasks:**
    *   **1. Scale Model and Data:**
        *   **[TODO]** Train a larger CRSM (e.g., 1B+ parameters) on a large-scale dataset, utilizing the secured compute resources.
        *   **[TODO]** Implement and test multi-GPU training (DDP) for efficient scaling.
    *   **2. Rigorous Benchmarking:**
        *   **[TODO]** Evaluate the scaled model on standard reasoning benchmarks (e.g., GSM8K, MATH, ARC).
        *   **[TODO]** Conduct a formal ablation study comparing the full CRSM against:
            *   A baseline Mamba model of the same size (MCTS disabled).
            *   A standard Transformer model of a similar size.
        *   **[TODO]** Measure and compare latency, throughput, and computational cost for generation.

*   **Exit Criteria:**
    *   A large-scale CRSM model is successfully trained.
    *   Benchmark results are collected and show a clear advantage for the CRSM architecture on specific tasks (e.g., better reasoning, lower latency for planning).

## Phase 5: Publication and Release

This final phase focuses on sharing the research and the model with the community.

*   **Tasks:**
    *   **1. Write Research Paper:**
        *   **[TODO]** Draft a paper detailing the CRSM architecture, the state delta mechanism, the training methodology, and the experimental results from Phase 4.
        *   **[TODO]** Submit the paper to a relevant AI conference (e.g., NeurIPS, ICML, ICLR) or as an arXiv preprint.
    *   **2. Open Source Release:**
        *   **[TODO]** Clean up the codebase and ensure all documentation (`README.md`, `USAGE.md`, `ARCHITECTURE.md`) is complete and up-to-date.
        *   **[TODO]** Publish the trained model weights for the best-performing CRSM on the Hugging Face Hub.
        *   **[TODO]** Create a simple Gradio or Streamlit demo to allow users to interact with the model.
        *   **[TODO]** Announce the project and paper on relevant social media and forums.

*   **Exit Criteria:**
    *   Research paper is published.
    *   Model, code, and demo are publicly available.
