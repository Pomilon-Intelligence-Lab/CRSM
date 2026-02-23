# CRSM: Continuous Reasoning State Model

> ⚠️ **STATUS: EXPERIMENTAL PROTOTYPE**
> This is a research experiment exploring whether a continuous background planner can guide a language model without pausing generation. While the core "Gated State Injection" mathematics have been verified for stability, the model is currently a proof-of-concept.

## The Core Concept: "Thinking" Without Tokens

Standard Transformers typically face a latency trade-off: to perform "System 2" reasoning (deep planning), they must generate intermediate tokens ("System 1" output) by writing out a "Chain of Thought." This increases latency and computational cost.

**CRSM** explores an alternative approach: decoupling reasoning from generation. It moves this process into the background by combining a **Mamba** backbone (efficient linear-time memory) with an **Asynchronous MCTS** planner. Instead of writing text, the planner directly edits the model's internal memory (hidden states) as it types.

### 🎯 ARC-AGI Focus

The project is currently optimized for benchmarking on **ARC-AGI**, targeting **Nano-scale implementations (100k - 500k parameters)**. The modular architecture allows for rapid iteration on different reasoning strategies and task-specific logic (e.g., Grid-based spatial reasoning).

---

## Documentation Hub

* **[Architecture Deep Dive](https://www.google.com/search?q=docs/ARCHITECTURE.md):** A detailed look at the Backbone, Dynamics, and Planner integration.
* **[Visual Architecture Diagram](https://www.google.com/search?q=docs/ARCHITECTURE_DIAGRAM.md):** Schematic of the asynchronous interaction loop.
* **[Technical Retrospective](https://www.google.com/search?q=docs/technical_report.md):** An informal discussion on the engineering challenges and lessons learned.
* **[Project Roadmap](https://www.google.com/search?q=docs/ROADMAP.md):** Current capabilities and future research goals.

---

## Architectural Breakdown

| Component | What it actually is |
| --- | --- |
| **System 1 Backbone** | A standard **Mamba SSM**. It handles fast, instinctive text generation. |
| **System 2 Planner** | An **MCTS** loop running on a separate thread/process to simulate future paths. |
| **State Blending** | A function that mixes the model's current state with the planner's "better" state. |
| **Latency Buffer** | It "fast-forwards" the planner's advice to match the model's current token position. |
| **Consensus Heads** | A committee of **Value Heads** that decide if a reasoning path is actually stable. |

---

## Core Mechanics & Key Experiments

### 1. Sparse-Gated Hierarchical Injection (Weighted Blending)

We don't force-feed the model. To maintain stability while allowing deep planning, CRSM uses **Sparse-Gated Injection**. Each layer in the Mamba hierarchy is treated as a sovereign entity with its own "gate." The planner injects state updates independently into each layer based on a simple "mix" ratio.

* **High Influence:** The planner aggressively updates high-level strategy layers.
* **Low Influence:** Lower-level sensory/syntax layers are left untouched to ensure fluid text.

### 2. Forward-Projected Planning (The Latency Buffer)

Planning takes time. In an asynchronous system, by the time MCTS finds a better state, the generator is already 3 tokens ahead. CRSM solves this via **Forward Projection**. The planner uses its internal dynamics model to "fast-forward" the current state to the target position before starting the search. Updates are held in a **Targeted Delta Buffer** and applied at the exact micro-second they align with the generation loop.

### 3. Multi-Headed Consensus & Uncertainty Penalties

Instead of a single "Value Head," CRSM employs a **Multi-Headed Value Critic (MV-Critic)**, one for every layer. The planner's utility score is a weighted consensus of these heads. If the different layers disagree on whether a path is good (high variance in values), the system applies an **Uncertainty Penalty**. This effectively tells the planner: "If the layers aren't in consensus, don't change the state," preventing the search from accidentally breaking the model's logic.

---

## Quick Start

### Installation

```bash
git clone https://github.com/Pomilon-Intelligence-Lab/CRSM.git
cd CRSM
pip install -e .

```

### Autonomous Inference

Run the model with the "Thinking" loop active:

```python
import torch
import asyncio
from crsm.core import CRSMModel, CRSMConfig

async def main():
    # 1. Load Model (Nano configuration)
    config = CRSMConfig(
        vocab_size=1024, 
        hidden_size=256, 
        num_hidden_layers=4,
        injection_rate=0.05
    )
    model = CRSMModel(config).cuda()
    
    # 2. Generate with Asynchronous Deliberation
    prompt = torch.tensor([[10, 20, 30]]).cuda()
    
    output = await model.think_and_generate(
        prompt, 
        max_length=50, 
        use_deliberation=True,
        deliberation_lag=3 # Latency buffer window
    )
    print("Generated:", output)

if __name__ == "__main__":
    asyncio.run(main())

```

---

## Usage, Benchmarking & Training

### Unified Benchmarking & Validation

The central tool for verifying both functional and operational validity is `scripts/eval/benchmark.py`. It automates backbone training, subconscious reasoning training, and ablation studies.

* **Synthetic Sanity Check (Fast):** Verify the architecture can learn identity and simple translations.
```bash
python scripts/eval/benchmark.py --config configs/arc_nano.yaml --type sanity

```


* **Official ARC-AGI Benchmark:** Run the full pipeline on the official fchollet/ARC dataset.
```bash
python scripts/eval/benchmark.py --config configs/arc_official.yaml --type official

```



### Understanding Operational Proofs

The benchmark reports two critical signals of "Working Reasoning":

* **Discrimination Accuracy:** Measures if the Multi-Headed Value Critic can distinguish correct states from noisy ones. An accuracy > 50% proves the subconscious is **learning to judge**.
* **MCTS Improvement Delta:** The performance gain of MCTS over Greedy search. A positive delta proves the search engine is **operationally steering** the model towards better solutions.

### Modular Training

To train a model on general tasks using the modular trainer:

```bash
python run.py --task lm --config configs/training_config.yaml

```

---

## Verification

The repository includes a test suite to verify the stability of the state injection math and the functionality of the components.

* **Architecture Stability:** `tests/test_architecture_stability.py` (Verifies Gated Injection properties).
* **Capabilities:** `tests/verify_capabilities.py` (Basic capability checks).

Run the core stability verification:

```bash
python tests/test_architecture_stability.py

```

---

## Project Origins & Transparency

This project follows a **"Centaur" workflow**—combining human direction and engineering with AI-assisted research.

* **The Spark:** The core concept—replacing linear token-based planning with a continuous "thinking module"—originated from a research session I conducted with **Gemini 2.5 Flash**.
* **Foundational Research:** The initial feasibility study and architectural concepts were generated by AI and are preserved in `docs/FOUNDATIONAL_RESEARCH.md`.
* **Implementation:** I utilized LLMs (ChatGPT, Claude, Gemini) to assist in drafting complex component code.
* **Verification & Engineering:** I personally handled the system integration, testing, debugging, and critical mathematical verification (such as the "Gated Injection" solution).

I believe this transparency is important to accurately represent the collaborative nature of modern experimental coding.

---

## Inspirations & Acknowledgements

This project is an experimental synthesis of existing breakthrough research. I claim no credit for the foundational concepts, only for the specific implementation of their integration (CRSM).

### Core Theoretical Foundations & Frameworks

* **MuZero (Schrittwieser et al., DeepMind):** The primary inspiration for performing MCTS entirely within a learned latent space, without decoding back to observations.
* **Mamba (Gu & Dao):** The efficient SSM backbone whose fixed-size, linear-time state enables the direct state manipulation that would be prohibitive with the KV-cache of Transformers.
* **Tree of Thoughts (Yao et al.) & Chain of Thought (Wei et al.):** The inspiration for treating reasoning as a search problem over a space of intermediate steps.
* **System 1 & System 2 (Daniel Kahneman):** The guiding conceptual framework where System 1 is the Mamba backbone and System 2 is the Asynchronous MCTS planner.
* **Global Workspace Theory (Baars):** The idea of a "working memory" inspired the design of the Latent State as a shared workspace.

### Architectural Mechanics & Methodologies

* **Coconut (Chain of Continuous Thought):** A parallel line of research exploring reasoning in continuous latent space.
* **JEPA (LeCun):** Influenced the design of the Latent Dynamics Model by learning to predict representations rather than pixel/token details.
* **World Models / Dreamer (Ha & Schmidhuber, Hafner et al.):** The concept of learning a compact model of the environment to simulate futures for planning.
* **State Delta Communication (Tang et al.):** Explores using "state deltas" to convey reasoning dynamics.
* **Representation Engineering (RepE) (Zou et al.):** The concept of "Top-Down" control of model behavior by manipulating the latent space.
* **Reasoning via Planning (RAP) (Hao et al.) & AlphaLLM:** Pioneered the integration of MCTS with Large Language Models.
* **Plug and Play Language Models (PPLM) & Activation Addition:** Established the foundation for steering model generation by modifying hidden states.
* **RLHF (Christiano et al. / OpenAI):** The methodology of training a separate Value Head to estimate the utility of a language model's state.
* **Speculative Decoding (Leviathan et al.):** Shares DNA with CRSM's asynchronous "draft-then-verify" computational pattern.
* **Polyak Averaging (Lillicrap et al.):** The Gated Injection formula is mathematically identical to the "soft target updates" used in RL algorithms like DDPG.
* **Quiet-STaR (Zelikman et al.):** Explores generating "internal thoughts" at every token step to improve reasoning.

I am deeply grateful to the researchers behind these works for sharing their code and insights with the open-source community.

---

## Reference

```bibtex
@software{crsm2025,
  title = {CRSM: Continuous Reasoning State Model},
  author = {Pomilon},
  year = {2025},
  url = {https://github.com/Pomilon-Intelligence-Lab/CRSM}
}

```

## License

MIT License. **Pomilon Intelligence Lab (2025)**