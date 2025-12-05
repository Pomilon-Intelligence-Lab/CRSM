# CRSM: Continuous Reasoning State Model

> ⚠️ **STATUS: EXPERIMENTAL PROTOTYPE**
> This is a research experiment exploring whether a continuous background planner can guide a language model without pausing generation. While the core "Gated State Injection" mathematics have been verified for stability, the model is currently a proof-of-concept.

**Exploring Asynchronous "System 2" Reasoning with Mamba**

Standard Transformers typically face a latency trade-off: to perform "System 2" reasoning (deep planning), they must generate intermediate tokens ("System 1" output), which increases latency and computational cost.

**CRSM** explores an alternative approach: decoupling reasoning from generation. It combines a **Mamba** backbone (efficient linear-time memory) with an **Asynchronous MCTS** planner. This "thinking module" runs in the background, exploring future possibilities and injecting "thought vectors" into the main model's state in real-time.

---

## 📚 Documentation Hub

*   **[Architecture Deep Dive](docs/ARCHITECTURE.md)**: A detailed look at the Backbone, Dynamics, and Planner integration.
*   **[Visual Architecture Diagram](docs/ARCHITECTURE_DIAGRAM.md)**: Schematic of the asynchronous interaction loop.
*   **[Technical Retrospective](docs/technical_report.md)**: An informal discussion on the engineering challenges and lessons learned.
*   **[Usage & Training Guide](docs/USAGE.md)**: Instructions for inference and the multi-stage training pipeline.
*   **[Installation Guide](docs/INSTALL.md)**: Environment setup.
*   **[Project Roadmap](docs/ROADMAP.md)**: Current capabilities and future research goals.

---

## 💡 Key Architectural Experiments

### 1. Gated State Injection (Stability Control)
Directly modifying a model's high-dimensional latent state often leads to "state explosion."
The proposed mechanism is **Gated Injection**:
$$h_{t} \leftarrow (1 - \alpha_{eff}) \cdot h_{t} + \alpha_{eff} \cdot h_{target}$$
This acts as a "low-pass filter" for thoughts. It allows the planner to gently nudge the model's intuition towards a better trajectory without disrupting the stability of the underlying state manifold.

### 2. Asynchronous Deliberation Loop
The planner executes in a background thread (`asyncio`) using a distilled **Latent Dynamics Model**. This lightweight neural network predicts state transitions, allowing the planner to simulate thousands of future steps quickly. This design intends to allow the main model to maintain fluent generation while "thinking" occurs in parallel.

### 3. Dynamic Confidence Scaling
To prevent an uncertain planner from degrading the model's output, the injection rate ($\alpha$) is scaled by the planner's confidence. If the MCTS Value Head is unsure, the system defaults back to the pure, stable Mamba backbone.

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/Pomilon-Intelligence-Lab/CRSM.git
cd CRSM
pip install -e .
```

### Autonomous Inference

> **Note:** This repository contains the architecture and training code. Pre-trained weights are not yet available (see Roadmap). Running inference now will use an untrained model with random weights, producing random tokens.

Run the model with the "Thinking" loop active:

```python
import torch
import asyncio
from crsm.model import CRSMModel, CRSMConfig

async def main():
    # 1. Load Model (0.05 injection rate is the current experimental sweet spot)
    config = CRSMConfig(vocab_size=50257, injection_rate=0.05)
    model = CRSMModel(config).cuda()
    
    # 2. Generate with Asynchronous Deliberation
    prompt = torch.tensor([[502, 10, 99]]).cuda()
    
    output = await model.crsm.think_and_generate(
        prompt, 
        max_length=100, 
        use_deliberation=True,  # <--- Activates the background MCTS thread
        deliberation_lag=3      # Plan 3 tokens into the future
    )
    print("Generated:", output)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛠️ Training Pipeline

The architecture requires a 4-stage training pipeline to function correctly (orchestrated by `scripts/training/train_full_crsm.py`):

1.  **Backbone Pre-training**: Standard CLM training for the Mamba model.
2.  **Dynamics Distillation**: Training a small MLP to predict state transitions ($h_t \to h_{t+1}$) from the frozen backbone.
3.  **Assembly**: Integrating the backbone and dynamics model.
4.  **Value Head Fine-tuning**: Training the planner's value estimator to recognize high-quality states.

To run the full pipeline on a small baseline:
```bash
python scripts/training/train_full_crsm.py --config configs/baseline_27m.json
```

---

## 🧪 Verification

The repository includes a test suite to verify the stability of the state injection math and the functionality of the components.

*   **Architecture Stability**: `tests/test_architecture_stability.py` (Verifies Gated Injection properties).
*   **Capabilities**: `tests/verify_capabilities.py` (Basic capability checks).

Run the core stability verification:
```bash
python tests/test_architecture_stability.py
```

---

## 🧬 Project Origins & Transparency

This project follows a **"Centaur" workflow**—combining human direction and engineering with AI-assisted research.

**The Spark:**
The core concept—replacing linear token-based planning with a continuous "thinking module"—originated from a research session I conducted with **Gemini 2.5 Flash**.

**Original Prompt:**
> "Help me research ways to develop the next SOTA open-source mode. My idea is that instead of relying on architectures like Transformers, which just predict linearly the next token in a sequence and thinks in the tokens that it generates... we could develop a new architecture that instead includes an internal reasoning component or a thinking module..."

**Development Process:**
*   **Foundational Research:** The initial feasibility study and architectural concepts were generated by AI and are preserved in `docs/FOUNDATIONAL_RESEARCH.md`.
*   **Implementation:** I utilized LLMs (ChatGPT, Claude, Gemini) to assist in drafting complex component code.
*   **Verification & Engineering:** I personally handled the system integration, testing, debugging, and critical mathematical verification (such as the "Gated Injection" solution).

I believe this transparency is important to accurately represent the collaborative nature of modern experimental coding.

---

## 🧠 Inspirations & Acknowledgements

This project is an experimental synthesis of existing breakthrough research. It attempts to combine these distinct ideas into a unified architecture. I claim no credit for the foundational concepts, only for the specific implementation of their integration (CRSM).

### Core Theoretical Foundations
*   **[MuZero (Schrittwieser et al., DeepMind)](https://arxiv.org/abs/1911.08265)**: The primary inspiration for performing **Monte Carlo Tree Search (MCTS)** entirely within a learned **latent space**, without decoding back to observations. CRSM adapts this "planning in latent space" concept to the continuous state of a language model.
*   **[Mamba (Gu & Dao)](https://arxiv.org/abs/2312.00752)**: The efficient **State Space Model (SSM)** backbone is the engine of this architecture. Its fixed-size, linear-time state enables the direct state manipulation and injection that would be computationally prohibitive with the KV-cache of Transformers.
*   **[Tree of Thoughts (Yao et al.)](https://arxiv.org/abs/2305.10601)** & **[Chain of Thought (Wei et al.)](https://arxiv.org/abs/2201.11903)**: The inspiration for treating reasoning as a search problem over a space of intermediate steps. CRSM attempts to make this search internal and continuous rather than external and discrete.

### Cognitive Frameworks
*   **[System 1 & System 2 (Daniel Kahneman)](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)**: The guiding conceptual framework.
    *   **System 1 (Intuition):** Represented by the Mamba backbone (fast, heuristic generation).
    *   **System 2 (Deliberation):** Represented by the Asynchronous MCTS planner (slow, logical search).
*   **[Global Workspace Theory (Baars)](https://en.wikipedia.org/wiki/Global_workspace_theory)**: The idea of a "working memory" where conscious processing occurs inspired the design of the **Latent State** as a shared workspace that both the planner and generator can access and modify.

### Emerging Research
*   **[Coconut (Chain of Continuous Thought)](https://arxiv.org/abs/2412.06769)**: A parallel line of research exploring reasoning in continuous latent space. While Coconut feeds the last hidden state back as input to the next step, CRSM modifies the internal state directly in real-time during the generation process.

### Architectural Components
*   **[JEPA (LeCun)](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)**: The design of the **Latent Dynamics Model** is heavily influenced by Joint Embedding Predictive Architectures—learning to predict the representation of the next state rather than the pixel/token details.
*   **[World Models / Dreamer (Ha & Schmidhuber, Hafner et al.)](https://arxiv.org/abs/1803.10122)**: The concept of learning a compact model of the environment to simulate futures ("dreaming") for planning is directly implemented in CRSM's dynamics distillation pipeline.

### Related Mechanics
*   **[State Delta Communication (Tang et al.)](https://aclanthology.org/2025.emnlp-main.518/)**: While CRSM uses "state deltas" for *intra-agent* self-correction (Planner → Backbone), Tang et al. explore a similar mechanic for *inter-agent* communication, passing "state deltas" between models to convey reasoning dynamics that are lost in discrete token communication.

### Related Methodologies
*   **[Representation Engineering (RepE) (Zou et al.)](https://arxiv.org/abs/2310.01405)**: The concept of "Top-Down" control of model behavior by manipulating the latent space is central to CRSM. Our "Gated Injection" can be viewed as a control-theoretic application of RepE, where the control vector is dynamically generated by the planner rather than a static concept vector.
*   **[Reasoning via Planning (RAP) (Hao et al.)](https://arxiv.org/abs/2305.14992)** & **[AlphaLLM (Tencent AI Lab)](https://arxiv.org/abs/2404.12253v1)**: These works pioneered the integration of MCTS with Large Language Models to enable self-improvement and strategic planning. CRSM builds on this by moving the planning process into the *asynchronous* and *continuous* domain.
*   **[Plug and Play Language Models (PPLM) (Dathathri et al.)](https://arxiv.org/abs/1912.02164)** & **[Activation Addition (Turner et al.)](https://arxiv.org/abs/2308.10248)**: These works established the foundation for steering model generation by modifying hidden states (via gradients or vector addition). CRSM extends this by using a *dynamic planner* to generate the steering vectors in real-time, rather than using static vectors or classifiers.
*   **[RLHF (Christiano et al. / OpenAI)](https://arxiv.org/abs/1706.03741)**: The methodology of training a separate **Value Head** to estimate the utility of a language model's state is adapted directly from the foundational work on Reinforcement Learning from Human Feedback.

### Mathematical & Engineering Parallels
*   **[Speculative Decoding (Leviathan et al.)](https://arxiv.org/abs/2211.17192)**: The "draft-then-verify" computational pattern in speculative decoding shares DNA with CRSM's asynchronous design. In CRSM, the "dynamics model" acts as a latent drafter, while the MCTS planner acts as a verifier/improver running in parallel.
*   **[Polyak Averaging (Lillicrap et al.)](https://arxiv.org/abs/1509.02971)**: The Gated Injection formula ($h_{new} = (1-\tau)h + \tau h_{target}$) is mathematically identical to the "soft target updates" used in DDPG and other RL algorithms. We apply this standard control-theory technique to maintain stability in the language model's latent manifold.
*   **[Quiet-STaR (Zelikman et al.)](https://arxiv.org/abs/2403.09629)**: This work explores generating "internal thoughts" at every token step to improve reasoning. CRSM shares this goal but seeks to make these thoughts continuous and asynchronous rather than discrete and interleaved.

I am deeply grateful to the researchers behind these works for sharing their code and insights with the open-source community.

---

## Reference (If you find this useful)

```bibtex
@software{crsm2025,
  title = {CRSM: Continuous Reasoning State Model},
  author = {Pomilon},
  year = {2025},
  url = {https://github.com/Pomilon-Intelligence-Lab/CRSM}
}
```

## License

MIT License.