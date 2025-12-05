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
My solution is **Gated Injection**:
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