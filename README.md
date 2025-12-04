# CRSM: Continuous Reasoning State Model

> ⚠️ **STATUS: ARCHITECTURE VERIFIED**
> The core "Gated State Injection" mechanism has been mathematically verified to prevent state explosion while ensuring guaranteed loss reduction when the planner finds better states. We are now in the large-scale training phase.

**A Hybrid Neuro-Algorithmic Architecture for Autonomous Reasoning**

The **Continuous Reasoning State Model (CRSM)** overcomes the fundamental "Thinking takes Latency" bottleneck of standard Transformers. By decoupling reasoning from token generation, CRSM allows a model to "think" (plan) and "speak" (generate) simultaneously.

It combines a **Mamba (State Space Model)** backbone for linear-time context processing with an **Asynchronous Monte Carlo Tree Search (MCTS)** planner for deep lookahead, fused together by a novel **Gated State Injection** mechanism.

---

## 📚 Documentation Hub

Navigate the detailed documentation to understand the system:

*   **[Architecture Deep Dive](docs/ARCHITECTURE.md)**: Detailed breakdown of the components (Backbone, Dynamics, Planner) and the Gated Injection math.
*   **[Visual Architecture Diagram](docs/ARCHITECTURE_DIAGRAM.md)**: A schematic view of the "System 1" (Fast) and "System 2" (Slow) interaction.
*   **[Research Paper](docs/research_paper.md)**: The theoretical framework, motivations, and design philosophy behind CRSM.
*   **[Usage & Training Guide](docs/USAGE.md)**: Practical instructions for running inference, training the backbone, and fine-tuning the value head.
*   **[Installation Guide](docs/INSTALL.md)**: Detailed environment setup and dependency management.
*   **[Project Roadmap](docs/ROADMAP.md)**: Current status, completed milestones, and future research directions.

---

## 💡 Core Innovations

### 1. Gated State Injection (Stability Solved)
Unlike standard RLHF which updates weights offline, CRSM updates its **latent state** online. Early experiments showed that simply adding thought vectors caused "state explosion." We solved this with **Gated Injection**:
$$h_{t} \leftarrow (1 - \alpha_{eff}) \cdot h_{t} + \alpha_{eff} \cdot h_{target}$$
This mechanism acts as a "low-pass filter" for thoughts, mathematically guaranteeing manifold stability while allowing the planner to guide the model's intuition.

### 2. Asynchronous "System 2"
The MCTS planner runs in a background thread (`asyncio`), performing rollouts using a distilled **Latent Dynamics Model**. This allows the model to generate tokens at full speed while the planner continuously refines the state in the background, injecting corrections only when high-confidence plans are found.

### 3. Confidence Scaling
The impact of the planner is dynamic. If the MCTS Value Head is unsure (low confidence), the injection rate ($\alpha$) drops to near zero, preventing an untrained planner from "lobotomizing" the coherent language model.

---

## ⚡ Quick Start

### Installation

```bash
git clone https://github.com/pomilon/CRSM.git
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
    # 1. Load Model (0.05 injection rate is the verified sweet spot)
    config = CRSMConfig(vocab_size=50257, injection_rate=0.05)
    model = CRSMModel(config).cuda()
    
    # 2. Generate with Asynchronous Deliberation
    prompt = torch.tensor([[502, 10, 99]]).cuda()
    
    output = await model.crsm.think_and_generate(
        prompt, 
        max_length=100, 
        use_deliberation=True,  # <--- Activates MCTS
        deliberation_lag=3      # Plan 3 tokens into the future
    )
    print("Generated:", output)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🛠️ Training Pipeline

CRSM requires a 4-stage training process (orchestrated by `scripts/training/train_full_crsm.py`):

1.  **Backbone Pre-training**: Train Mamba on text (CLM objective).
2.  **Dynamics Distillation**: Train the lightweight MLP to predict state transitions ($h_t \to h_{t+1}$). 
3.  **Assembly**: Combine Backbone + Dynamics into a unified CRSM checkpoint.
4.  **Value Head Fine-tuning**: Train the MCTS Value estimator to predict future loss/reward.

To run the full pipeline on a small baseline:
```bash
python scripts/training/train_full_crsm.py --config configs/baseline_27m.json
```

---

## 🧪 Verification

We include a rigorous test suite to ensure the complex architecture is behaving correctly.

*   **Architecture Stability**: `tests/test_architecture_stability.py` (Verifies the math of Gated Injection).
*   **Capabilities**: `tests/verify_capabilities.py` (Checks if MCTS improves reasoning on toy tasks).

Run the core stability proof:
```bash
python tests/test_architecture_stability.py
```

---

## Citation

```bibtex
@software{crsm2025,
  title = {CRSM: Continuous Reasoning State Model},
  author = {Pomilon},
  year = {2025},
  url = {https://github.com/pomilon/CRSM}
}
```

## License

MIT License.
