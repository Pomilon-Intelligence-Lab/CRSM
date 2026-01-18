# CRSM: Continuous Reasoning State Model

> ⚠️ **EXPERIMENTAL PROTOTYPE**
> A research project testing if a background search process (MCTS) can steer a Mamba model's hidden states in real-time without stopping generation.

## The Core Concept: "Thinking" Without Tokens

Standard models face a trade-off: to "think" deeply (System 2), they must write out a "Chain of Thought." This is slow and consumes tokens.

**CRSM** moves this process into the background. It uses a **Mamba** backbone (fast generation) and an **Asynchronous MCTS** planner (look-ahead search). Instead of writing text, the planner directly edits the model's internal memory (hidden states) as it types.

---

## 🛠 Architectural Breakdown

| Component | What it actually is |
| --- | --- |
| **System 1 Backbone** | A standard **Mamba SSM**. It handles fast, instinctive text generation. |
| **System 2 Planner** | An **MCTS** loop running on a separate thread/process to simulate future paths. |
| **State Blending** | A function that mixes the model's current state with the planner's "better" state. |
| **Latency Buffer** | It "fast-forwards" the planner's advice to match the model's current token position. |
| **Consensus Heads** | A committee of **Value Heads** that decide if a reasoning path is actually stable. |

---

## 🔬 Core Mechanics

### 1. Weighted State Blending (Layer-by-Layer)

We don't force-feed the model. Instead, we use a simple "mix" ratio () at each layer to determine how much the planner's advice should override the model's intuition.


* **High Influence ():** Higher-level strategy layers are heavily corrected by the planner.
* **Low Influence ():** Lower-level syntax and grammar layers are left untouched to ensure fluid text.

### 2. The Latency Buffer (Forward Projection)

Planning takes time. By the time the search loop finds a better state, the generator is already 3 tokens ahead. CRSM uses a lightweight **Dynamics Model** to "fast-forward" the planner’s target state so it aligns perfectly with the model’s current position when the update is applied.

### 3. Uncertainty Penalties

If the different layers of the model disagree on whether a path is good, the system applies an **Uncertainty Penalty**. This effectively tells the planner: "If the layers aren't in consensus, don't change the state," preventing the search from accidentally breaking the model's logic.

---

## 📚 References & Inspirations

### Core Foundations

* **Mamba (SSM):** Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)
* **Latent Planning (MuZero):** Schrittwieser, J., et al. (2019). *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model.* [arXiv:1911.08265](https://arxiv.org/abs/1911.08265)
* **System 1 & 2:** Kahneman, D. (2011). *Thinking, Fast and Slow.*

### State Steering & Reasoning

* **Chain of Continuous Thought (Coconut):** Hao, S., et al. (2024). *Training Language Models to Reason in a Continuous Latent Space.* [arXiv:2412.06769](https://arxiv.org/abs/2412.06769)
* **Representation Engineering (RepE):** Zou, A., et al. (2023). *Representation Engineering: A Top-Down Approach to AI Transparency.* [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)
* **Quiet-STaR:** Zelikman, E., et al. (2024). *Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking.* [arXiv:2403.09629](https://arxiv.org/abs/2403.09629)

### Technical Mechanics

* **Speculative Decoding:** Leviathan, Y., et al. (2022). *Fast Inference from Transformers via Predictive Sampling.* [arXiv:2211.17192](https://arxiv.org/abs/2211.17192)
* **JEPA Architecture:** LeCun, Y. (2023). *A Path Towards Autonomous Machine Intelligence.* [Meta AI Research](https://ai.meta.com/blog/yann-lecun-ai-model-i-jepa/)
* **Soft Target Updates:** Lillicrap, T. P., et al. (2015). *Continuous Control with Deep Reinforcement Learning.* [arXiv:1509.02971](https://arxiv.org/abs/1509.02971)

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/Pomilon-Intelligence-Lab/CRSM.git
cd CRSM && pip install -e .

```

### Inference with Deliberation

```python
output = await model.think_and_generate(
    prompt, 
    max_length=50, 
    use_deliberation=True, # Activates the background search
    deliberation_lag=3     # Latency buffer window
)

```

---

## License

MIT License. **Pomilon Intelligence Lab (2025)**
