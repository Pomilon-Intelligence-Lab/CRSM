# CRSM: An Informal Technical Report
**Subtitle:** Trying to give Mamba a "Subconscious"

## 1. The Big Idea
Look, everyone knows LLMs are basically just fancy "next token predictors". They don't really plan; they just ramble based on statistics. If they start a sentence with a bad assumption, they're stuck with it.

The idea behind CRSM (Continuous Reasoning State Model) was pretty simple: **What if we gave the model a background process—a "subconscious"—that thinks ahead while the model speaks?**

We wanted to separate the "fast" talking part (System 1) from the "slow" thinking part (System 2), just like how humans (supposedly) think.

## 2. The Architecture (How we built it)

### The "Brain": Mamba
We picked **Mamba** (a State Space Model) instead of a Transformer. Why? Because Transformers have a Key-Value cache that grows huge and is hard to mess with. Mamba has a fixed-size latent state `h`. It's just a tensor. We figured: *"Hey, if we modify `h` directly, can we change the model's mind?"*

### The "Planner": Async MCTS
We attached a **Monte Carlo Tree Search (MCTS)** engine to run in the background. While Mamba is generating tokens `t, t+1, t+2`, the MCTS engine grabs the state at `t`, pauses, and simulates 50 different futures. It looks for a path that leads to a "better" outcome.

To make this fast, we trained a tiny **Dynamics Model** (a small neural net) that simulates Mamba. It's like a low-res map of the territory, so the planner can run quickly without needing the full heavy model for every step.

## 3. The "Brain Surgery" Problem
Our first attempt was obvious: just add the planner's result to the model's state.
`state = state + thought_vector`

**Result:** Absolute chaos. The model started outputting garbage.
It turns out Mamba states are incredibly sensitive mathematical manifolds (like polynomial coefficients). You can't just add random numbers to them without breaking the history.

### The Fix: Gated Injection
We switched to a **Gated** approach. Instead of adding, we *interpolate*.
`new_state = (95% * old_state) + (5% * target_state)`

This worked! It acts like a "low-pass filter" for thoughts. The planner gently nudges the model's intuition towards the better path instead of shoving it off a cliff.

## 4. Current Status (What actually works)
*   **Stability:** We ran a stress test injecting 1000 thoughts in a row. The model didn't explode. The math holds up.
*   **Safety:** We added a "Confidence" check. If the planner isn't sure (because it's untrained or confused), we scale the injection down to 0. This prevents a dumb planner from ruining a smart model.
*   **Proof of Concept:** We verified that *if* the planner finds a better state, the injection mechanism *does* reduce the loss.

## 5. The "Real Talk" Limitations
Let's be honest about the current bottlenecks:

1.  **Python is Slow:** We use Python threads (`asyncio`). They aren't *truly* parallel because of the Global Interpreter Lock (GIL). The "background" thinking sometimes stutters the "foreground" speaking. A production version would need the planner written in C++ or Rust.
2.  **VRAM is Tight:** Storing a tree of states eats memory. We optimized this by re-computing states on the fly (trading compute for memory), but it's still heavy.
3.  **The "Chicken and Egg" Problem:** The planner needs a "Value Head" to tell it what a good state looks like. But the Value Head needs data from the planner to learn. Right now, until we train for thousands of steps, the planner is basically guessing.

## 6. Conclusion
CRSM isn't a solved product; it's an experiment. We proved that you *can* inject state modifications into Mamba without breaking it, and that you *can* run a planner in parallel. The next step is just brute-force training to make that planner actually smart.
