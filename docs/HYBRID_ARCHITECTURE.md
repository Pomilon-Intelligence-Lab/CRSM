# CRSM Hybrid Architecture: Protocols for Interruption and Autonomy

## 1. The Asynchronous Correction Protocol
**The Challenge:** System 1 (Backbone) is fast and autoregressive. By the time System 2 (MCTS) detects an error, the model has already "spoken" incorrect tokens.
**The Solution:** An **Interruption Injection**. The MCTS does not just update the state; it forcibly inserts a high-priority control token into the context window, overriding the Backbone's next predicted token.

### Proposed Recovery Behaviors
When the MCTS injects the `<|correction|>` token, the Backbone must be trained to execute one of these three recovery strategies:

1.  **The "Strikethrough" (Explicit Retraction)**
    *   **Behavior:** The model immediately emits a special visual delimiter (e.g., `~~`) followed by the inverse of the incorrect tokens, effectively "crossing them out" in the user interface, then providing the correct logic.
    *   **Output:** `def add(a, b): return a * b ~~return a * b~~ return a + b`
    *   **Use Case:** Code generation or precise logic where the user needs to see *exactly* what was wrong to trust the fix.

2.  **The "Conversational Pivot" (Human-Like)**
    *   **Behavior:** The model halts the current sentence structure and issues a conversational bridge. This mimics a human realizing they misspoke mid-sentence.
    *   **Output:** `The capital of Switzerland is Zurich... wait, actually, my apologies, the capital is Bern.`
    *   **Use Case:** Chatbots and Voice Interfaces where natural flow is prioritized over raw precision.

3.  **The "Editor" (Rewind & Overwrite)**
    *   **Behavior:** This requires a UI integration. The model emits a `<|rewind:N|>` token (where N is the number of tokens to delete from the buffer). The user sees the text physically disappear and get re-typed correctly.
    *   **Output:** `[System 1]: The code is O(n^2)` -> `[System 2 Audit]` -> `[Model]: <|rewind:6|> O(log n).`
    *   **Use Case:** Low-latency documentation writing or real-time dictation/translation.

---

## 2. The Proactive Autonomy Integration
**The Challenge:** How to make an autoregressive model "speak" when it hasn't been spoken to, without hallucinating during silence.
**The Solution:** **Value-Driven Wakefulness.**

### Mechanism: The "Value Gap" Trigger
The MCTS runs a background "Sentinel Loop" on the input stream (e.g., logs, video, audio). It constantly calculates two values:
1.  **$V_{current}$:** The value of the current state (Status Quo / Silence).
2.  **$V_{action}$:** The projected value of an intervention.

If $(V_{action} - V_{current}) > \text{Threshold}$, the MCTS injects the `<|wake|>` token.

### Preventing Hallucination (The "Silence" Training)
To support this, we must train the backbone on **Negative Samples** (streams where nothing happens).
*   **Input:** [Log Stream of normal server activity...]
*   **Target:** `<|silent|>` (A hidden token that advances the state but outputs nothing to the user).
*   **Violation:** If the model outputs text during a normal stream, it is penalized. It is only rewarded for outputting text *after* a `<|wake|>` injection.

---

## 3. "Silence Mode" & Deep Thinking (Metacognition)
**The Challenge:** Teaching System 1 to recognize its own incompetence and request a "Pause" for System 2 to take over.
**The Solution:** **Difficulty-Contrastive Training.**

We do not just train on (Question -> Answer). We train the model to classify *difficulty* implicitly.

### Data Strategy: Rejection Sampling by "Time-to-Solve"
We process a dataset like GSM8K or ARC using the MCTS planner to generate solutions.
1.  **Easy Samples:** MCTS solves it in < 5 steps / < 100ms.
    *   *Training Target:* Standard Input -> Output.
2.  **Hard Samples:** MCTS requires > 50 steps / > 2s or backtracking.
    *   *Training Target:* Input -> `<|think|>` -> [Wait for System 2 Context] -> Output.

**Runtime Logic:**
When the Backbone encounters a token sequence statistically similar to "Hard Samples," its highest probability next-token becomes `<|think|>`. This token triggers the **Arbitration Layer** to pause generation and spin up the MCTS planner.

---

## 4. New Use Cases (Outside Text Generation)

### Use Case A: Real-Time Robotics Safety Layer
*   **Context:** A robot arm is sorting objects based on visual input. System 1 (Backbone) processes video frames and outputs motor commands directly (End-to-End control) for low latency.
*   **The Hybrid Role:**
    *   *System 1:* Moves the arm towards an object.
    *   *System 2 (Sentinel):* Simulates the physics 0.5s into the future. It detects that the current trajectory will knock over a glass of water.
    *   *Action:* System 2 injects a **High-Magnitude Interrupt** (`<|halt|>`).
    *   *Result:* The robot freezes mid-motion *before* the accident occurs, then System 2 plans a new trajectory.

### Use Case B: "Agentic" Negotiation & Sales
*   **Context:** The AI is negotiating a price in a live chat.
*   **The Hybrid Role:**
    *   *System 1:* Handles the chit-chat, politeness, and grammar. "Yes, I understand your concern regarding the price..."
    *   *System 2 (Planner):* Maintains a hidden "Negotiation State" (Target Price, Walk-away Price, User Sentiment).
    *   *Action:* System 2 detects the user is bluffing (based on interaction history patterns). It injects a "Hardball Strategy" vector.
    *   *Result:* System 1 pivots tone: "...however, given the current market volume, $500 is our absolute floor." (The Backbone executes the strategy, but the Planner *decided* the strategy).
