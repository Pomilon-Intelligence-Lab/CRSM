# Contributing to CRSM

Thank you for your interest in contributing to the **Continuous Reasoning State Model (CRSM)**.

Please note that this repositoryand the "Pomilon Intelligence Lab" organizationserves as a personal experimentation ground for hybrid AI architectures. It is currently a solo project maintained by **@Pomilon**. While this is not a formal research institution, the goal is to maintain high-quality, reproducible, and experimental code.

Contributions that help stabilize the architecture, improve performance, or clarify documentation are very welcome.

## Code of Conduct

  * **Respect:** Treat everyone with respect.
  * **Constructive Feedback:** Keep discussions focused on technical improvements.
  * **Experimental Context:** Acknowledge that this is active, experimental software. Breaking changes and instability are part of the process.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
```bash
git clone https://github.com/Pomilon-Intelligence-Lab/CRSM.git
cd CRSM
```
3.  **Create a new branch** for your feature:
```bash
git checkout -b feature/your-feature-name
```

## Development Setup

**Note:** CRSM has specific dependencies regarding CUDA versions due to the Mamba backbone.

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode (includes testing tools)
pip install -e ".[dev]"

# Install pre-commit hooks for auto-formatting
pip install pre-commit
pre-commit install
```

## Making Changes

### Code Style

To keep this project maintainable, strict adherence to code style is required:

  * **Python:** Follow PEP 8 guidelines.
  * **Typing:** Strict type hints (`typing.List`, `torch.Tensor`, etc.) are mandatory. This is critical for managing the complex state within the MCTS planner.
  * **Async:** The Planner executes asynchronously. Avoid blocking the main thread; heavy computation should be offloaded or batched appropriately.

**Format code before committing:**

```bash
black crsm/ tests/
isort crsm/ tests/
```

**Check for linting errors:**

```bash
flake8 crsm/ tests/
```

### Commit Messages

Please write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed, wrapped at 72 characters.
Explain what changed and why.

- Use bullet points for multiple changes
- Reference issues with "Fixes #123"
```

## Testing

Testing CRSM requires attention to the asynchronous planner loop.

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_architecture_stability.py
```

### Writing Tests

  * **Location:** Place new tests in `tests/`.
  * **Async:** Ensure `pytest.mark.asyncio` is used for planner logic.
  * **Mocking:** You **must** mock the large Mamba backbone when testing MCTS logic. Loading full model weights for unit tests is inefficient.

## Pull Request Process

1.  **Sync with main** before starting work to avoid conflicts:

    ```bash
    git fetch origin
    git rebase origin/main
    ```

2.  **Push your branch**:

    ```bash
    git push origin feature/your-feature-name
    ```

3.  **Create a Pull Request** using the template below:

```markdown
## Description
Brief description of the changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Verification
- [ ] Tests pass locally (`pytest`)
- [ ] Code follows style guidelines (`black`, `flake8`)
```

## Areas for Contribution

Since this is a solo project, assistance in the following areas is particularly helpful:

### High Priority

1.  **Planner Optimization (C++/Rust):** The current Python `asyncio` planner encounters GIL contention. Porting the MCTS logic to a compiled extension is a primary goal.
2.  **MCTS Visualization:** Development of tools to visualize the "Tree of Thoughts" generated during inference for debugging.
3.  **Kernel Optimization:** Optimizing the Gated Injection mechanism (custom CUDA kernel) to reduce memory overhead.

### Medium Priority

4.  **Documentation:** Technical tutorials on interpreting "State Deltas."
5.  **Benchmarks:** Scripts to evaluate CRSM on standard reasoning benchmarks (GSM8K, ARC).
6.  **Distillation Pipeline:** Improving the stability of the Dynamics Model training loop.

## Questions?

  * **Technical Issues:** Open an Issue on GitHub.
  * **Architecture Discussion:** Open a GitHub Discussion.
  * **Contact:** Reach out to **@Pomilon**.

Thank you for helping push this experiment forward.