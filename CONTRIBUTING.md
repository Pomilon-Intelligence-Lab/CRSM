# Contributing to CRSM

Thank you for your interest in contributing to CRSM! This document provides guidelines and instructions for contributing.

## Code of Conduct

- Be respectful and inclusive
- Welcome diverse perspectives
- Focus on constructive feedback
- Keep discussions professional

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/CRSM.git
   cd CRSM
   ```
3. **Create a new branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Making Changes

### Code Style

- Follow PEP 8
- Use type hints where possible
- Max line length: 100 characters
- Use meaningful variable names

Format code with:
```bash
black crsm/ tests/ examples/
isort crsm/ tests/ examples/
```

Check with:
```bash
flake8 crsm/ tests/
```

### Commit Messages

Write clear, descriptive commit messages:

```
Short summary (50 chars or less)

More detailed explanation if needed, wrapped at 72 characters.
Explain what changed and why, not how.

- Use bullet points for multiple changes
- Reference issues with "Fixes #123"
```

Examples:
```
Add efficient S4 layer implementation
Refactor dataset loading for streaming support
Fix CUDA memory leak in training loop
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_crsm.py

# Run with coverage
pytest tests/ --cov=crsm --cov-report=html
```

### Writing Tests

- Place new tests in `tests/`
- Use descriptive test function names: `test_module_functionality`
- Include docstrings explaining what's tested
- Test both success and failure cases

Example:
```python
def test_crsm_config_initialization():
    """Test that CRSMConfig initializes with correct defaults."""
    config = CRSMConfig()
    assert config.vocab_size == 32000
    assert config.hidden_size == 2048
    assert config.num_hidden_layers == 24

def test_model_forward_shape():
    """Test that model forward pass produces correct output shape."""
    config = CRSMConfig(vocab_size=1000, hidden_size=256)
    model = CRSMModel(config)
    
    input_ids = torch.randint(0, 1000, (2, 128))
    logits, states = model(input_ids)
    
    assert logits.shape == (2, 128, 1000)
```

## Pull Request Process

1. **Sync with main** before starting work:
   ```bash
   git fetch origin
   git rebase origin/main
   ```

2. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create Pull Request** on GitHub with:
   - Clear title describing the change
   - Detailed description of what and why
   - Reference to related issues (if any)
   - Screenshots or examples (if applicable)

4. **PR Template**:
   ```markdown
   ## Description
   Brief description of the changes.
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   How to test these changes?
   
   ## Checklist
   - [ ] Tests pass locally
   - [ ] Code follows style guidelines
   - [ ] Documentation is updated
   - [ ] Commit messages are descriptive
   
   ## Related Issues
   Fixes #123
   ```

5. **Wait for review** - Maintainers will review and request changes if needed

## Areas for Contribution

### High Priority

1. **Performance Optimization**
   - Optimize S4/Mamba kernels
   - Reduce memory usage
   - Improve training throughput

2. **Testing**
   - Add integration tests
   - Add edge case tests
   - Add benchmark tests

3. **Documentation**
   - Add tutorials
   - Improve API docs
   - Add architecture explanations

### Medium Priority

4. **Features**
   - Distillation loss implementation
   - Multi-GPU training (DDP)
   - Quantization support
   - Export formats (ONNX, etc.)

5. **Examples**
   - Add more example scripts
   - Create Jupyter notebook tutorials
   - Add deployment examples

### Lower Priority

6. **Quality of Life**
   - Better error messages
   - CLI improvements
   - Configuration management
   - Logging improvements

## Documentation

- Keep docs in `docs/` folder
- Use Markdown format
- Update README.md for major changes
- Add docstrings to all public functions

```python
def function_name(arg1: str, arg2: int) -> bool:
    """
    Brief description of what the function does.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
    
    Returns:
        Description of return value
    
    Example:
        >>> result = function_name("hello", 42)
        >>> print(result)
        True
    """
    pass
```

## Release Process

Releases follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes

Steps for maintainers:
1. Update version in `setup.py`
2. Update `CHANGELOG.md`
3. Create tag: `git tag v0.1.0`
4. Push tag: `git push origin v0.1.0`
5. Build and upload to PyPI

## Questions?

- Check existing issues and discussions
- Open a new discussion for questions
- Open an issue for bugs
- Contact maintainers via email

## Thank You!

Your contributions help make CRSM better for everyone. We appreciate your time and effort!
