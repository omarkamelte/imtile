# Contributing to imtile

Thanks for your interest in contributing! Here's how to get started.

## Development Setup

```bash
# Clone the repo
git clone https://github.com/omarkamelte/imtile.git
cd imtile

# Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest tests/ -v
```

## How to Contribute

1. **Fork** the repository and create a feature branch from `main`.
2. **Write tests** for any new functionality.
3. **Ensure all tests pass** before submitting a PR.
4. **Open a Pull Request** with a clear description of what you changed and why.

## Reporting Bugs

Open an [issue](https://github.com/omarkamelte/imtile/issues) with:
- A minimal code snippet reproducing the problem
- Your Python version and OS
- The full traceback (if applicable)

## Feature Requests

We welcome ideas! Open an issue tagged `[Feature Request]` describing:
- The problem you're trying to solve
- Your proposed solution (if any)

## Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use type hints where practical.
- Write clear docstrings for public APIs.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
