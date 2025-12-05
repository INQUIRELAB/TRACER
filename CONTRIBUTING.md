# Contributing Guidelines

Thank you for your interest in contributing to this project!

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/Gourab562/TRACER.git`
3. Create a virtual environment: `python -m venv venv && source venv/bin/activate`
4. Install in development mode: `pip install -e ".[dev]"`

## Code Style

- Follow PEP 8 style guidelines
- Use type hints for all functions
- Run `ruff check .` before committing
- Run `mypy src/` for type checking

## Testing

- Write tests for new features
- Run tests: `pytest tests/ -v`
- Ensure all tests pass before submitting PR

## Commit Messages

Use clear, descriptive commit messages:
- `feat: Add new feature`
- `fix: Fix bug in X`
- `docs: Update documentation`
- `refactor: Refactor code`

## Pull Request Process

1. Update documentation if needed
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Submit PR with clear description

