# Development Guide

## Code Formatting with black-with-tabs

This project uses [black-with-tabs](https://github.com/Amar1729/black-with-tabs) for consistent Python code formatting.

### Installation

black-with-tabs is included in the project dependencies:

```bash
pip install -r requirements.txt
```

Or install it separately:

```bash
pip install black-with-tabs
```

### Configuration

Formatter configuration is defined in `pyproject.toml`:

- **Line length**: 120 characters
- **Indentation**: Tabs (not spaces)
- **String quotes**: Single quotes preserved (skip normalization)
- **Target Python versions**: 3.8, 3.9, 3.10, 3.11
- **Blank lines**: 2 blank lines between top-level definitions (enforced by default, per PEP 8)
- **Excluded directories**: `env/`, `venv/`, `third_party/`, build artifacts

### Usage

#### Format all code

```bash
# Format src/ and examples/ directories
python3 -m black src/ examples/ --exclude='env|third_party'
```

#### Check formatting without changes

```bash
# Check if code is formatted
python3 -m black --check src/ examples/ --exclude='env|third_party'
```

#### Format specific file

```bash
python3 -m black src/run_autotuner.py
```

#### Format with verbose output

```bash
python3 -m black src/ examples/ --exclude='env|third_party' --verbose
```

### Pre-commit Hook (Optional)

To automatically format code before commits, you can add a git pre-commit hook:

```bash
# Create .git/hooks/pre-commit
cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
python3 -m black --check src/ examples/ --exclude='env|third_party'
if [ $? -ne 0 ]; then
    echo "Code is not formatted. Running black..."
    python3 -m black src/ examples/ --exclude='env|third_party'
    echo "Code has been formatted. Please review and commit again."
    exit 1
fi
EOF

chmod +x .git/hooks/pre-commit
```

### IDE Integration

#### VS Code

Install the Python extension and add to `.vscode/settings.json`:

```json
{
    "python.formatting.provider": "black",
    "python.formatting.blackArgs": ["--line-length", "120", "--config", "pyproject.toml"],
    "editor.formatOnSave": true,
    "editor.detectIndentation": false,
    "editor.insertSpaces": false,
    "editor.tabSize": 4
}
```

#### PyCharm

1. Go to Settings → Tools → Black
2. Enable "Black"
3. Set line length to 120
4. Set arguments: `--config pyproject.toml`
5. Enable "Run Black on save"

## Code Style Guidelines

Beyond black-with-tabs formatting, follow these conventions:

1. **Imports**: Use absolute imports from project root
2. **Docstrings**: Use Google-style docstrings for all public functions/classes
3. **Type hints**: Use type hints for function signatures (Python 3.8+ syntax)
4. **Comments**: Write clear, concise comments explaining "why", not "what"
5. **Blank lines**:
   - 2 blank lines between top-level definitions (classes, functions)
   - 1 blank line between methods inside a class
   - Black enforces this automatically

### Example

```python
from typing import Dict, Any, Optional


def deploy_service(
    name: str,
    namespace: str,
    parameters: Dict[str, Any],
    timeout: Optional[int] = 600
) -> bool:
    """Deploy an inference service with given parameters.

    Args:
        name: Service name
        namespace: Kubernetes namespace
        parameters: Runtime parameter configuration
        timeout: Max wait time in seconds (default: 600)

    Returns:
        True if deployment succeeded, False otherwise
    """
    # Implementation details...
    pass
```

## Testing

Before submitting changes:

1. **Format code**: `black src/ examples/ --exclude='env|third_party'`
2. **Run tests**: `pytest tests/` (when test suite is available)
3. **Test manually**: Run autotuner with sample tasks
4. **Check docs**: Update documentation if adding features

## Documentation

- Update `README.md` for user-facing changes
- Update `CLAUDE.md` for development guidelines
- Add detailed docs to `docs/` for complex features
- Update `agentlog.md` for mini-milestones (per CLAUDE.md)
