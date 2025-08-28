# UV Migration Guide

This project has been migrated from pixi to uv for dependency management.

## Installation

1. Install uv if not already installed:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Sync dependencies:
   ```bash
   uv sync
   ```

3. Install the project in editable mode:
   ```bash
   uv pip install -e .
   ```

## Common Commands

### Development
- Install dependencies: `uv sync`
- Add a dependency: `uv add package-name`
- Add a dev dependency: `uv add --dev package-name`
- Remove a dependency: `uv remove package-name`

### Running the project
- Activate virtual environment: `source .venv/bin/activate` (or use `uv run`)
- Run scripts with uv: `uv run python script.py`
- Run the visualizer: `uv run quac-visualizer` (replaces `pixi run visualizer`)

### Environments
The project supports different environments through optional dependencies:
- Development: `uv sync --extra dev` (installs pytest, black, mypy, etc.)
- Web: `uv sync --extra web` (installs flask, werkzeug)
- Docs: `uv sync --extra docs` (installs sphinx and related packages)

### Migration Notes
- Pixi's conda channels (nvidia, pytorch) are replaced by PyPI packages
- The `funlib.learn.torch` git dependency is handled via `[tool.uv.sources]`
- The pixi task `visualizer` is replaced by the `quac-visualizer` script
- Python version is constrained to `>=3.10,<3.13` for PyTorch compatibility
