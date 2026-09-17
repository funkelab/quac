# Installation

1. Clone this repository
2. Choose one of the following:
    - **uv**: Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
      and run `uv sync` in the repository. Run your scripts with `uv run`, or
      activate the `.venv` and run your scripts as normal.
      On linux, use `uv sync --group cpu` if you don't need CUDA.
    - **pixi**: Install [`pixi`](https://pixi.sh/latest/installation/). Run your
      scripts in this repository with `pixi run` OR start a `pixi shell` and run
      your scripts as normal.
