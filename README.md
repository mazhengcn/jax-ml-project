# JAX-AI-Stack ML Project Template

This repository is a [cookiecutter](https://cookiecutter.readthedocs.io/) template for machine learning projects using the [jax-ai-stack](https://github.com/jax-ai/jax-ai-stack). It provides a structured starting point for developing, training, and deploying ML models with JAX and related tools.

## Features

- Pre-configured project structure for reproducible ML workflows
- Integration with JAX, Flax, Optax, Grain, Clu and other libraries from the jax-ai-stack
- Example scripts for training and evaluation
- Utilities for data handling and experiment tracking
- Ready for GPU acceleration

## Getting Started

Generate a new project using cookiecutter through [uv](https://github.com/astral-sh/uv) (recommended):

```bash
uvx cookiecutter https://github.com/mazhengcn/jax-ml-project
```

Or, by using pip:

```bash
pip install cookiecutter
cookiecutter https://github.com/mazhengcn/jax-ml-project
```

Follow the prompts to customize your project.

## Usage

After generating your project, use [uv](https://github.com/astral-sh/uv) to manage dependencies and run scripts:

```bash
cd your-project-name
uv sync
```

You can also use `uv` to run scripts in the project environment:

```bash
uv run python examples/run_train.py
```

## Dev Container

This project includes a [Dev Container](https://containers.dev/) configuration in the `.devcontainer` directory. You can use this to quickly get a reproducible development environment with all dependencies pre-installed.

To use the Dev Container:

1. Install [Visual Studio Code](https://code.visualstudio.com/) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers).
2. Open the project folder in VS Code.
3. When prompted, reopen the project in the Dev Container.

This will automatically build the container and set up the environment for you.

## Requirements

- Python 3.12+
- jax-ai-stack (JAX, Flax, Optax, etc.)
- NumPy
- [uv](https://github.com/astral-sh/uv) (for dependency management)
- (Other dependencies listed in `pyproject.toml`)

## License

This template is licensed under the MIT License.
