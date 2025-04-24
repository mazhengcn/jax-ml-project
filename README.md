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

### The resulting directory structure

The directory structure of your new project will look something like this (depending on the settings that you choose):

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         {{ cookiecutter.module_name }} and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── {{ cookiecutter.module_name }}   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes {{ cookiecutter.module_name }} a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling
    │   ├── __init__.py
    │   ├── predict.py          <- Code to run model inference with trained models
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```
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
