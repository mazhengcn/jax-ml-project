# First, build the application in the `/app` directory
FROM ghcr.io/astral-sh/uv:latest AS builder
# Install system packages
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
# Configure the Python directory so it is consistent
ENV UV_PYTHON_INSTALL_DIR=/python
# Only use the managed Python version
ENV UV_PYTHON_PREFERENCE=only-managed

WORKDIR /{{ cookiecutter.repo_name }}
# Install Python before the project for caching
RUN --mount=type=bind,source=.python-version,target=.python-version \
    uv python install
# Install the project dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
# Install the project
COPY . /{{ cookiecutter.repo_name }}
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Then, use a final image without uv
FROM ubuntu:latest
# Copy the Python version
COPY --from=builder --chown=python:python /python /python
# Copy the application from the builder
COPY --from=builder --chown={{ cookiecutter.repo_name }}:{{ cookiecutter.repo_name }} /{{ cookiecutter.repo_name }} /{{ cookiecutter.repo_name }}
# Place executables in the environment at the front of the path
ENV PATH="/{{ cookiecutter.repo_name }}/.venv/bin:$PATH"
