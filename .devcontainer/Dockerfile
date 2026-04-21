# Ubuntu 22.04-based dev image. Installs ktir-mlir-frontend (mlir_ktdp bindings)
# via uv sync from the lockfile — no need to build LLVM from source.
# The ktir-mlir-frontend commit is pinned in pyproject.toml [tool.uv.sources].
#
# Before building this image, copy the lockfile and project config:
#   cp pyproject.toml uv.lock .devcontainer/
#
# Then build with:
#   podman build -f .devcontainer/Dockerfile -t ktir-ubuntu-22.04 .devcontainer/

FROM ubuntu:22.04

ARG UV_PYTHON=3.12
ENV UV_PYTHON=${UV_PYTHON}

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash-completion \
    ca-certificates \
    ccache \
    clang \
    cmake \
    curl \
    git \
    lld \
    make \
    ninja-build \
    patch \
    unzip \
    xz-utils \
  && rm -rf /var/lib/apt/lists/*

# Install uv and use it to manage Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Keep all uv/Python data under /workspace so non-root users can access it
RUN mkdir -p /workspace && chmod -R 777 /workspace
ENV UV_PYTHON_INSTALL_DIR=/workspace/python
ENV UV_CACHE_DIR=/workspace/.uv-cache

RUN uv python install ${UV_PYTHON}
RUN ln -s $(uv python find ${UV_PYTHON}) /usr/local/bin/python3

# Create shared venv
RUN uv venv --python ${UV_PYTHON} /workspace/.venv && chmod -R 777 /workspace/.venv

ENV UV_PROJECT_ENVIRONMENT=/workspace/.venv
ENV PYTHON_EXE=/workspace/.venv/bin/python

# Install project dependencies + mlir extra (ktir-mlir-frontend[mlir]) from lockfile
# pyproject.toml and uv.lock are copied into .devcontainer/ by build_image.sh.
COPY pyproject.toml uv.lock /workspace/code/
RUN uv sync --frozen --extra mlir --no-install-project \
    --python ${PYTHON_EXE} \
    --project /workspace/code

# All build steps above run as root, so files in /workspace are owned by root.
# This final chmod ensures non-root users (e.g. mapped via --userns=keep-id) can
# read, write, and execute everything under /workspace including the venv and build outputs.
RUN chmod -R 777 /workspace

WORKDIR /workspace/code
