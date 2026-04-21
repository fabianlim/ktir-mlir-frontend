# Devcontainer Setup

## Building the image

This runs automatically via `initializeCommand` in `devcontainer.json`:
```bash
bash .devcontainer/build_image.sh
```

This will:
1. Build the `ktir-frontend-ubuntu-22.04` image if it doesn't already exist

## Prerequisites

[Podman](https://podman.io/) (rootless) and the [Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers) for VS Code. The dev container is configured to use `podman` with `--userns=keep-id` so bind-mounted files retain your host UID.

## One-time setup

VS Code picks up the podman socket via `DOCKER_HOST`. Add to your shell profile (`~/.zprofile`, `~/.bashrc`, etc.) so it is set before VS Code launches:

```bash
export DOCKER_HOST=unix://${XDG_RUNTIME_DIR}/podman/podman.sock
```

Enable the podman socket service:

- **Linux (systemd):**
  ```bash
  systemctl --user enable --now podman.socket
  ```
- **macOS (podman machine):**
  ```bash
  podman machine start
  ```

Then restart VS Code. To verify the socket path:

```bash
podman info --format '{{.Host.RemoteSocket.Path}}'
```

## Directory layout inside the container

| Path | Contents |
|---|---|
| `/workspace/code` | This repository (bind-mounted from your local clone) |
| `/workspace` | Home directory (`$HOME`) |

## Opening in a Dev Container

1. Open the repository folder in VS Code
2. When prompted, click **Reopen in Container** — or run the command palette (`Cmd+Shift+P`) → **Dev Containers: Reopen in Container**
3. VS Code will use the pre-built image and mount the repo at `/workspace/code`

## First-time setup inside the container

This runs automatically via `postCreateCommand` in `devcontainer.json`:

```bash
uv sync --extra mlir && uv pip install -e .
```

## Manual builds

To do a full CMake build manually (scikit-build-core drives cmake):

```bash
uv pip install -e .
```

The build directory defaults to `build/` but can be overridden with `KTIR_BUILD_DIR`:

```bash
KTIR_BUILD_DIR=/tmp/ktir-build uv pip install -e . 
```

`UV_PROJECT_ENVIRONMENT` is set to `/workspace/.venv` in the container, so `uv` will never touch the host's `.venv` inside `/workspace/code`.

## Testing

**Run a single file with `ktdp-walk`:**

```bash
uv run ktdp-walk test/Ktdp/add-with-control-flow-runtime-arg.mlir
```

**Run the lit test suite via CMake:**

```bash
cmake -S . -B build -GNinja \
  -DMLIR_DIR=/workspace/.venv/lib/python3.12/site-packages/mlir_wheel/lib/cmake/mlir \
  -DKTIR_ENABLE_PYTHON_BINDINGS=ON \
  -DPython3_EXECUTABLE=/workspace/.venv/bin/python3 \
  -DLLVM_EXTERNAL_LIT=/workspace/.venv/bin/lit \
  -DLLVM_LIT_ARGS="-v"

cmake --build build -j$(nproc)
cmake --build build --target check-ktir
```
