#!/bin/bash
set -e

KTIR_IMAGE=ktir-frontend-ubuntu-22.04

# Build the image only if it doesn't exist
if ! podman image exists ${KTIR_IMAGE}; then
    echo "Image ${KTIR_IMAGE} not found, building..."
    cp pyproject.toml uv.lock .devcontainer/
    podman build -f .devcontainer/Dockerfile -t ${KTIR_IMAGE} .devcontainer/
fi
