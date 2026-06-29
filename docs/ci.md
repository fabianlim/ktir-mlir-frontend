# CI Overview

> The pinned-LLVM CI flow is adopted from
> [triton-lang/triton](https://github.com/triton-lang/triton).
> Publishing the build as a GitHub Release asset and the `mlir_wheel` fallback
> are extensions of that pattern.

This document describes the three CI flows for `ktir-mlir-frontend`, how they
relate to each other, and how to use them for local development.

---

## MLIR source strategy

All CI flows build the KTIR project against an MLIR installation.  There are
two sources:

| Source | When used | Stability |
|--------|-----------|-----------|
| **Custom LLVM build** | `cmake/llvm-hash.txt` is present | Stable — pinned to an official LLVM release tag, published as a **GitHub Release asset** (`llvm-<short-hash>`): public, no token, never expires. (A legacy Actions-artifact download remains in `setup_mlir.py` as a token-gated fallback for hashes built before releases were adopted.) |
| **mlir_wheel** | No hash file, or explicit `--wheel` override | Bleeding-edge — tracks LLVM `main`, individual versions expire after 30 days |

The file `cmake/llvm-hash.txt` is the single control point.  Its presence
switches all three flows to the custom artifact path automatically.

---

## Flow 1 — Standard CI (`ci.yml`)

**Triggers:** push or pull request to `main`

**What it does:**

1. Reads `cmake/llvm-hash.txt` to determine the MLIR source
2. Runs `uv sync --extra test` to install Python test dependencies (venv must
   exist before MLIR setup in case the wheel fallback needs to `pip install`)
3. Calls `scripts/setup_mlir.py` to resolve and cache the MLIR installation:
   - Default: downloads the pinned LLVM build from the `llvm-<short-hash>`
     GitHub Release (no token), falling back to the legacy Actions artifact
   - `--wheel`: installs `mlir_wheel` from the eudsl index (explicit opt-in only)
4. Configures and builds KTIR with CMake
5. Runs LIT tests (`check-ktir`)
6. Builds and installs the Python wheel (`uv pip install .`)
7. Runs Python tests (`pytest python/test/`)

**Normal developer workflow:** open a PR → Flow 1 runs automatically.  No
manual steps needed as long as the LLVM release for the pinned hash exists.

---

## Flow 2 — LLVM Build (`llvm-build.yml`)

**Triggers:**
- Push to `main` that changes `cmake/llvm-hash.txt` (hash bump)
- Manual: `workflow_dispatch` with an optional hash override
- Pull request that touches `llvm-build.yml` (builds + packages to validate the
  workflow, but does **not** publish a release)

**What it does** depends on whether the release asset already exists:

### Hash bump (release asset does not exist)

1. Reads the new hash from `cmake/llvm-hash.txt`
2. Checks the `llvm-<short-hash>` release — no asset for this platform yet
3. Checks out `llvm-project` at the pinned commit
4. Builds LLVM/MLIR with `MLIR_ENABLE_BINDINGS_PYTHON=ON` (required for
   downstream Python wheel builds)
5. Runs `check-mlir` to validate the build
6. Packages the tarball and publishes it as an asset on the `llvm-<short-hash>`
   prerelease (created once, shared by all 3 platform jobs; `--clobber` keeps
   re-runs idempotent)
7. Triggers Flow 1 (`ci.yml`) against the new release
8. Prunes old `llvm-*` releases, keeping the newest `LLVM_RELEASE_KEEP` (10)

Release assets never expire and download without a token, so there is no
scheduled-refresh job — the old 90-day retention clock is gone. Storage is
bounded only by the prune step (step 8).

### When to trigger manually

```bash
# Rebuild for the current pinned hash (e.g. after accidental artifact deletion):
gh workflow run llvm-build.yml

# Build a specific hash (overrides cmake/llvm-hash.txt):
gh workflow run llvm-build.yml -f llvm-hash=<full-40-char-sha>
```

---

## Flow 3 — Bleeding-edge (`workflow_dispatch` with mlir_wheel)

**Trigger:** manual `workflow_dispatch` on `ci.yml` with `mlir-source=mlir_wheel`

**What it does:** runs the same build + test pipeline as Flow 1 but sources
MLIR from the latest `mlir_wheel` on the eudsl index instead of the pinned
release.

Use this to:
- Test compatibility with the latest LLVM `main` before bumping the hash
- Quickly verify a fix without waiting for a full LLVM build

**Note:** `mlir_wheel` builds expire after 30 days and are not tied to
official LLVM release tags.  Do not rely on a specific wheel version being
available long-term.

```bash
gh workflow run ci.yml -f mlir-source=mlir_wheel
```

---

## Local development

### Set up MLIR

There are two ways to obtain an MLIR installation for local builds:

**Option A — Download the prebuilt LLVM (recommended)**

`scripts/setup_mlir.py` downloads the pre-built LLVM produced by Flow 2
(`llvm-build.yml`).  It reads `cmake/llvm-hash.txt`, checks a local cache, then
pulls the `llvm-<short-hash>` **GitHub Release asset** — public, **no token
needed**.  Only if the release is absent (a hash built before releases were
adopted) does it fall back to the legacy Actions artifact, which *does* require
a token.  Pass `--wheel` to explicitly opt in to `mlir_wheel` instead.

```bash
# Release exists (the normal case) — no token needed, cached or not:
MLIR_DIR=$(uv run python scripts/setup_mlir.py)

# The script resolves the repo from git remote automatically; on a fork whose
# release lives in the upstream repo, pass --repo explicitly:
MLIR_DIR=$(uv run python scripts/setup_mlir.py --repo torch-spyre/ktir-mlir-frontend)

# Legacy fallback only: if no release exists, GIT_PAT/GITHUB_TOKEN is needed to
# download the Actions artifact instead.
GIT_PAT=<your-token> MLIR_DIR=$(uv run python scripts/setup_mlir.py)

# Force mlir_wheel (no token required, no cache):
MLIR_DIR=$(uv run python scripts/setup_mlir.py --wheel)
```

The build is cached at `~/.cache/ktir-mlir/<artifact-name>/`.  Once cached,
subsequent calls with the same hash return immediately with no network access.

**Option B — Build MLIR manually**

If you have built LLVM/MLIR from source yourself, skip the script entirely and
set `MLIR_DIR` directly to the `lib/cmake/mlir` directory of your build:

```bash
MLIR_DIR=/path/to/your/llvm-build/lib/cmake/mlir
```

### Build

```bash
# Install Python test dependencies
uv sync --extra test

# Build and install the Python wheel
CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv pip install .

# Run tests
cmake --build build --target check-ktir   # LIT tests
uv run pytest python/test/                # Python tests
```

---

## Hash bump procedure

To adopt a new LLVM release:

1. Update `cmake/llvm-hash.txt` with the full 40-character commit SHA
2. Push to `main` (or merge a PR that changes the file)
3. Flow 2 fires automatically — builds LLVM, uploads artifact, triggers Flow 1
4. Monitor the `llvm-build` and `cmake-py-test` workflow runs

```bash
# Example: adopt llvmorg-22.1.3
echo "e9846648fd6183ee6d8cbdb4502213fcf902a211" > cmake/llvm-hash.txt
git add cmake/llvm-hash.txt
git commit -m "Bump LLVM to llvmorg-22.1.3"
git push
```
