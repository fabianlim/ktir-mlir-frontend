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
| **Custom LLVM build** | `cmake/llvm-hash.txt` is present | Stable — pinned to an official LLVM release tag, published as a **GitHub Release asset** (`llvm-<short-hash>`): public, no token, never expires. **Migration in progress (issue #24):** the legacy Actions-artifact path is still produced and refreshed alongside the release asset and remains a token-gated fallback in `setup_mlir.py`; it will be removed once the release path is confirmed. |
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

`llvm-build.yml` builds the pinned LLVM/MLIR once per platform and publishes it
through **two parallel mechanisms** that currently run side by side:

- **Release Flow** — the LLVM build is published as a **GitHub Release asset**
  on the `llvm-<short-hash>` prerelease (public, no token, never expires). This
  is the promoted, default path.
- **Artifacts Flow [Marked for Deprecation]** — the same tarball is also
  uploaded as a GitHub **Actions artifact** (90-day expiry, token-gated). This
  is the legacy path, kept alive during the migration (issue #24) and removed
  once the Release Flow is confirmed and all consumers have cut over.

Both mechanisms exist on purpose right now; `setup_mlir.py` resolves the release
asset first and falls back to the artifact.

**Triggers:**
- Push to `main` that changes `cmake/llvm-hash.txt` (hash bump)
- Manual: `workflow_dispatch` with an optional hash override
- Pull request that touches `llvm-build.yml` (builds + packages to validate the
  workflow, but does **not** publish a release)
- Scheduled cron (1st of every other month) — drives the Artifacts Flow refresh
  only (see *Scheduled Refresh* below); **[Marked for Deprecation]**

A single build job produces the tarball; a `Publish release asset` step (Release
Flow) and an `Upload artifact (legacy)` step (Artifacts Flow) each fire only
when their own target is missing. The build itself runs whenever *either* target
is absent. The Artifacts Flow is removed once the Release Flow is confirmed and
all consumers have cut over (issue #24).

### Hash Bump

On a hash bump (or any dispatch where the target is missing), a single build job
checks out `llvm-project` at the pinned commit, builds LLVM/MLIR with
`MLIR_ENABLE_BINDINGS_PYTHON=ON` (required for downstream Python wheel builds),
runs `check-mlir`, packages the tarball, then feeds both flows and triggers
Flow 1 (`ci.yml`) against the new build.

#### Release Flow

1. `create-release` job ensures the `llvm-<short-hash>` prerelease exists (made
   once, shared by all 3 platform jobs; idempotent)
2. `gh release upload --clobber`s the tarball to the release

#### Artifacts Flow [Marked for Deprecation]

`actions/upload-artifact` uploads the same tarball as a 90-day Actions artifact.

### Manual Invocation

Use this to (re)produce a build without bumping `cmake/llvm-hash.txt` (e.g.
after an accidental delete, or a one-off hash). A single dispatch builds once
and feeds both flows, each upload step skipping if its own target already
exists — so check first. The check differs per flow:

#### Release Flow

```bash
gh release view llvm-<short-hash>            # public, no token
```

#### Artifacts Flow [Marked for Deprecation]

```bash
# Actions-artifacts query — needs a token, unlike the release check above:
gh api "repos/<owner>/<repo>/actions/artifacts?name=llvm-<short-hash>-<os>-<arch>"
```

Then dispatch — uploads the release asset *and* the legacy artifact:

```bash
# Current pinned hash:
gh workflow run llvm-build.yml
# Specific hash (overrides cmake/llvm-hash.txt):
gh workflow run llvm-build.yml -f llvm-hash=<full-40-char-sha>
```

### Scheduled Refresh

#### Release Flow

None — release assets never expire, so there is nothing to refresh. There is
also **no** automated pruning (few LLVM versions are expected); delete a stale
release manually when needed:

```bash
gh release delete <tag> --cleanup-tag --yes   # e.g. tag = llvm-<short-hash>
```

#### Artifacts Flow [Marked for Deprecation]

A `refresh` job runs on the bi-monthly `schedule` cron: it downloads the
existing artifact zip and re-uploads it to reset the 90-day retention clock (no
rebuild — the content is unchanged). `trigger-ci` is **not** called after a
refresh. Removed in Stage 2 along with the rest of the Artifacts Flow.

---

## Flow 3 — Bleeding-edge (`workflow_dispatch` with mlir_wheel)

**Trigger:** manual `workflow_dispatch` on `ci.yml` with `mlir-source=mlir_wheel`

**What it does:** runs the same build + test pipeline as Flow 1 but sources
MLIR from the latest `mlir_wheel` on the eudsl index instead of the pinned
artifact.

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
3. Flow 2 fires automatically — builds LLVM, publishes the release asset (and,
   during migration, the legacy artifact), triggers Flow 1
4. Monitor the `llvm-build` and `cmake-py-test` workflow runs

```bash
# Example: adopt llvmorg-22.1.3
echo "e9846648fd6183ee6d8cbdb4502213fcf902a211" > cmake/llvm-hash.txt
git add cmake/llvm-hash.txt
git commit -m "Bump LLVM to llvmorg-22.1.3"
git push
```
