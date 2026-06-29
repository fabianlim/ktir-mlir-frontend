# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

KTIR (Kernel Tile IR) frontend: an out-of-tree MLIR dialect named `ktdp` (Kernel
Tile Data-Parallel) that models tile-based, data-parallel kernels targeting
multi-core accelerators (Spyre). The dialect supplies **only** memory-view,
access-tile, and data-movement constructs; compute and control flow are
delegated to upstream MLIR dialects (Arith, Math, Linalg, SCF, Tensor, MemRef,
Func) — see `dependentDialects` in `include/Ktdp/KtdpDialect.td`.

## Build prerequisite: MLIR_DIR

Every build needs an MLIR install. The single control point is
`cmake/llvm-hash.txt` (a pinned 40-char LLVM SHA). `scripts/setup_mlir.py`
resolves it (cached at `~/.cache/ktir-mlir/`): downloads the pinned LLVM from
the `llvm-<short-hash>` **GitHub Release asset** — public, no token, never
expires. Token (`GIT_PAT`/`GITHUB_TOKEN`) is needed only for the legacy
Actions-artifact fallback (hashes built before releases existed). `--wheel`
forces `mlir_wheel` (LLVM `main`, no token, expires in 30 days). If you already
have a local LLVM/MLIR build, skip the script and point `MLIR_DIR` at its
`lib/cmake/mlir`.

```bash
MLIR_DIR=$(uv run --no-project python scripts/setup_mlir.py)
```

## Common commands

```bash
# Python bindings + wheel (the recommended dev flow)
uv venv --python 3.12
uv sync --extra test                         # installs pytest + lit
CMAKE_ARGS="-DMLIR_DIR=$MLIR_DIR" uv pip install .

# Bare C++ build (no bindings); add -DKTIR_ENABLE_PYTHON_BINDINGS=ON for them
cmake -S . -B build -GNinja -DMLIR_DIR=$MLIR_DIR
cmake --build build -j$(nproc)

# LIT tests (C++ / IR)
cmake --build build --target check-ktir
llvm-lit -sv build/test/Ktdp/                # whole suite
llvm-lit -sv build/test/Ktdp/add.mlir        # single test
uv run lit -sv build/test/Ktdp/              # via uv (lit from --extra test)

# Python tests
uv run pytest python/test/
uv run pytest python/test/test_basic.py::test_name   # single test

# Driver tool
build/bin/ktir-opt file.mlir
build/bin/ktir-opt --verify-roundtrip file.mlir
uv run ktdp-walk file.mlir                    # walk + print op tree
```

LIT must point at the **build** dir (where cmake generates `lit.site.cfg.py`),
not the source `test/`.

## Architecture

**C++ dialect** is split across four TableGen `.td` files in `include/Ktdp/`,
each with a matching `.cpp` in `lib/Ktdp/`: `KtdpDialect`, `KtdpAttrs`,
`KtdpTypes`, `KtdpOps`. The `.td` is the source of truth — adding/changing ops,
types, or attrs means editing the `.td`, and `include/Ktdp/CMakeLists.txt`
controls which `*.inc` files TableGen emits. The C++ `.cpp` files hold only
hand-written logic (verifiers, custom parsers/printers). `useDefaultType/
AttributePrinterParser = 1` means most types/attrs get auto-generated syntax.

**Key abstractions** (see README "Dialect Overview"): memory views
(`construct_memory_view`, `construct_distributed_memory_view`), access tiles
(`construct_access_tile`, `construct_indirect_access_tile`), data movement
(`load`/`store`), plus `!ktdp.access_tile`/`!ktdp.runtime_arg` types and
`#ktdp.spyre_memory_space` attr.

**Python packaging is two separate trees** — do not conflate them:
- `python/mlir_ktdp/` — the dialect bindings, **fully cmake-managed** (nanobind
  extension `KtdpExtensionNanobind.cpp` + MLIR-generated Python from
  `dialects/KtdpOps.td`). Installed via cmake component `KtdpPythonModules`.
  `python/CMakeLists.txt` globs `*.py` **at configure time**, so a new `.py`
  file under `mlir_ktdp/` requires re-running `cmake -S . -B build`.
- `python/tools_ktdp/` — pure-Python tools (e.g. `ktdp-walk`), **not** cmake-managed;
  packaged by scikit-build-core via `wheel.packages` in `pyproject.toml`.

`pyproject.toml` drives cmake through scikit-build-core (always builds with
`-DKTIR_ENABLE_PYTHON_BINDINGS=ON`). For a bare cmake build (no pip install),
`python/test/conftest.py` auto-adds `build/python_packages/ktdp` to `sys.path`
(override the build dir with `KTIR_BUILD_DIR`).

## CI / LLVM hash bumps

Three flows (full detail in `docs/ci.md`). To adopt a new LLVM release, write
the full 40-char SHA into `cmake/llvm-hash.txt` and push to `main`:
`llvm-build.yml` then builds LLVM, publishes it as the `llvm-<short>` release
asset, triggers the standard CI (`ci.yml`), and prunes old `llvm-*` releases
(keeps newest `LLVM_RELEASE_KEEP`=10). Release assets never expire, so there is
no scheduled refresh job.
