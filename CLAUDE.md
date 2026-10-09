# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

mwatershed is a Rust-based Python library implementing the mutex watershed algorithm for affinity-based image segmentation. It uses PyO3/maturin to expose Rust performance to Python consumers. Supports up to 6-dimensional arrays (tested on 2D and 3D).

## Build & Development Commands

**Requires Rust nightly toolchain and Python 3.11+. Always use `uv` to run Python tooling (pytest, ruff, maturin, etc.) — never call them directly.**

### Python (via uv + maturin)
```bash
uv run --extra dev pytest tests                          # run all Python tests
uv run --extra dev pytest tests/test_affs.py::test_name  # run a single test
uv run --extra dev maturin develop --release             # rebuild Rust extension in-place
```

### Rust
```bash
cargo test      # run Rust unit tests (nightly)
cargo bench     # run Criterion benchmarks
cargo clippy    # lint
cargo fmt       # format
```

### Linting
```bash
uvx ruff check mwatershed tests   # Python linting
```

### Releasing
```bash
bumpversion patch  # or minor/major — updates Cargo.toml, pyproject.toml, mwatershed/__init__.py
git push && git push --tags  # triggers publish workflow
```

## Architecture

### Rust core (`src/`)
- **`lib.rs`** — Entry point. Contains `get_edges<D>()` (extracts sorted edges from affinity arrays), `agglomerate<D>()` (main algorithm, supports 1–6D via const generics), `cluster_edges()` (arbitrary graph clustering), and PyO3 bindings (`agglom_rs`, `cluster`).
- **`clustering.rs`** — Core algorithm state: `Positives` (Union-Find merges), `Negatives` (mutex constraint tracking), `Clustering` (combined state). `process_edges()` does greedy merge/split based on sorted edge ordering.
- **`labels.rs`** — `LabelWrapper` maps between spatial labels and node IDs, handles filtering and merging on label arrays.

### Python wrapper (`mwatershed/__init__.py`)
Thin layer over Rust bindings. Exports `agglom()`, `cluster_edges()`, and `cluster()`.

### Version sync
Three files must stay in sync: `Cargo.toml`, `pyproject.toml`, `mwatershed/__init__.py`. Use `bumpversion` to update all three.
