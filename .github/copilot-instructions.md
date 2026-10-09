# Copilot instructions for `pyimcom`

## Big picture architecture
- `pyimcom` is a Roman image-processing package with two major pipelines:
  1. **Coaddition** (`src/pyimcom/coadd.py`) driven by `Block`, `InImage`, `OutStamp`.
  2. **Destriping** (`src/pyimcom/imdestripe.py`) driven by `Sca_img`, `Parameters`, and conjugate-gradient optimization.
- Runtime configuration is centralized in `Config` (`src/pyimcom/config.py`): JSON keys are uppercase (e.g. `INDATA`, `OUT`, `DSMODEL`) but attributes are lowercase (e.g. `cfg.inpath`, `cfg.outstem`, `cfg.ds_model`).
- Data model is hierarchical: pixel → postage stamp → block → mosaic (see `docs/hierarchy.rst`).
- Performance-critical interpolation/LA paths prefer `furry_parakeet.pyimcom_croutines`; modules fall back to local extension `pyimcom_croutines`, then pure/numba code in `src/pyimcom/routine.py`.

## Core data flows to preserve
- Coadd flow: `Config` → `Block.parse_config()` → `InImage` ingest/WCS mapping → layer extraction (`layer.get_all_data`) → LA kernel (`lakernel.py`) → FITS outputs.
- Destripe flow: load SCA images + masks (`Sca_img`) → build overlaps/interpolations (`interpolate_image_bilinear`) → optimize stripe params (`conjugate_gradient`) → write destriped FITS and diagnostics.
- Destriping uses large temporary memmaps (`*_geff.dat`, `*_Neff.dat`, `psi_all.dat`) and optional interpolation cache (`interp_mappings.h5`). Keep I/O patterns and shapes intact when refactoring.

## Developer workflows (actual project conventions)
- Setup dev env:
  - `python -m pip install -e '.[dev]'`
  - Optional: run `./.setup_dev.sh` to install editable deps + pre-commit hooks.
- Tests:
  - Standard: `python -m pytest`
  - CI coverage style: `NUMBA_DISABLE_JIT=1 python -m pytest --cov=pyimcom --cov-report=xml`
- Lint/format uses Ruff via pre-commit (`.pre-commit-config.yaml`), with import sorting enabled.
- Docs build is part of pre-commit/CI and uses Sphinx (`docs/Makefile` and local `sphinx-build` hook). Changes that break docs are considered regressions.

## Project-specific coding patterns
- Prefer existing fallback import pattern for C routines (`try furry_parakeet` → `try pyimcom_croutines` → local fallback).
- Preserve Roman detector assumptions unless intentionally changing science behavior:
  - 18 SCAs, default active size `4088x4088` (`Settings.sca_nside`).
  - Filename regex patterns like `_<obsid>_<scaid>` are used in multiple modules (`imdestripe.py`, `layer_wrapper.py`).
- `Config` values are often derived; after mutating numerical config fields in code, call `cfg()` to recompute dependent parameters.
- Many workflows run on HPC. Avoid introducing unnecessary in-memory copies of full-SCA arrays; prefer existing memmap/chunked/parallel structures.

## Integration points and external dependencies
- File formats: FITS + ASDF/GWCS (`PyIMCOM_WCS` wrapper in `src/pyimcom/wcsutil.py`).
- Diagnostics/reporting entry point: `python -m pyimcom.diagnostics.run <input_fits> <output_stem>`.
- Layer generation wrapper for precomputing cached inputs: `src/pyimcom/layer_wrapper.py` (`build_all_layers`).
- Tests include both unit-scale synthetic data and heavier integration-style cases (`tests/pyimcom/test_pyimcom.py`, `tests/pyimcom/test_imdestripe.py`); prefer synthetic fixtures for fast iterations.
