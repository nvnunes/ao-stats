# Testing

This document defines how to verify AO Stats calculations, public interfaces,
packaging, and documentation.

## Environment

Use the repository-local environment defined in
[Development](development.md). That document owns bootstrap, dependency
installation, and Git hook activation.

## Canonical Commands

Refresh the editable installation when dependencies or package metadata
change:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
```

Run the test suite and strict documentation build with:

```bash
./.conda/bin/python -m pytest -q
./.conda/bin/mkdocs build --strict
```

Build the package when changing packaging metadata or public exports:

```bash
./.conda/bin/python -m build
```

## Completion Expectations

Run the full test suite for changes to calculations, validation, selectors,
preprocessing, units, shapes, dtypes, warnings, or errors. Build documentation
in strict mode for public API, docstring, README, or documentation changes.
Package-root imports and the generated reference must remain aligned.

Tests for calculation changes should cover both a single PSF and a cube when
their result shapes differ. Contract tests should make unit, dtype, selector,
validation, warning, and failure behavior explicit. Numerical changes should
include a focused regression case that states the expected scientific
behavior.

Targeted tests are appropriate during iteration, but final verification should
cover the complete affected surface.

## Pre-commit Verification

The versioned pre-commit hook runs the test suite and strict documentation
build shown above. See [Development](development.md#pre-commit-hook) for hook
activation.
