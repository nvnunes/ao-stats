# Development

This document is the source of truth for the local environment, bootstrap,
Git hook activation, and routine contributor workflow for AO Stats.

## Bootstrap

Run:

```bash
./scripts/bootstrap.sh
```

The script creates `./.conda`, installs the package with development and
documentation dependencies, configures the repository hook, runs the tests,
and builds the documentation in strict mode.

## Daily Commands

After bootstrap, use commands from the repository-local environment:

```bash
./.conda/bin/python -m pip install -e ".[dev,docs]"
./.conda/bin/mkdocs serve
```

Use [Testing](testing.md) for canonical verification commands and completion
expectations. Consult [Architecture](architecture.md) and the
[Python API](api.md) before changing package ownership or a public calculation
contract.

## Documentation Toolchain

The supported documentation stack uses MkDocs 1.x with the Material theme.
The dependency bounds in `pyproject.toml` keep that toolchain stable until a
deliberate migration.

## Pre-commit Hook

The versioned `.githooks/pre-commit` hook runs the canonical checks defined in
[Testing](testing.md) on every commit. If the repository is not using that
hooks path, configure it with:

```bash
git config core.hooksPath .githooks
```
