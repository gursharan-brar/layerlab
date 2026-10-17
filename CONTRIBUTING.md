# Contributing to layerlab

Thanks for your interest in layerlab. This guide covers how to set up a development environment and what we expect from changes.

## Ground rules

- Please read the [Code of Conduct](CODE_OF_CONDUCT.md).
- For anything larger than a small fix, open an issue first so we can agree on the approach.
- Security issues do not go in public issues. See [SECURITY.md](SECURITY.md).

## Development setup

Requires Python 3.10 or newer.

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Checks

Every pull request must pass these locally and in CI:

```bash
ruff check .
ruff format --check .
mypy
pytest
```

CI runs the same checks on Linux, macOS and Windows with Python 3.10 and 3.12.

## Architecture

Read [CLAUDE.md](CLAUDE.md) before adding a processing step or a model backend. In short: each step is independent, reads from and writes to a job folder, and records its results in `manifest.json`. Model code sits behind interfaces so backends can be swapped.

## Pull requests

- Keep PRs focused. One concern per PR.
- Add or update tests for behaviour changes.
- Fill in the PR template.
- Do not commit model weights, sample footage with unclear rights, or generated job outputs.

## Licensing of contributions

layerlab is licensed under the Apache License 2.0. Unless you state otherwise, any contribution you submit for inclusion is licensed under the same terms, as described in Section 5 of the license.
