# GemPy Plugins

Extra, optional functionality for [GemPy](https://github.com/gempy-project/gempy) that
doesn't live in the core package. Plugins here can depend on extra third-party
libraries, move independently of GemPy's own release cycle, and don't need to meet the
same stability bar as the core repo.

## Installation

```bash
pip install gempy_plugins
```

For development, clone the repo and install it editable into the same environment as
your `gempy` checkout:

```bash
pip install -e .
```

## Structure

- **Active plugin directories** (e.g. `kriging/`, `topology_analysis/`) -- actively
  maintained plugins.
- **`archive/`** -- stale or outdated code that isn't actively maintained, kept as a
  reference.
- **`wip/`** -- features still being built out. Expect incomplete functionality and
  breaking changes.