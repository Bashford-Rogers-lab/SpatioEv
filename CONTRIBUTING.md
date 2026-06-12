# Contributing to SpatioEv

Thank you for your interest in contributing! SpatioEv is an academic research
package and we welcome bug reports, documentation improvements, and new
spatial analysis functions.

---

## Reporting Issues

Please use [GitHub Issues](https://github.com/Bashford-Rogers-lab/SpatioEv/issues)
to report bugs or request features. Include:

- a minimal reproducible example (an `AnnData` object with synthetic data is
  ideal);
- the exact error message and traceback;
- your Python version and SpatioEv version (`import spatioev; spatioev.__version__`).

---

## Development Setup

```bash
# 1. Fork the repository and clone your fork
git clone https://github.com/<your-username>/SpatioEv.git
cd SpatioEv

# 2. Create a fresh environment
conda create -n spatioev-dev python=3.11
conda activate spatioev-dev

# 3. Install in editable mode with dev extras
pip install -e ".[dev,scanpy,trajectory]"

# 4. Run the test suite to confirm everything works
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest
```

---

## Code Style

SpatioEv uses **Ruff** for linting and **Black** for formatting (both included
in the `dev` extras). Before opening a pull request, run:

```bash
ruff check spatioev/
black spatioev/
```

Key conventions:

- **Type annotations** on all public function signatures (PEP 484).
- **NumPy-style docstrings** (`Parameters`, `Returns`, `Examples`) on all
  public functions.
- **Lazy imports** for any heavy optional dependency (Scanpy, ElPiGraph,
  Napari) — do not import them at module level; use `import X` inside the
  function body, or guard with `try/except ImportError`.
- Target Python ≥ 3.10 — use `list[str]` not `List[str]`, `X | None` not
  `Optional[X]`.

---

## Adding a New Function

1. Identify the correct submodule for your function:

   | Namespace | Submodule | Purpose |
   |-----------|-----------|---------|
   | `sv.pp`   | `spatioev/pp/` | QC, normalization, pixel features |
   | `sv.tl`   | `spatioev/tl/` | Spatial stats, niches, pseudotime, ECM |
   | `sv.pl`   | `spatioev/pl/` | Plots |
   | `sv.xe`   | `spatioev/xe/` | Xenium / spatial transcriptomics |
   | `sv.hl`   | `spatioev/hl/` | Reusable building blocks |

2. Write the function with a full NumPy-style docstring and type annotations.

3. Add it to the `__all__` list of the relevant submodule file
   (e.g., `spatioev/tl/stats.py`).

4. Re-export it from the corresponding namespace `__init__.py`
   (e.g., `spatioev/tl/__init__.py`).

5. Write at least one test in `tests/` that exercises the happy path with a
   small synthetic `AnnData` object (see `tests/conftest.py` for the
   `toy_adata` fixture).

6. Run `ruff check`, `black`, and `pytest` before opening a PR.

---

## Pull Request Checklist

- [ ] All new public functions have NumPy-style docstrings and type hints.
- [ ] New functions are exported from the correct namespace `__init__.py`.
- [ ] At least one test covers the new function.
- [ ] `ruff check` and `black` pass with no errors.
- [ ] `pytest` passes (14 tests minimum).
- [ ] `CHANGELOG.md` updated under `[Unreleased]`.

---

## Licence

By contributing, you agree that your contributions will be licensed under the
[GPL-3.0-or-later](LICENSE) licence.
