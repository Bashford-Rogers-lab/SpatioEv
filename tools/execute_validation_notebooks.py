"""Execute generated public-API validation notebooks in-memory.

The script is intentionally lightweight: it runs notebooks without writing
executed outputs back to disk, then prints a pass/fail table for release QA.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def configure_temp_runtime() -> None:
    tmp = Path(tempfile.gettempdir())
    defaults = {
        "IPYTHONDIR": tmp / "spatioev-ipython",
        "JUPYTER_RUNTIME_DIR": tmp / "spatioev-jupyter-runtime",
        "MPLCONFIGDIR": tmp / "spatioev-mplconfig",
        "NUMBA_CACHE_DIR": tmp / "spatioev-numba",
    }
    for key, value in defaults.items():
        value.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(key, str(value))


def execute_notebook(path: Path, timeout: int) -> tuple[bool, str]:
    nb = nbformat.read(path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=timeout,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    try:
        client.execute()
    except Exception as exc:  # pragma: no cover - diagnostic script
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="+", type=Path)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    configure_temp_runtime()
    results: list[tuple[Path, bool, str]] = []
    for notebook in args.notebooks:
        ok, message = execute_notebook(notebook, timeout=args.timeout)
        results.append((notebook, ok, message))
        status = "PASS" if ok else "FAIL"
        print(f"{status}\t{notebook}")
        if message:
            print(message)

    failed = [result for result in results if not result[1]]
    print(f"\nExecuted {len(results)} notebooks; {len(failed)} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
