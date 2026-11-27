"""Shared filesystem and timestamp helpers for workflow entry points.

The batch workflows and the Streamlit pages each carried their own copies of
these. Two variants had drifted apart in ways that mattered:

- ``write_json`` existed in an atomic form (write to ``.tmp``, then replace)
  in the workflows and a plain-write form in the apps. A crash or a concurrent
  read during the plain write leaves a truncated status file, which the UI
  then fails to parse. The atomic form is used everywhere now.
- ``read_json`` caught ``(OSError, json.JSONDecodeError)`` in two places and
  bare ``Exception`` in a third. The broader catch is used, so no caller can
  start raising where it previously returned ``None``.

Not unified here: ``slugify``, ``update_status`` and ``dense_matrix`` also
exist in multiple copies but genuinely differ between modules, so merging them
would change behaviour rather than remove duplication.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path

__all__ = [
    "LOG_DIRNAME",
    "configure_caches",
    "find_run_file",
    "now",
    "read_json",
    "run_files",
    "start_logged_process",
    "write_json",
]

#: Every stage keeps its run bookkeeping -- the settings it was started with,
#: the live status file the app polls, and the worker's console output -- in
#: one ``logs/`` folder inside its output folder, so the results themselves
#: are not buried under a dozen ``*_config.json`` / ``*_status.json`` /
#: ``*_worker.log`` files.
LOG_DIRNAME = "logs"


def configure_caches() -> Path:
    """Point matplotlib/numba/XDG caches at one shared SpatioEv cache folder.

    The workers used to create a differently named cache folder each
    (``spatioev_clustering_matplotlib``, ``spatioev_scimap_numba``, ...), about
    twenty in the system temporary folder. Values already set in the
    environment are kept. Must run before matplotlib or numba is imported.
    """
    root = Path(tempfile.gettempdir()) / "spatioev_cache"
    for variable, folder in (
        ("MPLCONFIGDIR", "matplotlib"),
        ("NUMBA_CACHE_DIR", "numba"),
        ("XDG_CACHE_HOME", "xdg"),
    ):
        os.environ.setdefault(variable, str(root / folder))
        Path(os.environ[variable]).mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))
    return root


def run_files(output_dir: Path, stem: str) -> tuple[Path, Path, Path]:
    """Config, status and log paths for one run, all inside ``<output_dir>/logs``."""
    logs = Path(output_dir) / LOG_DIRNAME
    return (
        logs / f"{stem}_config.json",
        logs / f"{stem}_status.json",
        logs / f"{stem}.log",
    )


def find_run_file(output_dir: Path, name: str) -> Path:
    """Locate a run file in ``logs/``, falling back to the pre-``logs/`` layout.

    Output folders written by earlier SpatioEv versions kept these files next
    to the results. They are still found, so an old run can be resumed.
    """
    output_dir = Path(output_dir)
    current = output_dir / LOG_DIRNAME / name
    legacy = output_dir / name
    return legacy if not current.exists() and legacy.exists() else current


def start_logged_process(command: list[str], log_path: Path) -> int:
    """Start *command* in its own session, writing its output to *log_path*.

    The log is overwritten, not appended: it describes the latest run only.
    Appending made the logs grow without bound and mixed the messages of
    every earlier attempt into the one being read.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
    return process.pid


def now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def write_json(path: Path, payload: dict) -> None:
    """Write *payload* as JSON atomically.

    The file is written to a sibling ``.tmp`` path and then moved into place,
    so a reader never observes a partially written file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict | None:
    """Read JSON from *path*, returning ``None`` if it is missing or unreadable."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
