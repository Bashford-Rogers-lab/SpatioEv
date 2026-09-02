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
from datetime import UTC, datetime
from pathlib import Path

__all__ = ["now", "read_json", "write_json"]


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
