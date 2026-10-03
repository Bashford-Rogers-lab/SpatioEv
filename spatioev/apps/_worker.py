"""Background-worker plumbing shared by the spatial-analysis pages (04-07).

Each page writes a config JSON, starts ``python -m <workflow module>`` in its
own session, and polls the status JSON the worker keeps up to date. Polling
runs inside a fragment so widget changes made while a job runs are kept.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from spatioev.apps._common import module_command
from spatioev.workflows._io import read_json, write_json

AUTO_REFRESH_SECONDS = 2.0


def start_worker(module: str, config: dict, output_dir: Path, prefix: str) -> tuple[int, Path, Path]:
    """Launch ``module`` on ``config``; return its PID, status and log paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{config.get('sample_id') or config.get('comparison_name') or 'run'}_{prefix}"
    config_path = output_dir / f"{stem}_config.json"
    status_path = output_dir / f"{stem}_status.json"
    log_path = output_dir / f"{stem}_worker.log"
    write_json(config_path, config)
    write_json(status_path, {
        "state": "queued",
        "stage": "queued",
        "message": "Starting worker",
        "progress": 0.01,
        "updated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    })
    command = module_command(module, "--config", str(config_path), "--status", str(status_path))
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process.pid, status_path, log_path


@st.fragment(run_every=AUTO_REFRESH_SECONDS)
def _running(status_path: Path, log_path: Path) -> None:
    status = read_json(status_path)
    if status is None:
        return
    if status.get("state") in {"complete", "failed"}:
        st.rerun()
    st.progress(float(status.get("progress", 0.02)), text=status.get("message", "Running"))
    st.caption(f"Stage: {status.get('stage', 'queued')}")
    if log_path.exists():
        with st.expander("Worker log"):
            st.code(log_path.read_text(encoding="utf-8", errors="replace")[-8000:])


def render_worker(state_prefix: str, render_outputs: Callable[[dict], None]) -> None:
    """Show progress for the page's current job, then its outputs when done."""
    status_text = st.session_state.get(f"{state_prefix}_status_path")
    log_text = st.session_state.get(f"{state_prefix}_log_path")
    if not status_text or not log_text:
        return
    status_path, log_path = Path(status_text), Path(log_text)
    status = read_json(status_path)
    if status is None:
        return
    state = status.get("state")
    if state == "failed":
        st.error(status.get("message", "Worker failed"), icon=":material/error:")
        with st.expander("Error details"):
            st.code(status.get("traceback", "No traceback recorded"))
        return
    if state != "complete":
        _running(status_path, log_path)
        return
    st.success(status.get("message", "Complete"), icon=":material/check_circle:")
    render_outputs(status.get("outputs", {}))


def remember_worker(state_prefix: str, pid: int, status_path: Path, log_path: Path) -> None:
    st.session_state[f"{state_prefix}_status_path"] = str(status_path)
    st.session_state[f"{state_prefix}_log_path"] = str(log_path)
    st.toast(f"Worker started (PID {pid})", icon=":material/rocket_launch:")


def show_images(paths: list[str | None], columns: int = 2) -> None:
    paths = [p for p in paths if p and Path(p).exists()]
    for start in range(0, len(paths), columns):
        for column, path in zip(st.columns(columns), paths[start : start + columns]):
            column.image(path, width="stretch")
            column.caption(Path(path).name)


def path_value(key: str) -> str:
    """A path typed or pasted into a text box, without stray whitespace."""
    return str(st.session_state.get(key) or "").strip()


def parse_numbers(text: str) -> list[float]:
    """Parse ``"10, 20, 30"`` into floats, ignoring blanks."""
    values = []
    for token in str(text).replace(";", ",").split(","):
        token = token.strip()
        if token:
            values.append(float(token))
    return values
