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


def _render_batch_outputs(outputs: dict) -> None:
    columns = st.columns(4)
    for column, (key, label) in zip(columns, [("n_done", "Done"), ("n_skipped", "Already done"), ("n_missing_input", "Missing input"), ("n_failed", "Failed")]):
        column.metric(label, outputs.get(key, 0))
    if outputs.get("summary") and Path(outputs["summary"]).exists():
        import pandas as pd

        summary = pd.read_csv(outputs["summary"]).fillna("")
        st.dataframe(summary, hide_index=True, width="stretch", height=min(420, 38 * len(summary) + 40))
        st.caption(f"Saved as {outputs['summary']}. Open each core's QC images in its results folder before using the numbers.")


def render_batch(state_prefix: str) -> None:
    """After the page's run has finished, offer to repeat it for every core."""
    status_text = st.session_state.get(f"{state_prefix}_status_path")
    if not status_text or (read_json(Path(status_text)) or {}).get("state") != "complete":
        return
    config_path = Path(status_text.replace("_status.json", "_config.json"))
    template = read_json(config_path) if config_path.exists() else None
    if not template or not template.get("sample_id") or not template.get("output_dir"):
        return
    from spatioev.workflows.batch import project_root

    core = template["sample_id"]
    batch_prefix = f"{state_prefix}_batch"
    st.subheader("Run for every core")
    with st.container(border=True):
        st.caption(
            f"Happy with {core}? Repeat this run for every core folder with exactly the same settings; only the core name "
            "in the paths changes. Cores that are already done are skipped, so after an interruption just press the "
            "button again."
        )
        left, middle, right = st.columns([0.6, 0.2, 0.2])
        root = left.text_input("Folder holding the core folders", value=str(project_root(template)), key=f"{batch_prefix}_root")
        jobs = middle.number_input(
            "Cores at a time", 1, 16, 1, 1, key=f"{batch_prefix}_jobs",
            help="Each core needs as much memory as one run. Keep 1 on a laptop.",
        )
        redo = right.checkbox("Redo finished cores", value=False, key=f"{batch_prefix}_redo")
        if st.button("Run for every core", type="primary", icon=":material/dynamic_feed:", key=f"{batch_prefix}_run"):
            output_dir = Path(root.strip()).expanduser().resolve() / "results"
            config = {
                "sample_id": "all_cores", "template": str(config_path), "roots": [root.strip()],
                "overwrite": bool(redo), "jobs": int(jobs),
            }
            remember_worker(batch_prefix, *start_worker("spatioev.workflows.batch", config, output_dir, f"{state_prefix}_batch"))
            st.rerun()
    render_worker(batch_prefix, _render_batch_outputs)


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
