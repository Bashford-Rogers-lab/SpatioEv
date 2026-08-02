#!/usr/bin/env python3
"""Interactive napari application for reviewing Phenocycler marker gates.

The application reuses marker gates calculated by ``marker_gating_qc.py`` and
adds an exact, auditable manual-review layer. Images are read lazily from the
OME-TIFF pyramid; moving a gate only updates the positive-cell point layer.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "spatioev_marker_gating_review_matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "spatioev_marker_gating_review_numba"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
Path(os.environ["NUMBA_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

import anndata as ad
import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import zarr

from spatioev.workflows import marker_gating as mgq

from ._io import now as utc_now

STAINING_OPTIONS = [
    "clear_specific",
    "diffuse_background",
    "high_background_specific_tail",
    "negative_or_absent",
    "artifact_dominated",
    "failed_or_unusable",
]
COMPARTMENT_OPTIONS = [
    "membrane",
    "cytoplasmic",
    "nuclear",
    "membrane_cytoplasmic",
    "extracellular",
    "mixed_or_uncertain",
]
EXPRESSION_OPTIONS = ["bimodal", "multi_level", "broad_gradient"]
ARTIFACT_OPTIONS = ["low", "medium", "high", "severe"]


@dataclass
class MarkerReview:
    marker: str
    calculated_gate: float
    calculated_fraction: float
    current_gate: float
    current_fraction: float
    range_low: float
    range_high: float
    accepted_gate: float | None = None
    accepted_fraction: float | None = None
    status: str = "unreviewed"
    reviewed_at: str = ""




def resolve_gate_table(output_dir: Path, sample_id: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    # The calculated first pass is the neutral default. A refined table can be
    # passed explicitly when an already-reviewed sample is used as a reference.
    path = output_dir / f"{sample_id}_selected_marker_gates.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Calculated gate table not found: {path}. Run marker_gating_qc.py first "
            "or provide --gate-table."
        )
    return path


def adaptive_gate_range(values: np.ndarray, gate: float) -> tuple[float, float]:
    """Return a useful local threshold range centered around a calculated gate."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return max(0.0, gate - 1.0), gate + 1.0

    positive_fraction = float(np.mean(x >= gate))
    # Span a broad empirical positivity interval while preserving resolution
    # around the calculated gate. The explicit percentage allowance matters for
    # rare markers, where a simple fold-change remains too restrictive.
    lower_target_fraction = min(0.80, max(positive_fraction * 4.0, positive_fraction + 0.10))
    upper_target_fraction = max(0.0001, positive_fraction * 0.15)
    candidate_low = float(np.quantile(x, 1.0 - lower_target_fraction))
    candidate_high = float(np.quantile(x, 1.0 - upper_target_fraction))

    distance = 1.25 * max(gate - candidate_low, candidate_high - gate, 0.05)
    low = max(float(np.min(x)), gate - distance)
    high = min(float(np.max(x)), gate + distance)
    if high - low < 1e-5:
        low = max(0.0, gate - 0.1)
        high = gate + 0.1
    return low, high


def gate_to_slider(gate: float, low: float, high: float) -> int:
    return int(np.clip(round(1000 * (gate - low) / (high - low)), 0, 1000))


def slider_to_gate(value: int, low: float, high: float) -> float:
    return float(low + (high - low) * value / 1000.0)


class PyramidImage:
    """Keep one OME-TIFF and its lazy pyramid stores open for napari."""

    def __init__(self, path: Path, fallback_markers: list[str] | None = None):
        self.path = path
        self.tif = tifffile.TiffFile(path)
        self.series = self.tif.series[0]
        if "C" not in self.series.axes or not self.series.axes.endswith("YX"):
            raise ValueError(f"Expected a CYX-compatible image, found axes={self.series.axes!r}")
        self.channels = mgq.canonical_channel_names(path, fallback_markers)
        self.stores = []
        self.levels = []
        for level in self.series.levels:
            store = level.aszarr()
            root = zarr.open(store, mode="r")
            array = root["0"] if isinstance(root, zarr.hierarchy.Group) else root
            self.stores.append(store)
            self.levels.append(array)

    def channel_data(self, channel_index: int) -> list[da.Array]:
        return [da.from_zarr(level)[channel_index] for level in self.levels]

    def contrast_limits(self, channel_index: int) -> tuple[float, float]:
        small = np.asarray(self.levels[-1][channel_index], dtype=float)
        finite = small[np.isfinite(small)]
        if len(finite) == 0:
            return 0.0, 1.0
        low, high = np.percentile(finite, [0.5, 99.8])
        return float(low), float(max(high, low + 1.0))

    def close(self) -> None:
        for store in self.stores:
            close = getattr(store, "close", None)
            if close is not None:
                close()
        self.tif.close()


class MarkerRow:
    """Small adapter around one marker's controls in the scrollable list."""

    def __init__(self, marker: str, parent, on_select, on_change, on_accept):
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import (
            QFrame,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QSlider,
            QVBoxLayout,
        )

        self.marker = marker
        self.frame = QFrame(parent)
        self.frame.setObjectName("markerRow")
        outer = QVBoxLayout(self.frame)
        outer.setContentsMargins(8, 6, 8, 6)
        outer.setSpacing(3)

        header = QHBoxLayout()
        self.name = QPushButton(marker)
        self.name.setObjectName("markerName")
        self.name.setToolTip(f"Show {marker} and its positive-cell overlay")
        self.name.clicked.connect(lambda: on_select(marker))
        self.fraction = QLabel("0.0%")
        self.fraction.setMinimumWidth(48)
        self.fraction.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.status = QLabel("unreviewed")
        self.status.setMinimumWidth(76)
        self.status.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        header.addWidget(self.name, 1)
        header.addWidget(self.fraction)
        header.addWidget(self.status)
        outer.addLayout(header)

        slider_line = QHBoxLayout()
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setTracking(True)
        self.slider.valueChanged.connect(lambda value: on_change(marker, value))
        self.calculated = QLabel("calc --")
        self.calculated.setMinimumWidth(76)
        self.accept = QPushButton("Accept")
        self.accept.setToolTip("Accept this exact gate and continue")
        self.accept.clicked.connect(lambda: on_accept(marker))
        slider_line.addWidget(self.slider, 1)
        slider_line.addWidget(self.calculated)
        slider_line.addWidget(self.accept)
        outer.addLayout(slider_line)

    def set_active(self, active: bool) -> None:
        self.frame.setProperty("active", active)
        self.frame.style().unpolish(self.frame)
        self.frame.style().polish(self.frame)

    def set_status(self, status: str) -> None:
        self.status.setText(status)
        self.frame.setProperty("reviewStatus", status)
        self.frame.style().unpolish(self.frame)
        self.frame.style().polish(self.frame)


class MarkerGatingController:
    def __init__(
        self,
        viewer,
        *,
        sample_id: str,
        adata_path: Path,
        image_path: Path,
        condition_path: Path,
        gate_table_path: Path,
        output_dir: Path,
        layer: str | None,
        imageid: str | None = None,
    ):
        from qtpy.QtCore import QTimer

        self.viewer = viewer
        self.sample_id = sample_id
        self.adata_path = adata_path
        self.image_path = image_path
        self.condition_path = condition_path
        self.gate_table_path = gate_table_path
        self.output_dir = output_dir
        self.layer = layer
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.adata = mgq.load_adata(adata_path)
        self.conditions = mgq.read_marker_conditions(condition_path)
        for column in ["expected_positive_min", "expected_positive_max"]:
            if column not in self.conditions:
                self.conditions[column] = np.nan
        self.gates = pd.read_csv(gate_table_path)
        self.gates = self.gates.loc[self.gates["marker"].isin(self.adata.var_names)].copy()
        nuclear_markers = {"HOECHST", "HOECHST2", "DAPI", "DNA_1"}
        gate_markers = self.gates["marker"].astype(str).str.upper()
        self.gates = self.gates.loc[~gate_markers.isin(nuclear_markers)].reset_index(drop=True)
        if self.gates.empty:
            raise ValueError("Gate table has no markers matching the AnnData matrix")

        self.xmat = mgq.expression_matrix(self.adata, layer=layer)
        self.marker_index = {str(marker): i for i, marker in enumerate(self.adata.var_names)}
        self.log_values: dict[str, np.ndarray] = {}
        self.reviews: dict[str, MarkerReview] = {}
        for row in self.gates.itertuples(index=False):
            marker = str(row.marker)
            values = np.log1p(np.clip(np.asarray(self.xmat[:, self.marker_index[marker]], dtype=float), 0, None))
            self.log_values[marker] = values
            gate = float(row.selected_log1p_gate)
            fraction = float(np.mean(values >= gate))
            low, high = adaptive_gate_range(values, gate)
            self.reviews[marker] = MarkerReview(
                marker=marker,
                calculated_gate=gate,
                calculated_fraction=fraction,
                current_gate=gate,
                current_fraction=fraction,
                range_low=low,
                range_high=high,
            )

        required_coordinates = {"Y_centroid", "X_centroid"}
        missing_coordinates = required_coordinates - set(self.adata.obs.columns)
        if missing_coordinates:
            raise ValueError(f"AnnData is missing spatial coordinates: {sorted(missing_coordinates)}")
        self.display_mask = np.ones(self.adata.n_obs, dtype=bool)
        if imageid is not None:
            if "imageid" not in self.adata.obs:
                raise KeyError("AnnData has no 'imageid' column for FOV-specific review")
            self.display_mask = self.adata.obs["imageid"].astype(str).eq(str(imageid)).to_numpy()
            if not self.display_mask.any():
                raise ValueError(f"No cells have imageid={imageid!r}")
        self.coordinates = self.adata.obs.loc[
            self.display_mask, ["Y_centroid", "X_centroid"]
        ].to_numpy(dtype=float)

        self.pyramid = PyramidImage(image_path, list(self.adata.var_names.astype(str)))
        self.channel_index = {marker: i for i, marker in enumerate(self.pyramid.channels)}
        missing_channels = sorted(set(self.reviews) - set(self.channel_index))
        if missing_channels:
            raise ValueError(f"Image is missing marker channels: {missing_channels}")

        self.rows: dict[str, MarkerRow] = {}
        self.active_marker = next(iter(self.reviews))
        self._updating_controls = False
        self.overlay_timer = QTimer()
        self.overlay_timer.setSingleShot(True)
        self.overlay_timer.setInterval(140)
        self.overlay_timer.timeout.connect(self.update_overlay)

        self._add_image_layers()
        self.panel = self._build_panel()
        self.select_marker(self.active_marker)
        self.viewer.window._qt_window.destroyed.connect(self.pyramid.close)

    def _add_image_layers(self) -> None:
        hoechst_name = "HOECHST2" if "HOECHST2" in self.channel_index else self.pyramid.channels[0]
        hoechst_idx = self.channel_index[hoechst_name]
        self.hoechst_layer = self.viewer.add_image(
            self.pyramid.channel_data(hoechst_idx),
            name="Hoechst",
            multiscale=True,
            colormap="blue",
            blending="additive",
            contrast_limits=self.pyramid.contrast_limits(hoechst_idx),
        )
        marker_idx = self.channel_index[self.active_marker]
        self.marker_layer = self.viewer.add_image(
            self.pyramid.channel_data(marker_idx),
            name=self.active_marker,
            multiscale=True,
            colormap="red",
            blending="additive",
            contrast_limits=self.pyramid.contrast_limits(marker_idx),
        )
        point_kwargs = {
            "name": "Gate-positive cells",
            "size": 9,
            "face_color": "#25e6d5",
            "opacity": 0.82,
            "blending": "translucent",
        }
        try:
            # napari <=0.4 uses edge_* names for point outlines.
            self.positive_layer = self.viewer.add_points(
                np.empty((0, 2)), edge_width=0, **point_kwargs
            )
        except TypeError as exc:
            if "edge_width" not in str(exc):
                raise
            # Newer napari releases renamed the outline API to border_*.
            self.positive_layer = self.viewer.add_points(
                np.empty((0, 2)), border_width=0, **point_kwargs
            )

    def _build_panel(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFrame,
            QGridLayout,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QVBoxLayout,
            QWidget,
        )

        panel = QWidget()
        panel.setObjectName("gatePanel")
        panel.setMinimumWidth(480)
        panel.setStyleSheet(
            """
            QWidget { font-size: 12px; }
            QWidget#gatePanel { background: #f4f5f6; color: #182026; }
            QFrame#markerRow { border-bottom: 1px solid #3b3b3b; background: #252525; }
            QFrame#markerRow[active="true"] { background: #243b3a; border-left: 3px solid #25e6d5; }
            QFrame#markerRow QLabel { color: #d8dde0; }
            QFrame#markerRow[reviewStatus="accepted"] QLabel { color: #8ddf9a; }
            QFrame#markerRow[reviewStatus="adjusted"] QLabel { color: #f1c96b; }
            QFrame#markerRow[reviewStatus="flagged"] QLabel { color: #ff8989; }
            QPushButton#markerName { text-align: left; border: 0; font-weight: 600; color: #f4f6f7; }
            QLineEdit, QComboBox, QDoubleSpinBox { min-height: 25px; }
            QPushButton { min-height: 25px; }
            """
        )
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)

        title = QLabel(f"{self.sample_id} marker gate review")
        title.setStyleSheet("font-size: 17px; font-weight: 650;")
        layout.addWidget(title)
        self.progress_label = QLabel("")
        layout.addWidget(self.progress_label)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Search markers")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self.filter_rows)
        layout.addWidget(self.search)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_body = QWidget()
        self.rows_layout = QVBoxLayout(scroll_body)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(0)
        self._updating_controls = True
        for marker, review in self.reviews.items():
            row = MarkerRow(marker, scroll_body, self.select_marker, self.slider_changed, self.accept_marker)
            row.calculated.setText(f"calc {review.calculated_gate:.2f}")
            self.rows[marker] = row
            row.slider.setValue(gate_to_slider(review.current_gate, review.range_low, review.range_high))
            row.fraction.setText(f"{review.current_fraction:.1%}")
            self.rows_layout.addWidget(row.frame)
        self._updating_controls = False
        self.rows_layout.addStretch(1)
        scroll.setWidget(scroll_body)
        layout.addWidget(scroll, 3)

        detail = QFrame()
        detail.setFrameShape(QFrame.Shape.StyledPanel)
        detail_layout = QVBoxLayout(detail)
        self.active_title = QLabel("")
        self.active_title.setStyleSheet("font-size: 15px; font-weight: 650;")
        self.gate_detail = QLabel("")
        detail_layout.addWidget(self.active_title)
        detail_layout.addWidget(self.gate_detail)

        self.figure = Figure(figsize=(4.5, 1.75), tight_layout=True, facecolor="#ffffff")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.hist_ax = self.figure.add_subplot(111)
        self.hist_gate_line = None
        detail_layout.addWidget(self.canvas)

        condition_grid = QGridLayout()
        self.staining_combo = QComboBox()
        self.staining_combo.addItems(STAINING_OPTIONS)
        self.compartment_combo = QComboBox()
        self.compartment_combo.addItems(COMPARTMENT_OPTIONS)
        self.expression_combo = QComboBox()
        self.expression_combo.addItems(EXPRESSION_OPTIONS)
        self.artifact_combo = QComboBox()
        self.artifact_combo.addItems(ARTIFACT_OPTIONS)
        condition_grid.addWidget(QLabel("Staining"), 0, 0)
        condition_grid.addWidget(self.staining_combo, 0, 1)
        condition_grid.addWidget(QLabel("Compartment"), 1, 0)
        condition_grid.addWidget(self.compartment_combo, 1, 1)
        condition_grid.addWidget(QLabel("Expression"), 2, 0)
        condition_grid.addWidget(self.expression_combo, 2, 1)
        condition_grid.addWidget(QLabel("Artifact"), 3, 0)
        condition_grid.addWidget(self.artifact_combo, 3, 1)
        self.expected_min = QDoubleSpinBox()
        self.expected_max = QDoubleSpinBox()
        for spin in [self.expected_min, self.expected_max]:
            spin.setRange(0.0, 100.0)
            spin.setDecimals(1)
            spin.setSuffix("%")
            spin.setSpecialValueText("Not set")
        condition_grid.addWidget(QLabel("Expected positivity"), 4, 0)
        expected_line = QHBoxLayout()
        expected_line.addWidget(self.expected_min)
        expected_line.addWidget(QLabel("to"))
        expected_line.addWidget(self.expected_max)
        condition_grid.addLayout(expected_line, 4, 1)
        detail_layout.addLayout(condition_grid)

        for widget in [
            self.staining_combo,
            self.compartment_combo,
            self.expression_combo,
            self.artifact_combo,
        ]:
            widget.currentTextChanged.connect(self.condition_changed)
        self.expected_min.valueChanged.connect(self.condition_changed)
        self.expected_max.valueChanged.connect(self.condition_changed)

        nav = QHBoxLayout()
        previous_button = QPushButton("Previous")
        previous_button.clicked.connect(lambda: self.navigate(-1))
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset_active)
        expand_button = QPushButton("Expand")
        expand_button.setToolTip("Extend this marker's gate range in both directions")
        expand_button.clicked.connect(self.expand_active_range)
        flag_button = QPushButton("Flag")
        flag_button.clicked.connect(self.flag_active)
        accept_button = QPushButton("Accept and next")
        accept_button.setStyleSheet("font-weight: 650; background: #176c63;")
        accept_button.clicked.connect(lambda: self.accept_marker(self.active_marker))
        next_button = QPushButton("Next")
        next_button.clicked.connect(lambda: self.navigate(1))
        nav.addWidget(previous_button)
        nav.addWidget(reset_button)
        nav.addWidget(expand_button)
        nav.addWidget(flag_button)
        nav.addWidget(accept_button, 1)
        nav.addWidget(next_button)
        detail_layout.addLayout(nav)
        layout.addWidget(detail, 2)

        export_line = QHBoxLayout()
        self.write_h5ad = QCheckBox("Write gated h5ad")
        self.write_h5ad.setStyleSheet(
            "QCheckBox { color: #182026; font-weight: 600; spacing: 6px; }"
        )
        self.write_h5ad.setToolTip(
            "Write a new AnnData file containing gate-positive, margin, and log1p columns"
        )
        export_button = QPushButton("Export reviewed gates")
        export_button.clicked.connect(self.export_review)
        export_line.addWidget(self.write_h5ad)
        export_line.addStretch(1)
        export_line.addWidget(export_button)
        layout.addLayout(export_line)
        self.export_message = QMessageBox
        self.update_progress()
        return panel

    def filter_rows(self, text: str) -> None:
        query = text.strip().lower()
        for marker, row in self.rows.items():
            row.frame.setVisible(query in marker.lower())

    def select_marker(self, marker: str) -> None:
        if marker not in self.reviews:
            return
        self.active_marker = marker
        for name, row in self.rows.items():
            row.set_active(name == marker)

        channel_idx = self.channel_index[marker]
        self.marker_layer.data = self.pyramid.channel_data(channel_idx)
        self.marker_layer.name = marker
        self.marker_layer.contrast_limits = self.pyramid.contrast_limits(channel_idx)
        self._load_condition_controls(marker)
        self._draw_histogram(marker)
        self.update_gate_detail()
        self.update_overlay()

    def slider_changed(self, marker: str, slider_value: int) -> None:
        if self._updating_controls:
            return
        if marker != self.active_marker:
            self.select_marker(marker)
        review = self.reviews[marker]
        review.current_gate = slider_to_gate(slider_value, review.range_low, review.range_high)
        review.current_fraction = float(np.mean(self.log_values[marker] >= review.current_gate))
        if review.accepted_gate is not None and not np.isclose(review.current_gate, review.accepted_gate, atol=1e-5):
            review.status = "unreviewed"
            self.rows[marker].set_status(review.status)
        self.rows[marker].fraction.setText(f"{review.current_fraction:.1%}")
        self.update_gate_detail()
        if self.hist_gate_line is not None:
            self.hist_gate_line.set_xdata([review.current_gate, review.current_gate])
            self.canvas.draw_idle()
        self.overlay_timer.start()

    def update_overlay(self) -> None:
        review = self.reviews[self.active_marker]
        positive = self.log_values[self.active_marker] >= review.current_gate
        self.positive_layer.data = self.coordinates[positive[self.display_mask]]
        self.positive_layer.name = f"{self.active_marker}+ cells ({positive.mean():.1%})"

    def _draw_histogram(self, marker: str) -> None:
        review = self.reviews[marker]
        values = self.log_values[marker]
        self.hist_ax.clear()
        self.hist_ax.hist(values, bins=100, color="#8b9298", edgecolor="none")
        self.hist_ax.axvline(review.calculated_gate, color="#d8af4f", linewidth=1.4, linestyle="--")
        self.hist_gate_line = self.hist_ax.axvline(review.current_gate, color="#25bfb3", linewidth=2)
        self.hist_ax.set_xlim(review.range_low, review.range_high)
        self.hist_ax.set_yticks([])
        self.hist_ax.set_xlabel("log1p cell expression")
        self.hist_ax.spines[["top", "right", "left"]].set_visible(False)
        self.canvas.draw_idle()

    def update_gate_detail(self) -> None:
        review = self.reviews[self.active_marker]
        self.active_title.setText(self.active_marker)
        self.gate_detail.setText(
            f"Gate {review.current_gate:.4f} log1p / {np.expm1(review.current_gate):.3f} raw    "
            f"Positive {review.current_fraction:.2%} ({int(round(review.current_fraction * len(self.adata))):,} cells)"
        )

    def _condition_row_index(self, marker: str) -> int | None:
        matches = np.flatnonzero(self.conditions["marker"].astype(str).to_numpy() == marker)
        return int(matches[0]) if len(matches) else None

    def _load_condition_controls(self, marker: str) -> None:
        index = self._condition_row_index(marker)
        if index is None:
            return
        row = self.conditions.iloc[index]
        self._updating_controls = True
        for combo, column in [
            (self.staining_combo, "staining_condition"),
            (self.compartment_combo, "compartment_pattern"),
            (self.expression_combo, "expression_condition"),
            (self.artifact_combo, "artifact_level"),
        ]:
            value = str(row[column])
            if combo.findText(value) < 0:
                combo.addItem(value)
            combo.setCurrentText(value)
        minimum = row.get("expected_positive_min", np.nan)
        maximum = row.get("expected_positive_max", np.nan)
        self.expected_min.setValue(0.0 if pd.isna(minimum) else float(minimum) * 100.0)
        self.expected_max.setValue(0.0 if pd.isna(maximum) else float(maximum) * 100.0)
        self._updating_controls = False

    def condition_changed(self, *_args) -> None:
        if self._updating_controls:
            return
        index = self._condition_row_index(self.active_marker)
        if index is None:
            return
        self.conditions.loc[index, "staining_condition"] = self.staining_combo.currentText()
        self.conditions.loc[index, "compartment_pattern"] = self.compartment_combo.currentText()
        self.conditions.loc[index, "expression_condition"] = self.expression_combo.currentText()
        self.conditions.loc[index, "artifact_level"] = self.artifact_combo.currentText()
        self.conditions.loc[index, "expected_positive_min"] = self.expected_min.value() / 100.0 or np.nan
        self.conditions.loc[index, "expected_positive_max"] = self.expected_max.value() / 100.0 or np.nan

    def accept_marker(self, marker: str) -> None:
        review = self.reviews[marker]
        review.accepted_gate = review.current_gate
        review.accepted_fraction = review.current_fraction
        review.reviewed_at = utc_now()
        tolerance = max(1e-5, (review.range_high - review.range_low) / 1000.0)
        review.status = "accepted" if abs(review.current_gate - review.calculated_gate) <= tolerance else "adjusted"
        self.rows[marker].set_status(review.status)
        self.autosave_session()
        self.update_progress()
        self.navigate_to_next_unreviewed(marker)

    def reset_active(self) -> None:
        review = self.reviews[self.active_marker]
        review.current_gate = review.calculated_gate
        review.current_fraction = review.calculated_fraction
        review.accepted_gate = None
        review.accepted_fraction = None
        review.reviewed_at = ""
        review.status = "unreviewed"
        self.rows[self.active_marker].set_status(review.status)
        self._updating_controls = True
        self.rows[self.active_marker].slider.setValue(
            gate_to_slider(review.current_gate, review.range_low, review.range_high)
        )
        self._updating_controls = False
        self.rows[self.active_marker].fraction.setText(f"{review.current_fraction:.1%}")
        self._draw_histogram(self.active_marker)
        self.update_gate_detail()
        self.update_overlay()
        self.update_progress()

    def expand_active_range(self) -> None:
        review = self.reviews[self.active_marker]
        values = self.log_values[self.active_marker]
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return
        span = max(review.range_high - review.range_low, 0.1)
        review.range_low = max(float(np.min(finite)), review.range_low - span)
        review.range_high = min(float(np.max(finite)), review.range_high + span)
        self._updating_controls = True
        self.rows[self.active_marker].slider.setValue(
            gate_to_slider(review.current_gate, review.range_low, review.range_high)
        )
        self._updating_controls = False
        self._draw_histogram(self.active_marker)
        self.update_gate_detail()

    def flag_active(self) -> None:
        review = self.reviews[self.active_marker]
        review.status = "flagged"
        review.reviewed_at = utc_now()
        self.rows[self.active_marker].set_status("flagged")
        self.autosave_session()
        self.update_progress()
        self.navigate_to_next_unreviewed(self.active_marker)

    def navigate(self, direction: int) -> None:
        markers = list(self.reviews)
        index = markers.index(self.active_marker)
        self.select_marker(markers[(index + direction) % len(markers)])

    def navigate_to_next_unreviewed(self, marker: str) -> None:
        markers = list(self.reviews)
        start = markers.index(marker)
        for offset in range(1, len(markers) + 1):
            candidate = markers[(start + offset) % len(markers)]
            if self.reviews[candidate].status == "unreviewed":
                self.select_marker(candidate)
                return

    def update_progress(self) -> None:
        statuses = pd.Series([review.status for review in self.reviews.values()]).value_counts()
        completed = int(statuses.get("accepted", 0) + statuses.get("adjusted", 0) + statuses.get("flagged", 0))
        self.progress_label.setText(
            f"Reviewed {completed}/{len(self.reviews)}    Accepted {int(statuses.get('accepted', 0))}    "
            f"Adjusted {int(statuses.get('adjusted', 0))}    Flagged {int(statuses.get('flagged', 0))}"
        )

    def review_table(self) -> pd.DataFrame:
        rows = []
        for review in self.reviews.values():
            final_gate = review.accepted_gate if review.accepted_gate is not None else review.current_gate
            final_fraction = (
                review.accepted_fraction if review.accepted_fraction is not None else review.current_fraction
            )
            rows.append(
                {
                    "marker": review.marker,
                    "calculated_log1p_gate": review.calculated_gate,
                    "calculated_raw_gate": float(np.expm1(review.calculated_gate)),
                    "calculated_positive_fraction": review.calculated_fraction,
                    "final_log1p_gate": final_gate,
                    "final_raw_gate": float(np.expm1(final_gate)),
                    "final_positive_fraction": final_fraction,
                    "review_status": review.status,
                    "gate_changed": not np.isclose(final_gate, review.calculated_gate, atol=1e-5),
                    "reviewed_at": review.reviewed_at,
                }
            )
        return pd.DataFrame(rows)

    def autosave_session(self) -> Path:
        path = self.output_dir / f"{self.sample_id}_interactive_gate_review_session.csv"
        self.review_table().to_csv(path, index=False)
        return path

    def final_gate_table(self) -> pd.DataFrame:
        final = self.gates.copy()
        review = self.review_table().set_index("marker")
        for index, row in final.iterrows():
            marker = str(row["marker"])
            if marker not in review.index:
                continue
            final.loc[index, "calculated_log1p_gate"] = review.loc[marker, "calculated_log1p_gate"]
            final.loc[index, "calculated_positive_fraction"] = review.loc[
                marker, "calculated_positive_fraction"
            ]
            final.loc[index, "selected_log1p_gate"] = review.loc[marker, "final_log1p_gate"]
            final.loc[index, "selected_raw_gate"] = review.loc[marker, "final_raw_gate"]
            final.loc[index, "selected_positive_fraction"] = review.loc[
                marker, "final_positive_fraction"
            ]
            final.loc[index, "review_status"] = review.loc[marker, "review_status"]
            final.loc[index, "interactive_gate_changed"] = review.loc[marker, "gate_changed"]
            final.loc[index, "interactive_reviewed_at"] = review.loc[marker, "reviewed_at"]
            if bool(review.loc[marker, "gate_changed"]):
                final.loc[index, "gate_source"] = "interactive napari review"
        return final

    def export_review(self) -> None:
        from qtpy.QtWidgets import QMessageBox

        unreviewed = [marker for marker, review in self.reviews.items() if review.status == "unreviewed"]
        if unreviewed:
            answer = QMessageBox.question(
                self.panel,
                "Export draft review?",
                f"{len(unreviewed)} markers are still unreviewed. Export a draft using their current gates?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        session_path = self.autosave_session()
        condition_path = self.output_dir / f"{self.sample_id}_marker_conditions_interactive.csv"
        gates_path = self.output_dir / f"{self.sample_id}_selected_marker_gates_interactive.csv"
        scimap_path = self.output_dir / f"{self.sample_id}_scimap_manual_gates_interactive.csv"
        manifest_path = self.output_dir / f"{self.sample_id}_interactive_gate_review_manifest.json"
        final = self.final_gate_table()
        self.conditions.to_csv(condition_path, index=False)
        final.to_csv(gates_path, index=False)
        mgq.write_scimap_manual_gates(final, scimap_path)

        h5ad_path = None
        if self.write_h5ad.isChecked():
            h5ad_path = self.output_dir / f"{self.sample_id}_autogated_interactive.h5ad"
            mgq.apply_selected_gates(self.adata, final, layer=self.layer).write_h5ad(h5ad_path)

        manifest = {
            "sample_id": self.sample_id,
            "created_at": utc_now(),
            "adata_path": str(self.adata_path),
            "image_path": str(self.image_path),
            "source_gate_table": str(self.gate_table_path),
            "review_session": str(session_path),
            "marker_conditions": str(condition_path),
            "selected_gates": str(gates_path),
            "scimap_manual_gates": str(scimap_path),
            "gated_h5ad": str(h5ad_path) if h5ad_path else None,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        QMessageBox.information(
            self.panel,
            "Gate review exported",
            f"Wrote reviewed gates for {self.sample_id} to:\n{gates_path}",
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-id", default="sample")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--adata-path", type=Path)
    parser.add_argument("--image-path", type=Path)
    parser.add_argument("--condition-path", type=Path)
    parser.add_argument("--gate-table", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--layer", default=None, help="AnnData layer; use 'raw' for adata.raw")
    parser.add_argument("--imageid", default=None, help="Only display cells from this image/FOV")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print a summary without launching napari.",
    )
    return parser


def resolve_paths(args) -> dict[str, Path]:
    defaults = mgq.default_sample_paths(args.sample_id, args.project_root)
    output_dir = args.output_dir or defaults.output_dir
    return {
        "adata_path": args.adata_path or defaults.adata_path,
        "image_path": args.image_path or defaults.image_path,
        "condition_path": args.condition_path or defaults.marker_condition_path,
        "output_dir": output_dir,
        "gate_table_path": resolve_gate_table(output_dir, args.sample_id, args.gate_table),
    }


def validate_inputs(args, paths: dict[str, Path]) -> dict[str, object]:
    for name in ["adata_path", "image_path", "condition_path", "gate_table_path"]:
        if not paths[name].exists():
            raise FileNotFoundError(f"{name} does not exist: {paths[name]}")
    adata = ad.read_h5ad(paths["adata_path"], backed="r")
    conditions = mgq.read_marker_conditions(paths["condition_path"])
    gates = pd.read_csv(paths["gate_table_path"])
    adata_markers = [str(marker) for marker in adata.var_names]
    channels = mgq.canonical_channel_names(paths["image_path"], adata_markers)
    nuclear_markers = {"HOECHST", "HOECHST2", "DAPI", "DNA_1"}
    markers = [
        marker
        for marker in gates["marker"].astype(str)
        if marker.upper() not in nuclear_markers
    ]
    summary = {
        "sample_id": args.sample_id,
        "cells": int(adata.n_obs),
        "adata_markers": int(adata.n_vars),
        "review_markers": len(markers),
        "condition_rows": len(conditions),
        "image_channels": len(channels),
        "missing_in_adata": sorted(set(markers) - set(adata.var_names)),
        "missing_in_image": sorted(set(markers) - set(channels)),
        **{name: str(path) for name, path in paths.items()},
    }
    adata.file.close()
    return summary


def main() -> None:
    args = build_parser().parse_args()
    paths = resolve_paths(args)
    summary = validate_inputs(args, paths)
    if args.validate_only:
        print(json.dumps(summary, indent=2))
        return

    import napari

    viewer = napari.Viewer(title=f"{args.sample_id} marker autogating review")
    controller = MarkerGatingController(
        viewer,
        sample_id=args.sample_id,
        layer=args.layer,
        imageid=args.imageid,
        **paths,
    )
    viewer.window.add_dock_widget(controller.panel, name="Marker gate review", area="right")
    viewer.window._qt_window.resize(1680, 980)
    napari.run()


if __name__ == "__main__":
    main()
