"""Shared plotting style for SpatioEv manuscript figures."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


PSEUDOTIME_CMAP = "viridis"
BRANCH_CMAP = "tab20"

DISEASE_COLORS = {
    "NormalPancreas": "#2f4858",
    "PDAC": "#a63d40",
}

SAMPLE_COLORS = {
    "40331_1": "#2f4858",
    "34434_1": "#33658a",
    "33694_1": "#f26419",
    "35559_1": "#7a5195",
}

XENIUM_SAMPLE_COLORS = {
    "normal_nondiseased_v1": "#2f4858",
    "pdac_pancreas_v1": "#33658a",
    "pdac_addon_v1": "#f26419",
    "pdac_io_v1": "#7a5195",
}

PHENOTYPE_COLORS = {
    "pancreatic ductal epithelium": "#2f6fbd",
    "Fibroblasts": "#2a9d8f",
    "Vimentin only mesenchyme": "#b07aa1",
    "Endothelial cells": "#59a14f",
    "pancreatic acinar epithelium": "#f28e2b",
    "T cells": "#d64f4f",
    "B lineage": "#edc948",
    "Muscularis externa": "#9c755f",
    "Vascular smooth muscle": "#76b7b2",
    "Nerves": "#8e6bbf",
    "Myeloid cells": "#7f7f7f",
    "Islets": "#1b9e77",
    "Mucosa gland": "#e7298a",
    "Unknown": "#bdbdbd",
    "noise": "#bab0ab",
}

MODULE_COLORS = {
    "early duct": "#2f4858",
    "PanIN-like": "#e17c05",
    "architecture": "#6a994e",
    "invasion/desmoplasia": "#8c1d40",
    "proliferation": "#ca6702",
    "dedifferentiation": "#6a4c93",
}


def configure() -> None:
    sns.set_theme(context="paper", style="white")
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 600,
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.3,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.3,
            "ytick.labelsize": 6.3,
            "legend.fontsize": 6.2,
            "axes.linewidth": 0.65,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def panel_letter(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.10) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="#111111",
    )


def clean_spatial_axis(ax: plt.Axes) -> None:
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def clean_axis(ax: plt.Axes) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def shorten(text: object, width: int = 30) -> str:
    value = str(text).replace("__", " ").replace("_", " ")
    return value if len(value) <= width else value[: width - 1] + "..."


def add_scale_bar(ax: plt.Axes, length_um: float = 500, pixel_size_um: float = 0.325) -> None:
    """Add a simple lower-right scale bar to a spatial plot.

    Coordinates are assumed to be pixels for multiplexed imaging. Use
    ``pixel_size_um=1`` when coordinates are already micrometers.
    """

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    width = abs(xlim[1] - xlim[0])
    height = abs(ylim[1] - ylim[0])
    length_data = length_um / pixel_size_um
    x0 = min(xlim) + width * 0.66
    y0 = max(ylim) - height * 0.06
    ax.plot([x0, x0 + length_data], [y0, y0], color="#111111", lw=1.5, solid_capstyle="butt")
    ax.text(
        x0 + length_data / 2,
        y0 - height * 0.025,
        f"{int(length_um)} um",
        ha="center",
        va="top",
        fontsize=6.2,
        color="#111111",
    )


def export_figure(fig: plt.Figure, out_base: Path, dpi: int = 600) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(out_base.with_suffix(f".{ext}"), bbox_inches="tight", facecolor="white", dpi=dpi)

