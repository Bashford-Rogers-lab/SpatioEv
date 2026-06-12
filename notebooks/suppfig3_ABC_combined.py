"""
Supplementary Figure 3 — Combined A + B + C
============================================
Assembles the three pre-rendered panels into one publication-ready figure:

  A — Branch-composition Muller plot + surround LOWESS (4 cell types)
  B — PanIN morphological validation scores vs epithelial pseudotime
  C — Branch–module score profile heatmap + patient trajectory fingerprint

Width: 170 mm   Height: ~230 mm (fits on A4 page)

Dependencies: pre-rendered PNGs in notebooks/results/suppfig3/
Run individual panel scripts first if they don't exist:
    python notebooks/suppfig3A_branch_surround_lowess.py
    python notebooks/suppfig3B_panin_validation.py
    python notebooks/suppfig3C_trajectory_clinical.py

Run:
    python notebooks/suppfig3_ABC_combined.py

Output: notebooks/results/suppfig3/suppfig3_combined.pdf (.png)
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as mgridspec
import matplotlib.image as mpimg
import numpy as np

matplotlib.rcParams.update({
    "font.family":  "Arial",
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
})

MM2IN      = 1 / 25.4
_HERE       = Path(__file__).resolve().parent
RESULT_DIR  = _HERE / "results" / "suppfig3"

# Pre-rendered panel PNGs (300 dpi each)
PANEL_PNGS = {
    "A": RESULT_DIR / "suppfig3A_branch_surround_lowess.png",
    "B": RESULT_DIR / "suppfig3B_panin_validation.png",
    "C": RESULT_DIR / "suppfig3C_trajectory_clinical.png",
}

# Height contributions in mm (from actual pixel measurements at 300 dpi)
PANEL_H_MM = {"A": 90, "B": 57, "C": 68}
TOTAL_CONTENT_H = sum(PANEL_H_MM.values())   # 215 mm
VGAP_MM = 4       # gap between panels
LABEL_H_MM = 3    # extra headroom above each panel for the letter label
N_PANELS = 3

FIG_W_MM = 172
FIG_H_MM = TOTAL_CONTENT_H + (N_PANELS * LABEL_H_MM) + ((N_PANELS - 1) * VGAP_MM) + 4
# ≈ 215 + 9 + 8 + 4 = 236 mm


def make_figure():
    for key, path in PANEL_PNGS.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Panel {key} PNG not found: {path}\n"
                f"Run suppfig3{key}_*.py first."
            )

    fig_w = FIG_W_MM * MM2IN
    fig_h = FIG_H_MM * MM2IN
    fig   = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    # Build height ratios including label space
    # Each panel gets LABEL_H_MM + PANEL_H_MM; gaps handled by hspace
    h_ratios = [LABEL_H_MM + PANEL_H_MM[k] for k in ("A", "B", "C")]

    gs = mgridspec.GridSpec(
        N_PANELS, 1,
        figure=fig,
        left=0.0, right=1.0,
        top=1.0, bottom=0.0,
        hspace=VGAP_MM / (FIG_H_MM / N_PANELS),
        height_ratios=h_ratios,
    )

    LABELS = ["A", "B", "C"]
    for i, key in enumerate(LABELS):
        img  = mpimg.imread(str(PANEL_PNGS[key]))
        cell = gs[i, 0]

        # Total cell height fraction
        cell_pos  = cell.get_position(fig)  # needs a draw to be accurate
        # Instead use a nested gridspec: label row + image row
        sub_gs = mgridspec.GridSpecFromSubplotSpec(
            2, 1,
            subplot_spec=cell,
            height_ratios=[LABEL_H_MM, PANEL_H_MM[key]],
            hspace=0.0,
        )

        # Invisible label axis
        ax_lab = fig.add_subplot(sub_gs[0, 0])
        ax_lab.set_axis_off()
        ax_lab.text(
            0.0, 0.0, key,
            transform=ax_lab.transAxes,
            fontsize=9, fontweight="bold",
            va="bottom", ha="left",
        )

        # Image axis
        ax_img = fig.add_subplot(sub_gs[1, 0])
        ax_img.imshow(img, aspect="auto", interpolation="lanczos")
        ax_img.set_axis_off()

    # ── Save ─────────────────────────────────────────────────────────────────────
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_pdf = RESULT_DIR / "suppfig3_combined.pdf"
    out_png = RESULT_DIR / "suppfig3_combined.png"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved:\n  {out_pdf}\n  {out_png}")
    print(f"Figure size: {fig_w*25.4:.0f} × {fig_h*25.4:.0f} mm")
    plt.show()


if __name__ == "__main__":
    make_figure()
