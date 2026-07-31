import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-cache")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT_DIR = Path("/Users/shihongwu/SpatioEv/outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PNG_PATH = OUT_DIR / "ST_TMAG_E1_31_analysis_workflow_schematic.png"
PDF_PATH = OUT_DIR / "ST_TMAG_E1_31_analysis_workflow_schematic.pdf"


steps = [
    {
        "title": "Quantification",
        "detail": "cell x marker\nspatial coordinates",
        "color": "#4E79A7",
    },
    {
        "title": "AnnData",
        "detail": ".h5ad object\nobs / X / uns",
        "color": "#59A14F",
    },
    {
        "title": "Clustering",
        "detail": "level 0 + level 1\nper FOV review",
        "color": "#F28E2B",
    },
    {
        "title": "Clean\nAnnotation",
        "detail": "anno_broad\nanno_fine",
        "color": "#E15759",
    },
    {
        "title": "Tubule\nTerritories",
        "detail": "mask contact cores\nnearest territory",
        "color": "#76B7B2",
    },
    {
        "title": "Spatial\nMetrics",
        "detail": "composition\ndistance / density",
        "color": "#B07AA1",
    },
    {
        "title": "Neighborhood\nMotifs",
        "detail": "local niches\nco-occurrence",
        "color": "#EDC948",
    },
]


def add_box(ax, x, y, width, height, title, detail, color):
    shadow = FancyBboxPatch(
        (x + 0.035, y - 0.035),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        linewidth=0,
        facecolor="#000000",
        alpha=0.10,
        zorder=1,
    )
    ax.add_patch(shadow)

    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.025,rounding_size=0.045",
        linewidth=1.6,
        edgecolor=color,
        facecolor="#FFFFFF",
        zorder=2,
    )
    ax.add_patch(box)

    ax.text(
        x + width / 2,
        y + height * 0.67,
        title,
        ha="center",
        va="center",
        fontsize=14.5,
        fontweight="bold",
        color="#202124",
        linespacing=0.95,
        zorder=3,
    )
    ax.text(
        x + width / 2,
        y + height * 0.25,
        detail,
        ha="center",
        va="center",
        fontsize=11,
        color="#4A4F55",
        linespacing=1.2,
        zorder=3,
    )

    ax.plot(
        [x + 0.14, x + width - 0.14],
        [y + height * 0.43, y + height * 0.43],
        color=color,
        linewidth=2.2,
        solid_capstyle="round",
        zorder=3,
    )


def add_arrow(ax, x0, y0, x1, y1):
    arrow = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        arrowstyle="-|>",
        mutation_scale=19,
        linewidth=1.9,
        color="#5F6368",
        shrinkA=8,
        shrinkB=8,
        zorder=2,
    )
    ax.add_patch(arrow)


fig, ax = plt.subplots(figsize=(16, 9), facecolor="white")
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis("off")

ax.text(
    0.8,
    8.25,
    "ST_TMAG_E1_31 Spatial Analysis Workflow",
    fontsize=26,
    fontweight="bold",
    color="#202124",
    ha="left",
    va="center",
)
ax.text(
    0.8,
    7.78,
    "From cell-level marker quantification to tubule-centered spatial organization",
    fontsize=15,
    color="#5F6368",
    ha="left",
    va="center",
)

box_w = 2.45
box_h = 1.35
row1_y = 5.7
row2_y = 3.1
x_positions_row1 = [0.8, 4.0, 7.2, 10.4]
x_positions_row2 = [10.4, 7.2, 4.0]

positions = [
    (x_positions_row1[0], row1_y),
    (x_positions_row1[1], row1_y),
    (x_positions_row1[2], row1_y),
    (x_positions_row1[3], row1_y),
    (x_positions_row2[0], row2_y),
    (x_positions_row2[1], row2_y),
    (x_positions_row2[2], row2_y),
]

for step, (x, y) in zip(steps, positions):
    add_box(ax, x, y, box_w, box_h, step["title"], step["detail"], step["color"])

for i in range(3):
    x0 = positions[i][0] + box_w
    y0 = positions[i][1] + box_h / 2
    x1 = positions[i + 1][0]
    y1 = positions[i + 1][1] + box_h / 2
    add_arrow(ax, x0, y0, x1, y1)

add_arrow(
    ax,
    positions[3][0] + box_w / 2,
    positions[3][1],
    positions[4][0] + box_w / 2,
    positions[4][1] + box_h,
)

for i in [4, 5]:
    x0 = positions[i][0]
    y0 = positions[i][1] + box_h / 2
    x1 = positions[i + 1][0] + box_w
    y1 = positions[i + 1][1] + box_h / 2
    add_arrow(ax, x0, y0, x1, y1)

ax.text(
    0.8,
    1.45,
    "Core outputs for presentation",
    fontsize=14,
    fontweight="bold",
    color="#202124",
    ha="left",
)
ax.text(
    0.8,
    1.08,
    "Clean FOV maps  |  matched 3-tubule regions  |  composition and distance summaries  |  neighborhood motif shifts",
    fontsize=13,
    color="#5F6368",
    ha="left",
)

fig.savefig(PNG_PATH, dpi=300, bbox_inches="tight")
fig.savefig(PDF_PATH, bbox_inches="tight")
plt.close(fig)

print(PNG_PATH)
print(PDF_PATH)
