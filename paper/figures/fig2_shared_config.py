"""
Shared constants for Figure 2 scripts (combined 4-sample pseudotime).
Import with:  from fig2_shared_config import *
"""

from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT      = Path("/Users/shihongwu/SpatioEv")
CACHE_DIR = ROOT / "data" / "combined_exp_2_3_4_5"
OUT_DIR   = ROOT  / "paper" / "notebooks" / "results" / "fig2"

MM2IN = 1 / 25.4

# ── Palettes ──────────────────────────────────────────────────────────────────
DISEASE_PALETTE = {"NormalPancreas": "#4CAF50", "PDAC": "#D32F2F"}

SAMPLE_PALETTE = {
    "40331_1": "#4CAF50",   # normal
    "34434_1": "#1565C0",   # PDAC exp_2 (invasive)
    "33694_1": "#E65100",   # PDAC exp_3 (intermediate)
    "35559_1": "#7B1FA2",   # PDAC exp_4 (early PanIN-like)
}

SAMPLE_LABELS = {
    "40331_1": "Normal\npancreas",
    "34434_1": "PDAC\n(invasive)",
    "33694_1": "PDAC\n(intermediate)",
    "35559_1": "PDAC\n(early PanIN)",
}

DIFF_ORDER = ["NormalPancreas", "PDAC_exp_4", "PDAC_exp_3", "PDAC_exp_2"]
DIFF_LABEL = {
    "NormalPancreas": "Normal",
    "PDAC_exp_4":     "PDAC early",
    "PDAC_exp_3":     "PDAC inter.",
    "PDAC_exp_2":     "PDAC invasive",
}

# Ordered for trajectory display: normal first
SAMPLE_ORDER = ["40331_1", "35559_1", "33694_1", "34434_1"]

# ── Module columns ────────────────────────────────────────────────────────────
MODULE_COLS = [
    "pdac_early_duct_anchor_score",
    "pdac_panin_like_dysplasia_score",
    "pdac_invasive_gland_forming_score",
    "pdac_invasion_desmoplasia_axis",
    "pdac_proliferation_axis",
    "pdac_dedifferentiation_axis",
]

MODULE_LABELS = {
    "pdac_early_duct_anchor_score":      "Early-duct\nanchor",
    "pdac_panin_like_dysplasia_score":   "PanIN-like\ndysplasia",
    "pdac_invasive_gland_forming_score": "Invasive\ngland-forming",
    "pdac_invasion_desmoplasia_axis":    "Invasion–\ndesmoplasia",
    "pdac_proliferation_axis":           "Proliferation",
    "pdac_dedifferentiation_axis":       "Dedifferentiation",
}

# ── Branch context features ───────────────────────────────────────────────────
BRANCH_CONTEXT_FEATURES = [
    "surround_prop__pancreatic_ductal_epithelium",
    "surround_prop__Fibroblasts",
    "surround_prop__T_cells",
    "surround_prop__B_lineage",
    "surround_prop__Endothelial_cells",
    "surround_prop__Vimentin_only_mesenchyme",
    "surround_prop__pancreatic_acinar_epithelium",
]
BRANCH_CONTEXT_LABELS = {
    "surround_prop__pancreatic_ductal_epithelium": "Ductal (self)",
    "surround_prop__Fibroblasts":                  "Fibroblasts",
    "surround_prop__T_cells":                      "T cells",
    "surround_prop__B_lineage":                    "B lineage",
    "surround_prop__Endothelial_cells":            "Endothelial",
    "surround_prop__Vimentin_only_mesenchyme":     "Mesenchyme",
    "surround_prop__pancreatic_acinar_epithelium": "Acinar",
}
BRANCH_CONTEXT_TREND_FEATURES = [
    "surround_prop__pancreatic_ductal_epithelium",
    "surround_prop__Fibroblasts",
    "surround_prop__T_cells",
    "surround_prop__B_lineage",
]
BRANCH_FIBRO_MARKER_FEATURES = [
    "surround__Fibroblasts__FAP_expr_z__mean",
    "surround__Fibroblasts__aSMA_expr_z__mean",
    "surround__Fibroblasts__PDPN_expr_z__mean",
    "surround__Fibroblasts__Thy1_expr_z__mean",
]
FIBRO_LABEL = {
    "surround__Fibroblasts__FAP_expr_z__mean":  "FAP",
    "surround__Fibroblasts__aSMA_expr_z__mean": "αSMA",
    "surround__Fibroblasts__PDPN_expr_z__mean": "PDPN",
    "surround__Fibroblasts__Thy1_expr_z__mean": "Thy1",
}

# ── PanIN validation columns ──────────────────────────────────────────────────
PANIN_VALIDATION_SCORE_COLS = [
    "panin_validation__normal_duct_like_score",
    "panin_validation__lg_panin_like_score",
    "panin_validation__hg_panin_like_score",
    "panin_validation__invasive_desmoplastic_context_score",
]
PANIN_LABELS = {
    "panin_validation__normal_duct_like_score":              "Normal duct-like",
    "panin_validation__lg_panin_like_score":                 "LG PanIN-like",
    "panin_validation__hg_panin_like_score":                 "HG PanIN-like",
    "panin_validation__invasive_desmoplastic_context_score": "Invasive-desmoplastic",
}

# ── Spatial cell table path helper ────────────────────────────────────────────
SPATIAL_TABLE_VERSION = "auto_branch_n36_v5_contextual"

def spatial_pkl(sample_id: str) -> Path:
    return CACHE_DIR / f"spatial_cells_{SPATIAL_TABLE_VERSION}_{sample_id}.pkl"

# ── rcParams ──────────────────────────────────────────────────────────────────
def set_pub_rc():
    matplotlib.rcParams.update({
        "font.family":    "Arial",
        "font.size":       6,
        "axes.labelsize":  6,
        "axes.titlesize":  6,
        "xtick.labelsize": 5.5,
        "ytick.labelsize": 5.5,
        "axes.linewidth":  0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size":  2.5,
        "ytick.major.size":  2.5,
        "pdf.fonttype":    42,
        "svg.fonttype":    "none",
        "axes.spines.top":   False,
        "axes.spines.right": False,
    })

# ── Branch name auto-assignment (replicates notebook logic) ──────────────────
BRANCH_NAME_RULES = [
    ("early_duct_anchor",        "Normal/early-duct"),
    ("panin_like_dysplasia",     "PanIN-like"),
    ("invasive_gland_forming",   "Invasive gland-forming"),
    ("invasion_desmoplasia",     "Invasive-desmoplastic"),
    ("proliferation",            "Proliferative"),
    ("dedifferentiation",        "Dedifferentiated"),
]

def assign_branch_bio_names(df: pd.DataFrame,
                             module_cols: list,
                             branch_col: str = "major_branch") -> dict:
    """Return {branch_id: bio_name} using dominant module z-enrichment."""
    import re
    valid = df[branch_col].dropna().unique()
    valid = [b for b in valid if b not in {"unassigned"}]

    bm = df[df[branch_col].isin(valid)].groupby(branch_col, observed=True)[module_cols].mean()
    bz = bm.apply(lambda col: (col - col.mean()) / max(col.std(ddof=0), 1e-8), axis=0)

    names = {}
    for branch in bz.index:
        if str(branch) == "trunk":
            names[branch] = "Trunk"
            continue
        dom_col = bz.loc[branch].idxmax()
        bio = branch
        for key, label in BRANCH_NAME_RULES:
            if key in dom_col:
                bio = label
                break
        names[branch] = bio
    return names

# ── Branch colour palette (qualitative) ──────────────────────────────────────
BRANCH_COLORS_BASE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78",
]

def make_branch_palette(branches: list) -> dict:
    pal = {}
    ci = 0
    for b in branches:
        if b == "trunk":
            pal[b] = "#555555"
        elif b == "unassigned":
            pal[b] = "#cccccc"
        else:
            pal[b] = BRANCH_COLORS_BASE[ci % len(BRANCH_COLORS_BASE)]
            ci += 1
    return pal
