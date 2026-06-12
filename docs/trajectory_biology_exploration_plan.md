# Trajectory Biology Exploration Plan

Date: 2026-05-10

Scope:

- Multiplexed imaging trajectory notebooks:
  - `/Users/shihongwu/SpatioEv/notebooks/06_dev_graph_pseudotime_v2_combined_exp_2_3_4_5.ipynb`
  - `/Users/shihongwu/SpatioEv/notebooks/06_dev_graph_pseudotime_v2_exp_2.ipynb`
  - `/Users/shihongwu/SpatioEv/notebooks/06_dev_graph_pseudotime_v2_exp_3.ipynb`
  - `/Users/shihongwu/SpatioEv/notebooks/06_dev_graph_pseudotime_v2_exp_4.ipynb`
  - `/Users/shihongwu/SpatioEv/notebooks/06_dev_graph_pseudotime_v2_exp_5.ipynb`
- Xenium trajectory notebooks:
  - `/Users/shihongwu/SpatioEv/notebooks/07_xenium_02_epithelial_niche_features.ipynb`
  - `/Users/shihongwu/SpatioEv/notebooks/07_xenium_03_pooled_pseudotime.ipynb`
- Literature synthesis:
  - `/Users/shihongwu/SpatioEv/docs/panin_pdac_evolution_literature_workflow_review.md`

## Central Framing

The strongest interpretation is not "we reconstructed chronological time." It is:

SpatioEv learns a spatial epithelial state landscape where ductal niches separate into normal-like, ADM/PanIN-like, glandular, intraductal/cancerization-like, interface-disrupted/desmoplastic, immune-inflamed, immune-excluded, and gland-poor/undifferentiated states.

The biological question is then:

Which spatial and molecular changes define the paths from normal ductal organization toward PanIN-like remodeling and invasive tumor states, and are these paths shared or modality/sample-specific?

## What The Trajectory Can Tell Us

### 1. Whether tumor development is linear or branched

Literature expectation:

PanIN and PDAC evolution is multifocal and branched. Many PanIN-like lesions are indolent. Different invasive programs can arise from related but not identical precursor states.

Analysis:

- Compare branch occupancy by sample/disease.
- Separate trunk, early disease branches, glandular tumor branches, desmoplastic branches, immune-inflamed branches, and immune-excluded/gland-poor branches.
- Avoid treating pseudotime as one universal line.

Visualization:

- UMAP/tree roadmap with branch labels.
- Branch composition stacked bars by sample and disease.
- Branch heatmap of biological scores.
- A simplified "state map" cartoon: normal-like trunk, PanIN/glandular branch, desmoplastic-interface branch, immune-excluded/gland-poor branch.

### 2. Which features change earliest

Literature expectation:

Early changes can include ADM/PanIN-like mucin programs, ductal remodeling, stromal activation, immune changes, and altered epithelial architecture before frank invasive morphology.

Analysis:

- Along pseudotime, plot normal duct-like score, ADM/PanIN-like score, duct/lumen topology, epithelial identity, proliferation, desmoplasia, immune scores.
- Compare contextual trajectory versus intrinsic epithelial trajectory.
- Identify features whose change starts before late desmoplasia or gland-poor states.

Visualization:

- LOWESS trend panels along pseudotime.
- "Earliest changing features" ranked plot using correlation or GAM/LOWESS effect size.
- Heatmap of binned pseudotime feature z-scores.

### 3. Whether PanIN-like states are terminal, transitional, or side branches

Literature expectation:

PanINs are common, multifocal, and often do not progress. PanIN-like branches should not always be interpreted as mandatory intermediates.

Analysis:

- For multiplexed imaging: use `panin_validation__normal_duct_like_score`, `panin_validation__lg_panin_like_score`, `panin_validation__hg_panin_like_score`, `panin_validation__panin_grade_like_axis`, and contextual PDAC modules.
- For Xenium: use `histology__adm_panin_like_score`, `xenium_panin_like_remodeling_score`, `histology__ductal_continuity_cancerization_score`, and `histology__glandular_architecture_score`.
- Ask whether PanIN-like states sit near branchpoints, terminal branches, or the normal-like trunk.

Visualization:

- PanIN-like score overlaid on UMAP/tree.
- Pseudotime density of PanIN-like high niches.
- Spatial maps of PanIN-like high niches in residual normal pancreas and early PDAC.

### 4. Whether ductal continuity/cancerization is separable from invasive/desmoplastic remodeling

Literature expectation:

Precancerous cells can spread through ducts, and cancerization of ducts can be clinically meaningful. Intraductal spread is not the same as stromal invasion.

Analysis:

- In Xenium, use `histology__ductal_continuity_cancerization_score`, `xenium_duct_continuity_cancerization_score`, and raw `duct_continuity__*` features.
- In multiplexed imaging, use skeleton length, branchpoints, duct organization, architectural complexity, and epithelial-intrinsic PanIN validation scores as analogues.
- Compare duct-continuity-high niches with desmoplastic/interface-high niches.

Visualization:

- 2D plot: ductal continuity score versus interface/disruption or desmoplasia score.
- Spatial maps of duct-continuity-high niches.
- Representative niche panels showing continuous duct-like versus invasive/interface-disrupted morphology.

### 5. How invasion emerges at the epithelial-stromal interface

Literature expectation:

The transition to invasive PDAC is spatial: epithelial boundaries become disrupted, CAFs/fibroblasts engage, and tumor cells interface with or escape into stroma.

Analysis:

- In Xenium, use `histology__epithelial_stromal_interface_disruption_score`, `xenium_epithelial_stromal_interface_disruption_score`, `interface__epithelial_boundary_roughness`, `interface__fibroblast_contact_fraction_hop1`, and `interface__nearby_unassigned_epithelial_cells_per_100um_boundary`.
- In multiplexed imaging, use boundary-minus-core pixel features, invasion/desmoplasia axis, fibroblast surround markers, and CK19/NaKATPase polarity/pixel features.
- Test whether interface disruption increases before, with, or after desmoplastic context.

Visualization:

- Pseudotime trend: interface disruption, fibroblast contact, desmoplasia.
- Spatial maps colored by interface-disruption score.
- Branch heatmap highlighting interface/disruption branch enrichment.
- Near/far fibroblast comparison for epithelial marker states.

### 6. Whether immune-inflamed and immune-excluded paths are distinct

Literature expectation:

PanIN and PDAC immune responses are heterogeneous, with immune-hot and immune-cold/excluded regions.

Analysis:

- Xenium: compare `histology__immune_inflamed_score`, `histology__immune_exclusion_score`, T/B/myeloid proportions, checkpoint/plasma markers.
- Multiplexed imaging: compare T cells, B lineage, fibroblasts, endothelial cells, and marker expression along trajectory.
- Ask whether immune-inflamed niches are early inflammatory contexts, tumor-reactive contexts, or separate side branches.

Visualization:

- Combined immune composition trends along pseudotime.
- Branch-level immune heatmap.
- Spatial maps of immune-inflamed versus immune-excluded branches.
- Cross-Ripley/local-near-vs-far analyses around branch-specific epithelial niches.

### 7. Whether glandular and gland-poor tumor states are divergent outcomes

Literature expectation:

PDAC has differentiated/gland-forming and basal-like/gland-poor/undifferentiated states. These can represent divergent malignant programs rather than a single severity scale.

Analysis:

- Xenium: compare glandular architecture, duct/lumen topology, epithelial identity, proliferation, immune exclusion, duodenum context.
- Multiplexed imaging: compare invasive gland-forming score, dedifferentiation axis, duct organization, CK19/NaKATPase polarity, and pixel texture.
- Focus on `pdac_io_v1` as gland-poor/immune-exclusive and `pdac_addon_v1` or parts of `pdac_pancreas_v1` as glandular/PanIN-preserved.

Visualization:

- Bifurcation figure showing glandular/lumen-rich versus gland-poor/interface-disrupted branches.
- Sample-separated UMAPs colored by branch and by pseudotime.
- Representative spatial panels from each branch.

### 8. What is conserved between multiplexed imaging and Xenium

Expected value:

If similar biological axes appear in both modalities, SpatioEv looks robust rather than dataset-specific.

Analysis:

Create a shared axis table:

| Biological axis | Multiplexed imaging readout | Xenium readout |
| --- | --- | --- |
| Normal duct-like | `pdac_early_duct_anchor_score`, duct organization, CK19/NaKATPase polarity | `histology__normal_duct_like_score`, epithelial identity, CFTR/FXYD2/TM4SF4 |
| ADM/PanIN-like | `pdac_panin_like_dysplasia_score`, PanIN validation scores | `histology__adm_panin_like_score`, MUC/TFF/CEACAM/AGR/SOX9 |
| Glandular architecture | invasive gland-forming, duct organization, circularity/topology | `histology__glandular_architecture_score`, duct/lumen topology |
| Ductal continuity/cancerization | skeleton length/branching, duct organization | `histology__ductal_continuity_cancerization_score` |
| Interface disruption/invasion | invasion-desmoplasia axis, boundary-minus-core features | `histology__epithelial_stromal_interface_disruption_score` |
| Desmoplasia | fibroblast surround, FAP/aSMA/PDPN/Thy1 | desmoplastic context, fibroblast contact/ACTA2/PDGFRA/THY1/PDPN |
| Immune context | T/B lineage proportions and markers | T/B/myeloid proportions and immune/checkpoint scores |
| Gland-poor/dedifferentiated | dedifferentiation axis, polarity loss, texture | gland-poor score, low glandular architecture, immune exclusion |

Visualization:

- Cross-modality alignment heatmap: rows are shared axes, columns are modality-specific readouts.
- Two-panel roadmap: multiplexed imaging tree and Xenium tree using the same biological color labels.
- "Concordance matrix" of which axes are supported by each modality.

## Concrete Analysis Modules To Add Next

### Module A: Unified trajectory atlas figures

Goal:

Make one clean figure set that explains the whole result.

Outputs:

- Multiplexed UMAP/tree colored by major biological branch.
- Xenium UMAP/tree colored by matched biological branch.
- Spatial maps for representative samples.
- Branch composition bars.
- Branch biology heatmap.

### Module B: Along-pseudotime trend atlas

Goal:

Show what changes along the trajectory.

Features:

- Normal duct-like
- ADM/PanIN-like
- glandular architecture
- ductal continuity/cancerization
- interface disruption
- desmoplasia/fibroblast activation
- immune inflammation
- immune exclusion
- proliferation
- dedifferentiation/gland-poor

Outputs:

- LOWESS line plots with optional binned confidence intervals.
- Separate panels for contextual trajectory and epithelial-intrinsic trajectory.
- Separate sample/disease overlays to avoid overinterpreting pooled effects.

### Module C: Branch contrast analysis

Goal:

For each branch, identify what makes it biologically different.

Outputs:

- Branch versus trunk effect-size table.
- Top enriched morphology, transcript, and microenvironment features.
- Branch labels generated from evidence, not only tree topology.
- Representative niche examples for each major branch.

### Module D: Spatial validation panels

Goal:

Show that inferred states land in plausible tissue regions.

Outputs:

- Spatial scatter by pseudotime.
- Spatial scatter by branch.
- Spatial scatter by top biological score.
- Zoomed representative regions for normal duct, PanIN-like, glandular tumor, desmoplastic/interface-disrupted tumor, immune-inflamed, immune-excluded/gland-poor.

### Module E: Microenvironment coupling

Goal:

Ask how epithelial progression couples to fibroblasts, T cells, B lineage, myeloid cells, endothelial cells.

Analyses:

- Surrounding cell proportions along pseudotime.
- Compartment marker expression along pseudotime.
- Near-versus-far local comparisons using cross-Ripley/local contact functions.
- Cross-Moran or spatial coupling between epithelial scores and surrounding features.

### Module F: Cross-modality validation

Goal:

Use multiplexed imaging and Xenium as complementary evidence.

Analyses:

- Match biological axes rather than branch IDs.
- Compare trend directions across modalities.
- Highlight where modalities agree:
  - PanIN-like remodeling
  - desmoplasia/interface disruption
  - immune exclusion/inflammation
  - glandular versus gland-poor divergence
- Highlight where each modality is uniquely informative:
  - multiplexed imaging: polarity, protein, CK19/NaKATPase, pixel texture
  - Xenium: duct/lumen proxy, gene programs, immune/fibroblast transcript states

## Suggested Biological Storyline

1. Normal-like ducts occupy a trunk/reference state.

2. ADM/PanIN-like remodeling emerges as a branch-like state, not a guaranteed linear precursor.

3. A duct-continuity/cancerization-like program may represent intraductal spread or multifocal ductal remodeling.

4. A glandular/lumen-rich branch captures differentiated tumor or PanIN/gland-forming architecture.

5. An interface-disrupted/desmoplastic branch captures epithelial-stromal invasion pressure.

6. Immune-inflamed and immune-excluded branches represent different microenvironmental outcomes.

7. Gland-poor/immune-excluded states, especially in `pdac_io_v1`, look like a divergent tumor state rather than simply "late PanIN."

## Highest-Value Figures

1. Cross-modality state roadmap

One schematic showing shared biological axes, with multiplexed imaging and Xenium evidence side by side.

2. Branch biology heatmap

Rows: branches. Columns: normal duct, ADM/PanIN, glandular, duct continuity, interface disruption, desmoplasia, immune-inflamed, immune-excluded, proliferation, gland-poor.

3. Pseudotime trend atlas

LOWESS panels for the key biological axes, shown separately for contextual and epithelial-intrinsic pseudotime.

4. Spatial validation montage

Representative normal, PanIN-like, glandular, desmoplastic/interface-disrupted, immune-inflamed, and gland-poor/immune-excluded regions.

5. Modality complementarity figure

Same biological question, two measurements:

- multiplexed imaging: protein/polarity/pixel morphology
- Xenium: gene programs/DAPI/cell-boundary topology/spatial context

## Immediate Next Step

I would create two notebook sections next:

1. A "Trajectory Biology Atlas" section in the Xenium notebook and the combined multiplexed notebook with matched plots:

- branch heatmap
- pseudotime trends
- sample-separated spatial maps
- representative niches

2. A new comparison notebook:

`notebooks/08_cross_modality_pseudotime_biology_atlas.ipynb`

This notebook should load:

- `/Users/shihongwu/SpatioEv/data/combined_exp_2_3_4_5/pooled_niche_result_df.pkl`
- `/Users/shihongwu/SpatioEv/data/combined_exp_2_3_4_5/epithelial_intrinsic_pseudotime_result_df.pkl`
- `/Users/shihongwu/SpatioEv/data/xenium_pancreas_10x/pseudotime/xenium_pseudotime_result_df.pkl`
- `/Users/shihongwu/SpatioEv/data/xenium_pancreas_10x/pseudotime/xenium_branch_biology_summary.csv`

Then it should make harmonized figures using shared biological axes rather than trying to force one-to-one branch matching.

