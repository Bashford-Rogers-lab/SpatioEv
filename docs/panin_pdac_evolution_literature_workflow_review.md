# PanIN to PDAC Evolution Literature Review and SpatioEv Workflow Sanity Check

Date: 2026-05-10

Input literature table: `/Users/shihongwu/Desktop/for_PanIN_evolution_research_proposal.csv`

Workflow reviewed: `/Users/shihongwu/SpatioEv/notebooks/07_xenium_03_pooled_pseudotime.ipynb`

Generated QC table: `/Users/shihongwu/SpatioEv/data/xenium_pancreas_10x/pseudotime/literature_marker_availability.csv`

## Scope

I reviewed all 31 entries in the PanIN / PDAC evolution paper list using the local title, abstract, DOI, PMID/PMCID, notes, and journal metadata. For the major workflow-defining papers, I also checked accessible article pages or DOI landing pages. This is therefore a strong literature-level sanity check, but not a replacement for pathologist ROI validation or manual full-text extraction of every figure and supplement.

## Main Literature Model

The papers converge on a more complex model than a single linear "normal duct -> low-grade PanIN -> high-grade PanIN -> invasive PDAC" path.

### 1. PanINs are common, multifocal, and often indolent

PanINs are frequent even in grossly normal pancreas, and most do not become clinically relevant PDAC. Three-dimensional genomic mapping estimates many PanINs per adult pancreas, with most arising as independent clones rather than a single spatially continuous precursor lineage. This means a pseudotime framework should not imply that every PanIN-like niche is destined to become invasive tumor.

Relevant papers:

- [3D genomic mapping reveals multifocality of human pancreatic precancers](https://doi.org/10.1038/s41586-024-07359-3)
- [Analysis of donor pancreata defines the transcriptomic signature and microenvironment of early neoplastic lesions](https://doi.org/10.1158/2159-8290.CD-23-0013)
- [Progression to pancreatic ductal adenocarcinoma from pancreatic intraepithelial neoplasia: Results of a simulation model](https://doi.org/10.1016/j.pan.2018.07.009)
- [The Prevalence and Clinicopathological Characteristics of High-Grade Pancreatic Intraepithelial Neoplasia](https://doi.org/10.1097/MPA.0000000000000786)

Implication for SpatioEv:

Our tree should be presented as a branched epithelial state landscape, not as a literal clock where all niches pass through the same stages.

### 2. Intraductal spread and cancerization of ducts are central

Several papers emphasize that neoplastic epithelial cells can move through the pancreatic ductal system before invasion, and that intraductal PDAC or cancerization of ducts is clinically meaningful. A 2D field may therefore contain spatially separated but biologically connected ductal lesions.

Relevant papers:

- [Precancerous neoplastic cells can move through the pancreatic ductal system](https://doi.org/10.1038/s41586-018-0481-8)
- [Clinical Relevance of Cancerization of Ducts in Resected Pancreatic Ductal Adenocarcinoma](https://doi.org/10.1097/MPA.0000000000002326)
- [Intraductal pancreatic cancer is less responsive than cancer in the stroma to neoadjuvant chemotherapy](https://doi.org/10.1038/s41379-020-0572-6)

Implication for SpatioEv:

Our current niche construction is local and morphology/spatial-context driven, but it does not explicitly model ductal continuity, duct-network connectedness, or cancerization along a duct. This is one of the clearest missing morphology-driven features.

### 3. The PanIN-to-invasive transition involves leaving the duct, not only accumulating dysplasia

The 2026 Cancer Cell paper frames malignancy as activation of a conserved re-epithelialization / wound-repair-like program during PanIN-to-PDAC transition, with FOSL1, integrin/hemidesmosome features, EGFR signaling, and CTHRC1-high myCAF coupling. This matches the idea that invasion is a spatial event at the epithelial-stromal interface.

Relevant papers:

- [A conserved re-epithelialization program underlies malignancy in pancreatic ductal adenocarcinoma](https://doi.org/10.1016/j.ccell.2026.03.021)
- [Morphology-guided transcriptomic analysis of human pancreatic cancer organoids reveals microenvironmental signals that enhance cancer cell fitness](https://doi.org/10.1172/jci162054)
- [Transfer learning reveals cancer-associated fibroblasts are associated with epithelial-mesenchymal transition and inflammatory signaling](https://doi.org/10.1158/0008-5472.CAN-23-1660)

Implication for SpatioEv:

We already include desmoplastic context, immune context, epithelial state, and gland-poor/undifferentiated proxies. We should add explicit literature-inspired module scores where the Xenium panel supports them:

- MP10 / re-epithelialization candidate score
- EGFR-ligand / epithelial wound-response score
- CTHRC1-high CAF / myCAF coupling score
- epithelial-stromal interface disruption score

Marker availability is limited. Across the 10x panels, many canonical MP10 genes are absent, but EGFR, AREG in `pdac_io_v1`, COL17A1 in three samples, LAMB3 in `pdac_addon_v1`, and CTHRC1 in `pdac_addon_v1` are available. These can support proxy scores, but we should not oversell them as a full MP10 score.

### 4. ECM architecture is a progression feature, not just a background context

The stromal architecture literature is especially important for a morphology-driven pseudotime method. Periductal collagen architecture, TACS-2/TACS-3 patterns, epithelial extrusion, and contact-guided invasion can appear early and can make histologically premalignant lesions biologically more invasive.

Relevant papers:

- [Stromal architecture directs early dissemination in pancreatic ductal adenocarcinoma](https://doi.org/10.1172/jci.insight.150330)
- [Epithelial and stromal co-evolution and complicity in pancreatic cancer](https://doi.org/10.1038/s41568-022-00530-w)
- [Tissue architecture in tumor initiation and progression](https://doi.org/10.1016/j.trecan.2022.02.007)

Implication for SpatioEv:

This is the biggest modality gap for Xenium. DAPI morphology and transcript proximity can approximate some effects, but DAPI cannot directly measure collagen orientation or TACS. If we use H&E or other morphology images, the highest-value new features would be:

- periductal collagen-like texture / eosinophilic stromal texture
- stromal fiber orientation relative to epithelial boundary
- epithelial boundary protrusion or extrusion-like irregularity
- distance-weighted fibroblast / CAF coupling around ducts
- local epithelial-stromal interface length and interface roughness

### 5. Duct and gland geometry matter

PanINs are defined histologically by mucinous ductal epithelium and dysplasia, but 3D reconstruction challenges simple 2D size definitions. Tissue curvature, tubular geometry, lumen shape, exophytic versus endophytic growth, and apicobasal mechanical imbalance can influence cancer morphogenesis.

Relevant papers:

- [Early neoplastic lesions of the pancreas: initiation, progression, and opportunities for precancer interception](https://doi.org/10.1172/JCI191937)
- [Tissue curvature and apicobasal mechanical tension imbalance instruct cancer morphogenesis](https://doi.org/10.1038/s41586-019-0891-2)
- [Tissue clearing and 3D reconstruction of digitized, serially sectioned slides provide novel insights into pancreatic cancer](https://doi.org/10.1016/j.medj.2022.11.009)
- [Power-law growth models explain incidences and sizes of pancreatic cancer precursor lesions](https://doi.org/10.1126/sciadv.ado5103)

Implication for SpatioEv:

Our current nuclear / epithelial-niche features include useful components such as area, hull geometry, circularity, orientation, epithelial identity, PanIN-like remodeling, and DAPI texture. The main missing duct/gland features are:

- lumen area and lumen fraction
- duct caliber / epithelial ring diameter
- lumen eccentricity and cystic dilation
- epithelial thickness around lumen
- tubular branching or duct-network connectedness
- exophytic versus endophytic growth proxy
- ductal component continuity across nearby epithelial niches

These features would make the morphology-driven part of the pseudotime much closer to the PanIN pathology literature.

### 6. ADM, pancreatitis, and acinar context matter

PanINs can arise from ductal cells or acinar-to-ductal metaplasia, with strong evidence for ADM-like origins in many contexts. Pancreatitis and fibroinflammatory states can precondition stroma and may resemble early tumor-permissive environments.

Relevant papers:

- [Early neoplastic lesions of the pancreas](https://doi.org/10.1172/JCI191937)
- [From precursor to cancer: decoding the intrinsic and extrinsic pathways of pancreatic intraepithelial neoplasia progression](https://doi.org/10.1093/carcin/bgae064)
- [Stromal architecture directs early dissemination in pancreatic ductal adenocarcinoma](https://doi.org/10.1172/jci.insight.150330)

Implication for SpatioEv:

Our current `histology__adm_panin_like_score` is a good start. It would be stronger if paired with:

- acinar-loss / acinar-neighborhood depletion
- acinar-to-ductal transition gene score, where genes are available
- lobule-preservation score for `pdac_addon_v1`
- pancreatitis-like fibroinflammatory context score

### 7. Immune architecture is heterogeneous

PanIN and PDAC immune responses are spatially heterogeneous, including immune-hot and immune-cold regions. Recent metastatic PDAC work also highlights plasma cell exclusion and chemokine-mediated immune compartmentalization.

Relevant papers:

- [3D histology reveals that immune response to pancreatic precancers is heterogeneous and depends on global pancreas structure](https://doi.org/10.1101/2024.08.03.606493)
- [Spatial mapping of transcriptomic plasticity in metastatic pancreatic cancer](https://doi.org/10.1038/s41586-025-08927-x)
- [Single-cell multi-stage spatial evolutional map of esophageal carcinogenesis](https://doi.org/10.1016/j.ccell.2025.02.009)

Implication for SpatioEv:

Our immune-inflamed and immune-exclusion scores are aligned with this literature. `pdac_io_v1` is particularly important because it has a strong duodenum-invasion / gland-poor / immune-exclusive biological prior from manual review. The current workflow correctly treats this as a branch-like state rather than forcing it into the same desmoplastic progression as the other PDAC samples.

## Cross-Check Against Current Xenium Pseudotime Workflow

### What the workflow already does well

1. It uses a branched tree rather than a single linear trajectory.

This matches the literature better than a forced normal-to-PanIN-to-PDAC line. Multifocal PanINs, ductal spread, independent clones, ADM, desmoplasia, immune-exclusion, and gland-poor tumor states can all form separate branches.

2. It separates original, sample-centered, and intrinsic epithelial sensitivity trajectories.

This is important because the four 10x datasets have strong sample/histology differences. A single pooled trajectory can accidentally become a sample axis. The sample-centered trajectory is a good sanity check, and the intrinsic epithelial trajectory is useful when we want to ask which ordering remains after reducing microenvironment dominance.

3. It adds histology proxy modules.

The current modules map well onto literature-supported axes:

- normal duct-like
- ADM/PanIN-like
- glandular architecture
- desmoplastic tumor
- immune-inflamed
- immune-exclusion
- duodenum-invasion context
- gland-poor / undifferentiated

4. It keeps spatial backprojection.

This is essential. The trajectory only becomes biologically credible when branches and pseudotime localize to plausible regions on tissue: residual normal pancreas, PanIN/ADM areas, desmoplastic tumor, duodenum-invasive tumor, immune-excluded regions, etc.

5. It now uses corrected `pdac_io_v1` mucosa/submucosa annotation.

This correction matters because duodenal mucosa/submucosa contamination can otherwise be mistaken for pancreatic ductal biology.

### Main concerns

1. Pseudotime can be mistaken for true chronological evolution.

The literature strongly argues against a single literal progression path. We should present our output as a morphology-informed epithelial state manifold with branch-level biological interpretation.

2. The normal sample can contain high-pseudotime or PanIN-like states.

That is not automatically wrong. Healthy or grossly normal pancreas can contain PanINs and KRAS-mutant precursors. It becomes concerning only if high pseudotime is diffusely spread across normal tissue rather than localized to plausible ductal / ADM-like / inflamed regions.

3. Xenium DAPI-only morphology does not capture collagen architecture.

This is a key limitation because TACS / ECM architecture is one of the clearest progression-related morphology signals. H&E-derived stromal texture or collagen-oriented imaging would strengthen this substantially.

4. The gene panel limits subtype scoring.

The literature-marker availability check shows that many canonical markers are absent:

- MP10/re-epithelialization markers are sparse. EGFR and COL17A1 are broadly available; AREG is available in `pdac_io_v1`; LAMB3 and CTHRC1 appear in `pdac_addon_v1`.
- Basal/classical genes are very sparse. EPCAM is broadly present, but many common basal/classical markers are absent from these panels.
- Immune-exclusion markers are best represented, especially in `pdac_io_v1`, where CXCL12, CXCR4, TGFB1, IL6, immune lineage genes, MZB1, and JCHAIN are all present.

Therefore, marker modules should be labeled as proxies unless the relevant genes are available in a sample.

## Missing or Underdeveloped Feature Classes

### Highest priority

1. Duct/lumen architecture

Add features derived from epithelial masks and/or H&E:

- lumen fraction within each epithelial niche
- number of lumens per niche
- lumen size distribution
- epithelial ring thickness
- duct caliber
- cystic dilation proxy
- gland/tubule completeness
- gland fragmentation
- epithelial connected-component branching

Why it matters:

PanIN and gland-forming PDAC are fundamentally duct/gland/lumen phenotypes.

2. Ductal continuity and cancerization-of-ducts proxies

Add graph features that ask whether epithelial niches form elongated connected chains through tissue:

- epithelial component length
- skeleton branch count
- duct-network tortuosity
- pseudotime gradient along connected epithelial components
- local continuity of PanIN-like state along ductal structures
- distance to nearest high-grade/tumor-like epithelial component

Why it matters:

The literature emphasizes intraductal spread and cancerization. Our current local niche features do not fully capture this.

3. Epithelial-stromal interface and extrusion-like features

Add:

- epithelial boundary roughness
- boundary protrusion / budding index
- isolated epithelial cells near duct boundary
- tumor cell single-cell extrusion proxy
- epithelial-fibroblast interface length
- CAF density immediately outside epithelial boundary

Why it matters:

Invasion is not only a change in cell state. It is a spatial escape from duct/gland structure.

4. ECM / collagen architecture proxy

If H&E is available:

- stromal texture orientation near epithelial boundary
- eosin-rich stromal density proxy
- fiber-like anisotropy proxy
- radial versus tangential stromal texture relative to duct boundary

If only DAPI / Xenium transcriptomics is used:

- use fibroblast density, fibroblast activation markers, and interface geometry as imperfect proxies

Why it matters:

TACS is one of the strongest morphology-linked early dissemination signals in the literature.

### Medium priority

5. ADM and acinar-loss context

Add:

- acinar-neighborhood depletion
- distance from epithelial niche to acinar-rich lobules
- ADM-like gene score if genes are available
- lobule preservation / interlobular fibrosis proxy

6. Immune-hotspot and immune-exclusion spatial heterogeneity

Add:

- local immune hotspot score
- distance from epithelial niche to immune aggregates
- B/plasma-cell exclusion around tumor epithelium
- T-cell exclusion around gland-poor branches

7. Duodenum invasion and non-pancreatic epithelial contamination checks

Keep the corrected `Mucosa gland` and `Submucosa` labels in `pdac_io_v1`, and continue excluding or explicitly modeling these contexts.

## Recommended Workflow Refinements Before Finalizing

### Notebook 01: annotation

Keep graphclust as the primary annotation unit and Leiden as QC/fallback. Add a final per-sample marker dotplot and cell group export as already implemented. For `pdac_io_v1`, preserve the graphclust 2 and 17 overrides as mucosa gland and submucosa.

### Notebook 02: epithelial niche features

Recommended additions:

1. Add epithelial connected-component / duct-network summaries.

2. Add lumen-aware features if epithelial masks or polygon geometry can define holes/background inside epithelial components.

3. Add H&E-derived or morphology-image-derived stromal texture features around epithelial boundaries if feasible.

4. Add local epithelial-stromal interface features:

- boundary-to-fibroblast distance
- boundary-to-immune distance
- epithelial-fibroblast contact fraction
- boundary roughness

### Notebook 03: pooled pseudotime

Recommended additions:

1. Add literature marker module scores with availability flags:

- MP10/re-epithelialization proxy
- CTHRC1-high CAF / wound CAF proxy
- ADM/PanIN mucin proxy
- basal-like / classical proxy only when marker coverage is sufficient
- immune exclusion / CXCL12-CXCR4 axis proxy

2. Keep both microenvironment-aware and intrinsic epithelial trajectories.

3. Make the final narrative branch-based:

- trunk: normal / residual normal duct-like
- early disease / ADM-PanIN-lobule-preserved branch
- desmoplastic glandular tumor branch
- immune-inflamed branch
- gland-poor / duodenum-invasive / immune-excluded branch
- mixed/unclear branches that need manual ROI validation

4. Add leave-one-sample-out or sample-weighted sensitivity:

- build trajectory without each PDAC sample
- project held-out niches onto the tree
- check whether key branch labels remain stable

5. Add figure-level warnings:

- "pseudotime is a state coordinate, not elapsed time"
- "branch labels are automatic and literature-guided, not pathologist-confirmed"
- "Xenium DAPI morphology cannot directly measure collagen TACS"

## Evaluation of Current Workflow

Overall, the current workflow is scientifically coherent and well aligned with the modern PanIN/PDAC evolution literature if we frame it correctly.

The strongest claim we can make:

SpatioEv builds a spatially grounded epithelial state landscape that organizes pancreatic epithelial niches by morphology, gene expression, and microenvironmental context, revealing branched normal-like, ADM/PanIN-like, desmoplastic glandular, immune-inflamed, and gland-poor/immune-excluded tumor states.

The claim to avoid:

This reconstructs a universal chronological normal-to-PanIN-to-PDAC progression path.

The biggest missing morphology features are duct/lumen topology, ductal continuity, epithelial-stromal interface disruption, and collagen/ECM architecture. Adding even partial versions of these features would make the pseudotime analysis much more defensible as a morphology-driven PanIN-to-PDAC progression framework.

## Practical Next Step

Before finalizing the workflow, I would implement one more feature-review pass in this order:

1. Add literature proxy modules and marker availability flags to notebook 03.

2. Add duct/lumen and epithelial connected-component features to notebook 02 if masks/polygons support them. Implemented in the notebook generator on 2026-05-10 as feature version `boundary_components_v3_lumen_continuity_interface`.

3. Add a branch validation panel that shows, for each branch, sample composition, spatial location, top morphology features, top transcript modules, and representative tissue regions.

4. Keep the final paper/seminar narrative branch-based rather than linear.

## Appendix: Per-Paper Reading Map

This is a compact map of how each paper in the CSV informs the SpatioEv workflow.

| Year | DOI | Main idea for our workflow |
| --- | --- | --- |
| 2014 | [10.1158/0008-5472.CAN-14-0734](https://doi.org/10.1158/0008-5472.CAN-14-0734) | Early detection requires identifying curable precursor and earliest invasive states, so the trajectory should emphasize clinically interpretable early-state features rather than only late tumor separation. |
| 2015 | [10.1136/gutjnl-2014-308653](https://doi.org/10.1136/gutjnl-2014-308653) | Long latent phases can be followed by rapid clinically detectable progression, supporting the need to distinguish indolent PanIN-like states from invasive/high-risk states. |
| 2017 | [10.1097/MPA.0000000000000786](https://doi.org/10.1097/MPA.0000000000000786) | High-grade PanIN is multifocal and associated with fibrosis/cystic changes, supporting duct/lumen and stromal remodeling features. |
| 2018 | [10.1038/s41586-018-0481-8](https://doi.org/10.1038/s41586-018-0481-8) | Precancerous cells can spread through ducts, so duct-network continuity and cancerization-of-ducts proxies are important missing features. |
| 2018 | [10.1016/j.pan.2018.07.009](https://doi.org/10.1016/j.pan.2018.07.009) | Simulation suggests many PanINs do not progress, reinforcing that pseudotime should be state-risk ordering, not deterministic destiny. |
| 2019 | [10.1038/s41586-019-0891-2](https://doi.org/10.1038/s41586-019-0891-2) | Tissue curvature and apicobasal mechanics influence growth modes, motivating lumen shape, duct curvature, epithelial thickness, and exophytic/endophytic proxies. |
| 2020 | [10.1158/2159-8290.CD-20-0133](https://doi.org/10.1158/2159-8290.CD-20-0133) | Intraductal models show molecular subtype transition, motivating classical/basal or glandular/gland-poor sensitivity scores if marker coverage allows. |
| 2020 | [10.1038/s41379-020-0572-6](https://doi.org/10.1038/s41379-020-0572-6) | Intraductal PDAC can differ from stromal invasive cancer, supporting separate intraductal/cancerization and stromal-invasion axes. |
| 2020 | [10.1038/s41379-019-0409-3](https://doi.org/10.1038/s41379-019-0409-3) | 3D visualization reveals invasion patterns not captured well by 2D morphology, so our 2D workflow should be explicit about that limitation. |
| 2021 | [10.1007/s10555-020-09953-z](https://doi.org/10.1007/s10555-020-09953-z) | Pancreatic pathology can be interpreted through evolution, supporting branch-level rather than single-axis interpretation. |
| 2021 | [10.1172/jci.insight.150330](https://doi.org/10.1172/jci.insight.150330) | Periductal collagen architecture directs early dissemination, making ECM/TACS-like features a high-priority missing morphology class. |
| 2022 | [10.1053/j.gastro.2022.03.056](https://doi.org/10.1053/j.gastro.2022.03.056) | Broad PDAC pathogenesis review supports integrating precursor biology, diagnosis, and treatment relevance in the workflow narrative. |
| 2022 | [10.1038/s41568-021-00418-1](https://doi.org/10.1038/s41568-021-00418-1) | Evolution and heterogeneity review supports separating sample effects, subtype diversity, and microenvironmental states. |
| 2022 | [10.1016/j.trecan.2022.02.007](https://doi.org/10.1016/j.trecan.2022.02.007) | Tissue architecture shapes routes of invasion, supporting topology and spatial-context features beyond expression. |
| 2023 | [10.1158/2159-8290.CD-23-0013](https://doi.org/10.1158/2159-8290.CD-23-0013) | Donor pancreas PanINs have tumor-like epithelial signatures and altered microenvironment, validating why normal tissue can contain PanIN-like branches. |
| 2023 | [10.1038/s41568-022-00530-w](https://doi.org/10.1038/s41568-022-00530-w) | Epithelial-stromal co-evolution supports including CAF/fibroblast, immune, and interface context in pseudotime. |
| 2023 | [10.1172/JCI162054](https://doi.org/10.1172/JCI162054) | Morphology-guided organoids show that morphology and microenvironmental signals track tumor fitness, supporting our morphology-plus-context design. |
| 2023 | [10.1016/j.medj.2022.11.009](https://doi.org/10.1016/j.medj.2022.11.009) | Tissue clearing and 3D reconstruction highlight the importance of whole-structure context, not isolated fields. |
| 2024 | [10.1038/s41586-024-07359-3](https://doi.org/10.1038/s41586-024-07359-3) | Human PanINs are numerous, multifocal, and often independent, supporting a branched landscape and cautioning against a single lineage story. |
| 2024 | [10.1101/2024.08.03.606493](https://doi.org/10.1101/2024.08.03.606493) | Immune response to PanIN is heterogeneous and depends on global pancreas structure, supporting immune-hot/cold and sample-spatial backprojection checks. |
| 2024 | [10.1097/MPA.0000000000002326](https://doi.org/10.1097/MPA.0000000000002326) | Cancerization of ducts is clinically relevant, strengthening the need for duct-continuity features. |
| 2024 | [10.1093/carcin/bgae064](https://doi.org/10.1093/carcin/bgae064) | Intrinsic and extrinsic PanIN progression pathways justify both epithelial-intrinsic and microenvironment-aware trajectories. |
| 2024 | [10.1016/j.cels.2024.07.001](https://doi.org/10.1016/j.cels.2024.07.001) | PanIN and CAF transitions from spatial integration support adding CAF-transition and inflammatory-proliferative modules. |
| 2024 | [10.1126/sciadv.ado5103](https://doi.org/10.1126/sciadv.ado5103) | PanIN size/growth and spatial genomic validation motivate quantifying lesion size, expansion, and connected component architecture. |
| 2024 | [10.1158/0008-5472.CAN-23-1660](https://doi.org/10.1158/0008-5472.CAN-23-1660) | CAFs are linked to epithelial EMT and inflammation, supporting epithelial-fibroblast coupling features. |
| 2025 | [10.1126/scitranslmed.adq3110](https://doi.org/10.1126/scitranslmed.adq3110) | Protease activation as early detection biology suggests invasion/remodeling activity may be important but is poorly captured by current Xenium panels. |
| 2025 | [10.1172/JCI191937](https://doi.org/10.1172/JCI191937) | The JCI review synthesizes PanIN/IPMN morphology, ADM origin, and interception opportunities, providing the main framing for workflow validation. |
| 2025 | [10.1016/j.ccell.2025.02.009](https://doi.org/10.1016/j.ccell.2025.02.009) | Although esophageal, the multi-stage spatial map is a useful analogy for epithelial-stromal interface disruption and immune-shielding niches. |
| 2025 | [10.1038/s41586-025-08927-x](https://doi.org/10.1038/s41586-025-08927-x) | Spatial transcriptomic plasticity in PDAC highlights basal-like/CAF colocalization, plasma-cell exclusion, and CXCR4/CXCL12 immune organization. |
| 2025 | [10.1158/2159-8290.CD-23-1541](https://doi.org/10.1158/2159-8290.CD-23-1541) | Evolutionary forest concept fits our branch-based interpretation and warns against over-linearizing PDAC evolution. |
| 2026 | [10.1016/j.ccell.2026.03.021](https://doi.org/10.1016/j.ccell.2026.03.021) | Conserved re-epithelialization/MP10 program, FOSL1, EGFR, and CTHRC1-high CAF coupling provide the clearest candidate module for PanIN-to-invasive transition. |
