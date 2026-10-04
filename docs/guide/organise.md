# 3. Organise your data

SpatioEv finds everything by **folder and file names**, so it never asks you
to pick files one by one. Getting the layout right once saves a lot of time
later.

This guide starts from cores that have already been **segmented** (cell
outlines found) and **quantified** (marker brightness measured per cell).
If you only have the raw images, ask whoever runs the preprocessing for your
lab; they will give you folders in the layout below.

## The layout

Put all core folders side by side in **one folder**. This folder is your
**project root**:

```text
TMA_072/                                   ← project root
├── OXF047_6-C/                            ← one folder per core, named after the core
│   ├── background/
│   │   └── OXF047_6-C.ome.tiff            ← the multiplexed image (all channels)
│   ├── segmentation/
│   │   ├── OXF047_6-C_whole_cell.tiff     ← cell outlines (one number per cell)
│   │   └── OXF047_6-C_nuclear.tiff        ← nucleus outlines
│   └── quantification/
│       ├── cell_table_arcsinh_transformed.csv   ← marker levels per cell
│       ├── cell_table_size_normalized.csv
│       └── cell_table_raw.csv
├── OXF047_6-E/
│   └── … same four sub-folders …
└── OXF048_6-G/
    └── …
```

The rules:

1. **One folder per core**, named after the core.
2. Inside it, the files **start with the same name** as the folder:
   `OXF047_6-C/background/OXF047_6-C.ome.tiff`,
   `OXF047_6-C/segmentation/OXF047_6-C_whole_cell.tiff`, and so on.
3. The sub-folders are called exactly `background`, `segmentation` and
   `quantification` (lower case).
4. Every core folder sits directly inside the project root, not in deeper
   sub-folders.

Other files and folders in a core folder (for example `segmentation_input`
or `batch_state` from the preprocessing) are ignored, so leave them where they
are.

!!! warning "Do not rename anything inside a core folder"
    The core name links the image, the masks, the cell table, the QuPath
    project and every result. If a core really must be renamed, rename the
    folder **and** every file that starts with the old name, before you start
    the analysis.

!!! tip "Several TMAs in one cohort"
    If the core names are unique across TMAs (for example they include the
    TMA or block number, like `OXF047_6-C`), put all cores of the cohort in one
    project root. Comparing groups is then a single step. If names repeat
    between TMAs, keep one project root per TMA.

## Where to keep it

- Use a folder in your home folder, for example `/Users/you/Data/TMA_072`, or
  an external drive.
- Avoid **Desktop** and **Documents** if they sync to iCloud Drive: macOS may
  move files to the cloud to save space, and the analysis then stalls while
  they download.
- Avoid **spaces** in folder names. They work, but every Terminal command then
  needs quotes around the path. Use `_` or `-` instead.
- Plan for about **0.3 GB per core** for the data, plus up to **0.1 GB per
  core** for results and the QuPath project. 200 cores need roughly 80 GB.

## What SpatioEv adds as you go

You do not create these; SpatioEv and QuPath do:

```text
TMA_072/
├── OXF047_6-C/
│   ├── …
│   └── qupath/                            ← step 5: cells for QuPath, classes back from QuPath
├── results/                               ← steps 6–9
│   ├── OXF047_6-C_fiber_segmentation/
│   ├── OXF047_6-C_tissue_regions/
│   ├── OXF047_6-C_colocalization/
│   └── cohort_comparison/
├── qupath_project/                        ← your QuPath project (step 5)
└── sample_sheet.csv                       ← step 9: which group each core belongs to
```

## Check the layout

SpatioEv can list every core it recognises. In Terminal (with
`conda activate spatioev_env` done), type `spatioev qupath status ` with a
space at the end, then **drag your project-root folder from Finder onto the
Terminal window** to paste its path, and press ++return++:

```bash
spatioev qupath status /Users/you/Data/TMA_072
```

You get one line per core, with `True` for every file it found:

```text
      core  image  cell_mask  nuclear_mask  table  cells_geojson  classes  annotations  phenotyped
OXF047_6-C   True       True          True   True          False    False        False       False
OXF047_6-E   True       True          True   True          False    False        False       False
```

- The first four columns (`image`, `cell_mask`, `nuclear_mask`, `table`)
  should all be `True`. A `False` means that file is missing or named
  differently: compare it with the layout above.
- The last four columns fill in during [step 5](qupath.md).
- `No core folders found` means the path is not the project root, or the
  core folders lack `segmentation` and `quantification` sub-folders.

Next: [Open the app](app.md).
