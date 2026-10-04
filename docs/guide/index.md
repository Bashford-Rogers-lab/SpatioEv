# Step-by-step guide

This guide takes you from a folder of preprocessed images to cell types,
matrix measurements, tissue regions, co-localisation and group comparisons.
It is written for people who have **never used Terminal or Python**. Every
step says what to do, why you are doing it, what you should see, and what to
do if you see something else.

It uses a Mac. You will copy commands into Terminal, click through the
SpatioEv app in your web browser, and classify cells in QuPath.

!!! tip "Try it on practice data first"
    [Step 2](demo.md) creates a small synthetic dataset that looks like real
    TMA cores. Going through the whole guide on it takes an afternoon at
    most, and means you meet every screen once before using your own data.

## What you will get

For every core (or slide):

- **Cell types**, which you assign in QuPath with an object classifier.
- **Matrix fibers** in each matrix channel (for example COL1, COL4, FN): how
  much matrix there is, how aligned it is, and TWOMBLI-style architecture
  (branching, curvature, gaps, fibre length).
- **Tissue regions**: every cell labelled *tumour*, *tumour envelope* or
  *stroma*, with cell composition and matrix inside each region.
- **Co-localisation**: which cell types sit together more than chance, and
  which cell types sit in dense or aligned matrix.

Across your cohort:

- **Group comparisons** (for example HPV+ vs HPV−) of all of the above, with
  one value per patient and p-values corrected for multiple testing.

## The big picture

```text
 Your preprocessed cores           (image + cell outlines + cell measurements,
        │                           made before this guide starts)
        │
        ├──► 5. Cell types in QuPath      SpatioEv hands the cells to QuPath,
        │                                 you classify them, SpatioEv takes the
        │                                 classes back
        │
        ├──► 6. Matrix fibers             runs on the image only, so it can run
        │                                 before or while you work in QuPath
        │
        └──► 7. Tissue regions            needs 5 (and uses 6)
                   │
                   └──► 8. Co-localisation      needs 7 (and uses 6)
                              │
                              └──► 9. Compare groups   needs 7 and 8 for
                                                        every core
```

| Step | Where you do it | What it needs | What it gives you |
|---|---|---|---|
| [5. Cell types](qupath.md) | SpatioEv app **and** QuPath | cell outlines and measurements | a cell type for every cell |
| [6. Matrix fibers](fibers.md) | SpatioEv app | the image | fibers, matrix area and architecture per channel |
| [7. Tissue regions](regions.md) | SpatioEv app | cell types (+ fibers) | tumour / envelope / stroma, composition, matrix per region |
| [8. Co-localisation](colocalisation.md) | SpatioEv app | tissue regions (+ fibers) | cell–cell and cell–matrix co-localisation |
| [9. Compare groups](cohort.md) | SpatioEv app | steps 7–8 for every core, and a sample sheet | statistics between groups |

You try the settings of steps 6–8 on **one core**, check the pictures, and
then press **Run for every core**, which repeats exactly the same settings for
all the others.

## Words you will meet

| Word | What it means here |
|---|---|
| **Core** | One tissue spot of a tissue microarray (TMA). Each core has its own folder. A whole slide works the same way: one folder per slide. |
| **Channel** / **marker** | One stain in the multiplexed image, such as `PCK`, `CD68` or `COL1`. The image holds one greyscale picture per channel. |
| **Segmentation mask** | A picture in which every cell is filled with its own number (its *label*). It says which pixels belong to which cell. |
| **Cell table** | A spreadsheet (CSV file) with one row per cell: its label, position, size and the average brightness of every marker. |
| **AnnData / `.h5ad`** | The file format SpatioEv uses for "cell table + cell types + positions" in one file. You never need to open it yourself. |
| **Terminal** | The Mac app where you type commands. You only copy and paste them. |
| **Environment** (`spatioev_env`) | A self-contained box holding SpatioEv and the exact software versions it needs, so it cannot clash with other programs. |
| **Folder path** | The address of a folder, like `/Users/you/TMA_072`. [How to get it without typing](app.md#getting-a-folder-path). |
| **Project root** | The folder that holds your core folders. SpatioEv writes its results into a `results` folder inside it. |
| **Sample ID** | The name of one core (or slide), the same as its folder name. |

## How to read this guide

Grey boxes like this are **commands**. Click the copy button on the right of
the box, paste into Terminal with ++cmd+v++, and press ++return++:

```bash
echo "Hello from Terminal"
```

Run one box at a time and wait for it to finish (the prompt, ending in `%`,
comes back) before the next.

!!! note "Notes like this explain *why*"
    You can skip them, but they help when something does not look as
    expected.

!!! warning "Warnings mark the places where people most often go wrong"

When you are stuck, see [Troubleshooting](troubleshooting.md), or send the
person who supports you the **full** error text and a screenshot.
