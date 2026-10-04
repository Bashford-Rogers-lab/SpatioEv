# 2. Practise on demo data

SpatioEv can create a small **synthetic** dataset: six made-up TMA cores in
exactly the layout real data uses, with the same 17-channel panel (DAPI, PCK,
CD3, CD31, CD45, CD68, CD90, PDPN, SMA, VIM, WNT5A and the matrix channels
COL1, COL4, COL6, COL12, FN). It contains no real patient data. All the
screenshots in this guide were made with it.

Working through the guide on demo data first means you see every screen,
and what a good result looks like, before it matters.

## Create the demo data

In Terminal:

```bash
conda activate spatioev_env
```

```bash
spatioev demo ~/SpatioEv_demo
```

This takes under a minute and creates the folder `SpatioEv_demo` in your home
folder:

```text
SpatioEv_demo/
├── DEMO_01/ … DEMO_06/          ← six cores, each with background/, segmentation/, quantification/
├── demo_sample_sheet.csv        ← DEMO_01–03 are group A, DEMO_04–06 group B
└── README.txt
```

The two groups differ on purpose (group B has more aligned collagen and more
WNT5A+ fibroblasts), so [comparing groups](cohort.md) has something to find.

## Two ways to practise

=== "With QuPath (the full workflow)"

    Use the folder as it is and follow every step from
    [Organise your data](organise.md) onwards, including classifying the
    demo cells yourself in [QuPath](qupath.md).

=== "Without QuPath (SpatioEv steps only)"

    If you want to try the SpatioEv steps before learning QuPath, create the
    demo with ready-made cell classes instead, as if they had been exported
    from QuPath:

    ```bash
    spatioev demo ~/SpatioEv_demo --with-classes
    ```

    Then, in [step 5](qupath.md), skip the QuPath part and go straight to
    [F. Collect the classes](qupath.md#f-collect-the-classes-in-spatioev).

!!! tip "Start again at any time"
    Delete the `SpatioEv_demo` folder in Finder (nothing else on your Mac
    depends on it) and run the command again. `--overwrite` replaces only the
    core folders, not the results you made.

Next: [Organise your data](organise.md), or, for the demo,
[Open the app](app.md) with:

```bash
spatioev ui --project-root ~/SpatioEv_demo
```
