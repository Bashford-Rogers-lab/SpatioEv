# Troubleshooting

Find the message you see, or the step you are on. If nothing here helps, see
[Asking for help](#asking-for-help) at the bottom.

## Terminal

| What you see | Why | What to do |
|---|---|---|
| `command not found: conda` | Anaconda is not installed, or Terminal was opened before it was installed. | Quit Terminal with ++cmd+q++, reopen it. If it persists, [install Anaconda](install.md#install-anaconda-skip-if-you-have-it). |
| `command not found: spatioev` | Terminal is not in SpatioEv's environment. | Run `conda activate spatioev_env`. The prompt should start with `(spatioev_env)`. |
| `No such file or directory` after pasting a path | The path is mistyped, or contains spaces without quotes. | Drag the folder from Finder onto Terminal instead of typing. Paths with spaces need quotes: `"/Users/you/My data"`. |
| `Operation not permitted` | macOS blocks Terminal from that folder (Desktop, Documents, Downloads or an external drive). | When macOS asks *"Terminal" would like to access files in…*, click **Allow**. If you clicked *Don't Allow* earlier:  › *System Settings* › *Privacy & Security* › *Files and Folders* › *Terminal*, and switch the folder on. |
| The prompt does not come back | A command is still running. | Wait. To stop it, press ++ctrl+c++. |

## The app

| What you see | Why | What to do |
|---|---|---|
| The browser does not open | Some browsers block it. | Open `http://localhost:8501` yourself. |
| `Port 8501 is already in use` | The app is already running in another Terminal window. | Use that window's app, or start a second one with `spatioev ui --port 8502 --project-root …` and open `http://localhost:8502`. |
| *Connection error* in the browser | The Terminal window running the app was closed or stopped. | Start the app again ([Open the app](app.md)). Background runs that had started keep going. |
| `No core folders found` | The folder is not the project root, or the core folders lack `segmentation` and `quantification` sub-folders. | Check the layout with `spatioev qupath status <folder>` ([Organise your data](organise.md#check-the-layout)). |
| *Inspect inputs* says a file is not found | The Sample ID does not match a core folder, or the paths were not refreshed. | Check the Sample ID spelling (it is case-sensitive), press **Fill standard sample paths**, then **Inspect inputs** again. |
| A run shows a red error | Something in the input or settings did not work. | Open **Error details** and read the last line; it usually names the problem. Send the whole text if you ask for help. |
| A progress bar seems stuck | Large images take a while; the bar moves between steps, not continuously. | Open **Worker log** under the progress bar. If the last line has not changed for more than 15 minutes, ask for help. |
| *Run for every core* reports **missing input** for some cores | An earlier step has not been done for those cores. | The message says which input is missing, for example *no cell AnnData* (run [QuPath collect](qupath.md) for that core) or *no fiber segmentation result* (run [step 6](fibers.md) for it). Then press **Run for every core** again; finished cores are skipped. |
| *Run for every core* reports **failed** for a core | That core's run hit an error. | Open `logs/<core>_<stage>.log` in that core's results folder; the end of the file shows the error. Other cores are not affected. |

## Clustering and gating (steps 00–03)

| What you see | Why | What to do |
|---|---|---|
| *Marker order CSV lists … markers but … has … image channels* (page 00) | The marker CSV found does not belong to these images, often an older copy in a parent folder. | Type the path of the right `markers_bs.csv` into **Marker order CSV**. Check the **Marker order** tab before building. |
| A warning that the AnnData *was built by an older SpatioEv* (pages 01–02) | Made with SpatioEv 0.2.0 or earlier, which paired markers with the wrong image planes. | Rebuild it with page 00, then redo clustering and gating ([step 5b](clustering_gating.md#b-prepare-anndata-page-00)). |
| Clusters on page 01 come mostly from one or two FOVs | Cores were stained or imaged unevenly, and scaling over all cells keeps those differences. | Tick **Z-score within each FOV** and run Level-0 clustering again ([step 5b](clustering_gating.md#c-find-the-tissue-structures-page-01)). |
| Image review is greyed out, or a FOV is missing from the FOV list (pages 01–03) | That FOV has no image in the image folder. Its cells are still converted and analysed. | Copy its OME-TIFF into the folder (name it with the FOV number, e.g. `22.ome.tif`), or review another FOV. |
| *Workflow markers absent from AnnData* (page 03) | A marker column of the rules file matches no marker, not even after ignoring suffixes such as `_1`/`_2`. | Rename the column to the marker name in your panel, or delete it. |
| *Workflow markers absent from gate CSV* (page 03) | The gate table has no row for a marker the rules use. | Add a gate for it (step 02 calculates one for every marker), or remove it from the rules. |
| *No gate for SOX9 in fov2, …* (page 03) | A per-FOV gate table has no value for those FOVs. | Choose how to fill them under **Images without a gate for a marker**, or list only the gated FOVs in **Image IDs to phenotype**. |
| A note that a row uses a single `pos` / `neg` marker (page 03) | scimap only averages that marker into the score, so cells that break it can still get the type. | Use `allpos` / `allneg` when the marker must be positive / negative ([why](clustering_gating.md#the-rules-file)). |
| *N cells … were set to Unknown because their phenotype had fewer than K cells in that image* | **Minimum cells per phenotype per image** removes small populations per FOV. | Set it to 0 for small cores, or lower it. |
| Many cells of a broadly expressed marker are called negative | The marker was left blank in the staining questionnaire, so it got a cautious upper-tail gate. | Answer **Staining condition**, **Compartment** and **Artifact** for it on page 02 and calculate again. |
| A marker is flagged `3GMM_high_is_majority` and called positive in only a few percent of cells (page 02) | Its values have no separate positive peak. The usual gate would have called most cells positive, so a stricter upper-tail gate was used, which calls at most about a quarter of cells positive. | Check the marker on the image (**Generate spatial QC images**). If it really is in many cells, move its gate in napari review. |
| Distances or areas look 3× too large or too small (pages 04–06), or page 04 says **Check the pixel size** | The pixel size is wrong. Some scanners write a wrong one into the OME-TIFF, and page 04 fills it in from there. Pages 05 and 06 copy it from the fiber result. | Check it: a typical nucleus is 5–10 µm across. Type your scan's pixel size (0.325 µm/px for our TMA scans) on page 04 and run again. Then check that pages 05 and 06 show the same value. |

## QuPath

| What you see (in QuPath's log) | Why | What to do |
|---|---|---|
| `SpatioEv: no image is open` | The script ran with nothing open. | Use the script editor's *Run › Run for project*, or open a core first by double-clicking it in the *Project* tab. |
| `SpatioEv: no cell file for <core>` | SpatioEv has not written the cells for that core, or the image is not in its core's `background` folder. | Press **Write QuPath cell files** in the app ([step A](qupath.md#a-prepare-the-cells-in-spatioev)). Add images to QuPath from `<core>/background/`, not from a copy elsewhere. |
| `already holds … SpatioEv cells - skipping` | The cells were imported before. This protects your classifications. | Nothing to do. To re-import on purpose, open the import script, set `REPLACE_EXISTING = true`, run it, and set it back to `false`. |
| `… cells have no SpatioEv cell_id … will be ignored` | Cells were created in QuPath (for example with *Cell detection*). | Delete those cells. Only cells imported from SpatioEv carry the ID that links them back. |
| The cells do not line up with the tissue | The QuPath image is not the core's `background/<core>.ome.tiff`. | Remove the image from the project and add the right file. |

## Results look wrong

| What you see | What to try |
|---|---|
| Fiber QC: fibers missed, or filled patches only outlined | Tick **Also count bright matrix** ([step 6](fibers.md#settings)). |
| Fiber QC: many tiny specks on background | Raise **Minimum fiber area** (50 px works for the 0.325 µm/px TMA scans). |
| Fiber QC: whole diffuse areas filled in magenta | Raise **Bright matrix: minimum level**, or set the local window to 0. |
| Tissue regions: nests are rings or broken up | Raise **Gap closing** or **Boundary grid** in *Nest detection settings* ([step 7](regions.md)). |
| Tissue regions: separate nests merged into one | Lower **Linking radius scale**. |
| Compare groups: warning about the *smallest possible p-value* | There are too few patients per group for any test to reach p < 0.05. Treat the results as descriptive, or add patients. |

## Asking for help

Send the person who supports you:

1. What you were doing (which step, which core).
2. A screenshot of the screen (++cmd+shift+4++, then drag over the area; the
   picture appears on your Desktop).
3. The **full** error text: in the app, open *Error details* and copy
   everything; in Terminal, select all the text and copy it.
4. The output of this command:

```bash
conda activate spatioev_env && cd ~/SpatioEv && git log -1 --format="%h %cd" && python -c "import spatioev; print(spatioev.__version__)"
```
