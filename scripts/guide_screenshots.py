#!/usr/bin/env python3
"""Regenerate the app screenshots of the step-by-step guide (docs/guide/images).

The screenshots show the synthetic demo data, so they can be published. To
regenerate them:

1. ``spatioev demo ~/SpatioEv_demo`` (fresh folder) and do the QuPath part of
   the guide (or use ``--with-classes``), so stages 05-07 have cell types.
2. ``spatioev ui --project-root ~/SpatioEv_demo --port 8517 --no-browser``
3. In another environment with Playwright (``pip install playwright``; it
   drives the installed Google Chrome, no browser download needed):
   ``python scripts/guide_screenshots.py home qupath fibers ...``
4. ``python scripts/guide_screenshots.py optimise`` to shrink the PNGs
   (needs Pillow) before committing them.

Steps run in the order given and build on each other's results (fibers before
regions, regions before colocalisation, all three before cohort). Your home
folder is shown as ``/Users/you`` in every picture.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

URL = "http://localhost:8517"
HOME = str(Path.home())
DEMO = Path.home() / "SpatioEv_demo"
IMAGES = Path(__file__).resolve().parents[1] / "docs" / "guide" / "images"
WIDTH = 1440


def idle(page: Page, timeout: float = 600.0) -> None:
    """Wait until Streamlit has finished rerunning (twice in a row, to skip short gaps)."""
    deadline = time.monotonic() + timeout
    calm = 0
    page.wait_for_timeout(400)
    while time.monotonic() < deadline:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        calm = 0 if running else calm + 1
        if calm >= 3:
            return
        page.wait_for_timeout(300)
    raise TimeoutError("Streamlit kept running")


def wait_text(page: Page, text: str, timeout: float = 900.0) -> None:
    page.get_by_text(text).first.wait_for(timeout=timeout * 1000)
    idle(page)


def open_page(page: Page, path: str = "") -> None:
    page.goto(f"{URL}/{path}")
    page.get_by_test_id("stMain").wait_for()
    idle(page)


def fill(page: Page, label: str, value: str) -> None:
    box = page.get_by_label(label, exact=True)
    box.fill(value)
    box.press("Enter")
    idle(page)


def number(page: Page, label: str, value: float | int) -> None:
    fill(page, label, str(value))


def click(page: Page, name: str) -> None:
    page.get_by_role("button", name=name).first.click()
    idle(page)


def tick(page: Page, label_start: str) -> None:
    page.get_by_test_id("stCheckbox").filter(has_text=label_start).first.locator("label").click()
    idle(page)


def expand(page: Page, title: str) -> None:
    page.get_by_test_id("stExpander").filter(has_text=title).first.locator("summary").click()
    idle(page)


def multiselect_add(page: Page, label: str, options: list[str]) -> None:
    widget = page.get_by_test_id("stMultiSelect").filter(has_text=label).first
    for option in options:
        widget.locator("input").click()
        page.get_by_role("option", name=option, exact=True).click()
        idle(page)
    page.keyboard.press("Escape")
    idle(page)


def select(page: Page, label: str, option: str) -> None:
    page.get_by_test_id("stSelectbox").filter(has_text=label).first.locator("input").click()
    page.get_by_role("option", name=option, exact=True).click()
    idle(page)


def mask_home(page: Page) -> None:
    """Show the home folder as /Users/you, in boxes and in text, until unmask_home()."""
    page.evaluate(
        """(home) => {
            const swap = (s) => s.split(home).join('/Users/you');
            window.__guideMasked = window.__guideMasked || [];
            document.querySelectorAll('input, textarea').forEach((el) => {
                if (el.value && el.value.includes(home)) {
                    window.__guideMasked.push([el, 'value', el.value]);
                    el.value = swap(el.value);
                }
            });
            const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
                const node = walker.currentNode;
                if (node.nodeValue.includes(home)) {
                    window.__guideMasked.push([node, 'nodeValue', node.nodeValue]);
                    node.nodeValue = swap(node.nodeValue);
                }
            }
        }""",
        HOME,
    )


def unmask_home(page: Page) -> None:
    """Put the real paths back, so later typing edits the real values (Streamlit inputs are React-controlled)."""
    page.evaluate(
        """() => {
            (window.__guideMasked || []).reverse().forEach(([target, key, value]) => { target[key] = value; });
            window.__guideMasked = [];
        }"""
    )


def shot(page: Page, name: str, start: str | None = None, end: str | None = None, *, sidebar: bool = False, pad: int = 12, pad_end: int = 12) -> Path:
    """Screenshot the main area from the element containing ``start`` to the one containing ``end``."""
    mask_home(page)
    # Streamlit scrolls inside its own container, so the window must be tall enough to hold the whole page.
    needed = page.evaluate("document.querySelector('[data-testid=\"stMainBlockContainer\"]').scrollHeight") + 200
    if needed > page.viewport_size["height"]:
        unmask_home(page)
        page.set_viewport_size({"width": WIDTH, "height": int(needed)})
        idle(page)
        mask_home(page)
    page.wait_for_timeout(300)
    main = page.get_by_test_id("stMain").bounding_box()
    block = page.locator('[data-testid="stMainBlockContainer"]').bounding_box() or main
    top = block["y"]
    bottom = block["y"] + block["height"]
    if start:
        box = page.get_by_test_id("stMain").get_by_text(start, exact=False).first.bounding_box()
        top = box["y"] - pad
    if end:
        box = page.get_by_test_id("stMain").get_by_text(end, exact=False).last.bounding_box()
        bottom = box["y"] + box["height"] + pad_end
    left = 0 if sidebar else block["x"] - pad
    right = block["x"] + block["width"] + pad
    path = IMAGES / f"{name}.png"
    page.screenshot(path=str(path), clip={"x": left, "y": max(0, top), "width": right - left, "height": bottom - max(0, top)})
    unmask_home(page)
    print("saved", path.name)
    return path


# --------------------------------------------------------------------------- steps
def step_home(page: Page) -> None:
    open_page(page)
    fill(page, "Project root", str(DEMO))
    fill(page, "Sample ID", "DEMO_01")
    click(page, "Apply")
    shot(page, "app_home", "SpatioEv image analysis", "07 Cohort comparison", sidebar=True, pad_end=70)


def step_qupath(page: Page) -> None:
    open_page(page, "qupath_bridge")
    fill(page, "TMA or project folder (one sub-folder per core)", str(DEMO))
    click(page, "Scan cores")
    shot(page, "qupath_scan", "QuPath bridge", "1 · Prepare cells for QuPath")
    click(page, "Write QuPath cell files")
    wait_text(page, "Wrote QuPath cell files")
    shot(page, "qupath_prepare", "1 · Prepare cells for QuPath", "Last run")
    shot(page, "qupath_prepare_done", "Last run")
    shot(page, "qupath_scripts", "2 · Classify in QuPath", "3 · Collect QuPath classes")


def step_collect(page: Page) -> None:
    open_page(page, "qupath_bridge")
    fill(page, "TMA or project folder (one sub-folder per core)", str(DEMO))
    click(page, "Collect classes")
    wait_text(page, "Built phenotyped AnnData")
    shot(page, "qupath_collect", "3 · Collect QuPath classes")


def step_fibers(page: Page) -> None:
    open_page(page, "fiber_segmentation")
    fill(page, "Sample ID", "DEMO_01")
    fill(page, "Project root", str(DEMO))
    click(page, "Fill standard sample paths")
    click(page, "Inspect inputs")
    multiselect_add(page, "Matrix channels to segment", ["COL12", "COL4", "COL6", "FN"])
    shot(page, "fibers_inputs", "Fiber segmentation", "Segmentation and quantification settings")
    expand(page, "Segmentation and quantification settings")
    number(page, "Minimum fiber area (px)", 50)
    tick(page, "Also count bright matrix")
    shot(page, "fibers_settings", "Segmentation and quantification settings", "HDM intensity range clip", pad_end=95)
    select(page, "Channel", "COL1")
    click(page, "Preview segmentation steps")
    wait_text(page, "Fibers in crop")
    page.wait_for_timeout(1500)
    shot(page, "fibers_preview", "Preview on a crop", "Run fiber segmentation")
    click(page, "Run fiber segmentation")
    wait_text(page, "Run for every core")
    page.wait_for_timeout(1500)
    shot(page, "fibers_results", "Fiber segmentation run", "fiber_manifest.json")
    shot(page, "batch_panel", "Run for every core", "Run for every core")
    click(page, "Run for every core")
    wait_text(page, "for every core:")
    shot(page, "fibers_batch", "Run for every core")
    qc = DEMO / "results" / "DEMO_04_fiber_segmentation" / "qc" / "DEMO_04__COL1_fiber_overlay.png"
    shutil.copy(qc, IMAGES / "fibers_qc.png")


def step_regions(page: Page) -> None:
    open_page(page, "tissue_regions")
    fill(page, "Sample ID", "DEMO_01")
    fill(page, "Project root", str(DEMO))
    click(page, "Fill standard sample paths")
    click(page, "Inspect inputs")
    shot(page, "regions_inputs", "Tissue regions", "Nest detection settings")
    click(page, "Assign tissue regions")
    wait_text(page, "Run for every core")
    page.wait_for_timeout(1500)
    shot(page, "regions_results", "Tissue regions run", "Matrix inside each region")
    shot(page, "regions_matrix", "Matrix inside each region", "compare each region with the same region")
    click(page, "Run for every core")
    wait_text(page, "for every core:")


def step_coloc(page: Page) -> None:
    open_page(page, "colocalization")
    fill(page, "Sample ID", "DEMO_01")
    fill(page, "Project root", str(DEMO))
    click(page, "Fill standard sample paths")
    click(page, "Inspect inputs")
    shot(page, "coloc_inputs", "Co-localisation", "High-matrix tile quantile", pad_end=90)
    click(page, "Run co-localisation")
    wait_text(page, "Run for every core")
    page.wait_for_timeout(1500)
    shot(page, "coloc_cell_cell", "Cell–cell co-localisation", "Neighbour ratio = observed")
    shot(page, "coloc_cell_matrix", "Cell–matrix co-localisation", "enrichment ratio")
    click(page, "Run for every core")
    wait_text(page, "for every core:")


def step_cohort(page: Page) -> None:
    open_page(page, "cohort_comparison")
    shot(page, "cohort_sheet", "Cohort comparison", "Save sample sheet")
    fill(page, "Sample sheet CSV", str(DEMO / "demo_sample_sheet.csv"))
    fill(page, "Comparison name", "demo_A_vs_B")
    shot(page, "cohort_settings", "Sample sheet CSV", "Run comparison")
    click(page, "Run comparison")
    wait_text(page, "Comparison run")
    wait_text(page, "Compared")
    page.wait_for_timeout(1500)
    shot(page, "cohort_results", "Comparison run", "Show feature families")


def step_fiber_preview(page: Page) -> None:
    open_page(page, "fiber_segmentation")
    fill(page, "Sample ID", "DEMO_01")
    fill(page, "Project root", str(DEMO))
    click(page, "Fill standard sample paths")
    click(page, "Inspect inputs")
    expand(page, "Segmentation and quantification settings")
    number(page, "Minimum fiber area (px)", 50)
    tick(page, "Also count bright matrix")
    select(page, "Channel", "COL1")
    click(page, "Preview segmentation steps")
    wait_text(page, "Fibers in crop")
    page.wait_for_timeout(1500)
    shot(page, "fibers_preview", "Preview on a crop", "Run fiber segmentation")


def step_optimise(page: Page | None = None) -> None:
    """Shrink every guide image to at most 1600 px wide and 256 colours (needs Pillow)."""
    from PIL import Image

    for path in sorted(IMAGES.glob("*.png")):
        before = path.stat().st_size
        image = Image.open(path).convert("RGB")
        if image.width > 1600:
            image = image.resize((1600, round(image.height * 1600 / image.width)), Image.LANCZOS)
        image.quantize(colors=256, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(path, optimize=True)
        print(f"{path.name}: {before // 1024} -> {path.stat().st_size // 1024} KB")


STEPS = {
    "home": step_home, "fiber_preview": step_fiber_preview, "optimise": step_optimise, "qupath": step_qupath, "collect": step_collect, "fibers": step_fibers,
    "regions": step_regions, "coloc": step_coloc, "cohort": step_cohort,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=list(STEPS))
    parser.add_argument("--show", action="store_true", help="Show the browser window")
    args = parser.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)
    if args.steps == ["optimise"]:
        step_optimise()
        return
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=not args.show)
        page = browser.new_page(viewport={"width": WIDTH, "height": 4200}, device_scale_factor=2)
        for name in args.steps:
            print("step", name)
            STEPS[name](page)
        browser.close()


if __name__ == "__main__":
    main()
