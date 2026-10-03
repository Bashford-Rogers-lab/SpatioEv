/**
 * SpatioEv -> QuPath: import the segmented cells of one core.
 *
 * Each cell arrives with its exact outline, its matched nucleus, its SpatioEv
 * cell_id as the object name, and the marker measurements (bare marker name =
 * whole-cell mean; "<marker> nucleus" = nuclear mean).
 *
 * Use: open the core image (or select all cores) and run this script, or
 * Automate > Run for project. It looks for
 *     <core folder>/qupath/<core>_cells.geojson
 * next to the image <core folder>/background/<core>.ome.tiff, as written by
 *     spatioev qupath prepare <TMA folder>
 * Set BRIDGE_DIR if the GeoJSON files are all in one folder instead.
 *
 * Import once per image. Re-running skips images that already hold SpatioEv
 * cells, so classifications are never wiped by accident; set REPLACE_EXISTING
 * to true to re-import (this deletes those cells and their classes).
 */

// --- settings ---------------------------------------------------------------
def BRIDGE_DIR = null          // e.g. "/path/to/geojson_folder"; null = per-core layout
def REPLACE_EXISTING = false
// ----------------------------------------------------------------------------

def scriptArgs = binding.hasVariable("args") ? (args as List) : []
def imageData = getCurrentImageData()
if (imageData == null) {
    print "SpatioEv: no image is open. Double-click a core in the Project list and press Run, " +
          "or use Run > Run for project to import cells into every image at once."
    return
}
def uris = imageData.getServer().getURIs()
if (uris == null || uris.isEmpty()) {
    print "SpatioEv: cannot tell which file this image came from, so the core folder is unknown. Set BRIDGE_DIR."
    return
}
def imageFile = new File(uris.iterator().next())
def core = imageFile.getName().replaceAll(/(?i)(\.ome)?\.(tiff?|zarr)$/, "")
def geojson
if (scriptArgs.size() > 0 && scriptArgs[0]) {
    geojson = new File(scriptArgs[0] as String)
} else if (BRIDGE_DIR != null) {
    geojson = new File(BRIDGE_DIR as String, core + "_cells.geojson")
} else {
    geojson = new File(new File(imageFile.getParentFile().getParentFile(), "qupath"), core + "_cells.geojson")
}
if (!geojson.exists()) {
    print "SpatioEv: no cell file for ${core} at ${geojson} - run 'spatioev qupath prepare' first. Skipping."
    return
}

def existing = getCellObjects().findAll { it.getName() != null && it.getName().startsWith(core + "_") }
if (!existing.isEmpty()) {
    if (!REPLACE_EXISTING) {
        print "SpatioEv: ${core} already holds ${existing.size()} SpatioEv cells - skipping (set REPLACE_EXISTING = true to re-import)."
        return
    }
    removeObjects(existing, false)
    print "SpatioEv: removed ${existing.size()} previously imported cells from ${core}."
}

importObjectsFromFile(geojson.getAbsolutePath())
def imported = getCellObjects().findAll { it.getName() != null && it.getName().startsWith(core + "_") }
def withNucleus = imported.count { it.getNucleusROI() != null }
print "SpatioEv: imported ${imported.size()} cells into ${core} (${withNucleus} with a nucleus) from ${geojson.getName()}."
