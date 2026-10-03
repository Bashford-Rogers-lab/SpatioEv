/**
 * QuPath -> SpatioEv: export the cell classes (and annotations) of one core.
 *
 * Writes, next to the cells file that was imported,
 *     <core folder>/qupath/<core>_cell_classes.csv     cell_id -> QuPath class
 *     <core folder>/qupath/<core>_annotations.geojson  all annotations
 * Then on the SpatioEv side:
 *     spatioev qupath collect <TMA folder>
 *
 * Use after classifying: run for the open image, or Automate > Run for
 * project. Re-run whenever the classifier changes; nothing else needs
 * re-importing.
 *
 * Annotation classes starting with "Exclude", "Ignore" or "Artefact" remove
 * the cells inside them from the SpatioEv analysis. Any other classified
 * annotation (e.g. "Tumour", "Stroma") is recorded for each cell it contains.
 */

// --- settings ---------------------------------------------------------------
def BRIDGE_DIR = null          // same as in spatioev_import_cells.groovy
// ----------------------------------------------------------------------------

def scriptArgs = binding.hasVariable("args") ? (args as List) : []
def imageData = getCurrentImageData()
if (imageData == null) {
    print "SpatioEv: no image is open. Double-click a core in the Project list and press Run, " +
          "or use Run > Run for project to export every image at once."
    return
}
def uris = imageData.getServer().getURIs()
if (uris == null || uris.isEmpty()) {
    print "SpatioEv: cannot tell which file this image came from, so the core folder is unknown. Set BRIDGE_DIR."
    return
}
def imageFile = new File(uris.iterator().next())
def core = imageFile.getName().replaceAll(/(?i)(\.ome)?\.(tiff?|zarr)$/, "")
def outDir
if (scriptArgs.size() > 0 && scriptArgs[0]) {
    outDir = new File(scriptArgs[0] as String)
} else if (BRIDGE_DIR != null) {
    outDir = new File(BRIDGE_DIR as String)
} else {
    outDir = new File(imageFile.getParentFile().getParentFile(), "qupath")
}
outDir.mkdirs()

def quote = { value ->
    def text = value == null ? "" : value.toString()
    (text.contains(",") || text.contains("\"")) ? "\"" + text.replace("\"", "\"\"") + "\"" : text
}

def cells = getCellObjects()
def unnamed = cells.count { it.getName() == null || !it.getName().startsWith(core + "_") }
def lines = new StringBuilder("image,cell_id,qupath_class,centroid_x_px,centroid_y_px\n")
cells.each { cell ->
    def pathClass = cell.getPathClass()
    lines << [core, cell.getName(), pathClass == null ? "Unclassified" : pathClass.toString(),
              cell.getROI().getCentroidX(), cell.getROI().getCentroidY()].collect(quote).join(",") << "\n"
}
def classesFile = new File(outDir, core + "_cell_classes.csv")
def temporary = new File(outDir, classesFile.getName() + ".tmp")
temporary.text = lines.toString()
java.nio.file.Files.move(temporary.toPath(), classesFile.toPath(), java.nio.file.StandardCopyOption.REPLACE_EXISTING)

def annotations = getAnnotationObjects()
def annotationsFile = new File(outDir, core + "_annotations.geojson")
exportObjectsToGeoJson(annotations, annotationsFile.getAbsolutePath(), "FEATURE_COLLECTION")

def counts = cells.countBy { it.getPathClass() == null ? "Unclassified" : it.getPathClass().toString() }
print "SpatioEv: exported ${cells.size()} cells from ${core} to ${classesFile.getName()} " +
      "(${annotations.size()} annotations). Classes: ${counts}"
if (unnamed > 0) {
    print "SpatioEv: warning - ${unnamed} cells in ${core} have no SpatioEv cell_id (created in QuPath?) and will be ignored."
}
