from ..visualization.spatial import inspect_clusters
from .annotation import annotate_interactive

def annotate_refinements(results, image_path):
    annotated = {}
    mappings = {}

    for name, ad in results.items():
        print(f"\nInspecting {name}\n")

        inspect_clusters(ad, image_path=image_path, label="leiden", block=True)

        ad, mapping = annotate_interactive(ad)

        annotated[name] = ad
        mappings[name] = mapping

    return annotated, mappings
