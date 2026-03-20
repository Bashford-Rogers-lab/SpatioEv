import scimap as sm
import napari


def inspect_clusters(
    adata,
    image_path,
    label="leiden",
    block=True
):

    # open viewer via scimap
    sm.pl.image_viewer(
        image_path=image_path,
        adata=adata,
        overlay=label,
        point_size=10,
        point_color="white"
    )

    if not block:
        return

    print("\nInspect clusters in napari. Close the viewer to continue.\n")

    viewer = napari.current_viewer()

    if viewer is None:
        print("Warning: napari viewer not detected.")
        return

    # Wait until viewer closes
    from qtpy.QtCore import QEventLoop

    loop = QEventLoop()

    viewer.window._qt_window.destroyed.connect(loop.quit)

    loop.exec_()