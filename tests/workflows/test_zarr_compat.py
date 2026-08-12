"""Image reading must work across zarr majors.

Regression test for a user-reported crash:

    AttributeError: module 'zarr' has no attribute 'hierarchy'

``zarr.hierarchy`` is a zarr-2-only module; it was removed in zarr 3. The
image-overview readers used ``isinstance(root, zarr.hierarchy.Group)`` to tell
a pyramid group from a bare array, which raises on zarr 3.

Pinning zarr below 3 is not an option: ``tifffile.aszarr()`` requires zarr 3
from 2025 onwards and raises "zarr 2.x < 3 is not supported" otherwise. So the
code has to be version-agnostic, which is what ``zarr_array`` provides.
"""

from __future__ import annotations

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
zarr = pytest.importorskip("zarr")

from spatioev.workflows.image_collection import zarr_array  # noqa: E402


def test_zarr_group_type_is_reachable_without_hierarchy():
    """``zarr.Group`` exists in both majors; ``zarr.hierarchy`` does not."""
    assert hasattr(zarr, "Group"), "zarr.Group is the portable group type"


def test_zarr_array_unwraps_a_pyramid_group(tmp_path):
    """A multi-resolution OME-TIFF yields a group keyed by level."""
    path = tmp_path / "pyramid.ome.tif"
    data = np.random.default_rng(0).integers(0, 255, (3, 128, 128), dtype=np.uint16)
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(data, subifds=2, metadata={"axes": "CYX"})
        writer.write(data[:, ::2, ::2], subfiletype=1)
        writer.write(data[:, ::4, ::4], subfiletype=1)

    with tifffile.TiffFile(path) as tif:
        for level in range(len(tif.series[0].levels)):
            root = zarr.open(tif.series[0].levels[level].aszarr(), mode="r")
            array = zarr_array(root)
            out = np.asarray(array[:, ::2, ::2])
            assert out.ndim == 3
            assert out.shape[0] == 3


def test_zarr_array_passes_through_a_plain_array(tmp_path):
    """A flat OME-TIFF yields an array directly, which must pass through."""
    path = tmp_path / "flat.ome.tif"
    data = np.random.default_rng(1).integers(0, 255, (2, 64, 64), dtype=np.uint16)
    tifffile.imwrite(path, data, metadata={"axes": "CYX"}, ome=True)

    with tifffile.TiffFile(path) as tif:
        root = zarr.open(tif.series[0].levels[0].aszarr(), mode="r")
        array = zarr_array(root)
        np.testing.assert_array_equal(np.asarray(array), data)


def test_read_overview_array_end_to_end(tmp_path):
    """The function that actually crashed for the user."""
    marker_gating = pytest.importorskip("spatioev.workflows.marker_gating")

    path = tmp_path / "overview.ome.tif"
    data = np.random.default_rng(2).integers(0, 4000, (4, 512, 512), dtype=np.uint16)
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(data, subifds=1, metadata={"axes": "CYX"})
        writer.write(data[:, ::2, ::2], subfiletype=1)

    with tifffile.TiffFile(path) as tif:
        overview, *_ = marker_gating.read_overview_array(tif, max_dimension=128)

    assert overview.ndim == 3
    assert overview.shape[0] == 4
    assert max(overview.shape[1:]) <= 512


def test_no_module_uses_the_zarr2_only_hierarchy_api():
    """Guard against the pattern coming back anywhere in the package."""
    import pathlib

    import spatioev

    root = pathlib.Path(spatioev.__file__).parent
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if "_vendor" not in p.parts and "zarr.hierarchy" in p.read_text()
        and "image_collection" not in p.name  # the docstring explains why
    ]
    assert not offenders, f"zarr.hierarchy is zarr-2 only; found in {offenders}"
