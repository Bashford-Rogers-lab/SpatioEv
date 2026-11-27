"""Resolve single-image and multi-FOV OME-TIFF collections."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import tifffile

OME_PATTERNS = ("*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff")


def zarr_array(root):
    """Return the pixel array from an opened zarr store, group or not.

    ``tifffile``'s ``aszarr()`` yields either a bare array or a group whose
    first pyramid level is under key ``"0"``, depending on the file.

    The obvious check, ``isinstance(root, zarr.hierarchy.Group)``, only works
    on zarr 2: ``zarr.hierarchy`` was removed in zarr 3, so it raises
    ``AttributeError: module 'zarr' has no attribute 'hierarchy'``. ``zarr.Group``
    exists in both majors and is the *same object* as ``zarr.hierarchy.Group``
    on zarr 2, so it is the portable spelling.
    """
    import zarr

    group_type = getattr(zarr, "Group", None)
    if group_type is not None and isinstance(root, group_type):
        return root["0"]
    # Fall back to duck-typing for any store type we do not recognise.
    if hasattr(root, "keys") and not hasattr(root, "shape"):
        return root["0"]
    return root


def natural_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
    ]


def image_files(source: Path) -> list[Path]:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    found: dict[Path, None] = {}
    for pattern in OME_PATTERNS:
        for path in source.glob(pattern):
            if (
                path.is_file()
                and not path.name.startswith("._")
                and "mask" not in path.stem.lower()
            ):
                found[path.resolve()] = None
    return sorted(found, key=lambda path: natural_key(path.name))


def imageid_key(value: str) -> str:
    value = str(value).strip().lower()
    value = re.sub(r"\.ome$", "", value)
    digits = re.findall(r"\d+", value)
    if digits:
        return str(int(digits[-1]))
    return re.sub(r"[^a-z0-9]+", "", value)


def image_map(source: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in image_files(source):
        key = imageid_key(path.stem)
        if key in mapping:
            raise ValueError(
                f"Multiple images resolve to FOV key {key!r}: {mapping[key]} and {path}"
            )
        mapping[key] = path
    return mapping


def resolve_image(source: Path, imageid: str | None = None) -> Path:
    source = Path(source).expanduser().resolve()
    if source.is_file():
        return source
    mapping = image_map(source)
    if not mapping:
        raise FileNotFoundError(f"No OME-TIFF images found under {source}")
    if imageid is None:
        return next(iter(mapping.values()))
    key = imageid_key(imageid)
    if key not in mapping:
        raise KeyError(f"No image under {source} matches image ID {imageid!r}")
    return mapping[key]


def raw_channel_names(image_path: Path) -> list[str]:
    image_path = Path(image_path).expanduser().resolve()
    with tifffile.TiffFile(image_path) as tif:
        if not tif.ome_metadata:
            return [f"C{index}" for index in range(tif.series[0].shape[0])]
        root = ET.fromstring(tif.ome_metadata)
        channels = [
            node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Channel"
        ]
        return [
            node.attrib.get("Name", f"C{index}") for index, node in enumerate(channels)
        ]


def generic_channel_names(channels: list[str]) -> bool:
    return bool(channels) and all(
        re.fullmatch(r"C\d+", str(channel)) for channel in channels
    )


def plane_marker_names(adata) -> list[str]:
    """One marker name per image plane, as recorded when the AnnData was built.

    Prepare AnnData stores the plane order in ``adata.uns["all_markers"]``
    (the marker order CSV's row order, or the OME channel names). That, not
    ``adata.var_names``, is what names the planes of an unnamed image: the two
    differ whenever a plane is not a marker (a repeated nuclear stain, a
    channel flagged ``remove``). Older files without it fall back to
    ``var_names``.
    """
    recorded = getattr(adata, "uns", {}).get("all_markers")
    if recorded is not None:
        names = [str(name) for name in np.asarray(recorded).ravel().tolist()]
        if names:
            return names
    return [str(name) for name in adata.var_names]


def channel_names(image_path: Path, fallback: list[str] | None = None) -> list[str]:
    channels = raw_channel_names(image_path)
    if generic_channel_names(channels) and fallback is not None:
        if len(fallback) != len(channels):
            raise ValueError(
                f"Unnamed image has {len(channels)} channels but the supplied marker list has {len(fallback)}"
            )
        return [str(marker) for marker in fallback]
    return channels


def collection_manifest(
    source: Path, imageids: list[str], *, allow_missing: bool = False
) -> list[dict[str, str]]:
    """One ``{imageid, image_path}`` row per image ID that has an image.

    An image ID without an image is an error unless ``allow_missing``; then it
    is simply left out (see :func:`imageids_with_images`).
    """
    mapping = image_map(source)
    rows = []
    missing = []
    for imageid in imageids:
        path = mapping.get(imageid_key(imageid))
        if path is None:
            missing.append(str(imageid))
        else:
            rows.append({"imageid": str(imageid), "image_path": str(path)})
    if missing and not allow_missing:
        raise ValueError(f"No OME-TIFF was found for FOVs: {missing}")
    return rows


def imageids_with_images(source: Path | str | None, imageids: list[str]) -> list[str]:
    """The image IDs that have an image under ``source``, in the given order.

    A single image file serves every image ID. A folder serves the FOVs whose
    OME-TIFF it holds. A TMA's images are around 1 GB each, so often only a
    few FOVs are copied locally while the cell tables cover all of them:
    everything computed from the cell tables still works for the others, and
    only image review and overlays need one of these.
    """
    if source is None or not str(source).strip():
        return []
    source = Path(source).expanduser()
    if source.is_file():
        return [str(imageid) for imageid in imageids]
    if not source.is_dir():
        return []
    keys = set(image_map(source))
    return [str(imageid) for imageid in imageids if imageid_key(imageid) in keys]
