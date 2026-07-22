"""Resolve single-image and multi-FOV OME-TIFF collections."""

from __future__ import annotations

import re
from pathlib import Path
from xml.etree import ElementTree as ET

import tifffile


OME_PATTERNS = ("*.ome.tif", "*.ome.tiff", "*.tif", "*.tiff")


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


def channel_names(image_path: Path, fallback: list[str] | None = None) -> list[str]:
    channels = raw_channel_names(image_path)
    if generic_channel_names(channels) and fallback is not None:
        if len(fallback) != len(channels):
            raise ValueError(
                f"Unnamed image has {len(channels)} channels but the supplied marker list has {len(fallback)}"
            )
        return [str(marker) for marker in fallback]
    return channels


def collection_manifest(source: Path, imageids: list[str]) -> list[dict[str, str]]:
    mapping = image_map(source)
    rows = []
    missing = []
    for imageid in imageids:
        path = mapping.get(imageid_key(imageid))
        if path is None:
            missing.append(str(imageid))
        else:
            rows.append({"imageid": str(imageid), "image_path": str(path)})
    if missing:
        raise ValueError(f"No OME-TIFF was found for FOVs: {missing}")
    return rows
