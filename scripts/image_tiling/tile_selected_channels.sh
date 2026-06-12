#!/bin/bash

#SBATCH -A immune-rep.prj
#SBATCH -J tile_selected_channels
#SBATCH -o tile_selected_channels-%j.out
#SBATCH -e tile_selected_channels-%j.err
#SBATCH -p short
#SBATCH -c 4

set -euo pipefail

module load Anaconda3/2024.02-1
eval "$(conda shell.bash hook)"
conda activate ashlar

INPUT_IMAGE=""
WORKDIR=""
OUTPUT_DIR="background"
CHANNELS="0,1,26"
TILE_LIMIT=25000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-image)
            INPUT_IMAGE="$2"
            shift 2
            ;;
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        --tile-limit)
            TILE_LIMIT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            WORKDIR="$1"
            shift
            ;;
    esac
done

WORKDIR=${WORKDIR:-$PWD}
cd "$WORKDIR"
echo "Working in: $WORKDIR"

if [[ -z "$INPUT_IMAGE" ]]; then
    INPUT_IMAGE=$(find "$OUTPUT_DIR" -maxdepth 1 -type f ! -name "._*" \( -name "*.ome.tif" -o -name "*.ome.tiff" -o -name "*.tif" -o -name "*.tiff" \) | sort | head -n 1 || true)
fi

if [[ -z "$INPUT_IMAGE" ]]; then
    echo "No input image found. Pass --input-image or place one in $OUTPUT_DIR." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "Input image: $INPUT_IMAGE"
echo "Channels to tile: $CHANNELS"
echo "Tile limit: $TILE_LIMIT"
echo "Output directory: $OUTPUT_DIR"

python - "$INPUT_IMAGE" "$CHANNELS" "$TILE_LIMIT" "$OUTPUT_DIR" <<'PY'
import math
import sys
from pathlib import Path

import tifffile


def parse_channels(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(int(part))
    return values


def resolve_cyx_shape(series):
    axes = series.axes
    shape = series.shape
    axis_map = {ax: i for i, ax in enumerate(axes)}
    if "C" not in axis_map or "Y" not in axis_map or "X" not in axis_map:
        raise ValueError(f"Unsupported axes layout {axes!r}; expected at least C, Y, and X.")
    return axes, shape, axis_map


def read_channel(input_image, channel_index, axes, shape, axis_map):
    # Fast path for the common OME-TIFF layout where each channel can be read
    # directly by page index, matching the original cluster helper behavior.
    try:
        channel = tifffile.imread(input_image, key=channel_index)
        if channel.ndim == 2:
            return channel
    except Exception:
        pass

    # Fallback for files where tifffile exposes a higher-dimensional series.
    arr = tifffile.imread(input_image)
    while arr.ndim > len(axes):
        arr = arr[0]
    if arr.ndim != len(axes):
        raise ValueError(f"Unexpected array shape {arr.shape} for axes {axes}.")

    selectors = []
    for ax, size in zip(axes, shape):
        if ax == "C":
            if channel_index >= size:
                raise IndexError(
                    f"Requested channel {channel_index} but image only has {size} channels."
                )
            selectors.append(channel_index)
        elif ax in {"Y", "X"}:
            selectors.append(slice(None))
        else:
            selectors.append(0)
    channel = arr[tuple(selectors)]
    if channel.ndim != 2:
        raise ValueError(
            f"Channel {channel_index} resolved to shape {channel.shape}; expected 2D tileable image."
        )
    return channel


input_image = Path(sys.argv[1])
channels = parse_channels(sys.argv[2])
tile_limit = int(sys.argv[3])
output_dir = Path(sys.argv[4])

with tifffile.TiffFile(input_image) as tf:
    series = tf.series[0]
    axes, shape, axis_map = resolve_cyx_shape(series)
    height = shape[axis_map["Y"]]
    width = shape[axis_map["X"]]
    n_channels = shape[axis_map["C"]]

print(f"Detected axes={axes}, shape={shape}")
print(f"Spatial size: {height}x{width}, channels: {n_channels}")

n_rows = math.ceil(height / tile_limit)
n_cols = math.ceil(width / tile_limit)
print(f"Tiling into {n_rows} row(s) x {n_cols} column(s)")

for channel_index in channels:
    print(f"Reading channel {channel_index}")
    channel_img = read_channel(input_image, channel_index, axes, shape, axis_map)

    for row in range(n_rows):
        y0 = row * tile_limit
        y1 = min((row + 1) * tile_limit, height)
        for col in range(n_cols):
            x0 = col * tile_limit
            x1 = min((col + 1) * tile_limit, width)
            tile = channel_img[y0:y1, x0:x1]
            tile_name = f"c{channel_index}_r{row}c{col}.tiff"
            tifffile.imwrite(output_dir / tile_name, tile)

print("Done.")
PY

echo "✅ Finished tiling selected channels into $OUTPUT_DIR."
