# Image Tiling Helpers

These shell scripts tile selected channels from large OME-TIFF images into
smaller TIFF files for local QC/background inspection.

- `tile_selected_channels.sh` is the SLURM/cluster version.
- `tile_selected_channels_local.sh` is the local workstation version.

Example:

```bash
bash scripts/image_tiling/tile_selected_channels_local.sh \
  --input-image background/OnTIMEr18_n_a_backsub.ome.tif \
  --channels 0,1,26 \
  --tile-limit 25000 \
  --output-dir background
```

The generated TIFF tiles are ignored by Git through the root `.gitignore`.

