"""Removing each cell's overall brightness before refinement clustering (page 01)."""

from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("scanpy")

from spatioev.workflows import clustering  # noqa: E402
from spatioev.workflows.clustering import remove_cell_brightness  # noqa: E402


def _adata(values, markers):
    adata = ad.AnnData(X=np.asarray(values, dtype=np.float32))
    adata.var_names = markers
    adata.obs_names = [f"c{i}" for i in range(adata.n_obs)]
    return adata


def test_brightness_removal_keeps_each_cells_marker_profile():
    source = _adata([[1, 2, 3, 50], [11, 12, 13, 50], [np.nan, 2, 4, 50]], ["A", "B", "C", "D"])

    out = remove_cell_brightness(source, ["A", "B", "C"])

    assert list(out.var_names) == ["A", "B", "C"]
    # A dim and a bright cell with the same profile end up identical.
    np.testing.assert_allclose(out.X[0], [-1, 0, 1])
    np.testing.assert_allclose(out.X[1], [-1, 0, 1])
    # A missing marker is skipped, not counted as zero.
    np.testing.assert_allclose(out.X[2], [np.nan, -1, 1])


def test_brightness_removal_needs_two_markers():
    with pytest.raises(ValueError, match="at least two markers"):
        remove_cell_brightness(_adata([[1, 2]], ["A", "B"]), ["A"])


def _project(tmp_path: Path) -> tuple[dict, pd.Series]:
    """Two cell types that differ in which marker stands out, under a strong
    per-cell brightness that raises every marker together."""
    rng = np.random.default_rng(0)
    n = 400
    kind = np.repeat(["A", "B"], n // 2)
    profile = np.where(kind[:, None] == "A", [1.0, 0.0, 0.5], [0.0, 1.0, 0.5])
    brightness = rng.normal(0, 3, (n, 1))
    values = profile + brightness + rng.normal(0, 0.15, (n, 3))
    source = _adata(values, ["SOX9", "MAGEA4", "AMH"])
    source.obs["imageid"] = "fov1"
    source_path = tmp_path / "source.h5ad"
    source.write_h5ad(source_path)

    level0 = source.copy()
    level0.obs["leiden"] = "0"
    level0_path = tmp_path / "level0.h5ad"
    level0.write_h5ad(level0_path)
    mapping_path = tmp_path / "mapping.csv"
    pd.DataFrame({"cluster": ["0"], "annotation": ["Tubule"]}).to_csv(mapping_path, index=False)

    config = {
        "sample_id": "S",
        "adata_path": str(source_path),
        "output_dir": str(tmp_path / "out"),
        "level0_h5ad": str(level0_path),
        "level0_mapping_path": str(mapping_path),
        "refinements": [
            {
                "annotation": "Tubule",
                "markers": ["SOX9", "MAGEA4", "AMH"],
                "resolution": 0.3,
                "n_neighbors": 15,
                "n_pcs": 2,
                "scale": True,
                "remove_brightness": True,
            }
        ],
    }
    return config, pd.Series(kind, index=source.obs_names)


def test_refinement_clusters_by_profile_once_brightness_is_removed(tmp_path: Path):
    config, kind = _project(tmp_path)

    outputs = clustering.run_refinement(config, tmp_path / "status.json")["Tubule"]

    assert outputs["remove_brightness"] is True
    clustered = ad.read_h5ad(outputs["clustered_h5ad"])
    np.testing.assert_allclose(np.asarray(clustered.X).mean(axis=1), 0, atol=1e-5)
    labels = clustered.obs["leiden"].astype(str)
    majority = pd.crosstab(labels, kind.reindex(labels.index)).max(axis=1).sum() / len(labels)
    assert majority > 0.95
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "complete"


def test_brightness_removal_is_off_unless_asked(tmp_path: Path):
    config, _ = _project(tmp_path)
    del config["refinements"][0]["remove_brightness"]

    outputs = clustering.run_refinement(config, tmp_path / "status.json")["Tubule"]

    assert outputs["remove_brightness"] is False
    clustered = ad.read_h5ad(outputs["clustered_h5ad"])
    assert np.abs(np.asarray(clustered.X).mean(axis=1)).max() > 1


def test_brightness_removal_needs_z_scored_markers(tmp_path: Path):
    config, _ = _project(tmp_path)
    config["refinements"][0]["scale"] = False

    with pytest.raises(ValueError, match="needs z-scored markers"):
        clustering.run_refinement(config, tmp_path / "status.json")


def test_heatmap_colours_stop_at_three_and_every_mean_is_printed(tmp_path: Path, monkeypatch):
    """A tiny cluster at -7 must not wash out the colours of the others."""
    captured = {}

    def fake_heatmap(data, **options):
        captured.update(options, data=data)

    monkeypatch.setattr(clustering.sns, "heatmap", fake_heatmap)
    clustered = _adata([[1.0, 0.5], [-7.0, 0.2], [0.8, -0.4]], ["CYP17A1", "SMA"])
    clustered.obs["leiden"] = ["0", "1", "0"]

    clustering.save_heatmap(clustered, tmp_path / "scaled.png", title="t")
    assert (captured["vmin"], captured["vmax"], captured["center"]) == (-3.0, 3.0, 0)
    assert captured["annot"] is True
    assert captured["data"].loc["1", "CYP17A1"] == pytest.approx(-7.0)

    captured.clear()
    clustering.save_heatmap(clustered, tmp_path / "raw.png", title="t", scaled=False)
    assert "vmin" not in captured and captured["annot"] is True
