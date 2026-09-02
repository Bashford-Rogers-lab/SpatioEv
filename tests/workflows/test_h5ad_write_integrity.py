"""An interrupted conversion must not leave a half-written H5AD behind.

Writing straight to the destination left a truncated file there whenever the
worker was killed mid-write. HDF5 keeps its superblock, so the file still looks
openable and only failed later in the clustering page with h5py's opaque
"unable to determine object type".
"""

import anndata as ad
import numpy as np
import pandas as pd
import pytest

# spatioev.workflows.cellsam imports tifffile at module level, so a missing
# image-IO stack must skip this file rather than abort collection.
pytest.importorskip("tifffile")

from spatioev.workflows.cellsam import (  # noqa: E402
    describe_unreadable_h5ad,
    write_h5ad_atomically,
)


def _adata(rows=5, columns=3):
    return ad.AnnData(
        X=np.zeros((rows, columns), dtype=np.float32),
        obs=pd.DataFrame(index=[f"cell{index}" for index in range(rows)]),
        var=pd.DataFrame(index=[f"M{index}" for index in range(columns)]),
    )


def test_atomic_write_produces_a_readable_file(tmp_path):
    target = tmp_path / "out.h5ad"
    write_h5ad_atomically(_adata(), target)
    assert describe_unreadable_h5ad(target) is None
    assert ad.read_h5ad(target).shape == (5, 3)


def test_a_failed_write_leaves_no_file_at_the_target(tmp_path):
    class Failing(ad.AnnData):
        def write_h5ad(self, *args, **kwargs):
            raise RuntimeError("simulated interruption mid-write")

    target = tmp_path / "out.h5ad"
    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_h5ad_atomically(Failing(X=np.zeros((2, 2), dtype=np.float32)), target)
    assert not target.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_a_failed_rewrite_keeps_the_previous_good_file(tmp_path):
    """The destructive case: re-running a conversion over a working file."""

    class Failing(ad.AnnData):
        def write_h5ad(self, *args, **kwargs):
            raise RuntimeError("simulated interruption mid-write")

    target = tmp_path / "out.h5ad"
    write_h5ad_atomically(_adata(rows=7), target)
    with pytest.raises(RuntimeError):
        write_h5ad_atomically(Failing(X=np.zeros((2, 2), dtype=np.float32)), target)
    assert describe_unreadable_h5ad(target) is None
    assert ad.read_h5ad(target).n_obs == 7


def test_no_partial_files_are_left_beside_a_successful_write(tmp_path):
    target = tmp_path / "out.h5ad"
    write_h5ad_atomically(_adata(), target)
    assert [path.name for path in tmp_path.iterdir()] == ["out.h5ad"]


# --------------------------------------------------------------------------
# describe_unreadable_h5ad: name the problem instead of raising from h5py
# --------------------------------------------------------------------------


def test_truncated_file_is_reported_as_truncated(tmp_path):
    good = tmp_path / "good.h5ad"
    write_h5ad_atomically(_adata(), good)
    truncated = tmp_path / "truncated.h5ad"
    truncated.write_bytes(good.read_bytes()[:2048])
    message = describe_unreadable_h5ad(truncated)
    assert message is not None
    assert "truncated or corrupted" in message
    assert "re-run step 1" in message.lower()


def test_empty_file_is_reported_as_empty(tmp_path):
    empty = tmp_path / "empty.h5ad"
    empty.write_bytes(b"")
    message = describe_unreadable_h5ad(empty)
    assert message is not None
    assert "empty" in message


def test_non_hdf5_file_mentions_a_cloud_placeholder(tmp_path):
    # A cloud-storage stub that has not downloaded yet is a plain text file.
    stub = tmp_path / "stub.h5ad"
    stub.write_text("placeholder, not yet downloaded")
    message = describe_unreadable_h5ad(stub)
    assert message is not None
    assert "not an HDF5 file" in message


def test_missing_file_is_reported_as_missing(tmp_path):
    message = describe_unreadable_h5ad(tmp_path / "nope.h5ad")
    assert message is not None
    assert "does not exist" in message


def test_a_healthy_file_reports_nothing(tmp_path):
    target = tmp_path / "out.h5ad"
    write_h5ad_atomically(_adata(), target)
    assert describe_unreadable_h5ad(target) is None
