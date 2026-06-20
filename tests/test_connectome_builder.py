from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import pytest

from neurosynth.connectome.builder import ConnectomeBuilder


def test_build_connectivity_graph_with_synthetic_nifti(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    shape = (8, 8, 8, 40)
    fmri = nib.Nifti1Image(np.random.randn(*shape).astype(np.float32), affine=np.eye(4))
    fmri.header["pixdim"][4] = 2.0

    t1 = nib.Nifti1Image(np.random.randn(8, 8, 8).astype(np.float32), affine=np.eye(4))

    fmri_path = tmp_path / "sub-01_task-rest_bold.nii.gz"
    t1_path = tmp_path / "sub-01_T1w.nii.gz"
    nib.save(fmri, fmri_path)
    nib.save(t1, t1_path)

    atlas_img = nib.Nifti1Image(np.random.randint(0, 117, (8, 8, 8), dtype=np.int16), affine=np.eye(4))
    atlas_path = tmp_path / "atlas.nii.gz"
    nib.save(atlas_img, atlas_path)

    monkeypatch.setattr(
        "neurosynth.connectome.builder.datasets.fetch_atlas_aal",
        lambda version="SPM12": SimpleNamespace(maps=str(atlas_path), labels=[f"ROI_{i:03d}" for i in range(116)]),
    )

    class DummyMasker:
        def __init__(self, **kwargs):
            _ = kwargs

        def fit_transform(self, fmri_file, confounds=None):
            _ = (fmri_file, confounds)
            return np.random.randn(40, 116).astype(np.float32)

    monkeypatch.setattr("neurosynth.connectome.builder.NiftiLabelsMasker", DummyMasker)

    builder = ConnectomeBuilder()
    data = builder.build_connectivity_graph(
        fmri_path=fmri_path,
        t1_path=t1_path,
        patient_id="p1",
        scan_date="2026-01-01",
        site_id="s1",
    )

    assert data.x.shape == (116, 128)
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape[1] == 1
