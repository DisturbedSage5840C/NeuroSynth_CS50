from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from neurosynth.genomic.dataset import GenomicDataset, genomic_collate_fn


def test_genomic_collate_dynamic_padding(tmp_path: Path) -> None:
    h5_path = tmp_path / "genomic_cache.h5"
    with h5py.File(h5_path, "w") as h5:
        for i, n in enumerate([10, 20]):
            g = h5.create_group(f"p{i}")
            g.create_dataset("variant_features", data=np.random.randn(n, 16).astype(np.float32))
            g.create_dataset("sequence_context", data=np.random.randn(n, 256).astype(np.float32))
            g.create_dataset("gene_ids", data=np.random.randint(0, 5, size=(n,), dtype=np.int64))
            g.create_dataset("consequence_category", data=np.random.randint(0, 10, size=(n,), dtype=np.int64))
            g.create_dataset("gene_symbols", data=np.array([b"APOE"] * n, dtype="S8"))
            labels = g.create_group("labels")
            labels.create_dataset("prs", data=np.array([0.1, 0.2, -0.1], dtype=np.float32))
            labels.attrs["apoe_count"] = 1
            labels.attrs["diagnosis_class"] = 2
            labels.attrs["cdrsb_at_next_visit"] = 3.5
            labels.attrs["pathogenicity_class"] = 1

    ds = GenomicDataset(hdf5_path=h5_path, patient_ids=["p0", "p1"])
    batch = genomic_collate_fn([ds[0], ds[1]])
    assert batch["variant_features"].shape == (2, 20, 16)
    assert batch["sequence_context"].shape == (2, 20, 256)
    assert batch["variant_mask"].sum().item() == 30
