# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import torch
from torch.utils.data import Dataset


class GenomicDataset(Dataset):
    def __init__(self, hdf5_path: Path, patient_ids: list[str]) -> None:
        self.hdf5_path = hdf5_path
        self.patient_ids = patient_ids

    def __len__(self) -> int:
        return len(self.patient_ids)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        patient_id = self.patient_ids[idx]
        with h5py.File(self.hdf5_path, "r") as h5:
            grp = h5[patient_id]
            variant_features = torch.from_numpy(grp["variant_features"][...]).float()
            seq_context = torch.from_numpy(grp["sequence_context"][...]).float()
            gene_ids = torch.from_numpy(grp["gene_ids"][...]).long()
            consequence = torch.from_numpy(grp["consequence_category"][...]).long()

            modality = grp.attrs.get("modality", "WGS")
            modality_mask = torch.tensor([1, 0] if modality == "WGS" else [0, 1], dtype=torch.float32)

            labels = {
                "prs": torch.tensor(grp["labels/prs"][...], dtype=torch.float32),
                "apoe_count": torch.tensor(int(grp["labels"].attrs.get("apoe_count", 0)), dtype=torch.long),
                "diagnosis_class": torch.tensor(int(grp["labels"].attrs.get("diagnosis_class", 0)), dtype=torch.long),
                "cdrsb_at_next_visit": torch.tensor(float(grp["labels"].attrs.get("cdrsb_at_next_visit", 0.0)), dtype=torch.float32),
                "pathogenicity_class": torch.tensor(int(grp["labels"].attrs.get("pathogenicity_class", 0)), dtype=torch.long),
            }

            meta = {
                "patient_id": patient_id,
                "site_id": grp.attrs.get("site_id", "unknown"),
                "sex": grp.attrs.get("sex", "U"),
                "age": float(grp.attrs.get("age", 0.0)),
            }

        return {
            "variant_features": variant_features,
            "sequence_context": seq_context,
            "gene_ids": gene_ids,
            "consequence_category": consequence,
            "patient_meta": meta,
            "labels": labels,
            "modality_mask": modality_mask,
        }


def genomic_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_v = max(item["variant_features"].shape[0] for item in batch)
    bsz = len(batch)

    vf = torch.zeros((bsz, max_v, 16), dtype=torch.float32)
    sc = torch.zeros((bsz, max_v, 256), dtype=torch.float32)
    gid = torch.zeros((bsz, max_v), dtype=torch.long)
    ccat = torch.zeros((bsz, max_v), dtype=torch.long)
    vmask = torch.zeros((bsz, max_v), dtype=torch.bool)

    prs = torch.zeros((bsz, 3), dtype=torch.float32)
    apoe = torch.zeros((bsz,), dtype=torch.long)
    diag = torch.zeros((bsz,), dtype=torch.long)
    cdrsb = torch.zeros((bsz,), dtype=torch.float32)
    patho = torch.zeros((bsz,), dtype=torch.long)
    modality_mask = torch.zeros((bsz, 2), dtype=torch.float32)

    meta: list[dict[str, Any]] = []

    for i, item in enumerate(batch):
        n = item["variant_features"].shape[0]
        vf[i, :n] = item["variant_features"]
        sc[i, :n] = item["sequence_context"]
        gid[i, :n] = item["gene_ids"]
        ccat[i, :n] = item["consequence_category"]
        vmask[i, :n] = True

        prs[i] = item["labels"]["prs"]
        apoe[i] = item["labels"]["apoe_count"]
        diag[i] = item["labels"]["diagnosis_class"]
        cdrsb[i] = item["labels"]["cdrsb_at_next_visit"]
        patho[i] = item["labels"]["pathogenicity_class"]
        modality_mask[i] = item["modality_mask"]
        meta.append(item["patient_meta"])

    return {
        "variant_features": vf,
        "sequence_context": sc,
        "gene_ids": gid,
        "consequence_category": ccat,
        "variant_mask": vmask,
        "modality_mask": modality_mask,
        "labels": {
            "prs": prs,
            "apoe_count": apoe,
            "diagnosis_class": diag,
            "cdrsb_at_next_visit": cdrsb,
            "pathogenicity_class": patho,
        },
        "patient_meta": meta,
    }
