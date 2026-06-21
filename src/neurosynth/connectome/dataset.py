# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.data import Batch, Data, InMemoryDataset

from neurosynth.connectome.builder import ConnectomeBuilder
from neurosynth.connectome.types import BatchSequence, PatientSequence
from neurosynth.connectome.utils import month_delta


class TemporalBrainDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        cohort: str,
        min_timepoints: int = 2,
        max_timepoints: int = 6,
        transform=None,
        pre_transform=None,
    ) -> None:
        self.cohort = cohort
        self.min_timepoints = min_timepoints
        self.max_timepoints = max_timepoints
        self.builder = ConnectomeBuilder()
        self._sequences: list[PatientSequence] = []
        super().__init__(root, transform, pre_transform)
        self._sequences = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> list[str]:
        return [f"{self.cohort}_manifest.csv"]

    @property
    def processed_file_names(self) -> list[str]:
        return [f"{self.cohort.lower()}_temporal_sequences.pt"]

    def process(self) -> None:
        manifest = Path(self.raw_paths[0])
        frame = pd.read_csv(manifest)
        frame = frame.sort_values(["patient_id", "visit_date"]).reset_index(drop=True)

        grouped: dict[str, list[dict]] = defaultdict(list)
        for _, row in frame.iterrows():
            grouped[str(row["patient_id"])].append(row.to_dict())

        sequences: list[PatientSequence] = []
        for patient_id, visits in grouped.items():
            graphs: list[Data] = []
            for visit in visits[: self.max_timepoints]:
                try:
                    graph = self.builder.build_connectivity_graph(
                        fmri_path=Path(visit["fmri_path"]),
                        t1_path=Path(visit["t1_path"]),
                        pet_path=Path(visit["pet_path"]) if pd.notna(visit.get("pet_path")) else None,
                        y_class=int(visit["label_stage"]),
                        y_regression=float(visit["label_cdrsb"]),
                        patient_id=patient_id,
                        scan_date=str(visit.get("visit_date", "")),
                        site_id=str(visit["site_id"]),
                    )
                    graphs.append(graph)
                except Exception:
                    continue

            if len(graphs) >= self.min_timepoints:
                sequences.append(
                    PatientSequence(
                        patient_id=patient_id,
                        graphs=graphs,
                        label_stage=int(visits[-1]["label_stage"]),
                        label_cdrsb=float(visits[-1]["label_cdrsb"]),
                        site_id=str(visits[-1]["site_id"]),
                    )
                )

        torch.save(sequences, self.processed_paths[0])

    def len(self) -> int:
        return len(self._sequences)

    def get(self, idx: int) -> PatientSequence:
        return self._sequences[idx]


def collate_temporal_batch(batch: list[PatientSequence]) -> BatchSequence:
    max_len = max(len(item.graphs) for item in batch)
    bsz = len(batch)

    graph_batches: list[Data | None] = []
    mask = torch.zeros((bsz, max_len), dtype=torch.bool)
    deltas = torch.zeros((bsz, max_len), dtype=torch.float32)
    y_class = torch.tensor([item.label_stage for item in batch], dtype=torch.long)
    y_reg = torch.tensor([item.label_cdrsb for item in batch], dtype=torch.float32)

    for t in range(max_len):
        to_batch: list[Data] = []
        for i, item in enumerate(batch):
            if t < len(item.graphs):
                g = item.graphs[t]
                to_batch.append(g)
                mask[i, t] = True
                if t > 0:
                    d_days = max(1, t * 30)
                    deltas[i, t] = month_delta(d_days)
        graph_batches.append(Batch.from_data_list(to_batch) if to_batch else None)

    return BatchSequence(
        graph_batches=graph_batches,
        padding_mask=mask,
        time_deltas_months=deltas,
        y_class=y_class,
        y_regression=y_reg,
        patient_ids=[item.patient_id for item in batch],
    )


class TemporalLengthBatchSampler:
    def __init__(self, dataset: TemporalBrainDataset, batch_size: int, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        lengths = [(idx, len(self.dataset.get(idx).graphs)) for idx in range(self.dataset.len())]
        lengths.sort(key=lambda x: x[1])
        buckets = [lengths[i : i + self.batch_size] for i in range(0, len(lengths), self.batch_size)]
        if self.shuffle:
            import random

            random.shuffle(buckets)
        for bucket in buckets:
            yield [idx for idx, _ in bucket]

    def __len__(self) -> int:
        return (self.dataset.len() + self.batch_size - 1) // self.batch_size


def make_stratified_group_splits(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[list[str], list[str]]]:
    patient_frame = (
        manifest.sort_values("visit_date")
        .groupby("patient_id")
        .tail(1)
        .reset_index(drop=True)
    )

    y = patient_frame["label_stage"].astype(str) + "::" + patient_frame["site_id"].astype(str)
    groups = patient_frame["patient_id"].astype(str)

    min_class_count = int(pd.Series(y).value_counts().min()) if len(y) else 0
    effective_splits = min(n_splits, len(patient_frame))
    out: list[tuple[list[str], list[str]]] = []

    if min_class_count >= 2 and effective_splits >= 2:
        splitter = StratifiedGroupKFold(n_splits=min(effective_splits, min_class_count), shuffle=True, random_state=random_state)
        for train_idx, val_idx in splitter.split(patient_frame, y, groups=groups):
            train_ids = patient_frame.iloc[train_idx]["patient_id"].astype(str).tolist()
            val_ids = patient_frame.iloc[val_idx]["patient_id"].astype(str).tolist()
            out.append((train_ids, val_ids))
        return out

    if effective_splits < 2:
        return [(groups.tolist(), [])]

    splitter = GroupKFold(n_splits=effective_splits)
    for train_idx, val_idx in splitter.split(patient_frame, groups=groups):
        train_ids = patient_frame.iloc[train_idx]["patient_id"].astype(str).tolist()
        val_ids = patient_frame.iloc[val_idx]["patient_id"].astype(str).tolist()
        out.append((train_ids, val_ids))
    return out
