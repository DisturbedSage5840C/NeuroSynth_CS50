from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch_geometric.data import Data

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class PatientVisit:
    patient_id: str
    visit_code: str
    visit_date: str
    fmri_path: Path
    t1_path: Path
    pet_path: Path | None
    label_cdrsb: float
    label_stage: int
    site_id: str
    age: float
    sex: str


@dataclass
class PatientSequence:
    patient_id: str
    graphs: list[Data]
    label_stage: int
    label_cdrsb: float
    site_id: str


@dataclass
class BatchSequence:
    graph_batches: list[Data | None]
    padding_mask: torch.Tensor
    time_deltas_months: torch.Tensor
    y_class: torch.Tensor
    y_regression: torch.Tensor
    patient_ids: list[str]


@dataclass
class ExplanationResult:
    patient_id: str
    node_importance: torch.Tensor
    edge_importance: torch.Tensor
    edge_pairs: list[tuple[int, int]] = field(default_factory=list)
    top_10_regions: list[str] = field(default_factory=list)
    top_10_connections: list[str] = field(default_factory=list)
