from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import torch

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class CausalInput:
    patient_matrix: np.ndarray
    patient_delta_matrix: np.ndarray
    population_matrix: np.ndarray
    variable_names: list[str]
    modifiability_mask: torch.BoolTensor


@dataclass
class TrainingResult:
    W_continuous: torch.Tensor
    W_binary: torch.Tensor
    causal_graph: nx.DiGraph
    training_history: dict[str, list[float]]
    final_h: float
    final_nll: float


@dataclass
class PatientCausalGraph:
    adjacency: np.ndarray
    dag: nx.DiGraph
    ancestors: dict[str, set[str]]
    descendant_effects: dict[str, float]
    modifiable_high_impact: list[str]
    causal_paths_to_dci: list[tuple[list[str], float]]


@dataclass
class InterventionResult:
    factual_dci_trajectory: np.ndarray
    counterfactual_dci_trajectory: np.ndarray
    dci_difference: np.ndarray
    dci_difference_ci_80: np.ndarray
    affected_variables: dict[str, dict[str, float]]
    interpretation: str


@dataclass
class ValidationReport:
    precision: float
    recall: float
    f1: float
    shd: int
    agreement_edges: list[tuple[str, str]] = field(default_factory=list)
    disagreement_edges: list[tuple[str, str]] = field(default_factory=list)
