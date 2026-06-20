from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class QCReport:
    input_vcf: Path
    output_prefix: Path
    reference_genome: str
    n_input_samples: int = 0
    n_post_qc_samples: int = 0
    n_input_variants: int = 0
    n_post_qc_variants: int = 0
    pca_covariates_path: Path | None = None
    vep_tsv_path: Path | None = None
    prs_path: Path | None = None
    qc_commands: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class VariantFeatureMatrix:
    patient_id: str
    matrix: pd.DataFrame


@dataclass
class GeneAttentionOutput:
    gene_symbols: list[str]
    gene_attention: pd.DataFrame


@dataclass
class VariantImportanceOutput:
    variant_ids: list[str]
    importance_scores: list[float]
