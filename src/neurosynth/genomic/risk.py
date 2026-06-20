from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

@dataclass
class VariantRiskScorer:
    meta_model: LogisticRegression | None = None

    def compute_cadd_burden(self, patient_variant_df: pd.DataFrame, gene: str) -> float:
        r"""Compute CADD-weighted rare burden.

        $$B_g = \sum_{i \in g} CADD_i \cdot \min(1, -\log_{10}(AF_i + 10^{-6}))$$
        """
        df = patient_variant_df.copy()
        df = df[df["gene_symbol"] == gene]
        df = df[df["impact_score"].isin([1, 2])]
        df = df[df["gnomad_af"] < 0.01]
        if df.empty:
            return 0.0

        af_weight = np.minimum(1.0, -np.log10(df["gnomad_af"].to_numpy() + 1e-6))
        burden = float(np.sum(df["cadd_phred"].to_numpy() * af_weight))
        return burden

    def compute_pathway_burden(self, patient_variant_df: pd.DataFrame, pathway_gene_set: set[str]) -> float:
        burdens = [self.compute_cadd_burden(patient_variant_df, gene) for gene in pathway_gene_set]
        return float(np.mean(burdens)) if burdens else 0.0

    def fit_meta_score(self, train_df: pd.DataFrame, label_col: str = "case_label") -> None:
        x = train_df[["prs_score", "rare_burden"]].to_numpy()
        y = train_df[label_col].to_numpy()
        self.meta_model = LogisticRegression(max_iter=1000)
        self.meta_model.fit(x, y)

    def integrate_common_rare(self, prs_score: float, rare_burden_dict: dict[str, float]) -> dict[str, float]:
        rare_total = float(np.mean(list(rare_burden_dict.values()))) if rare_burden_dict else 0.0

        if self.meta_model is None:
            # Fallback linear blend when no trained meta-model is available.
            score = 0.7 * prs_score + 0.3 * np.tanh(rare_total)
            prob = 1.0 / (1.0 + np.exp(-score))
        else:
            prob = float(self.meta_model.predict_proba([[prs_score, rare_total]])[0, 1])
            score = float(np.log(prob / max(1e-6, 1 - prob)))

        return {
            "meta_score": score,
            "risk_probability": float(prob),
            "prs_component": float(prs_score),
            "rare_component": rare_total,
        }
