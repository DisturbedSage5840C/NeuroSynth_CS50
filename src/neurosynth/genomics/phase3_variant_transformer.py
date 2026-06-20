from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from opacus import PrivacyEngine
from torch import nn

try:
    import allel
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
except Exception:  # pragma: no cover
    allel = None


@dataclass
class Phase3Config:
    embed_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.2
    mc_samples: int = 30


class _HierarchicalTransformer(nn.Module):
    def __init__(self, in_dim: int, cfg: Phase3Config) -> None:
        super().__init__()
        self.in_proj = nn.Linear(in_dim, cfg.embed_dim)
        self.variant_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.n_layers,
        )
        self.gene_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=max(1, cfg.n_layers - 1),
        )
        self.pathway_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.embed_dim,
                nhead=cfg.n_heads,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=1,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.risk_head = nn.Sequential(nn.Linear(cfg.embed_dim, 1), nn.Sigmoid())
        self.pathogenic_head = nn.Linear(cfg.embed_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.in_proj(x)
        z = self.variant_encoder(z)

        gene_tokens = z.mean(dim=1, keepdim=True)
        gene_tokens = self.gene_encoder(gene_tokens)

        pathway_tokens = self.pathway_encoder(gene_tokens)
        pooled = self.dropout(pathway_tokens[:, 0, :])

        risk = self.risk_head(pooled).squeeze(-1)
        pathogenic = self.pathogenic_head(pooled).squeeze(-1)
        return risk, pathogenic, pooled


class GenomicPhase3Model:
    """Phase 3 hierarchical genomic variant transformer with DP-SGD option."""

    def __init__(self, config: Phase3Config | None = None) -> None:
        self.config = config or Phase3Config()
        self.model: _HierarchicalTransformer | None = None
        self.privacy_engine: PrivacyEngine | None = None

    def load_variant_feature_matrix(self, vcf_path: str) -> np.ndarray:
        if allel is None:
            raise ImportError("scikit-allel is required to parse VCF into variant feature matrix")

        callset = allel.read_vcf(vcf_path, fields=["variants/CHROM", "variants/POS", "variants/QUAL"], alt_number=1)
        chrom = np.asarray(callset["variants/CHROM"])
        pos = np.asarray(callset["variants/POS"], dtype=float)
        qual = np.asarray(callset.get("variants/QUAL", np.zeros_like(pos)), dtype=float)

        # Temporary stand-in for MAF/CADD/pathway membership until full annotation sources are plugged in.
        maf = np.clip((pos % 100) / 100.0, 0.0, 1.0)
        cadd = np.clip(qual / (qual.max() + 1e-6), 0.0, 1.0)
        pathway_membership = (np.char.str_len(chrom.astype(str)) % 3).astype(float)

        features = np.stack([maf, cadd, pathway_membership], axis=1)
        return features

    def initialize(self, in_dim: int) -> None:
        self.model = _HierarchicalTransformer(in_dim=in_dim, cfg=self.config)

    def fit(
        self,
        variant_features: np.ndarray,
        risk_target: np.ndarray,
        pathogenic_target: np.ndarray,
        use_dp: bool = True,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> dict[str, float]:
        if self.model is None:
            self.initialize(variant_features.shape[-1])
        assert self.model is not None

        x = torch.tensor(variant_features[None, :, :], dtype=torch.float32)
        y_risk = torch.tensor(risk_target.reshape(-1), dtype=torch.float32)
        y_path = torch.tensor(pathogenic_target.reshape(-1), dtype=torch.float32)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        if use_dp:
            self.privacy_engine = PrivacyEngine()
            # Opacus typically expects DataLoader; this call keeps API-ready compatibility.
            self.model, optimizer, _ = self.privacy_engine.make_private_with_epsilon(
                module=self.model,
                optimizer=optimizer,
                data_loader=[(x, y_risk)],
                epochs=5,
                target_epsilon=8.0,
                target_delta=1e-5,
                max_grad_norm=max_grad_norm,
            )

        bce = nn.BCELoss()
        bce_logits = nn.BCEWithLogitsLoss()

        self.model.train()
        for _ in range(5):
            pred_risk, pred_path, _ = self.model(x)
            loss = bce(pred_risk, y_risk) + bce_logits(pred_path, y_path)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return {"loss": float(loss.detach().cpu().item()), "dp_enabled": float(use_dp)}

    def predict_with_uncertainty(self, x: np.ndarray) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        self.model.train()
        tensor_x = torch.tensor(x[None, :, :], dtype=torch.float32)
        samples = []
        emb_samples = []
        for _ in range(self.config.mc_samples):
            with torch.no_grad():
                risk, _, emb = self.model(tensor_x)
                samples.append(float(risk.cpu().item()))
                emb_samples.append(emb.cpu().numpy().squeeze(0))

        arr = np.asarray(samples)
        lower_80, upper_80 = np.quantile(arr, [0.10, 0.90])
        lower_95, upper_95 = np.quantile(arr, [0.025, 0.975])

        mean_emb = np.mean(np.asarray(emb_samples), axis=0)
        top_idx = np.argsort(np.abs(mean_emb))[::-1][:10]
        shap_values = [{"feature": int(i), "value": float(mean_emb[i])} for i in top_idx]

        return {
            "mean": float(arr.mean()),
            "lower_80": float(lower_80),
            "upper_80": float(upper_80),
            "lower_95": float(lower_95),
            "upper_95": float(upper_95),
            "shap_values": shap_values,
        }
