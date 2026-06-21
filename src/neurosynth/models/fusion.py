# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""CrossAttentionFusion — learns which modalities to trust per sample.

Replaces the v4 hardcoded modality weights (tabular 40%, GNN 20%, etc.)
with a learned 2-head cross-attention layer trained end-to-end on the
validation fold. Optuna tunes the attention hyperparameters separately
(see scripts/tune_fusion_weights.py).

Usage:
    fusion = CrossAttentionFusion(n_modalities=5)
    fused_prob, attn_weights = fusion(tabular_prob, genomic_prob, tft_prob, causal_prob, gnn_prob)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:
    class CrossAttentionFusion(nn.Module):
        """2-head cross-attention over modality probability tokens.

        Each scalar modality probability is projected to a D-dim token,
        then cross-attended so the model learns which modalities to weight
        per sample (e.g. trust genomic more when APOE4_dosage is high).
        """

        def __init__(
            self,
            n_modalities: int = 5,
            embed_dim: int = 64,
            n_heads: int = 2,
        ) -> None:
            super().__init__()
            self.n_modalities = n_modalities
            self.embed = nn.Linear(1, embed_dim)
            self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
            self.norm = nn.LayerNorm(embed_dim)
            self.out = nn.Linear(embed_dim, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(
            self, *probs: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor"]:
            """
            Args:
                *probs: n_modalities scalar tensors of shape (B,) or (B, 1)
            Returns:
                fused_prob: (B,) fused probability
                attn_weights: (B, n_modalities) attention weight per modality
            """
            # Stack → (B, M, 1) then embed → (B, M, D)
            x = torch.stack([p.unsqueeze(-1) if p.dim() == 1 else p for p in probs], dim=1)
            x = self.embed(x)                          # (B, M, D)
            attn_out, attn_w = self.attn(x, x, x)     # cross-attend
            attn_out = self.norm(attn_out + x)         # residual
            fused = self.out(attn_out.mean(dim=1))     # (B, 1)
            fused_prob = self.sigmoid(fused).squeeze(-1)  # (B,)
            return fused_prob, attn_w.mean(dim=1)      # average over heads


class FusionWeights:
    """Loads and applies Optuna-tuned modality fusion weights.

    Falls back to the v4 hardcoded weights if no tuned weights are found.
    Compatible with both torch CrossAttentionFusion and simple weighted sum.
    """

    V4_DEFAULTS = {
        "tabular":  0.40,
        "genomic":  0.15,
        "tft":      0.15,
        "causal":   0.10,
        "gnn":      0.20,
    }

    def __init__(self, weights_path: str | Path | None = None) -> None:
        self.weights: dict[str, float] = dict(self.V4_DEFAULTS)
        self._fusion_model: Any = None

        if weights_path is not None:
            p = Path(weights_path)
            if p.exists():
                try:
                    loaded = json.loads(p.read_text())
                    self.weights.update(loaded.get("weights", {}))
                except Exception:
                    pass

    def weighted_sum(self, modality_probs: dict[str, float]) -> float:
        """Simple weighted sum — used when CrossAttentionFusion is not loaded."""
        total_w = sum(self.weights.get(k, 0.0) for k in modality_probs)
        if total_w == 0:
            return float(np.mean(list(modality_probs.values())))
        return float(sum(
            v * self.weights.get(k, 0.0) / total_w
            for k, v in modality_probs.items()
        ))

    def fuse(self, modality_probs: dict[str, float]) -> dict[str, Any]:
        """Fuse modality probabilities, returning final prob + per-modality weights used."""
        if _TORCH_AVAILABLE and self._fusion_model is not None:
            import torch as _t
            probs = [_t.tensor([v], dtype=_t.float32) for v in modality_probs.values()]
            with _t.no_grad():
                fused, attn_w = self._fusion_model(*probs)
            return {
                "fused_probability": float(fused[0].item()),
                "modality_weights": {
                    k: float(w) for k, w in zip(modality_probs.keys(), attn_w[0].tolist())
                },
                "method": "cross_attention",
            }
        # Fallback to weighted sum
        prob = self.weighted_sum(modality_probs)
        return {
            "fused_probability": prob,
            "modality_weights": {k: self.weights.get(k, 0.0) for k in modality_probs},
            "method": "weighted_sum",
        }

    def load_cross_attention(self, checkpoint_path: str | Path) -> bool:
        """Load a trained CrossAttentionFusion checkpoint."""
        if not _TORCH_AVAILABLE:
            return False
        p = Path(checkpoint_path)
        if not p.exists():
            return False
        try:
            import torch as _t
            n_mods = len(self.weights)
            model = CrossAttentionFusion(n_modalities=n_mods)
            model.load_state_dict(_t.load(p, map_location="cpu"))
            model.eval()
            self._fusion_model = model
            return True
        except Exception:
            return False
