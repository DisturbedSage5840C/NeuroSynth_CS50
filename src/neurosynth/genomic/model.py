from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class AttentionEncoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.norm1(x)
        attn_out, attn_w = self.attn(y, y, y, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.drop(self.ff(self.norm2(x)))
        return x, attn_w


class HierarchicalVariantTransformer(nn.Module):
    def __init__(self, gene_vocab: dict[str, int], resources_dir: Path | None = None, max_variants_per_gene: int = 500) -> None:
        super().__init__()
        self.gene_vocab = gene_vocab
        self.id_to_gene = {v: k for k, v in gene_vocab.items()}
        self.max_variants_per_gene = max_variants_per_gene

        self.variant_fuse = nn.Sequential(
            nn.Linear(272, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )
        self.variant_type_emb = nn.Embedding(11, 256)

        self.gene_cls_token = nn.Parameter(torch.randn(1, 1, 256) * 0.02)
        self.gene_layers = nn.ModuleList([AttentionEncoderLayer(256, 8, 1024, 0.1) for _ in range(4)])

        self.gene_identity_emb = nn.Embedding(10000, 256)

        self.pathway_layers = nn.ModuleList([AttentionEncoderLayer(256, 8, 1024, 0.15) for _ in range(6)])
        self.disease_queries = nn.Parameter(torch.randn(4, 256) * 0.02)
        self.final_proj = nn.Sequential(
            nn.Linear(4 * 256, 512),
            nn.LayerNorm(512),
        )

        self.apoe_emb = nn.Embedding(3, 32)
        self.apoe_fuse = nn.Linear(544, 512)

        self.prs_regression = nn.Linear(512, 3)
        self.apoe_prediction = nn.Linear(512, 3)
        self.pathogenicity_classification = nn.Linear(512, 4)
        self.diagnosis_head = nn.Linear(512, 3)
        self.dirichlet_head = nn.Sequential(nn.Linear(512, 3), nn.Softplus())

        self._load_gene_sets(resources_dir)

    def _load_gene_sets(self, resources_dir: Path | None) -> None:
        base = resources_dir or (Path(__file__).parent / "resources")
        with (base / "gene_sets.json").open("r", encoding="utf-8") as f:
            payload = json.load(f)

        disease_genes = set(payload["ad_genes"] + payload["pd_genes"] + payload["als_genes"] + payload["shared_neuro"])
        self.register_buffer(
            "disease_gene_ids",
            torch.tensor([self.gene_vocab[g] for g in disease_genes if g in self.gene_vocab], dtype=torch.long),
            persistent=False,
        )

    def _group_by_gene(self, fused: torch.Tensor, gene_ids: torch.Tensor, mask: torch.Tensor) -> tuple[list[torch.Tensor], list[int]]:
        valid = mask.bool()
        ids = gene_ids[valid]
        feats = fused[valid]
        unique = ids.unique(sorted=True)

        groups: list[torch.Tensor] = []
        group_gene_ids: list[int] = []
        for gid in unique.tolist():
            idx = ids == gid
            x = feats[idx][: self.max_variants_per_gene]
            if x.numel() == 0:
                continue
            groups.append(x)
            group_gene_ids.append(gid)
        return groups, group_gene_ids

    def _encode_gene(self, variants: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cls = self.gene_cls_token.expand(1, -1, -1)
        seq = torch.cat([cls, variants.unsqueeze(0)], dim=1)
        key_padding = torch.zeros((1, seq.shape[1]), dtype=torch.bool, device=seq.device)

        attn_last = torch.zeros((1, 8, seq.shape[1], seq.shape[1]), device=seq.device)
        for layer in self.gene_layers:
            seq, attn_last = layer(seq, key_padding_mask=key_padding)
        return seq[:, 0, :].squeeze(0), attn_last.squeeze(0)

    def _disease_filter(self, gene_embs: torch.Tensor, gene_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.disease_gene_ids.numel() == 0:
            return gene_embs, gene_ids
        keep = torch.isin(gene_ids, self.disease_gene_ids.to(gene_ids.device))
        if keep.sum() == 0:
            return gene_embs, gene_ids
        return gene_embs[keep], gene_ids[keep]

    def _pathway_encode(self, gene_embs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        seq = gene_embs.unsqueeze(0)
        attns: list[torch.Tensor] = []
        for layer in self.pathway_layers:
            seq, a = layer(seq)
            attns.append(a)

        q = self.disease_queries.unsqueeze(0)
        attn_scores = torch.matmul(q, seq.transpose(1, 2)) / (256 ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        pooled = torch.matmul(attn_probs, seq)
        out = pooled.reshape(1, -1)
        return out.squeeze(0), attns

    def forward(
        self,
        variant_features: torch.Tensor,
        sequence_context: torch.Tensor,
        gene_ids: torch.Tensor,
        consequence_category: torch.Tensor,
        variant_mask: torch.Tensor,
        apoe_e4_count: torch.Tensor,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        r"""Forward for hierarchical genomic encoding.

        The fused variant representation is

        $$h_v = \mathrm{GELU}(\mathrm{LN}(W[x_v; s_v])) + e_{type(v)}$$
        """

        bsz = variant_features.shape[0]
        patient_embeddings: list[torch.Tensor] = []
        gene_attentions: list[torch.Tensor] = []
        variant_importance: list[torch.Tensor] = []

        fused_all = self.variant_fuse(torch.cat([variant_features, sequence_context], dim=-1))
        fused_all = fused_all + self.variant_type_emb(consequence_category.clamp(0, 10))

        for b in range(bsz):
            groups, group_ids = self._group_by_gene(fused_all[b], gene_ids[b], variant_mask[b])
            if not groups:
                patient_embeddings.append(torch.zeros(512, device=variant_features.device))
                gene_attentions.append(torch.zeros(1, 1, 1, 1, device=variant_features.device))
                variant_importance.append(torch.zeros(1, device=variant_features.device))
                continue

            gene_embs: list[torch.Tensor] = []
            gene_attn: list[torch.Tensor] = []
            var_scores: list[torch.Tensor] = []
            for g in groups:
                emb, att = self._encode_gene(g)
                gene_embs.append(emb)
                gene_attn.append(att)
                # CLS-to-token attention mean over heads as variant importance proxy.
                var_scores.append(att[:, 0, 1:].mean(dim=0))

            g_emb = torch.stack(gene_embs, dim=0)
            g_ids = torch.tensor(group_ids, device=g_emb.device, dtype=torch.long)
            g_emb = g_emb + self.gene_identity_emb(g_ids.clamp(0, self.gene_identity_emb.num_embeddings - 1))
            g_emb, g_ids = self._disease_filter(g_emb, g_ids)

            pooled, _ = self._pathway_encode(g_emb)
            patient = self.final_proj(pooled.unsqueeze(0)).squeeze(0)
            apoe_emb = self.apoe_emb(apoe_e4_count[b].clamp(0, 2))
            patient = self.apoe_fuse(torch.cat([patient, apoe_emb], dim=-1))

            patient_embeddings.append(patient)
            gene_attentions.append(torch.mean(torch.stack([a.mean(dim=(-1, -2)) for a in gene_attn]), dim=0))
            variant_importance.append(torch.cat(var_scores) if var_scores else torch.zeros(1, device=g_emb.device))

        z = torch.stack(patient_embeddings, dim=0)
        out = {
            "embedding": z,
            "prs_pred": self.prs_regression(z),
            "apoe_logits": self.apoe_prediction(z),
            "pathogenicity_logits": self.pathogenicity_classification(z),
            "diagnosis_logits": self.diagnosis_head(z),
            "dirichlet_alpha": self.dirichlet_head(z) + 1.0,
            "gene_attention": gene_attentions,
            "variant_importance": variant_importance,
        }
        return out

    # ------------------------------------------------------------------
    # MC Dropout inference
    # ------------------------------------------------------------------

    def enable_mc_dropout(self) -> None:
        """Set all dropout layers to training mode for MC Dropout inference.

        Call this before running n_passes of forward() at inference time to
        get a distribution over predictions rather than a point estimate.
        All other layers (LayerNorm, Linear) stay in eval mode.
        """
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def disable_mc_dropout(self) -> None:
        """Restore full eval mode (disable MC Dropout)."""
        self.eval()

    @torch.no_grad()
    def predict_mc(self, *args, n_passes: int = 20, **kwargs) -> dict[str, torch.Tensor]:
        """Run n_passes stochastic forward passes and return mean + std.

        Returns a dict with every key from forward() plus:
          - ``<key>_std``: per-output standard deviation across passes
          - ``mc_passes``: number of stochastic passes used

        Usage:
            model.enable_mc_dropout()
            result = model.predict_mc(variant_features, ..., n_passes=20)
            model.disable_mc_dropout()
        """
        results: list[dict] = []
        for _ in range(n_passes):
            results.append(self(*args, **kwargs))

        # Stack scalar/tensor outputs across passes and compute statistics
        keys = [k for k in results[0] if isinstance(results[0][k], torch.Tensor)]
        aggregated: dict[str, torch.Tensor] = {}
        for k in keys:
            stacked = torch.stack([r[k] for r in results], dim=0)  # (passes, ...)
            aggregated[k] = stacked.mean(dim=0)
            aggregated[f"{k}_std"] = stacked.std(dim=0)

        aggregated["mc_passes"] = torch.tensor(n_passes)
        # Keep list outputs (gene_attention, variant_importance) from the last pass
        for k in results[0]:
            if k not in aggregated:
                aggregated[k] = results[-1][k]
        return aggregated
