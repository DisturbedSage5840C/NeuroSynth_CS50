from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced neurological disease classification.

    γ=2 down-weights well-classified examples so the model focuses on rare
    disease cases (ALS, Huntington's) rather than the easy majority class.
    Replaces plain cross-entropy in the clinical diagnosis head.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # Gather the log-probability and probability of the correct class
        log_p = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal_weight = (1.0 - p) ** self.gamma
        loss = -focal_weight * log_p
        if self.weight is not None:
            class_w = self.weight.gather(0, targets)
            loss = loss * class_w
        return loss.mean()


class WeightedMultiTaskLoss(nn.Module):
    def __init__(
        self,
        w_clinical: float = 1.0,
        w_prs: float = 0.3,
        w_apoe: float = 0.5,
        w_dirichlet_kl: float = 0.1,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.w_clinical = w_clinical
        self.w_prs = w_prs
        self.w_apoe = w_apoe
        self.w_dirichlet_kl = w_dirichlet_kl
        self._focal = FocalLoss(gamma=focal_gamma)

    def _dirichlet_kl(self, alpha: torch.Tensor) -> torch.Tensor:
        k = alpha.shape[-1]
        prior = torch.ones_like(alpha)
        sum_alpha = alpha.sum(dim=-1, keepdim=True)
        sum_prior = prior.sum(dim=-1, keepdim=True)
        term1 = torch.lgamma(sum_alpha) - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
        term2 = torch.lgamma(prior).sum(dim=-1, keepdim=True) - torch.lgamma(sum_prior)
        term3 = ((alpha - prior) * (torch.digamma(alpha) - torch.digamma(sum_alpha))).sum(dim=-1, keepdim=True)
        _ = k
        return (term1 + term2 + term3).mean()

    def forward(self, outputs: dict[str, torch.Tensor], labels: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        clinical = self._focal(outputs["diagnosis_logits"], labels["diagnosis_class"])
        prs = F.mse_loss(outputs["prs_pred"], labels["prs"])
        apoe = F.cross_entropy(outputs["apoe_logits"], labels["apoe_count"])
        kl = self._dirichlet_kl(outputs["dirichlet_alpha"])

        total = self.w_clinical * clinical + self.w_prs * prs + self.w_apoe * apoe + self.w_dirichlet_kl * kl
        return total, {
            "clinical_loss": float(clinical.detach()),
            "prs_loss": float(prs.detach()),
            "apoe_loss": float(apoe.detach()),
            "dirichlet_kl": float(kl.detach()),
        }
