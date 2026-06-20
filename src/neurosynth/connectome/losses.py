from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class EvidentialClassificationLoss(nn.Module):
    def __init__(self, lambda_reg: float = 0.1) -> None:
        super().__init__()
        self.lambda_reg = lambda_reg

    def forward(self, evidence: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        alpha = evidence + 1.0
        s = alpha.sum(dim=-1, keepdim=True)
        probs = alpha / s

        if target.ndim == 1:
            target_oh = F.one_hot(target, num_classes=evidence.shape[-1]).float()
        else:
            target_oh = target.float()
        cls = torch.sum((target_oh - probs) ** 2 + probs * (1 - probs) / (s + 1.0), dim=-1)

        kl_alpha = (alpha - 1) * (1 - target_oh) + 1
        kl = torch.sum((kl_alpha - 1) * (torch.digamma(kl_alpha) - torch.digamma(kl_alpha.sum(dim=-1, keepdim=True))), dim=-1)
        return (cls + self.lambda_reg * kl).mean()


class NIGLoss(nn.Module):
    def __init__(self, lambda_coef: float = 0.01) -> None:
        super().__init__()
        self.lambda_coef = lambda_coef

    def forward(
        self,
        y: torch.Tensor,
        gamma: torch.Tensor,
        v: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        anneal: float = 1.0,
    ) -> torch.Tensor:
        two_beta_v = 2 * beta * (1 + v)
        nll = 0.5 * torch.log(torch.pi / v) - alpha * torch.log(two_beta_v)
        nll = nll + (alpha + 0.5) * torch.log(v * (y - gamma) ** 2 + two_beta_v)
        nll = nll + torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

        reg = torch.abs(y - gamma) * (2 * v + alpha)
        return (nll + anneal * self.lambda_coef * reg).mean()


class CombinedNeuroLoss(nn.Module):
    def __init__(self, cls_weight: float = 0.6, reg_weight: float = 0.4, lambda_reg: float = 0.1, lambda_nig: float = 0.01) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.cls_loss = EvidentialClassificationLoss(lambda_reg=lambda_reg)
        self.reg_loss = NIGLoss(lambda_coef=lambda_nig)

    def forward(
        self,
        out: dict[str, torch.Tensor],
        y_class: torch.Tensor,
        y_regression: torch.Tensor,
        epoch: int = 0,
        y_class_soft: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        anneal = min(1.0, epoch / 30.0)
        cls_target = y_class_soft if y_class_soft is not None else y_class
        cls = self.cls_loss(out["evidence"], cls_target)
        reg = self.reg_loss(
            y=y_regression[:, None],
            gamma=out["nig_gamma"],
            v=out["nig_v"],
            alpha=out["nig_alpha"],
            beta=out["nig_beta"],
            anneal=anneal,
        )
        total = self.cls_weight * cls + self.reg_weight * reg
        return total, {"cls_loss": float(cls.detach()), "reg_loss": float(reg.detach()), "anneal": float(anneal)}
