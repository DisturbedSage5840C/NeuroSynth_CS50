from __future__ import annotations

import torch

from neurosynth.genomic.losses import WeightedMultiTaskLoss


def test_weighted_multitask_loss() -> None:
    loss_fn = WeightedMultiTaskLoss()
    outputs = {
        "diagnosis_logits": torch.randn(4, 3),
        "prs_pred": torch.randn(4, 3),
        "apoe_logits": torch.randn(4, 3),
        "dirichlet_alpha": torch.rand(4, 3) + 1,
    }
    labels = {
        "diagnosis_class": torch.tensor([0, 1, 2, 1]),
        "prs": torch.randn(4, 3),
        "apoe_count": torch.tensor([0, 1, 2, 1]),
    }
    loss, parts = loss_fn(outputs, labels)
    assert loss.item() > 0
    assert "clinical_loss" in parts
