from __future__ import annotations

import torch

from neurosynth.connectome.losses import CombinedNeuroLoss


def test_combined_loss_computes() -> None:
    loss_fn = CombinedNeuroLoss()
    out = {
        "evidence": torch.rand(4, 3),
        "nig_gamma": torch.rand(4, 1),
        "nig_v": torch.rand(4, 1) + 1,
        "nig_alpha": torch.rand(4, 1) + 1,
        "nig_beta": torch.rand(4, 1) + 1,
    }
    y_class = torch.tensor([0, 1, 2, 1])
    y_reg = torch.rand(4)

    loss, parts = loss_fn(out, y_class, y_reg, epoch=5)
    assert loss.item() > 0
    assert "cls_loss" in parts
    assert "reg_loss" in parts
