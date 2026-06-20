from __future__ import annotations

import numpy as np
import torch

from neurosynth.causal.model import NeuralCausalDiscovery
from neurosynth.causal.trainer import NotearsTrainer


def test_notears_compute_loss() -> None:
    m = NeuralCausalDiscovery(n_vars=6, hidden_dims=[16, 16])
    x = torch.randn(32, 6)
    ld = m.compute_loss(x)
    assert "total" in ld and torch.isfinite(ld["total"]).item()


def test_notears_train_outputs_dag() -> None:
    x = torch.tensor(np.random.RandomState(42).randn(128, 6).astype(np.float32))
    m = NeuralCausalDiscovery(n_vars=6, hidden_dims=[16, 16])
    t = NotearsTrainer(model=m, device=torch.device("cpu"))
    res = t.train(x, max_outer_iters=2, max_inner_iters=20)
    assert res.W_continuous.shape == (6, 6)
    assert res.W_binary.shape == (6, 6)
