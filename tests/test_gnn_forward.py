from __future__ import annotations

import torch
from torch_geometric.data import Data

from neurosynth.connectome.model import BrainConnectomeGNN


def _make_graph(n_nodes: int = 116) -> Data:
    x = torch.randn(n_nodes, 128)
    edge_src = torch.arange(0, n_nodes - 1)
    edge_dst = torch.arange(1, n_nodes)
    edge_index = torch.stack([torch.cat([edge_src, edge_dst]), torch.cat([edge_dst, edge_src])], dim=0)
    edge_attr = torch.randn(edge_index.shape[1], 1)
    batch = torch.zeros(n_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)


def test_gnn_forward_shapes() -> None:
    model = BrainConnectomeGNN()
    seq = [_make_graph(), _make_graph()]
    deltas = torch.tensor([[0.0, 6.0]], dtype=torch.float32)
    mask = torch.tensor([[True, True]])

    out = model(seq, deltas, mask)
    assert out["embedding"].shape == (1, 256)
    assert out["logits"].shape == (1, 3)
    assert out["cdrsb"].shape == (1, 1)
    assert out["uncertainty"].shape == (1, 1)
