# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATv2Conv, global_add_pool, global_max_pool, global_mean_pool

try:
    from torchdiffeq import odeint
except Exception:  # pragma: no cover
    odeint = None


@dataclass
class NIGParams:
    gamma: torch.Tensor
    v: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B]
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(0, -9.0, half, device=t.device))
        angle = t[:, None] * freqs[None, :]
        return torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)


class ODEFunc(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, _t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ODELSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.odefunc = ODEFunc(hidden_size)
        self.cell = nn.LSTMCell(input_size, hidden_size)

    def _flow(self, h: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if odeint is None:
            return h + dt[:, None] * self.odefunc(torch.zeros(1, device=h.device), h)

        t0 = torch.zeros(1, device=h.device)
        out = []
        for i in range(h.shape[0]):
            ts = torch.cat([t0, dt[i : i + 1]])
            h_i = odeint(self.odefunc, h[i : i + 1], ts, rtol=1e-3, atol=1e-4, method="dopri5")[-1]
            out.append(h_i)
        return torch.cat(out, dim=0)

    def forward(self, x_seq: torch.Tensor, deltas: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, T, D], deltas/mask: [B, T]
        bsz, t_steps, _ = x_seq.shape
        h = torch.zeros(bsz, self.hidden_size, device=x_seq.device)
        c = torch.zeros_like(h)

        for t in range(t_steps):
            valid = mask[:, t]
            if t > 0:
                h = self._flow(h, deltas[:, t])
            if valid.any():
                h_new, c_new = self.cell(x_seq[:, t], (h, c))
                h = torch.where(valid[:, None], h_new, h)
                c = torch.where(valid[:, None], c_new, c)
        return h


class DirichletHead(nn.Module):
    def __init__(self, in_dim: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, n_classes),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BrainConnectomeGNN(nn.Module):
    def __init__(self, in_dim: int = 128, use_gradient_checkpointing: bool = True) -> None:
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        self.gat1 = GATv2Conv(256, 128, heads=8, edge_dim=1, dropout=0.3, add_self_loops=True, share_weights=False)
        self.bn1 = nn.BatchNorm1d(1024)
        self.gat2 = GATv2Conv(1024, 256, heads=4, edge_dim=1, dropout=0.2, concat=True)
        self.gat3 = GATv2Conv(1024, 256, heads=1, edge_dim=1, dropout=0.1, concat=False)

        self.readout_proj = nn.Sequential(
            nn.Linear(768, 256),
            nn.LayerNorm(256),
        )

        self.time_embedding = SinusoidalTimeEmbedding(dim=32)
        self.temporal = ODELSTM(input_size=288, hidden_size=256)

        self.stage_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 3),
        )
        self.reg_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Softplus(),
        )
        self.dirichlet_head = DirichletHead(256, 3)

        self.nig_head = nn.Linear(256, 4)

    def encode_graph(self, data, return_attention_weights: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_proj(data.x)

        if self.use_gradient_checkpointing and self.training:
            x = checkpoint(lambda z: z, x)

        x1, att1 = self.gat1(x, data.edge_index, data.edge_attr, return_attention_weights=True)
        x1 = F.elu(self.bn1(x1))
        x2, att2 = self.gat2(x1, data.edge_index, data.edge_attr, return_attention_weights=True)
        x3, att3 = self.gat3(x2, data.edge_index, data.edge_attr, return_attention_weights=True)

        g = torch.cat(
            [
                global_mean_pool(x3, data.batch),
                global_max_pool(x3, data.batch),
                global_add_pool(x3, data.batch),
            ],
            dim=-1,
        )
        emb = self.readout_proj(g)

        att = {
            "layer1": att1[1],
            "layer2": att2[1],
            "layer3": att3[1],
        }
        return emb, att

    def _nig_params(self, hidden: torch.Tensor) -> NIGParams:
        out = self.nig_head(hidden)
        gamma = out[:, 0:1]
        v = F.softplus(out[:, 1:2]) + 1e-6
        alpha = F.softplus(out[:, 2:3]) + 1.0
        beta = F.softplus(out[:, 3:4]) + 1e-6
        return NIGParams(gamma=gamma, v=v, alpha=alpha, beta=beta)

    def forward(self, graph_sequence: list, time_deltas_months: torch.Tensor, padding_mask: torch.Tensor) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        bsz, t_steps = padding_mask.shape
        device = time_deltas_months.device

        seq_emb = torch.zeros(bsz, t_steps, 256, device=device)
        all_att: dict[str, torch.Tensor] = {}

        for t, graph_batch in enumerate(graph_sequence):
            if graph_batch is None:
                continue
            graph_batch = graph_batch.to(device)
            emb, att = self.encode_graph(graph_batch, return_attention_weights=True)
            seq_emb[: emb.shape[0], t, :] = emb
            all_att = att

        t_emb = self.time_embedding(time_deltas_months.reshape(-1)).reshape(bsz, t_steps, 32)
        temporal_in = torch.cat([seq_emb, t_emb], dim=-1)

        hidden = self.temporal(temporal_in, time_deltas_months, padding_mask)
        logits = self.stage_head(hidden)
        cdrsb = self.reg_head(hidden)
        uncertainty = self.uncertainty_head(hidden)
        evidence = self.dirichlet_head(hidden)
        nig = self._nig_params(hidden)

        return {
            "embedding": hidden,
            "logits": logits,
            "cdrsb": cdrsb,
            "uncertainty": uncertainty,
            "evidence": evidence,
            "nig_gamma": nig.gamma,
            "nig_v": nig.v,
            "nig_alpha": nig.alpha,
            "nig_beta": nig.beta,
            "attention": all_att,
        }
