# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


def _act(name: str) -> nn.Module:
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "relu":
        return nn.ReLU()
    return nn.ELU()


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], output_dim: int, activation: str = "gelu", use_batch_norm: bool = True) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(_act(activation))
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralCausalDiscovery(nn.Module):
    def __init__(self, n_vars: int = 28, hidden_dims: list[int] | None = None, activation: str = "gelu") -> None:
        super().__init__()
        self.n_vars = n_vars
        hidden_dims = hidden_dims or [64, 64]

        self.W_logits = nn.Parameter(torch.zeros(n_vars, n_vars))
        nn.init.uniform_(self.W_logits, -0.01, 0.01)

        self.mechanisms = nn.ModuleList(
            [
                MLP(input_dim=n_vars, hidden_dims=hidden_dims, output_dim=1, activation=activation, use_batch_norm=True)
                for _ in range(n_vars)
            ]
        )
        self.log_noise_vars = nn.Parameter(torch.zeros(n_vars))

    def get_adjacency_matrix(self) -> torch.Tensor:
        W = torch.sigmoid(self.W_logits)
        W = W * (1 - torch.eye(self.n_vars, device=W.device, dtype=W.dtype))
        return W

    def acyclicity_constraint(self, W: torch.Tensor) -> torch.Tensor:
        # Use float64 for numerical stability at large rho values.
        M = (W * W).to(torch.float64)
        E = torch.eye(self.n_vars, device=W.device, dtype=torch.float64)
        power = E.clone()
        exp_M = E.clone()
        for k in range(1, 10):
            power = power @ M / k
            exp_M = exp_M + power
        return torch.trace(exp_M) - self.n_vars

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        W = self.get_adjacency_matrix()
        X_hat_list = []
        for i, mechanism in enumerate(self.mechanisms):
            parent_mask = W[i].unsqueeze(0)
            X_masked = X * parent_mask
            X_hat_i = mechanism(X_masked)
            X_hat_list.append(X_hat_i)
        X_hat = torch.cat(X_hat_list, dim=1)
        return X_hat, W

    def compute_loss(self, X: torch.Tensor, lambda1: float = 0.01, lambda2: float = 10.0, rho: float = 1.0, alpha: float = 0.0) -> dict[str, torch.Tensor]:
        X_hat, W = self.forward(X)
        noise_vars = torch.exp(self.log_noise_vars).clamp(min=1e-6, max=1e6)

        nll = 0.5 * ((X - X_hat) ** 2 / noise_vars + torch.log(noise_vars)).mean()
        h = self.acyclicity_constraint(W)
        dag_penalty = rho / 2.0 * h**2 + alpha * h
        l1_penalty = lambda1 * W.abs().sum()
        var_reg = lambda2 * (noise_vars - 1.0).abs().mean()

        total = nll + dag_penalty + l1_penalty + var_reg
        return {"total": total, "nll": nll, "h": h, "l1": l1_penalty}
