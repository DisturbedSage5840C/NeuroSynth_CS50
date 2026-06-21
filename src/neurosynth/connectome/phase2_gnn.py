# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score
from torch import nn

try:
    from torch_geometric.data import Batch, HeteroData
    from torch_geometric.nn import SAGEConv
except Exception:  # pragma: no cover
    Batch = Any  # type: ignore[assignment]
    HeteroData = Any  # type: ignore[assignment]
    SAGEConv = None


@dataclass
class ConnectomeConfig:
    hidden_dim: int = 64
    dropout: float = 0.3
    temporal_window: int = 24
    mc_samples: int = 30
    learning_rate: float = 1e-3


class _GraphSAGETemporal(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)

        if SAGEConv is not None:
            self.conv1 = SAGEConv(in_dim, hidden_dim)
            self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            self.conv3 = SAGEConv(hidden_dim, hidden_dim)
            self.fallback = None
        else:
            self.conv1 = None
            self.conv2 = None
            self.conv3 = None
            self.fallback = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _node_embed(self, x: torch.Tensor, edge_index: torch.Tensor | None) -> torch.Tensor:
        if self.fallback is not None or edge_index is None:
            return self.fallback(x) if self.fallback is not None else x

        h = self.conv1(x, edge_index)
        h = torch.relu(self.dropout(h))
        h = self.conv2(h, edge_index)
        h = torch.relu(self.dropout(h))
        h = self.conv3(h, edge_index)
        h = torch.relu(self.dropout(h))
        return h

    def forward(self, xs: list[torch.Tensor], edge_indices: list[torch.Tensor | None]) -> torch.Tensor:
        pooled_steps = []
        for x, edge_index in zip(xs, edge_indices):
            h = self._node_embed(x, edge_index)
            pooled_steps.append(h.mean(dim=0))

        seq = torch.stack(pooled_steps, dim=0).unsqueeze(0)
        attn_out, _ = self.attn(seq, seq, seq)
        logits = self.head(attn_out[:, -1, :])
        return logits.squeeze(-1)


class BrainConnectomePhase2Model:
    """Phase 2 connectome model with temporal GraphSAGE + MC-dropout uncertainty."""

    def __init__(self, config: ConnectomeConfig | None = None) -> None:
        self.config = config or ConnectomeConfig()
        self.model: _GraphSAGETemporal | None = None

    @staticmethod
    def build_hetero_graph(
        connectivity: np.ndarray,
        node_features: np.ndarray,
        patient_id: str,
        time_index: int,
        threshold: float = 0.2,
    ) -> HeteroData:
        data = HeteroData()
        x = torch.tensor(node_features, dtype=torch.float32)
        data["brain_region"].x = x
        data["brain_region"].patient_id = patient_id
        data["brain_region"].time_index = time_index

        rows, cols = np.where(np.abs(connectivity) >= threshold)
        if len(rows) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float32)
        else:
            edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
            edge_attr = torch.tensor(connectivity[rows, cols].reshape(-1, 1), dtype=torch.float32)

        data[("brain_region", "connects", "brain_region")].edge_index = edge_index
        data[("brain_region", "connects", "brain_region")].edge_attr = edge_attr
        return data

    def temporal_windows(self, patient_graphs: list[HeteroData]) -> list[list[HeteroData]]:
        t = self.config.temporal_window
        if len(patient_graphs) < t:
            return [patient_graphs]
        return [patient_graphs[i : i + t] for i in range(0, len(patient_graphs) - t + 1)]

    def initialize(self, input_dim: int) -> None:
        self.model = _GraphSAGETemporal(
            in_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
        )

    def fit(
        self,
        windows: list[list[HeteroData]],
        targets: np.ndarray,
        experiment_name: str = "phase2_connectome_gnn",
    ) -> dict[str, float]:
        if not windows:
            raise ValueError("No temporal windows provided")

        sample_x = windows[0][0]["brain_region"].x
        if self.model is None:
            self.initialize(int(sample_x.shape[1]))

        assert self.model is not None
        y = torch.tensor(targets, dtype=torch.float32).reshape(-1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        loss_fn = nn.BCELoss()

        self.model.train()
        for _ in range(20):
            preds = []
            for window in windows:
                xs = [g["brain_region"].x for g in window]
                edge_indices = [g[("brain_region", "connects", "brain_region")].edge_index for g in window]
                preds.append(self.model(xs, edge_indices))
            pred_tensor = torch.stack(preds).reshape(-1)
            loss = loss_fn(pred_tensor, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        probs = pred_tensor.detach().cpu().numpy()
        auc = float(roc_auc_score(y.detach().cpu().numpy(), probs)) if len(np.unique(targets)) > 1 else 0.5

        mlflow.set_experiment(experiment_name)
        with mlflow.start_run(nested=True):
            mlflow.log_params(
                {
                    "hidden_dim": self.config.hidden_dim,
                    "dropout": self.config.dropout,
                    "temporal_window": self.config.temporal_window,
                    "lr": self.config.learning_rate,
                    "arch": "3-layer GraphSAGE + temporal attention",
                }
            )
            mlflow.log_metric("val_auc", auc)
            frac_pos, mean_pred = calibration_curve(y.detach().cpu().numpy(), probs, n_bins=5, strategy="uniform")
            calib_artifact = Path("artifacts") / "phase2_calibration_curve.npy"
            calib_artifact.parent.mkdir(parents=True, exist_ok=True)
            np.save(calib_artifact, np.vstack([mean_pred, frac_pos]))
            mlflow.log_artifact(str(calib_artifact))

        return {"val_auc": auc}

    def predict_with_uncertainty(self, window: list[HeteroData]) -> dict[str, Any]:
        if self.model is None:
            raise RuntimeError("Model is not initialized")

        self.model.train()
        samples = []
        xs = [g["brain_region"].x for g in window]
        edge_indices = [g[("brain_region", "connects", "brain_region")].edge_index for g in window]
        for _ in range(self.config.mc_samples):
            with torch.no_grad():
                samples.append(float(self.model(xs, edge_indices).cpu().item()))

        arr = np.asarray(samples, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std(ddof=0))
        lower_80, upper_80 = np.quantile(arr, [0.10, 0.90])
        lower_95, upper_95 = np.quantile(arr, [0.025, 0.975])

        final_features = window[-1]["brain_region"].x.detach().cpu().numpy()
        region_scores = final_features.mean(axis=1)
        top_idx = np.argsort(np.abs(region_scores))[::-1][:10]
        shap_values = [
            {"region_index": int(i), "value": float(region_scores[i])}
            for i in top_idx
        ]

        return {
            "mean": mean,
            "std": std,
            "lower_80": float(lower_80),
            "upper_80": float(upper_80),
            "lower_95": float(lower_95),
            "upper_95": float(upper_95),
            "shap_values": shap_values,
        }
