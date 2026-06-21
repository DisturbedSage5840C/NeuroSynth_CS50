# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_size: int = 32) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            dropout=0.4,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class TemporalProgressionModel:
    def __init__(self, feature_names: list[str], models_dir: str | Path = "models") -> None:
        self.feature_names = feature_names
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.model = TrajectoryLSTM(input_size=len(feature_names))
        self.mmse_idx = self.feature_names.index("MMSE") if "MMSE" in self.feature_names else 0
        self.func_idx = self.feature_names.index("FunctionalAssessment") if "FunctionalAssessment" in self.feature_names else 0
        self.adl_idx = self.feature_names.index("ADL") if "ADL" in self.feature_names else 0

    @staticmethod
    def _age_bracket(age: float) -> int:
        if age < 60:
            return 0
        if age < 70:
            return 1
        if age < 80:
            return 2
        return 3

    def _build_pseudo_sequences(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        age_idx = self.feature_names.index("Age") if "Age" in self.feature_names else 0
        groups: dict[tuple[int, int], list[tuple[np.ndarray, int]]] = {}

        for row, label in zip(X, y):
            age_bucket = self._age_bracket(float(row[age_idx]))
            mmse_score = float(row[self.mmse_idx])
            mmse_bucket = int(np.clip((30 - mmse_score) // 5, 0, 5))
            key = (age_bucket, mmse_bucket)
            groups.setdefault(key, []).append((row, int(label)))

        sequences: list[np.ndarray] = []
        labels: list[int] = []

        for items in groups.values():
            items = sorted(items, key=lambda v: float(v[0][self.mmse_idx]))
            if len(items) < 4:
                continue
            rows = [v[0] for v in items]
            labs = [v[1] for v in items]
            for i in range(0, len(rows) - 3):
                seq = np.stack(rows[i : i + 4], axis=0)
                label = int(max(labs[i : i + 4]))
                sequences.append(seq)
                labels.append(label)

        if not sequences:
            seqs = np.repeat(X[: min(64, len(X))][:, None, :], repeats=4, axis=1)
            return seqs.astype(np.float32), y[: len(seqs)].astype(np.float32)

        return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.float32)

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 30, lr: float = 1e-3) -> None:
        seqs, labels = self._build_pseudo_sequences(X_train, y_train)

        x_tensor = torch.tensor(seqs, dtype=torch.float32)
        y_tensor = torch.tensor(labels.reshape(-1, 1), dtype=torch.float32)

        optim = torch.optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.BCELoss()

        self.model.train()
        for _ in range(epochs):
            optim.zero_grad()
            probs = self.model(x_tensor)
            loss = loss_fn(probs, y_tensor)
            loss.backward()
            optim.step()

        torch.save(self.model.state_dict(), self.models_dir / "lstm_model.pt")

    def predict_trajectory(self, patient_features: np.ndarray, current_probability: float) -> dict[str, Any]:
        months = np.array([6, 12, 18, 24, 30, 36], dtype=float)
        self.model.eval()
        with torch.no_grad():
            seq = np.repeat(np.asarray(patient_features, dtype=np.float32)[None, :], repeats=4, axis=0)
            seq_tensor = torch.tensor(seq[None, :, :], dtype=torch.float32)
            lstm_prob = float(self.model(seq_tensor).squeeze().item())

        mmse = float(patient_features[self.mmse_idx])
        functional = float(patient_features[self.func_idx])
        adl = float(patient_features[self.adl_idx])

        impairment = np.clip((30 - mmse) / 30.0, 0.0, 1.0)
        functional_risk = np.clip((10 - functional) / 10.0, 0.0, 1.0)
        adl_risk = np.clip((10 - adl) / 10.0, 0.0, 1.0)
        rate = 0.45 * impairment + 0.3 * functional_risk + 0.25 * adl_risk

        _ = current_probability
        p0 = float(np.clip(lstm_prob, 0.01, 0.99))

        if p0 > 0.7:
            growth = 1 - np.exp(-0.09 * (months / 6.0) * (1 + rate))
            traj = p0 + (0.95 - p0) * growth
            band = 0.04
        elif p0 > 0.4:
            x = (months / 6.0) - 3.0
            sig = 1 / (1 + np.exp(-0.9 * x * (0.8 + rate)))
            traj = p0 + (0.82 - p0) * sig
            band = 0.06
        else:
            slow = (months / 36.0) ** 1.2
            traj = p0 + (0.58 - p0) * slow * (0.4 + 0.6 * rate)
            band = 0.08

        traj = np.clip(traj, 0.01, 0.98)
        lower = np.clip(traj - band, 0.0, 1.0)
        upper = np.clip(traj + band, 0.0, 1.0)

        return {
            "trajectory": [round(float(v), 4) for v in traj.tolist()],
            "confidence_bands": {
                "lower": [round(float(v), 4) for v in lower.tolist()],
                "upper": [round(float(v), 4) for v in upper.tolist()],
            },
        }
