from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
DEFAULT_VARIABLES = [
    "Age",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
    "MemoryComplaints",
    "BehavioralProblems",
    "Depression",
    "SleepQuality",
    "PhysicalActivity",
    "Diagnosis",
]

DISEASE_VARIABLES = {
    "Alzheimer's Disease": DEFAULT_VARIABLES,
    "Parkinson's Disease": [
        "Age",
        "MMSE",
        "FunctionalAssessment",
        "ADL",
        "Depression",
        "SleepQuality",
        "PhysicalActivity",
        "Diagnosis",
    ],
    "Multiple Sclerosis": [
        "Age",
        "FunctionalAssessment",
        "ADL",
        "Depression",
        "SleepQuality",
        "PhysicalActivity",
        "Diagnosis",
    ],
    "Epilepsy": [
        "Age",
        "MMSE",
        "SleepQuality",
        "Depression",
        "PhysicalActivity",
        "Diagnosis",
    ],
    "ALS": [
        "Age",
        "FunctionalAssessment",
        "ADL",
        "PhysicalActivity",
        "Depression",
        "Diagnosis",
    ],
    "Huntington's Disease": [
        "Age",
        "MMSE",
        "FunctionalAssessment",
        "ADL",
        "MemoryComplaints",
        "Depression",
        "Diagnosis",
    ],
}


class _NodeMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NeuralCausalDiscovery(nn.Module):
    def __init__(self, models_dir: str | Path = "models", variables: list[str] | None = None) -> None:
        super().__init__()
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.variables = list(variables) if variables else list(DEFAULT_VARIABLES)
        self.n_vars = len(self.variables)

        self.W_logits = nn.Parameter(torch.zeros(self.n_vars, self.n_vars))
        self.mlps = nn.ModuleList([_NodeMLP(self.n_vars) for _ in range(self.n_vars)])
        self.latest_W: np.ndarray | None = None
        self.min_vals: np.ndarray | None = None
        self.max_vals: np.ndarray | None = None

    def get_adjacency(self) -> torch.Tensor:
        W = torch.sigmoid(self.W_logits)
        W = W * (1 - torch.eye(self.n_vars, device=W.device))

        flat = W.flatten()
        k = max(1, int(flat.numel() * 0.3))
        threshold = torch.topk(flat, k).values.min()
        mask = (W >= threshold).float()
        return W * mask

    @staticmethod
    def acyclicity_constraint(W: torch.Tensor) -> torch.Tensor:
        return torch.trace(torch.matrix_exp(W * W)) - W.shape[0]

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        W = self.get_adjacency()
        outs = []
        for j in range(self.n_vars):
            xj = X * W[:, j]
            outs.append(self.mlps[j](xj))
        X_hat = torch.cat(outs, dim=1)
        return X_hat, W

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 1000,
        outer_iters: int = 15,
        inner_iters: int = 100,
        lr: float = 0.01,
        lambda1: float = 0.01,
        lambda2: float = 10.0,
    ) -> None:
        X = np.asarray(X, dtype=np.float32)
        self.min_vals = X.min(axis=0)
        self.max_vals = X.max(axis=0)

        X_t = torch.tensor(X, dtype=torch.float32)
        optim = torch.optim.Adam(self.parameters(), lr=lr)

        alpha = torch.tensor(0.0)
        for _ in range(outer_iters):
            for _ in range(inner_iters):
                optim.zero_grad()
                X_hat, W = self.forward(X_t)
                recon = ((X_hat - X_t) ** 2).mean()
                sparsity = lambda1 * torch.sum(torch.abs(W))
                h = self.acyclicity_constraint(W)
                loss = recon + sparsity + alpha * h + 0.5 * lambda2 * h * h
                loss.backward()
                optim.step()
            with torch.no_grad():
                h_val = self.acyclicity_constraint(self.get_adjacency())
                alpha = alpha + lambda2 * h_val

        self.latest_W = self.get_adjacency().detach().cpu().numpy()
        np.save(self.models_dir / "causal_graph.npy", self.latest_W)
        (self.models_dir / "causal_vars.json").write_text(json.dumps(self.variables, indent=2), encoding="utf-8")

    def _load_if_needed(self) -> None:
        if self.latest_W is None:
            graph_file = self.models_dir / "causal_graph.npy"
            if graph_file.exists():
                self.latest_W = np.load(graph_file)
            else:
                self.latest_W = np.zeros((self.n_vars, self.n_vars), dtype=float)

    def get_causal_graph(self) -> dict[str, Any]:
        self._load_if_needed()
        W = self.latest_W

        # Safe variable lookups — avoid ValueError if variables are missing.
        diag_idx = self.variables.index("Diagnosis") if "Diagnosis" in self.variables else None
        mmse_idx = self.variables.index("MMSE") if "MMSE" in self.variables else None

        edges = []
        for i, src in enumerate(self.variables):
            for j, dst in enumerate(self.variables):
                if i == j:
                    continue
                strength = float(W[i, j])
                if strength > 0.25:
                    edge_type = "direct" if abs(i - j) <= 2 else "indirect"
                    edges.append(
                        {
                            "from": src,
                            "to": dst,
                            "strength": round(strength, 4),
                            "type": edge_type,
                        }
                    )

        def top_causes(target_idx: int | None, k: int) -> list[dict[str, float]]:
            if target_idx is None:
                return []
            vals = []
            for i, name in enumerate(self.variables):
                if i == target_idx:
                    continue
                vals.append((name, float(W[i, target_idx])))
            vals.sort(key=lambda x: x[1], reverse=True)
            return [{"variable": n, "strength": round(s, 4)} for n, s in vals[:k]]

        top_d = top_causes(diag_idx, 4)
        top_m = top_causes(mmse_idx, 3)

        protective = []
        amplifiers = []
        modifiable = []
        if diag_idx is not None:
            for i in range(len(self.variables)):
                if i == diag_idx:
                    continue
                name = self.variables[i]
                eff = float(W[i, diag_idx])
                if name in {"PhysicalActivity", "SleepQuality", "MMSE", "FunctionalAssessment", "ADL", "Depression"}:
                    if eff < 0.35:
                        protective.append({"variable": name, "effect": round(-abs(eff), 4)})
                    if eff > 0.45:
                        amplifiers.append({"variable": name, "effect": round(eff, 4)})
                    modifiable.append(
                        {
                            "variable": name,
                            "current_effect": round(eff, 4),
                            "intervention_direction": "increase" if name in {"PhysicalActivity", "SleepQuality", "MMSE", "FunctionalAssessment", "ADL"} else "decrease",
                            "expected_impact": "Potential reduction in modeled diagnosis risk",
                        }
                    )

        return {
            "variables": self.variables,
            "edges": sorted(edges, key=lambda x: x["strength"], reverse=True),
            "adjacency_matrix": np.round(W, 4).tolist(),
            "top_causes_of_Diagnosis": top_d,
            "top_causes_of_MMSE": top_m,
            "protective_factors": protective,
            "risk_amplifiers": amplifiers,
            "modifiable_interventions": modifiable,
        }

    def simulate_intervention(
        self,
        variable: str,
        new_value_normalized: float,
        patient_data: dict[str, float],
    ) -> dict[str, Any]:
        self._load_if_needed()
        if variable not in self.variables:
            raise ValueError(f"Unknown variable: {variable}")

        idx_map = {v: i for i, v in enumerate(self.variables)}
        x = np.zeros(self.n_vars, dtype=float)
        for v in self.variables:
            if v == "Diagnosis":
                x[idx_map[v]] = 0.0
            else:
                x[idx_map[v]] = float(patient_data.get(v, 0.0))

        if self.min_vals is None or self.max_vals is None:
            self.min_vals = np.zeros(self.n_vars)
            self.max_vals = np.ones(self.n_vars)

        denom = np.where((self.max_vals - self.min_vals) == 0, 1.0, self.max_vals - self.min_vals)
        x_norm = (x - self.min_vals) / denom
        x_norm = np.clip(x_norm, 0.0, 1.0)

        diag_idx = idx_map["Diagnosis"]

        incoming = self.latest_W[:, diag_idx]
        original_score = float(np.dot(incoming, x_norm))
        original_risk = float(1.0 / (1.0 + np.exp(-original_score)))

        x_new = x_norm.copy()
        x_new[idx_map[variable]] = float(np.clip(new_value_normalized, 0.0, 1.0))
        intervened_score = float(np.dot(incoming, x_new))
        intervened_risk = float(1.0 / (1.0 + np.exp(-intervened_score)))

        arr = np.abs(self.latest_W[idx_map[variable], :])
        top_downstream_idx = np.argsort(arr)[::-1][:4]
        downstream = [self.variables[i] for i in top_downstream_idx if i != idx_map[variable] and arr[i] > 0.1]

        absolute_reduction = max(0.0, original_risk - intervened_risk)
        relative_reduction = (absolute_reduction / max(original_risk, 1e-6)) * 100.0

        interpretation = (
            f"Adjusting {variable} to normalized value {new_value_normalized:.2f} is estimated to change modeled "
            f"Alzheimer's risk from {original_risk:.1%} to {intervened_risk:.1%}. "
            f"Estimated absolute reduction is {absolute_reduction:.1%} through downstream effects on {', '.join(downstream) if downstream else 'key cognitive pathways'}."
        )

        return {
            "original_risk": round(original_risk, 4),
            "intervened_risk": round(intervened_risk, 4),
            "absolute_risk_reduction": round(absolute_reduction, 4),
            "relative_risk_reduction_pct": round(relative_reduction, 2),
            "affected_downstream_vars": downstream,
            "interpretation": interpretation,
        }
