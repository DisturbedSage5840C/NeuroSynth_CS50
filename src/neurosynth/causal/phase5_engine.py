# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


@dataclass
class Phase5Config:
    bootstrap_samples: int = 50


class CausalPhase5Engine:
    """Phase 5 causal engine: graph discovery, DoWhy effects, and counterfactual simulation."""

    def __init__(self, config: Phase5Config | None = None) -> None:
        self.config = config or Phase5Config()
        self._dag_edges: list[tuple[str, str]] = []

    def discover_dag(self, frame: pd.DataFrame) -> list[tuple[str, str]]:
        from causallearn.search.ConstraintBased.PC import pc

        cols = [c for c in frame.columns if np.issubdtype(frame[c].dtype, np.number)]
        data = frame[cols].to_numpy(dtype=float)
        result = pc(data, alpha=0.05)

        edges: list[tuple[str, str]] = []
        graph = result.G.graph
        for i in range(graph.shape[0]):
            for j in range(graph.shape[1]):
                if graph[i, j] != 0 and i != j:
                    edges.append((cols[i], cols[j]))

        self._dag_edges = edges
        return edges

    def estimate_effect(
        self,
        frame: pd.DataFrame,
        treatment: str,
        outcome: str,
        common_causes: list[str],
    ) -> float:
        from dowhy import CausalModel

        model = CausalModel(
            data=frame,
            treatment=treatment,
            outcome=outcome,
            common_causes=common_causes,
        )
        estimand = model.identify_effect(proceed_when_unidentifiable=True)
        estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
        return float(estimate.value)

    def counterfactual_forecast(
        self,
        patient_row: pd.Series,
        biomarker: str,
        reduction_frac: float,
        forecast_fn: Callable[[pd.DataFrame], np.ndarray],
    ) -> dict[str, Any]:
        before = patient_row.to_frame().T.copy()
        after = patient_row.to_frame().T.copy()
        after[biomarker] = after[biomarker] * (1.0 - reduction_frac)

        before_forecast = np.asarray(forecast_fn(before), dtype=float)
        after_forecast = np.asarray(forecast_fn(after), dtype=float)

        return {
            "before": before_forecast.tolist(),
            "after": after_forecast.tolist(),
            "delta": (after_forecast - before_forecast).tolist(),
        }

    def predict_with_uncertainty(self, X: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        x_num = X.select_dtypes(include=[np.number]).copy()
        y_arr = y.to_numpy(dtype=float)

        preds = []
        importances = []
        for _ in range(self.config.bootstrap_samples):
            idx = np.random.choice(len(x_num), size=len(x_num), replace=True)
            x_boot = x_num.iloc[idx]
            y_boot = y_arr[idx]
            reg = RandomForestRegressor(n_estimators=100, random_state=None)
            reg.fit(x_boot, y_boot)
            preds.append(reg.predict(x_num))
            importances.append(reg.feature_importances_)

        pred_arr = np.stack(preds, axis=0)
        mean = pred_arr.mean(axis=0)
        lower_80, upper_80 = np.quantile(pred_arr, [0.10, 0.90], axis=0)
        lower_95, upper_95 = np.quantile(pred_arr, [0.025, 0.975], axis=0)

        imp = np.mean(np.stack(importances, axis=0), axis=0)
        top_idx = np.argsort(np.abs(imp))[::-1][:10]
        shap_values = [
            {"feature": x_num.columns[int(i)], "value": float(imp[int(i)])}
            for i in top_idx
        ]

        return {
            "mean": mean.tolist(),
            "lower_80": lower_80.tolist(),
            "upper_80": upper_80.tolist(),
            "lower_95": lower_95.tolist(),
            "upper_95": upper_95.tolist(),
            "shap_values": shap_values,
        }
