from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class VariableImportanceAnalyzer:
    def __init__(self, feature_catalog_path: Path | None = None) -> None:
        catalog_path = feature_catalog_path or (Path(__file__).parent / "resources" / "feature_catalog.json")
        with catalog_path.open("r", encoding="utf-8") as f:
            self.feature_catalog = json.load(f)

    def get_tft_variable_selection(self, model, dataset) -> pd.DataFrame:
        raw = model.predict(dataset.to_dataloader(train=False, batch_size=64), mode="raw")
        interp = model.interpret_output(raw, reduction="mean")

        rows = []
        for key in ["encoder_variables", "decoder_variables", "static_variables"]:
            vars_dict = interp.get(key, {})
            for var, val in vars_dict.items():
                arr = np.asarray(val)
                rows.append(
                    {
                        "variable": var,
                        "scope": key,
                        "mean_importance": float(arr.mean()),
                        "std_importance": float(arr.std(ddof=0)),
                    }
                )
        return pd.DataFrame(rows).sort_values("mean_importance", ascending=False)

    def compute_shap_values(self, model, background_data, test_patients, n_samples: int = 100):
        sample_bg = background_data[: min(len(background_data), n_samples)]
        sample_test = test_patients[: min(len(test_patients), n_samples)]
        explainer = shap.DeepExplainer(model, sample_bg)
        return explainer(sample_test)

    def generate_clinical_insight_report(self, patient_id: str, model, data) -> str:
        shap_values = self.compute_shap_values(model, data[:10], data[:1], n_samples=10)
        vals = np.abs(np.asarray(shap_values.values)).mean(axis=tuple(range(np.asarray(shap_values.values).ndim - 1)))
        feature_names = shap_values.feature_names if getattr(shap_values, "feature_names", None) else [f"feature_{i}" for i in range(vals.shape[0])]

        modifiable = set(self.feature_catalog.get("modifiable", []))
        ranked = sorted(zip(feature_names, vals.tolist()), key=lambda x: x[1], reverse=True)
        top_mod = [x for x in ranked if x[0] in modifiable][:3]
        if not top_mod:
            top_mod = ranked[:3]

        lines = [f"Patient {patient_id} clinical insight report:"]
        for feat, score in top_mod:
            lines.append(
                f"Feature {feat} is a primary driver of predicted deterioration, contributing {score:.3f} units to 12-month forecast."
            )
        return "\n".join(lines)
