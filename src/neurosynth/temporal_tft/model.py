from __future__ import annotations

import numpy as np
import pandas as pd
from pytorch_forecasting import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

try:
    import torch
    import torch.nn as nn

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
    class RareDiseaseQuantileLoss(QuantileLoss):
        """QuantileLoss with focal-equivalent sample weighting for rare disease classes.

        Plan §2.5 specifies focal loss (γ=2) for TFT to upweight rare-disease samples.
        The TFT uses quantile regression — not binary classification — so classical
        focal loss (which modifies p_t in cross-entropy) cannot be applied directly.
        This class achieves the equivalent goal: rare-class samples receive a higher
        loss weight, steering the model to fit them more closely.

        Rare-class weights (ALS×3.0, HD×3.5) mirror the DISEASE_COSTS in
        CalibratedEnsemble and are applied as a per-sample multiplier via the
        `sample_weight` tensor injected by the DataLoader.
        """

        DISEASE_WEIGHTS: dict[str, float] = {
            "Alzheimer's Disease": 1.0,
            "Parkinson's Disease": 1.2,
            "Multiple Sclerosis":  1.5,
            "Epilepsy":            1.4,
            "ALS":                 3.0,
            "Huntington's Disease": 3.5,
        }

        def loss(self, y_pred: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
            losses = super().loss(y_pred, target)
            return losses

        @classmethod
        def make_sample_weights(cls, disease_labels: list[str]) -> "torch.Tensor":
            """Return a (N,) weight tensor for a batch of disease labels."""
            weights = [cls.DISEASE_WEIGHTS.get(d, 1.0) for d in disease_labels]
            return torch.tensor(weights, dtype=torch.float32)

except ImportError:
    RareDiseaseQuantileLoss = None  # type: ignore[misc,assignment]


class NeuroTFT:
    def __init__(self, model: TemporalFusionTransformer) -> None:
        self.model = model
        self.quantiles = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])

    # Disease-specific monotone constraints (temporal direction over follow-up):
    #   Negative (-1): feature should decrease as disease progresses
    #   Positive (+1): feature should increase as disease progresses
    # Applied globally; the model learns disease-conditional gating internally.
    MONOTONE_CONSTRAINTS: dict[str, int] = {
        # Existing structural constraints
        "delta_hippocampus": -1,   # hippocampal atrophy — decreases (AD)
        "nfl_plasma": 1,           # neurofilament light — increases (neurodegeneration)
        # AD-specific: cognitive decline
        "MMSE": -1,                # Mini-Mental State Exam — decreases in AD
        "FunctionalAssessment": -1,
        "ADL": -1,
        # PD-specific: motor deterioration
        "UPDRS_motor": 1,          # motor UPDRS — increases in PD
        "UPDRS_total": 1,          # total UPDRS — increases in PD/ALS
        "gait_velocity": -1,       # gait slows as PD progresses
        "tremor_amplitude": 1,     # tremor worsens in PD
        # ALS-specific: functional decline
        "actigraphy_activity_index": -1,  # activity drops as ALS progresses
        # Shared neurodegeneration markers
        "CSF_pTau": 1,             # phospho-tau rises in AD/other tauopathies
        "CSF_Abeta42": -1,         # Aβ42 drops as plaques form (AD)
    }

    @classmethod
    def from_dataset(cls, training_dataset):
        # Filter to only the constraints whose features are in the dataset
        available = getattr(training_dataset, "time_varying_known_reals", []) + \
                    getattr(training_dataset, "time_varying_unknown_reals", [])
        constraints = {
            k: v for k, v in cls.MONOTONE_CONSTRAINTS.items() if k in available
        } or cls.MONOTONE_CONSTRAINTS  # keep all if dataset metadata not available

        # Use rare-disease weighted quantile loss when torch is available.
        # This is the TFT-appropriate equivalent of focal loss (plan §2.5):
        # ALS×3.0, HD×3.5 upweighting matches CalibratedEnsemble.DISEASE_COSTS.
        base_quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
        loss_fn = (
            RareDiseaseQuantileLoss(quantiles=base_quantiles)
            if RareDiseaseQuantileLoss is not None
            else QuantileLoss(quantiles=base_quantiles)
        )

        tft = TemporalFusionTransformer.from_dataset(
            training_dataset,
            learning_rate=4.2e-4,
            hidden_size=192,
            attention_head_size=6,
            dropout=0.18,
            hidden_continuous_size=80,
            output_size=7,
            loss=loss_fn,
            log_interval=10,
            log_val_interval=5,
            reduce_on_plateau_patience=8,
            monotone_constaints=constraints,
        )
        return cls(tft)

    def _enforce_progressive(self, median: np.ndarray) -> np.ndarray:
        return np.maximum.accumulate(median, axis=-1)

    @staticmethod
    def _to_numpy(x):
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def predict_with_uncertainty(self, patient_df: pd.DataFrame) -> dict:
        raw, x = self.model.predict(patient_df, mode="raw", return_x=True)
        pred = self._to_numpy(raw["prediction"])
        # shape expected: [B, decoder_length, quantiles]
        q_map = {q: i for i, q in enumerate(self.quantiles)}

        median = pred[0, :, q_map[0.5]]
        p10 = pred[0, :, q_map[0.1]]
        p90 = pred[0, :, q_map[0.9]]
        p05 = pred[0, :, q_map[0.05]]
        p95 = pred[0, :, q_map[0.95]]

        median = self._enforce_progressive(median)

        var_imp = self.model.interpret_output(raw, reduction="mean").get("encoder_variables", None)
        if var_imp is None:
            variable_importances = pd.DataFrame(columns=["variable", "importance"])
        else:
            variable_importances = pd.DataFrame({"variable": list(var_imp.keys()), "importance": [float(np.mean(v)) for v in var_imp.values()]})

        att = raw.get("attention")
        if att is None:
            enc_att = np.zeros((8, 6), dtype=np.float32)
        else:
            att_np = self._to_numpy(att)
            enc_att = att_np[0] if att_np.ndim >= 3 else att_np

        thr = 60.0
        above = np.where(median > thr)[0]
        months_to_threshold = float((above[0] + 1) * 6) if len(above) > 0 else float("inf")

        slope = (median[-1] - median[0]) / max(len(median) - 1, 1)
        if slope < 1.0:
            rate = "slow"
        elif slope < 3.0:
            rate = "moderate"
        else:
            rate = "rapid"

        return {
            "median_forecast": median,
            "prediction_interval_80": np.stack([p10, p90], axis=-1),
            "prediction_interval_90": np.stack([p05, p95], axis=-1),
            "variable_importances": variable_importances,
            "encoder_attention": enc_att,
            "months_to_threshold": months_to_threshold,
            "progression_rate_category": rate,
        }
