# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CalibratedTFT:
    model: object
    quantile_calibrators: dict[float, object]
    coverage_table: pd.DataFrame


@dataclass
class ValidationReport:
    overall_metrics: dict[str, float]
    subgroup_metrics: pd.DataFrame
    calibration_metrics: dict[str, float]
    temporal_examples: pd.DataFrame
    per_horizon_metrics: pd.DataFrame
    notes: list[str] = field(default_factory=list)


@dataclass
class PredictionWithUncertainty:
    median_forecast: np.ndarray
    prediction_interval_80: np.ndarray
    prediction_interval_90: np.ndarray
    variable_importances: pd.DataFrame
    encoder_attention: np.ndarray
    months_to_threshold: float
    progression_rate_category: str
