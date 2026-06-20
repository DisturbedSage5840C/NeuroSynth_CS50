from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from neurosynth.causal.types import CausalInput

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

VARIABLE_CONFIG = {
    "abeta42": {"scale": "log", "clip": (100, 2000), "modifiable": False},
    "ptau181": {"scale": "log", "clip": (5, 500), "modifiable": False},
    "total_tau": {"scale": "log", "clip": (50, 2000), "modifiable": False},
    "alpha_syn": {"scale": "log", "clip": (500, 5000), "modifiable": False},
    "nfl": {"scale": "log", "clip": (1, 500), "modifiable": False},
    "hippocampus": {"scale": "linear", "clip": (1000, 5000), "modifiable": False},
    "entorhinal": {"scale": "linear", "clip": (0, 5000), "modifiable": False},
    "fusiform": {"scale": "linear", "clip": (0, 5000), "modifiable": False},
    "midtemp": {"scale": "linear", "clip": (0, 5000), "modifiable": False},
    "ventricles": {"scale": "linear", "clip": (2000, 80000), "modifiable": False},
    "wholebrain": {"scale": "linear", "clip": (500000, 1800000), "modifiable": False},
    "cdrsb": {"scale": "linear", "clip": (0, 18), "modifiable": True},
    "mmse": {"scale": "linear", "clip": (0, 30), "modifiable": True},
    "moca": {"scale": "linear", "clip": (0, 30), "modifiable": True},
    "adas13": {"scale": "linear", "clip": (0, 85), "modifiable": True},
    "updrs3": {"scale": "linear", "clip": (0, 132), "modifiable": True},
    "gait_speed": {"scale": "linear", "clip": (0, 2), "modifiable": True},
    "sleep_efficiency": {"scale": "linear", "clip": (0, 1), "modifiable": True},
    "step_count": {"scale": "linear", "clip": (0, 40000), "modifiable": True},
    "tremor_index": {"scale": "linear", "clip": (0, 10), "modifiable": True},
    "bradykinesia_score": {"scale": "linear", "clip": (0, 10), "modifiable": True},
    "age": {"scale": "linear", "clip": (40, 100), "modifiable": False},
    "sex_male": {"scale": "linear", "clip": (0, 1), "modifiable": False},
    "education_years": {"scale": "linear", "clip": (0, 30), "modifiable": False},
    "apoe_e4_count": {"scale": "linear", "clip": (0, 2), "modifiable": False},
    "prs_ad": {"scale": "linear", "clip": (-5, 5), "modifiable": False},
    "inflammation_proxy": {"scale": "linear", "clip": (-5, 5), "modifiable": True},
    "dci": {"scale": "linear", "clip": (0, 100), "modifiable": False},
}


@dataclass
class CausalDataPreparer:
    norm_mean_: pd.Series | None = None
    norm_std_: pd.Series | None = None

    @property
    def variable_names(self) -> list[str]:
        return list(VARIABLE_CONFIG.keys())

    def _scale_variable(self, s: pd.Series, scale: str, clip_low: float, clip_high: float) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce").clip(clip_low, clip_high)
        if scale == "log":
            x = np.log1p(x)
        return x

    def _apply_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for v, cfg in VARIABLE_CONFIG.items():
            if v not in out.columns:
                out[v] = np.nan
            out[v] = self._scale_variable(out[v], cfg["scale"], cfg["clip"][0], cfg["clip"][1])
        return out

    def _normalize(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        vals = df[self.variable_names]
        if fit or self.norm_mean_ is None or self.norm_std_ is None:
            self.norm_mean_ = vals.mean()
            self.norm_std_ = vals.std(ddof=0).replace(0, 1.0)
        out = df.copy()
        out[self.variable_names] = (vals - self.norm_mean_) / self.norm_std_
        out[self.variable_names] = out[self.variable_names].fillna(0.0)
        return out

    def prepare_causal_matrix(self, patient_longitudinal_df: pd.DataFrame) -> CausalInput:
        df = patient_longitudinal_df.copy()
        if "patient_id" not in df.columns:
            raise ValueError("Input must include patient_id")
        if "time_idx" not in df.columns:
            raise ValueError("Input must include time_idx")

        df = self._apply_scaling(df)
        df = self._normalize(df, fit=True)

        patient0 = str(df["patient_id"].astype(str).iloc[0])
        p = df[df["patient_id"].astype(str) == patient0].sort_values("time_idx")
        patient_matrix = p[self.variable_names].to_numpy(dtype=np.float32)

        delta = np.diff(patient_matrix, axis=0)
        patient_delta_matrix = np.concatenate([patient_matrix[1:], delta], axis=1).astype(np.float32)

        pop = df.sort_values(["patient_id", "time_idx"])
        population_matrix = pop[self.variable_names].to_numpy(dtype=np.float32)

        modifiability_mask = torch.tensor([bool(VARIABLE_CONFIG[v]["modifiable"]) for v in self.variable_names], dtype=torch.bool)
        return CausalInput(
            patient_matrix=patient_matrix,
            patient_delta_matrix=patient_delta_matrix,
            population_matrix=population_matrix,
            variable_names=self.variable_names,
            modifiability_mask=modifiability_mask,
        )
