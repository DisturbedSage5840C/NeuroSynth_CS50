# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
from pandera import Column, DataFrameSchema
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge

try:
    from neuroCombat import neuroCombat
except Exception:  # pragma: no cover
    neuroCombat = None


class BiomarkerTimeSeriesPreprocessor:
    VISIT_MAP = {
        "BL": 0,
        "M06": 6,
        "M12": 12,
        "M18": 18,
        "M24": 24,
        "M36": 36,
        "M48": 48,
        "M60": 60,
    }

    def _harmonize_sites(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain"]
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return df

        covars = pd.DataFrame(
            {
                "batch": df.get("RID", "UNK").astype(str),
                "AGE": pd.to_numeric(df.get("AGE", np.nan), errors="coerce").fillna(df.get("AGE", 0)),
                "PTGENDER": df.get("PTGENDER", "UNK").astype(str),
                "PTEDUCAT": pd.to_numeric(df.get("PTEDUCAT", np.nan), errors="coerce").fillna(0),
                "DX_bl": df.get("DX_bl", "UNK").astype(str),
            }
        )

        matrix = df[cols].apply(pd.to_numeric, errors="coerce").fillna(df[cols].median(numeric_only=True)).to_numpy(dtype=float).T
        if neuroCombat is not None:
            harmonized = neuroCombat(dat=matrix, covars=covars, batch_col="batch", categorical_cols=["PTGENDER", "DX_bl"], continuous_cols=["AGE", "PTEDUCAT"])["data"].T
            df.loc[:, cols] = harmonized
        else:
            for c in cols:
                df[c] = df[c] - df.groupby("RID")[c].transform("mean") + df[c].mean()
        return df

    def _impute_biomarkers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["patient_id", "month"]).copy()
        csf_cols = [c for c in ["ABETA", "PTAU", "TAU", "nfl_pgml", "alpha_syn_csf"] if c in out.columns]
        img_cols = [c for c in ["Hippocampus", "Entorhinal", "Fusiform", "MidTemp", "Ventricles", "WholeBrain", "L_hippo", "R_hippo"] if c in out.columns]
        cog_cols = [c for c in ["CDRSB", "ADAS13", "MMSE", "MOCA"] if c in out.columns]
        wear_cols = [c for c in ["gait_speed", "tremor_index", "bradykinesia_score", "step_count_daily", "sleep_efficiency"] if c in out.columns]

        for col in csf_cols:
            out[col] = out.groupby("patient_id")[col].transform(lambda s: s.interpolate(limit_direction="both").ffill().bfill())

        if img_cols:
            imp = IterativeImputer(estimator=BayesianRidge(), max_iter=10, random_state=42)
            out[img_cols] = imp.fit_transform(out[img_cols])

        for col in cog_cols:
            out[col] = out.groupby("patient_id")[col].transform(lambda s: s.ffill(limit=2))
            out[f"{col}_usable"] = out.groupby("patient_id")[col].transform(lambda s: s.isna().rolling(3, min_periods=1).sum().le(2)).astype(bool)

        for col in wear_cols:
            out[col] = out.groupby("patient_id")[col].transform(lambda s: s.fillna(s.mean()))
        return out

    def _z(self, series: pd.Series, baseline_mask: pd.Series) -> pd.Series:
        ref = pd.to_numeric(series[baseline_mask], errors="coerce")
        mu = ref.mean()
        sd = ref.std(ddof=0) or 1.0
        return (pd.to_numeric(series, errors="coerce") - mu) / sd

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["patient_id", "month"]).copy()
        out["time_idx"] = (out["month"] // 6).astype(int)
        out["visit_number"] = out.groupby("patient_id").cumcount() + 1
        out["months_between"] = out.groupby("patient_id")["month"].diff().replace(0, np.nan)

        for col in ["Hippocampus", "nfl_pgml", "CDRSB"]:
            if col in out.columns:
                delta = out.groupby("patient_id")[col].diff() / out["months_between"]
                out[f"delta_{col.lower()}"] = delta.replace([np.inf, -np.inf], 0).fillna(0)
                out[f"accel_{col.lower()}"] = out.groupby("patient_id")[f"delta_{col.lower()}"] .diff().fillna(0)

        if "medication_dose" in out.columns and "medication_class" in out.columns:
            out["total_drug_burden_score"] = out.groupby(["patient_id", "medication_class"])["medication_dose"].cumsum()
        else:
            out["total_drug_burden_score"] = 0.0

        if "ABETA" in out.columns and "PTAU" in out.columns:
            out["csf_ratio"] = out["ABETA"] / (out["PTAU"] + 1)
        else:
            out["csf_ratio"] = 0.0

        if "L_hippo" in out.columns and "R_hippo" in out.columns:
            mean_hip = (out["L_hippo"] + out["R_hippo"]) / 2.0
            out["atrophy_asymmetry"] = (out["L_hippo"] - out["R_hippo"]).abs() / mean_hip.replace(0, np.nan)
            out["atrophy_asymmetry"] = out["atrophy_asymmetry"].fillna(0)
        else:
            out["atrophy_asymmetry"] = 0.0

        return out

    def _compute_dci(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        baseline_mask = out.get("DX_bl", "CN").astype(str).eq("CN")

        cdr = self._z(out.get("CDRSB", 0), baseline_mask)
        hippo = self._z(-pd.to_numeric(out.get("Hippocampus", 0), errors="coerce"), baseline_mask)
        nfl = self._z(out.get("nfl_pgml", 0), baseline_mask)
        adas = self._z(out.get("ADAS13", 0), baseline_mask)

        out["dci"] = (0.35 * cdr + 0.25 * hippo + 0.25 * nfl + 0.15 * adas).clip(0, 100)
        out["target"] = out["dci"]
        return out

    def _schema(self) -> DataFrameSchema:
        return DataFrameSchema(
            {
                "time_idx": Column(int, nullable=False),
                "patient_id": Column(str, nullable=False),
                "dci": Column(float, nullable=False),
                "target": Column(float, nullable=False),
            },
            coerce=True,
            strict=False,
        )

    def build_longitudinal_dataset(self, adni_path: Path, ppmi_path: Path, output_path: Path) -> pd.DataFrame:
        adni = pd.read_csv(adni_path)
        ppmi = pd.read_csv(ppmi_path)

        if "PTID" in adni.columns:
            adni["patient_id"] = adni["PTID"].astype(str)
        if "VISCODE" in adni.columns:
            adni["month"] = adni["VISCODE"].astype(str).map(self.VISIT_MAP).fillna(0).astype(int)

        merged = pd.concat([adni, ppmi], ignore_index=True, sort=False)
        merged = self._harmonize_sites(merged)
        merged = self._impute_biomarkers(merged)
        merged = self._feature_engineering(merged)
        merged = self._compute_dci(merged)

        merged["patient_id"] = merged["patient_id"].astype(str)
        merged["group_id"] = merged["patient_id"]

        schema = self._schema()
        schema.validate(merged, lazy=True)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(output_path, index=False)
        return merged
