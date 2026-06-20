from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

class DataPipeline:
    """Data pipeline for alzheimers_disease_data.csv with 34-feature schema."""

    target_column = "Diagnosis"
    drop_columns = ["PatientID", "DoctorInCharge"]
    categorical_columns = ["Gender", "Ethnicity", "EducationLevel", "DiseaseType"]

    # Resolution order when no explicit path is given: the realistic v4 synthetic
    # set (parquet preferred, csv fallback) takes priority, then the legacy CSVs.
    DATA_CANDIDATES = (
        "data/realistic_v4.parquet",
        "data/realistic_v4.csv",
        "neurological_disease_data.csv",
        "alzheimers_disease_data.csv",
    )

    def __init__(self, csv_path: str | None = None, models_dir: str | Path = "models") -> None:
        if csv_path is None:
            csv_path = next(
                (c for c in self.DATA_CANDIDATES if Path(c).exists()),
                "alzheimers_disease_data.csv",
            )
        self.csv_path = Path(csv_path)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.df_raw: pd.DataFrame | None = None
        self.df_processed: pd.DataFrame | None = None
        self.feature_names: list[str] = []
        self.scaler: StandardScaler | None = None
        self.dataset_stats: dict[str, Any] = {}

    def _load(self) -> pd.DataFrame:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found: {self.csv_path}. "
                "Run: python scripts/data/build_realistic_synthetic.py"
            )
        if self.csv_path.suffix == ".parquet":
            return pd.read_parquet(self.csv_path)
        return pd.read_csv(self.csv_path)

    @staticmethod
    def _safe_stats(series: pd.Series) -> dict[str, float]:
        return {
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std(ddof=0)), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
        }

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = df.copy()
        for col in self.categorical_columns:
            if col in encoded.columns:
                encoded[col] = encoded[col].astype(str).str.strip().str.lower()
                encoded[col], _ = pd.factorize(encoded[col], sort=True)
        return encoded

    def _build_dataset_stats(self, df: pd.DataFrame, features: list[str]) -> dict[str, Any]:
        n_total = int(len(df))
        n_alz = int((df[self.target_column] == 1).sum())
        n_healthy = int((df[self.target_column] == 0).sum())
        pct = round((n_alz / n_total) * 100.0, 2) if n_total else 0.0

        distributions: dict[str, dict[str, float]] = {}
        for col in features:
            distributions[col] = self._safe_stats(df[col])

        return {
            "n_patients": n_total,
            "n_alzheimers": n_alz,
            "n_healthy": n_healthy,
            "pct_alzheimers": pct,
            "mean_age": round(float(df["Age"].mean()), 4) if "Age" in df.columns else 0.0,
            "mean_mmse": round(float(df["MMSE"].mean()), 4) if "MMSE" in df.columns else 0.0,
            "feature_distributions": distributions,
        }

    def process(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list[str], StandardScaler, dict[str, Any]]:
        df = self._load()
        self.df_raw = df.copy()
        source_rows = int(len(df))

        for col in self.drop_columns:
            if col in df.columns:
                df = df.drop(columns=[col])

        if self.target_column not in df.columns:
            raise ValueError("Diagnosis target column is missing from dataset")

        df = self._encode_categoricals(df)

        numeric_cols = df.columns.tolist()
        for col in numeric_cols:
            if col == self.target_column:
                continue
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

        df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce").fillna(0).astype(int)

        if "DiseaseType" in df.columns:
            df = df.drop(columns=["DiseaseType"])

        feature_names = [c for c in df.columns if c != self.target_column]
        self.feature_names = feature_names

        X = df[feature_names]
        y = df[self.target_column]

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index)
        X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)

        self.scaler = scaler
        self.df_processed = df
        self.dataset_stats = self._build_dataset_stats(df, feature_names)
        self.dataset_stats["source_n_patients"] = source_rows
        self.dataset_stats["training_cohort"] = "full_dataset"

        joblib.dump(scaler, self.models_dir / "scaler.pkl")

        return X_train, X_test, y_train, y_test, feature_names, scaler, self.dataset_stats

    def split_by_disease(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
        if self.df_raw is None or self.scaler is None or not self.feature_names:
            self.process(test_size=test_size, random_state=random_state)

        assert self.df_raw is not None
        assert self.scaler is not None

        if "DiseaseType" not in self.df_raw.columns:
            return {}

        splits: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]] = {}

        for disease_name, raw_subset in self.df_raw.groupby("DiseaseType"):
            df = raw_subset.copy()
            for col in self.drop_columns:
                if col in df.columns:
                    df = df.drop(columns=[col])

            if self.target_column not in df.columns:
                continue

            df = self._encode_categoricals(df)

            for col in df.columns:
                if col == self.target_column:
                    continue
                df[col] = pd.to_numeric(df[col], errors="coerce")
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0.0
                df[col] = df[col].fillna(median_val)

            df[self.target_column] = pd.to_numeric(df[self.target_column], errors="coerce").fillna(0).astype(int)

            if "DiseaseType" in df.columns:
                df = df.drop(columns=["DiseaseType"])

            for fname in self.feature_names:
                if fname not in df.columns:
                    df[fname] = 0.0

            X = df[self.feature_names]
            y = df[self.target_column]

            if len(X) < 10:
                continue

            stratify = y if y.nunique() > 1 else None
            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

            X_train = pd.DataFrame(
                self.scaler.transform(X_train_raw),
                columns=self.feature_names,
                index=X_train_raw.index,
            )
            X_test = pd.DataFrame(
                self.scaler.transform(X_test_raw),
                columns=self.feature_names,
                index=X_test_raw.index,
            )

            splits[str(disease_name)] = (X_train, X_test, y_train, y_test)

        return splits
