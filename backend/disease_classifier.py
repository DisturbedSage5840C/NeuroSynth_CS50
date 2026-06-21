# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""
Disease-type classifier. Takes patient features and returns the most likely
neurological disease type. Used to route to disease-specific risk models.

v2 FIX: Replaced synthetic data generation with real dataset loading.
The previous generate_synthetic_training_data() used rng.normal() which
produced clinically impossible values. The classifier now trains on the
actual dataset features, ensuring feature alignment during inference.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

DISEASES = [
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Multiple Sclerosis",
    "Epilepsy",
    "ALS",
    "Huntington's Disease",
]

# Features used for disease classification.  Must match the features
# available in the real dataset after preprocessing.
CLASSIFICATION_FEATURES = [
    "Age",
    "Gender",
    "BMI",
    "SystolicBP",
    "DiastolicBP",
    "CholesterolTotal",
    "PhysicalActivity",
    "SleepQuality",
    "Depression",
    "MMSE",
    "FunctionalAssessment",
    "ADL",
    "MemoryComplaints",
    "FamilyHistoryAlzheimers",
    # Additional features available in the full dataset:
    "Smoking",
    "AlcoholConsumption",
    "DietQuality",
    "CardiovascularDisease",
    "Diabetes",
    "HeadInjury",
    "Hypertension",
    "CholesterolLDL",
    "CholesterolHDL",
    "CholesterolTriglycerides",
    "BehavioralProblems",
    "Confusion",
    "Disorientation",
    "PersonalityChanges",
    "DifficultyCompletingTasks",
    "Forgetfulness",
    "EducationLevel",
    "Ethnicity",
]


class DiseaseClassifier:
    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.le = LabelEncoder()
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------
    # v2: Load training data from real dataset instead of generating
    # synthetic data.  Falls back to synthetic ONLY as a last resort.
    # ------------------------------------------------------------------

    @staticmethod
    def _find_dataset() -> Path | None:
        """Locate the best available dataset file."""
        candidates = [
            Path("neurological_disease_data.csv"),
            Path("alzheimers_disease_data.csv"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _load_real_training_data(self) -> tuple[pd.DataFrame, pd.Series] | None:
        """Load training data from the real dataset.

        For datasets with a ``DiseaseType`` column, that is used directly
        as the label.  For datasets without it (e.g. the original
        alzheimers_disease_data.csv), we generate probabilistic labels
        based on the actual feature distributions in the data — not from
        ``rng.normal()``.
        """
        dataset_path = self._find_dataset()
        if dataset_path is None:
            logger.warning("No dataset file found for DiseaseClassifier training")
            return None

        df = pd.read_csv(dataset_path)
        logger.info("DiseaseClassifier: loaded %d rows from %s", len(df), dataset_path)

        # Determine which classification features are available.
        available = [f for f in CLASSIFICATION_FEATURES if f in df.columns]
        if len(available) < 5:
            logger.warning("Too few classification features (%d) in dataset", len(available))
            return None

        # Encode categoricals the same way the main pipeline does.
        for col in ["Gender", "Ethnicity", "EducationLevel"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip().str.lower()
                df[col], _ = pd.factorize(df[col], sort=True)

        # Fill missing values with column medians.
        for col in available:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0.0
            df[col] = df[col].fillna(median_val)

        # --- Case 1: DiseaseType column exists → use directly ---
        if "DiseaseType" in df.columns:
            labels = df["DiseaseType"].astype(str).str.strip()
            X = df[available]
            return X, labels

        # --- Case 2: No DiseaseType → generate probabilistic labels ---
        # Uses actual patient feature profiles to assign most-likely disease.
        labels = self._assign_probabilistic_labels(df, available)
        X = df[available]
        return X, labels

    @staticmethod
    def _assign_probabilistic_labels(df: pd.DataFrame, features: list[str]) -> pd.Series:
        """Assign disease labels based on real feature distributions.

        Uses clinically-informed scoring rules on actual data values
        (not random generation), so the resulting labels are consistent
        with the feature profiles present in the dataset.
        """
        scores = pd.DataFrame(index=df.index)

        age = df["Age"] if "Age" in df.columns else pd.Series(70, index=df.index)
        mmse = df["MMSE"] if "MMSE" in df.columns else pd.Series(25, index=df.index)
        func = df["FunctionalAssessment"] if "FunctionalAssessment" in df.columns else pd.Series(6, index=df.index)
        adl = df["ADL"] if "ADL" in df.columns else pd.Series(6, index=df.index)
        memory = df["MemoryComplaints"] if "MemoryComplaints" in df.columns else pd.Series(0, index=df.index)
        fam_hist = df["FamilyHistoryAlzheimers"] if "FamilyHistoryAlzheimers" in df.columns else pd.Series(0, index=df.index)
        depression = df["Depression"] if "Depression" in df.columns else pd.Series(0, index=df.index)
        sleep = df["SleepQuality"] if "SleepQuality" in df.columns else pd.Series(5, index=df.index)
        phys = df["PhysicalActivity"] if "PhysicalActivity" in df.columns else pd.Series(5, index=df.index)
        behav = df["BehavioralProblems"] if "BehavioralProblems" in df.columns else pd.Series(0, index=df.index)

        # Alzheimer's: older age, low MMSE, memory complaints, family history
        scores["Alzheimer's Disease"] = (
            (age - 60).clip(lower=0) / 40.0 * 0.25
            + (30 - mmse).clip(lower=0) / 30.0 * 0.35
            + memory * 0.20
            + fam_hist * 0.15
            + (10 - func).clip(lower=0) / 10.0 * 0.05
        )

        # Parkinson's: mid-to-older age, moderate MMSE, low physical activity
        scores["Parkinson's Disease"] = (
            ((age - 55).clip(lower=0) / 45.0) * 0.20
            + ((30 - mmse).clip(lower=0) / 30.0) * 0.15
            + depression * 0.20
            + (10 - phys).clip(lower=0) / 10.0 * 0.25
            + (10 - sleep).clip(lower=0) / 10.0 * 0.20
        )

        # MS: younger age, moderate function
        scores["Multiple Sclerosis"] = (
            (60 - age).clip(lower=0) / 60.0 * 0.35
            + depression * 0.20
            + (10 - func).clip(lower=0) / 10.0 * 0.20
            + (10 - sleep).clip(lower=0) / 10.0 * 0.15
            + behav * 0.10
        )

        # Epilepsy: younger age, moderate sleep issues
        scores["Epilepsy"] = (
            (55 - age).clip(lower=0) / 55.0 * 0.30
            + (10 - sleep).clip(lower=0) / 10.0 * 0.30
            + depression * 0.15
            + behav * 0.15
            + (10 - phys).clip(lower=0) / 10.0 * 0.10
        )

        # ALS: mid-age, very low ADL and function
        scores["ALS"] = (
            ((age - 45).clip(lower=0) / 55.0) * 0.15
            + (10 - func).clip(lower=0) / 10.0 * 0.35
            + (10 - adl).clip(lower=0) / 10.0 * 0.35
            + (10 - phys).clip(lower=0) / 10.0 * 0.15
        )

        # Huntington's: mid-age, memory + behavioral problems
        scores["Huntington's Disease"] = (
            ((age - 30).clip(lower=0) / 50.0) * 0.15
            + memory * 0.25
            + behav * 0.25
            + (30 - mmse).clip(lower=0) / 30.0 * 0.20
            + depression * 0.15
        )

        # Add noise to prevent deterministic ties.
        rng = np.random.RandomState(42)
        for col in scores.columns:
            scores[col] += rng.uniform(0, 0.05, size=len(scores))

        return scores.idxmax(axis=1)

    def generate_synthetic_training_data(self, n_per_class: int = 500):
        """Kept for backward compatibility but deprecated.

        Prefer train() which loads real data first.
        """
        import warnings
        warnings.warn(
            "generate_synthetic_training_data is deprecated. "
            "DiseaseClassifier.train() now loads from real datasets.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Original synthetic generation with clipping applied.
        all_rows: list[dict[str, float]] = []
        all_labels: list[str] = []
        rng = np.random.RandomState(42)

        profiles: dict[str, dict[str, Any]] = {
            "Alzheimer's Disease": {"Age": (75, 8), "MMSE": (18, 6), "MemoryComplaints": 0.85, "FamilyHistoryAlzheimers": 0.35},
            "Parkinson's Disease": {"Age": (68, 9), "MMSE": (24, 4), "MemoryComplaints": 0.35, "FamilyHistoryAlzheimers": 0.10},
            "Multiple Sclerosis": {"Age": (38, 10), "MMSE": (27, 2), "MemoryComplaints": 0.30, "FamilyHistoryAlzheimers": 0.05},
            "Epilepsy": {"Age": (35, 18), "MMSE": (27, 2), "MemoryComplaints": 0.25, "FamilyHistoryAlzheimers": 0.08},
            "ALS": {"Age": (58, 10), "MMSE": (27, 2), "MemoryComplaints": 0.15, "FamilyHistoryAlzheimers": 0.05},
            "Huntington's Disease": {"Age": (45, 12), "MMSE": (22, 5), "MemoryComplaints": 0.60, "FamilyHistoryAlzheimers": 0.05},
        }

        for disease, profile in profiles.items():
            for _ in range(n_per_class):
                row = {}
                for feat in CLASSIFICATION_FEATURES[:14]:
                    if feat in profile:
                        v = profile[feat]
                        if isinstance(v, tuple):
                            row[feat] = float(rng.normal(v[0], v[1]))
                        else:
                            row[feat] = float(rng.binomial(1, v))
                    else:
                        row[feat] = float(rng.normal(5, 2))
                all_rows.append(row)
                all_labels.append(disease)

        df = pd.DataFrame(all_rows)
        df["Age"] = df["Age"].clip(20, 100)
        df["MMSE"] = df["MMSE"].clip(0, 30)
        for col in ["FunctionalAssessment", "ADL", "PhysicalActivity", "SleepQuality"]:
            if col in df.columns:
                df[col] = df[col].clip(0, 10)

        return df, pd.Series(all_labels)

    def train(self) -> None:
        """Train the disease classifier, preferring real data over synthetic."""
        real_data = self._load_real_training_data()

        if real_data is not None:
            X, labels = real_data
            feature_cols = list(X.columns)
            logger.info(
                "DiseaseClassifier: training on real data (%d samples, %d features, %d classes)",
                len(X), len(feature_cols), labels.nunique(),
            )
        else:
            logger.warning("DiseaseClassifier: falling back to synthetic training data")
            X, labels = self.generate_synthetic_training_data(n_per_class=800)
            feature_cols = list(X.columns)

        y = self.le.fit_transform(labels)
        self.feature_names = feature_cols
        self.clf.fit(X.values, y)

        # Log cross-validation score for quality check.
        try:
            cv = StratifiedKFold(n_splits=min(5, len(np.unique(y))), shuffle=True, random_state=42)
            cv_scores = cross_val_score(self.clf, X.values, y, cv=cv, scoring="accuracy")
            logger.info("DiseaseClassifier CV accuracy: %.4f ± %.4f", cv_scores.mean(), cv_scores.std())
        except Exception as e:
            logger.warning("DiseaseClassifier CV failed: %s", e)

        joblib.dump(self.clf, self.models_dir / "disease_clf.pkl")
        joblib.dump(self.le, self.models_dir / "disease_le.pkl")
        joblib.dump(self.feature_names, self.models_dir / "disease_features.pkl")

    def _lazy_load(self) -> None:
        if self.feature_names is not None:
            return
        clf_path = self.models_dir / "disease_clf.pkl"
        le_path = self.models_dir / "disease_le.pkl"
        feat_path = self.models_dir / "disease_features.pkl"

        if not clf_path.exists() or not le_path.exists() or not feat_path.exists():
            logger.warning("DiseaseClassifier artifacts not found, training now...")
            self.train()
            return

        self.feature_names = joblib.load(feat_path)
        self.clf = joblib.load(clf_path)
        self.le = joblib.load(le_path)

    def predict_disease(self, patient_features: dict) -> dict:
        """Predict disease type from patient features.

        Handles feature alignment: uses only the features the classifier
        was trained on, filling missing features with 0.0.
        """
        self._lazy_load()
        assert self.feature_names is not None

        row = {f: float(patient_features.get(f, 0.0)) for f in self.feature_names}
        df = pd.DataFrame([row])

        probs = self.clf.predict_proba(df.values)[0]
        pred_idx = int(np.argmax(probs))
        pred_disease = self.le.inverse_transform([pred_idx])[0]

        all_probs = {
            self.le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probs)
        }

        top_prob = float(probs[pred_idx])
        confidence = "High" if top_prob > 0.6 else "Medium" if top_prob > 0.4 else "Low"

        return {
            "predicted_disease": pred_disease,
            "disease_probabilities": all_probs,
            "confidence": confidence,
        }


class DiseaseClassifierV5:
    """Duck-type compatible wrapper around the v5 CatBoost disease classifier."""

    def __init__(self, clf, le, feature_names: list[str]) -> None:
        self.clf = clf
        self.le = le
        self.feature_names = feature_names

    def predict_disease(self, patient_features: dict) -> dict:
        row = np.array(
            [[float(patient_features.get(f, 0.0)) for f in self.feature_names]],
            dtype=float,
        )
        probs = self.clf.predict_proba(row)[0]
        pred_idx = int(np.argmax(probs))
        pred_disease = self.le.inverse_transform([pred_idx])[0]
        all_probs = {
            self.le.inverse_transform([i])[0]: round(float(p), 4)
            for i, p in enumerate(probs)
        }
        top_prob = float(probs[pred_idx])
        confidence = "High" if top_prob > 0.6 else "Medium" if top_prob > 0.4 else "Low"
        return {
            "predicted_disease": pred_disease,
            "disease_probabilities": all_probs,
            "confidence": confidence,
        }
