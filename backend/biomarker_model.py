# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

try:
    import shap  # type: ignore
except Exception:
    shap = None


class BiomarkerPredictor:
    def __init__(
        self,
        feature_names: list[str],
        models_dir: str | Path = "models",
        enable_lgbm: bool = True,
    ) -> None:
        self.feature_names = feature_names
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # enable_lgbm=False is used for the per-disease models: training LightGBM
        # repeatedly in one process alongside torch segfaults on macOS (clashing
        # OpenMP runtimes). The main predictor keeps it for the AUC headroom.
        self._enable_lgbm = enable_lgbm

        self.rf = RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        self.gb = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
        )
        if XGBClassifier is not None:
            self.third = XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=42,
                eval_metric="logloss",
            )
            self.third_name = "xgboost"
        else:
            self.third = ExtraTreesClassifier(
                n_estimators=300,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            )
            self.third_name = "extra_trees"

        self.lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)

        # LightGBM as an optional 5th base learner — strongest single-model AUC on
        # tabular clinical data, which is what lifts the ensemble past the 0.92 gate.
        if LGBMClassifier is not None and enable_lgbm:
            # n_jobs=1: this predictor is trained repeatedly in one process (main +
            # one per disease), and LightGBM's OpenMP pool deadlocks under repeated
            # fits on macOS. Single-threaded is sub-second on this data anyway.
            self.lgbm = LGBMClassifier(
                n_estimators=600,
                learning_rate=0.03,
                num_leaves=63,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=1,
                random_state=42,
                verbose=-1,
            )
            self.has_lgbm = True
        else:
            self.lgbm = None
            self.has_lgbm = False

        self.tree_explainer: Any | None = None
        self.decision_threshold: float = 0.5
        self._refresh_weights()

    def _refresh_weights(self) -> None:
        """Set ensemble weights for the currently-active base models.

        Order: [rf, gb, third, lr] plus lgbm when present. Tree learners carry
        more weight than the linear model; lgbm shares the top tier with rf/gb.
        """
        if self.has_lgbm and self.lgbm is not None:
            self.weights = np.array([0.24, 0.24, 0.18, 0.10, 0.24], dtype=float)
        else:
            self.weights = np.array([0.35, 0.35, 0.20, 0.10], dtype=float)

    def load_from_disk(self) -> None:
        self.rf = joblib.load(self.models_dir / "rf_model.pkl")
        self.gb = joblib.load(self.models_dir / "gb_model.pkl")

        xgb_path = self.models_dir / "xgboost_model.pkl"
        et_path = self.models_dir / "extra_trees_model.pkl"
        if xgb_path.exists():
            self.third = joblib.load(xgb_path)
            self.third_name = "xgboost"
        elif et_path.exists():
            self.third = joblib.load(et_path)
            self.third_name = "extra_trees"
        else:
            # Neither third-model artifact exists; create an untrained fallback
            # so that the ensemble can still function with RF+GB+LR.
            import logging
            logging.getLogger(__name__).warning(
                "No third model artifact found (xgboost_model.pkl or extra_trees_model.pkl). "
                "Using a fresh ExtraTreesClassifier — predictions may be degraded."
            )
            self.third = ExtraTreesClassifier(
                n_estimators=300, random_state=42, class_weight="balanced", n_jobs=-1,
            )
            self.third_name = "extra_trees"

        lr_path = self.models_dir / "lr_model.pkl"
        if lr_path.exists():
            self.lr = joblib.load(lr_path)

        # LightGBM participates only if enabled for this predictor AND both the
        # library and a trained artifact exist (guards against stale per-disease
        # lgbm artifacts left by older runs).
        lgbm_path = self.models_dir / "lgbm_model.pkl"
        if self._enable_lgbm and LGBMClassifier is not None and lgbm_path.exists():
            self.lgbm = joblib.load(lgbm_path)
            self.has_lgbm = True
        else:
            self.lgbm = None
            self.has_lgbm = False
        self._refresh_weights()

        threshold_path = self.models_dir / "decision_threshold.pkl"
        if threshold_path.exists():
            self.decision_threshold = float(joblib.load(threshold_path))

    @staticmethod
    def _risk_level(prob: float) -> str:
        if prob >= 0.8:
            return "Critical"
        if prob >= 0.65:
            return "High"
        if prob >= 0.4:
            return "Moderate"
        return "Low"

    @staticmethod
    def _confidence(prob: float) -> str:
        margin = abs(prob - 0.5)
        if margin >= 0.3:
            return "High"
        if margin >= 0.15:
            return "Medium"
        return "Low"

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self.rf.fit(X_train, y_train)
        self.gb.fit(X_train, y_train)
        self.third.fit(X_train, y_train)
        self.lr.fit(X_train, y_train)
        if self.has_lgbm and self.lgbm is not None:
            self.lgbm.fit(X_train, y_train)

        if shap is not None:
            self.tree_explainer = shap.TreeExplainer(self.rf)

        probs, _ = self._ensemble_probs(X_train)
        thresholds = np.linspace(0.35, 0.75, 41)
        best_t = 0.5
        best_score = -1.0
        for t in thresholds:
            y_hat = (probs >= t).astype(int)
            score = 0.6 * balanced_accuracy_score(y_train, y_hat) + 0.4 * accuracy_score(y_train, y_hat)
            if score > best_score:
                best_score = float(score)
                best_t = float(t)
        self.decision_threshold = best_t

        joblib.dump(self.rf, self.models_dir / "rf_model.pkl")
        joblib.dump(self.gb, self.models_dir / "gb_model.pkl")
        joblib.dump(self.third, self.models_dir / f"{self.third_name}_model.pkl")
        joblib.dump(self.lr, self.models_dir / "lr_model.pkl")
        if self.has_lgbm and self.lgbm is not None:
            joblib.dump(self.lgbm, self.models_dir / "lgbm_model.pkl")
        joblib.dump(self.decision_threshold, self.models_dir / "decision_threshold.pkl")

    def _ensemble_probs(self, X: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        rf_p = self.rf.predict_proba(X)[:, 1]
        gb_p = self.gb.predict_proba(X)[:, 1]
        third_p = self.third.predict_proba(X)[:, 1]
        lr_p = self.lr.predict_proba(X)[:, 1]

        columns = [rf_p, gb_p, third_p, lr_p]
        per_model = {
            "random_forest": rf_p,
            "gradient_boosting": gb_p,
            self.third_name: third_p,
            "logistic_regression": lr_p,
        }
        if self.has_lgbm and self.lgbm is not None:
            lgbm_p = self.lgbm.predict_proba(X)[:, 1]
            columns.append(lgbm_p)
            per_model["lightgbm"] = lgbm_p

        stacked = np.vstack(columns)
        ensemble = np.average(stacked, axis=0, weights=self.weights)
        return ensemble, per_model

    def get_shap_values(self, X: np.ndarray) -> np.ndarray:
        if shap is None:
            return np.zeros((X.shape[0], X.shape[1]), dtype=float)

        if self.tree_explainer is None:
            self.tree_explainer = shap.TreeExplainer(self.rf)

        shap_values = self.tree_explainer.shap_values(X)
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                return np.asarray(shap_values[1])
            return np.asarray(shap_values[0])

        arr = np.asarray(shap_values)
        # Some SHAP versions return (n_samples, n_features, n_classes) for tree models.
        if arr.ndim == 3:
            if arr.shape[-1] == 2:
                return arr[:, :, 1]
            return arr.mean(axis=-1)
        return arr

    def predict(self, X: np.ndarray) -> dict[str, Any]:
        ensemble, per_model = self._ensemble_probs(X)
        prob = float(np.clip(ensemble[0], 0.0, 1.0))
        pred = int(prob >= self.decision_threshold)

        shap_vals = self.get_shap_values(X[:1])[0]
        top_idx = np.argsort(np.abs(shap_vals))[::-1][:5]
        top_risk_factors = [self.feature_names[i] for i in top_idx]

        return {
            "prediction": pred,
            "probability": round(prob, 4),
            "confidence": self._confidence(prob),
            "risk_level": self._risk_level(prob),
            "individual_model_probs": {
                k: round(float(v[0]), 4) for k, v in per_model.items()
            },
            "top_risk_factors": top_risk_factors,
        }

    def get_feature_importance(self) -> dict[str, float]:
        importances = []
        tree_models = [self.rf, self.gb, self.third]
        if self.has_lgbm and self.lgbm is not None:
            tree_models.append(self.lgbm)
        for model in tree_models:
            if hasattr(model, "feature_importances_"):
                imp = np.asarray(model.feature_importances_, dtype=float)
                total = imp.sum()
                if total > 0:  # lgbm uses raw split counts; normalize so scales match
                    imp = imp / total
                importances.append(imp)
        if not importances:
            return {name: 0.0 for name in self.feature_names}

        avg = np.mean(np.vstack(importances), axis=0)
        ranking = sorted(
            [(name, float(score)) for name, score in zip(self.feature_names, avg)],
            key=lambda x: x[1],
            reverse=True,
        )
        return {k: round(v, 6) for k, v in ranking}

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
        ensemble, _ = self._ensemble_probs(X_test)
        y_pred = (ensemble >= self.decision_threshold).astype(int)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
            "roc_auc": round(float(roc_auc_score(y_test, ensemble)), 4),
            "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            "decision_threshold": round(float(self.decision_threshold), 4),
        }


class MultiDiseasePredictor:
    def __init__(self, feature_names: list[str], diseases: list[str], models_dir: str | Path = "models/multi") -> None:
        self.feature_names = feature_names
        self.diseases = diseases
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.predictors: dict[str, BiomarkerPredictor] = {}

        for disease in diseases:
            disease_dir = self.models_dir / self._slug(disease)
            disease_dir.mkdir(parents=True, exist_ok=True)
            # enable_lgbm=False: avoids the repeated-LightGBM OpenMP segfault on macOS
            # (the 4-model ensemble is sufficient for secondary per-disease scores).
            self.predictors[disease] = BiomarkerPredictor(
                feature_names=feature_names, models_dir=disease_dir, enable_lgbm=False
            )

    @staticmethod
    def _slug(name: str) -> str:
        return (
            name.lower()
            .replace("'", "")
            .replace(" ", "_")
            .replace("-", "_")
        )

    def train_all(self, splits: dict[str, tuple[Any, Any, Any, Any]]) -> None:
        for disease, predictor in self.predictors.items():
            if disease not in splits:
                continue
            X_train, _X_test, y_train, _y_test = splits[disease]
            predictor.train(X_train.values, y_train.values)

    def load_from_disk(self) -> None:
        loaded: dict[str, BiomarkerPredictor] = {}
        for disease, predictor in self.predictors.items():
            try:
                predictor.load_from_disk()
                loaded[disease] = predictor
            except Exception:
                continue
        self.predictors = loaded

    def predict_for_disease(self, disease: str, X: np.ndarray) -> dict[str, Any]:
        predictor = self.predictors.get(disease)
        if predictor is None:
            raise KeyError(f"Unknown disease model: {disease}")
        return predictor.predict(X)

    def predict_all(self, X: np.ndarray) -> dict[str, float]:
        risk: dict[str, float] = {}
        for disease, predictor in self.predictors.items():
            pred = predictor.predict(X)
            risk[disease] = float(pred["probability"])
        return risk
