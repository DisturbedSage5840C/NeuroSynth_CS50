# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import joblib
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# V5 adapter — bridges old 32-feature scaled input → v5 56-feature unscaled
# ---------------------------------------------------------------------------

class _V5PredictorAdapter:
    """Wraps CalibratedEnsemble so it accepts the old 32-feature scaled X.

    predictions.py builds X by scaling the patient feature dict through the
    old StandardScaler (trained on 32 features). The v5 ensemble expects 56
    unscaled features. This adapter:
      1. Inverse-transforms the 32 scaled features back to original scale.
      2. Maps each feature into the correct index in the 56-feature v5 space.
      3. Calls CalibratedEnsemble.predict() on the expanded X.
    SHAP values are projected back to 32-feature space so the existing
    predictions.py SHAP extraction continues to work unchanged.
    """

    def __init__(
        self,
        ensemble,
        old_scaler,
        old_feature_names: list[str],
        v5_feature_names: list[str],
    ) -> None:
        self._ensemble = ensemble
        self._old_scaler = old_scaler
        self._v5_len = len(v5_feature_names)
        # Map: old_feature_index → v5_feature_index
        v5_idx_map: dict[str, int] = {n: i for i, n in enumerate(v5_feature_names)}
        self._idx_map: dict[int, int] = {
            old_i: v5_idx_map[name]
            for old_i, name in enumerate(old_feature_names)
            if name in v5_idx_map
        }

    def _to_v5(self, X_scaled: np.ndarray) -> np.ndarray:
        X_orig = self._old_scaler.inverse_transform(X_scaled)
        X_v5 = np.zeros((X_orig.shape[0], self._v5_len), dtype=float)
        for old_i, v5_i in self._idx_map.items():
            X_v5[:, v5_i] = X_orig[:, old_i]
        return X_v5

    def predict(self, X_scaled: np.ndarray) -> dict[str, Any]:
        return self._ensemble.predict(self._to_v5(X_scaled))

    def get_shap_values(self, X_scaled: np.ndarray) -> np.ndarray:
        v5_shap = self._ensemble.get_shap_values(self._to_v5(X_scaled))
        n_old = self._old_scaler.n_features_in_
        shap_out = np.zeros((X_scaled.shape[0], n_old), dtype=float)
        for old_i, v5_i in self._idx_map.items():
            shap_out[:, old_i] = v5_shap[:, v5_i]
        return shap_out


class ModelRegistry:
    def __init__(self, models_dir: str | Path = "models") -> None:
        self.models_dir = Path(models_dir)

    def _load_feature_names(self, scaler: Any) -> list[str]:
        names = getattr(scaler, "feature_names_in_", None)
        if names is None:
            return []
        return [str(n) for n in names]

    # ------------------------------------------------------------------
    # v5 loading helpers
    # ------------------------------------------------------------------

    def _load_v5_ensemble(self, old_scaler, old_feature_names: list[str]):
        """Try to load v5 CalibratedEnsemble + disease classifier.

        Returns (_V5PredictorAdapter, DiseaseClassifierV5) or (None, None).
        """
        v5_dir = self.models_dir / "ensemble_v5"
        if not (v5_dir / "model_manifest_v5.json").exists():
            return None, None

        root = Path(__file__).resolve().parents[1]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        try:
            import types as _t
            _pkg = _t.ModuleType("neurosynth")
            _pkg.__path__ = [str(root / "src" / "neurosynth")]
            sys.modules.setdefault("neurosynth", _pkg)

            from src.neurosynth.models.calibrated_ensemble import CalibratedEnsemble
            from scripts.data.v5.schema import ALL_FEATURES

            v5_feature_names = list(ALL_FEATURES)
            ensemble = CalibratedEnsemble(
                feature_names=v5_feature_names,
                models_dir=v5_dir / "ensemble",
                n_cv_folds=5,
                enable_tabnet=False,
            )
            ensemble.load_from_disk()

            adapter = _V5PredictorAdapter(
                ensemble, old_scaler, old_feature_names, v5_feature_names
            )

            # v5 CatBoost disease classifier
            disease_clf_v5 = None
            clf_path = v5_dir / "disease_classifier_v5.pkl"
            le_path = v5_dir / "disease_label_encoder_v5.pkl"
            if clf_path.exists() and le_path.exists():
                from backend.disease_classifier import DiseaseClassifierV5
                disease_clf_v5 = DiseaseClassifierV5(
                    joblib.load(clf_path),
                    joblib.load(le_path),
                    v5_feature_names,
                )

            logger.info("v5_ensemble_loaded models_dir=%s", v5_dir)
            return adapter, disease_clf_v5

        except Exception as exc:
            import traceback as _tb
            logger.warning("v5_ensemble_load_failed error=%s trace=%s", exc, _tb.format_exc())
            # Re-raise so the caller can surface the actual failure reason instead of
            # silently falling back to the legacy path (which requires rf_model.pkl).
            raise

    # ------------------------------------------------------------------

    def load_all(self) -> SimpleNamespace:
        from backend.biomarker_model import BiomarkerPredictor, MultiDiseasePredictor
        from backend.disease_classifier import DISEASES, DiseaseClassifier

        # torch-dependent modules — skip gracefully when torch is not installed
        # (e.g. Render free-tier deploy uses requirements-deploy.txt without torch)
        try:
            from backend.causal_engine import NeuralCausalDiscovery as _NCD
        except Exception:
            _NCD = None  # type: ignore[assignment,misc]

        try:
            from backend.temporal_model import TemporalProgressionModel as _TPM
        except Exception:
            _TPM = None  # type: ignore[assignment,misc]

        scaler = joblib.load(self.models_dir / "scaler.pkl")
        feature_names = self._load_feature_names(scaler)

        # ---- Try v5 ensemble first ----
        v5_predictor, v5_disease_clf = self._load_v5_ensemble(scaler, feature_names)

        if v5_predictor is not None:
            predictor = v5_predictor
            logger.info("Using v5 CalibratedEnsemble (CatBoost+LightGBM+RF+GB+LR) as primary predictor")
        else:
            # Fall back to legacy BiomarkerPredictor
            rf_path = self.models_dir / "rf_model.pkl"
            if not rf_path.exists():
                # Expose diagnostic: what IS in models_dir so we can debug tarball extraction
                try:
                    present = sorted(str(p) for p in self.models_dir.rglob("*") if p.is_file())
                except Exception:
                    present = ["<error listing>"]
                raise FileNotFoundError(
                    f"v5 ensemble failed (see v5_ensemble_load_failed log) and legacy "
                    f"rf_model.pkl missing at {rf_path}. "
                    f"Files in models_dir: {present}"
                )
            predictor = BiomarkerPredictor(feature_names)
            predictor.rf = joblib.load(rf_path)
            predictor.gb = joblib.load(self.models_dir / "gb_model.pkl")

            third_model = self.models_dir / "xgboost_model.pkl"
            if third_model.exists():
                predictor.third = joblib.load(third_model)
                predictor.third_name = "xgboost"
            else:
                predictor.third = joblib.load(self.models_dir / "extra_trees_model.pkl")
                predictor.third_name = "extra_trees"

            lr_model = self.models_dir / "lr_model.pkl"
            if lr_model.exists():
                predictor.lr = joblib.load(lr_model)

            lgbm_model = self.models_dir / "lgbm_model.pkl"
            if predictor.has_lgbm and lgbm_model.exists():
                predictor.lgbm = joblib.load(lgbm_model)
            else:
                predictor.lgbm = None
                predictor.has_lgbm = False
            predictor._refresh_weights()

        try:
            if _TPM is None:
                raise ImportError("torch not available")
            temporal = _TPM(feature_names)
            import torch
            lstm_state = torch.load(
                self.models_dir / "lstm_model.pt",
                map_location="cpu",
                weights_only=True,
            )
            temporal.model.load_state_dict(lstm_state)
        except Exception as _te:
            logger.warning("temporal_model_load_skipped reason=%s", _te)
            temporal = None

        variables = None
        vars_file = self.models_dir / "causal_vars.json"
        if vars_file.exists():
            variables = json.loads(vars_file.read_text(encoding="utf-8"))

        try:
            if _NCD is None:
                raise ImportError("torch not available")
            causal_model = _NCD(variables=variables)
            causal_path = self.models_dir / "causal_graph.npy"
            if causal_path.exists():
                causal_model.latest_W = np.load(causal_path)
        except Exception as _ce:
            logger.warning("causal_model_load_skipped reason=%s", _ce)
            causal_model = None

        if v5_disease_clf is not None:
            disease_clf = v5_disease_clf
            logger.info("Using v5 CatBoost disease classifier")
        else:
            disease_clf = DiseaseClassifier(models_dir=self.models_dir)
            disease_clf._lazy_load()

        multi_predictor = MultiDiseasePredictor(
            feature_names=feature_names,
            diseases=DISEASES,
            models_dir=self.models_dir / "multi",
        )
        try:
            multi_predictor.load_from_disk()
        except Exception as e:
            logger.warning("Failed to load multi-disease predictor: %s", e)
            multi_predictor = None

        manifest = {}
        manifest_path = self.models_dir / "model_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Merge v5 metrics into manifest so /performance shows updated numbers
        v5_manifest_path = self.models_dir / "ensemble_v5" / "model_manifest_v5.json"
        if v5_manifest_path.exists():
            try:
                v5_manifest = json.loads(v5_manifest_path.read_text(encoding="utf-8"))
                manifest["v5"] = v5_manifest
                manifest.setdefault("metrics", {}).update(
                    v5_manifest.get("binary_metrics", {})
                )
            except Exception:
                pass

        return SimpleNamespace(
            scaler=scaler,
            predictor=predictor,
            temporal=temporal,
            causal=causal_model,
            disease_classifier=disease_clf,
            multi_predictor=multi_predictor,
            feature_names=feature_names,
            manifest=manifest,
            dataset_stats=manifest.get("dataset_stats", {}),
        )
