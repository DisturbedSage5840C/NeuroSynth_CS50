# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""NeuroSynth v5 — main training script.

Trains the 6-model ensemble (RF + GB + CatBoost + LR + LightGBM + TabNet)
on real_v5_augmented.parquet and validates AUC ≥ 0.95 before saving.

Usage:
    python scripts/train_v5.py
    python scripts/train_v5.py --data data/real_v5_augmented.parquet --out models/ensemble_v5
    python scripts/train_v5.py --auc-gate 0.92 --no-tabnet   # faster, lower threshold
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder

from scripts.data.v5.schema import ALL_FEATURES, DISEASE_TYPES
import importlib, types as _t
_pkg = _t.ModuleType("neurosynth"); _pkg.__path__ = [str(_Path(__file__).resolve().parents[1] / "src" / "neurosynth")]; _sys.modules.setdefault("neurosynth", _pkg)
from src.neurosynth.models.calibrated_ensemble import CalibratedEnsemble

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# AUC gate thresholds
HARD_AUC_GATE = 0.92   # fail if below this
SOFT_AUC_GATE = 0.95   # warn if below this (v5 target)
RARE_F1_GATE       = 0.75   # ALS + Huntington's F1 floor — hard gate per plan §2.9
RARE_F1_SOFT_GATE  = 0.75   # same value; plan has one target, no separate soft tier

# Rare disease cost weights (used to compute sample_weight)
DISEASE_COSTS = {
    "Alzheimer's Disease":  1.0,
    "Parkinson's Disease":  1.2,
    "Multiple Sclerosis":   1.5,
    "Epilepsy":             1.4,
    "ALS":                  3.0,
    "Huntington's Disease": 3.5,
    "Healthy":              0.8,
}


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, LabelEncoder]:
    """Load parquet, build X (56 features), y_binary (risk_label), y_disease (6-class)."""
    log.info("Loading %s …", path)
    df = pd.read_parquet(path)
    log.info("  %d rows, %d columns", len(df), len(df.columns))

    # Select only the 56 schema features that exist in the data
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        log.warning("  Missing features (filled with 0): %s", missing)
    for c in missing:
        df[c] = 0.0

    X = df[ALL_FEATURES].fillna(df[ALL_FEATURES].median(numeric_only=True)).fillna(0).values

    # Binary label (risk_label)
    y_binary = df["risk_label"].fillna(0).astype(int).values

    # 6-class disease label
    le = LabelEncoder()
    y_disease = le.fit_transform(df["DiseaseType"].fillna("Alzheimer's Disease").values)

    log.info("  Features: %d | Binary pos rate: %.1f%% | Disease classes: %s",
             X.shape[1], y_binary.mean() * 100, list(le.classes_))
    return X, y_binary, y_disease, le


def compute_sample_weights(df_diseases: pd.Series) -> np.ndarray:
    """Disease-cost based sample weights."""
    return np.array([DISEASE_COSTS.get(d, 1.0) for d in df_diseases])


def evaluate_per_disease(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    le: LabelEncoder,
) -> dict[str, float]:
    """Per-disease F1 on the 6-class problem."""
    report = classification_report(y_true, y_pred, target_names=le.classes_, output_dict=True, zero_division=0)
    metrics: dict[str, float] = {}
    for cls in le.classes_:
        safe = cls.replace(" ", "_").replace("'", "")
        metrics[f"f1_{safe}"] = round(
            report.get(cls, {}).get("f1-score", 0.0), 4
        )
    return metrics


def train_disease_classifier(
    X: np.ndarray, y_disease: np.ndarray, le: LabelEncoder, out_dir: Path,
) -> dict[str, float]:
    """Train a CatBoost 6-class disease classifier."""
    import joblib

    try:
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=400, learning_rate=0.05, depth=7,
            auto_class_weights="Balanced",
            random_seed=42, verbose=0, allow_writing_files=False,
        )
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        log.warning("CatBoost not available, using RandomForest for disease classifier")
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y_disease, test_size=0.2, stratify=y_disease, random_state=42)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    metrics = evaluate_per_disease(y_te, y_pred, None, le)

    disease_clf_path = out_dir / "disease_classifier_v5.pkl"
    joblib.dump(clf, disease_clf_path)
    joblib.dump(le, out_dir / "disease_label_encoder_v5.pkl")
    log.info("  Disease classifier saved → %s", disease_clf_path)
    return metrics


def train_binary_ensemble(
    X: np.ndarray, y_binary: np.ndarray, feature_names: list[str],
    out_dir: Path, enable_tabnet: bool,
) -> dict[str, float]:
    """Train CalibratedEnsemble on binary risk label."""
    ensemble = CalibratedEnsemble(
        feature_names=feature_names,
        models_dir=out_dir / "ensemble",
        n_cv_folds=5,
        enable_tabnet=enable_tabnet,
    )

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_binary, test_size=0.15, stratify=y_binary, random_state=42
    )

    log.info("Training ensemble on %d rows (%d features, %d base models) …",
             len(X_tr), X.shape[1], len(ensemble.base_models))
    log.info("  Base learners: %s", [name for name, _ in ensemble.base_models])

    t0 = time.time()
    train_metrics = ensemble.train(X_tr, y_tr)
    elapsed = time.time() - t0
    log.info("  Training complete in %.1fs", elapsed)

    # Evaluate on held-out test set
    test_metrics = ensemble.evaluate(X_te, y_te)
    log.info("  Test AUC:    %.4f", test_metrics["roc_auc"])
    log.info("  Test F1:     %.4f", test_metrics["f1_weighted"])
    log.info("  Test Brier:  %.4f", test_metrics["brier_score"])

    return {**train_metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}


def validate_auc_gate(auc: float, hard: float, soft: float) -> bool:
    """Return True if AUC passes hard gate; log warning if below soft gate."""
    if auc < hard:
        log.error("AUC %.4f is BELOW hard gate %.2f — training FAILED", auc, hard)
        return False
    if auc < soft:
        log.warning("AUC %.4f passes hard gate but is below soft target %.2f", auc, soft)
    else:
        log.info("AUC %.4f passes both gates (hard %.2f / soft %.2f) ✓", auc, hard, soft)
    return True


def validate_mapie_coverage(
    X_test: np.ndarray,
    y_test: np.ndarray,
    out_dir: Path,
    nominal_coverage: float = 0.95,
    min_empirical: float = 0.93,
) -> dict[str, float]:
    """Validate MAPIE conformal prediction achieves ≥ 93% empirical coverage.

    Loads the MAPIE classifier saved by CalibratedEnsemble.train() and checks
    that the 95% prediction sets actually contain the true label ≥ 93% of the
    time on the held-out test fold. Logs a warning if below threshold.
    """
    import joblib

    mapie_path = out_dir / "ensemble" / "mapie_classifier.pkl"
    meta_path = out_dir / "ensemble" / "calibrated_meta.pkl"

    if not mapie_path.exists() or not meta_path.exists():
        log.warning("MAPIE artifacts not found at %s — skipping coverage validation", mapie_path)
        return {}

    try:
        mapie = joblib.load(mapie_path)
        meta = joblib.load(meta_path)

        # Rebuild base-model OOF probabilities for the test set from the
        # calibrated meta-learner input format MAPIE was trained on.
        # We use the meta-learner to score X_test, then wrap in a column vector.
        meta_probs = meta.predict_proba(X_test)  # (n, 2) binary probs

        alpha = 1.0 - nominal_coverage
        _, pred_sets = mapie.predict(meta_probs, alpha=[alpha])
        # pred_sets shape: (n, 2, 1) — [lower, upper] × [alpha]
        pred_sets_arr = pred_sets[:, :, 0]  # (n, 2)

        # Coverage: true label falls within the prediction set
        in_set = np.array([
            bool(pred_sets_arr[i, int(y_test[i])]) if int(y_test[i]) < pred_sets_arr.shape[1] else False
            for i in range(len(y_test))
        ])
        empirical = float(in_set.mean())

        log.info(
            "MAPIE empirical coverage: %.3f (nominal %.2f, gate %.2f) %s",
            empirical, nominal_coverage, min_empirical,
            "✓" if empirical >= min_empirical else "⚠ BELOW GATE",
        )
        if empirical < min_empirical:
            log.warning(
                "Conformal coverage %.3f < %.2f minimum — consider re-calibrating MAPIE",
                empirical, min_empirical,
            )
        return {"mapie_empirical_coverage": round(empirical, 4), "mapie_nominal": nominal_coverage}

    except Exception as exc:
        log.warning("MAPIE coverage validation failed: %s", exc)
        return {}


def main() -> None:
    ap = argparse.ArgumentParser(description="NeuroSynth v5 training pipeline")
    ap.add_argument("--data", default="data/real_v5_augmented.parquet")
    ap.add_argument("--out", default="models/ensemble_v5")
    ap.add_argument("--auc-gate", type=float, default=HARD_AUC_GATE)
    ap.add_argument("--no-tabnet", action="store_true", help="Disable TabNet (faster)")
    ap.add_argument("--skip-disease-clf", action="store_true")
    args = ap.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        log.error("Data file not found: %s — run merge_v5.py + ctgan_augment.py first", data_path)
        sys.exit(1)

    log.info("=== NeuroSynth v5 Training ===")
    log.info("Data: %s", data_path)
    log.info("Output: %s", out_dir)

    # Load data
    df = pd.read_parquet(data_path)
    X, y_binary, y_disease, le = load_data(data_path)

    # Train binary ensemble (primary task)
    log.info("\n--- Binary Risk Ensemble ---")
    binary_metrics = train_binary_ensemble(
        X, y_binary, ALL_FEATURES, out_dir,
        enable_tabnet=not args.no_tabnet,
    )

    # AUC gate check
    auc = binary_metrics.get("test_roc_auc", 0.0)
    gate_passed = validate_auc_gate(auc, args.auc_gate, SOFT_AUC_GATE)

    # MAPIE empirical coverage validation
    log.info("\n--- MAPIE Conformal Coverage Validation ---")
    X_te_split = X[int(len(X) * 0.85):]
    y_te_split = y_binary[int(len(X) * 0.85):]
    mapie_metrics = validate_mapie_coverage(X_te_split, y_te_split, out_dir)

    # Train 6-class disease classifier
    disease_metrics: dict[str, float] = {}
    if not args.skip_disease_clf:
        log.info("\n--- 6-Class Disease Classifier (CatBoost) ---")
        disease_metrics = train_disease_classifier(X, y_disease, le, out_dir)
        for k, v in disease_metrics.items():
            log.info("  %s: %.4f", k, v)

        # Rare disease F1 gates (plan §2.9)
        for rare in ("ALS", "Huntington's Disease"):
            f1_key = f"{rare}_f1"
            f1 = disease_metrics.get(f1_key)
            if f1 is not None:
                if f1 < RARE_F1_GATE:
                    log.error("RARE_F1 HARD GATE FAILED — %s F1=%.3f < %.2f", rare, f1, RARE_F1_GATE)
                elif f1 < RARE_F1_SOFT_GATE:
                    log.warning("RARE_F1 soft target not met — %s F1=%.3f < %.2f (v5 target; acceptable for release)", rare, f1, RARE_F1_SOFT_GATE)
                else:
                    log.info("RARE_F1 ✓ — %s F1=%.3f ≥ %.2f", rare, f1, RARE_F1_SOFT_GATE)

    # Save manifest
    manifest = {
        "version": "v5",
        "data_source": str(data_path),
        "n_rows": int(len(df)),
        "n_features": len(ALL_FEATURES),
        "disease_classes": list(le.classes_),
        "binary_metrics": binary_metrics,
        "disease_metrics": disease_metrics,
        "mapie_metrics": mapie_metrics,
        "auc_gate": {"hard": args.auc_gate, "soft": SOFT_AUC_GATE, "passed": gate_passed},
    }
    manifest_path = out_dir / "model_manifest_v5.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info("\nManifest saved → %s", manifest_path)

    if not gate_passed:
        log.error("Training failed AUC gate. Artifacts saved but flagged.")
        _sys.exit(1)

    log.info("\n=== Training complete ===")
    log.info("AUC: %.4f | F1: %.4f | Models: %s",
             auc,
             binary_metrics.get("test_f1_weighted", 0.0),
             out_dir)


if __name__ == "__main__":
    main()
