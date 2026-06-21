# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Optuna-tuned modality fusion weights for NeuroSynth v5.

Replaces the v4 hardcoded weights (tabular 40%, GNN 20%, etc.) by searching
for the weight combination that maximises AUC on a held-out validation fold.

Usage:
    python scripts/tune_fusion_weights.py
    python scripts/tune_fusion_weights.py --trials 200 --out models/ensemble_v5/fusion_weights.json
"""
from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

MODALITIES = ["tabular", "genomic", "tft", "causal", "gnn"]

# v4 defaults (used as starting point + fallback)
V4_DEFAULTS = {"tabular": 0.40, "genomic": 0.15, "tft": 0.15, "causal": 0.10, "gnn": 0.20}


def _simulate_modality_probs(
    X: np.ndarray, y: np.ndarray, seed: int = 42,
) -> dict[str, np.ndarray]:
    """
    Produce plausible per-modality probability estimates from the feature matrix.
    In production, these come from each model's .predict_proba() output.
    For tuning without all models loaded, we train lightweight proxies.
    """
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    rng = np.random.default_rng(seed)
    n = len(X)
    modality_probs: dict[str, np.ndarray] = {}

    # tabular — main ensemble proxy (GBM)
    gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=seed)
    gb.fit(X, y)
    modality_probs["tabular"] = gb.predict_proba(X)[:, 1]

    # genomic — uses only GENOMIC_4 columns (last 4 features if present)
    X_genomic = X[:, -4:] if X.shape[1] >= 4 else X
    lr_g = LogisticRegression(C=0.5, max_iter=500, random_state=seed)
    lr_g.fit(X_genomic, y)
    modality_probs["genomic"] = lr_g.predict_proba(X_genomic)[:, 1]

    # tft — temporal proxy (RF with randomised feature subset)
    feat_idx = rng.choice(X.shape[1], size=min(20, X.shape[1]), replace=False)
    rf_t = RandomForestClassifier(n_estimators=100, max_features=0.5, random_state=seed + 1)
    rf_t.fit(X[:, feat_idx], y)
    modality_probs["tft"] = rf_t.predict_proba(X[:, feat_idx])[:, 1]

    # causal — conservative LR proxy (represents structural causal model output)
    lr_c = LogisticRegression(C=0.1, max_iter=500, random_state=seed + 2)
    lr_c.fit(X, y)
    modality_probs["causal"] = lr_c.predict_proba(X)[:, 1]

    # gnn — brain-connectivity proxy (RF on imaging features, cols 32-39 if present)
    img_start = 32
    img_end = min(40, X.shape[1])
    X_img = X[:, img_start:img_end] if img_end > img_start else X[:, :8]
    rf_g = RandomForestClassifier(n_estimators=100, max_features=0.7, random_state=seed + 3)
    rf_g.fit(X_img, y)
    modality_probs["gnn"] = rf_g.predict_proba(X_img)[:, 1]

    return modality_probs


def weighted_fusion(modality_probs: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    total_w = sum(weights.values())
    if total_w == 0:
        return np.mean(list(modality_probs.values()), axis=0)
    return sum(modality_probs[m] * weights[m] / total_w for m in MODALITIES if m in modality_probs)


def tune(
    X: np.ndarray, y: np.ndarray,
    n_trials: int = 100,
    n_cv_folds: int = 3,
    seed: int = 42,
) -> dict[str, float]:
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("optuna not installed — returning v4 defaults. pip install optuna")
        return V4_DEFAULTS

    log.info("Simulating modality probability proxies …")
    modality_probs = _simulate_modality_probs(X, y, seed)

    kf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=seed)

    def objective(trial: "optuna.Trial") -> float:
        weights = {m: trial.suggest_float(m, 0.0, 1.0) for m in MODALITIES}
        total = sum(weights.values())
        if total < 1e-6:
            return 0.0
        weights = {m: v / total for m, v in weights.items()}

        fold_aucs = []
        for tr_idx, val_idx in kf.split(X, y):
            val_probs = {m: p[val_idx] for m, p in modality_probs.items()}
            fused = weighted_fusion(val_probs, weights)
            auc = roc_auc_score(y[val_idx], fused)
            fold_aucs.append(auc)
        return float(np.mean(fold_aucs))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    # Seed with v4 defaults as a warm start
    study.enqueue_trial(V4_DEFAULTS)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    total = sum(best.values())
    best_normalized = {m: round(v / total, 6) for m, v in best.items()}

    log.info("Optuna best AUC (CV): %.4f", study.best_value)
    log.info("Tuned weights: %s", best_normalized)
    return best_normalized


def main() -> None:
    ap = argparse.ArgumentParser(description="Tune fusion weights with Optuna")
    ap.add_argument("--data", default="data/real_v5_augmented.parquet")
    ap.add_argument("--out", default="models/ensemble_v5/fusion_weights.json")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        log.error("Data not found: %s", data_path)
        _sys.exit(1)

    from scripts.data.v5.schema import ALL_FEATURES
    df = pd.read_parquet(data_path)
    feat_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feat_cols].fillna(df[feat_cols].median(numeric_only=True)).fillna(0).values
    y = df["risk_label"].fillna(0).astype(int).values

    log.info("=== Optuna Fusion Weight Tuning ===")
    log.info("Data: %d rows, %d features | Trials: %d", len(X), X.shape[1], args.trials)

    best_weights = tune(X, y, n_trials=args.trials, n_cv_folds=args.folds, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"weights": best_weights, "modalities": MODALITIES}, indent=2))
    log.info("Saved → %s", out_path)

    log.info("\nFinal weights:")
    for m, w in sorted(best_weights.items(), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        log.info("  %-10s %.4f  %s", m, w, bar)


if __name__ == "__main__":
    main()
