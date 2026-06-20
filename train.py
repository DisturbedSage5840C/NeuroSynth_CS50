#!/usr/bin/env python3
"""NeuroSynth v4 — unified training entry point.

Trains the full production model stack (calibrated ensemble, per-disease models,
temporal progression, disease classifier, causal graph) by delegating to the
canonical ``scripts.pretrain.run_pretrain`` routine, then optionally enforces the
AUC release gate.

This replaces the v1 stub that trained a lone RandomForest on an unrelated
``oasis_longitudinal.csv`` — that file was never wired into the backend.

Usage:
    python train.py                                  # auto-detect data source
    python train.py --data data/realistic_v4.parquet
    python train.py --validate                       # enforce AUC >= 0.92 gate
    python train.py --data data/oasis3.parquet --validate --models-dir models/
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Cap native thread pools before numpy/sklearn/lightgbm import — prevents the
# OpenMP/BLAS oversubscription deadlock seen when many models are fit in sequence
# on macOS. Set before heavy imports to take effect.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
# Allow multiple OpenMP runtimes to coexist (LightGBM's libomp vs sklearn/numpy)
# — without this, repeated fits segfault on macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Resolution order mirrors backend.data_pipeline.DataPipeline.DATA_CANDIDATES.
DATA_CANDIDATES = (
    "data/realistic_v4.parquet",
    "data/realistic_v4.csv",
    "neurological_disease_data.csv",
    "alzheimers_disease_data.csv",
)

AUC_GATE = 0.92


def main() -> int:
    parser = argparse.ArgumentParser(description="NeuroSynth v4 training pipeline")
    parser.add_argument("--data", default=None, help="Path to training CSV or parquet")
    parser.add_argument("--models-dir", default="models", help="Output directory for artifacts")
    parser.add_argument("--validate", action="store_true", help="Enforce AUC >= 0.92 gate")
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    args = parser.parse_args()

    def log(msg: str) -> None:
        if not args.quiet:
            print(f"[train] {msg}")

    # 1. Resolve data source
    candidates = [args.data, *DATA_CANDIDATES]
    data_path = next((c for c in candidates if c and Path(c).exists()), None)
    if data_path is None:
        print("ERROR: No training dataset found.", file=sys.stderr)
        print("Run: python scripts/data/build_realistic_synthetic.py", file=sys.stderr)
        return 1
    log(f"dataset     : {data_path}")

    # 2. Train the full stack via the canonical pretrain routine.
    #    Use joblib's threading backend: it keeps RF/ExtraTrees parallel (their
    #    Cython releases the GIL) while avoiding the loky multiprocessing pools
    #    that deadlock on macOS when many n_jobs=-1 fits run back-to-back.
    import joblib

    from scripts.pretrain import run_pretrain

    with joblib.parallel_backend("threading"):
        manifest = run_pretrain(dataset_path=Path(data_path), models_dir=Path(args.models_dir))
    metrics = manifest.get("metrics", {}) or {}
    auc = float(metrics.get("roc_auc", 0.0))
    f1 = float(metrics.get("f1_weighted", 0.0))
    acc = float(metrics.get("accuracy", 0.0))
    log(f"features    : {len(manifest.get('feature_schema', []))}")
    log(f"ensemble AUC: {auc:.4f}  F1: {f1:.4f}  Acc: {acc:.4f}")
    log(f"manifest    : {Path(args.models_dir) / 'model_manifest.json'}")

    # 3. AUC release gate
    if args.validate:
        if auc >= AUC_GATE:
            log(f"gate        : PASS  AUC {auc:.4f} >= {AUC_GATE}")
        else:
            log(f"gate        : FAIL  AUC {auc:.4f} < {AUC_GATE} — model NOT promoted")
            return 2  # non-zero so CI/drift-retrain can detect a non-promotable model

    log(f"artifacts   : {args.models_dir}/")
    log("run API     : uvicorn backend.api:app --host 0.0.0.0 --port 8888")
    return 0


if __name__ == "__main__":
    sys.exit(main())
