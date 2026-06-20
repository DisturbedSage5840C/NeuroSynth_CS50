from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from backend.biomarker_model import BiomarkerPredictor, MultiDiseasePredictor
from backend.causal_engine import DISEASE_VARIABLES, NeuralCausalDiscovery
from backend.data_pipeline import DataPipeline
from backend.disease_classifier import DISEASES, DiseaseClassifier
from backend.temporal_model import TemporalProgressionModel


def dataset_md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_baseline_profile(pipeline: DataPipeline, disease_clf: DiseaseClassifier) -> dict[str, float]:
    profile: dict[str, float] = {}
    if pipeline.df_raw is None or not disease_clf.feature_names:
        return profile

    for fname in disease_clf.feature_names:
        if fname not in pipeline.df_raw.columns:
            continue
        series = pipeline.df_raw[fname]
        if str(series.dtype) in {"object", "string"}:
            profile[fname] = float(series.astype(str).str.len().median())
        else:
            profile[fname] = float(series.fillna(series.median()).median())
    return profile


def run_pretrain(dataset_path: Path, models_dir: Path) -> dict[str, object]:
    models_dir.mkdir(parents=True, exist_ok=True)

    pipeline = DataPipeline(csv_path=str(dataset_path), models_dir=models_dir)
    X_train, X_test, y_train, y_test, feature_names, _scaler, dataset_stats = pipeline.process()

    predictor = BiomarkerPredictor(feature_names=feature_names, models_dir=models_dir)
    predictor.train(X_train.values, y_train.values)
    metrics = predictor.evaluate(X_test.values, y_test.values)

    multi = MultiDiseasePredictor(feature_names=feature_names, diseases=DISEASES, models_dir=models_dir / "multi")
    multi.train_all(pipeline.split_by_disease())

    temporal = TemporalProgressionModel(feature_names=feature_names, models_dir=models_dir)
    temporal.train_model(X_train.values, y_train.values)

    disease_clf = DiseaseClassifier(models_dir=models_dir)
    disease_clf.train()

    predicted_disease = "Alzheimer's Disease"
    baseline_profile = _build_baseline_profile(pipeline, disease_clf)
    if baseline_profile:
        predicted_disease = disease_clf.predict_disease(baseline_profile).get("predicted_disease", predicted_disease)

    causal_vars = DISEASE_VARIABLES.get(predicted_disease, DISEASE_VARIABLES["Alzheimer's Disease"])
    causal_model = NeuralCausalDiscovery(models_dir=models_dir, variables=causal_vars)
    if pipeline.df_processed is not None:
        causal_cols = [c for c in causal_model.variables if c in pipeline.df_processed.columns]
        if len(causal_cols) == len(causal_model.variables):
            causal_model.fit(
                pipeline.df_processed[causal_model.variables].values.astype(float),
                outer_iters=2,
                inner_iters=20,
            )
        else:
            zeros = np.zeros((len(causal_model.variables), len(causal_model.variables)), dtype=float)
            np.save(models_dir / "causal_graph.npy", zeros)
            (models_dir / "causal_vars.json").write_text(json.dumps(causal_model.variables, indent=2), encoding="utf-8")

    manifest = {
        "trained_at": datetime.now(tz=UTC).isoformat(),
        "dataset_path": str(dataset_path),
        "dataset_md5": dataset_md5(dataset_path),
        "feature_schema": feature_names,
        "bootstrap_disease": predicted_disease,
        "dataset_stats": dataset_stats,
        "metrics": metrics,
    }
    (models_dir / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain NeuroSynth ML artifacts")
    parser.add_argument("--dataset", default="neurological_disease_data.csv")
    parser.add_argument("--models-dir", default="models")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    models_dir = Path(args.models_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    manifest = run_pretrain(dataset_path=dataset_path, models_dir=models_dir)
    print(json.dumps({"status": "ok", "manifest": manifest.get("dataset_md5")}, indent=2))


if __name__ == "__main__":
    main()
