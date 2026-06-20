"""Priority 3 verification tests — Model Upgrade."""
import logging
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import pandas as pd

print("=" * 60)
print("NeuroSynth v2 — Priority 3 Model Upgrade Verification")
print("=" * 60)

# 1. CalibratedEnsemble — Training and prediction
print("\n[1/4] CalibratedEnsemble (5-model + meta-learner)...")
from neurosynth.models.calibrated_ensemble import CalibratedEnsemble

# Use real data
from backend.data_pipeline import DataPipeline
pipeline = DataPipeline()
X_train, X_test, y_train, y_test, feature_names, scaler, stats = pipeline.process()

ensemble = CalibratedEnsemble(feature_names=feature_names, n_cv_folds=3)
metrics = ensemble.train(X_train.values, y_train.values)
print(f"  Meta AUC:  {metrics['meta_auc']}")
print(f"  Brier:     {metrics['meta_brier']}")
print(f"  Threshold: {metrics['threshold']}")
print(f"  Models:    {metrics['n_base_models']}")

# Predict
pred = ensemble.predict(X_test.values[:1])
print(f"  Prediction: {pred['prediction']} (prob={pred['probability']}, risk={pred['risk_level']})")
print(f"  Calibrated: {pred['calibrated']}")
print(f"  Per-model:  {pred['individual_model_probs']}")
print(f"  Top risk:   {pred['top_risk_factors']}")

# Evaluate
eval_metrics = ensemble.evaluate(X_test.values, y_test.values)
print(f"  Test AUC:   {eval_metrics['roc_auc']}")
print(f"  Test F1:    {eval_metrics['f1_weighted']}")
assert eval_metrics["roc_auc"] > 0.80, f"AUC too low: {eval_metrics['roc_auc']}"
print("  PASSED")

# 2. ModelHub — Registration and prediction
print("\n[2/4] ModelHub (unified multi-modal prediction)...")
from neurosynth.models.model_hub import ModelHub, Modality

hub = ModelHub(feature_names=feature_names)
hub.register_ensemble(ensemble)

# Check available modalities
assert Modality.CLINICAL in hub.available_modalities
assert len(hub.available_modalities) == 1  # Only ensemble registered
print(f"  Available modalities: {hub.available_modalities}")

# Predict with clinical only
result = hub.predict(clinical_features=X_test.values[:1])
print(f"  Fused probability: {result.probability}")
print(f"  Risk level: {result.risk_level}")
print(f"  Confidence: {result.confidence}")
print(f"  Models used: {result.model_contributions}")
assert 0.0 <= result.probability <= 1.0
assert result.risk_level in ("Low", "Moderate", "High", "Critical")
print("  PASSED")

# 3. Phase model imports
print("\n[3/4] Phase Model Imports...")
from neurosynth.connectome.phase2_gnn import BrainConnectomePhase2Model, ConnectomeConfig
print("  BrainConnectomePhase2Model: OK")

from neurosynth.genomics.phase3_variant_transformer import GenomicPhase3Model, Phase3Config
print("  GenomicPhase3Model: OK")

from neurosynth.forecasting.phase4_tft import ForecastingPhase4Model, Phase4Config
print("  ForecastingPhase4Model: OK")

from neurosynth.causal.phase5_engine import CausalPhase5Engine, Phase5Config
print("  CausalPhase5Engine: OK")

# Register all phase models (even without data, they should degrade gracefully)
gnn = BrainConnectomePhase2Model(ConnectomeConfig())
hub.register_connectome(gnn)

genomic = GenomicPhase3Model(Phase3Config())
hub.register_genomic(genomic)

causal = CausalPhase5Engine(Phase5Config())
hub.register_causal(causal)

print(f"  Available modalities after registration: {hub.available_modalities}")
assert len(hub.available_modalities) == 4
print("  PASSED")

# 4. Graceful degradation — predict with missing data
print("\n[4/4] Graceful Degradation (missing modalities)...")
result_partial = hub.predict(
    clinical_features=X_test.values[:1],
    connectome_data=None,     # No imaging data
    genomic_data=None,        # No genomic data
    longitudinal_df=None,     # No longitudinal data
    causal_df=None,           # No causal data
)
# Should still produce a valid prediction from clinical only
assert 0.0 <= result_partial.probability <= 1.0

# Count available models
available_count = sum(1 for p in result_partial.per_model if p.available)
unavailable_count = sum(1 for p in result_partial.per_model if not p.available)
print(f"  Available models: {available_count}, Unavailable: {unavailable_count}")
print(f"  Fused probability: {result_partial.probability}")
print(f"  Risk level: {result_partial.risk_level}")
assert available_count >= 1, "At least clinical model should be available"
print("  PASSED")

print("\n" + "=" * 60)
print("ALL PRIORITY 3 VERIFICATION TESTS PASSED")
print("=" * 60)
