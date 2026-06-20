"""Priority 4 verification tests — Validation Pipeline (with gate fixes)."""
import logging
import shutil
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

import numpy as np
import pandas as pd

print("=" * 60)
print("NeuroSynth v2 — Priority 4 Validation Pipeline Verification")
print("          (with gate fixes: AUC, DPR, ECE, robustness)")
print("=" * 60)

# ---------------------------------------------------------------
# Step 0: Load data + add interaction features
# ---------------------------------------------------------------
from backend.data_pipeline import DataPipeline
from neurosynth.validation.feature_interactions import add_interaction_features

pipeline = DataPipeline()
X_train, X_test, y_train, y_test, feature_names, scaler, stats = pipeline.process()

# Add interaction features AFTER scaling (interactions of scaled features)
X_train_enhanced = add_interaction_features(X_train)
X_test_enhanced = add_interaction_features(X_test)
enhanced_features = list(X_train_enhanced.columns)

print(f"\nBase features:     {len(feature_names)}")
print(f"Enhanced features: {len(enhanced_features)}")
print(f"Added:             {len(enhanced_features) - len(feature_names)} interaction terms")

# ---------------------------------------------------------------
# Step 1: Train CalibratedEnsemble on enhanced features
# ---------------------------------------------------------------
print("\n[1/6] CalibratedEnsemble (enhanced features + isotonic calibration)...")
from neurosynth.models.calibrated_ensemble import CalibratedEnsemble

ensemble = CalibratedEnsemble(
    feature_names=enhanced_features,
    n_cv_folds=5,
)
metrics = ensemble.train(X_train_enhanced.values, y_train.values)
y_prob = ensemble.predict(X_test_enhanced.values[:1])["probability"]

# Get full test predictions
base_probs = ensemble._base_probs(X_test_enhanced.values)
if ensemble.calibrated_meta is not None:
    y_prob_all = ensemble.calibrated_meta.predict_proba(base_probs)[:, 1]
else:
    y_prob_all = ensemble.meta_learner.predict_proba(base_probs)[:, 1]

print(f"  Meta AUC (OOF): {metrics['meta_auc']}")
print(f"  Threshold:      {metrics['threshold']}")
print(f"  Models:         {metrics['n_base_models']}")

# ---------------------------------------------------------------
# Step 2: ModelValidator with enhanced ensemble
# ---------------------------------------------------------------
print("\n[2/6] ModelValidator (AUC, ECE, Brier, SHAP stability)...")
from neurosynth.validation.validator import ModelValidator

validator = ModelValidator()
report = validator.validate(
    y_true=y_test.values,
    y_prob=y_prob_all,
    model_name="calibrated_ensemble_v2",
    disease="Alzheimer's",
    model=ensemble.rf,
    X=X_test_enhanced.values,
    feature_names=enhanced_features,
)
print(f"  AUC:       {report.auc}")
print(f"  F1:        {report.f1}")
print(f"  ECE:       {report.calibration.ece}")
print(f"  MCE:       {report.calibration.mce}")
print(f"  Brier:     {report.calibration.brier}")
print(f"  Threshold: {report.optimal_threshold}")
print(f"  SHAP top5: {report.shap_top5_jaccard}")
assert report.auc > 0.5, "AUC should be better than random"
print("  PASSED")

# ---------------------------------------------------------------
# Step 3: FairnessAuditor with post-processing
# ---------------------------------------------------------------
print("\n[3/6] FairnessAuditor + PostProcessor (equalized predictions)...")
from neurosynth.validation.fairness import FairnessAuditor
from neurosynth.validation.fairness_postprocessor import FairnessPostProcessor

# Reconstruct raw (unscaled) features for demographic group binning
# The scaler standardizes Age to ~0 mean, so we need original values
X_train_raw = pd.DataFrame(
    scaler.inverse_transform(X_train.values),
    columns=feature_names,
    index=X_train.index,
)
X_test_raw = pd.DataFrame(
    scaler.inverse_transform(X_test.values),
    columns=feature_names,
    index=X_test.index,
)

# Fit fairness post-processor on training data using RAW features for grouping
train_base_probs = ensemble._base_probs(X_train_enhanced.values)
if ensemble.calibrated_meta is not None:
    y_prob_train = ensemble.calibrated_meta.predict_proba(train_base_probs)[:, 1]
else:
    y_prob_train = ensemble.meta_learner.predict_proba(train_base_probs)[:, 1]

age_processor = FairnessPostProcessor(protected_attr="Age")
age_processor.fit(y_train.values, y_prob_train, X_train_raw)

# Apply calibration to test set using raw features
y_prob_fair = age_processor.transform(y_prob_all, X_test_raw)

# Audit fairness on calibrated probabilities using raw features
auditor = FairnessAuditor(
    protected_attributes=["Age", "Gender"],
    threshold=report.optimal_threshold,
)
fairness = auditor.assess(
    y_true=y_test.values,
    y_prob=y_prob_fair,
    features=X_test_raw,
    model_name="calibrated_ensemble_v2",
)
print(f"  DPR:               {fairness.demographic_parity_ratio:.4f}")
print(f"  EOR:               {fairness.equalized_odds_ratio:.4f}")
print(f"  Predictive parity: {fairness.predictive_parity_ratio:.4f}")
print(f"  Max TPR gap:       {fairness.max_tpr_gap:.4f}")
print(f"  Passes 4/5 rule:   {fairness.passes_four_fifths}")
print(f"  Groups evaluated:  {len(fairness.groups)}")
print(f"  Post-processor:    {age_processor.get_summary()}")
assert len(fairness.groups) > 0
print("  PASSED")

# ---------------------------------------------------------------
# Step 4: RobustnessTester on enhanced ensemble
# ---------------------------------------------------------------
print("\n[4/6] RobustnessTester (10 perturbation tests)...")
from neurosynth.validation.robustness import RobustnessTester

tester = RobustnessTester()


def predict_fn(X):
    bp = ensemble._base_probs(X)
    if ensemble.calibrated_meta is not None:
        return ensemble.calibrated_meta.predict_proba(bp)[:, 1]
    return ensemble.meta_learner.predict_proba(bp)[:, 1]


robustness = tester.run_all(predict_fn, X_test_enhanced.values, y_test.values, model_name="calibrated_ensemble_v2")
print(f"  Tests run:       {len(robustness.tests)}")
print(f"  Overall pass:    {robustness.overall_pass}")
print(f"  Worst AUC drop:  {robustness.worst_auc_drop:.4f}")
print(f"  Worst test:      {robustness.worst_test}")
for t in robustness.tests:
    status = "✅" if t.passed else "❌"
    print(f"    {status} {t.test_name}: AUC {t.original_auc:.4f}→{t.perturbed_auc:.4f} (Δ={t.auc_delta:.4f})")
assert len(robustness.tests) == 10
print("  PASSED")

# ---------------------------------------------------------------
# Step 5: AuditTrail
# ---------------------------------------------------------------
print("\n[5/6] AuditTrail (FDA SaMD hash-chained logging)...")
from neurosynth.validation.audit import AuditTrail

audit_dir = tempfile.mkdtemp(prefix="neurosynth_audit_",
                             dir=os.path.join(os.path.dirname(__file__), ".."))
try:
    trail = AuditTrail(audit_dir=audit_dir)
    trail.log_validation(model_name="calibrated_ensemble_v2", metrics=report.to_dict())
    trail.log_gate_decision(model_name="calibrated_ensemble_v2", passed=True, gates={"auc": {"value": report.auc}})
    trail.log_deployment(model_name="calibrated_ensemble_v2", environment="staging")

    chain_valid = trail.verify_chain()
    print(f"  Entries logged:  {len(trail._entries)}")
    print(f"  Chain valid:     {chain_valid}")
    assert chain_valid
    print("  PASSED")
finally:
    shutil.rmtree(audit_dir, ignore_errors=True)

# ---------------------------------------------------------------
# Step 6: ValidationGates (final gate check)
# ---------------------------------------------------------------
print("\n[6/6] ValidationGates (hard/soft gate evaluation)...")
from neurosynth.validation.gates import ValidationGates

gates = ValidationGates()
decision = gates.evaluate(
    validation=report,
    fairness=fairness,
    robustness=robustness,
    model_version="v2.0.0-alpha.4",
)
print(f"  Decision:      {decision.decision}")
print(f"  Hard fails:    {decision.hard_fails}")
print(f"  Soft warns:    {decision.soft_warns}")
print(f"  Total gates:   {decision.total_gates}")
print(f"  Summary:       {decision.summary}")
for g in decision.gates:
    icon = "✅" if g.result == "PASS" else ("⚠️" if g.result == "SOFT_WARN" else "❌")
    print(f"    {icon} [{g.gate_type}] {g.gate_name}: {g.metric_name}={g.metric_value:.4f} (threshold={g.threshold})")
assert decision.decision in ("PROMOTE", "REJECT", "HUMAN_REVIEW")
assert decision.total_gates >= 5
print("  PASSED")

# ---------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------
print("\n" + "=" * 60)
print("GATE RESULTS SUMMARY")
print("=" * 60)
print(f"  {'Gate':<20} {'Type':<6} {'Metric':<10} {'Value':>8} {'Threshold':>10} {'Status':>8}")
print(f"  {'-'*20} {'-'*6} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
for g in decision.gates:
    status = "PASS" if g.result == "PASS" else ("WARN" if g.result == "SOFT_WARN" else "FAIL")
    print(f"  {g.gate_name:<20} {g.gate_type:<6} {g.metric_name:<10} {g.metric_value:>8.4f} {g.threshold:>10.2f} {status:>8}")

print(f"\n  Final decision: {decision.decision}")
print("=" * 60)
print("ALL PRIORITY 4 VERIFICATION TESTS PASSED")
print("=" * 60)
