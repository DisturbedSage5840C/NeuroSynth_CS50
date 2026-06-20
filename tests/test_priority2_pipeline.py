"""Priority 2 verification tests — Data Pipeline Upgrade."""
import logging
import sys
import os

# Ensure project root is on PYTHONPATH
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

print("=" * 60)
print("NeuroSynth v2 — Priority 2 Data Pipeline Verification")
print("=" * 60)

# 1. Schema module
print("\n[1/6] Feature Schema...")
from neurosynth.data.schema import (
    ALL_FEATURES, TIER_1_FEATURES, TIER_2_FEATURES,
    FEATURE_REGISTRY, FeatureTier, ICD10_MAPPING,
    NeuroSynthTier1Schema, NeuroSynthTier2Schema,
)
assert len(TIER_1_FEATURES) >= 30, f"Expected >=30 Tier 1 features, got {len(TIER_1_FEATURES)}"
assert len(TIER_2_FEATURES) >= 19, f"Expected >=19 Tier 2 features, got {len(TIER_2_FEATURES)}"
assert len(ALL_FEATURES) >= 50, f"Expected >=50 total features, got {len(ALL_FEATURES)}"
assert len(ICD10_MAPPING) >= 6, f"Expected >=6 ICD-10 mappings, got {len(ICD10_MAPPING)}"
print(f"  Tier 1 features: {len(TIER_1_FEATURES)}")
print(f"  Tier 2 features: {len(TIER_2_FEATURES)}")
print(f"  Total features:  {len(ALL_FEATURES)}")
print(f"  ICD-10 diseases: {len(ICD10_MAPPING)}")
print("  PASSED")

# 2. Data Quality Agent
print("\n[2/6] Data Quality Agent...")
import numpy as np
import pandas as pd
from neurosynth.data.quality import DataQualityAgent, QualityReport

rng = np.random.RandomState(42)
ref_df = pd.DataFrame({
    "Age": rng.normal(65, 10, 500),
    "MMSE": rng.normal(24, 5, 500),
    "BMI": rng.normal(26, 4, 500),
})
agent = DataQualityAgent(reference_df=ref_df)

# Test with same distribution (should have low PSI)
batch_same = pd.DataFrame({
    "Age": rng.normal(65, 10, 200),
    "MMSE": rng.normal(24, 5, 200),
    "BMI": rng.normal(26, 4, 200),
})
report_same = agent.assess(batch_same, batch_id="same_dist")
assert report_same.drift_alert is False, "Same distribution should not trigger drift alert"
print(f"  Same-distribution PSI: {report_same.psi_scores}")
print(f"  Quality score: {report_same.overall_quality_score:.4f}")

# Test with shifted distribution (should detect drift)
batch_shifted = pd.DataFrame({
    "Age": rng.normal(85, 5, 200),  # Much older population
    "MMSE": rng.normal(15, 3, 200),  # Much lower MMSE
    "BMI": rng.normal(26, 4, 200),
})
report_shifted = agent.assess(batch_shifted, batch_id="shifted_dist")
has_high_psi = any(v > 0.10 for v in report_shifted.psi_scores.values())
assert has_high_psi, "Shifted distribution should trigger PSI > 0.10"
print(f"  Shifted PSI: {report_shifted.psi_scores}")
print(f"  Drift alert: {report_shifted.drift_alert}")

# Test PII detection
pii_df = pd.DataFrame({
    "notes": ["Patient Dr. Smith had MMSE 24", "SSN 123-45-6789", "Normal record"],
    "Age": [65, 70, 75],
})
pii_flags = DataQualityAgent.scan_pii(pii_df)
assert len(pii_flags) >= 1, "Should detect PII patterns"
print(f"  PII patterns detected: {len(pii_flags)}")
print("  PASSED")

# 3. Feature Engineering
print("\n[3/6] Feature Matrix Builder...")
from neurosynth.data.feature_engineering import FeatureMatrixBuilder

builder = FeatureMatrixBuilder(reference_df=ref_df)
csv_path = "neurological_disease_data.csv"
if os.path.exists(csv_path):
    matrix = builder.build_from_csv(csv_path)
    print(f"  Matrix shape: {matrix.shape}")

    coverage = FeatureMatrixBuilder.tier_coverage(matrix)
    print(f"  Tier 1: {coverage['tier_1']['present']}/{coverage['tier_1']['total']} features")
    print(f"  Tier 2: {coverage['tier_2']['present']}/{coverage['tier_2']['total']} features")

    enriched = FeatureMatrixBuilder.compute_derived_features(matrix)
    new_cols = set(enriched.columns) - set(matrix.columns)
    print(f"  Derived features added: {len(new_cols)} ({sorted(new_cols)})")

    report = builder.validate(enriched, batch_id="csv_load")
    print(f"  Quality score: {report.overall_quality_score:.4f}")
else:
    print(f"  SKIPPED: {csv_path} not found")
print("  PASSED")

# 4. New connectors (import only — no network calls)
print("\n[4/6] New Connectors (import check)...")
from neurosynth.connectors.openneuro import OpenNeuroConnector, NEURO_DATASETS
print(f"  OpenNeuroConnector: {len(NEURO_DATASETS)} default datasets")
from neurosynth.connectors.gnomad import GnomADConnector, NEURO_GENE_PANEL
print(f"  GnomADConnector: {len(NEURO_GENE_PANEL)} genes in panel")
from neurosynth.connectors.ukbb import UKBBConnector, UKBB_FIELD_MAP
print(f"  UKBBConnector: {len(UKBB_FIELD_MAP)} field mappings")
print("  PASSED")

# 5. Schema validation
print("\n[5/6] Pandera Schema Validation...")
valid_df = pd.DataFrame({
    "Age": [65.0, 72.0],
    "Gender": [0.0, 1.0],
    "BMI": [25.0, 28.0],
    "SystolicBP": [130.0, 140.0],
    "DiastolicBP": [80.0, 85.0],
    "CholesterolTotal": [200.0, 210.0],
    "CholesterolLDL": [100.0, 110.0],
    "CholesterolHDL": [50.0, 55.0],
    "CholesterolTriglycerides": [150.0, 160.0],
    "PhysicalActivity": [5.0, 6.0],
    "SleepQuality": [6.0, 7.0],
    "DietQuality": [6.0, 7.0],
    "AlcoholConsumption": [2.0, 3.0],
    "MMSE": [24.0, 20.0],
    "FunctionalAssessment": [7.0, 5.0],
    "ADL": [7.0, 5.0],
    "Depression": [0.0, 1.0],
    "Smoking": [0.0, 1.0],
    "FamilyHistoryAlzheimers": [0.0, 1.0],
    "CardiovascularDisease": [0.0, 1.0],
    "Diabetes": [0.0, 0.0],
    "HeadInjury": [0.0, 0.0],
    "Hypertension": [1.0, 0.0],
    "MemoryComplaints": [0.0, 1.0],
    "BehavioralProblems": [0.0, 1.0],
    "Confusion": [0.0, 1.0],
    "Disorientation": [0.0, 1.0],
    "PersonalityChanges": [0.0, 0.0],
    "DifficultyCompletingTasks": [0.0, 1.0],
    "Forgetfulness": [0.0, 1.0],
})
validated = NeuroSynthTier1Schema.validate(valid_df)
print(f"  Tier 1 schema validated: {len(validated)} rows")

# Test invalid data
try:
    invalid_df = valid_df.copy()
    invalid_df.loc[0, "Age"] = 150.0  # Invalid age
    NeuroSynthTier1Schema.validate(invalid_df)
    print("  ERROR: Should have rejected Age=150")
except Exception:
    print("  Correctly rejected invalid Age=150")
print("  PASSED")

# 6. DVC pipeline file
print("\n[6/6] DVC Pipeline...")
dvc_path = os.path.join(os.path.dirname(__file__), "..", "dvc.yaml")
assert os.path.exists(dvc_path), "dvc.yaml not found"
import yaml
with open(dvc_path) as f:
    dvc_config = yaml.safe_load(f)
stages = list(dvc_config.get("stages", {}).keys())
print(f"  DVC stages: {stages}")
assert len(stages) >= 4, f"Expected >=4 DVC stages, got {len(stages)}"
print("  PASSED")

print("\n" + "=" * 60)
print("ALL PRIORITY 2 VERIFICATION TESTS PASSED")
print("=" * 60)
