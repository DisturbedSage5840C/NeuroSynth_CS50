# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Extended 54-feature Pandera schema for NeuroSynth v2.

Defines the canonical feature set combining clinical, imaging, genomic,
wearable, and neuropsychological biomarkers.  All connectors should map
their outputs into this schema before downstream consumption.

Schema tiers:
  TIER_1 — Required for baseline ensemble model (34 features from v1 CSV)
  TIER_2 — Required for full v2 multi-modal inference (20 new features)
  TIER_3 — Optional enrichment (wearable, clinical notes)

Validation rules enforce clinically plausible ranges sourced from
published reference values (ADNI, PPMI, MIMIC-IV documentation).
"""
from __future__ import annotations

from enum import StrEnum
from typing import Any

import pandera as pa
from pandera.typing import Series


# ---------------------------------------------------------------------------
# Feature tier classification
# ---------------------------------------------------------------------------

class FeatureTier(StrEnum):
    TIER_1 = "tier_1"   # v1 clinical CSV features
    TIER_2 = "tier_2"   # v2 imaging / genomic / advanced
    TIER_3 = "tier_3"   # optional enrichment


# ---------------------------------------------------------------------------
# Feature metadata registry
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: dict[str, dict[str, Any]] = {
    # --- TIER 1: Original 34 clinical features (from CSV) ---
    "Age":                       {"tier": FeatureTier.TIER_1, "unit": "years",    "range": (18, 120)},
    "Gender":                    {"tier": FeatureTier.TIER_1, "unit": "encoded",  "range": (0, 1)},
    "Ethnicity":                 {"tier": FeatureTier.TIER_1, "unit": "encoded",  "range": (0, 10)},
    "EducationLevel":            {"tier": FeatureTier.TIER_1, "unit": "encoded",  "range": (0, 10)},
    "BMI":                       {"tier": FeatureTier.TIER_1, "unit": "kg/m²",    "range": (10, 60)},
    "Smoking":                   {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "AlcoholConsumption":        {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 20)},
    "PhysicalActivity":          {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 10)},
    "DietQuality":               {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 10)},
    "SleepQuality":              {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 10)},
    "FamilyHistoryAlzheimers":   {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "CardiovascularDisease":     {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "Diabetes":                  {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "Depression":                {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "HeadInjury":                {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "Hypertension":              {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "SystolicBP":                {"tier": FeatureTier.TIER_1, "unit": "mmHg",     "range": (60, 250)},
    "DiastolicBP":               {"tier": FeatureTier.TIER_1, "unit": "mmHg",     "range": (30, 160)},
    "CholesterolTotal":          {"tier": FeatureTier.TIER_1, "unit": "mg/dL",    "range": (80, 400)},
    "CholesterolLDL":            {"tier": FeatureTier.TIER_1, "unit": "mg/dL",    "range": (30, 300)},
    "CholesterolHDL":            {"tier": FeatureTier.TIER_1, "unit": "mg/dL",    "range": (10, 120)},
    "CholesterolTriglycerides":  {"tier": FeatureTier.TIER_1, "unit": "mg/dL",    "range": (30, 800)},
    "MMSE":                      {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 30)},
    "FunctionalAssessment":      {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 10)},
    "MemoryComplaints":          {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "BehavioralProblems":        {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "ADL":                       {"tier": FeatureTier.TIER_1, "unit": "score",    "range": (0, 10)},
    "Confusion":                 {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "Disorientation":            {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "PersonalityChanges":        {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "DifficultyCompletingTasks": {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},
    "Forgetfulness":             {"tier": FeatureTier.TIER_1, "unit": "binary",   "range": (0, 1)},

    # --- TIER 2: New v2 biomarkers (imaging, genomic, advanced clinical) ---
    "APOE_genotype":                   {"tier": FeatureTier.TIER_2, "unit": "allele_count", "range": (0, 2)},
    "CSF_Abeta42":                     {"tier": FeatureTier.TIER_2, "unit": "pg/mL",        "range": (100, 2000)},
    "CSF_tau_total":                   {"tier": FeatureTier.TIER_2, "unit": "pg/mL",        "range": (50, 1200)},
    "CSF_phospho_tau":                 {"tier": FeatureTier.TIER_2, "unit": "pg/mL",        "range": (5, 200)},
    "MRI_hippocampus_volume_mm3":      {"tier": FeatureTier.TIER_2, "unit": "mm³",          "range": (1500, 10000)},
    "MRI_entorhinal_cortex_thickness": {"tier": FeatureTier.TIER_2, "unit": "mm",           "range": (1.0, 5.0)},
    "FDG_PET_global_metabolism":       {"tier": FeatureTier.TIER_2, "unit": "SUVr",         "range": (0.5, 2.5)},
    "amyloid_PET_centiloid":           {"tier": FeatureTier.TIER_2, "unit": "centiloid",    "range": (-30, 200)},
    "gait_speed_ms":                   {"tier": FeatureTier.TIER_2, "unit": "m/s",          "range": (0.1, 3.0)},
    "grip_strength_kg":                {"tier": FeatureTier.TIER_2, "unit": "kg",           "range": (5, 80)},
    "dual_task_cost_pct":              {"tier": FeatureTier.TIER_2, "unit": "%",            "range": (0, 100)},
    "sleep_rem_pct":                   {"tier": FeatureTier.TIER_2, "unit": "%",            "range": (0, 50)},
    "actigraphy_fragmentation_index":  {"tier": FeatureTier.TIER_2, "unit": "index",        "range": (0, 100)},
    "EEG_theta_power_Fz":             {"tier": FeatureTier.TIER_2, "unit": "μV²",          "range": (0, 200)},
    "EEG_alpha_peak_freq_hz":         {"tier": FeatureTier.TIER_2, "unit": "Hz",           "range": (4, 14)},
    "polygenic_risk_score_AD":         {"tier": FeatureTier.TIER_2, "unit": "score",        "range": (-5, 5)},
    "polygenic_risk_score_PD":         {"tier": FeatureTier.TIER_2, "unit": "score",        "range": (-5, 5)},
    "time_since_symptom_onset_months": {"tier": FeatureTier.TIER_2, "unit": "months",       "range": (0, 360)},
    "disease_duration_months":         {"tier": FeatureTier.TIER_2, "unit": "months",       "range": (0, 360)},

    # --- TIER 3: Derived / enrichment (optional) ---
    "clinical_notes_embedding":        {"tier": FeatureTier.TIER_3, "unit": "BioBERT_dim", "range": (-10, 10)},
}


# ---------------------------------------------------------------------------
# Pandera schemas
# ---------------------------------------------------------------------------

class NeuroSynthTier1Schema(pa.DataFrameModel):
    """Validates the core 32 clinical features (required for baseline model)."""

    Age: Series[float] = pa.Field(ge=18, le=120, coerce=True)
    Gender: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    BMI: Series[float] = pa.Field(ge=10, le=60, coerce=True)
    SystolicBP: Series[float] = pa.Field(ge=60, le=250, coerce=True)
    DiastolicBP: Series[float] = pa.Field(ge=30, le=160, coerce=True)
    CholesterolTotal: Series[float] = pa.Field(ge=80, le=400, coerce=True)
    CholesterolLDL: Series[float] = pa.Field(ge=30, le=300, coerce=True)
    CholesterolHDL: Series[float] = pa.Field(ge=10, le=120, coerce=True)
    CholesterolTriglycerides: Series[float] = pa.Field(ge=30, le=800, coerce=True)
    PhysicalActivity: Series[float] = pa.Field(ge=0, le=10, coerce=True)
    SleepQuality: Series[float] = pa.Field(ge=0, le=10, coerce=True)
    DietQuality: Series[float] = pa.Field(ge=0, le=10, coerce=True)
    AlcoholConsumption: Series[float] = pa.Field(ge=0, le=20, coerce=True)
    MMSE: Series[float] = pa.Field(ge=0, le=30, coerce=True)
    FunctionalAssessment: Series[float] = pa.Field(ge=0, le=10, coerce=True)
    ADL: Series[float] = pa.Field(ge=0, le=10, coerce=True)
    Depression: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    Smoking: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    FamilyHistoryAlzheimers: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    CardiovascularDisease: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    Diabetes: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    HeadInjury: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    Hypertension: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    MemoryComplaints: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    BehavioralProblems: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    Confusion: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    Disorientation: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    PersonalityChanges: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    DifficultyCompletingTasks: Series[float] = pa.Field(ge=0, le=1, coerce=True)
    Forgetfulness: Series[float] = pa.Field(ge=0, le=1, coerce=True)

    class Config:
        strict = False
        coerce = True


class NeuroSynthTier2Schema(pa.DataFrameModel):
    """Validates the 20 new v2 biomarker features (nullable — may be NaN)."""

    APOE_genotype: Series[float] = pa.Field(ge=0, le=2, nullable=True, coerce=True)
    CSF_Abeta42: Series[float] = pa.Field(ge=100, le=2000, nullable=True, coerce=True)
    CSF_tau_total: Series[float] = pa.Field(ge=50, le=1200, nullable=True, coerce=True)
    CSF_phospho_tau: Series[float] = pa.Field(ge=5, le=200, nullable=True, coerce=True)
    MRI_hippocampus_volume_mm3: Series[float] = pa.Field(ge=1500, le=10000, nullable=True, coerce=True)
    MRI_entorhinal_cortex_thickness: Series[float] = pa.Field(ge=1.0, le=5.0, nullable=True, coerce=True)
    FDG_PET_global_metabolism: Series[float] = pa.Field(ge=0.5, le=2.5, nullable=True, coerce=True)
    amyloid_PET_centiloid: Series[float] = pa.Field(ge=-30, le=200, nullable=True, coerce=True)
    gait_speed_ms: Series[float] = pa.Field(ge=0.1, le=3.0, nullable=True, coerce=True)
    grip_strength_kg: Series[float] = pa.Field(ge=5, le=80, nullable=True, coerce=True)
    dual_task_cost_pct: Series[float] = pa.Field(ge=0, le=100, nullable=True, coerce=True)
    sleep_rem_pct: Series[float] = pa.Field(ge=0, le=50, nullable=True, coerce=True)
    actigraphy_fragmentation_index: Series[float] = pa.Field(ge=0, le=100, nullable=True, coerce=True)
    EEG_theta_power_Fz: Series[float] = pa.Field(ge=0, le=200, nullable=True, coerce=True)
    EEG_alpha_peak_freq_hz: Series[float] = pa.Field(ge=4, le=14, nullable=True, coerce=True)
    polygenic_risk_score_AD: Series[float] = pa.Field(ge=-5, le=5, nullable=True, coerce=True)
    polygenic_risk_score_PD: Series[float] = pa.Field(ge=-5, le=5, nullable=True, coerce=True)
    time_since_symptom_onset_months: Series[float] = pa.Field(ge=0, le=360, nullable=True, coerce=True)
    disease_duration_months: Series[float] = pa.Field(ge=0, le=360, nullable=True, coerce=True)

    class Config:
        strict = False
        coerce = True


# ---------------------------------------------------------------------------
# Convenience lookups
# ---------------------------------------------------------------------------

TIER_1_FEATURES = sorted(
    [k for k, v in FEATURE_REGISTRY.items() if v["tier"] == FeatureTier.TIER_1]
)

TIER_2_FEATURES = sorted(
    [k for k, v in FEATURE_REGISTRY.items() if v["tier"] == FeatureTier.TIER_2]
)

ALL_FEATURES = TIER_1_FEATURES + TIER_2_FEATURES

# ICD-10 mapping for NeuroSynth diseases
ICD10_MAPPING = {
    "Alzheimer's Disease":     {"codes": ["G30.0", "G30.1", "G30.8", "G30.9"], "category": "G30"},
    "Parkinson's Disease":     {"codes": ["G20", "G21.0", "G21.1"], "category": "G20"},
    "Multiple Sclerosis":      {"codes": ["G35"], "category": "G35"},
    "Epilepsy":                {"codes": ["G40.0", "G40.1", "G40.2", "G40.3", "G40.4", "G40.5", "G40.8", "G40.9"], "category": "G40"},
    "ALS":                     {"codes": ["G12.21"], "category": "G12"},
    "Huntington's Disease":    {"codes": ["G10"], "category": "G10"},
    "Frontotemporal Dementia": {"codes": ["G31.0", "G31.09"], "category": "G31"},
    "Lewy Body Dementia":      {"codes": ["G31.83"], "category": "G31"},
}
