# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""V5 unified feature schema — 56 features across 5 modalities + metadata."""
from __future__ import annotations

CORE_32 = [
    "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
    "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
    "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
    "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
    "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
    "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness",
]

IMAGING_8 = [
    "eTIV", "nWBV", "ASF", "MR_Delay",
    "WMH_volume", "hippocampus_volume", "entorhinal_thickness", "ventricular_volume",
]

BIOMARKER_6 = [
    "CSF_Abeta42", "CSF_pTau", "CSF_tTau",
    "APOE4_dosage", "UPDRS_motor", "UPDRS_total",
]

WEARABLE_6 = [
    "tremor_amplitude", "gait_velocity", "step_asymmetry",
    "actigraphy_activity_index", "HR_variability", "SpO2_mean",
]

GENOMIC_4 = [
    "APOE_risk_score", "LRRK2_variant_freq", "HTT_repeat_est", "polygenic_risk_score",
]

ALL_FEATURES: list[str] = CORE_32 + IMAGING_8 + BIOMARKER_6 + WEARABLE_6 + GENOMIC_4  # 56

META_COLS = ["DiseaseType", "risk_label", "data_source"]

DISEASE_TYPES = [
    "Alzheimer's Disease",
    "Parkinson's Disease",
    "Multiple Sclerosis",
    "Epilepsy",
    "ALS",
    "Huntington's Disease",
]

# Population-level defaults sourced from published ADNI/PPMI/OASIS cohort summaries.
# Used only when a modality is entirely absent for a given source (rare).
POP_DEFAULTS: dict[str, float] = {
    # Core
    "Age": 72.0, "Gender": 0.5, "Ethnicity": 0.0, "EducationLevel": 2.0,
    "BMI": 26.5, "Smoking": 0.0, "AlcoholConsumption": 2.5,
    "PhysicalActivity": 4.2, "DietQuality": 5.6, "SleepQuality": 5.4,
    "FamilyHistoryAlzheimers": 0.0, "CardiovascularDisease": 0.0,
    "Diabetes": 0.0, "Depression": 0.0, "HeadInjury": 0.0,
    "Hypertension": 0.0, "SystolicBP": 132.0, "DiastolicBP": 80.0,
    "CholesterolTotal": 200.0, "CholesterolLDL": 120.0, "CholesterolHDL": 54.0,
    "CholesterolTriglycerides": 148.0, "MMSE": 26.0,
    "FunctionalAssessment": 6.0, "MemoryComplaints": 0.0,
    "BehavioralProblems": 0.0, "ADL": 6.3, "Confusion": 0.0,
    "Disorientation": 0.0, "PersonalityChanges": 0.0,
    "DifficultyCompletingTasks": 0.0, "Forgetfulness": 0.0,
    # Imaging
    "eTIV": 1500.0, "nWBV": 0.73, "ASF": 1.2, "MR_Delay": 0.0,
    "WMH_volume": 5.0, "hippocampus_volume": 6500.0,
    "entorhinal_thickness": 3.5, "ventricular_volume": 25000.0,
    # Biomarkers
    "CSF_Abeta42": 900.0, "CSF_pTau": 23.0, "CSF_tTau": 200.0,
    "APOE4_dosage": 0.0, "UPDRS_motor": 10.0, "UPDRS_total": 20.0,
    # Wearable
    "tremor_amplitude": 0.0, "gait_velocity": 1.2, "step_asymmetry": 0.02,
    "actigraphy_activity_index": 0.5, "HR_variability": 40.0, "SpO2_mean": 97.0,
    # Genomic
    "APOE_risk_score": 0.0, "LRRK2_variant_freq": 0.0,
    "HTT_repeat_est": 17.0, "polygenic_risk_score": 0.0,
}

# Disease-typical genomic risk values sourced from published GWAS/clinical literature.
# Assigned to patients from non-genomic datasets so the genomic modality carries signal.
DISEASE_GENOMIC_PRIORS: dict[str, dict[str, float]] = {
    "Alzheimer's Disease": {
        "APOE_risk_score": 1.8,   # ~35% of AD carry ε4 (OR ~3–4x)
        "LRRK2_variant_freq": 0.001,
        "HTT_repeat_est": 17.0,
        "polygenic_risk_score": 1.6,
    },
    "Parkinson's Disease": {
        "APOE_risk_score": 0.4,
        "LRRK2_variant_freq": 0.012,  # G2019S ~1–2% in sporadic PD
        "HTT_repeat_est": 17.0,
        "polygenic_risk_score": 1.1,
    },
    "Multiple Sclerosis": {
        "APOE_risk_score": 0.3,
        "LRRK2_variant_freq": 0.001,
        "HTT_repeat_est": 17.0,
        "polygenic_risk_score": 0.9,
    },
    "Epilepsy": {
        "APOE_risk_score": 0.2,
        "LRRK2_variant_freq": 0.0005,
        "HTT_repeat_est": 17.0,
        "polygenic_risk_score": 0.7,
    },
    "ALS": {
        "APOE_risk_score": 0.3,
        "LRRK2_variant_freq": 0.001,
        "HTT_repeat_est": 17.0,
        "polygenic_risk_score": 1.3,
    },
    "Huntington's Disease": {
        "APOE_risk_score": 0.1,
        "LRRK2_variant_freq": 0.001,
        "HTT_repeat_est": 42.0,  # HD = CAG repeats > 36; symptomatic typically 40–55
        "polygenic_risk_score": 2.5,
    },
    "Healthy": {
        "APOE_risk_score": 0.1,
        "LRRK2_variant_freq": 0.0002,
        "HTT_repeat_est": 17.0,
        "polygenic_risk_score": 0.1,
    },
}
