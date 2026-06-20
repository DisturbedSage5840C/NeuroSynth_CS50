"""
Generates an expanded multi-disease neurological dataset combining:
1. The existing alzheimers_disease_data.csv
2. Synthetic patients for Parkinson's and MS cohorts
3. Unified schema for multi-disease training
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_parkinsons_cohort(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "Age": rng.normal(67, 9, n).clip(40, 95),
        "Gender": rng.binomial(1, 0.6, n),
        "Ethnicity": rng.randint(0, 4, n),
        "EducationLevel": rng.randint(0, 4, n),
        "BMI": rng.normal(25.5, 4.5, n).clip(16, 44),
        "Smoking": rng.binomial(1, 0.25, n),
        "AlcoholConsumption": rng.exponential(2.5, n).clip(0, 20),
        "PhysicalActivity": rng.normal(3.8, 2.2, n).clip(0, 10),
        "DietQuality": rng.normal(5.5, 2.0, n).clip(0, 10),
        "SleepQuality": rng.normal(5.2, 2.1, n).clip(0, 10),
        "FamilyHistoryAlzheimers": rng.binomial(1, 0.08, n),
        "CardiovascularDisease": rng.binomial(1, 0.3, n),
        "Diabetes": rng.binomial(1, 0.18, n),
        "Depression": rng.binomial(1, 0.45, n),
        "HeadInjury": rng.binomial(1, 0.15, n),
        "Hypertension": rng.binomial(1, 0.38, n),
        "SystolicBP": rng.normal(128, 18, n).clip(85, 210),
        "DiastolicBP": rng.normal(80, 11, n).clip(45, 135),
        "CholesterolTotal": rng.normal(195, 33, n).clip(110, 390),
        "CholesterolLDL": rng.normal(118, 28, n).clip(45, 290),
        "CholesterolHDL": rng.normal(52, 14, n).clip(22, 115),
        "CholesterolTriglycerides": rng.normal(145, 55, n).clip(42, 480),
        "MMSE": rng.normal(25, 3.5, n).clip(10, 30),
        "FunctionalAssessment": rng.normal(5.8, 2.2, n).clip(0, 10),
        "MemoryComplaints": rng.binomial(1, 0.30, n),
        "BehavioralProblems": rng.binomial(1, 0.35, n),
        "ADL": rng.normal(6.0, 2.2, n).clip(0, 10),
        "Confusion": rng.binomial(1, 0.20, n),
        "Disorientation": rng.binomial(1, 0.15, n),
        "PersonalityChanges": rng.binomial(1, 0.40, n),
        "DifficultyCompletingTasks": rng.binomial(1, 0.55, n),
        "Forgetfulness": rng.binomial(1, 0.35, n),
        "DiseaseType": "Parkinson's Disease",
        "Diagnosis": rng.binomial(1, 0.65, n),
    })


def generate_ms_cohort(n: int = 1200, seed: int = 43) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "Age": rng.normal(37, 11, n).clip(18, 70),
        "Gender": rng.binomial(1, 0.30, n),
        "Ethnicity": rng.randint(0, 4, n),
        "EducationLevel": rng.randint(1, 4, n),
        "BMI": rng.normal(25.2, 5.0, n).clip(16, 44),
        "Smoking": rng.binomial(1, 0.35, n),
        "AlcoholConsumption": rng.exponential(2.0, n).clip(0, 20),
        "PhysicalActivity": rng.normal(5.2, 2.3, n).clip(0, 10),
        "DietQuality": rng.normal(6.0, 2.0, n).clip(0, 10),
        "SleepQuality": rng.normal(6.2, 1.9, n).clip(0, 10),
        "FamilyHistoryAlzheimers": rng.binomial(1, 0.05, n),
        "CardiovascularDisease": rng.binomial(1, 0.08, n),
        "Diabetes": rng.binomial(1, 0.06, n),
        "Depression": rng.binomial(1, 0.50, n),
        "HeadInjury": rng.binomial(1, 0.08, n),
        "Hypertension": rng.binomial(1, 0.15, n),
        "SystolicBP": rng.normal(115, 14, n).clip(85, 180),
        "DiastolicBP": rng.normal(74, 10, n).clip(45, 120),
        "CholesterolTotal": rng.normal(182, 28, n).clip(110, 350),
        "CholesterolLDL": rng.normal(108, 24, n).clip(45, 280),
        "CholesterolHDL": rng.normal(60, 15, n).clip(22, 115),
        "CholesterolTriglycerides": rng.normal(125, 48, n).clip(42, 400),
        "MMSE": rng.normal(27.5, 2.0, n).clip(18, 30),
        "FunctionalAssessment": rng.normal(6.8, 1.9, n).clip(0, 10),
        "MemoryComplaints": rng.binomial(1, 0.35, n),
        "BehavioralProblems": rng.binomial(1, 0.25, n),
        "ADL": rng.normal(7.2, 1.8, n).clip(0, 10),
        "Confusion": rng.binomial(1, 0.18, n),
        "Disorientation": rng.binomial(1, 0.12, n),
        "PersonalityChanges": rng.binomial(1, 0.30, n),
        "DifficultyCompletingTasks": rng.binomial(1, 0.45, n),
        "Forgetfulness": rng.binomial(1, 0.30, n),
        "DiseaseType": "Multiple Sclerosis",
        "Diagnosis": rng.binomial(1, 0.70, n),
    })


def main() -> None:
    output_path = Path("neurological_disease_data.csv")

    try:
        alz_df = pd.read_csv("alzheimers_disease_data.csv")
        alz_df["DiseaseType"] = "Alzheimer's Disease"
        print(f"Loaded {len(alz_df)} Alzheimer's records")
    except FileNotFoundError:
        print("WARNING: alzheimers_disease_data.csv not found, generating synthetic only")
        alz_df = pd.DataFrame()

    pd_df = generate_parkinsons_cohort(n=1500)
    ms_df = generate_ms_cohort(n=1200)

    dfs = [df for df in [alz_df, pd_df, ms_df] if len(df) > 0]
    combined = pd.concat(dfs, ignore_index=True)

    all_cols = [
        "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
        "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
        "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
        "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
        "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
        "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
        "Forgetfulness", "DiseaseType", "Diagnosis",
    ]
    for col in all_cols:
        if col not in combined.columns:
            combined[col] = 0

    combined = combined[all_cols]
    combined.to_csv(output_path, index=False)

    print(f"\nExpanded dataset saved to {output_path}")
    print(f"Total records: {len(combined)}")
    print(f"Disease distribution:\n{combined['DiseaseType'].value_counts()}")
    print(f"Diagnosis rate: {combined['Diagnosis'].mean():.1%}")


if __name__ == "__main__":
    main()
