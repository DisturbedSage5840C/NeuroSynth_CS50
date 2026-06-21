# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Process OASIS-2/3 longitudinal data into the NeuroSynth 32-feature schema.

OASIS is free with registration (https://www.oasis-brains.org). Download
``oasis_longitudinal.csv`` into ``data/raw/oasis/`` then run this script. Columns
not present in OASIS are filled with cohort-level population means so the row is
schema-complete; the learnable signal comes from the genuine OASIS fields
(Age, M/F, MMSE, CDR, EDUC). The label is ``Diagnosis = 1`` when CDR > 0.

Usage:
    python scripts/data/process_oasis.py \
        --raw data/raw/oasis/oasis_longitudinal.csv \
        --out data/oasis3_processed.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Population means for fields OASIS does not carry (used as schema fillers).
POP_DEFAULTS: dict[str, float] = {
    "BMI": 26.5, "Smoking": 0, "AlcoholConsumption": 2.5, "PhysicalActivity": 4.2,
    "DietQuality": 5.6, "SleepQuality": 5.4, "FamilyHistoryAlzheimers": 0,
    "CardiovascularDisease": 0, "Diabetes": 0, "Depression": 0, "HeadInjury": 0,
    "Hypertension": 0, "SystolicBP": 132, "DiastolicBP": 80, "CholesterolTotal": 200,
    "CholesterolLDL": 120, "CholesterolHDL": 54, "CholesterolTriglycerides": 148,
    "Ethnicity": 0,
}


def process_oasis_longitudinal(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    out = pd.DataFrame()

    out["Age"] = df["Age"].clip(45, 100)
    out["Gender"] = (df["M/F"].astype(str).str.upper() == "M").astype(int)
    out["MMSE"] = df["MMSE"].clip(0, 30)
    cdr = df["CDR"].fillna(0).clip(0, 3)
    out["Diagnosis"] = (cdr > 0).astype(int)
    out["EducationLevel"] = df.get("EDUC", pd.Series([2] * len(df))).fillna(2).clip(0, 3).astype(int)

    for col, val in POP_DEFAULTS.items():
        out[col] = val

    # CDR-derived clinical proxies (higher CDR → worse function / more symptoms).
    out["FunctionalAssessment"] = (10 - cdr * 3).clip(0, 10)
    out["ADL"] = (10 - cdr * 2.5).clip(0, 10)
    out["MemoryComplaints"] = (cdr > 0).astype(int)
    out["BehavioralProblems"] = (cdr >= 1).astype(int)
    out["Confusion"] = (cdr >= 1).astype(int)
    out["Disorientation"] = (cdr >= 1).astype(int)
    out["PersonalityChanges"] = (cdr >= 0.5).astype(int)
    out["DifficultyCompletingTasks"] = (cdr >= 0.5).astype(int)
    out["Forgetfulness"] = (cdr > 0).astype(int)

    out["DiseaseType"] = "Alzheimer's Disease"

    out = out.dropna(subset=["Age", "MMSE", "Diagnosis"]).reset_index(drop=True)
    print(f"OASIS processed: {len(out)} rows, {out['Diagnosis'].mean():.2%} positive")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Process OASIS into NeuroSynth schema")
    ap.add_argument("--raw", default="data/raw/oasis/oasis_longitudinal.csv")
    ap.add_argument("--out", default="data/oasis3_processed.parquet")
    args = ap.parse_args()

    if not Path(args.raw).exists():
        raise FileNotFoundError(
            f"OASIS file not found: {args.raw}. Register and download from "
            "https://www.oasis-brains.org into data/raw/oasis/."
        )

    df = process_oasis_longitudinal(args.raw)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(out, index=False)
    except Exception:
        out = out.with_suffix(".csv")
        df.to_csv(out, index=False)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
