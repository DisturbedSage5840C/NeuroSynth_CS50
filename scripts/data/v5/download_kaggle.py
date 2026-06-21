# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Download and process Kaggle neurological datasets into the v5 schema.

Datasets:
  1. rabieelkharoua/alzheimers-disease-dataset  (2,149 rows — perfect column match)
  2. shashwatwork/dementia-prediction-dataset    (  373 rows — OASIS-2 tabular + imaging)
  3. tanishchavaan/neurological-disease-prediction (~5,000 rows — 6-class multi-disease)

Prerequisites (kaggle 2.x new token format):
    mkdir -p ~/.kaggle
    echo "KGAT_<your_token>" > ~/.kaggle/access_token
    chmod 600 ~/.kaggle/access_token
  OR:
    export KAGGLE_API_TOKEN=KGAT_<your_token>

Usage:
    python scripts/data/v5/download_kaggle.py [--out-dir data/raw/kaggle]
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import io
import os
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts.data.v5.schema import (
    ALL_FEATURES,
    DISEASE_GENOMIC_PRIORS,
    DISEASE_TYPES,
    META_COLS,
    POP_DEFAULTS,
)

_DATASETS = {
    "alzheimers": "rabieelkharoua/alzheimers-disease-dataset",
    "dementia": "shashwatwork/dementia-prediction-dataset",
    # Stroke dataset: neurological vascular disease — maps Age, BMI, hypertension,
    # heart disease, glucose, smoking → stroke (binary). 5,110 rows.
    "stroke": "fedesoriano/stroke-prediction-dataset",
}

# Canonical 6-class disease name mapping for fuzzy column values
_DISEASE_ALIAS: dict[str, str] = {
    "alzheimer": "Alzheimer's Disease",
    "alzheimers": "Alzheimer's Disease",
    "alzheimer's": "Alzheimer's Disease",
    "ad": "Alzheimer's Disease",
    "parkinson": "Parkinson's Disease",
    "parkinsons": "Parkinson's Disease",
    "parkinson's": "Parkinson's Disease",
    "pd": "Parkinson's Disease",
    "multiple sclerosis": "Multiple Sclerosis",
    "ms": "Multiple Sclerosis",
    "epilepsy": "Epilepsy",
    "seizure": "Epilepsy",
    "als": "ALS",
    "amyotrophic": "ALS",
    "lou gehrig": "ALS",
    "huntington": "Huntington's Disease",
    "huntingtons": "Huntington's Disease",
    "huntington's": "Huntington's Disease",
    "hd": "Huntington's Disease",
    "healthy": "Healthy",
    "normal": "Healthy",
    "control": "Healthy",
    "non-demented": "Healthy",
    "nondemented": "Healthy",
}


def _resolve_token() -> str:
    """Resolve Kaggle API token — supports kaggle 2.x new KGAT_* format."""
    # 1. Explicit env var (works with both old and new kaggle)
    token = os.getenv("KAGGLE_API_TOKEN", "").strip()
    if token:
        return token
    # 2. New format: ~/.kaggle/access_token
    access_token_path = Path.home() / ".kaggle" / "access_token"
    if access_token_path.exists():
        token = access_token_path.read_text().strip()
        if token:
            return token
    # 3. Old format: ~/.kaggle/kaggle.json  (username + key)
    json_path = Path.home() / ".kaggle" / "kaggle.json"
    if json_path.exists():
        import json
        creds = json.loads(json_path.read_text())
        os.environ["KAGGLE_USERNAME"] = creds.get("username", "")
        os.environ["KAGGLE_KEY"] = creds.get("key", "")
        return ""  # old-style: env vars set, no token needed

    raise RuntimeError(
        "Kaggle credentials not found.\n\n"
        "  Option A (kaggle 2.x — your API token from kaggle.com/settings):\n"
        "    mkdir -p ~/.kaggle\n"
        "    echo 'KGAT_<your_token>' > ~/.kaggle/access_token\n"
        "    chmod 600 ~/.kaggle/access_token\n\n"
        "  Option B (env var):\n"
        "    export KAGGLE_API_TOKEN=KGAT_<your_token>"
    )


def _download_dataset(token: str, dataset: str, out_dir: Path) -> Path:
    """Download and unzip a Kaggle dataset using the CLI (works with kaggle 2.x)."""
    dest = out_dir / dataset.split("/")[-1]
    dest.mkdir(parents=True, exist_ok=True)

    env = {**os.environ}
    if token:
        env["KAGGLE_API_TOKEN"] = token

    cmd = [
        _sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", dataset,
        "--path", str(dest),
        "--unzip",
    ]
    print(f"  Running: kaggle datasets download -d {dataset} --path {dest} --unzip")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle CLI failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout.strip()}\n"
            f"  stderr: {result.stderr.strip()}"
        )
    if result.stdout:
        print(f"  {result.stdout.strip()}")
    return dest


def _scaffold(n: int, disease_type: str, data_source: str) -> pd.DataFrame:
    df = pd.DataFrame({col: [POP_DEFAULTS.get(col, np.nan)] * n for col in ALL_FEATURES})
    genomic = DISEASE_GENOMIC_PRIORS.get(disease_type, DISEASE_GENOMIC_PRIORS["Alzheimer's Disease"])
    for col, val in genomic.items():
        df[col] = val
    df["DiseaseType"] = disease_type
    df["data_source"] = data_source
    return df


def _normalize_disease(raw: str) -> str:
    clean = str(raw).strip().lower()
    for key, canonical in _DISEASE_ALIAS.items():
        if key in clean:
            return canonical
    return "Alzheimer's Disease"  # safe fallback for ambiguous labels


# ─── Dataset 1: Alzheimer's Disease Dataset (rabieelkharoua) ─────────────────

def process_alzheimers(folder: Path) -> pd.DataFrame:
    """Perfect 1:1 column match with CORE_32 schema."""
    csv = next(folder.glob("*.csv"), None)
    if csv is None:
        raise FileNotFoundError(f"No CSV found in {folder}")

    raw = pd.read_csv(csv)
    print(f"[alzheimers] raw shape: {raw.shape}, columns: {list(raw.columns[:10])}...")

    n = len(raw)
    df = _scaffold(n, "Alzheimer's Disease", "kaggle_alzheimers")

    # Direct column mapping (dataset columns match FEATURE_ORDER almost exactly)
    direct_cols = [
        "Age", "Gender", "Ethnicity", "EducationLevel", "BMI", "Smoking",
        "AlcoholConsumption", "PhysicalActivity", "DietQuality", "SleepQuality",
        "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
        "HeadInjury", "Hypertension", "SystolicBP", "DiastolicBP", "CholesterolTotal",
        "CholesterolLDL", "CholesterolHDL", "CholesterolTriglycerides", "MMSE",
        "FunctionalAssessment", "MemoryComplaints", "BehavioralProblems", "ADL",
        "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
        "Forgetfulness",
    ]
    for col in direct_cols:
        if col in raw.columns:
            df[col] = raw[col].values

    # Label
    diag_col = next((c for c in raw.columns if c.lower() in ("diagnosis", "label", "target")), None)
    if diag_col:
        df["risk_label"] = raw[diag_col].astype(int)
    else:
        df["risk_label"] = 0

    # Patients with Diagnosis=0 (cognitively normal) retain disease_type as "Alzheimer's Disease"
    # (they are in the study cohort, at risk) — matches original dataset encoding

    pos = df["risk_label"].mean()
    print(f"[alzheimers] {n} rows, Diagnosis=1: {pos:.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── Dataset 2: Dementia Prediction Dataset (OASIS-2 tabular) ────────────────

def process_dementia(folder: Path) -> pd.DataFrame:
    """OASIS-2 tabular: imaging features (eTIV, nWBV, ASF) + MMSE/CDR."""
    csv = next(folder.glob("*.csv"), None)
    if csv is None:
        raise FileNotFoundError(f"No CSV found in {folder}")

    raw = pd.read_csv(csv)
    print(f"[dementia] raw shape: {raw.shape}, columns: {list(raw.columns)}...")

    n = len(raw)
    df = _scaffold(n, "Alzheimer's Disease", "kaggle_dementia_oasis2")

    col = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}

    # Demographics
    if "age" in col:
        df["Age"] = raw[col["age"]].clip(45, 100)
    sex_col = col.get("m/f") or col.get("sex") or col.get("gender")
    if sex_col:
        df["Gender"] = (raw[sex_col].astype(str).str.upper() == "M").astype(int)
    educ_col = col.get("educ") or col.get("education")
    if educ_col:
        df["EducationLevel"] = raw[educ_col].fillna(12).clip(0, 23) / 23 * 3  # normalize

    # Cognitive
    if "mmse" in col:
        df["MMSE"] = raw[col["mmse"]].clip(0, 30)
    cdr_col = col.get("cdr")
    if cdr_col:
        cdr = raw[cdr_col].fillna(0).clip(0, 3)
        df["FunctionalAssessment"] = (10 - cdr * 3).clip(0, 10)
        df["ADL"] = (10 - cdr * 2.5).clip(0, 10)
        df["MemoryComplaints"] = (cdr > 0).astype(int)
        df["BehavioralProblems"] = (cdr >= 1).astype(int)
        df["Confusion"] = (cdr >= 1).astype(int)
        df["Disorientation"] = (cdr >= 1).astype(int)
        df["PersonalityChanges"] = (cdr >= 0.5).astype(int)
        df["DifficultyCompletingTasks"] = (cdr >= 0.5).astype(int)
        df["Forgetfulness"] = (cdr > 0).astype(int)

    # Imaging — the key value-add of this dataset
    for src, dst in [("etiv", "eTIV"), ("nwbv", "nWBV"), ("asf", "ASF"), ("mr_delay", "MR_Delay")]:
        if src in col:
            df[dst] = raw[col[src]]

    # Label
    group_col = col.get("group") or col.get("diagnosis") or col.get("label")
    if group_col:
        group = raw[group_col].astype(str).str.lower()
        df["risk_label"] = group.isin(["demented", "converted", "1", "yes"]).astype(int)
    else:
        # Fall back to CDR > 0 if group column not found
        if cdr_col:
            df["risk_label"] = (raw[cdr_col].fillna(0) > 0).astype(int)

    # MR_Delay: present in dataset as visit interval in days
    mr_delay = col.get("mr_delay") or col.get("mr delay")
    if mr_delay:
        df["MR_Delay"] = raw[mr_delay].fillna(0)

    pos = df["risk_label"].mean()
    print(f"[dementia] {n} rows, Demented fraction: {pos:.2%}")
    return df[ALL_FEATURES + META_COLS]


# ─── Dataset 3: Multi-Class Neurological Disease Prediction ──────────────────

def process_multiclass(folder: Path) -> pd.DataFrame:
    """6-class neurological disease prediction dataset."""
    csvs = list(folder.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV found in {folder}")
    # Use largest CSV if multiple
    csv = max(csvs, key=lambda p: p.stat().st_size)
    raw = pd.read_csv(csv)
    print(f"[multiclass] raw shape: {raw.shape}, columns: {list(raw.columns[:15])}...")

    n = len(raw)
    df = _scaffold(n, "Alzheimer's Disease", "kaggle_multiclass_neuro")

    col = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}

    # Find disease/label column
    disease_col = None
    for candidate in ("disease", "condition", "diagnosis", "label", "target", "class",
                       "disease_type", "diseasetype", "type"):
        if candidate in col:
            disease_col = col[candidate]
            break

    if disease_col:
        df["DiseaseType"] = raw[disease_col].astype(str).apply(_normalize_disease)
        df["risk_label"] = (df["DiseaseType"] != "Healthy").astype(int)
    else:
        print("[multiclass] WARNING: no disease column found — labeling all as high risk")
        df["risk_label"] = 1

    # Map genomic priors per row based on DiseaseType
    for disease in df["DiseaseType"].unique():
        mask = df["DiseaseType"] == disease
        priors = DISEASE_GENOMIC_PRIORS.get(disease, DISEASE_GENOMIC_PRIORS["Alzheimer's Disease"])
        for gcol, gval in priors.items():
            df.loc[mask, gcol] = gval

    # Clinical features — map common naming patterns
    clinical_map = {
        "age": "Age", "gender": "Gender", "sex": "Gender",
        "mmse": "MMSE", "mmse_score": "MMSE",
        "bmi": "BMI", "body_mass_index": "BMI",
        "smoking": "Smoking", "smoker": "Smoking",
        "diabetes": "Diabetes", "diabetic": "Diabetes",
        "hypertension": "Hypertension", "high_bp": "Hypertension",
        "depression": "Depression",
        "education": "EducationLevel", "education_level": "EducationLevel",
        "alcohol": "AlcoholConsumption", "alcohol_consumption": "AlcoholConsumption",
        "physical_activity": "PhysicalActivity", "exercise": "PhysicalActivity",
        "sleep": "SleepQuality", "sleep_quality": "SleepQuality",
        "cholesterol": "CholesterolTotal", "total_cholesterol": "CholesterolTotal",
        "systolic_bp": "SystolicBP", "diastolic_bp": "DiastolicBP",
        "memory_complaints": "MemoryComplaints",
        "family_history": "FamilyHistoryAlzheimers",
        "cardiovascular": "CardiovascularDisease",
    }
    for raw_col_key, target in clinical_map.items():
        if raw_col_key in col:
            df[target] = pd.to_numeric(raw[col[raw_col_key]], errors="coerce")

    dist = df["DiseaseType"].value_counts().to_dict()
    print(f"[multiclass] {n} rows, disease distribution: {dist}")
    return df[ALL_FEATURES + META_COLS]


# ─── Dataset 3: Stroke Prediction Dataset (fedesoriano) ──────────────────────

def process_stroke(folder: Path) -> pd.DataFrame:
    """
    Stroke dataset: 5,110 rows. Columns: id, gender, age, hypertension,
    heart_disease, ever_married, work_type, Residence_type,
    avg_glucose_level, bmi, smoking_status, stroke (0/1).
    Maps to vascular neurological risk profile.
    """
    csv = next(folder.glob("*.csv"), None)
    if csv is None:
        raise FileNotFoundError(f"No CSV found in {folder}")

    raw = pd.read_csv(csv)
    print(f"[stroke] raw shape: {raw.shape}, columns: {list(raw.columns)}")

    n = len(raw)
    col = {c.strip().lower().replace(" ", "_"): c for c in raw.columns}
    df = _scaffold(n, "Alzheimer's Disease", "kaggle_stroke")  # vascular neuro → AD risk proxy

    # Direct column mappings
    if "age" in col:
        df["Age"] = pd.to_numeric(raw[col["age"]], errors="coerce").clip(10, 110)
    if "gender" in col:
        df["Gender"] = (raw[col["gender"]].astype(str).str.lower() == "male").astype(float)
    if "bmi" in col:
        df["BMI"] = pd.to_numeric(raw[col["bmi"]], errors="coerce").clip(10, 60)
    if "hypertension" in col:
        df["Hypertension"] = pd.to_numeric(raw[col["hypertension"]], errors="coerce").clip(0, 1)
    if "heart_disease" in col:
        df["CardiovascularDisease"] = pd.to_numeric(raw[col["heart_disease"]], errors="coerce").clip(0, 1)
    if "avg_glucose_level" in col:
        glucose = pd.to_numeric(raw[col["avg_glucose_level"]], errors="coerce")
        df["Diabetes"] = (glucose > 126).astype(float)  # fasting glucose threshold
    if "smoking_status" in col:
        df["Smoking"] = raw[col["smoking_status"]].astype(str).str.lower().isin(
            ["smokes", "formerly smoked"]
        ).astype(float)
    if "ever_married" in col:
        pass  # not a clinical feature we use

    # Stroke label: stroke patients are high neurological risk
    stroke_col = col.get("stroke")
    if stroke_col:
        df["risk_label"] = pd.to_numeric(raw[stroke_col], errors="coerce").fillna(0).astype(int)
    else:
        df["risk_label"] = 0

    pos = df["risk_label"].mean()
    print(f"[stroke] {n} rows, stroke fraction: {pos:.2%}")
    return df[ALL_FEATURES + META_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Kaggle neurological datasets")
    ap.add_argument("--out-dir", default="data/raw/kaggle")
    ap.add_argument("--only", choices=list(_DATASETS.keys()), nargs="+",
                    help="Download only specific datasets")
    ap.add_argument("--skip-download", action="store_true",
                    help="Process already-downloaded files in out-dir (skip API call)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    token = ""
    if not args.skip_download:
        token = _resolve_token()

    targets = args.only or list(_DATASETS.keys())
    frames: list[pd.DataFrame] = []

    processors = {
        "alzheimers": process_alzheimers,
        "dementia": process_dementia,
        "stroke": process_stroke,
    }

    for name in targets:
        dataset_id = _DATASETS[name]
        folder = out_dir / dataset_id.split("/")[-1]

        if not args.skip_download:
            print(f"\n[{name}] downloading kaggle dataset: {dataset_id} ...")
            try:
                folder = _download_dataset(token, dataset_id, out_dir)
            except Exception as exc:
                print(f"[{name}] download failed: {exc} — skipping")
                continue

        if not folder.exists():
            print(f"[{name}] folder {folder} not found — skipping (run without --skip-download)")
            continue

        try:
            df = processors[name](folder)
            out_path = out_dir / f"kaggle_{name}_v5.parquet"
            df.to_parquet(out_path, index=False)
            print(f"[{name}] saved {len(df)} rows → {out_path}")
            frames.append(df)
        except Exception as exc:
            print(f"[{name}] processing failed: {exc}")
            import traceback; traceback.print_exc()

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        combined_path = out_dir / "kaggle_combined_v5.parquet"
        combined.to_parquet(combined_path, index=False)
        dist = combined["DiseaseType"].value_counts().to_dict()
        print(f"\nKaggle combined: {len(combined)} rows, distribution: {dist}")
        print(f"Saved → {combined_path}")
    else:
        print("No Kaggle data processed.")


if __name__ == "__main__":
    main()
