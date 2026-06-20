"""Download clinical sidecars (participants.tsv) from OpenNeuro BIDS datasets.

OpenNeuro hosts neurological BIDS datasets with participants.tsv files
containing real demographics + diagnoses. No account required.

Target datasets (neurological, publicly accessible):
  ds004292  Parkinson's Disease MRI (PPMI-derived)
  ds003843  Parkinson's Disease (De Novo PD)
  ds003653  Epilepsy (intracranial EEG, demographics only)
  ds002393  Multiple Sclerosis (MRI)
  ds003826  Alzheimer's Disease (ADNI-style)
  ds004169  ALS clinical dataset

Usage:
    python scripts/data/v5/scrape_openneuro.py [--out-dir data/raw/openneuro]
    # Optional: pip install openneuro-py (falls back to direct HTTP if unavailable)
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scripts.data.v5.schema import (
    ALL_FEATURES,
    DISEASE_GENOMIC_PRIORS,
    META_COLS,
    POP_DEFAULTS,
)

_OPENNEURO_BASE = "https://openneuro.org/crn/datasets"
_GH_RAW = "https://raw.githubusercontent.com"
_RATE_LIMIT = 1.5  # seconds between requests

# Known neurological OpenNeuro datasets with their expected disease types
_DATASETS = {
    "ds004292": {"disease": "Parkinson's Disease",   "desc": "Parkinson's Disease MRI"},
    "ds003843": {"disease": "Parkinson's Disease",   "desc": "De Novo Parkinson's Disease"},
    "ds003653": {"disease": "Epilepsy",              "desc": "Intracranial EEG Epilepsy"},
    "ds002393": {"disease": "Multiple Sclerosis",    "desc": "Multiple Sclerosis MRI"},
    "ds003826": {"disease": "Alzheimer's Disease",   "desc": "Alzheimer's Disease MRI"},
    "ds004169": {"disease": "ALS",                   "desc": "ALS Clinical Dataset"},
}

# URL templates to try for participants.tsv
_URL_TEMPLATES = [
    "https://openneuro.org/crn/datasets/{dsid}/snapshots/1.0.0/files/participants.tsv",
    "https://openneuro.org/crn/datasets/{dsid}/snapshots/2.0.0/files/participants.tsv",
    "https://openneuro.org/crn/datasets/{dsid}/snapshots/1.0.1/files/participants.tsv",
    # GitHub mirror (some datasets are mirrored)
    "https://raw.githubusercontent.com/OpenNeuroDatasets/{dsid}/main/participants.tsv",
    "https://raw.githubusercontent.com/OpenNeuroDatasets/{dsid}/master/participants.tsv",
]


def _fetch_participants_tsv(dsid: str) -> pd.DataFrame | None:
    """Try multiple URL patterns to fetch participants.tsv for a dataset."""
    for template in _URL_TEMPLATES:
        url = template.format(dsid=dsid)
        try:
            time.sleep(_RATE_LIMIT)
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200 and len(resp.content) > 100:
                df = pd.read_csv(io.BytesIO(resp.content), sep="\t")
                print(f"  [{dsid}] fetched {len(df)} subjects from {url.split('//')[1].split('/')[0]}")
                return df
        except Exception:
            continue
    return None


def _try_openneuro_py(dsid: str, out_dir: Path) -> pd.DataFrame | None:
    """Use openneuro-py library if installed (downloads participants.tsv only)."""
    try:
        import openneuro as on  # type: ignore
        ds_dir = out_dir / dsid
        ds_dir.mkdir(parents=True, exist_ok=True)
        # Download only participants.tsv (not full dataset)
        on.download(
            dataset=dsid,
            include=["participants.tsv", "participants.json"],
            target_dir=str(ds_dir),
        )
        tsv = ds_dir / "participants.tsv"
        if tsv.exists():
            return pd.read_csv(tsv, sep="\t")
    except ImportError:
        pass  # openneuro-py not installed, use HTTP fallback
    except Exception as exc:
        print(f"  [{dsid}] openneuro-py failed: {exc}")
    return None


def _scaffold(n: int, disease_type: str, data_source: str) -> pd.DataFrame:
    # Use NaN for all features — only genomic priors and observed columns will be filled.
    # Leaving unknown features as NaN is more honest than imputing population defaults here;
    # merge_v5.py fills medians after all sources are combined.
    df = pd.DataFrame({col: [np.nan] * n for col in ALL_FEATURES})
    genomic = DISEASE_GENOMIC_PRIORS.get(disease_type, DISEASE_GENOMIC_PRIORS["Alzheimer's Disease"])
    for col, val in genomic.items():
        df[col] = val
    df["DiseaseType"] = disease_type
    df["data_source"] = data_source
    return df


def process_participants_tsv(
    raw: pd.DataFrame,
    dsid: str,
    default_disease: str,
) -> pd.DataFrame:
    """Convert a BIDS participants.tsv to the v5 schema."""
    if raw is None or len(raw) == 0:
        return pd.DataFrame()

    col = {c.strip().lower().replace("-", "_").replace(" ", "_"): c for c in raw.columns}
    n = len(raw)
    df = _scaffold(n, default_disease, f"openneuro_{dsid}")

    # participant_id → skip (not a feature)

    # Demographics
    age_col = col.get("age") or col.get("age_at_visit") or col.get("participant_age")
    if age_col:
        df["Age"] = pd.to_numeric(raw[age_col], errors="coerce").clip(10, 100)

    sex_col = col.get("sex") or col.get("gender")
    if sex_col:
        sex_vals = raw[sex_col].astype(str).str.lower()
        df["Gender"] = sex_vals.isin(["m", "male", "1"]).astype(float)

    educ_col = col.get("education") or col.get("education_years") or col.get("years_education")
    if educ_col:
        df["EducationLevel"] = pd.to_numeric(raw[educ_col], errors="coerce").clip(0, 23) / 23 * 3

    # Diagnosis / group
    group_col = (col.get("group") or col.get("diagnosis") or col.get("diagnosis_group")
                 or col.get("condition") or col.get("pathology"))
    if group_col:
        labels = raw[group_col].astype(str).str.lower()
        is_case = ~labels.isin(["hc", "healthy", "control", "nc", "normal", "td", "0", "cn"])
        df["risk_label"] = is_case.astype(int)
        df.loc[~is_case, "DiseaseType"] = "Healthy"

        # Refine DiseaseType from label if informative
        disease_keywords = {
            "parkinson": "Parkinson's Disease", "pd": "Parkinson's Disease",
            "alzheimer": "Alzheimer's Disease", "ad": "Alzheimer's Disease", "mci": "Alzheimer's Disease",
            "sclerosis": "Multiple Sclerosis", "ms": "Multiple Sclerosis",
            "epilep": "Epilepsy", "seizure": "Epilepsy",
            "als": "ALS", "amyotrophic": "ALS",
            "huntington": "Huntington's Disease",
        }
        for idx, label_val in enumerate(labels):
            label_str = str(label_val) if label_val is not None else ""
            for key, disease in disease_keywords.items():
                if key in label_str and is_case.iloc[idx]:
                    df.loc[idx, "DiseaseType"] = disease
                    break
    else:
        df["risk_label"] = 1  # assume all subjects are cases if no group column

    # MMSE / cognitive scores
    for src_key, dst in [("mmse", "MMSE"), ("moca", "MMSE"), ("cdr", None),
                          ("updrs_total", "UPDRS_total"), ("updrs_motor", "UPDRS_motor"),
                          ("bmi", "BMI")]:
        if src_key in col:
            vals = pd.to_numeric(raw[col[src_key]], errors="coerce")
            if dst:
                df[dst] = vals
            elif src_key == "cdr":
                cdr = vals.fillna(0).clip(0, 3)
                df["FunctionalAssessment"] = (10 - cdr * 3).clip(0, 10)
                df["MemoryComplaints"] = (cdr > 0).astype(float)

    # Genomic priors: update per detected disease type
    for disease in df["DiseaseType"].unique():
        mask = df["DiseaseType"] == disease
        priors = DISEASE_GENOMIC_PRIORS.get(disease, {})
        for gcol, gval in priors.items():
            df.loc[mask, gcol] = gval

    pos = df["risk_label"].mean()
    print(f"  [{dsid}] {n} subjects, case fraction: {pos:.2%}, diseases: {df['DiseaseType'].value_counts().to_dict()}")
    return df[ALL_FEATURES + META_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Scrape OpenNeuro participants.tsv files")
    ap.add_argument("--out-dir", default="data/raw/openneuro")
    ap.add_argument("--datasets", nargs="+", default=list(_DATASETS.keys()),
                    help="OpenNeuro dataset IDs to fetch")
    ap.add_argument("--use-openneuro-py", action="store_true",
                    help="Use openneuro-py library instead of HTTP (requires: pip install openneuro-py)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    failed: list[str] = []

    for dsid in args.datasets:
        meta = _DATASETS.get(dsid, {"disease": "Parkinson's Disease", "desc": dsid})
        print(f"\n[{dsid}] {meta['desc']} (default: {meta['disease']})")

        raw = None
        if args.use_openneuro_py:
            raw = _try_openneuro_py(dsid, out_dir)
        if raw is None:
            raw = _fetch_participants_tsv(dsid)

        if raw is None:
            print(f"  [{dsid}] could not fetch — dataset may require login or different version")
            failed.append(dsid)
            continue

        # Cache raw TSV
        raw.to_csv(out_dir / f"{dsid}_participants.tsv", sep="\t", index=False)

        df = process_participants_tsv(raw, dsid, meta["disease"])
        if len(df) > 0:
            p = out_dir / f"{dsid}_v5.parquet"
            df.to_parquet(p, index=False)
            frames.append(df)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        p = out_dir / "openneuro_combined_v5.parquet"
        combined.to_parquet(p, index=False)
        dist = combined["DiseaseType"].value_counts().to_dict()
        print(f"\nOpenNeuro combined: {len(combined)} rows → {p}")
        print(f"Disease distribution: {dist}")
    else:
        print("\nNo OpenNeuro data processed.")

    if failed:
        print(f"\nFailed datasets: {failed}")
        print("These may require login. Try:")
        print("  1. Visit https://openneuro.org and create a free account")
        print("  2. pip install openneuro-py && openneuro login")
        print("  3. Re-run with --use-openneuro-py")


if __name__ == "__main__":
    main()
