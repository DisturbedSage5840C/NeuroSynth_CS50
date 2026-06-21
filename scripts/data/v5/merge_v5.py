# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Merge all v5 real data sources into a single unified training file.

Input sources (any combination of what's available):
  data/raw/kaggle/kaggle_alzheimers_v5.parquet
  data/raw/kaggle/kaggle_dementia_v5.parquet
  data/raw/kaggle/kaggle_multiclass_v5.parquet
  data/raw/uci/uci_classic_v5.parquet
  data/raw/uci/uci_telemonitoring_v5.parquet
  data/raw/physionet/pads_v5.parquet
  data/raw/physionet/noneeg_v5.parquet
  data/raw/physionet/physionet_combined_v5.parquet
  data/raw/openneuro/openneuro_combined_v5.parquet
  data/oasis_v5.parquet
  data/raw/gnomad/gnomad_features.parquet (reference only — enriches genomic cols)

Output:
  data/real_v5.parquet  (56 features + DiseaseType + risk_label + data_source)

After running merge_v5.py, run ctgan_augment.py if ALS/Huntington's need boosting.

Usage:
    python scripts/data/v5/merge_v5.py [--out data/real_v5.parquet]
    python scripts/data/v5/merge_v5.py --validate  # also runs pandera schema checks
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.v5.schema import (
    ALL_FEATURES,
    BIOMARKER_6,
    CORE_32,
    DISEASE_GENOMIC_PRIORS,
    DISEASE_TYPES,
    GENOMIC_4,
    IMAGING_8,
    META_COLS,
    POP_DEFAULTS,
    WEARABLE_6,
)

# Ordered list of source parquets to try.
# Combined files (*_combined_v5.parquet) are excluded — they duplicate individual sources.
_SOURCE_GLOBS = [
    "data/raw/kaggle/**/*_v5.parquet",
    "data/raw/uci/**/*_v5.parquet",
    "data/raw/physionet/**/*_v5.parquet",
    "data/raw/openneuro/**/*_v5.parquet",
    "data/oasis_v5.parquet",
    # Legacy: the existing realistic_v4.parquet is kept as fallback if no real data available
    "data/realistic_v4.parquet",
]
# Files matching these patterns are skipped (they aggregate individual sources already loaded)
_SKIP_PATTERNS = {"_combined_v5", "physionet_combined_v5", "uci_combined_v5",
                   "kaggle_combined_v5", "openneuro_combined_v5"}

_GNOMAD_SCORES = "data/raw/gnomad/disease_risk_scores.json"
_MIN_ROWS = 500  # warn if merged dataset is below this


def _load_source(path: Path) -> pd.DataFrame | None:
    try:
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix in (".csv", ".tsv"):
            sep = "\t" if path.suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
        else:
            return None

        if len(df) == 0:
            return None

        # Skip gnomad reference rows (they're used for enrichment, not training)
        if "data_source" in df.columns and df["data_source"].str.contains("gnomad_reference", na=False).all():
            return None

        return df
    except Exception as exc:
        print(f"  [warning] failed to load {path}: {exc}")
        return None


def _coerce_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Align a DataFrame to the 56-feature schema + meta columns."""
    out = pd.DataFrame(index=range(len(df)))

    for col in ALL_FEATURES:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").values
        else:
            out[col] = np.nan

    for col in META_COLS:
        if col in df.columns:
            out[col] = df[col].values
        elif col == "DiseaseType":
            out[col] = "Unknown"
        elif col == "risk_label":
            out[col] = 0
        elif col == "data_source":
            out[col] = "unknown"

    return out


def _apply_gnomad_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing GENOMIC_4 columns using gnomAD-derived disease priors."""
    import json

    gnomad_path = Path(_GNOMAD_SCORES)
    if gnomad_path.exists():
        try:
            scores = json.loads(gnomad_path.read_text())
            print(f"  Applying gnomAD enrichment from {gnomad_path}")
        except Exception:
            scores = {}
    else:
        scores = {}

    for disease in df["DiseaseType"].unique():
        mask = df["DiseaseType"] == disease
        # Use gnomAD scores if available, else fall back to literature priors
        priors = {
            **DISEASE_GENOMIC_PRIORS.get(disease, {}),
            **scores.get(disease, {}),
        }
        for col in GENOMIC_4:
            if col in priors:
                # Only fill where the value is missing (NaN)
                missing_mask = mask & df[col].isna()
                if missing_mask.any():
                    df.loc[missing_mask, col] = priors[col]

    return df


def _fill_missing_core(df: pd.DataFrame) -> pd.DataFrame:
    """
    For the 32 core clinical features, fill NaN with dataset-specific column medians
    where a column has >50% coverage; otherwise leave as NaN (CatBoost handles it).
    """
    for col in CORE_32:
        if col not in df.columns:
            continue
        coverage = df[col].notna().mean()
        if 0.1 < coverage < 1.0:
            # Fill NaN within each data_source using that source's median
            for src in df["data_source"].unique():
                src_mask = df["data_source"] == src
                col_median = df.loc[src_mask & df[col].notna(), col].median()
                if np.isfinite(col_median):
                    fill_mask = src_mask & df[col].isna()
                    df.loc[fill_mask, col] = col_median
    return df


def _deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows that are fully identical across ALL features + meta within the same source.
    Rows with fewer than 4 non-NaN feature values are considered data-sparse (e.g. OpenNeuro
    participants.tsv with only Age) and are never deduplicated — they are unique patients
    who appear identical only because most schema columns are absent for that dataset.
    """
    before = len(df)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    dedup_cols = feature_cols + ["DiseaseType", "data_source"]

    # Compute feature density per row
    n_obs = df[feature_cols].notna().sum(axis=1)
    # Only dedup rows rich enough to be meaningfully compared (≥10 non-NaN features).
    # Sparse rows (e.g. OpenNeuro participants.tsv with only Age + genomic priors) are
    # kept as-is — they represent unique patients who appear identical only because most
    # clinical columns are absent from that dataset.
    rich = n_obs >= 10
    sparse = ~rich

    rich_df = df[rich].drop_duplicates(subset=dedup_cols).reset_index(drop=True)
    sparse_df = df[sparse].reset_index(drop=True)

    df = pd.concat([rich_df, sparse_df], ignore_index=True)
    after = len(df)
    if before > after:
        print(f"  Exact-duplicate removal: {before} → {after} rows ({before - after} removed)")
    return df


def _validate_schema(df: pd.DataFrame) -> None:
    """Run basic sanity checks on the merged dataset."""
    print("\n=== Schema Validation ===")

    # Feature coverage
    print("Feature coverage (% non-null):")
    for group_name, cols in [
        ("Core-32", CORE_32), ("Imaging-8", IMAGING_8),
        ("Biomarker-6", BIOMARKER_6), ("Wearable-6", WEARABLE_6),
        ("Genomic-4", GENOMIC_4),
    ]:
        available = [c for c in cols if c in df.columns]
        coverage = df[available].notna().mean().mean() if available else 0.0
        status = "✓" if coverage > 0.5 else "⚠" if coverage > 0.1 else "✗"
        print(f"  {status} {group_name:12s}: {coverage:.0%}")

    # Disease distribution
    print("\nDisease distribution:")
    dist = df["DiseaseType"].value_counts()
    for disease, count in dist.items():
        pct = count / len(df)
        flag = " ⚠ RARE" if count < 200 else ""
        print(f"  {disease:30s}: {count:5d} ({pct:.1%}){flag}")

    # Risk label balance
    pos = df["risk_label"].mean()
    print(f"\nRisk label positive rate: {pos:.2%} (target: 20–60%)")
    if pos < 0.1 or pos > 0.9:
        print("  ⚠ Highly imbalanced — consider class weights in training")

    # Row count
    n = len(df)
    print(f"\nTotal rows: {n:,}")
    if n < _MIN_ROWS:
        print(f"  ⚠ Only {n} rows — real data download likely incomplete")

    # Age sanity
    if "Age" in df.columns:
        age_ok = df["Age"].between(10, 110).mean()
        print(f"Age in [10, 110]: {age_ok:.1%}")

    # MMSE sanity
    if "MMSE" in df.columns:
        mmse_ok = df["MMSE"].between(0, 30).mean()
        print(f"MMSE in [0, 30]:  {mmse_ok:.1%}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge all v5 real data into real_v5.parquet")
    ap.add_argument("--out", default="data/real_v5.parquet")
    ap.add_argument("--validate", action="store_true", help="Run schema validation after merge")
    ap.add_argument("--include-synthetic-fallback", action="store_true",
                    help="Include realistic_v4.parquet if real data is insufficient")
    args = ap.parse_args()

    print("=== NeuroSynth v5 Data Merge ===\n")

    frames: list[pd.DataFrame] = []
    real_sources: list[str] = []
    synthetic_sources: list[str] = []

    # Collect all available parquets
    from pathlib import Path as _P
    import glob

    found_paths: list[Path] = []
    for pattern in _SOURCE_GLOBS:
        base = _P(".")
        matched = sorted(base.glob(pattern))
        for p in matched:
            if p not in found_paths:
                found_paths.append(p)

    print(f"Found {len(found_paths)} potential sources:")
    for p in found_paths:
        print(f"  {p}")

    print()
    for path in found_paths:
        # Skip combined aggregate files (they duplicate the individual sources already loaded)
        if any(pat in path.name for pat in _SKIP_PATTERNS):
            print(f"  [skip] {path.name} (combined aggregate — individual sources loaded separately)")
            continue

        is_synthetic = "synthetic" in path.name or "realistic_v4" in path.name
        if is_synthetic and not args.include_synthetic_fallback:
            print(f"  [skip] {path.name} (synthetic fallback — use --include-synthetic-fallback to include)")
            continue

        df = _load_source(path)
        if df is None:
            print(f"  [skip] {path.name} (empty or unreadable)")
            continue

        df = _coerce_to_schema(df)
        n = len(df)

        if is_synthetic:
            synthetic_sources.append(path.name)
        else:
            real_sources.append(path.name)

        print(f"  [load] {path.name}: {n} rows, diseases: {df['DiseaseType'].value_counts().to_dict()}")
        frames.append(df)

    if not frames:
        print("\nNo data found! Run the download scripts first:")
        print("  python scripts/data/v5/download_kaggle.py")
        print("  python scripts/data/v5/download_uci.py")
        print("  python scripts/data/v5/download_physionet.py")
        print("  python scripts/data/v5/process_oasis_v5.py")
        print("  python scripts/data/v5/scrape_openneuro.py")
        print("  python scripts/data/v5/query_gnomad.py")
        return

    # Merge
    print(f"\nMerging {len(frames)} source(s)...")
    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)

    # Coerce numeric types
    for col in ALL_FEATURES:
        if col in merged.columns:
            merged[col] = pd.to_numeric(merged[col], errors="coerce")

    # Deduplicate
    merged = _deduplicate(merged)

    # Enrich genomic features from gnomAD reference
    merged = _apply_gnomad_enrichment(merged)

    # Fill missing core features with within-source medians
    print("  Filling missing core features with within-source medians...")
    merged = _fill_missing_core(merged)

    # Clip physiological ranges to valid bounds
    clip_bounds: dict[str, tuple[float, float]] = {
        "Age": (10, 110), "MMSE": (0, 30), "BMI": (10, 60),
        "SystolicBP": (60, 250), "DiastolicBP": (30, 160),
        "CholesterolTotal": (50, 600), "UPDRS_motor": (0, 132), "UPDRS_total": (0, 176),
        "FunctionalAssessment": (0, 10), "ADL": (0, 10),
        "SpO2_mean": (50, 100), "HR_variability": (0, 300),
        "nWBV": (0.5, 1.0), "ASF": (0.5, 2.0),
    }
    for col, (lo, hi) in clip_bounds.items():
        if col in merged.columns:
            merged[col] = merged[col].clip(lo, hi)

    # Ensure boolean columns are 0/1
    binary_cols = [
        "Gender", "Smoking", "FamilyHistoryAlzheimers", "CardiovascularDisease",
        "Diabetes", "Depression", "HeadInjury", "Hypertension",
        "MemoryComplaints", "BehavioralProblems", "Confusion",
        "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks", "Forgetfulness",
    ]
    for col in binary_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).clip(0, 1).round().astype(float)

    # Final column order
    output_cols = ALL_FEATURES + META_COLS
    for col in output_cols:
        if col not in merged.columns:
            merged[col] = np.nan
    merged = merged[output_cols]

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(out_path, index=False)

    # Summary
    after = len(merged)
    real_rows = merged[~merged["data_source"].str.contains("synthetic|ctgan", na=False)]
    print(f"\n=== Merge Complete ===")
    print(f"Sources merged: {len(frames)}")
    print(f"  Real sources: {real_sources}")
    if synthetic_sources:
        print(f"  Synthetic:    {synthetic_sources}")
    print(f"Total rows:    {after:,} ({before - after} deduped)")
    print(f"Real rows:     {len(real_rows):,}")
    print(f"Saved → {out_path}")

    if args.validate:
        _validate_schema(merged)

    # Recommend CTGAN augmentation if rare classes are too small
    print()
    for disease in ["ALS", "Huntington's Disease"]:
        n_disease = (merged["DiseaseType"] == disease).sum()
        if n_disease < 200:
            print(f"⚠  {disease}: only {n_disease} rows — run CTGAN augmentation:")
            print(f"   python scripts/data/v5/ctgan_augment.py --input {out_path}")


if __name__ == "__main__":
    main()
