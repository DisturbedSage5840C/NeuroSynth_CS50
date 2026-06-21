# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""CTGAN-based augmentation for rare disease classes (ALS, Huntington's Disease).

After merging all real data sources into real_v5.parquet, ALS and Huntington's
will likely have <200 samples each. CTGAN learns the joint distribution of the
real samples and generates synthetic ones that are statistically indistinguishable.

Rules:
  - Only augment classes with fewer than MIN_REAL_SAMPLES real samples
  - Synthetic fraction ≤ 30% of any class after augmentation
  - Every synthetic row is tagged: data_source = "ctgan_augmented"
  - Synthetic rows are excluded from drift detection (flagged separately)

Usage:
    python scripts/data/v5/ctgan_augment.py \
        --input data/real_v5.parquet \
        --output data/real_v5_augmented.parquet

Requires: pip install ctgan
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.data.v5.schema import (
    ALL_FEATURES, META_COLS, DISEASE_GENOMIC_PRIORS, POP_DEFAULTS, DISEASE_TYPES,
)

# Augmentation thresholds
MIN_REAL_SAMPLES = 200     # augment classes below this count
MAX_SYNTHETIC_FRACTION = 0.30   # cap: synthetic ≤ 30% of final class size

# CTGAN hyperparameters (tuned for small tabular datasets)
_CTGAN_EPOCHS = 300
_BATCH_SIZE = 500

# Columns to treat as discrete (binary / ordinal) in CTGAN
_DISCRETE_COLUMNS = [
    "Gender", "Ethnicity", "EducationLevel", "Smoking", "AlcoholConsumption",
    "FamilyHistoryAlzheimers", "CardiovascularDisease", "Diabetes", "Depression",
    "HeadInjury", "Hypertension", "MemoryComplaints", "BehavioralProblems",
    "Confusion", "Disorientation", "PersonalityChanges", "DifficultyCompletingTasks",
    "Forgetfulness", "APOE4_dosage",
]

warnings.filterwarnings("ignore", category=FutureWarning)


def _load_ctgan():
    try:
        from ctgan import CTGAN
        return CTGAN
    except ImportError:
        raise ImportError(
            "ctgan not installed. Run: pip install ctgan\n"
            "Or add to requirements.txt and re-install."
        )


def augment_class(
    real_rows: pd.DataFrame,
    target_total: int,
    disease_type: str,
    feature_cols: list[str],
    seed: int = 42,
) -> pd.DataFrame:
    """Train CTGAN on real_rows and generate (target_total - len(real_rows)) synthetic rows."""
    n_real = len(real_rows)
    n_synth = target_total - n_real
    if n_synth <= 0:
        return pd.DataFrame()

    print(f"  [{disease_type}] training CTGAN on {n_real} real rows → generating {n_synth} synthetic...")

    CTGAN = _load_ctgan()

    # Prepare training data: features only (no meta columns in CTGAN input)
    X = real_rows[feature_cols].copy()

    # Drop columns that are entirely NaN (CTGAN cannot train on them)
    non_null_cols = X.columns[X.notna().any()].tolist()
    X = X[non_null_cols]

    # CTGAN requires float for continuous, discrete_columns for categoricals
    discrete_in_data = [c for c in _DISCRETE_COLUMNS if c in X.columns]

    # Fill remaining NaN with column median before feeding to CTGAN (CTGAN doesn't handle NaN)
    col_medians = X.median(numeric_only=True)
    X = X.fillna(col_medians).fillna(0)  # second fillna handles all-NaN-median cols

    ctgan = CTGAN(
        epochs=_CTGAN_EPOCHS,
        batch_size=min(_BATCH_SIZE, max(100, n_real)),
        verbose=False,
        cuda=False,  # CPU-only for free-tier compatibility
    )
    ctgan.fit(X, discrete_columns=discrete_in_data)

    synth_X = ctgan.sample(n_synth, condition_column=None)
    synth_X = synth_X.reset_index(drop=True)

    # Reconstruct full schema rows (fill non-trained columns with NaN)
    synth_df = pd.DataFrame(index=range(n_synth))
    for col in feature_cols:
        if col in synth_X.columns:
            synth_df[col] = synth_X[col].values
        else:
            synth_df[col] = np.nan

    # Copy disease-level metadata from real rows
    synth_df["DiseaseType"] = disease_type
    synth_df["risk_label"] = real_rows["risk_label"].mode().iloc[0]
    synth_df["data_source"] = "ctgan_augmented"
    synth_df["is_synthetic"] = True

    print(f"  [{disease_type}] generated {len(synth_df)} synthetic rows")
    return synth_df


def generate_from_priors(disease_type: str, n: int, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic rows for a disease with 0 real samples using schema priors.
    Used for Huntington's Disease which has no real training data available.
    """
    rng = np.random.default_rng(seed)
    rows = []
    priors = DISEASE_GENOMIC_PRIORS.get(disease_type, {})
    for _ in range(n):
        row = {col: POP_DEFAULTS.get(col, np.nan) for col in ALL_FEATURES}
        row.update(priors)
        # Jitter continuous features ±15% to create variety
        for col, val in row.items():
            if col in _DISCRETE_COLUMNS or not isinstance(val, (int, float)):
                continue
            if np.isfinite(val) and val != 0:
                row[col] = float(val) + rng.normal(0, abs(val) * 0.15)
        row["DiseaseType"] = disease_type
        row["risk_label"] = 1
        row["data_source"] = "prior_augmented"
        rows.append(row)
    df = pd.DataFrame(rows)
    # Clip age to valid range
    if "Age" in df.columns:
        df["Age"] = df["Age"].clip(30, 90)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="CTGAN augmentation for rare disease classes")
    ap.add_argument("--input", default="data/real_v5.parquet")
    ap.add_argument("--output", default="data/real_v5_augmented.parquet")
    ap.add_argument("--min-samples", type=int, default=MIN_REAL_SAMPLES,
                    help=f"Augment classes below this count (default: {MIN_REAL_SAMPLES})")
    ap.add_argument("--max-fraction", type=float, default=MAX_SYNTHETIC_FRACTION,
                    help=f"Max synthetic fraction of final class size (default: {MAX_SYNTHETIC_FRACTION})")
    ap.add_argument("--epochs", type=int, default=_CTGAN_EPOCHS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--diseases", nargs="+",
                    default=["ALS", "Huntington's Disease", "Epilepsy",
                             "Multiple Sclerosis", "Healthy"],
                    help="Disease types to augment (default: all rare classes)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            "Run merge_v5.py first to create real_v5.parquet"
        )

    print(f"Loading {input_path}...")
    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    df["is_synthetic"] = df.get("data_source", "").str.contains("synthetic", na=False)

    print(f"Loaded {len(df)} rows. Disease distribution:")
    dist = df["DiseaseType"].value_counts()
    for disease, count in dist.items():
        real_count = (~df.loc[df["DiseaseType"] == disease, "is_synthetic"]).sum()
        print(f"  {disease:30s}: {count:5d} total ({real_count} real)")

    feature_cols = ALL_FEATURES  # 56 features

    augmented_frames: list[pd.DataFrame] = [df]
    total_synth = 0

    for disease in args.diseases:
        disease_rows = df[df["DiseaseType"] == disease]
        n_real = int((~disease_rows["is_synthetic"]).sum())

        if n_real == 0:
            # No real data: generate from disease-specific schema priors + jitter
            n_to_generate = min(args.min_samples, int(args.min_samples * args.max_fraction))
            print(f"\n[{disease}] 0 real samples — generating {n_to_generate} from schema priors")
            synth_df = generate_from_priors(disease, n_to_generate, args.seed)
            synth_df["is_synthetic"] = True
            augmented_frames.append(synth_df)
            total_synth += len(synth_df)
            continue

        if n_real >= args.min_samples:
            print(f"\n[{disease}] {n_real} real samples ≥ {args.min_samples} threshold — no augmentation needed")
            continue

        # Target: bring up to min_samples, but cap synthetic at max_fraction
        max_synth_allowed = int(n_real / (1 - args.max_fraction) * args.max_fraction)
        n_synth_needed = args.min_samples - n_real
        n_to_generate = min(n_synth_needed, max_synth_allowed)

        if n_to_generate <= 0:
            print(f"\n[{disease}] augmentation cap would not allow enough synthetic samples — skipping")
            continue

        target_total = n_real + n_to_generate
        synth_fraction = n_to_generate / target_total
        print(f"\n[{disease}] {n_real} real → generating {n_to_generate} synthetic "
              f"(target: {target_total}, synthetic fraction: {synth_fraction:.1%})")

        real_rows_only = disease_rows[~disease_rows["is_synthetic"]]
        try:
            synth_df = augment_class(real_rows_only, target_total, disease, feature_cols, args.seed)
            if len(synth_df) > 0:
                augmented_frames.append(synth_df)
                total_synth += len(synth_df)
        except Exception as exc:
            print(f"  [{disease}] CTGAN augmentation failed: {exc}")
            print("  Falling back to bootstrapping (random oversampling with noise)...")
            # Fallback: bootstrap with small noise
            needed = n_to_generate
            bootstrap_idx = np.random.default_rng(args.seed).choice(len(real_rows_only), size=needed, replace=True)
            bootstrap = real_rows_only.iloc[bootstrap_idx].copy().reset_index(drop=True)
            # Add small noise to continuous features
            for col in feature_cols:
                if col in bootstrap.columns and col not in _DISCRETE_COLUMNS:
                    noise = bootstrap[col].std() * 0.05
                    if np.isfinite(noise) and noise > 0:
                        bootstrap[col] += np.random.default_rng(args.seed).normal(0, noise, len(bootstrap))
            bootstrap["data_source"] = "bootstrap_augmented"
            bootstrap["is_synthetic"] = True
            augmented_frames.append(bootstrap)
            total_synth += len(bootstrap)
            print(f"  [{disease}] bootstrapped {needed} rows")

    # Combine and save
    result = pd.concat(augmented_frames, ignore_index=True)

    # Remove internal `is_synthetic` column from output (tracked via data_source instead)
    if "is_synthetic" in result.columns:
        result = result.drop(columns=["is_synthetic"])

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(out_path, index=False)

    print(f"\n=== Augmentation complete ===")
    print(f"Original: {len(df)} rows → Augmented: {len(result)} rows (+{total_synth} synthetic)")
    print(f"Saved → {out_path}")
    print("\nFinal disease distribution:")
    for disease, count in result["DiseaseType"].value_counts().items():
        synth_count = (result.loc[result["DiseaseType"] == disease, "data_source"]
                       .str.contains("augmented|synthetic", na=False).sum())
        print(f"  {disease:30s}: {count:5d} total ({synth_count} synthetic, "
              f"{synth_count/count:.1%} synthetic fraction)")


if __name__ == "__main__":
    main()
