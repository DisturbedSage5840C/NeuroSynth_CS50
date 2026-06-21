# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Merge every available processed data source into one training set.

Picks up whichever of the known sources exist (the realistic synthetic set is
always present; OASIS/PPMI/MIMIC appear once downloaded and processed), aligns
them on the shared schema, drops cross-source duplicates, and writes a single
parquet that ``train.py --data data/merged_training.parquet`` can consume.

Usage:
    python scripts/data/merge_sources.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCES = (
    "data/realistic_v4.parquet",
    "data/realistic_v4.csv",
    "data/oasis3_processed.parquet",
    "data/ppmi_processed.parquet",
    "data/mimic_processed.parquet",
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def main() -> None:
    frames: list[pd.DataFrame] = []
    seen_stems: set[str] = set()
    for s in SOURCES:
        p = Path(s)
        if not p.exists() or p.stem in seen_stems:
            continue
        seen_stems.add(p.stem)
        df = _read(p)
        df["_source"] = p.stem
        frames.append(df)
        print(f"loaded {s}: {len(df)} rows")

    if not frames:
        raise FileNotFoundError(
            "No data sources found. Run scripts/data/build_realistic_synthetic.py first."
        )

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    key = [c for c in ("Age", "MMSE", "Diagnosis") if c in merged.columns]
    if key:
        merged = merged.drop_duplicates(subset=key).reset_index(drop=True)

    out = Path("data/merged_training.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        merged.to_parquet(out, index=False)
    except Exception:
        out = out.with_suffix(".csv")
        merged.to_csv(out, index=False)

    pos = merged["Diagnosis"].mean() if "Diagnosis" in merged.columns else float("nan")
    print(f"merged {len(frames)} sources: {before} -> {len(merged)} rows ({pos:.2%} positive) -> {out}")


if __name__ == "__main__":
    main()
