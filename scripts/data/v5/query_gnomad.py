# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Query gnomAD GraphQL API for neurological disease variant frequencies.

gnomAD is a population-level database — it provides allele frequencies for
variants across the general population, not per-patient records. This script:

  1. Queries the gnomAD v4 GraphQL API for pathogenic variants in 15 neuro genes
  2. Computes population-level risk scores per disease category
  3. Saves a reference table: data/raw/gnomad/variant_frequencies.json
  4. Saves per-disease genomic feature rows: data/raw/gnomad/gnomad_features.parquet

These reference features are merged into real_v5.parquet by merge_v5.py to
populate the GENOMIC_4 columns for patients from non-genomic datasets.

gnomAD API: https://gnomad.broadinstitute.org/api  (no auth required)

Usage:
    python scripts/data/v5/query_gnomad.py
"""
from __future__ import annotations
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from scripts.data.v5.schema import (
    ALL_FEATURES,
    DISEASE_GENOMIC_PRIORS,
    DISEASE_TYPES,
    META_COLS,
    POP_DEFAULTS,
)

_GNOMAD_API = "https://gnomad.broadinstitute.org/api"
_RATE_LIMIT_DELAY = 2.0  # seconds between gene queries

# 15-gene neurological disease panel (from existing gnomad.py + plan additions)
NEURO_GENE_PANEL = {
    # Alzheimer's Disease
    "APOE":   {"disease": "Alzheimer's Disease", "chr": "19"},
    "APP":    {"disease": "Alzheimer's Disease", "chr": "21"},
    "PSEN1":  {"disease": "Alzheimer's Disease", "chr": "14"},
    "PSEN2":  {"disease": "Alzheimer's Disease", "chr": "1"},
    "TREM2":  {"disease": "Alzheimer's Disease", "chr": "6"},
    # Parkinson's Disease
    "LRRK2":  {"disease": "Parkinson's Disease", "chr": "12"},
    "SNCA":   {"disease": "Parkinson's Disease", "chr": "4"},
    "GBA":    {"disease": "Parkinson's Disease", "chr": "1"},
    # ALS
    "SOD1":   {"disease": "ALS", "chr": "21"},
    "C9orf72": {"disease": "ALS", "chr": "9"},
    "FUS":    {"disease": "ALS", "chr": "16"},
    "TARDBP": {"disease": "ALS", "chr": "1"},
    # Huntington's Disease
    "HTT":    {"disease": "Huntington's Disease", "chr": "4"},
    # MS / Epilepsy (channelopathies + immune)
    "SCN1A":  {"disease": "Epilepsy", "chr": "2"},
    "KCNQ2":  {"disease": "Epilepsy", "chr": "20"},
}

_GENE_QUERY = """
query GeneVariants($geneSymbol: String!, $datasetId: DatasetId!) {
  gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
    gene_id
    symbol
    variants(dataset: $datasetId) {
      variant_id
      consequence
      hgvsp
      exome { ac an af }
      genome { ac an af }
    }
  }
}
"""

_PATHOGENIC_CONSEQUENCES = {
    "missense_variant",
    "frameshift_variant",
    "stop_gained",
    "splice_donor_variant",
    "splice_acceptor_variant",
    "start_lost",
}


def query_gene(gene: str, dataset_id: str = "gnomad_r4", retries: int = 3) -> dict[str, Any]:
    """Query one gene from gnomAD, return summary dict."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                _GNOMAD_API,
                json={
                    "query": _GENE_QUERY,
                    "variables": {"geneSymbol": gene, "datasetId": dataset_id},
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            break
        except Exception as exc:
            if attempt == retries - 1:
                print(f"  [{gene}] failed after {retries} attempts: {exc}")
                return {"gene": gene, "error": str(exc), "pathogenic_variants": [], "n_pathogenic": 0}
            wait = (attempt + 1) * 3
            print(f"  [{gene}] attempt {attempt+1} failed, retry in {wait}s...")
            time.sleep(wait)

    gene_data = data.get("data", {}).get("gene")
    if not gene_data:
        return {"gene": gene, "error": "not found", "pathogenic_variants": [], "n_pathogenic": 0}

    variants = gene_data.get("variants", [])
    pathogenic = []
    for v in variants:
        if v.get("consequence") not in _PATHOGENIC_CONSEQUENCES:
            continue
        exome_af = (v.get("exome") or {}).get("af") or 0
        genome_af = (v.get("genome") or {}).get("af") or 0
        combined_af = max(float(exome_af), float(genome_af))
        pathogenic.append({
            "variant_id": v.get("variant_id"),
            "consequence": v.get("consequence"),
            "hgvsp": v.get("hgvsp"),
            "allele_frequency": round(combined_af, 8),
        })

    # Compute max pathogenic AF (most common disease-associated variant)
    max_af = max((p["allele_frequency"] for p in pathogenic), default=0.0)
    sum_af = sum(p["allele_frequency"] for p in pathogenic)

    result = {
        "gene": gene,
        "disease": NEURO_GENE_PANEL[gene]["disease"],
        "n_total_variants": len(variants),
        "n_pathogenic": len(pathogenic),
        "max_pathogenic_af": max_af,
        "sum_pathogenic_af": round(sum_af, 8),
        "pathogenic_variants": pathogenic[:20],  # keep top 20 to limit file size
    }
    print(f"  [{gene}] {len(variants)} variants, {len(pathogenic)} pathogenic, max_AF={max_af:.2e}")
    return result


def compute_disease_risk_scores(summaries: list[dict]) -> dict[str, dict[str, float]]:
    """
    Convert per-gene pathogenic variant summaries into per-disease risk scores.
    These scores become the GENOMIC_4 features in the v5 schema.
    """
    from collections import defaultdict
    by_disease: dict[str, list[dict]] = defaultdict(list)
    for s in summaries:
        if "error" not in s:
            by_disease[s["disease"]].append(s)

    scores: dict[str, dict[str, float]] = {}
    for disease, gene_summaries in by_disease.items():
        total_pathogenic = sum(g["n_pathogenic"] for g in gene_summaries)
        max_af = max((g["max_pathogenic_af"] for g in gene_summaries), default=0.0)

        # PRS proxy: normalized pathogenic burden (0–3 scale)
        prs = min(total_pathogenic * 0.08, 3.0)

        # Gene-specific feature assignment
        apoe_summary = next((g for g in gene_summaries if g["gene"] == "APOE"), None)
        lrrk2_summary = next((g for g in gene_summaries if g["gene"] == "LRRK2"), None)
        htt_summary = next((g for g in gene_summaries if g["gene"] == "HTT"), None)

        apoe_risk = (apoe_summary["n_pathogenic"] * 0.3) if apoe_summary else 0.0
        lrrk2_freq = (lrrk2_summary["max_pathogenic_af"] * 100) if lrrk2_summary else 0.0

        scores[disease] = {
            "APOE_risk_score": round(min(apoe_risk, 3.0), 4),
            "LRRK2_variant_freq": round(min(lrrk2_freq, 0.05), 6),
            "HTT_repeat_est": 42.0 if "huntington" in disease.lower() else 17.0,
            "polygenic_risk_score": round(prs, 4),
        }
        print(f"  [{disease}] PRS={prs:.3f}, APOE_risk={scores[disease]['APOE_risk_score']:.3f}")

    return scores


def build_reference_rows(disease_scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    """
    Build one representative row per disease type for the GENOMIC_4 features.
    These are merged into real_v5.parquet for datasets that lack genomic data.
    """
    rows = []
    for disease in DISEASE_TYPES:
        scores = disease_scores.get(disease, DISEASE_GENOMIC_PRIORS.get(disease, {}))
        # Use gnomAD-computed scores where available, fall back to literature priors
        final = {**DISEASE_GENOMIC_PRIORS.get(disease, {}), **scores}
        row = {col: POP_DEFAULTS.get(col, np.nan) for col in ALL_FEATURES}
        row.update(final)
        row["DiseaseType"] = disease
        row["risk_label"] = 0 if disease == "Healthy" else 1
        row["data_source"] = "gnomad_reference"
        rows.append(row)
    return pd.DataFrame(rows)[ALL_FEATURES + META_COLS]


def main() -> None:
    ap = argparse.ArgumentParser(description="Query gnomAD for neurological disease variant frequencies")
    ap.add_argument("--out-dir", default="data/raw/gnomad")
    ap.add_argument("--genes", nargs="+", default=list(NEURO_GENE_PANEL.keys()),
                    help="Genes to query (default: all 15)")
    ap.add_argument("--dataset", default="gnomad_r4",
                    help="gnomAD dataset ID (default: gnomad_r4 = latest)")
    ap.add_argument("--skip-api", action="store_true",
                    help="Use cached variant_frequencies.json instead of querying API")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "variant_frequencies.json"

    if args.skip_api and cache_path.exists():
        print(f"Loading cached results from {cache_path}")
        summaries = json.loads(cache_path.read_text())
    else:
        # Test API connectivity first
        print("Testing gnomAD API connectivity...")
        try:
            resp = requests.post(_GNOMAD_API, json={"query": "{ meta { gnomad_version } }"}, timeout=15)
            resp.raise_for_status()
            version = resp.json().get("data", {}).get("meta", {}).get("gnomad_version", "unknown")
            print(f"gnomAD API OK — version: {version}")
        except Exception as exc:
            print(f"gnomAD API unreachable: {exc}")
            print("Using literature-based priors from schema.py as fallback.")
            ref_df = build_reference_rows({})
            ref_df.to_parquet(out_dir / "gnomad_features.parquet", index=False)
            return

        summaries = []
        print(f"\nQuerying {len(args.genes)} genes from gnomAD ({args.dataset})...")
        for gene in args.genes:
            if gene not in NEURO_GENE_PANEL:
                print(f"  [{gene}] not in panel, skipping")
                continue
            print(f"  Querying {gene} ({NEURO_GENE_PANEL[gene]['disease']})...")
            result = query_gene(gene, args.dataset)
            summaries.append(result)
            time.sleep(_RATE_LIMIT_DELAY)

        cache_path.write_text(json.dumps(summaries, indent=2))
        print(f"\nSaved variant frequencies → {cache_path}")

    # Compute disease-level risk scores from gnomAD data
    print("\nComputing disease risk scores...")
    disease_scores = compute_disease_risk_scores(summaries)

    # Save the reference score table
    scores_path = out_dir / "disease_risk_scores.json"
    scores_path.write_text(json.dumps(disease_scores, indent=2))
    print(f"Saved disease risk scores → {scores_path}")

    # Build reference DataFrame
    ref_df = build_reference_rows(disease_scores)
    ref_path = out_dir / "gnomad_features.parquet"
    ref_df.to_parquet(ref_path, index=False)
    print(f"Saved reference rows ({len(ref_df)}) → {ref_path}")

    # Print summary
    print("\n=== gnomAD Summary ===")
    for s in summaries:
        if "error" not in s:
            print(f"  {s['gene']:8s} ({s['disease']:25s}): {s['n_pathogenic']:3d} pathogenic, max_AF={s['max_pathogenic_af']:.2e}")


if __name__ == "__main__":
    main()
