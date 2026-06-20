"""Fetch neurological PubMed abstracts via NCBI E-utilities.

Saves records to a JSONL file consumed by embed_corpus.py.
Uses Biopython's Bio.Entrez (already in requirements via biopython>=1.83).

Rate limits:
  • Without NCBI_API_KEY: 3 requests/second
  • With NCBI_API_KEY:    10 requests/second (set in env)

Usage:
    python scripts/data/v5/build_pubmed_corpus.py
    python scripts/data/v5/build_pubmed_corpus.py --max 5000 --out data/pubmed_corpus.jsonl
    python scripts/data/v5/build_pubmed_corpus.py --email you@example.com
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Search terms per disease — MeSH + free-text hybrid for breadth
DISEASE_QUERIES: dict[str, str] = {
    "alzheimer": (
        '("Alzheimer Disease"[MeSH] OR "Alzheimer\'s disease"[tiab]) '
        'AND ("biomarker"[tiab] OR "risk factor"[tiab] OR "cognitive"[tiab] '
        'OR "amyloid"[tiab] OR "tau"[tiab]) '
        'AND hasabstract[text] AND "2015/01/01"[PDAT]:"3000"[PDAT]'
    ),
    "parkinson": (
        '("Parkinson Disease"[MeSH] OR "Parkinson\'s disease"[tiab]) '
        'AND ("alpha-synuclein"[tiab] OR "UPDRS"[tiab] OR "dopamine"[tiab] '
        'OR "LRRK2"[tiab] OR "motor"[tiab]) '
        'AND hasabstract[text] AND "2015/01/01"[PDAT]:"3000"[PDAT]'
    ),
    "multiple_sclerosis": (
        '("Multiple Sclerosis"[MeSH] OR "multiple sclerosis"[tiab]) '
        'AND ("MRI"[tiab] OR "lesion"[tiab] OR "myelin"[tiab] '
        'OR "relapse"[tiab] OR "demyelination"[tiab]) '
        'AND hasabstract[text] AND "2015/01/01"[PDAT]:"3000"[PDAT]'
    ),
    "epilepsy": (
        '("Epilepsy"[MeSH] OR "epilepsy"[tiab] OR "seizure disorder"[tiab]) '
        'AND ("EEG"[tiab] OR "antiepileptic"[tiab] OR "drug-resistant"[tiab] '
        'OR "SCN1A"[tiab]) '
        'AND hasabstract[text] AND "2015/01/01"[PDAT]:"3000"[PDAT]'
    ),
    "als": (
        '("Amyotrophic Lateral Sclerosis"[MeSH] OR "ALS"[tiab] '
        'OR "motor neuron disease"[tiab]) '
        'AND ("SOD1"[tiab] OR "C9orf72"[tiab] OR "TDP-43"[tiab] '
        'OR "progression"[tiab] OR "FRS-ALS"[tiab]) '
        'AND hasabstract[text] AND "2015/01/01"[PDAT]:"3000"[PDAT]'
    ),
    "huntington": (
        '("Huntington Disease"[MeSH] OR "Huntington\'s disease"[tiab]) '
        'AND ("CAG repeat"[tiab] OR "HTT"[tiab] OR "striatum"[tiab] '
        'OR "neurodegeneration"[tiab]) '
        'AND hasabstract[text] AND "2010/01/01"[PDAT]:"3000"[PDAT]'
    ),
}

# PMIDs per disease — more for common diseases, rarer diseases get quota boost
QUOTA_PER_DISEASE: dict[str, int] = {
    "alzheimer": 2500,
    "parkinson": 2500,
    "multiple_sclerosis": 1500,
    "epilepsy": 1500,
    "als": 1500,
    "huntington": 500,
}


def _entrez_search(term: str, retmax: int, email: str, api_key: str | None) -> list[str]:
    """Run an eSearch and return up to retmax PMIDs."""
    from Bio import Entrez  # type: ignore
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    handle = Entrez.esearch(db="pubmed", term=term, retmax=retmax, usehistory="y")
    record = Entrez.read(handle)
    handle.close()
    return list(record.get("IdList", []))


def _entrez_fetch_batch(pmids: list[str], email: str, api_key: str | None) -> list[dict]:
    """Fetch XML records for a batch of PMIDs and parse abstract + metadata."""
    from Bio import Entrez, Medline  # type: ignore
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    ids = ",".join(pmids)
    handle = Entrez.efetch(db="pubmed", id=ids, rettype="medline", retmode="text")
    records = list(Medline.parse(handle))
    handle.close()

    out: list[dict] = []
    for rec in records:
        pmid = rec.get("PMID", "").strip()
        abstract = rec.get("AB", "").strip()
        if not pmid or not abstract:
            continue
        out.append({
            "pmid": pmid,
            "title": rec.get("TI", "").strip(),
            "abstract": abstract,
            "journal": rec.get("TA", "").strip(),
            "pub_year": _parse_year(rec.get("DP", "")),
            "diseases": [],  # filled in by caller
        })
    return out


def _parse_year(date_str: str) -> int | None:
    """Extract year from a date string like '2021 Jan' or '2021'."""
    if not date_str:
        return None
    try:
        return int(date_str.split()[0])
    except (ValueError, IndexError):
        return None


def fetch_corpus(
    max_total: int,
    email: str,
    api_key: str | None,
    batch_size: int = 200,
    sleep_sec: float = 0.35,   # ~3 req/s without key; 0.11 with key
) -> list[dict]:
    """Fetch up to max_total abstracts across all 6 disease groups."""
    all_docs: list[dict] = []
    seen_pmids: set[str] = set()

    for disease, query in DISEASE_QUERIES.items():
        quota = QUOTA_PER_DISEASE.get(disease, 500)
        quota = min(quota, max_total - len(all_docs))
        if quota <= 0:
            break

        log.info("Searching PubMed for '%s' (quota %d) …", disease, quota)
        try:
            pmids = _entrez_search(query, retmax=quota * 2, email=email, api_key=api_key)
        except Exception as exc:
            log.warning("eSearch failed for %s: %s", disease, exc)
            continue

        time.sleep(sleep_sec)

        # Filter already-seen PMIDs
        new_pmids = [p for p in pmids if p not in seen_pmids][:quota]
        log.info("  Found %d PMIDs, fetching %d unique …", len(pmids), len(new_pmids))

        for i in range(0, len(new_pmids), batch_size):
            batch = new_pmids[i: i + batch_size]
            try:
                docs = _entrez_fetch_batch(batch, email=email, api_key=api_key)
            except Exception as exc:
                log.warning("  Fetch batch %d failed: %s", i, exc)
                time.sleep(2.0)
                continue

            for doc in docs:
                doc["diseases"] = [disease]
                seen_pmids.add(doc["pmid"])
                all_docs.append(doc)

            log.info("  Fetched batch %d-%d → %d total docs", i, i + len(batch), len(all_docs))
            time.sleep(sleep_sec)

            if len(all_docs) >= max_total:
                break

        if len(all_docs) >= max_total:
            break

    return all_docs


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch PubMed neurology corpus")
    ap.add_argument("--max", type=int, default=10_000, help="Max total abstracts")
    ap.add_argument("--out", default="data/pubmed_corpus.jsonl")
    ap.add_argument("--email", default=os.getenv("NCBI_EMAIL", "neurosynth@example.com"))
    ap.add_argument("--api-key", default=os.getenv("NCBI_API_KEY", ""))
    ap.add_argument("--batch-size", type=int, default=200)
    args = ap.parse_args()

    try:
        from Bio import Entrez  # noqa: F401
    except ImportError:
        log.error("Biopython not installed. Run: pip install biopython")
        raise SystemExit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("=== PubMed Corpus Fetcher ===")
    log.info("Target: %d abstracts  |  Output: %s", args.max, out_path)

    docs = fetch_corpus(
        max_total=args.max,
        email=args.email,
        api_key=args.api_key or None,
        batch_size=args.batch_size,
        sleep_sec=0.11 if args.api_key else 0.35,
    )

    with out_path.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    log.info("Saved %d abstracts → %s", len(docs), out_path)

    # Disease breakdown
    from collections import Counter
    breakdown = Counter(d["diseases"][0] for d in docs if d["diseases"])
    for disease, count in sorted(breakdown.items()):
        log.info("  %-20s %d", disease, count)


if __name__ == "__main__":
    main()
