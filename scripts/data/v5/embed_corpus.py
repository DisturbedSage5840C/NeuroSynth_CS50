"""Embed PubMed corpus and store embeddings in Neon pgvector.

Reads the JSONL produced by build_pubmed_corpus.py, batches abstracts,
calls OpenAI text-embedding-3-small, then upserts into the
literature_embeddings table via asyncpg.

Requires:
  - OPENAI_API_KEY env var (or --api-key flag)
  - DATABASE_URL (postgres DSN) set in environment / .env

Usage:
    python scripts/data/v5/embed_corpus.py
    python scripts/data/v5/embed_corpus.py --input data/pubmed_corpus.jsonl --batch-size 100
    python scripts/data/v5/embed_corpus.py --dry-run   # counts records, no DB writes
"""
from __future__ import annotations

import argparse
import asyncio
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

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_TOKENS_PER_INPUT = 8_000  # chars (conservative — model limit is 8192 tokens)


def _truncate(text: str, max_chars: int = MAX_TOKENS_PER_INPUT) -> str:
    return text[:max_chars]


def _embed_batch_openai(texts: list[str], api_key: str) -> list[list[float]]:
    """Call OpenAI Embeddings API for a batch of texts. Returns list of vectors."""
    import openai  # type: ignore

    client = openai.OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[_truncate(t) for t in texts],
    )
    # response.data is sorted by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]


async def _upsert_batch(db, batch: list[dict], embeddings: list[list[float]]) -> None:
    """Upsert a batch of docs + embeddings into literature_embeddings."""
    for doc, emb in zip(batch, embeddings):
        await db.upsert_abstract(
            pmid=doc["pmid"],
            title=doc.get("title", ""),
            abstract=doc.get("abstract", ""),
            journal=doc.get("journal", ""),
            pub_year=doc.get("pub_year"),
            diseases=doc.get("diseases", []),
            embedding=emb,
        )


async def embed_and_store(
    docs: list[dict],
    api_key: str,
    batch_size: int = 100,
    sleep_sec: float = 0.5,
    dry_run: bool = False,
) -> None:
    from backend.db import get_db

    db = get_db()
    await db.connect()

    total = len(docs)
    stored = 0
    failed = 0

    for i in range(0, total, batch_size):
        batch = docs[i: i + batch_size]
        texts = [
            f"{doc.get('title', '')}. {doc.get('abstract', '')}" for doc in batch
        ]

        try:
            embeddings = _embed_batch_openai(texts, api_key)
        except Exception as exc:
            log.warning("Embedding batch %d-%d failed: %s", i, i + len(batch), exc)
            failed += len(batch)
            time.sleep(2.0)
            continue

        if not dry_run:
            try:
                await _upsert_batch(db, batch, embeddings)
                stored += len(batch)
            except Exception as exc:
                log.warning("DB upsert batch %d-%d failed: %s", i, i + len(batch), exc)
                failed += len(batch)
                continue
        else:
            stored += len(batch)

        log.info(
            "Embedded %d/%d  (stored=%d, failed=%d)",
            min(i + len(batch), total), total, stored, failed,
        )
        time.sleep(sleep_sec)

    await db.disconnect()
    log.info("Done. Total: %d | Stored: %d | Failed: %d", total, stored, failed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Embed PubMed corpus → pgvector")
    ap.add_argument("--input", default="data/pubmed_corpus.jsonl")
    ap.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    ap.add_argument("--batch-size", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true", help="Embed but don't write to DB")
    args = ap.parse_args()

    if not args.api_key:
        log.error("OPENAI_API_KEY not set. Pass --api-key or set the env var.")
        raise SystemExit(1)

    try:
        import openai  # noqa: F401
    except ImportError:
        log.error("openai package not installed. Run: pip install openai")
        raise SystemExit(1)

    corpus_path = Path(args.input)
    if not corpus_path.exists():
        log.error("Corpus file not found: %s — run build_pubmed_corpus.py first", corpus_path)
        raise SystemExit(1)

    docs: list[dict] = []
    with corpus_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    log.info("=== PubMed Corpus Embedder ===")
    log.info("Corpus: %d abstracts | Model: %s | Batch: %d", len(docs), EMBEDDING_MODEL, args.batch_size)
    if args.dry_run:
        log.info("DRY RUN — embeddings will NOT be written to DB")

    # Load .env if present (for DATABASE_URL)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    asyncio.run(embed_and_store(
        docs,
        api_key=args.api_key,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    ))


if __name__ == "__main__":
    main()
