"""PubMed-grounded RAG for NeuroSynth v5 report generation.

Provides sync and async retrieval over the pgvector literature_embeddings
table stored in Neon. Embeddings are generated via OpenAI
text-embedding-3-small (1536 dims).

Fallbacks at every layer:
  - No OPENAI_API_KEY   → query embedding returns None → no RAG, v3 fallback
  - Empty DB table      → search returns []            → no RAG, v3 fallback
  - pgvector unavailable → DB query fails gracefully   → no RAG, v3 fallback
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_INPUT_CHARS = 8_000


class PubMedRAG:
    """pgvector-backed retrieval over PubMed neurology abstracts.

    Designed to be called from a ThreadPoolExecutor thread (as used in
    FastAPI inference handlers). Sync methods use a private event loop;
    async methods are for direct await in async contexts.
    """

    def __init__(self, db: Any, openai_api_key: str | None = None) -> None:
        self.db = db
        self._api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")

    @property
    def enabled(self) -> bool:
        return bool(self._api_key)

    # ------------------------------------------------------------------
    # Sync API (safe to call from ThreadPoolExecutor threads)
    # ------------------------------------------------------------------

    def retrieve_sync(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Embed query_text and fetch top_k similar abstracts synchronously.

        Returns [] when OpenAI key is absent or DB/pgvector is unavailable.
        """
        if not self.enabled:
            return []
        embedding = self._embed_sync(query_text)
        if embedding is None:
            return []
        try:
            return asyncio.run(self.db.search_literature(embedding, top_k=top_k))
        except RuntimeError:
            # Already in a running event loop (shouldn't happen in a thread, but guard)
            logger.warning("rag_retrieve_event_loop_conflict")
            return []
        except Exception as exc:
            logger.warning("rag_retrieve_sync_failed error=%s", exc)
            return []

    def _embed_sync(self, text: str) -> list[float] | None:
        """Call OpenAI Embeddings API synchronously."""
        if not self._api_key:
            return None
        try:
            import openai  # type: ignore
            client = openai.OpenAI(api_key=self._api_key)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[: MAX_INPUT_CHARS],
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("rag_embed_sync_failed error=%s", exc)
            return None

    # ------------------------------------------------------------------
    # Async API (for use in async route handlers)
    # ------------------------------------------------------------------

    async def retrieve(self, query_text: str, top_k: int = 5) -> list[dict[str, Any]]:
        """Async version of retrieve_sync."""
        if not self.enabled:
            return []
        embedding = await self._embed_async(query_text)
        if embedding is None:
            return []
        try:
            return await self.db.search_literature(embedding, top_k=top_k)
        except Exception as exc:
            logger.warning("rag_retrieve_async_failed error=%s", exc)
            return []

    async def _embed_async(self, text: str) -> list[float] | None:
        """Call OpenAI Embeddings API asynchronously."""
        if not self._api_key:
            return None
        try:
            import openai  # type: ignore
# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
            client = openai.AsyncOpenAI(api_key=self._api_key)
            response = await client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text[: MAX_INPUT_CHARS],
            )
            return response.data[0].embedding
        except Exception as exc:
            logger.warning("rag_embed_async_failed error=%s", exc)
            return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def format_context(docs: list[dict[str, Any]]) -> str:
        """Format retrieved abstracts as numbered context for Claude prompt."""
        if not docs:
            return ""
        lines: list[str] = ["Relevant PubMed literature (use [PMID] to cite inline):\n"]
        for i, doc in enumerate(docs, start=1):
            pmid = doc.get("pmid", "")
            title = doc.get("title", "No title")
            abstract = doc.get("abstract", "")
            year = doc.get("pub_year") or "n.d."
            journal = doc.get("journal", "")
            lines.append(
                f"[{i}] PMID{pmid} ({year}, {journal})\n"
                f"Title: {title}\n"
                f"Abstract: {abstract[:600]}{'…' if len(abstract) > 600 else ''}\n"
            )
        return "\n".join(lines)

    @staticmethod
    def build_patient_query(patient_data: dict, disease: str | None) -> str:
        """Build a text query that captures this patient's clinical profile."""
        parts: list[str] = []
        if disease:
            parts.append(disease)
        age = patient_data.get("Age")
        if age:
            parts.append(f"age {int(age)}")
        mmse = patient_data.get("MMSE")
        if mmse is not None:
            parts.append(f"MMSE {mmse}")
        for feat in ("FamilyHistoryAlzheimers", "APOE4_dosage", "UPDRS_motor",
                     "CSF_Abeta42", "CSF_pTau", "tremor_amplitude"):
            val = patient_data.get(feat)
            if val and float(val) > 0:
                parts.append(feat.replace("_", " ").lower())
        return " ".join(parts) or (disease or "neurological disease risk")
