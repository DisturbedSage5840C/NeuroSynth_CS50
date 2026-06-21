# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""Literature endpoints — pgvector similarity search over PubMed corpus.

NOTE: Do NOT add ``from __future__ import annotations`` — FastAPI needs
runtime type resolution for Pydantic models.

Endpoints:
  POST /v3/literature/search   — embed query, return top-k similar abstracts
  GET  /v3/literature/cite/{pmid} — return a single abstract by PMID
  GET  /v3/literature/status   — corpus size + embedding status
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from backend.deps import get_current_user, get_database
from backend.db import Database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v3/literature", tags=["literature"])


class LiteratureSearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000,
                       description="Free-text query (patient profile or clinical question)")
    top_k: int = Field(default=5, ge=1, le=20,
                       description="Number of abstracts to return")


class AbstractResult(BaseModel):
    pmid: str
    title: str
    abstract: str
    journal: str
    pub_year: int | None
    diseases: list[str]
    similarity: float | None = None


class LiteratureSearchResponse(BaseModel):
    results: list[AbstractResult]
    total_retrieved: int
    rag_enabled: bool


# ---------------------------------------------------------------------------

@router.post("/search", response_model=LiteratureSearchResponse)
async def search_literature(
    body: LiteratureSearchRequest,
    request: Request,
    db: Database = Depends(get_database),
):
    """Embed the query and return the top-k most similar PubMed abstracts.

    Requires the literature_embeddings table to be populated via
    ``embed_corpus.py`` and ``OPENAI_API_KEY`` to be set in the environment.
    Returns an empty results list (not an error) when RAG is unavailable.
    """
    rag = getattr(request.app.state, "rag", None)
    if rag is None or not rag.enabled:
        return LiteratureSearchResponse(results=[], total_retrieved=0, rag_enabled=False)

    try:
        docs = await rag.retrieve(body.query, top_k=body.top_k)
    except Exception as exc:
        logger.warning("literature_search_failed error=%s", exc)
        raise HTTPException(status_code=502, detail="Literature search temporarily unavailable")

    results = [
        AbstractResult(
            pmid=str(d.get("pmid", "")),
            title=d.get("title") or "",
            abstract=d.get("abstract") or "",
            journal=d.get("journal") or "",
            pub_year=d.get("pub_year"),
            diseases=list(d.get("diseases") or []),
            similarity=round(float(d.get("similarity", 0.0)), 4),
        )
        for d in docs
    ]
    return LiteratureSearchResponse(
        results=results,
        total_retrieved=len(results),
        rag_enabled=True,
    )


@router.get("/cite/{pmid}", response_model=AbstractResult)
async def get_citation(
    pmid: str,
    db: Database = Depends(get_database),
):
    """Return a single PubMed abstract by PMID.

    Used by the frontend to resolve inline [PMIDxxxxxxx] citations in
    SOAP reports to their full title + abstract text.
    """
    record = await db.fetch_abstract(pmid)
    if record is None:
        raise HTTPException(status_code=404, detail=f"PMID {pmid} not in literature corpus")

    return AbstractResult(
        pmid=str(record.get("pmid", pmid)),
        title=record.get("title") or "",
        abstract=record.get("abstract") or "",
        journal=record.get("journal") or "",
        pub_year=record.get("pub_year"),
        diseases=list(record.get("diseases") or []),
        similarity=None,
    )


@router.get("/status")
async def literature_status(
    request: Request,
    db: Database = Depends(get_database),
):
    """Return RAG corpus status — abstract count and embedding availability."""
    count = await db.count_literature()
    rag = getattr(request.app.state, "rag", None)
    return {
        "abstracts_stored": count,
        "rag_enabled": bool(rag and rag.enabled),
        "embedding_model": "text-embedding-3-small",
        "vector_dims": 1536,
        "ready": count > 0 and bool(rag and rag.enabled),
    }
