# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
"""v3 Data endpoints — real dataset status, cohort stats, and provenance.

NOTE: Do NOT add ``from __future__ import annotations``.

Endpoints:
  GET  /v3/data/sources              — list all 11 data sources + status
  POST /v3/data/refresh/{source}     — admin: trigger source re-download
  GET  /v3/data/cohort/stats         — population-level statistics
  GET  /v3/data/provenance           — data lineage per source
"""
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.deps import get_current_user, get_database, require_role
from backend.db import Database
from backend.models_v3 import (
    AgeGroup,
    CohortStatsResponse,
    DataSourcesResponse,
    DataSourceStatus,
    DiseasePrevalence,
    ProvenanceResponse,
    ProvenanceRow,
    RefreshResponse,
)
from backend.services.data_pipeline_service import get_data_pipeline_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v3/data", tags=["data-v3"])


def _svc(request: Request):
    """Get or create the DataPipelineService (singleton on app state)."""
    svc = getattr(request.app.state, "data_pipeline_svc", None)
    if svc is None:
        db = getattr(request.app.state, "db", None)
        svc = get_data_pipeline_service(db=db)
        request.app.state.data_pipeline_svc = svc
    return svc


# ── GET /v3/data/sources ──────────────────────────────────────────────────────

@router.get("/sources", response_model=DataSourcesResponse)
async def list_data_sources(request: Request, db: Database = Depends(get_database)):
    """List all real data sources with row counts and status."""
    svc = _svc(request)
    raw = await svc.get_sources()

    sources = [
        DataSourceStatus(
            name=s["name"],
            display_name=s.get("display_name", s["name"]),
            tier=s.get("tier", "1"),
            url=s.get("url"),
            row_count=s.get("row_count"),
            feature_count=s.get("feature_count"),
            last_updated=s.get("last_updated"),
            status=s.get("status", "active"),
            features=s.get("features", ""),
        )
        for s in raw
    ]

    total_rows = sum((s.row_count or 0) for s in sources)
    active = sum(1 for s in sources if s.status == "active")
    return DataSourcesResponse(
        sources=sources,
        total_rows=total_rows,
        total_sources=len(sources),
        active_sources=active,
        last_refreshed=datetime.utcnow(),
    )


# ── POST /v3/data/refresh/{source} ────────────────────────────────────────────

@router.post("/refresh/{source}", response_model=RefreshResponse)
async def refresh_data_source(
    source: str,
    request: Request,
    user=Depends(get_current_user),
    db: Database = Depends(get_database),
):
    """Admin-only: trigger re-download of a data source."""
    if getattr(user, "role", None) != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")

    svc = _svc(request)
    result = await svc.refresh_source(source)

    return RefreshResponse(
        source=source,
        status=result["status"],
        message=result["message"],
    )


# ── GET /v3/data/cohort/stats ─────────────────────────────────────────────────

@router.get("/cohort/stats", response_model=CohortStatsResponse)
async def get_cohort_stats(request: Request, db: Database = Depends(get_database)):
    """Population-level statistics from the real_v5 dataset."""
    svc = _svc(request)
    stats = await svc.get_cohort_stats()

    prevalence = [
        DiseasePrevalence(
            name=p["name"],
            value=float(p["value"]),
            count=int(p.get("count", 0)),
            color=p.get("color", ""),
        )
        for p in stats.get("prevalence", [])
    ]

    age_dist = [
        AgeGroup(
            range=ag["range"],
            ad=ag.get("ad", 0), pd=ag.get("pd", 0),
            ms=ag.get("ms", 0), ep=ag.get("ep", 0),
            als=ag.get("als", 0), hd=ag.get("hd", 0),
        )
        for ag in stats.get("age_distribution", [])
    ]

    return CohortStatsResponse(
        total_patients=stats.get("total_patients", 0),
        data_sources=stats.get("data_sources", 11),
        prevalence=prevalence,
        age_distribution=age_dist,
        feature_count=stats.get("feature_count", 56),
        schema_version=stats.get("schema_version", "v5"),
        computed_at=datetime.fromisoformat(stats["computed_at"])
        if stats.get("computed_at") else None,
    )


# ── GET /v3/data/provenance ───────────────────────────────────────────────────

@router.get("/provenance", response_model=ProvenanceResponse)
async def get_provenance(request: Request, db: Database = Depends(get_database)):
    """Data lineage: per-source row counts, QC, and merge details."""
    svc = _svc(request)
    data = svc.get_provenance()

    provenance = [
        ProvenanceRow(
            source=p["source"],
            tier=p["tier"],
            rows_raw=p["rows_raw"],
            rows_after_qc=p["rows_after_qc"],
            features_mapped=p["features_mapped"],
            synthetic=p.get("synthetic", False),
        )
        for p in data.get("provenance", [])
    ]

    merged_at = None
    if data.get("merged_at"):
        try:
            merged_at = datetime.fromisoformat(data["merged_at"].rstrip("Z"))
        except (ValueError, TypeError):
            pass

    return ProvenanceResponse(
        total_rows=data.get("total_rows", 0),
        provenance=provenance,
        merge_file=data.get("merge_file", "data/real_v5.parquet"),
        schema_version=data.get("schema_version", "v5"),
        merged_at=merged_at,
    )
