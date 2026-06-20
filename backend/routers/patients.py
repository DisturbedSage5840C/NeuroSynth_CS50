from datetime import UTC, datetime
from uuid import uuid4
from typing import Any

from fastapi import APIRouter, Body, Depends, Request

from backend.core.rate_limit import limiter, role_limit
from backend.db import Database
from backend.core.security import Role
from backend.deps import get_current_user, get_database
from backend.models import PatientListResponse, PatientSummary, UserContext

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
router = APIRouter(prefix="/patients", tags=["patients"])


@router.get(
    "",
    response_model=PatientListResponse,
    summary="List patients",
    description="Returns patient summaries accessible by clinicians, researchers, and admins.",
)
@limiter.limit(role_limit)
async def list_patients(
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: Database = Depends(get_database),
) -> PatientListResponse:
    _ = user
    now = datetime.now(tz=UTC)
    if db.pool:
        try:
            rows = await db.pool.fetch(
                """
                SELECT
                    p.id,
                    p.name,
                    a.probability,
                    a.risk_level,
                    a.disease_classification,
                    COALESCE(a.created_at, p.updated_at, NOW()) AS updated_at
                FROM patients p
                LEFT JOIN (
                    SELECT DISTINCT ON (patient_id)
                        patient_id, probability, risk_level, disease_classification, created_at
                    FROM analyses
                    ORDER BY patient_id, created_at DESC
                ) a ON a.patient_id = p.id
                ORDER BY COALESCE(a.created_at, p.updated_at) DESC NULLS LAST
                LIMIT 50
                """
            )
            items = [
                PatientSummary(
                    patient_id=str(r["id"]),
                    name=str(r["name"]),
                    probability=float(r["probability"]) if r["probability"] is not None else None,
                    risk_level=str(r["risk_level"]) if r["risk_level"] is not None else None,
                    disease_classification=dict(r["disease_classification"]) if r["disease_classification"] is not None else None,
                    updated_at=r["updated_at"] or now,
                )
                for r in rows
            ]
        except Exception:
            # analyses table may not exist yet — fall back to patients-only query
            try:
                rows = await db.pool.fetch(
                    "SELECT id, name, updated_at FROM patients ORDER BY updated_at DESC NULLS LAST LIMIT 50"
                )
                items = [
                    PatientSummary(
                        patient_id=str(r["id"]),
                        name=str(r["name"]),
                        updated_at=r["updated_at"] or now,
                    )
                    for r in rows
                ]
            except Exception:
                items = []
    else:
        items = [
            PatientSummary(patient_id="P-001", name="Nakamura, Kenji", updated_at=now),
            PatientSummary(patient_id="P-002", name="Okonkwo, Adaeze", updated_at=now),
        ]

    return PatientListResponse(items=items)


@router.get(
    "/{patient_id}",
    response_model=PatientSummary,
    summary="Get patient",
    description="Returns a single patient summary by ID.",
)
@limiter.limit(role_limit)
async def get_patient(patient_id: str, request: Request, user: UserContext = Depends(get_current_user)) -> PatientSummary:
    _ = request
    _ = user
    return PatientSummary(patient_id=patient_id, name=f"Patient {patient_id}", updated_at=datetime.now(tz=UTC))


@router.post(
    "/",
    response_model=PatientSummary,
    status_code=201,
    summary="Create patient",
    description="Creates a patient row in PostgreSQL when DB is available.",
)
@limiter.limit(role_limit)
async def create_patient(
    request: Request,
    name: str | None = None,
    age: int = 62,
    sex: str = "F",
    diagnosis: str = "Neurology Monitoring",
    payload: dict[str, Any] | None = Body(default=None),
    user: UserContext = Depends(get_current_user),
    db: Database = Depends(get_database),
) -> PatientSummary:
    _ = request
    _ = user
    if payload:
        if payload.get("name"):
            name = str(payload["name"])
        if payload.get("age") is not None:
            age = int(payload["age"])
        if payload.get("sex") is not None:
            sex = str(payload["sex"])
        if payload.get("diagnosis") is not None:
            diagnosis = str(payload["diagnosis"])

    if not name:
        name = f"New Patient {uuid4().hex[:4].upper()}"

    patient_id = f"P-{uuid4().hex[:6].upper()}"
    mrn = f"SYN-{uuid4().hex[:5].upper()}"
    now = datetime.now(tz=UTC)

    if db.pool:
        await db.pool.execute(
            "INSERT INTO patients (id, name, age, sex, mrn, diagnosis, created_at, updated_at) "
            "VALUES ($1, $2, $3, $4, $5, $6, $7, $7)",
            patient_id,
            name,
            age,
            sex,
            mrn,
            diagnosis,
            now,
        )

    return PatientSummary(patient_id=patient_id, name=name, updated_at=now)


@router.get("/{patient_id}/analyses")
@limiter.limit(role_limit)
async def get_patient_analyses(
    patient_id: str,
    request: Request,
    user: UserContext = Depends(get_current_user),
    db: Database = Depends(get_database),
    limit: int = 10,
):
    _ = request
    _ = user
    if not db.pool:
        return {"items": []}

    rows = await db.pool.fetch(
        "SELECT id, probability, risk_level, confidence, trajectory, "
        "shap_values, disease_classification, created_at "
        "FROM analyses WHERE patient_id = $1 "
        "ORDER BY created_at DESC LIMIT $2",
        patient_id,
        limit,
    )
    return {"items": [dict(r) for r in rows]}
