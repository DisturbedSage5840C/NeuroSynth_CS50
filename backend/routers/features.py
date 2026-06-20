"""Feature schema endpoint (Gap 7).

Serves the human-readable encoding reference for every clinical input feature so the
frontend can render feature legends and SHAP labels without hard-coding them.
Public reference data — no authentication required.
"""
from fastapi import APIRouter

from backend.feature_schema import get_feature_schema

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
router = APIRouter(prefix="/v2/features", tags=["features-v2"])


@router.get(
    "/schema",
    summary="Feature encoding schema",
    description=(
        "Returns the complete clinical feature schema: human-readable labels, full "
        "names, types, valid ranges, categorical encodings (e.g. Gender 0=Female), "
        "units, and clinical notes. Used by the frontend to render feature legends."
    ),
)
async def feature_schema() -> dict:
    return get_feature_schema()
