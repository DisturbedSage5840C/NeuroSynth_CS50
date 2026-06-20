from __future__ import annotations

from math import ceil

from fastapi import Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from backend.core.security import Role
from backend.models import ErrorResponse

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

ROLE_LIMITS: dict[str, str] = {
    Role.CLINICIAN.value: "100/minute",
    Role.RESEARCHER.value: "500/minute",
    Role.ADMIN.value: "1000000/minute",
}


def _key_func(request: Request) -> str:
    user = getattr(request.state, "user", None)
    role = user["role"] if user else "ANON"
    subject = user["user_id"] if user else get_remote_address(request)
    return f"{role}:{subject}"


limiter = Limiter(key_func=_key_func, default_limits=[])


def role_limit(request: Request | None = None) -> str:
    # SlowAPI invokes limit provider callables without arguments.
    if request is None:
        return ROLE_LIMITS.get(Role.CLINICIAN.value, "30/minute")

    user = getattr(request.state, "user", None)
    if user is None:
        return "30/minute"
    return ROLE_LIMITS.get(user["role"], "30/minute")


async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    reset_after = 60
    if hasattr(exc, "limit") and getattr(exc.limit, "granularity", None):
        reset_after = ceil(float(exc.limit.granularity.seconds))
    payload = ErrorResponse(error="rate_limit_exceeded", detail="Too many requests").model_dump()
    return JSONResponse(status_code=429, content=payload, headers={"Retry-After": str(reset_after)})
