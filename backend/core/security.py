# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

import jwt

from backend.core.config import get_settings


class Role(StrEnum):
    CLINICIAN = "CLINICIAN"
    RESEARCHER = "RESEARCHER"
    ADMIN = "ADMIN"


ACCESS_COOKIE = "ns_access_token"
REFRESH_COOKIE = "ns_refresh_token"


def hash_patient_id(patient_id: str | None) -> str | None:
    if not patient_id:
        return None
    secret = get_settings().patient_hash_secret.encode("utf-8")
    digest = hmac.new(secret, patient_id.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:16]


def _encode_token(subject: str, role: Role, expires_delta: timedelta, token_type: str) -> str:
    settings = get_settings()
    now = datetime.now(tz=UTC)
    payload: dict[str, Any] = {
        "sub": subject,
        "role": role.value,
        "type": token_type,
        "iat": int(now.timestamp()),
        "exp": int((now + expires_delta).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def create_access_token(subject: str, role: Role) -> str:
    settings = get_settings()
    return _encode_token(
        subject=subject,
        role=role,
        expires_delta=timedelta(minutes=settings.access_token_minutes),
        token_type="access",
    )


def create_refresh_token(subject: str, role: Role) -> str:
    settings = get_settings()
    return _encode_token(
        subject=subject,
        role=role,
        expires_delta=timedelta(days=settings.refresh_token_days),
        token_type="refresh",
    )


def decode_token(token: str, expected_type: str | None = None) -> dict[str, Any]:
    settings = get_settings()
    payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    if expected_type and payload.get("type") != expected_type:
        raise jwt.InvalidTokenError("Invalid token type")
    return payload
