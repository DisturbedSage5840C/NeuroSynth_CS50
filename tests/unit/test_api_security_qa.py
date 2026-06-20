from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

jwt = pytest.importorskip("jwt")
respx = pytest.importorskip("respx")
httpx = pytest.importorskip("httpx")
from httpx import AsyncClient, Response as HttpxResponse

from backend.core.config import get_settings
from backend.core.security import ACCESS_COOKIE, Role, create_access_token


def _expired_access_token() -> str:
    settings = get_settings()
    now = datetime.now(tz=UTC)
    payload = {
        "sub": "expired-user",
        "role": Role.CLINICIAN.value,
        "type": "access",
        "iat": int((now - timedelta(minutes=5)).timestamp()),
        "exp": int((now - timedelta(minutes=1)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def test_jwt_expiry_returns_401(api_client) -> None:
    api_client.cookies.set(ACCESS_COOKIE, _expired_access_token())

    response = api_client.get("/patients")

    assert response.status_code == 401


def test_rate_limit_returns_429_with_retry_after(api_client, monkeypatch: pytest.MonkeyPatch) -> None:
    from backend.core import rate_limit

    # Keep this test deterministic and fast.
    monkeypatch.setitem(rate_limit.ROLE_LIMITS, Role.CLINICIAN.value, "2/minute")

    token = create_access_token("rate-limited-user", Role.CLINICIAN)
    api_client.cookies.set(ACCESS_COOKIE, token)

    assert api_client.get("/patients").status_code == 200
    assert api_client.get("/patients").status_code == 200
    third = api_client.get("/patients")

    assert third.status_code == 429
    assert "Retry-After" in third.headers
    assert int(third.headers["Retry-After"]) > 0


def test_phi_redaction_in_logs_does_not_emit_raw_patient_id(api_client, capsys) -> None:
    token = create_access_token("qa-redaction", Role.CLINICIAN)
    api_client.cookies.set(ACCESS_COOKIE, token)

    raw_patient_id = "MRN-99887766"
    response = api_client.get("/patients", headers={"x-patient-id": raw_patient_id})

    assert response.status_code == 200
    captured = capsys.readouterr()
    combined = f"{captured.out}\n{captured.err}"
    assert raw_patient_id not in combined


@pytest.mark.asyncio
@respx.mock
async def test_api_downstream_call_is_mocked_with_respx() -> None:
    # QA contract test: downstream ML service calls must be mockable and deterministic.
    route = respx.get("https://ml-inference.internal/health").mock(
        return_value=HttpxResponse(200, json={"status": "ok"})
    )

    async with AsyncClient() as client:
        resp = await client.get("https://ml-inference.internal/health")

    assert route.called
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
