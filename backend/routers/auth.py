# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from fastapi import APIRouter, HTTPException, Request, Response, status

from backend.core.config import get_settings
from backend.core.security import ACCESS_COOKIE, REFRESH_COOKIE, Role, create_access_token, create_refresh_token, decode_token
from backend.models import ApiMessage, LoginRequest, TokenEnvelope, UserContext

router = APIRouter(prefix="/auth", tags=["auth"])

DEMO_USERS = {
    "clinician": {"password": "neurosynth", "role": Role.CLINICIAN},
    "researcher": {"password": "neurosynth", "role": Role.RESEARCHER},
    "admin": {"password": "neurosynth", "role": Role.ADMIN},
    "clinician@neurosynth.local": {"password": "neurosynth", "role": Role.CLINICIAN},
    "researcher@neurosynth.local": {"password": "neurosynth", "role": Role.RESEARCHER},
    "admin@neurosynth.local": {"password": "neurosynth", "role": Role.ADMIN},
}


@router.post(
    "/login",
    response_model=TokenEnvelope,
    summary="Login and set auth cookies",
    description="Authenticates a user and issues access/refresh JWT tokens in httpOnly cookies.",
)
async def login(payload: LoginRequest, response: Response) -> TokenEnvelope:
    user_record = DEMO_USERS.get(payload.username)
    if not user_record or user_record["password"] != payload.password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    role = user_record["role"]

    settings = get_settings()
    access = create_access_token(payload.username, role)
    refresh = create_refresh_token(payload.username, role)

    response.set_cookie(ACCESS_COOKIE, access, httponly=True, secure=settings.auth_cookie_secure, samesite="lax", max_age=settings.access_token_minutes * 60)
    response.set_cookie(REFRESH_COOKIE, refresh, httponly=True, secure=settings.auth_cookie_secure, samesite="lax", max_age=settings.refresh_token_days * 86400)

    return TokenEnvelope(
        access_token=access,
        refresh_token=refresh,
        access_expires_in=settings.access_token_minutes * 60,
        refresh_expires_in=settings.refresh_token_days * 86400,
        user=UserContext(user_id=payload.username, role=role),
    )


@router.post(
    "/refresh",
    response_model=TokenEnvelope,
    summary="Refresh access token",
    description="Generates a new access token from refresh token cookie and rotates access cookie.",
)
async def refresh_token(request: Request, response: Response) -> TokenEnvelope:
    refresh_cookie = request.cookies.get(REFRESH_COOKIE)
    # Also accept refresh token from Authorization header for cross-origin SPAs
    if not refresh_cookie:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            refresh_cookie = auth_header[7:]
    if not refresh_cookie:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing refresh token")
    try:
        claims = decode_token(refresh_cookie, expected_type="refresh")
        user = UserContext(user_id=str(claims["sub"]), role=Role(str(claims["role"])))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token") from exc

    settings = get_settings()
    access = create_access_token(user.user_id, user.role)
    refresh = create_refresh_token(user.user_id, user.role)

    response.set_cookie(ACCESS_COOKIE, access, httponly=True, secure=settings.auth_cookie_secure, samesite="lax", max_age=settings.access_token_minutes * 60)
    response.set_cookie(REFRESH_COOKIE, refresh, httponly=True, secure=settings.auth_cookie_secure, samesite="lax", max_age=settings.refresh_token_days * 86400)

    return TokenEnvelope(
        access_token=access,
        refresh_token=refresh,
        access_expires_in=settings.access_token_minutes * 60,
        refresh_expires_in=settings.refresh_token_days * 86400,
        user=user,
    )


@router.post(
    "/logout",
    response_model=ApiMessage,
    summary="Logout",
    description="Clears authentication cookies from the client.",
)
async def logout(response: Response) -> ApiMessage:
    response.delete_cookie(ACCESS_COOKIE)
    response.delete_cookie(REFRESH_COOKIE)
    return ApiMessage(message="logged_out")
