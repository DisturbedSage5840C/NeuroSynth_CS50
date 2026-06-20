from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, Request, status
from redis.asyncio import Redis

from backend.core.security import Role
from backend.db import Database, get_db
from backend.models import UserContext

# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai

def get_redis(request: Request) -> Redis:
    redis_client: Redis | None = getattr(request.app.state, "redis", None)
    if redis_client is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Redis unavailable")
    return redis_client


def get_current_user(request: Request) -> UserContext:
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    return UserContext(user_id=user["user_id"], role=Role(user["role"]))


def require_role(*roles: Role) -> Callable[[UserContext], UserContext]:
    def checker(user: UserContext = Depends(get_current_user)) -> UserContext:
        if user.role not in roles:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")
        return user

    return checker


def get_database() -> Database:
    return get_db()
