# AI Assistance: Developed with assistance from Claude (Anthropic) — claude.ai
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="NEUROSYNTH_", extra="ignore")

    app_name: str = "NeuroSynth Clinical API"
    app_version: str = "3.0.0"
    app_env: Literal["dev", "staging", "prod"] = "dev"

    jwt_secret: str = "change-me"
    jwt_algorithm: str = "HS256"
    access_token_minutes: int = 15
    refresh_token_days: int = 7
    auth_cookie_secure: bool = False

    patient_hash_secret: str = "change-me-patient-hmac"

    postgres_dsn: str = "postgresql://postgres:postgres@localhost:5432/neurosynth"
    redis_url: str = "redis://localhost:6379/0"

    # LLM clinical report generation (Gap 4). Empty key -> deterministic Jinja2 fallback.
    # Accepts the prefixed NEUROSYNTH_ANTHROPIC_API_KEY or the bare ANTHROPIC_API_KEY.
    anthropic_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("NEUROSYNTH_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"),
    )
    anthropic_model: str = "claude-sonnet-4-6"

    kubeflow_host: str = "http://localhost:8080"

    metrics_enabled: bool = True
    allowed_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: object) -> object:
        if isinstance(value, str):
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @model_validator(mode="after")
    def _validate_secrets_in_prod(self) -> "Settings":
        """Ensure sensitive defaults are not used in staging/production."""
        if self.app_env in ("staging", "prod"):
            if self.jwt_secret == "change-me":
                raise ValueError(
                    "NEUROSYNTH_JWT_SECRET must be changed from the default in %s environment" % self.app_env
                )
            if self.patient_hash_secret == "change-me-patient-hmac":
                raise ValueError(
                    "NEUROSYNTH_PATIENT_HASH_SECRET must be changed from the default in %s environment" % self.app_env
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
