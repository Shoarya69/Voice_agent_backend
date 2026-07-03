"""Application configuration loaded from environment variables.

Uses pydantic-settings so all config is validated at startup and every
value can be overridden via the environment (or a local .env file for
development).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Strongly-typed application settings.

    All values can be supplied via environment variables (case-insensitive)
    or a `.env` file in the project root. See `.env.example` for the full
    list of supported variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    port: int = Field(default=8000, description="Port the ASGI server listens on")
    log_level: str = Field(default="INFO", description="Python/structlog log level")
    environment: str = Field(default="production", description="deployment environment name")

    # Lovable app (control plane)
    lovable_app_url: str = Field(
        default="https://nexovaaii.lovable.app",
        description="Base URL of the Lovable control-plane app",
    )
    lovable_api_secret: str = Field(
        default="",
        description="Shared secret used to authenticate callbacks to the Lovable app",
    )

    # ElevenLabs (STT via Scribe + TTS) - same API key powers both.
    elevenlabs_api_key: str = Field(default="", description="API key for ElevenLabs")
    elevenlabs_voice_id: str = Field(
        default="cgSgspJ2msm6clMCkdW9", description="Default ElevenLabs voice id"
    )

    # OpenAI (LLM)
    openai_api_key: str = Field(default="", description="API key for OpenAI")
    openai_model: str = Field(default="gpt-4o-mini", description="OpenAI chat model name")

    # Default STT/TTS language (BCP-47-ish; ElevenLabs Scribe language_code).
    language: str = Field(default="hin", description="Default language code for STT")

    # Optional WebSocket auth hardening: if set, the Exotel WS route requires
    # a `?token=<value>` query parameter matching this value before accepting
    # the connection. Leave unset to accept all connections (Exotel does not
    # support custom WS headers, so a query param is the only viable option).
    exotel_ws_auth_token: str = Field(
        default="",
        description="Optional shared secret required as a `token` query param on the WS route",
    )

    @property
    def is_production(self) -> bool:
        """Return True when running in the production environment."""
        return self.environment.lower() == "production"

    def missing_required_keys(self) -> list[str]:
        """Return the list of required API keys that are not configured.

        Used at startup to warn loudly (but not crash) about missing
        credentials so operators can fix their Railway env vars quickly.
        """
        required = {
            "LOVABLE_API_SECRET": self.lovable_api_secret,
            "ELEVENLABS_API_KEY": self.elevenlabs_api_key,
            "OPENAI_API_KEY": self.openai_api_key,
        }
        return [name for name, value in required.items() if not value]


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of the application settings."""
    return Settings()
