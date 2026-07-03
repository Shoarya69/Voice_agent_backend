"""Client for the Lovable control-plane app (nexovaaii.lovable.app).

Fetches per-agent voice configuration and reports completed call logs
back to the control plane. Agent config is cached in-memory for a short
TTL to keep the hot WebSocket path fast without hammering the Lovable API
on every call.
"""

from __future__ import annotations

import time
from typing import Any

import httpx
import structlog

from app.config import get_settings
from app.models import AgentConfig
from app.utils import retry_async

logger = structlog.get_logger(__name__)

_CACHE_TTL_SECONDS = 60
_REQUEST_TIMEOUT_SECONDS = 5.0
_TRANSIENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout)

# agent_id -> (AgentConfig, fetched_at_monotonic)
_config_cache: dict[str, tuple[AgentConfig, float]] = {}


def _auth_headers() -> dict[str, str]:
    settings = get_settings()
    return {"Authorization": f"Bearer {settings.lovable_api_secret}"}


async def fetch_agent_config(agent_id: str) -> AgentConfig:
    """Fetch voice-agent configuration for `agent_id` from the Lovable app.

    Results are cached in-memory for `_CACHE_TTL_SECONDS` to avoid an
    extra network round-trip on every incoming call for the same agent.
    Falls back to a safe default config (with a warning log) if the
    control plane is unreachable or returns an error, so a call never
    hard-fails just because the config API blipped.

    Args:
        agent_id: UUID of the agent, taken from Exotel's
            `start.custom_parameters.agent_id`.

    Returns:
        The resolved `AgentConfig` for this agent.
    """
    cached = _config_cache.get(agent_id)
    if cached is not None:
        config, fetched_at = cached
        if time.monotonic() - fetched_at < _CACHE_TTL_SECONDS:
            return config

    settings = get_settings()
    url = f"{settings.lovable_app_url}/api/public/voicebot/agent/{agent_id}"

    async def _do_fetch() -> AgentConfig:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.get(url, headers=_auth_headers())
            response.raise_for_status()
            data: dict[str, Any] = response.json()
            data.setdefault("agent_id", agent_id)
            return AgentConfig(**data)

    try:
        config = await retry_async(
            _do_fetch,
            retries=2,
            base_delay_seconds=0.2,
            retry_on=_TRANSIENT_ERRORS,
            op_name="lovable_api.fetch_agent_config",
            call_sid=agent_id,
        )
        _config_cache[agent_id] = (config, time.monotonic())
        return config
    except Exception as exc:  # noqa: BLE001 - must never crash the call
        logger.error(
            "lovable_api.fetch_agent_config_failed",
            agent_id=agent_id,
            error=str(exc),
        )
        return AgentConfig(agent_id=agent_id)


async def post_call_log(call_data: dict[str, Any]) -> None:
    """POST a completed call's transcript and metadata to the Lovable app.

    Args:
        call_data: Dict with keys call_sid, from, to, agent_id, transcript
            (list of turns), started_at, ended_at, duration_seconds.

    This is best-effort: failures are logged but never raised, since a
    logging failure should not affect an already-completed call.
    """
    settings = get_settings()
    url = f"{settings.lovable_app_url}/api/public/voicebot/call-log"
    call_sid = call_data.get("call_sid", "")

    async def _do_post() -> None:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(url, json=call_data, headers=_auth_headers())
            response.raise_for_status()

    try:
        await retry_async(
            _do_post,
            retries=2,
            base_delay_seconds=0.5,
            retry_on=_TRANSIENT_ERRORS,
            op_name="lovable_api.post_call_log",
            call_sid=call_sid,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort logging call
        logger.error(
            "lovable_api.post_call_log_failed",
            call_sid=call_sid,
            error=str(exc),
        )


def clear_config_cache() -> None:
    """Clear the in-memory agent config cache (useful for tests)."""
    _config_cache.clear()
