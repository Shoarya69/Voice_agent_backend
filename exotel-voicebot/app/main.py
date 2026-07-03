"""FastAPI application entry point for the Exotel Voicebot server.

Exposes:
    GET  /health            - liveness/readiness check for Railway
    GET  /metrics           - Prometheus-format metrics
    WS   /exotel/voicebot   - the Exotel Voicebot Applet WebSocket endpoint
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse

from app.config import get_settings
from app.websocket_handler import handle_exotel_websocket, metrics

# Tracks all in-flight WebSocket handler tasks so shutdown can wait for
# active calls to finish gracefully instead of killing them mid-turn.
_active_session_tasks: set[asyncio.Task] = set()


def _configure_logging(log_level: str) -> None:
    """Configure structlog to emit JSON logs (Railway-friendly) at `log_level`."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


settings = get_settings()
_configure_logging(settings.log_level)
logger = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Log configuration at startup and drain in-flight calls at shutdown."""
    logger.info(
        "server.startup",
        environment=settings.environment,
        port=settings.port,
        openai_model=settings.openai_model,
        language=settings.language,
    )
    missing = settings.missing_required_keys()
    if missing:
        logger.warning("server.missing_config", missing_keys=missing)
    else:
        logger.info("server.config_ok")

    yield

    if not _active_session_tasks:
        logger.info("server.shutdown", active_calls=0)
        return

    logger.info("server.shutdown_waiting", active_calls=len(_active_session_tasks))
    try:
        await asyncio.wait_for(
            asyncio.gather(*_active_session_tasks, return_exceptions=True),
            timeout=30.0,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "server.shutdown_forced", still_active=len(_active_session_tasks)
        )
    logger.info("server.shutdown_complete")


app = FastAPI(title="Exotel Voicebot Server", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness/readiness check used by Railway's health check."""
    return {"status": "ok"}


@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics() -> str:
    """Expose basic call metrics in Prometheus text exposition format."""
    lines = [
        "# HELP voicebot_active_calls Number of currently active calls",
        "# TYPE voicebot_active_calls gauge",
        f"voicebot_active_calls {metrics.active_calls}",
        "# HELP voicebot_total_calls Total calls handled since process start",
        "# TYPE voicebot_total_calls counter",
        f"voicebot_total_calls {metrics.total_calls}",
        "# HELP voicebot_avg_latency_ms Average time-to-first-reply-token per turn (ms)",
        "# TYPE voicebot_avg_latency_ms gauge",
        f"voicebot_avg_latency_ms {metrics.avg_latency_ms:.2f}",
    ]
    return "\n".join(lines) + "\n"


@app.websocket("/exotel/voicebot")
async def exotel_voicebot(websocket: WebSocket) -> None:
    """Handle an Exotel Voicebot Applet WebSocket connection for one call.

    If `EXOTEL_WS_AUTH_TOKEN` is configured, requires a matching `?token=`
    query parameter before accepting the connection. Exotel's Voicebot
    Applet lets you configure the WS URL with query params, so this is a
    lightweight way to stop random internet clients from connecting to a
    publicly reachable WebSocket endpoint. Left unset by default so the
    server works out of the box with a plain Exotel applet configuration.
    """
    settings = get_settings()
    if settings.exotel_ws_auth_token:
        provided_token = websocket.query_params.get("token", "")
        if provided_token != settings.exotel_ws_auth_token:
            logger.warning("websocket.auth_rejected", client=str(websocket.client))
            await websocket.close(code=4401)
            return

    task = asyncio.current_task()
    if task is not None:
        _active_session_tasks.add(task)
    try:
        await handle_exotel_websocket(websocket)
    finally:
        if task is not None:
            _active_session_tasks.discard(task)
