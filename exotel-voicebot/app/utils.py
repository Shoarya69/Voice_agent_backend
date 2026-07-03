"""Small shared utilities: async retry with exponential backoff."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    retries: int = 1,
    base_delay_seconds: float = 0.2,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    op_name: str = "operation",
    call_sid: str = "",
) -> T:
    """Call `func()` with exponential-backoff retries on transient failures.

    Args:
        func: Zero-arg async callable to invoke (wrap the real call in a
            lambda/closure so it can be re-invoked on retry).
        retries: Number of *additional* attempts after the first failure.
        base_delay_seconds: Base delay before the first retry; doubles on
            each subsequent attempt (0.2s, 0.4s, 0.8s, ...).
        retry_on: Tuple of exception types that should trigger a retry.
            Any other exception propagates immediately.
        op_name: Human-readable operation name, used only for logging.
        call_sid: Call identifier, included in logs for traceability.

    Returns:
        The result of the first successful call to `func()`.

    Raises:
        The last exception encountered if all attempts (including retries)
        fail, or immediately if an exception outside `retry_on` occurs.
    """
    attempt = 0
    while True:
        try:
            return await func()
        except retry_on as exc:
            if attempt >= retries:
                raise
            delay = base_delay_seconds * (2**attempt)
            logger.warning(
                f"{op_name}.retrying",
                call_sid=call_sid,
                attempt=attempt + 1,
                max_attempts=retries + 1,
                delay_seconds=delay,
                error=str(exc),
            )
            await asyncio.sleep(delay)
            attempt += 1
