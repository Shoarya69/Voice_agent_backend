"""OpenAI streaming chat completion for reply generation.

Streams `gpt-4o-mini`'s reply via the raw Chat Completions REST API
(httpx + hand-rolled Server-Sent-Events parsing - no `openai` SDK
needed) and yields complete sentences as soon as they're available, so
the caller can start TTS-ing the first sentence before the full reply
has finished generating.

There is intentionally NO fallback LLM provider: on any error or
timeout this yields a single apologetic sentence and gives up on the
turn, per the "ElevenLabs + OpenAI only, no fallbacks" architecture.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator

import httpx
import structlog

from app.config import get_settings
from app.models import ConversationTurn

logger = structlog.get_logger(__name__)

_OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"
_GENERATION_TIMEOUT_SECONDS = 4.0
_FALLBACK_REPLY = "Sorry, ek moment ruko."

# Split on sentence-ending punctuation, including Hindi's poorna viram (।).
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?।])\s+")


def _history_to_openai_messages(history: list[ConversationTurn]) -> list[dict[str, str]]:
    """Convert internal conversation turns into OpenAI's `messages` format."""
    return [{"role": turn.role, "content": turn.text} for turn in history]


async def _raw_delta_stream(
    payload: dict, headers: dict[str, str]
) -> AsyncIterator[str]:
    """Open the OpenAI streaming request and yield raw `delta.content` text chunks.

    Parses the `data: {...}` Server-Sent-Events lines OpenAI's streaming
    Chat Completions endpoint returns, stopping cleanly on the `[DONE]`
    sentinel line.
    """
    timeout = httpx.Timeout(connect=3.0, read=None, write=5.0, pool=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream(
            "POST", _OPENAI_CHAT_URL, headers=headers, json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    return
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                text = choices[0].get("delta", {}).get("content")
                if text:
                    yield text


async def stream_reply(
    history: list[ConversationTurn],
    user_text: str,
    system_prompt: str,
    call_sid: str = "",
) -> AsyncIterator[str]:
    """Stream an OpenAI chat completion, yielding complete sentences as they arrive.

    Args:
        history: Prior conversation turns for this call (excluding `user_text`).
        user_text: The latest transcribed user utterance.
        system_prompt: The agent's configured system prompt/persona.
        call_sid: Call identifier, included in logs for traceability.

    Yields:
        Complete sentences (or the final partial fragment) as soon as a
        sentence boundary is detected in the streamed output. On any
        failure or timeout, yields a single fallback sentence instead of
        raising - there is no fallback LLM provider to retry with.
    """
    settings = get_settings()
    messages = (
        [{"role": "system", "content": system_prompt}]
        + _history_to_openai_messages(history)
        + [{"role": "user", "content": user_text}]
    )
    payload = {"model": settings.openai_model, "messages": messages, "stream": True}
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }

    loop = asyncio.get_running_loop()
    deadline = loop.time() + _GENERATION_TIMEOUT_SECONDS
    buffer = ""
    got_any = False

    raw_iter = _raw_delta_stream(payload, headers).__aiter__()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            logger.warning("openai_llm.timeout", call_sid=call_sid)
            break
        try:
            text = await asyncio.wait_for(raw_iter.__anext__(), timeout=remaining)
        except StopAsyncIteration:
            break
        except asyncio.TimeoutError:
            logger.warning("openai_llm.timeout", call_sid=call_sid)
            break
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - must never crash the call loop
            logger.error("openai_llm.failed", call_sid=call_sid, error=str(exc))
            break

        buffer += text
        parts = _SENTENCE_BOUNDARY_RE.split(buffer)
        if len(parts) > 1:
            *complete, buffer = parts
            for sentence in complete:
                sentence = sentence.strip()
                if sentence:
                    got_any = True
                    yield sentence

    remainder = buffer.strip()
    if remainder:
        got_any = True
        yield remainder

    if not got_any:
        yield _FALLBACK_REPLY


__all__ = ["stream_reply"]
