"""ElevenLabs streaming text-to-speech, output as mu-law 8kHz.

Streams synthesized audio for a single sentence at a time so playback
can begin as soon as the first bytes arrive, minimizing time-to-first-audio.
The `ulaw_8000` output format means ElevenLabs already returns audio in the
exact format Exotel expects, so no local resampling/encoding is needed.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import httpx
import structlog

from app.config import get_settings

logger = structlog.get_logger(__name__)

_ELEVENLABS_STREAM_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream"
_MODEL_ID = "eleven_multilingual_v2"
_OUTPUT_FORMAT = "ulaw_8000"
# Maximum latency optimization (0-4): trades a little audio quality/normalization
# for the lowest possible time-to-first-byte, which matters most for a live call.
# Temporarily lowered from 4 -> 0 while diagnosing reported noise/static on real
# Exotel calls (level 4's compression artifacts are a suspect since the same
# audio plays clean through our own test client but not through Exotel).
_OPTIMIZE_STREAMING_LATENCY = 0
_CONNECT_TIMEOUT_SECONDS = 3.0
_TOTAL_TIMEOUT_SECONDS = 10.0
_MAX_CONNECT_RETRIES = 1

_VOICE_SETTINGS = {
    "stability": 0.4,
    "similarity_boost": 0.75,
    "style": 0.3,
    "speed": 1.05,
}


async def stream_tts(
    text: str,
    voice_id: str = "",
    call_sid: str = "",
) -> AsyncIterator[bytes]:
    """Stream mu-law 8kHz audio for `text` from ElevenLabs.

    Args:
        text: The sentence/text to synthesize.
        voice_id: ElevenLabs voice id to use. Falls back to the
            configured default voice if empty.
        call_sid: Call identifier, included in logs for traceability.

    Yields:
        Raw mu-law encoded audio chunks, ready to base64-encode and send
        directly to Exotel as `media` events. Yields nothing (empty) on
        failure, so callers should fall back to another TTS provider.
    """
    if not text.strip():
        return

    settings = get_settings()
    resolved_voice_id = voice_id or settings.elevenlabs_voice_id
    url = _ELEVENLABS_STREAM_URL.format(voice_id=resolved_voice_id)

    headers = {
        "xi-api-key": settings.elevenlabs_api_key,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": _MODEL_ID,
        "voice_settings": _VOICE_SETTINGS,
    }
    params = {
        "output_format": _OUTPUT_FORMAT,
        "optimize_streaming_latency": _OPTIMIZE_STREAMING_LATENCY,
    }

    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT_SECONDS, read=_TOTAL_TIMEOUT_SECONDS, write=5.0, pool=5.0
    )

    attempt = 0
    while True:
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST", url, headers=headers, params=params, json=payload
                ) as response:
                    response.raise_for_status()
                    async for audio_chunk in response.aiter_bytes():
                        if audio_chunk:
                            yield audio_chunk
            return
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            if attempt >= _MAX_CONNECT_RETRIES:
                logger.error(
                    "elevenlabs_tts.connect_failed", call_sid=call_sid, error=str(exc)
                )
                return
            logger.warning(
                "elevenlabs_tts.retrying", call_sid=call_sid, attempt=attempt + 1
            )
            await asyncio.sleep(0.15)
            attempt += 1
        except httpx.TimeoutException:
            logger.warning("elevenlabs_tts.timeout", call_sid=call_sid, text_len=len(text))
            return
        except Exception as exc:  # noqa: BLE001 - must never crash the call loop
            logger.error("elevenlabs_tts.failed", call_sid=call_sid, error=str(exc))
            return


__all__ = ["stream_tts"]
