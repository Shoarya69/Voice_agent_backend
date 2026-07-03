"""ElevenLabs Scribe speech-to-text client (batch mode).

Sends a buffered utterance (as a WAV file) to ElevenLabs' Scribe STT
endpoint and returns the transcribed text. Batch mode is used (rather
than streaming) because Exotel already buffers a full utterance for us
via VAD end-of-speech detection, keeping this integration simple and
robust. Uses the same `ELEVENLABS_API_KEY` as the TTS side.
"""

from __future__ import annotations

import httpx
import structlog

from app.audio_utils import pcm16_to_wav_bytes
from app.config import get_settings
from app.utils import retry_async

logger = structlog.get_logger(__name__)

_ELEVENLABS_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
_MODEL_ID = "scribe_v2"
_REQUEST_TIMEOUT_SECONDS = 3.0
_TRANSIENT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout)


async def transcribe(pcm16_bytes: bytes, sample_rate: int = 8000, call_sid: str = "") -> str:
    """Transcribe a buffered utterance using ElevenLabs Scribe batch STT.

    Args:
        pcm16_bytes: 16-bit little-endian mono PCM audio for the full
            utterance (as detected by VAD end-of-speech).
        sample_rate: Sample rate of `pcm16_bytes` in Hz.
        call_sid: Call identifier, included in logs for traceability.

    Returns:
        The transcribed text, or an empty string if transcription failed
        or produced no speech (never raises).
    """
    if not pcm16_bytes:
        return ""

    settings = get_settings()
    wav_bytes = pcm16_to_wav_bytes(pcm16_bytes, sample_rate)

    files = {"file": ("audio.wav", wav_bytes, "audio/wav")}
    data = {
        "model_id": _MODEL_ID,
        "language_code": settings.language,
    }
    headers = {"xi-api-key": settings.elevenlabs_api_key}

    async def _do_transcribe() -> str:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECONDS) as client:
            response = await client.post(
                _ELEVENLABS_STT_URL, headers=headers, data=data, files=files
            )
            response.raise_for_status()
            result = response.json()
            return (result.get("text") or "").strip()

    try:
        transcript = await retry_async(
            _do_transcribe,
            retries=1,
            base_delay_seconds=0.15,
            retry_on=_TRANSIENT_ERRORS,
            op_name="elevenlabs_stt.transcribe",
            call_sid=call_sid,
        )
        logger.info("elevenlabs_stt.transcribed", call_sid=call_sid, transcript=transcript)
        return transcript
    except httpx.TimeoutException:
        logger.warning("elevenlabs_stt.timeout", call_sid=call_sid)
        return ""
    except Exception as exc:  # noqa: BLE001 - must never crash the call loop
        logger.error("elevenlabs_stt.failed", call_sid=call_sid, error=str(exc))
        return ""


__all__ = ["transcribe"]
