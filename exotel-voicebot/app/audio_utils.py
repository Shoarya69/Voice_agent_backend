"""Audio format conversion helpers.

Exotel streams 8kHz, mono, 8-bit mu-law encoded PCM audio in base64-encoded
20ms frames. Most STT/TTS providers expect 16-bit linear PCM (and sometimes
a different sample rate), so this module provides the small set of pure
functions needed to convert between the two representations.

Uses `audioop` (Python < 3.13) or the `audioop-lts` backport (Python >=
3.13, since the stdlib module was removed) transparently.
"""

from __future__ import annotations

import io
import wave

try:
    import audioop  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - exercised only on Python 3.13+
    import audioop_lts as audioop  # type: ignore[import-not-found,no-redef]

import numpy as np

# Width in bytes of a single 16-bit linear PCM sample.
_PCM16_SAMPLE_WIDTH = 2


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    """Convert 8-bit mu-law encoded audio to 16-bit linear PCM.

    Args:
        mulaw_bytes: Raw mu-law encoded audio bytes (1 byte/sample).

    Returns:
        16-bit little-endian linear PCM bytes (2 bytes/sample).
    """
    if not mulaw_bytes:
        return b""
    return audioop.ulaw2lin(mulaw_bytes, _PCM16_SAMPLE_WIDTH)


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    """Convert 16-bit linear PCM audio to 8-bit mu-law encoding.

    Args:
        pcm_bytes: 16-bit little-endian linear PCM bytes (2 bytes/sample).

    Returns:
        Raw mu-law encoded audio bytes (1 byte/sample).
    """
    if not pcm_bytes:
        return b""
    return audioop.lin2ulaw(pcm_bytes, _PCM16_SAMPLE_WIDTH)


def chunk_audio(pcm: bytes, chunk_ms: int = 20, sample_rate: int = 8000) -> list[bytes]:
    """Split 16-bit PCM audio into fixed-duration chunks.

    Args:
        pcm: 16-bit little-endian linear PCM bytes.
        chunk_ms: Desired chunk duration in milliseconds.
        sample_rate: Sample rate of `pcm` in Hz.

    Returns:
        List of PCM byte chunks, each `chunk_ms` long (the final chunk may
        be shorter if `pcm` length isn't an exact multiple).
    """
    if not pcm:
        return []
    bytes_per_sample = _PCM16_SAMPLE_WIDTH
    samples_per_chunk = int(sample_rate * chunk_ms / 1000)
    chunk_size = samples_per_chunk * bytes_per_sample
    if chunk_size <= 0:
        return [pcm]
    return [pcm[i : i + chunk_size] for i in range(0, len(pcm), chunk_size)]


def resample(pcm: bytes, from_hz: int, to_hz: int) -> bytes:
    """Resample 16-bit mono linear PCM audio between sample rates.

    Args:
        pcm: 16-bit little-endian linear PCM bytes.
        from_hz: Source sample rate in Hz.
        to_hz: Target sample rate in Hz.

    Returns:
        Resampled 16-bit little-endian linear PCM bytes. Returns the input
        unchanged if `from_hz == to_hz`.
    """
    if from_hz == to_hz or not pcm:
        return pcm
    converted, _ = audioop.ratecv(pcm, _PCM16_SAMPLE_WIDTH, 1, from_hz, to_hz, None)
    return converted


def pcm16_bytes_to_numpy(pcm: bytes) -> np.ndarray:
    """Convert 16-bit PCM bytes to a numpy int16 array for buffer manipulation."""
    if not pcm:
        return np.array([], dtype=np.int16)
    return np.frombuffer(pcm, dtype=np.int16)


def numpy_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    """Convert a numpy int16 array back to raw 16-bit PCM bytes."""
    return samples.astype(np.int16).tobytes()


def rms_level(pcm: bytes) -> float:
    """Compute the RMS (root-mean-square) volume level of a PCM buffer.

    Useful for lightweight energy-based checks alongside VAD.
    """
    if not pcm:
        return 0.0
    return float(audioop.rms(pcm, _PCM16_SAMPLE_WIDTH))


def pcm16_to_wav_bytes(pcm: bytes, sample_rate: int = 8000) -> bytes:
    """Wrap raw 16-bit mono PCM bytes in a minimal WAV container.

    Used to package a buffered utterance as a `.wav` file for batch STT
    upload (multipart form file).

    Args:
        pcm: 16-bit little-endian mono PCM bytes.
        sample_rate: Sample rate of `pcm` in Hz.

    Returns:
        A complete in-memory WAV file as bytes.
    """
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(_PCM16_SAMPLE_WIDTH)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
