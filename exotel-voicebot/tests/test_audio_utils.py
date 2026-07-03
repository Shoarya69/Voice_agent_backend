"""Unit tests for app.audio_utils."""

from __future__ import annotations

import numpy as np

from app.audio_utils import (
    chunk_audio,
    mulaw_to_pcm16,
    numpy_to_pcm16_bytes,
    pcm16_bytes_to_numpy,
    pcm16_to_mulaw,
    resample,
    rms_level,
)


def _sine_wave_pcm16(freq_hz: float = 440.0, duration_s: float = 0.02, sample_rate: int = 8000) -> bytes:
    n_samples = int(sample_rate * duration_s)
    t = np.arange(n_samples) / sample_rate
    samples = (np.sin(2 * np.pi * freq_hz * t) * 10000).astype(np.int16)
    return samples.tobytes()


class TestMulawPcmRoundTrip:
    def test_pcm_to_mulaw_to_pcm_round_trip_is_close(self) -> None:
        original = _sine_wave_pcm16()
        mulaw = pcm16_to_mulaw(original)
        restored = mulaw_to_pcm16(mulaw)

        assert len(restored) == len(original)
        # mu-law is lossy; allow reasonable quantization error per sample.
        original_arr = pcm16_bytes_to_numpy(original)
        restored_arr = pcm16_bytes_to_numpy(restored)
        max_diff = np.max(np.abs(original_arr.astype(int) - restored_arr.astype(int)))
        assert max_diff < 2000

    def test_mulaw_halves_the_byte_size(self) -> None:
        pcm = _sine_wave_pcm16()
        mulaw = pcm16_to_mulaw(pcm)
        assert len(mulaw) == len(pcm) // 2

    def test_empty_input_returns_empty_output(self) -> None:
        assert mulaw_to_pcm16(b"") == b""
        assert pcm16_to_mulaw(b"") == b""


class TestChunkAudio:
    def test_chunk_audio_produces_correct_frame_count(self) -> None:
        pcm = _sine_wave_pcm16(duration_s=0.1)  # 100ms
        chunks = chunk_audio(pcm, chunk_ms=20, sample_rate=8000)
        # 100ms / 20ms = 5 chunks
        assert len(chunks) == 5
        for chunk in chunks:
            assert len(chunk) == 320  # 20ms * 8000Hz * 2 bytes

    def test_chunk_audio_handles_partial_final_chunk(self) -> None:
        pcm = _sine_wave_pcm16(duration_s=0.025)  # 25ms
        chunks = chunk_audio(pcm, chunk_ms=20, sample_rate=8000)
        assert len(chunks) == 2
        assert len(chunks[0]) == 320
        assert len(chunks[1]) < 320

    def test_chunk_audio_empty_input(self) -> None:
        assert chunk_audio(b"") == []


class TestResample:
    def test_resample_same_rate_is_noop(self) -> None:
        pcm = _sine_wave_pcm16()
        assert resample(pcm, 8000, 8000) == pcm

    def test_resample_upsamples_to_expected_length(self) -> None:
        pcm = _sine_wave_pcm16(duration_s=0.1, sample_rate=8000)
        resampled = resample(pcm, 8000, 16000)
        expected_samples = len(pcm) // 2 * 2
        assert abs(len(resampled) // 2 - expected_samples) <= 2

    def test_resample_empty_input(self) -> None:
        assert resample(b"", 8000, 16000) == b""


class TestNumpyHelpers:
    def test_pcm_numpy_round_trip(self) -> None:
        pcm = _sine_wave_pcm16()
        arr = pcm16_bytes_to_numpy(pcm)
        assert arr.dtype == np.int16
        restored = numpy_to_pcm16_bytes(arr)
        assert restored == pcm


class TestRmsLevel:
    def test_rms_of_silence_is_zero(self) -> None:
        silence = (np.zeros(160, dtype=np.int16)).tobytes()
        assert rms_level(silence) == 0.0

    def test_rms_of_tone_is_positive(self) -> None:
        pcm = _sine_wave_pcm16()
        assert rms_level(pcm) > 0.0

    def test_rms_of_empty_is_zero(self) -> None:
        assert rms_level(b"") == 0.0
