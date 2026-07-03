"""Unit tests for app.vad and a basic integration test of the WebSocket route."""

from __future__ import annotations

import base64
import json
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest
import structlog
from fastapi.testclient import TestClient

from app.vad import VADProcessor


def _frame(is_loud: bool, n_samples: int = 160) -> bytes:
    """Build a 20ms, 8kHz PCM16 frame that is either loud (speech-like) or silent."""
    if is_loud:
        t = np.arange(n_samples)
        samples = (np.sin(2 * np.pi * 300 * t / 8000) * 12000).astype(np.int16)
    else:
        samples = np.zeros(n_samples, dtype=np.int16)
    return samples.tobytes()


class TestVADProcessor:
    def test_silence_frames_return_silence(self) -> None:
        vad = VADProcessor(sample_rate=8000)
        for _ in range(10):
            assert vad.process_frame(_frame(False)) == "silence"
        assert vad.is_speaking is False

    def test_sustained_speech_then_silence_yields_end_of_utterance(self) -> None:
        vad = VADProcessor(
            sample_rate=8000, start_speech_ms=40, end_silence_ms=100, frame_ms=20
        )

        # A couple of loud frames to confirm start-of-speech.
        states = [vad.process_frame(_frame(True)) for _ in range(3)]
        assert "speech" in states
        assert vad.is_speaking is True

        # Silence frames should eventually flip to end_of_utterance once the
        # configured trailing-silence duration has elapsed.
        results = [vad.process_frame(_frame(False)) for _ in range(10)]
        assert "end_of_utterance" in results
        assert vad.is_speaking is False

    def test_reset_clears_speaking_state(self) -> None:
        vad = VADProcessor(sample_rate=8000, start_speech_ms=20)
        vad.process_frame(_frame(True))
        vad.process_frame(_frame(True))
        vad.reset()
        assert vad.is_speaking is False


class TestBargeIn:
    @pytest.mark.asyncio
    async def test_barge_in_does_not_reset_vad_speaking_state(self) -> None:
        """Regression test: barge-in must not force the VAD to forget it
        already confirmed the caller is speaking, otherwise the first
        ~60ms of the caller's new utterance gets dropped from the buffer
        while the VAD re-confirms speech from scratch.
        """
        from unittest.mock import AsyncMock

        from app.websocket_handler import SessionState, _handle_barge_in

        fake_ws = AsyncMock()
        session = SessionState(websocket=fake_ws, stream_sid="s1")
        session.vad.process_frame(_frame(True))
        session.vad.process_frame(_frame(True))
        session.vad.process_frame(_frame(True))
        assert session.vad.is_speaking is True

        session.is_bot_speaking = True
        await _handle_barge_in(session, structlog.get_logger())

        assert session.vad.is_speaking is True
        assert session.is_bot_speaking is False
        fake_ws.send_json.assert_awaited_once_with({"event": "clear", "stream_sid": "s1"})


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """A TestClient with required env vars set so Settings() doesn't warn/crash."""
    monkeypatch.setenv("LOVABLE_API_SECRET", "test-secret")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    from app.main import app

    return TestClient(app)


class TestHealthAndMetrics:
    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_metrics_endpoint(self, client: TestClient) -> None:
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "voicebot_active_calls" in response.text


class TestWebSocketFlow:
    def test_connected_start_and_stop_lifecycle(self, client: TestClient) -> None:
        """Drive a minimal connected -> start -> stop sequence through the
        real WebSocket route, with all external services mocked out, and
        verify the server sends back a greeting `media` frame and doesn't
        crash or hang.
        """
        from app.models import AgentConfig

        fake_config = AgentConfig(
            agent_id="agent-123",
            system_prompt="Be helpful.",
            voice_id="voice-1",
            first_message="Hello there!",
            language="en-US",
            temperature=0.5,
            max_tokens=128,
        )

        async def fake_stream_tts(text, voice_id="", call_sid=""):
            yield b"\x00" * 160

        with patch(
            "app.websocket_handler.lovable_api.fetch_agent_config",
            new=AsyncMock(return_value=fake_config),
        ), patch(
            "app.websocket_handler.lovable_api.post_call_log", new=AsyncMock()
        ), patch(
            "app.websocket_handler.elevenlabs_tts.stream_tts", new=fake_stream_tts
        ):
            with client.websocket_connect("/exotel/voicebot") as websocket:
                websocket.send_text(
                    json.dumps({"event": "connected", "protocol": "websocket", "version": "1.0.0"})
                )
                websocket.send_text(
                    json.dumps(
                        {
                            "event": "start",
                            "sequence_number": "1",
                            "start": {
                                "stream_sid": "stream-1",
                                "call_sid": "call-1",
                                "account_sid": "acct-1",
                                "from": "+919999999999",
                                "to": "+917971451588",
                                "custom_parameters": {"agent_id": "agent-123"},
                                "media_format": {
                                    "encoding": "audio/x-mulaw",
                                    "sample_rate": 8000,
                                    "channels": 1,
                                },
                            },
                        }
                    )
                )

                greeting_message = json.loads(websocket.receive_text())
                assert greeting_message["event"] == "media"
                assert greeting_message["stream_sid"] == "stream-1"
                assert base64.b64decode(greeting_message["media"]["payload"]) == b"\x00" * 160

                mark_message = json.loads(websocket.receive_text())
                assert mark_message["event"] == "mark"

                websocket.send_text(
                    json.dumps(
                        {
                            "event": "stop",
                            "sequence_number": "999",
                            "stop": {"call_sid": "call-1"},
                        }
                    )
                )
